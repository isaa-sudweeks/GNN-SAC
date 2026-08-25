import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, filter_overrides
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher
from omegaconf import OmegaConf, open_dict

from common.parser import multirun_work_dir


log = logging.getLogger(__name__)
_STEP_CHECKPOINT_RE = re.compile(r"^step_(\d+)\.pt$")
_CHECKPOINT_METADATA_VERSION = 1


def resolve_work_dir(cfg, job_num: int) -> Path:
    """Resolve one composed sweep job's training directory."""
    override_dirname = str(OmegaConf.select(cfg, "hydra.job.override_dirname", default=""))
    return multirun_work_dir(
        str(cfg.work_dir),
        isolate_multirun_runs=bool(cfg.get("isolate_multirun_runs", False)),
        job_num=job_num,
        override_dirname=override_dirname,
    )


def resolve_checkpoint_dir(cfg, job_num: int) -> Path:
    """Resolve one composed sweep job's checkpoint directory."""
    work_dir = resolve_work_dir(cfg, job_num)
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    if checkpoint_dir in (None, "", "???"):
        return work_dir / "checkpoints"
    checkpoint_dir = Path(str(checkpoint_dir))
    if checkpoint_dir.is_absolute():
        return checkpoint_dir
    return work_dir / checkpoint_dir


def _metadata_checkpoint_step(checkpoint_dir: Path) -> int | None:
    metadata_path = checkpoint_dir / "latest.metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
        if metadata.get("format_version") != _CHECKPOINT_METADATA_VERSION:
            raise ValueError("unsupported metadata format")
        step = int(metadata["step"])
        target_steps = int(metadata["target_steps"])
        checkpoint_name = str(metadata["checkpoint"])
        if step < 0 or target_steps <= 0 or step > target_steps:
            raise ValueError("invalid checkpoint progress")
        if checkpoint_name != f"step_{step}.pt":
            raise ValueError("checkpoint filename does not match saved step")
        if not (checkpoint_dir / checkpoint_name).is_file():
            raise ValueError("numbered checkpoint is missing")
        if not (checkpoint_dir / "latest.pt").is_file():
            raise ValueError("latest checkpoint is missing")
        return step
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        log.warning("Ignoring invalid checkpoint metadata at %s: %s", metadata_path, exc)
        return -1


def completed_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Return saved progress without deserializing a full training checkpoint."""
    metadata_step = _metadata_checkpoint_step(checkpoint_dir)
    if metadata_step is not None:
        return None if metadata_step < 0 else metadata_step

    highest_step = None
    try:
        for path in checkpoint_dir.iterdir():
            match = _STEP_CHECKPOINT_RE.fullmatch(path.name)
            if match is not None and path.is_file():
                step = int(match.group(1))
                highest_step = step if highest_step is None else max(highest_step, step)
    except OSError:
        return None
    return highest_step


class FilteringSlurmLauncher(SlurmLauncher):
    """Submitit launcher that omits sweep jobs with completed checkpoints."""

    def __init__(self, skip_completed_jobs: bool = True, **params: Any) -> None:
        super().__init__(**params)
        self.skip_completed_jobs = bool(skip_completed_jobs)

    def _compose_job(self, overrides: Sequence[str], job_num: int):
        assert self.hydra_context is not None
        assert self.config is not None
        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, list(overrides)
        )
        with open_dict(sweep_config.hydra.job) as job:
            job.num = job_num
        return sweep_config

    def _completed_job_return(self, overrides: Sequence[str], cfg, work_dir: Path) -> JobReturn:
        return JobReturn(
            overrides=list(overrides),
            cfg=cfg,
            hydra_cfg=cfg.hydra,
            working_dir=str(work_dir),
            task_name=str(OmegaConf.select(cfg, "hydra.job.name", default="")),
            status=JobStatus.COMPLETED,
            _return_value=None,
        )

    def _is_complete(self, cfg, checkpoint_dir: Path) -> tuple[bool, int | None]:
        if not self.skip_completed_jobs:
            return False, None
        if str(cfg.get("resume_from_checkpoint", "")).lower() != "latest":
            return False, None
        target_steps = int(cfg.steps)
        saved_step = completed_checkpoint_step(checkpoint_dir)
        return saved_step is not None and saved_step >= target_steps, saved_step

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None
        num_jobs = len(job_overrides)
        assert num_jobs > 0

        results: list[JobReturn | None] = [None] * num_jobs
        pending: list[tuple[int, int, Sequence[str], Any]] = []
        for position, overrides in enumerate(job_overrides):
            job_num = initial_job_idx + position
            sweep_config = self._compose_job(overrides, job_num)
            checkpoint_dir = resolve_checkpoint_dir(sweep_config, job_num)
            complete, saved_step = self._is_complete(sweep_config, checkpoint_dir)
            if complete:
                work_dir = resolve_work_dir(sweep_config, job_num)
                log.info(
                    "Skipping completed Hydra job #%d at step %d/%d: %s",
                    job_num,
                    saved_step,
                    int(sweep_config.steps),
                    work_dir,
                )
                results[position] = self._completed_job_return(
                    overrides, sweep_config, work_dir
                )
            else:
                pending.append((position, job_num, overrides, sweep_config))

        log.info(
            "Completed-job scan: skipped %d; submitting %d",
            num_jobs - len(pending),
            len(pending),
        )
        if not pending:
            return [result for result in results if result is not None]

        params = self.params
        init_params = {"folder": params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}
        init_params.update(
            **{
                f"{self._EXECUTOR}_{key}": value
                for key, value in params.items()
                if key in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        executor_params = {
            key if key in baseparams else f"{self._EXECUTOR}_{key}": value
            for key, value in params.items()
            if key not in init_keys
        }
        executor.update_parameters(**executor_params)

        log.info(
            "Submitit '%s' sweep output dir : %s",
            self._EXECUTOR,
            self.config.hydra.sweep.dir,
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            os.chmod(sweep_dir, mode=int(str(self.config.hydra.sweep.mode), 8))

        job_params = []
        singleton_state = Singleton.get_state()
        for _, job_num, overrides, _ in pending:
            log.info("\t#%d : %s", job_num, " ".join(filter_overrides(overrides)))
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    job_num,
                    f"job_id_for_{job_num}",
                    singleton_state,
                )
            )

        jobs = executor.map_array(self, *zip(*job_params))
        for (position, _, _, _), job in zip(pending, jobs):
            results[position] = job.results()[0]
        return [result for result in results if result is not None]
