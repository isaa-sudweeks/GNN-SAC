import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra.core.utils import JobReturn, JobStatus
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.submitit_launcher import (
    FilteringSlurmLauncher,
    completed_checkpoint_step,
    resolve_checkpoint_dir,
)


def sweep_cfg(
    root: Path,
    *,
    override_dirname: str,
    steps: int | str = 100,
    resume="latest",
):
    return OmegaConf.create(
        {
            "work_dir": str(root / "runs" / "experiment" / "seed_1"),
            "checkpoint_dir": "checkpoints",
            "isolate_multirun_runs": True,
            "resume_from_checkpoint": resume,
            "steps": steps,
            "hydra": {
                "job": {
                    "num": "???",
                    "name": "test-job",
                    "override_dirname": override_dirname,
                },
                "sweep": {"dir": str(root / "hydra")},
            },
        }
    )


def write_legacy_checkpoint(checkpoint_dir: Path, step: int, *, agent=False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".agent.pt" if agent else ".pt"
    (checkpoint_dir / f"step_{step}{suffix}").write_bytes(b"checkpoint")


def write_metadata_checkpoint(checkpoint_dir: Path, step: int, target_steps: int):
    write_legacy_checkpoint(checkpoint_dir, step)
    (checkpoint_dir / "latest.pt").write_bytes(b"latest")
    (checkpoint_dir / "latest.metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "step": step,
                "target_steps": target_steps,
                "checkpoint": f"step_{step}.pt",
            }
        )
    )


class CheckpointScanTest(unittest.TestCase):
    def test_metadata_reports_saved_step_at_or_below_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            write_metadata_checkpoint(checkpoint_dir, 80, 100)
            self.assertEqual(completed_checkpoint_step(checkpoint_dir), 80)

            write_metadata_checkpoint(checkpoint_dir, 100, 100)
            self.assertEqual(completed_checkpoint_step(checkpoint_dir), 100)

    def test_legacy_scan_uses_highest_full_checkpoint_and_ignores_agent_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            write_legacy_checkpoint(checkpoint_dir, 40)
            write_legacy_checkpoint(checkpoint_dir, 90)
            write_legacy_checkpoint(checkpoint_dir, 200, agent=True)

            self.assertEqual(completed_checkpoint_step(checkpoint_dir), 90)

    def test_invalid_metadata_fails_open_instead_of_using_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            write_legacy_checkpoint(checkpoint_dir, 100)
            (checkpoint_dir / "latest.metadata.json").write_text("not json")

            self.assertIsNone(completed_checkpoint_step(checkpoint_dir))

    def test_missing_checkpoint_directory_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertIsNone(completed_checkpoint_step(Path(tmp_dir) / "missing"))


class FakeConfigLoader:
    def __init__(self, configs):
        self.configs = configs

    def load_sweep_config(self, _master_config, overrides):
        return OmegaConf.create(
            OmegaConf.to_container(self.configs[overrides[0]], resolve=False)
        )


class FakeSubmittedJob:
    def __init__(self, job_num):
        self.job_num = job_num

    def results(self):
        return [
            JobReturn(
                overrides=[f"job={self.job_num}"],
                status=JobStatus.COMPLETED,
                _return_value=self.job_num,
            )
        ]


class FakeExecutor:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parameters = None
        self.map_args = None
        self.__class__.instances.append(self)

    def update_parameters(self, **kwargs):
        self.parameters = kwargs

    def map_array(self, _callable, overrides, job_dir_keys, job_nums, job_ids, states):
        self.map_args = (overrides, job_dir_keys, job_nums, job_ids, states)
        return [FakeSubmittedJob(job_num) for job_num in job_nums]


class FilteringLauncherTest(unittest.TestCase):
    def make_launcher(self, root, configs, *, enabled=True):
        launcher = FilteringSlurmLauncher(
            skip_completed_jobs=enabled,
            submitit_folder=str(root / "hydra" / ".submitit" / "%j"),
            max_num_timeout=2,
            timeout_min=60,
        )
        launcher.config = OmegaConf.create(
            {"hydra": {"sweep": {"dir": str(root / "hydra")}}}
        )
        launcher.hydra_context = SimpleNamespace(config_loader=FakeConfigLoader(configs))
        return launcher

    def test_all_complete_sweep_does_not_create_executor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = {
                "job=a": sweep_cfg(root, override_dirname="job=a"),
                "job=b": sweep_cfg(root, override_dirname="job=b"),
            }
            for job_num, cfg in enumerate(configs.values(), start=4):
                write_metadata_checkpoint(resolve_checkpoint_dir(cfg, job_num), 100, 100)

            launcher = self.make_launcher(root, configs)
            with patch("submitit.AutoExecutor", side_effect=AssertionError("must not submit")):
                results = launcher.launch([["job=a"], ["job=b"]], initial_job_idx=4)

            self.assertEqual(len(results), 2)
            self.assertTrue(all(result.status is JobStatus.COMPLETED for result in results))

    def test_mixed_sweep_preserves_sparse_job_numbers_and_result_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = {
                "job=a": sweep_cfg(root, override_dirname="job=a"),
                "job=b": sweep_cfg(root, override_dirname="job=b"),
                "job=c": sweep_cfg(root, override_dirname="job=c"),
            }
            write_metadata_checkpoint(resolve_checkpoint_dir(configs["job=a"], 5), 100, 100)
            write_metadata_checkpoint(resolve_checkpoint_dir(configs["job=c"], 7), 100, 100)
            FakeExecutor.instances.clear()

            launcher = self.make_launcher(root, configs)
            with patch("submitit.AutoExecutor", FakeExecutor):
                results = launcher.launch(
                    [["job=a"], ["job=b"], ["job=c"]], initial_job_idx=5
                )

            self.assertEqual(FakeExecutor.instances[0].map_args[2], (6,))
            self.assertEqual([result.return_value for result in results], [None, 6, None])
            self.assertIn("job_0005_", results[0].working_dir)
            self.assertIn("job_0007_", results[2].working_dir)

    def test_filtering_is_bypassed_when_disabled_or_not_resuming_latest(self):
        for enabled, resume in ((False, "latest"), (True, None), (True, "/tmp/checkpoint.pt")):
            with self.subTest(enabled=enabled, resume=resume), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                configs = {
                    "job=a": sweep_cfg(
                        root, override_dirname="job=a", resume=resume
                    )
                }
                write_metadata_checkpoint(resolve_checkpoint_dir(configs["job=a"], 2), 100, 100)
                FakeExecutor.instances.clear()
                launcher = self.make_launcher(root, configs, enabled=enabled)

                with patch("submitit.AutoExecutor", FakeExecutor):
                    results = launcher.launch([["job=a"]], initial_job_idx=2)

                self.assertEqual(FakeExecutor.instances[0].map_args[2], (2,))
                self.assertEqual(results[0].return_value, 2)

    def test_increasing_steps_resumes_and_decreasing_steps_skips(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            original = sweep_cfg(root, override_dirname="job=a", steps=100)
            checkpoint_dir = resolve_checkpoint_dir(original, 0)
            write_metadata_checkpoint(checkpoint_dir, 100, 100)

            increased = sweep_cfg(root, override_dirname="job=a", steps=150)
            decreased = sweep_cfg(root, override_dirname="job=a", steps=80)
            launcher = self.make_launcher(root, {"job=a": increased})
            self.assertEqual(launcher._is_complete(increased, checkpoint_dir), (False, 100))
            self.assertEqual(launcher._is_complete(decreased, checkpoint_dir), (True, 100))

    def test_arithmetic_step_expression_is_normalized_before_comparison(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = sweep_cfg(root, override_dirname="job=a", steps="1000*10")
            checkpoint_dir = resolve_checkpoint_dir(cfg, 0)
            write_metadata_checkpoint(checkpoint_dir, 10_000, 10_000)
            launcher = self.make_launcher(root, {"job=a": cfg})

            self.assertEqual(
                launcher._is_complete(cfg, checkpoint_dir),
                (True, 10_000),
            )


if __name__ == "__main__":
    unittest.main()
