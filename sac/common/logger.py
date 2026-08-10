"""
Logging utilities for SAC experiments.

The logger mirrors the TD-MPC2 logging style: metrics are grouped by category
(`train/...`, `eval/...`), W&B receives the full run config, evaluation videos
are logged as W&B videos, and local CSV/model outputs are written when enabled.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - only used in minimal tooling envs.
    torch = None

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - only used in minimal tooling envs.
    OmegaConf = None

try:
    from termcolor import colored
except ModuleNotFoundError:  # pragma: no cover - only used in minimal tooling envs.
    def colored(text: str, *args: Any, **kwargs: Any) -> str:
        return text


CONSOLE_FORMAT = [
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "Succ", "float"),
    ("episode_length", "Len", "float"),
    ("buffer_size", "Buf", "int"),
    ("optimizer_updates", "Upd", "int"),
    ("elapsed_time", "T", "time"),
    ("steps_per_sec", "SPS", "float"),
]

CAT_TO_COLOR = {
    "train": "blue",
    "safety": "red",
    "eval": "green",
    "training_rewards": "magenta",
    "gradient_diagnostics": "cyan",
    "profiling": "yellow",
}


def make_dir(dir_path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_plain_value(value: Any) -> Any:
    """Convert common ML scalar/container values into W&B/CSV friendly values."""
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def cfg_to_dict(cfg: Any) -> dict[str, Any]:
    """Return a serializable config dict for W&B."""
    if dataclasses.is_dataclass(cfg):
        raw_cfg = dataclasses.asdict(cfg)
    elif OmegaConf is not None and OmegaConf.is_config(cfg):
        raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        raw_cfg = dict(cfg)
    else:
        raw_cfg = {
            key: value
            for key, value in vars(cfg).items()
            if not key.startswith("_")
        }
    return _sanitize_for_wandb(raw_cfg)


def _sanitize_for_wandb(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_wandb(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_wandb(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return _sanitize_for_wandb(_to_plain_value(value))
    return value


def _safe_name(value: Any, fallback: str) -> str:
    value = fallback if value in (None, "", "???") else str(value)
    return re.sub(r"[^0-9a-zA-Z_.-]+", "-", value).strip("-") or fallback


def _checkpoint_dir(cfg: Any) -> Path:
    checkpoint_dir = _cfg_get(cfg, "checkpoint_dir", None)
    if checkpoint_dir in (None, "", "???"):
        return Path(_cfg_get(cfg, "work_dir", ".")) / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(_cfg_get(cfg, "work_dir", ".")) / checkpoint_dir
    return checkpoint_dir


def _resolve_checkpoint_path(cfg: Any) -> Path | None:
    checkpoint_ref = _cfg_get(cfg, "resume_from_checkpoint", None)
    if checkpoint_ref in (None, "", "???", False):
        return None
    if str(checkpoint_ref).lower() == "latest":
        return _checkpoint_dir(cfg) / "latest.pt"
    return Path(checkpoint_ref).expanduser()


def _wandb_metadata_path(log_dir: Path) -> Path:
    return log_dir / "wandb_run.json"


def _load_wandb_run_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _checkpoint_wandb_run_info(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists() or torch is None:
        return {}
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    logger_state = checkpoint.get("logger", {}) if isinstance(checkpoint, Mapping) else {}
    wandb_state = logger_state.get("wandb", {}) if isinstance(logger_state, Mapping) else {}
    return wandb_state if isinstance(wandb_state, dict) else {}


def wandb_resume_info(cfg: Any, log_dir: Path) -> dict[str, Any]:
    """Return persisted W&B identity when this launch is resuming a checkpoint."""
    checkpoint_path = _resolve_checkpoint_path(cfg)
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    explicit_id = _cfg_get(cfg, "wandb_id", None)
    if explicit_id not in (None, "", "???"):
        return {"id": str(explicit_id)}

    run_info = _load_wandb_run_info(_wandb_metadata_path(log_dir))
    if run_info.get("id"):
        return run_info

    return _checkpoint_wandb_run_info(checkpoint_path)


def cfg_to_group(cfg: Any, return_list: bool = False) -> str | list[str]:
    """Return a W&B-safe group name from available run identity fields."""
    env_name = _cfg_get(cfg, "env_name", _cfg_get(cfg, "task", "env"))
    exp_name = _cfg_get(cfg, "exp_name", _cfg_get(cfg, "experiment", "default"))
    parts = [_safe_name(env_name, "env"), _safe_name(exp_name, "default")]
    return parts if return_list else "-".join(parts)


def print_run(cfg: Any) -> None:
    """Pretty-print high level run metadata."""
    prefix, color, attrs = " ", "green", ["bold"]

    def _limstr(value: Any, maxlen: int = 48) -> str:
        value = str(value)
        return value[:maxlen] + "..." if len(value) > maxlen else value

    def _pprint(key: str, value: Any) -> None:
        print(prefix + colored(f'{key.capitalize() + ":":<15}', color, attrs=attrs), _limstr(value))

    kvs = [
        ("env", _cfg_get(cfg, "env_name", _cfg_get(cfg, "task", "unknown"))),
        ("steps", f"{int(_cfg_get(cfg, 'steps', 0)):,}"),
        ("seed", _cfg_get(cfg, "seed", "unknown")),
        ("experiment", _cfg_get(cfg, "exp_name", _cfg_get(cfg, "experiment", "default"))),
        ("work dir", _cfg_get(cfg, "work_dir", ".")),
    ]
    width = max(len(_limstr(value)) for _, value in kvs) + 25
    div = "-" * width
    print(div)
    for key, value in kvs:
        _pprint(key, value)
    print(div)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg: Any, wandb: Any, fps: int = 15):
        self.cfg = cfg
        self._save_dir = make_dir(Path(_cfg_get(cfg, "work_dir", ".")) / "eval_video")
        self._wandb = wandb
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.enabled = False

    def init(self, env: Any, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = bool(self._wandb and enabled)
        self.record(env)

    def record(self, env: Any) -> None:
        if not self.enabled:
            return
        try:
            frame = env.render()
        except TypeError:
            frame = env.render(mode="rgb_array")
        if frame is not None:
            self.frames.append(np.asarray(frame))

    def save(self, step: int, key: str = "videos/eval_video") -> None:
        if not self.enabled or not self.frames:
            return
        frames = np.stack(self.frames)
        video = self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format="mp4")
        self._wandb.log({key: video}, step=step)


class NullVideoRecorder:
    """No-op video recorder used when video logging is disabled."""

    def init(self, env: Any, enabled: bool = True) -> None:
        return

    def record(self, env: Any) -> None:
        return

    def save(self, step: int, key: str = "videos/eval_video") -> None:
        return


class Logger:
    """Primary experiment logger for console, local files, and W&B."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._log_dir = make_dir(_cfg_get(cfg, "work_dir", "outputs"))
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = bool(_cfg_get(cfg, "save_csv", True))
        self._save_agent = bool(_cfg_get(cfg, "save_agent", True))
        self._group = cfg_to_group(cfg)
        self._seed = _cfg_get(cfg, "seed", 0)
        self._multirun_id = _cfg_get(cfg, "multirun_id", None)
        self._eval_rows: list[dict[str, Any]] = []
        self._finished = False

        print_run(cfg)

        self.project = _cfg_get(cfg, "wandb_project", _cfg_get(cfg, "project", "none"))
        self.entity = _cfg_get(cfg, "wandb_entity", _cfg_get(cfg, "entity", None))
        enable_wandb = bool(_cfg_get(cfg, "enable_wandb", True))
        self._wandb_run_info: dict[str, Any] = {}

        self._wandb = None
        if enable_wandb and self.project not in (None, "", "none", "???"):
            self._init_wandb(cfg)
        else:
            print(colored("W&B disabled.", "blue", attrs=["bold"]))

        self._video = (
            VideoRecorder(cfg, self._wandb)
            if self._wandb and bool(_cfg_get(cfg, "save_video", False))
            else NullVideoRecorder()
        )

    @property
    def video(self) -> VideoRecorder | NullVideoRecorder:
        return self._video

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def _init_wandb(self, cfg: Any) -> None:
        os.environ["WANDB_SILENT"] = "true" if bool(_cfg_get(cfg, "wandb_silent", False)) else "false"
        import wandb

        name = _cfg_get(cfg, "wandb_name", None)
        if name in (None, "", "???"):
            name = str(self._seed)
        if self._multirun_id not in (None, "", "???"):
            name = f"{name}-{self._multirun_id}"
        tags = cfg_to_group(cfg, return_list=True) + [f"seed:{self._seed}"]
        wandb_dir = Path(_cfg_get(cfg, "wandb_dir", Path(__file__).resolve().parents[2]))
        if not wandb_dir.is_absolute():
            wandb_dir = Path(__file__).resolve().parents[2] / wandb_dir
        make_dir(wandb_dir)
        init_kwargs = dict(
            project=self.project,
            name=str(name),
            group=self._group,
            tags=tags,
            dir=str(wandb_dir),
            config=cfg_to_dict(cfg),
        )
        if self.entity not in (None, "", "none", "???"):
            init_kwargs["entity"] = self.entity
        if bool(_cfg_get(cfg, "set_wandb_offline", False)):
            init_kwargs["mode"] = "offline"

        run_info = wandb_resume_info(cfg, self._log_dir)
        if run_info.get("id"):
            init_kwargs["id"] = str(run_info["id"])
            init_kwargs["resume"] = str(_cfg_get(cfg, "wandb_resume", "allow"))

        run = wandb.init(**init_kwargs)
        self._wandb_run_info = {
            "id": getattr(run, "id", None) or getattr(wandb.run, "id", None),
            "project": self.project,
            "entity": self.entity,
            "name": str(name),
        }
        self._write_wandb_run_info()
        if bool(_cfg_get(cfg, "set_wandb_offline", False)):
            print(colored("W&B logging is offline.", "blue", attrs=["bold"]))
        else:
            print(colored("Logs will be synced with W&B.", "blue", attrs=["bold"]))
        self._wandb = wandb

    def _write_wandb_run_info(self) -> None:
        if not self._wandb_run_info.get("id"):
            return
        try:
            with _wandb_metadata_path(self._log_dir).open("w") as f:
                json.dump(self._wandb_run_info, f, indent=2, sort_keys=True)
        except OSError as exc:
            print(colored(f"Failed to write W&B run metadata: {exc}", "red"))

    def save_agent(self, agent: Any = None, identifier: str = "final") -> None:
        if not self._save_agent or agent is None:
            return
        fp = self._model_dir / f"{identifier}.pt"
        agent.save(fp)
        if self._wandb:
            artifact_name = f"{self._group}-{self._seed}"
            if self._multirun_id not in (None, "", "???"):
                artifact_name = f"{artifact_name}-{self._multirun_id}"
            artifact = self._wandb.Artifact(
                f"{artifact_name}-{identifier}",
                type="model",
            )
            artifact.add_file(str(fp))
            self._wandb.log_artifact(artifact)

    def finish(self, agent: Any = None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self.save_agent(agent)
        except Exception as exc:
            print(colored(f"Failed to save model: {exc}", "red"))
        if self._wandb:
            self._wandb.finish()

    def state_dict(self) -> dict[str, Any]:
        return {
            "eval_rows": list(self._eval_rows),
            "wandb": dict(self._wandb_run_info),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._eval_rows = list(state_dict.get("eval_rows", []))
        wandb_state = state_dict.get("wandb", {})
        if isinstance(wandb_state, Mapping):
            self._wandb_run_info = dict(wandb_state)

    def _format(self, key: str, value: Any, ty: str) -> str:
        value = _to_plain_value(value)
        if ty == "int":
            return f'{colored(key + ":", "blue")} {int(value):,}'
        if ty == "float":
            return f'{colored(key + ":", "blue")} {float(value):.03f}'
        if ty == "time":
            elapsed = str(_datetime.timedelta(seconds=int(value)))
            return f'{colored(key + ":", "blue")} {elapsed}'
        raise ValueError(f"invalid log format type: {ty}")

    def _print(self, metrics: dict[str, Any], category: str) -> None:
        category_color = CAT_TO_COLOR.get(category, "white")
        pieces = [f" {colored(category, category_color):<14}"]
        for key, display_key, ty in CONSOLE_FORMAT:
            if key in metrics:
                pieces.append(f"{self._format(display_key, metrics[key], ty):<22}")
        print(" ".join(pieces))

    def _write_eval_csv(self, metrics: dict[str, Any]) -> None:
        if not self._save_csv:
            return
        row = {
            key: _to_plain_value(value)
            for key, value in metrics.items()
            if np.isscalar(_to_plain_value(value))
        }
        self._eval_rows.append(row)
        keys = sorted({key for eval_row in self._eval_rows for key in eval_row.keys()})
        with (self._log_dir / "eval.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._eval_rows)

    def log(self, metrics: Mapping[str, Any], category: str = "train") -> None:
        if category not in CAT_TO_COLOR:
            raise ValueError(f"invalid category: {category}")

        clean_metrics = {
            key: _to_plain_value(value)
            for key, value in dict(metrics).items()
        }

        if self._wandb:
            step = clean_metrics.get("step")
            wandb_metrics = {
                key if key.startswith("safety/") else f"{category}/{key}": value
                for key, value in clean_metrics.items()
            }
            self._wandb.log(wandb_metrics, step=step)

        if category == "eval":
            self._write_eval_csv(clean_metrics)

        self._print(clean_metrics, category)
