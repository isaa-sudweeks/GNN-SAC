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
    ("elapsed_time", "T", "time"),
    ("steps_per_sec", "SPS", "float"),
]

CAT_TO_COLOR = {
    "train": "blue",
    "eval": "green",
    "training_rewards": "magenta",
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
        self._eval_rows: list[dict[str, Any]] = []
        self._finished = False

        print_run(cfg)

        self.project = _cfg_get(cfg, "wandb_project", _cfg_get(cfg, "project", "none"))
        self.entity = _cfg_get(cfg, "wandb_entity", _cfg_get(cfg, "entity", None))
        enable_wandb = bool(_cfg_get(cfg, "enable_wandb", True))

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
        tags = cfg_to_group(cfg, return_list=True) + [f"seed:{self._seed}"]
        init_kwargs = dict(
            project=self.project,
            name=str(name),
            group=self._group,
            tags=tags,
            dir=str(self._log_dir),
            config=cfg_to_dict(cfg),
        )
        if self.entity not in (None, "", "none", "???"):
            init_kwargs["entity"] = self.entity

        wandb.init(**init_kwargs)
        print(colored("Logs will be synced with W&B.", "blue", attrs=["bold"]))
        self._wandb = wandb

    def save_agent(self, agent: Any = None, identifier: str = "final") -> None:
        if not self._save_agent or agent is None:
            return
        fp = self._model_dir / f"{identifier}.pt"
        agent.save(fp)
        if self._wandb:
            artifact = self._wandb.Artifact(
                f"{self._group}-{self._seed}-{identifier}",
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
                f"{category}/{key}": value
                for key, value in clean_metrics.items()
            }
            self._wandb.log(wandb_metrics, step=step)

        if category == "eval":
            self._write_eval_csv(clean_metrics)

        self._print(clean_metrics, category)
