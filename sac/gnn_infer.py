from __future__ import annotations

from pathlib import Path
import argparse
import csv
import os
import platform
import re
import sys
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_default_mujoco_gl = "glfw" if platform.system() == "Darwin" else "egl"
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", _default_mujoco_gl)
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import hydra
import numpy as np
import torch
from omegaconf import open_dict
from termcolor import colored

if not getattr(argparse._ActionsContainer._check_help, "_hydra_py314_compat", False):
    _argparse_check_help = argparse._ActionsContainer._check_help

    def _check_help_compat(self, action):
        if action.help is not None and not isinstance(action.help, str):
            return
        return _argparse_check_help(self, action)

    _check_help_compat._hydra_py314_compat = True
    argparse._ActionsContainer._check_help = _check_help_compat

from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env
from gnn_sac import GNNSAC


def _is_wandb_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc.endswith("wandb.ai")


def _wandb_cache_dir(cfg) -> Path:
    return Path(cfg.work_dir) / "wandb_models"


def _select_checkpoint_file(download_dir: Path, model_file: str | None = None) -> Path:
    if model_file:
        candidate = download_dir / model_file
        if not candidate.exists():
            raise FileNotFoundError(f'W&B artifact did not contain "{model_file}" under {download_dir}')
        return candidate

    preferred = [download_dir / "models" / "final.pt", download_dir / "final.pt"]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    matches = sorted(download_dir.rglob("*.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .pt checkpoint found in downloaded W&B artifact {download_dir}")
    choices = "\n".join(str(path.relative_to(download_dir)) for path in matches)
    raise ValueError(
        "Multiple .pt files found in the W&B artifact. Set model_file to one of:\n"
        f"{choices}"
    )


def _artifact_ref_from_url(model_ref: str) -> tuple[str, str | None]:
    parsed = urlparse(model_ref)
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 5 or parts[2] != "artifacts":
        raise ValueError(f"Unsupported W&B artifact URL: {model_ref}")

    entity, project = parts[0], parts[1]
    artifact_name = parts[4]
    version = parts[5] if len(parts) > 5 else "latest"
    file_path = parse_qs(parsed.query).get("path", [None])[0]
    if file_path is None and "files" in parts:
        files_index = parts.index("files")
        file_path = "/".join(parts[files_index + 1 :]) or None
    if file_path is not None:
        file_path = file_path.lstrip("/")
    if version.startswith("v") and version[1:].isdigit():
        version = version
    elif ":" in artifact_name:
        artifact_name, version = artifact_name.rsplit(":", 1)
    return f"{entity}/{project}/{artifact_name}:{version}", file_path


def _download_wandb_artifact(model_ref: str, cfg) -> Path:
    import wandb

    if model_ref.startswith("wandb-artifact://"):
        artifact_ref = model_ref.removeprefix("wandb-artifact://")
        url_model_file = None
    elif _is_wandb_url(model_ref):
        artifact_ref, url_model_file = _artifact_ref_from_url(model_ref)
    else:
        artifact_ref = model_ref
        url_model_file = None

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    root = _wandb_cache_dir(cfg) / re.sub(r"[^0-9a-zA-Z_.-]+", "-", artifact_ref)
    download_dir = Path(artifact.download(root=str(root)))
    return _select_checkpoint_file(download_dir, cfg.model_file or url_model_file)


def _download_wandb_run_file(model_ref: str, cfg) -> Path:
    import wandb

    parsed = urlparse(model_ref)
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    try:
        runs_index = parts.index("runs")
        files_index = parts.index("files")
    except ValueError as exc:
        raise ValueError(f"Unsupported W&B run file URL: {model_ref}") from exc

    if runs_index < 2 or files_index <= runs_index + 1:
        raise ValueError(f"Unsupported W&B run file URL: {model_ref}")
    entity, project = parts[0], parts[1]
    run_id = parts[runs_index + 1]
    file_path = "/".join(parts[files_index + 1 :])
    if not file_path:
        raise ValueError(f"W&B run file URL does not include a file path: {model_ref}")

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    root = _wandb_cache_dir(cfg) / f"{entity}-{project}-{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    downloaded = run.file(file_path).download(root=str(root), replace=True)
    return Path(downloaded.name)


def resolve_checkpoint(model_ref: str, cfg) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        return path

    if _is_wandb_url(model_ref):
        if "/runs/" in urlparse(model_ref).path and "/files/" in urlparse(model_ref).path:
            return _download_wandb_run_file(model_ref, cfg)
        return _download_wandb_artifact(model_ref, cfg)

    if model_ref.startswith("wandb-artifact://") or re.match(r"^[^/]+/[^/]+/[^/]+:[^/]+$", model_ref):
        return _download_wandb_artifact(model_ref, cfg)

    raise FileNotFoundError(
        f"Model reference does not exist locally and is not a recognized W&B reference: {model_ref}"
    )


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def run_inference(cfg):
    if getattr(cfg, "device", "cuda") == "cuda":
        assert torch.cuda.is_available(), "CUDA not available, please run on a GPU"

    with open_dict(cfg):
        cfg.enable_wandb = False
        cfg.save_agent = False
        cfg.save_csv = False

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    checkpoint = resolve_checkpoint(str(cfg.model), cfg)
    print(colored("Checkpoint:", "yellow", attrs=["bold"]), checkpoint)
    print(colored("Task:", "yellow", attrs=["bold"]), cfg.task)

    env = make_env(cfg)
    agent = GNNSAC(cfg)
    try:
        agent.load(checkpoint)
        agent.model.eval()
        results = []
        for episode in range(int(cfg.episodes)):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_success = 0.0
            info = {}
            step = 0
            max_steps = getattr(cfg, "inference_max_steps", None)
            while not done:
                action = agent.act(obs, t0=step == 0, eval_mode=bool(cfg.deterministic))
                obs, reward, done, info = env.step(action)
                ep_reward += _to_float(reward)
                ep_success = float(info.get("success", ep_success))
                step += 1
                if max_steps is not None and step >= int(max_steps):
                    break
            row = {
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_success": ep_success,
                "episode_length": step,
                "terminated": _to_float(info.get("terminated", 0.0)),
                "truncated": _to_float(info.get("truncated", 0.0)),
            }
            results.append(row)
            print(
                f"episode={episode} reward={ep_reward:.6f} "
                f"success={ep_success:.3f} length={step}"
            )

        rewards = np.asarray([row["episode_reward"] for row in results], dtype=np.float64)
        successes = np.asarray([row["episode_success"] for row in results], dtype=np.float64)
        print(
            colored("Summary:", "cyan", attrs=["bold"]),
            f"reward_mean={rewards.mean():.6f}",
            f"reward_std={rewards.std():.6f}",
            f"success_mean={successes.mean():.6f}",
        )

        if cfg.output_csv not in (None, "", "???"):
            output_csv = Path(cfg.output_csv).expanduser()
            if not output_csv.is_absolute():
                output_csv = Path(cfg.work_dir) / output_csv
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(colored("Wrote:", "green", attrs=["bold"]), output_csv)

        return {
            "episode_reward": float(rewards.mean()),
            "episode_success": float(successes.mean()),
        }
    finally:
        env.close()


@hydra.main(config_name="gnn_inference", config_path="../config", version_base=None)
def infer(cfg):
    """
    Run a trained GNN SAC checkpoint on any configured graph environment.

    Examples:
        python sac/gnn_infer.py model=/path/to/final.pt task=octahedron-graph-right-realistic
        python sac/gnn_infer.py model=entity/project/artifact-name:latest task=octahedron-graph-right-realistic
    """
    return run_inference(cfg)


if __name__ == "__main__":
    infer()
