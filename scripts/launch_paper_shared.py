#!/usr/bin/env python3
"""Prepare or submit the twelve 10M-total-transition shared-controller runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESET = "paper_shared_10m"
SEEDS = (1, 2, 3)
TOTAL_TRANSITIONS = 10_000_000
ARMS = {
    "gnn_scratch": ("gnn", "scratch"),
    "padded_mlp_scratch": ("padded_mlp", "scratch"),
    "gnn_distill": ("gnn", "distill"),
    "padded_mlp_distill": ("padded_mlp", "distill"),
}
ALLOWED_OVERRIDES = {"run_root", "wandb_dir", "num_envs"}


def validate_overrides(overrides: Sequence[str]) -> None:
    """Allow resource/storage changes without silently changing the experiment."""
    for value in overrides:
        key, sep, _ = value.lstrip("+").partition("=")
        if not sep or (key not in ALLOWED_OVERRIDES and not key.startswith("hydra.launcher.")):
            raise ValueError(f"Override {value!r} is outside the resource/storage allowlist.")


def arm_overrides(arm: str, extra: Sequence[str] = ()) -> list[str]:
    """Return the complete ordered Hydra override list for one three-seed sweep."""
    validate_overrides(extra)
    backend, mode = ARMS[arm]
    distillation = "kl_paper_experts_best" if mode == "distill" else "disabled"
    result = [
        "platform=supercomputer", "sim_backend=mjx", f"sac_backend={backend}",
        f"distillation={distillation}", f"+experiment={PRESET}",
        f"exp_name=paper-shared-{backend.replace('_', '-')}-{mode}-10m",
        "seed=" + ",".join(map(str, SEEDS)),
    ]
    if mode == "distill":
        result.extend([
            "distillation.pretrain_updates=10000", "distillation.decay_steps=5000000",
            "distillation.offline_only=false",
        ])
    return [*result, *extra]


def resolved_arm(arm: str, extra: Sequence[str] = ()):
    """Compose one representative seed and validate budget and split invariants."""
    overrides = ["seed=1" if value.startswith("seed=") else value
                 for value in arm_overrides(arm, extra)]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    train = list(cfg.truss_topologies)
    heldout = list(cfg.eval_extra_topologies)
    if len(train) != 10 or len(set(train)) != 10 or len(heldout) != 3 or set(train) & set(heldout):
        raise ValueError("Expected ten unique training topologies and three disjoint held-out topologies.")
    if int(cfg.steps) * len(train) != TOTAL_TRANSITIONS:
        raise ValueError("Effective online budget must be exactly 10M transitions.")
    if int(cfg.num_envs) <= 0 or int(cfg.num_envs) % len(train):
        raise ValueError("num_envs must be positive and divisible by ten training topologies.")
    if not cfg.save_eval_agent:
        raise ValueError("Per-evaluation agent exports must remain enabled.")
    if ARMS[arm][1] == "distill":
        teachers = set(cfg.distillation.teachers)
        if teachers != set(train):
            raise ValueError("The shared teacher bank must contain exactly the ten training topologies.")
    return cfg


def check_teachers(cfg) -> None:
    """Require existing full trainer files before submitting a distilled arm."""
    if not cfg.distillation.enabled:
        return
    for topology, value in cfg.distillation.teachers.items():
        path = Path(str(value))
        if path.name.endswith(".agent.pt") or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/non-full teacher for {topology}: {path}")


def build_manifest(arms: Sequence[str], extra: Sequence[str] = (), *, check_files=False) -> dict:
    """Compose all selected arms before submitting any of them."""
    entries = []
    for arm in arms:
        cfg = resolved_arm(arm, extra)
        if check_files:
            check_teachers(cfg)
        command = ["uv", "run", "python", "sac/train.py", "-m", *arm_overrides(arm, extra)]
        entries.append({
            "arm": arm, "command": command, "command_display": shlex.join(command),
            "training_topologies": list(cfg.truss_topologies),
            "heldout_topologies": list(cfg.eval_extra_topologies),
            "per_topology_steps": int(cfg.steps), "total_online_transitions": TOTAL_TRANSITIONS,
            "offline_pretrain_updates": int(cfg.distillation.pretrain_updates),
            "teachers": OmegaConf.to_container(cfg.distillation.teachers, resolve=True),
            "jobs": [{"seed": seed, "total_online_transitions": TOTAL_TRANSITIONS} for seed in SEEDS],
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "dependency_lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "job_count": len(arms) * len(SEEDS), "arms": entries,
        "selection_note": "Retrospective held-out checkpoint selection is evaluation-selected performance, not an independent final test.",
    }


def submit_arms(entries: Sequence[dict], environment: dict, log_dir: Path) -> int:
    """Start every Hydra controller before waiting for any condition to finish.

    Submitit waits for its jobs inside each controller. Running the controllers
    concurrently queues all selected conditions without a training dependency.
    A failed condition does not cancel the other submitted jobs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    controllers = []
    for entry in entries:
        logfile = log_dir / f"{entry['arm']}.log"
        with logfile.open("w") as stream:
            process = subprocess.Popen(
                entry["command"], cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        controllers.append((entry["arm"], process, logfile))
        print(f"Started {entry['arm']} controller PID {process.pid}; log: {logfile}", flush=True)
    failed = []
    for arm, process, logfile in controllers:
        code = process.wait()
        if code:
            failed.append((arm, code))
            print(f"{arm} controller exited {code}; inspect {logfile}", flush=True)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join(ARMS), help="Comma-separated subset of the four named arms.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Submit all selected arm sweeps after validation.")
    mode.add_argument("--dry-run", action="store_true", help="Print/write commands without submitting (default).")
    parser.add_argument("--check-teachers", action="store_true", help="Check teacher files on this machine (always done before submission).")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs/paper_shared_10m/launch_manifest.json")
    parser.add_argument("overrides", nargs="*", help="Resource/storage Hydra overrides only.")
    args = parser.parse_intermixed_args(argv)
    arms = args.arms.split(",")
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        parser.error("--arms must be a comma-separated, duplicate-free subset of " + ",".join(ARMS))
    try:
        manifest = build_manifest(arms, args.overrides, check_files=args.check_teachers or args.submit)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {args.manifest}\nJobs: {manifest['job_count']}; online transitions per job: {TOTAL_TRANSITIONS:,}")
    for entry in manifest["arms"]:
        print(entry["command_display"])
    if args.submit:
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_dir = args.manifest.parent / "launch_logs" / timestamp
        return submit_arms(manifest["arms"], environment, log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
