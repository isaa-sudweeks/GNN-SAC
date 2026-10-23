#!/usr/bin/env python3
"""Preflight and submit the six-job, audit-backed performance-decay pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    sys.path.insert(0, str(path))

from common.cross_validation import resolve_cross_validation
from common.distillation import model_state_sha256
from common.distillation_schedule import PerformanceDecay
from common.parser import scale_learning_parameters_by_topology

AUDIT = ROOT / "docs/experiments/paper_v4_performance_audit.json"


def pilot_config(overrides: list[str]):
    """Resolve the actual training split and total transition budget without env creation."""
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="performance_decay_pilot", overrides=overrides)
    resolve_cross_validation(cfg)
    scale_learning_parameters_by_topology(cfg)
    return cfg


def validate_audit(cfg, *, check_teachers: bool) -> list[dict]:
    """Check split, measurements, runtime, and (on ORC) exact checkpoint model weights."""
    audit = json.loads(AUDIT.read_text())
    references = cfg.distillation.performance_decay.references
    training = list(cfg.truss_topologies)
    heldout = list(cfg.eval_extra_topologies)
    final_test = list(cfg.cross_validation.final_test)
    if set(training) & (set(heldout) | set(final_test)) or set(heldout) & set(final_test):
        raise ValueError("Training, development holdout, and final-test splits overlap.")
    if not cfg.eval_domain_randomization or cfg.eval_backend != "mujoco":
        raise ValueError("Audit references require native evaluation with audited initial-pose randomization.")
    # Check the full audit environment, including termination/physics and reset settings.
    for key, expected in audit["environment"].items():
        actual = OmegaConf.to_container(cfg[key], resolve=True) if OmegaConf.is_config(cfg[key]) else cfg[key]
        if actual != expected:
            raise ValueError(f"Pilot environment differs from the audited environment: {key}.")
    if check_teachers:
        for package in ("mujoco", "mujoco-truss-gen"):
            actual = importlib.metadata.version(package)
            expected = audit["runtime"]["packages"][package]
            if actual != expected:
                raise ValueError(f"Audit runtime mismatch for {package}: {actual} != {expected}.")
    rows = []
    for topology in training:
        evidence = audit["teachers"][topology]
        reference = references[topology]
        episodes = evidence["episodes"]
        distance = sum(row["distance"] for row in episodes) / len(episodes)
        survival = sum(row["length"] >= audit["horizon"] for row in episodes) / len(episodes)
        if (abs(reference.distance - distance) > 1e-10
                or abs(reference.survival - survival) > 1e-10
                or reference.horizon != audit["horizon"]
                or bool(reference.eligible) != (distance > 0 and survival > 0)
                or reference.teacher_model_sha256 != evidence["model_sha256"]
                or OmegaConf.to_container(reference.policy_config, resolve=True) != evidence["policy_config"]):
            raise ValueError(f"Reference no longer matches audited episodes: {topology}.")
        task = f"truss-graph:{topology}"
        path = Path(cfg.distillation.teachers[topology]).expanduser()
        sources = {task: {"model_sha256": evidence["model_sha256"]}}
        # Validate eligible/ineligible reference semantics even during dry runs.
        PerformanceDecay(cfg.distillation.performance_decay, {task: topology}, sources)
        row = dict(topology=topology, checkpoint=str(path),
                   audited_resets=len(episodes), distance=distance, survival=survival,
                   adaptive_eligible=bool(reference.eligible),
                   model_sha256=evidence["model_sha256"])
        if check_teachers:
            if not path.is_file():
                raise FileNotFoundError(f"Missing training teacher: {path}")
            # Memory-map tensor storages rather than eagerly loading large replay tensors.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if not all(key in checkpoint for key in ("config", "agent", "buffer")):
                raise ValueError(f"Teacher is not a full trainer checkpoint: {path}")
            model_hash = model_state_sha256(checkpoint["agent"]["model"])
            if model_hash != evidence["model_sha256"]:
                raise ValueError(f"Teacher weights differ from audited model: {path}")
            for key, expected in evidence["policy_config"].items():
                if checkpoint["config"].get(key) != expected:
                    raise ValueError(f"Teacher policy config differs from audit: {topology}: {key}.")
            teacher_topology = checkpoint["config"]["truss_topology"]
            if teacher_topology != topology:
                raise ValueError(f"Teacher topology mismatch: {path}")
            del checkpoint
            gc.collect()
            row["verified_model_sha256"] = model_hash
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check local config/evidence; do not open teacher files or submit.")
    parser.add_argument("--check-only", action="store_true", help="Verify teacher files on ORC without submitting.")
    parser.add_argument("--teacher-root", help="Relocate the existing kl_paper_v4 checkpoint tree.")
    parser.add_argument("--run-root", help="Override the isolated experiment output root.")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct integers.")
    overrides = []
    if args.teacher_root:
        overrides.append(f"distillation.teacher_root={args.teacher_root}")
    if args.run_root:
        overrides.append(f"run_root={args.run_root}")
    cfg = pilot_config(overrides)
    rows = validate_audit(cfg, check_teachers=not args.dry_run)
    command = ["uv", "run", "python", "sac/train.py", "--config-name", "performance_decay_pilot", "-m",
               "distillation.performance_decay.enabled=false,true",
               f"seed={','.join(str(seed) for seed in seeds)}", *overrides]
    deadline = (cfg.distillation.decay_steps if cfg.distillation.decay_steps is not None
                else cfg.steps * cfg.distillation.decay_fraction)
    payload = dict(command=command, job_count=2 * len(seeds),
                   slurm_partition=OmegaConf.load(ROOT / "config/performance_decay_pilot.yaml").hydra.launcher.partition,
                   teachers_verified=not args.dry_run,
                   total_transitions_per_job=int(cfg.steps), global_decay_steps=int(deadline),
                   training_topologies=list(cfg.truss_topologies),
                   heldout_topologies=list(cfg.eval_extra_topologies),
                   final_test_excluded=list(cfg.cross_validation.final_test), teachers=rows,
                   audit_sha256=hashlib.sha256(AUDIT.read_bytes()).hexdigest())
    manifest_path = args.manifest
    if manifest_path is None and not args.dry_run:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        manifest_path = Path(str(cfg.run_root)) / "launch" / f"preflight-{timestamp}.json"
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Preflight manifest: {manifest_path}")
    print(json.dumps(payload, indent=2))
    print("Launch:", shlex.join(command), flush=True)
    if args.dry_run or args.check_only:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
