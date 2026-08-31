#!/usr/bin/env python3
"""Launch every leave-one-configuration-group-out fold as a Hydra multirun."""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = PROJECT_ROOT / "sac"
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.cross_validation import validate_cross_validation_spec


def parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated, duplicate-free seed list."""
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers.") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer.")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds cannot contain duplicates.")
    return seeds


def config_name_from_override(value: str) -> str:
    """Accept either NAME or the Hydra-style cross_validation=NAME spelling."""
    prefix = "cross_validation="
    return value[len(prefix) :] if value.startswith(prefix) else value


def load_definition(name: str, config_root: Path = PROJECT_ROOT / "config") -> dict:
    path = config_root / "cross_validation" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Cross-validation config does not exist: {path}")
    loaded = OmegaConf.load(path)
    value = loaded.get("cross_validation", loaded)
    spec = validate_cross_validation_spec(value)
    if spec is None:
        raise ValueError(f"Cross-validation config '{name}' is disabled.")
    if spec["name"] != name:
        raise ValueError(
            f"Cross-validation config name '{spec['name']}' must match filename '{name}'."
        )
    return spec


def ordered_folds(spec: dict, shuffle_seed: int) -> list[str]:
    folds = list(spec["groups"])
    random.Random(shuffle_seed).shuffle(folds)
    return folds


def validate_overrides(overrides: Sequence[str]) -> None:
    reserved = ("seed=", "cross_validation=", "cross_validation.held_out_group=")
    for override in overrides:
        normalized = override.lstrip("+")
        if normalized.startswith(reserved):
            raise ValueError(
                f"Override '{override}' is owned by the cross-validation launcher."
            )


def build_launch(
    *,
    config_name: str,
    spec: dict,
    seeds: Sequence[int],
    shuffle_seed: int,
    overrides: Sequence[str],
    python_executable: str = sys.executable,
) -> tuple[list[str], list[dict]]:
    """Build the Hydra command and its explicit fold-by-seed manifest rows."""
    validate_overrides(overrides)
    folds = ordered_folds(spec, shuffle_seed)
    command = [
        python_executable,
        str(PROJECT_ROOT / "sac" / "gnn_train.py"),
        "-m",
        f"cross_validation={config_name}",
        f"cross_validation.held_out_group={','.join(folds)}",
        f"seed={','.join(str(seed) for seed in seeds)}",
        *overrides,
    ]
    group_names = list(spec["groups"])
    jobs = []
    for held_out in folds:
        training_groups = [name for name in group_names if name != held_out]
        training_topologies = [
            topology
            for group_name in training_groups
            for topology in spec["groups"][group_name]
        ]
        for seed in seeds:
            jobs.append(
                {
                    "fold_index": group_names.index(held_out),
                    "fold_name": f"holdout_{held_out}",
                    "held_out_group": held_out,
                    "training_groups": training_groups,
                    "training_topologies": training_topologies,
                    "heldout_topologies": list(spec["groups"][held_out]),
                    "seed": int(seed),
                }
            )
    return command, jobs


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cross_validation",
        help="Config name, optionally written as cross_validation=NAME.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, such as platform=supercomputer.",
    )
    parser.add_argument("--seeds", required=True, type=parse_seeds, help="For example: 1,2,3")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_intermixed_args(argv)
    config_name = config_name_from_override(args.cross_validation)
    spec = load_definition(config_name)
    command, jobs = build_launch(
        config_name=config_name,
        spec=spec,
        seeds=args.seeds,
        shuffle_seed=args.shuffle_seed,
        overrides=args.overrides,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = args.manifest or (
        PROJECT_ROOT / "cross_validation_manifests" / f"{config_name}-{timestamp}.json"
    )
    payload = {
        "cross_validation": spec["name"],
        "shuffle_seed": args.shuffle_seed,
        "seeds": list(args.seeds),
        "final_test": list(spec["final_test"]),
        "command": command,
        "command_display": shlex.join(command),
        "jobs": jobs,
    }
    write_manifest(manifest_path, payload)
    print(f"Cross-validation manifest: {manifest_path}")
    print(f"Jobs: {len(jobs)}")
    print(payload["command_display"])
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
