#!/usr/bin/env python3
"""Launch random five-fold CV with the pinned paper-v4 development teachers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.launch_cross_validation import main as launch_cv, parse_seeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    parser.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3])
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_intermixed_args(argv)
    defaults = {
        "platform": "supercomputer",
        "sim_backend": "mjx",
        "distillation": "kl_paper_v4",
        "exp_name": "diversity-random-cv",
        "run_root": "/home/isuds/nobackup/autodelete/GNN-SAC",
        "distillation.cache_dir": (
            "/home/isuds/nobackup/autodelete/GNN-SAC/distillation_cache/paper-v4"
        ),
    }
    supplied = {override.lstrip("+~").split("=", 1)[0] for override in args.overrides}
    overrides = [f"{key}={value}" for key, value in defaults.items() if key not in supplied]
    command = [
        "random_5fold",
        "--seeds", ",".join(str(seed) for seed in args.seeds),
        "--shuffle-seed", str(args.shuffle_seed),
        *overrides,
        *args.overrides,
    ]
    if args.manifest is not None:
        command.extend(["--manifest", str(args.manifest)])
    if args.dry_run:
        command.append("--dry-run")
    return launch_cv(command)


if __name__ == "__main__":
    raise SystemExit(main())
