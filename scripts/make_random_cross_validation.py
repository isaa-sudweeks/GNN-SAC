#!/usr/bin/env python3
"""Write a random-configuration cross-validation definition from an existing one.

Every development topology in the source definition is pooled and randomly
partitioned into folds, ignoring the source's grouping. The source's final-test
set is carried over unchanged and never enters any fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = PROJECT_ROOT / "sac"
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.cross_validation import random_partition, validate_cross_validation_spec
from scripts.launch_cross_validation import load_definition


CONFIG_ROOT = PROJECT_ROOT / "config"


def build_definition(
    *,
    source: str,
    num_folds: int,
    split_seed: int,
    name: str,
    config_root: Path = CONFIG_ROOT,
) -> dict:
    """Return the normalized random-fold definition, including split metadata."""
    source_spec = load_definition(source, config_root)
    pool = [
        topology
        for topologies in source_spec["groups"].values()
        for topology in topologies
    ]
    definition = {
        "enabled": True,
        "name": name,
        "groups": random_partition(pool, num_folds, split_seed),
        "final_test": list(source_spec["final_test"]),
        "held_out_group": None,
    }
    validate_cross_validation_spec(definition)
    definition["split"] = {
        "source": source,
        "num_folds": int(num_folds),
        "seed": int(split_seed),
    }
    return definition


def render_definition(definition: dict) -> str:
    """Render a definition in the hand-written config/cross_validation style."""
    split = definition["split"]
    command = (
        "python scripts/make_random_cross_validation.py "
        f"--source {split['source']} --num-folds {split['num_folds']} "
        f"--split-seed {split['seed']} --name {definition['name']}"
    )
    lines = [
        "# @package _global_",
        "",
        f"# Random-configuration folds pooled from '{split['source']}'.",
        "# Generated file; regenerate instead of editing by hand:",
        f"#   {command}",
        "cross_validation:",
        "  enabled: true",
        f"  name: {definition['name']}",
        "  split:",
        f"    source: {split['source']}",
        f"    num_folds: {split['num_folds']}",
        f"    seed: {split['seed']}",
        "  groups:",
    ]
    for group_name, topologies in definition["groups"].items():
        lines.append(f"    {group_name}:")
        lines.extend(f"      - {topology}" for topology in topologies)
    if definition["final_test"]:
        lines.append("  final_test:")
        lines.extend(f"    - {topology}" for topology in definition["final_test"])
    else:
        lines.append("  final_test: []")
    lines.append("  held_out_group: null")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="node_count_loso")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--name", default="random_5fold")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the definition without writing it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    definition = build_definition(
        source=args.source,
        num_folds=args.num_folds,
        split_seed=args.split_seed,
        name=args.name,
    )
    text = render_definition(definition)
    if args.dry_run:
        print(text, end="")
        return 0
    output = CONFIG_ROOT / "cross_validation" / f"{args.name}.yaml"
    output.write_text(text)
    print(f"Wrote {output}")
    for group_name, topologies in definition["groups"].items():
        print(f"  {group_name}: {', '.join(topologies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
