#!/usr/bin/env python3
"""Generate morphology-cluster CV using farthest-point prototypes, without rewards."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "sac"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.cross_validation import validate_cross_validation_spec
from common.topology_splits import FEATURE_NAMES, farthest_point_partition, morphology_features
from scripts.launch_cross_validation import load_definition


def build_definition(
    *, source: str, num_folds: int, name: str,
    config_root: Path = PROJECT_ROOT / "config",
) -> dict:
    """Read development presets only and save descriptors for an auditable split."""
    from mujoco_truss_gen import PRESETS

    spec = load_definition(source, config_root)
    pool = sorted(topology for group in spec["groups"].values() for topology in group)
    if not 2 <= num_folds <= len(pool):
        raise ValueError(f"num_folds must be between 2 and {len(pool)}.")
    features = {}
    for topology in pool:
        print(f"Describing {topology}", file=sys.stderr, flush=True)
        features[topology] = morphology_features(*PRESETS[topology]())
    groups, prototypes = farthest_point_partition(features, num_folds)
    definition = {
        "enabled": True, "name": name, "groups": groups,
        "final_test": list(spec["final_test"]), "held_out_group": None,
        "split": {
            "source": source, "method": "farthest_point_clusters", "version": 1,
            "num_folds": num_folds, "distance": "euclidean_development_zscore",
            "start": "farthest_from_centroid", "tie_break": "sorted_topology_name",
            "mujoco_truss_gen_version": version("mujoco-truss-gen"),
            "feature_names": list(FEATURE_NAMES), "prototypes": prototypes,
            "features": features,
        },
    }
    validate_cross_validation_spec(definition)
    return definition


def render_definition(definition: dict) -> str:
    """Render the frozen fold definition and descriptor provenance."""
    split = definition["split"]
    command = (
        "uv run python scripts/make_farthest_point_cross_validation.py "
        f"--source {split['source']} --num-folds {split['num_folds']} "
        f"--name {definition['name']}"
    )
    return (
        "# @package _global_\n\n"
        "# Farthest-point prototypes with nearest-prototype morphology clusters.\n"
        "# Generated file; regenerate instead of editing by hand:\n"
        f"#   {command}\n"
        + OmegaConf.to_yaml(OmegaConf.create({"cross_validation": definition}))
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="node_count_loso")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--name", default="farthest_point_5fold")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    # Validate the output name before evaluating potentially expensive presets.
    valid_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.name or any(c not in valid_characters for c in args.name):
        parser.error("--name must contain only letters, numbers, underscores, or hyphens")
    definition = build_definition(source=args.source, num_folds=args.num_folds, name=args.name)
    text = render_definition(definition)
    if args.dry_run:
        print(text, end="")
    else:
        output = PROJECT_ROOT / "config" / "cross_validation" / f"{args.name}.yaml"
        output.write_text(text)
        print(f"Wrote {output}")
        for group, topologies in definition["groups"].items():
            print(f"  {group}: {', '.join(topologies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
