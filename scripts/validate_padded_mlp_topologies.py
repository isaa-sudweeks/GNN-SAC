#!/usr/bin/env python3
"""Validate padded-MLP capacity for a configured split without collecting data."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import parse_cfg
from common.padded_mlp_actor_critic import PaddedMLPActorCritic
from common.training_validation import (
    configured_topologies, parameter_counts, padded_environment_contract,
    validate_padded_mlp_environment, validate_training_config,
)
from env import make_env


EXPECTED = {
    "tetrahedron": (4, 8, 4),
    "henneberg_n5_1tube_1": (5, 10, 8),
    "henneberg_n6_1tube_2": (6, 13, 11),
    "henneberg_n6_2tube_1": (6, 14, 10),
    "octahedron": (6, 12, 8),
    "henneberg_n8_1tube_57": (8, 19, 17),
    "henneberg_n8_2tube_187": (8, 20, 16),
    "henneberg_n8_3tube_64": (8, 21, 15),
    "usevitch_210272254_p1": (8, 18, 12),
    "usevitch_60243677150_p1": (9, 21, 14),
    "henneberg_n7_1tube_3": (7, 16, 14),
    "henneberg_n7_3tube_1": (7, 18, 12),
    "usevitch_1514879": (7, 15, 10),
}


def make_config(topology):
    """Retain the old single-topology helper for external audit callers."""
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="config", overrides=["sac_backend=padded_mlp"])
    config.truss_topology = topology
    config.domain_randomization = False
    config.save_video = False
    config.work_dir = "/tmp/padded-mlp-topology-audit"
    config.device = "cpu"
    return parse_cfg(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Full or partial run YAML; relative paths use the current directory.")
    parser.add_argument("--topologies", nargs="+", help="Explicit training topology identifiers (no paper set is assumed).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, including cross_validation and backend settings.")
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="config", overrides=["sac_backend=padded_mlp", *args.overrides])
    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))
    legacy = not args.config and not args.topologies and not args.overrides
    if args.topologies:
        config.truss_topologies = args.topologies
    elif legacy:
        config.truss_topologies = list(EXPECTED)
    config.work_dir = "/tmp/padded-mlp-topology-audit"
    config.device = "cpu"
    cfg = parse_cfg(config)
    validate_training_config(cfg)
    if cfg.sac_backend != "padded_mlp":
        parser.error("This validator requires sac_backend=padded_mlp.")
    print("topology,abstract_nodes,control_nodes,active_actions", flush=True)
    records = []
    task_records = []
    for topology in configured_topologies(cfg, include_evaluation=True):
        # Inspect each nominal model without resets, replay, teacher targets or
        # compiled MJX startup. Config compatibility was checked above.
        single = deepcopy(cfg)
        single.task = "truss-graph"
        single.tasks = [single.task]
        single.truss_topology = topology
        single.truss_topologies = None
        single.topologies = None
        single.multitask = False
        single.num_envs = 1
        single.mujoco_backend = "mujoco"
        single.domain_randomization = False
        single.save_video = False
        env = make_env(single)
        try:
            record = padded_environment_contract(cfg, env.unwrapped, owner=topology)
            abstract = len(env.unwrapped.mj_model.node_names)
            actual = (abstract, record["control_nodes"], record["active_actions"])
            if legacy and actual != EXPECTED[topology]:
                raise ValueError(f"{topology}: expected {EXPECTED[topology]}, got {actual}")
            if not records and getattr(cfg, "eval_task", None) not in (None, cfg.task):
                # Share runtime validation so registered eval-task aliases are
                # inspected too, instead of guessing their topology from text.
                task_records = validate_padded_mlp_environment(cfg, env)
            records.append(record)
            print(f"{topology},{actual[0]},{actual[1]},{actual[2]}", flush=True)
        finally:
            env.close()
    for record in task_records:
        if record not in records:
            records.append(record)
    print(json.dumps({
        "validated_capacity": int(cfg.padded_mlp_max_nodes),
        "observed_max_control_nodes": max(record["control_nodes"] for record in records),
        "training_topologies": configured_topologies(cfg),
        "evaluation_topologies": list(getattr(cfg, "eval_extra_topologies", None) or []),
        "evaluation_task": getattr(cfg, "eval_task", None),
        "effective_budgets": {name: int(getattr(cfg, name)) for name in ("steps", "batch_size", "buffer_size")},
        "per_topology_budgets": {name: int(getattr(cfg, "per_topology_" + name)) for name in ("steps", "batch_size", "buffer_size")},
        "parameters": parameter_counts(PaddedMLPActorCritic(cfg)),
        "contracts": records,
    }, indent=2))


if __name__ == "__main__":
    main()
