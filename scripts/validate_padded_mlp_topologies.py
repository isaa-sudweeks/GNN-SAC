#!/usr/bin/env python3
"""Validate the fixed padded-MLP capacity against the thesis topology set."""

from pathlib import Path
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import parse_cfg
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
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "sac_backend" / "padded_mlp.yaml"),
        OmegaConf.create(
            {
                "truss_topology": topology,
                "max_steps": 1,
                "nsubsteps": 1,
                "domain_randomization": False,
                "save_video": False,
                "work_dir": "/tmp/padded-mlp-topology-audit",
                "device": "cpu",
            }
        ),
    )
    return parse_cfg(config)


def main():
    failures = []
    backend_config = OmegaConf.load(
        ROOT / "config" / "sac_backend" / "padded_mlp.yaml"
    )
    capacity = int(backend_config.padded_mlp_max_nodes)
    print("topology,abstract_nodes,control_nodes,active_actions")
    for topology, expected in EXPECTED.items():
        env = make_env(make_config(topology))
        try:
            observation = env.reset()
            actual = (
                len(env.unwrapped.mj_model.node_names),
                int(observation.num_nodes),
                int(observation.action_mask.sum()),
            )
        finally:
            env.close()
        print(f"{topology},{actual[0]},{actual[1]},{actual[2]}")
        if actual != expected:
            failures.append(f"{topology}: expected {expected}, got {actual}")

    observed_max = max(value[1] for value in EXPECTED.values())
    if observed_max != capacity:
        failures.append(
            f"Configured padded_mlp_max_nodes={capacity}, but the frozen topology "
            f"table requires {observed_max}."
        )
    if failures:
        raise SystemExit("Padded MLP topology validation failed:\n" + "\n".join(failures))
    print(f"validated_capacity={capacity}")


if __name__ == "__main__":
    main()
