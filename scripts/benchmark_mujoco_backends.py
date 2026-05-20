from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env.mujoco  # noqa: E402,F401 - registers custom MuJoCo environments


TRUSS_TASKS = {
    "truss-velocity-command-right": "MujocoVelocityCommandEnvRight-v0",
    "truss-velocity-command-left": "MujocoVelocityCommandEnvLeft-v0",
    "truss-velocity-command-up": "MujocoVelocityCommandEnvUp-v0",
    "truss-velocity-command-down": "MujocoVelocityCommandEnvDown-v0",
}


def load_cfg(args: argparse.Namespace, backend: str):
    cfg = OmegaConf.load(ROOT / "config" / "environment.yaml")
    overrides = {
        "task": args.task,
        "save_video": False,
        "mujoco_backend": backend,
        "nsubsteps": args.nsubsteps,
        "max_steps": max(args.steps + args.warmup + 1, int(cfg.max_steps)),
    }
    if args.xml_path is not None:
        overrides["xml_path"] = args.xml_path
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def make_env(task: str, cfg):
    if task not in TRUSS_TASKS:
        raise ValueError(
            "This benchmark targets the local truss MuJoCo envs. "
            f"Supported tasks: {', '.join(sorted(TRUSS_TASKS))}"
        )
    return gym.make(TRUSS_TASKS[task], config=cfg, render_mode=None)


def run_backend(args: argparse.Namespace, backend: str):
    cfg = load_cfg(args, backend)
    env = make_env(args.task, cfg)
    rng = np.random.default_rng(args.seed)
    try:
        env.reset(seed=args.seed)

        for _ in range(args.warmup):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            env.step(action)

        start = time.perf_counter()
        for _ in range(args.steps):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            env.step(action)
        elapsed = time.perf_counter() - start

        steps_per_sec = args.steps / elapsed
        physics_steps_per_sec = (args.steps * args.nsubsteps) / elapsed
        return steps_per_sec, physics_steps_per_sec, elapsed
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark native MuJoCo vs MJX for local truss envs.")
    parser.add_argument("--task", default="truss-velocity-command-right", choices=sorted(TRUSS_TASKS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--nsubsteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--backend", choices=("both", "mjx", "mujoco"), default="both")
    args = parser.parse_args()

    backends = ("mjx", "mujoco") if args.backend == "both" else (args.backend,)
    for backend in backends:
        steps_per_sec, physics_steps_per_sec, elapsed = run_backend(args, backend)
        print(
            f"{backend:6s} elapsed={elapsed:.3f}s "
            f"env_steps/s={steps_per_sec:.2f} physics_steps/s={physics_steps_per_sec:.2f}"
        )


if __name__ == "__main__":
    main()
