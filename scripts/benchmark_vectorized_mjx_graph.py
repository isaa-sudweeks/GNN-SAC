from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import parse_cfg  # noqa: E402
from env import make_env  # noqa: E402


def load_cfg(args):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "gnn_config.yaml"),
        OmegaConf.create(
            {
                "task": args.task,
                "env_name": args.task,
                "save_video": False,
                "multitask": False,
                "num_envs": args.num_envs,
                "mujoco_backend": "mjx",
                "mjx_vectorized": True,
                "nsubsteps": args.nsubsteps,
                "max_steps": max(args.steps + args.warmup + 1, args.max_steps),
                "episode_length": max(args.steps + args.warmup + 1, args.max_steps),
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "work_dir": str(ROOT / "logs" / "benchmark-vectorized-mjx-graph"),
            }
        ),
    )
    return parse_cfg(cfg)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the batched MJX graph rollout path.")
    parser.add_argument("--task", default="octahedron-graph-right", choices=("octahedron-graph-right", "tetrehedron-graph-right"))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--nsubsteps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_cfg(args)
    env = make_env(cfg)
    try:
        env.reset_many(range(args.num_envs))
        actions = [env.rand_act() for _ in range(args.num_envs)]
        for _ in range(args.warmup):
            env.step_many(actions, range(args.num_envs))

        start = time.perf_counter()
        for _ in range(args.steps):
            env.step_many(actions, range(args.num_envs))
        elapsed = time.perf_counter() - start

        env_steps = args.steps * args.num_envs
        physics_steps = env_steps * args.nsubsteps
        print(
            f"task={args.task} num_envs={args.num_envs} elapsed={elapsed:.3f}s "
            f"env_steps/s={env_steps / elapsed:.2f} physics_steps/s={physics_steps / elapsed:.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
