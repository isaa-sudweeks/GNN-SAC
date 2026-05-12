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


def load_cfg(args: argparse.Namespace, num_envs: int, backend: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.create(
            {
                "task": args.task,
                "env_name": args.task,
                "save_video": False,
                "multitask": False,
                "num_envs": num_envs,
                "mujoco_backend": backend,
                "nsubsteps": args.nsubsteps,
                "max_steps": max(args.steps_per_env + args.warmup + 1, args.max_steps),
                "episode_length": max(args.steps_per_env + args.warmup + 1, args.max_steps),
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "work_dir": str(ROOT / "logs" / "benchmark-multi-env"),
            }
        ),
    )
    if args.xml_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"xml_path": args.xml_path}))
    return parse_cfg(cfg)


def step_all_envs(env, num_envs: int):
    for env_idx in range(num_envs):
        if num_envs > 1:
            env.set_active_env(env_idx)
        env.step(env.rand_act())


def reset_all_envs(env, num_envs: int):
    for env_idx in range(num_envs):
        if num_envs > 1:
            env.reset(task_idx=env_idx)
        else:
            env.reset()


def run_case(args: argparse.Namespace, num_envs: int, backend: str):
    cfg = load_cfg(args, num_envs, backend)
    env = make_env(cfg)
    try:
        reset_all_envs(env, num_envs)
        for _ in range(args.warmup):
            step_all_envs(env, num_envs)

        start = time.perf_counter()
        for _ in range(args.steps_per_env):
            step_all_envs(env, num_envs)
        elapsed = time.perf_counter() - start

        total_env_steps = args.steps_per_env * num_envs
        total_physics_steps = total_env_steps * args.nsubsteps
        return {
            "elapsed": elapsed,
            "env_steps_per_sec": total_env_steps / elapsed,
            "per_env_steps_per_sec": args.steps_per_env / elapsed,
            "physics_steps_per_sec": total_physics_steps / elapsed,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the repo's current serial multi-environment training step pattern."
    )
    parser.add_argument("--task", default="truss-velocity-command-right")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--backend", choices=("both", "mjx", "mujoco"), default="both")
    parser.add_argument("--steps-per-env", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--nsubsteps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--xml-path", default=None)
    args = parser.parse_args()

    backends = ("mjx", "mujoco") if args.backend == "both" else (args.backend,)
    print(
        "backend,num_envs,elapsed_s,total_env_steps_s,per_env_steps_s,total_physics_steps_s"
    )
    for backend in backends:
        for num_envs in args.num_envs:
            result = run_case(args, num_envs, backend)
            print(
                f"{backend},{num_envs},{result['elapsed']:.3f},"
                f"{result['env_steps_per_sec']:.2f},"
                f"{result['per_env_steps_per_sec']:.2f},"
                f"{result['physics_steps_per_sec']:.2f}"
            )


if __name__ == "__main__":
    main()
