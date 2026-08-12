from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import parse_cfg  # noqa: E402
from env import make_env  # noqa: E402


def make_cfg(args: argparse.Namespace, implementation: str):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "sac_backend" / "gnn.yaml"),
        OmegaConf.load(ROOT / "config" / "sim_backend" / "mjx.yaml"),
        OmegaConf.create(
            {
                "task": "truss-graph",
                "truss_topology": args.topology,
                "truss_realistic": args.realistic,
                "num_envs": args.num_envs,
                "nsubsteps": args.nsubsteps,
                "max_steps": args.warmup + args.steps + 1,
                "domain_randomization": False,
                "save_video": False,
                "enable_wandb": False,
                "device": args.device,
                "work_dir": str(ROOT / "logs" / "benchmark-mjx-implementations"),
                "mjx_impl": implementation,
                "warp_graph_mode": args.warp_graph_mode,
                "warp_naconmax": args.warp_naconmax,
                "warp_njmax": args.warp_njmax,
            }
        ),
    )
    return parse_cfg(cfg)


def synchronize(env) -> None:
    core_env = env.env
    core_env._jax.block_until_ready(core_env._state.data.qpos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(args: argparse.Namespace, implementation: str) -> dict[str, object]:
    cfg = make_cfg(args, implementation)
    env = make_env(cfg)
    try:
        env.reset_many()
        actions = [env.rand_act(env_idx=index) for index in range(args.num_envs)]

        compile_start = time.perf_counter()
        env.step_many(actions)
        synchronize(env)
        first_step_seconds = time.perf_counter() - compile_start

        for _ in range(max(args.warmup - 1, 0)):
            env.step_many(actions)
        synchronize(env)

        start = time.perf_counter()
        for _ in range(args.steps):
            env.step_many(actions)
        synchronize(env)
        elapsed = time.perf_counter() - start

        transitions = args.steps * args.num_envs
        return {
            "implementation": implementation,
            "topology": args.topology,
            "realistic": args.realistic,
            "num_envs": args.num_envs,
            "nsubsteps": args.nsubsteps,
            "warmup_steps": args.warmup,
            "measured_vector_steps": args.steps,
            "first_step_seconds": first_step_seconds,
            "elapsed_seconds": elapsed,
            "vector_steps_per_second": args.steps / elapsed,
            "transitions_per_second": transitions / elapsed,
            "physics_steps_per_second": transitions * args.nsubsteps / elapsed,
            "jax_device": str(env.env._jax.devices()[0]),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX and Warp through GNN-SAC's batch-native MJX adapter."
    )
    parser.add_argument("--implementation", choices=("both", "jax", "warp"), default="both")
    parser.add_argument("--topology", default="octahedron")
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--nsubsteps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--warp-graph-mode",
        choices=("warp", "warp_staged", "warp_staged_ex"),
        default="warp_staged",
    )
    parser.add_argument("--warp-naconmax", type=int, default=None)
    parser.add_argument("--warp-njmax", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    implementations = ("jax", "warp") if args.implementation == "both" else (args.implementation,)
    results = [benchmark(args, implementation) for implementation in implementations]
    if len(results) == 2:
        results[1]["speedup_vs_jax"] = (
            float(results[1]["transitions_per_second"])
            / float(results[0]["transitions_per_second"])
        )

    output = {"benchmark": "gnn_sac_mjx_implementations", "results": results}
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
