from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC


def make_cfg(args: argparse.Namespace):
    return SimpleNamespace(
        device=args.device,
        obs_dim=args.obs_dim,
        embedding_dim=args.embedding_dim,
        mlp_dim=args.mlp_dim,
        dropout=0.0,
        action_dim=args.action_dim,
        Q_output_dim=args.embedding_dim,
        head_hidden_dims=[args.mlp_dim, args.mlp_dim],
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy="auto",
        num_policy_actions=args.nodes * args.action_dim,
        episode_length=5_000,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        tau=0.005,
        grad_clip_norm=10.0,
    )


def make_graph(num_nodes: int, obs_dim: int, generator: torch.Generator) -> Data:
    source = torch.arange(num_nodes, dtype=torch.long)
    target = source.roll(-1)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])],
        dim=0,
    )
    return Data(x=torch.randn(num_nodes, obs_dim, generator=generator), edge_index=edge_index)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    synchronize(device)
    return time.perf_counter() - start


def benchmark_mode(agent, args, batch_size: int, mixed_sizes: bool) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(args.seed + batch_size + int(mixed_sizes))
    observations = [
        make_graph(args.nodes + (index % 3 if mixed_sizes else 0), args.obs_dim, generator)
        for index in range(batch_size)
    ]
    device = torch.device(args.device)
    kwargs = {"eval_mode": args.deterministic}

    serialized_elapsed = measure(
        lambda: [agent.act(observation, **kwargs) for observation in observations],
        device,
        args.warmup,
        args.repeats,
    )
    batched_elapsed = measure(
        lambda: agent.act_batch(observations, **kwargs),
        device,
        args.warmup,
        args.repeats,
    )
    graph_count = batch_size * args.repeats
    return graph_count / serialized_elapsed, graph_count / batched_elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare serialized and batched GNN actor inference.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", default="1,8,64,256")
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--obs-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(batch_size < 1 for batch_size in batch_sizes):
        parser.error("All batch sizes must be positive")

    torch.manual_seed(args.seed)
    agent = GNNSAC(make_cfg(args))
    print("mode   batch serialized_graphs/s batched_graphs/s speedup")
    for mixed_sizes in (False, True):
        mode = "mixed" if mixed_sizes else "same"
        for batch_size in batch_sizes:
            serialized, batched = benchmark_mode(agent, args, batch_size, mixed_sizes)
            print(f"{mode:5s} {batch_size:5d} {serialized:19.1f} {batched:16.1f} {batched / serialized:7.2f}x")


if __name__ == "__main__":
    main()
