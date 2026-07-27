from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer


def make_graph(
    node_count: int,
    feature_dim: int,
    generator: torch.Generator,
) -> Data:
    nodes = torch.arange(node_count, dtype=torch.long)
    next_nodes = nodes.roll(-1)
    edge_index = torch.stack(
        [
            torch.cat([nodes, next_nodes]),
            torch.cat([next_nodes, nodes]),
        ],
        dim=0,
    )
    return Data(
        x=torch.randn(node_count, feature_dim, generator=generator),
        edge_index=edge_index,
    )


def transition(
    node_count: int,
    feature_dim: int,
    generator: torch.Generator,
) -> list[dict[str, torch.Tensor | Data]]:
    observation = make_graph(node_count, feature_dim, generator)
    next_observation = make_graph(node_count, feature_dim, generator)
    action = torch.randn(1, node_count, 1, generator=generator)
    reward = torch.randn(1, generator=generator)
    terminated = torch.zeros(1)
    return [
        {
            "obs": observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
        },
        {
            "obs": next_observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
        },
    ]


def make_buffer(args: argparse.Namespace, node_counts: list[int]) -> GNNBuffer:
    task_names = [f"benchmark:task-{index}" for index in range(len(node_counts))]
    batch_size_per_task = args.batch_size // len(task_names)
    capacity_per_task = max(batch_size_per_task * 2, args.entries_per_task)
    config = SimpleNamespace(
        device=args.device,
        task="benchmark",
        tasks=task_names,
        multitask=True,
        mujoco_backend="mujoco",
        truss_topologies=None,
        buffer_size=capacity_per_task * len(task_names),
        batch_size=args.batch_size,
        steps=capacity_per_task * len(task_names),
        use_virtual_node=args.virtual_node,
    )
    buffer = GNNBuffer(config)
    generator = torch.Generator().manual_seed(args.seed)
    for task, node_count in zip(task_names, node_counts):
        for _ in range(capacity_per_task):
            buffer.add(
                transition(node_count, args.feature_dim, generator),
                task=task,
            )
    return buffer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def legacy_sample(buffer: GNNBuffer):
    task_batches = buffer.sample_task_batches()
    return GNNBuffer.combine_task_batches(task_batches)


def assert_batches_equal(first, second) -> None:
    for first_value, second_value in zip(first, second):
        if hasattr(first_value, "to_dict"):
            first_values = first_value.to_dict()
            second_values = second_value.to_dict()
            if first_values.keys() != second_values.keys():
                raise RuntimeError("Legacy and optimized graph fields differ")
            values = zip(first_values.values(), second_values.values())
        else:
            values = [(first_value, second_value)]
        for first_tensor, second_tensor in values:
            if not torch.equal(first_tensor, second_tensor):
                raise RuntimeError(
                    "Legacy and optimized replay batches are not exactly equal"
                )


def measure_call(call, device: torch.device) -> float:
    synchronize(device)
    started_at = time.perf_counter()
    call()
    synchronize(device)
    return time.perf_counter() - started_at


def summarize(samples: list[float], batch_size: int) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    mean_seconds = float(values.mean())
    return {
        "count": int(values.size),
        "mean_ms": 1000.0 * mean_seconds,
        "median_ms": 1000.0 * float(np.median(values)),
        "p95_ms": 1000.0 * float(np.percentile(values, 95)),
        "max_ms": 1000.0 * float(values.max()),
        "batches_per_second": 1.0 / mean_seconds,
        "transitions_per_second": batch_size / mean_seconds,
    }


def benchmark(args: argparse.Namespace) -> dict:
    node_counts = [int(value) for value in args.node_counts.split(",")]
    if not node_counts or any(value <= 0 for value in node_counts):
        raise ValueError("--node-counts must contain positive integers")
    if args.batch_size <= 0 or args.batch_size % len(node_counts) != 0:
        raise ValueError("--batch-size must be positive and divisible by task count")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    buffer = make_buffer(args, node_counts)
    torch.manual_seed(args.seed + 1)
    legacy_batch = legacy_sample(buffer)
    torch.manual_seed(args.seed + 1)
    optimized_batch = buffer.sample()
    assert_batches_equal(legacy_batch, optimized_batch)

    for iteration in range(args.warmup):
        seed = args.seed + 10_000 + iteration
        torch.manual_seed(seed)
        legacy_sample(buffer)
        torch.manual_seed(seed)
        buffer.sample()
    synchronize(device)

    legacy_samples = []
    optimized_samples = []
    for iteration in range(args.repeats):
        seed = args.seed + 20_000 + iteration
        modes = (
            (("legacy", legacy_sample), ("optimized", buffer.sample))
            if iteration % 2 == 0
            else (("optimized", buffer.sample), ("legacy", legacy_sample))
        )
        for mode, call in modes:
            torch.manual_seed(seed)
            elapsed = measure_call(lambda: call(buffer) if mode == "legacy" else call(), device)
            if mode == "legacy":
                legacy_samples.append(elapsed)
            else:
                optimized_samples.append(elapsed)

    legacy = summarize(legacy_samples, args.batch_size)
    optimized = summarize(optimized_samples, args.batch_size)
    return {
        "device": str(device),
        "batch_size": args.batch_size,
        "task_count": len(node_counts),
        "node_counts": node_counts,
        "feature_dim": args.feature_dim,
        "virtual_node": args.virtual_node,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "batch_equivalence": "exact",
        "legacy": legacy,
        "optimized": optimized,
        "speedup": legacy["mean_ms"] / optimized["mean_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy and direct-collation GNN replay sampling."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=255)
    parser.add_argument("--node-counts", default="4,6,6")
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--entries-per-task", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--virtual-node", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = benchmark(args)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
