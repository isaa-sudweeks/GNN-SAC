from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from types import SimpleNamespace
import sys
import tempfile
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
from common.tensor_gnn_buffer import TensorGNNBuffer
from gnn_sac import GNNSAC


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


def make_config(args: argparse.Namespace, node_counts: list[int], backend: str):
    task_names = [f"benchmark:task-{index}" for index in range(len(node_counts))]
    batch_size_per_task = args.batch_size // len(task_names)
    capacity_per_task = max(batch_size_per_task * 2, args.entries_per_task)
    return SimpleNamespace(
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
        obs_dim=args.feature_dim,
        action_dim=1,
        node_counts=node_counts,
        replay_backend=backend,
        replay_storage=args.storage,
        replay_gpu_fraction=args.replay_gpu_fraction,
        replay_gpu_max_gb=args.replay_gpu_max_gb,
        replay_gpu_reserve_gb=args.replay_gpu_reserve_gb,
        graph_features={},
        embedding_dim=64,
        mlp_dim=64,
        dropout=0.0,
        Q_output_dim=64,
        head_hidden_dims=[64],
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy="auto",
        num_policy_actions=max(node_counts),
        episode_length=100,
        discount_denom=500,
        discount_min=.95,
        discount_max=.995,
        tau=.005,
        grad_clip_norm=10.,
        pcgrad=False,
        gradient_diagnostics=False,
    )


def make_buffer(args: argparse.Namespace, node_counts: list[int], backend: str):
    config = make_config(args, node_counts, backend)
    buffer = GNNBuffer(config) if backend == "legacy" else TensorGNNBuffer(config)
    generator = torch.Generator().manual_seed(args.seed)
    started_at = time.perf_counter()
    for task, node_count in zip(config.tasks, node_counts):
        for _ in range(config.buffer_size // len(config.tasks)):
            buffer.add(
                transition(node_count, args.feature_dim, generator),
                task=task,
            )
    synchronize(torch.device(args.device))
    return buffer, time.perf_counter() - started_at


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


def checkpoint_benchmark(buffer, args, node_counts, backend, directory):
    clone_samples, serialize_samples, deserialize_samples = [], [], []
    restore_samples, sizes = [], []
    for iteration in range(args.checkpoint_repeats):
        path = Path(directory) / f"{backend}-{iteration}.pt"
        holder = {}
        clone_samples.append(measure_call(lambda: holder.update(state=buffer.state_dict()), torch.device(args.device)))
        serialize_samples.append(measure_call(lambda: torch.save(holder["state"], path), torch.device(args.device)))
        sizes.append(path.stat().st_size)
        loaded = {}
        deserialize_samples.append(measure_call(
            lambda: loaded.update(state=torch.load(path, map_location="cpu", weights_only=False)),
            torch.device(args.device),
        ))
        def restore_storage():
            restored = (
                GNNBuffer(make_config(args, node_counts, backend))
                if backend == "legacy"
                else TensorGNNBuffer(make_config(args, node_counts, backend))
            )
            restored.load_state_dict(loaded["state"])
        restore_samples.append(measure_call(restore_storage, torch.device(args.device)))
    return {
        "snapshot_clone": summarize(clone_samples, buffer.size),
        "serialization": summarize(serialize_samples, buffer.size),
        "deserialization": summarize(deserialize_samples, buffer.size),
        "device_restoration": summarize(restore_samples, buffer.size),
        "full_save_median_ms": summarize(clone_samples, buffer.size)["median_ms"] + summarize(serialize_samples, buffer.size)["median_ms"],
        "full_load_median_ms": summarize(deserialize_samples, buffer.size)["median_ms"] + summarize(restore_samples, buffer.size)["median_ms"],
        "file_bytes": sizes,
    }


def optimizer_benchmark(legacy_buffer, tensor_buffer, args, node_counts, device):
    cfg = make_config(args, node_counts, "legacy")
    torch.manual_seed(args.seed + 50_000)
    legacy_agent = GNNSAC(cfg)
    tensor_agent = GNNSAC(make_config(args, node_counts, "torchrl_tensor"))
    tensor_agent.load_training_state_dict(legacy_agent.training_state_dict())
    for iteration in range(min(args.warmup, 5)):
        seed = args.seed + 55_000 + iteration
        torch.manual_seed(seed)
        legacy_agent.update(legacy_buffer)
        torch.manual_seed(seed)
        tensor_agent.update(tensor_buffer)
    legacy_samples, tensor_samples = [], []
    for iteration in range(args.optimizer_repeats):
        seed = args.seed + 60_000 + iteration
        torch.manual_seed(seed)
        legacy_samples.append(measure_call(lambda: legacy_agent.update(legacy_buffer), device))
        torch.manual_seed(seed)
        tensor_samples.append(measure_call(lambda: tensor_agent.update(tensor_buffer), device))
    return {
        "legacy": summarize(legacy_samples, args.batch_size),
        "torchrl_tensor": summarize(tensor_samples, args.batch_size),
        "throughput_gain_fraction": (
            summarize(tensor_samples, args.batch_size)["batches_per_second"]
            / summarize(legacy_samples, args.batch_size)["batches_per_second"] - 1.0
        ),
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

    buffer, insertion_seconds = make_buffer(args, node_counts, "legacy")
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
    result = {
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
        "insertion": {"legacy_seconds": insertion_seconds},
    }
    if args.prototype:
        tensor, tensor_insertion_seconds = make_buffer(args, node_counts, "torchrl_tensor")
        torch.manual_seed(args.seed + 2)
        expected = buffer.sample()
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(args.seed + 2)
        actual = tensor.sample()
        actual_rng = torch.random.get_rng_state()
        assert_batches_equal(expected, actual)
        if not torch.equal(expected_rng, actual_rng):
            raise RuntimeError("Tensor coordinator changed the Torch RNG state")

        tensor_direct_samples, tensor_ensemble_samples = [], []
        ensemble_exact = None
        ensemble_error = None
        try:
            torch.manual_seed(args.seed + 3)
            direct = tensor.sample()
            direct_rng = torch.random.get_rng_state()
            torch.manual_seed(args.seed + 3)
            ensemble = tensor.sample_ensemble()
            ensemble_rng = torch.random.get_rng_state()
            assert_batches_equal(direct, ensemble)
            if not torch.equal(direct_rng, ensemble_rng):
                raise RuntimeError("ReplayBufferEnsemble changed the Torch RNG state")
            ensemble_exact = True
        except (RuntimeError, ValueError) as error:
            ensemble_exact, ensemble_error = False, str(error)

        for iteration in range(args.warmup):
            torch.manual_seed(args.seed + 30_000 + iteration)
            tensor.sample()
            if ensemble_exact:
                torch.manual_seed(args.seed + 30_000 + iteration)
                tensor.sample_ensemble()
        for iteration in range(args.repeats):
            seed = args.seed + 40_000 + iteration
            torch.manual_seed(seed)
            tensor_direct_samples.append(measure_call(tensor.sample, device))
            if ensemble_exact:
                torch.manual_seed(seed)
                tensor_ensemble_samples.append(measure_call(tensor.sample_ensemble, device))

        tensor_direct = summarize(tensor_direct_samples, args.batch_size)
        ensemble_summary = summarize(tensor_ensemble_samples, args.batch_size) if tensor_ensemble_samples else None
        ensemble_overhead = (
            ensemble_summary["median_ms"] / tensor_direct["median_ms"] - 1.0
            if ensemble_summary is not None else None
        )
        result["insertion"]["torchrl_tensor_seconds"] = tensor_insertion_seconds
        result["tensor_direct"] = tensor_direct
        result["tensor_equivalence"] = "exact"
        result["ensemble"] = {
            "exact": ensemble_exact,
            "error": ensemble_error,
            "timing": ensemble_summary,
            "median_overhead_fraction": ensemble_overhead,
            "adopt": bool(ensemble_exact and ensemble_overhead is not None and ensemble_overhead <= .05),
        }
        result["placement"] = tensor.runtime_storage_metadata()
        if args.optimizer_repeats:
            result["optimizer_update"] = optimizer_benchmark(
                buffer, tensor, args, node_counts, device
            )
        with tempfile.TemporaryDirectory(prefix="gnn-replay-benchmark-") as directory:
            result["checkpoint"] = {
                "legacy": checkpoint_benchmark(buffer, args, node_counts, "legacy", directory),
                "torchrl_tensor": checkpoint_benchmark(tensor, args, node_counts, "torchrl_tensor", directory),
            }
        result["process_peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if device.type == "cuda":
            result["cuda"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy and TorchRL tensorized GNN replay."
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
    parser.add_argument("--prototype", action="store_true")
    parser.add_argument("--storage", choices=("cpu_pinned", "cuda", "auto"), default="cpu_pinned")
    parser.add_argument("--replay-gpu-fraction", type=float, default=.20)
    parser.add_argument("--replay-gpu-max-gb", type=float, default=8.)
    parser.add_argument("--replay-gpu-reserve-gb", type=float, default=12.)
    parser.add_argument("--checkpoint-repeats", type=int, default=3)
    parser.add_argument("--optimizer-repeats", type=int, default=0)
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
