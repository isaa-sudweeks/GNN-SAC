"""Validate replay performance and learner equivalence from real checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from common.tensor_gnn_buffer import TensorGNNBuffer
from gnn_sac import GNNSAC


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def numeric_summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(samples),
        "mean_ms": 1000.0 * statistics.fmean(samples),
        "median_ms": 1000.0 * statistics.median(samples),
        "p95_ms": 1000.0 * ordered[p95_index],
        "max_ms": 1000.0 * max(samples),
    }


def make_config(state: dict, backend: str, storage: str) -> SimpleNamespace:
    values = dict(state["config"])
    values.update(device="cuda", replay_backend=backend, replay_storage=storage)
    return SimpleNamespace(**values)


def load_case(path: Path, backend: str, storage: str):
    started = time.perf_counter()
    state = torch.load(path, map_location="cpu", weights_only=False)
    deserialize_seconds = time.perf_counter() - started
    cfg = make_config(state, backend, storage)
    buffer = GNNBuffer(cfg) if backend == "legacy" else TensorGNNBuffer(cfg)
    agent = GNNSAC(cfg)
    started = time.perf_counter()
    buffer.load_state_dict(state["buffer"])
    agent.load_training_state_dict(state["agent"])
    synchronize()
    restore_seconds = time.perf_counter() - started
    agent_state = deepcopy(state["agent"])
    del state
    return cfg, buffer, agent, agent_state, deserialize_seconds, restore_seconds


def assert_equal(expected, actual, path="root") -> None:
    if isinstance(expected, torch.Tensor):
        if not torch.equal(expected, actual):
            raise RuntimeError(f"Tensor mismatch at {path}")
        return
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError(f"Key mismatch at {path}")
        for key in expected:
            assert_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"Length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equal(left, right, f"{path}[{index}]")
        return
    if expected != actual:
        raise RuntimeError(f"Value mismatch at {path}: {expected!r} != {actual!r}")


def assert_close(expected, actual, *, rtol: float, atol: float, path="root") -> None:
    if isinstance(expected, torch.Tensor):
        if expected.is_floating_point() or expected.is_complex():
            try:
                torch.testing.assert_close(
                    expected, actual, rtol=rtol, atol=atol, equal_nan=True
                )
            except AssertionError as error:
                raise RuntimeError(f"Numerical mismatch at {path}: {error}") from error
        elif not torch.equal(expected, actual):
            raise RuntimeError(f"Tensor mismatch at {path}")
        return
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError(f"Key mismatch at {path}")
        for key in expected:
            assert_close(
                expected[key], actual[key], rtol=rtol, atol=atol, path=f"{path}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"Length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_close(
                left, right, rtol=rtol, atol=atol, path=f"{path}[{index}]"
            )
        return
    if expected != actual:
        raise RuntimeError(f"Value mismatch at {path}: {expected!r} != {actual!r}")


def max_float_drift(expected, actual) -> tuple[float, float]:
    if isinstance(expected, torch.Tensor):
        if not (expected.is_floating_point() or expected.is_complex()) or not expected.numel():
            return 0.0, 0.0
        difference = (expected - actual).abs()
        scale = torch.maximum(expected.abs(), actual.abs()).clamp_min(torch.finfo(expected.dtype).eps)
        return float(difference.max().detach().cpu()), float((difference / scale).max().detach().cpu())
    if isinstance(expected, dict):
        values = [max_float_drift(expected[key], actual[key]) for key in expected]
    elif isinstance(expected, (list, tuple)):
        values = [max_float_drift(left, right) for left, right in zip(expected, actual)]
    else:
        return 0.0, 0.0
    return max((value[0] for value in values), default=0.0), max(
        (value[1] for value in values), default=0.0
    )


def assert_batch_equal(expected, actual) -> None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if hasattr(left, "to_dict"):
            assert_equal(left.to_dict(), right.to_dict(), f"batch[{index}]")
        else:
            assert_equal(left, right, f"batch[{index}]")


def performance(args: argparse.Namespace) -> dict:
    cfg, buffer, warmup_agent, initial_agent_state, load_seconds, restore_seconds = load_case(
        args.checkpoint, args.backend, args.storage
    )
    for index in range(args.warmup):
        torch.manual_seed(args.seed + index)
        warmup_agent.update(buffer)
    synchronize()

    sample_times = []
    for index in range(args.sample_repeats):
        torch.manual_seed(args.seed + 10_000 + index)
        synchronize()
        started = time.perf_counter()
        sample = buffer.sample()
        synchronize()
        sample_times.append(time.perf_counter() - started)
        del sample

    update_blocks = []
    for block in range(args.blocks):
        agent = GNNSAC(cfg)
        agent.load_training_state_dict(deepcopy(initial_agent_state))
        elapsed = []
        for index in range(args.updates):
            torch.manual_seed(args.seed + 20_000 + index)
            synchronize()
            started = time.perf_counter()
            agent.update(buffer)
            synchronize()
            elapsed.append(time.perf_counter() - started)
        update_blocks.append({
            "block": block,
            "timing": numeric_summary(elapsed),
            "updates_per_second": len(elapsed) / sum(elapsed),
        })

    return {
        "mode": "performance",
        "backend": args.backend,
        "storage": args.storage,
        "checkpoint": str(args.checkpoint),
        "buffer_size": buffer.size,
        "sizes_by_task": buffer.sizes_by_task,
        "deserialize_seconds": load_seconds,
        "restore_seconds": restore_seconds,
        "sample": numeric_summary(sample_times),
        "update_blocks": update_blocks,
        "median_block_updates_per_second": statistics.median(
            block["updates_per_second"] for block in update_blocks
        ),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "placement": (
            buffer.runtime_storage_metadata()
            if isinstance(buffer, TensorGNNBuffer) else {"mode": "legacy", "placements": {}}
        ),
    }


def correctness(args: argparse.Namespace) -> dict:
    left_checkpoint = (
        args.tensor_checkpoint if args.same_backend_control else args.legacy_checkpoint
    )
    left_backend = "torchrl_tensor" if args.same_backend_control else "legacy"
    left_storage = "auto" if args.same_backend_control else "cpu_pinned"
    _, legacy_buffer, legacy_agent, _, legacy_load, legacy_restore = load_case(
        left_checkpoint, left_backend, left_storage
    )
    _, tensor_buffer, tensor_agent, _, tensor_load, tensor_restore = load_case(
        args.tensor_checkpoint, "torchrl_tensor", "auto"
    )
    assert_equal(
        legacy_agent.training_state_dict(), tensor_agent.training_state_dict(), "initial_agent"
    )

    for index in range(args.sample_repeats):
        seed = args.seed + 30_000 + index
        torch.manual_seed(seed)
        expected = legacy_buffer.sample()
        expected_cpu_rng = torch.random.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)
        actual = tensor_buffer.sample()
        assert_batch_equal(expected, actual)
        if not torch.equal(expected_cpu_rng, torch.random.get_rng_state()):
            raise RuntimeError(f"CPU RNG mismatch after sample {index}")
        assert_equal(expected_cuda_rng, torch.cuda.get_rng_state_all(), f"sample_cuda_rng[{index}]")

    started = time.perf_counter()
    drift = []
    first_tolerance_violation = None
    max_metric_absolute = 0.0
    max_metric_relative = 0.0
    for index in range(args.correctness_updates):
        seed = args.seed + 40_000 + index
        torch.manual_seed(seed)
        expected_metrics = legacy_agent.update(legacy_buffer)
        expected_cpu_rng = torch.random.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)
        actual_metrics = tensor_agent.update(tensor_buffer)
        metric_absolute, metric_relative = max_float_drift(
            expected_metrics, actual_metrics
        )
        max_metric_absolute = max(max_metric_absolute, metric_absolute)
        max_metric_relative = max(max_metric_relative, metric_relative)
        try:
            assert_close(
                expected_metrics,
                actual_metrics,
                rtol=args.rtol,
                atol=args.atol,
                path=f"metrics[{index}]",
            )
        except RuntimeError as error:
            if first_tolerance_violation is None:
                first_tolerance_violation = str(error)
        if not torch.equal(expected_cpu_rng, torch.random.get_rng_state()):
            raise RuntimeError(f"CPU RNG mismatch after update {index}")
        assert_equal(expected_cuda_rng, torch.cuda.get_rng_state_all(), f"update_cuda_rng[{index}]")
        if (index + 1) % 100 == 0:
            expected_state = legacy_agent.training_state_dict()
            actual_state = tensor_agent.training_state_dict()
            max_absolute, max_relative = max_float_drift(expected_state, actual_state)
            try:
                assert_close(
                    expected_state,
                    actual_state,
                    rtol=args.rtol,
                    atol=args.atol,
                    path=f"agent_after_{index + 1}",
                )
            except RuntimeError as error:
                if first_tolerance_violation is None:
                    first_tolerance_violation = str(error)
            drift.append({
                "update": index + 1,
                "max_absolute": max_absolute,
                "max_relative": max_relative,
            })
    synchronize()
    expected_final = legacy_agent.training_state_dict()
    actual_final = tensor_agent.training_state_dict()
    try:
        assert_close(
            expected_final,
            actual_final,
            rtol=args.rtol,
            atol=args.atol,
            path="final_agent",
        )
    except RuntimeError as error:
        if first_tolerance_violation is None:
            first_tolerance_violation = str(error)
    return {
        "mode": "correctness",
        "comparison": (
            "tensor_auto_vs_tensor_auto_control"
            if args.same_backend_control else "legacy_vs_tensor_auto"
        ),
        "status": (
            "numerically_equivalent"
            if first_tolerance_violation is None else "tolerance_exceeded"
        ),
        "rtol": args.rtol,
        "atol": args.atol,
        "first_tolerance_violation": first_tolerance_violation,
        "max_metric_absolute_drift": max_metric_absolute,
        "max_metric_relative_drift": max_metric_relative,
        "sample_comparisons": args.sample_repeats,
        "updates_completed": args.correctness_updates,
        "agent_state_drift": drift,
        "elapsed_seconds": time.perf_counter() - started,
        "buffer_size": legacy_buffer.size,
        "sizes_by_task": legacy_buffer.sizes_by_task,
        "legacy_deserialize_seconds": legacy_load,
        "legacy_restore_seconds": legacy_restore,
        "tensor_deserialize_seconds": tensor_load,
        "tensor_restore_seconds": tensor_restore,
        "tensor_placement": tensor_buffer.runtime_storage_metadata(),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("performance", "correctness"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--tensor-checkpoint", type=Path)
    parser.add_argument("--backend", choices=("legacy", "torchrl_tensor"))
    parser.add_argument("--storage", choices=("cpu_pinned", "cuda", "auto"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sample-repeats", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--correctness-updates", type=int, default=1000)
    parser.add_argument("--same-backend-control", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=773)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "performance":
        if args.checkpoint is None or args.backend is None:
            parser.error("performance mode requires --checkpoint and --backend")
        report = performance(args)
    else:
        if args.tensor_checkpoint is None or (
            args.legacy_checkpoint is None and not args.same_backend_control
        ):
            parser.error(
                "correctness mode requires --tensor-checkpoint and, unless running "
                "the same-backend control, --legacy-checkpoint"
            )
        report = correctness(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
