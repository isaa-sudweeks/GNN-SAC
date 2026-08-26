"""Benchmark the exact PCGrad projection against its frozen legacy implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC


def legacy_pcgrad_project(task_gradients):
    """Frozen projection from before the synchronization optimization."""
    task_gradients = tuple(tuple(gradients) for gradients in task_gradients)
    projected = [
        [gradient.detach().clone() for gradient in gradients]
        for gradients in task_gradients
    ]
    for task_idx, task_gradient in enumerate(projected):
        for other_idx in torch.randperm(len(task_gradients)).tolist():
            if other_idx == task_idx:
                continue
            other_gradient = task_gradients[other_idx]
            other_norm_squared = GNNSAC._gradient_dot_native(
                other_gradient, other_gradient
            )
            if float(other_norm_squared) == 0.0:
                continue
            dot = GNNSAC._gradient_dot_native(task_gradient, other_gradient)
            if float(dot) >= 0.0:
                continue
            coefficient = dot / other_norm_squared
            task_gradient[:] = [
                gradient
                - coefficient.to(device=gradient.device, dtype=gradient.dtype)
                * other.to(device=gradient.device, dtype=gradient.dtype)
                for gradient, other in zip(task_gradient, other_gradient)
            ]
    return tuple(
        sum(
            (task_gradient[param_idx] for task_gradient in projected),
            start=torch.zeros_like(projected[0][param_idx]),
        )
        / len(projected)
        for param_idx in range(len(projected[0]))
    )


def parallel_pcgrad_project(task_gradients, streams):
    """Project each independent task on a separate CUDA stream."""
    task_gradients = tuple(tuple(gradients) for gradients in task_gradients)
    if not task_gradients:
        raise ValueError("PCGrad requires at least one task gradient")
    if len(streams) < len(task_gradients):
        raise ValueError("One CUDA stream is required per task gradient")
    device = task_gradients[0][0].device
    if device.type != "cuda":
        raise ValueError("Parallel projection requires CUDA gradients")

    source_stream = torch.cuda.current_stream(device)
    original_norm_squared = tuple(
        GNNSAC._gradient_dot_native(gradients, gradients)
        for gradients in task_gradients
    )
    projection_orders = tuple(
        torch.randperm(len(task_gradients)).tolist()
        for _ in task_gradients
    )
    projected = [None] * len(task_gradients)
    for task_idx, (gradients, stream) in enumerate(
        zip(task_gradients, streams)
    ):
        stream.wait_stream(source_stream)
        with torch.cuda.stream(stream):
            task_gradient = [gradient.detach().clone() for gradient in gradients]
            for gradient in gradients:
                gradient.record_stream(stream)
            for other_idx in projection_orders[task_idx]:
                if other_idx == task_idx:
                    continue
                other_gradient = task_gradients[other_idx]
                for gradient in other_gradient:
                    gradient.record_stream(stream)
                other_norm_squared = original_norm_squared[other_idx]
                other_norm_squared.record_stream(stream)
                dot = GNNSAC._gradient_dot_native(
                    task_gradient, other_gradient
                )
                nonzero_norm = other_norm_squared != 0
                should_project = nonzero_norm & ~(dot >= 0)
                safe_norm_squared = torch.where(
                    nonzero_norm,
                    other_norm_squared,
                    torch.ones_like(other_norm_squared),
                )
                coefficient = dot / safe_norm_squared
                task_gradient[:] = [
                    torch.where(
                        should_project,
                        gradient
                        - coefficient.to(
                            device=gradient.device, dtype=gradient.dtype
                        )
                        * other.to(
                            device=gradient.device, dtype=gradient.dtype
                        ),
                        gradient,
                    )
                    for gradient, other in zip(
                        task_gradient, other_gradient
                    )
                ]
            projected[task_idx] = task_gradient

    for stream in streams[: len(task_gradients)]:
        source_stream.wait_stream(stream)
    for task_gradient in projected:
        for gradient in task_gradient:
            gradient.record_stream(source_stream)
    return tuple(
        sum(
            (task_gradient[param_idx] for task_gradient in projected),
            start=torch.zeros_like(projected[0][param_idx]),
        )
        / len(projected)
        for param_idx in range(len(projected[0]))
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(call, device: torch.device) -> float:
    synchronize(device)
    started_at = time.perf_counter()
    call()
    synchronize(device)
    return time.perf_counter() - started_at


def statistics_for(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": 1000.0 * float(values.mean()),
        "median_ms": 1000.0 * float(np.median(values)),
        "p95_ms": 1000.0 * float(np.percentile(values, 95)),
        "max_ms": 1000.0 * float(values.max()),
    }


def make_task_gradients(parameters, task_count: int, seed: int):
    generators: dict[str, torch.Generator] = {}
    gradients = []
    for task_index in range(task_count):
        task = []
        for parameter_index, parameter in enumerate(parameters):
            device_key = str(parameter.device)
            generator = generators.get(device_key)
            if generator is None:
                generator = torch.Generator(device=parameter.device)
                generators[device_key] = generator
            generator.manual_seed(seed + 10_000 * task_index + parameter_index)
            task.append(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
            )
        gradients.append(tuple(task))
    return tuple(gradients)


def assert_exact(task_gradients, seed: int, streams=None) -> None:
    torch.manual_seed(seed)
    rng_state = torch.random.get_rng_state()
    reference = legacy_pcgrad_project(task_gradients)
    expected_rng_state = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_state)
    optimized = GNNSAC.pcgrad_project(task_gradients)
    if not torch.equal(expected_rng_state, torch.random.get_rng_state()):
        raise RuntimeError("Optimized projection changed CPU RNG consumption")
    for index, (expected, actual) in enumerate(zip(reference, optimized)):
        if not torch.equal(expected, actual):
            difference = float((expected - actual).abs().max())
            raise RuntimeError(
                f"Projection tensor {index} differs; max_abs_difference={difference}"
            )
    if streams is not None:
        torch.random.set_rng_state(rng_state)
        parallel = parallel_pcgrad_project(task_gradients, streams)
        if not torch.equal(expected_rng_state, torch.random.get_rng_state()):
            raise RuntimeError("Parallel projection changed CPU RNG consumption")
        for index, (expected, actual) in enumerate(zip(reference, parallel)):
            if not torch.equal(expected, actual):
                difference = float((expected - actual).abs().max())
                raise RuntimeError(
                    f"Parallel projection tensor {index} differs; "
                    f"max_abs_difference={difference}"
                )


def benchmark_projection(
    parameters,
    *,
    task_count: int,
    warmup: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict:
    task_gradients = make_task_gradients(parameters, task_count, seed)
    streams = (
        [torch.cuda.Stream(device=device) for _ in range(task_count)]
        if device.type == "cuda" and task_count > 1
        else None
    )
    assert_exact(task_gradients, seed + 1, streams=streams)

    for iteration in range(warmup):
        iteration_seed = seed + 100 + iteration
        torch.manual_seed(iteration_seed)
        legacy_pcgrad_project(task_gradients)
        torch.manual_seed(iteration_seed)
        GNNSAC.pcgrad_project(task_gradients)
        if streams is not None:
            torch.manual_seed(iteration_seed)
            parallel_pcgrad_project(task_gradients, streams)

    samples = {"legacy": [], "optimized": []}
    implementations = {
        "legacy": legacy_pcgrad_project,
        "optimized": GNNSAC.pcgrad_project,
    }
    if streams is not None:
        samples["parallel"] = []
        implementations["parallel"] = lambda gradients: parallel_pcgrad_project(
            gradients, streams
        )
    for iteration in range(repeats):
        names = tuple(implementations)
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            torch.manual_seed(seed + 1_000 + iteration)
            samples[name].append(
                timed_call(
                    lambda implementation=implementations[name]: implementation(
                        task_gradients
                    ),
                    device,
                )
            )

    legacy = statistics_for(samples["legacy"])
    optimized = statistics_for(samples["optimized"])
    result = {
        "task_count": task_count,
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(parameter.numel() for parameter in parameters),
        "legacy": legacy,
        "optimized": optimized,
        "speedup": legacy["mean_ms"] / optimized["mean_ms"],
        "mean_ms_saved": legacy["mean_ms"] - optimized["mean_ms"],
        "exact_match": True,
    }
    if "parallel" in samples:
        parallel = statistics_for(samples["parallel"])
        result["parallel"] = parallel
        result["parallel_speedup_over_legacy"] = (
            legacy["mean_ms"] / parallel["mean_ms"]
        )
        result["parallel_speedup_over_optimized"] = (
            optimized["mean_ms"] / parallel["mean_ms"]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task-counts", default="2,3,5")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    task_counts = [int(value) for value in args.task_counts.split(",")]
    if not task_counts or any(value <= 0 for value in task_counts):
        raise ValueError("--task-counts must contain positive integers")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["sac_backend=gnn", f"device={device}"],
        )
    torch.manual_seed(args.seed)
    agent = GNNSAC(cfg)
    parameter_groups = {
        "critic": tuple(agent.model._Qs.parameters()),
        "actor": tuple(agent.model.actor_parameters()),
    }
    results = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": {
            objective: [
                benchmark_projection(
                    parameters,
                    task_count=task_count,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    seed=args.seed + 100_000 * group_index + task_count,
                    device=device,
                )
                for task_count in task_counts
            ]
            for group_index, (objective, parameters) in enumerate(
                parameter_groups.items()
            )
        },
    }
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
