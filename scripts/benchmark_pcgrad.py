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
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC
from common.graph_transforms import graph_feature_flags, prepare_graph


class FrozenLegacyGNNSAC(GNNSAC):
    """Pre-optimization PCGrad update retained only for matched benchmarks."""

    @classmethod
    def pcgrad_project(cls, task_gradients):
        return legacy_pcgrad_project(task_gradients)

    @staticmethod
    def _set_parameter_gradients(parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.detach().clone()

    def _pcgrad_q_update(self, task_batches, **_kwargs):
        parameters = tuple(self.model._Qs.parameters())
        losses = []
        task_gradients = {}
        for task, batch in task_batches.items():
            obs, action, reward, terminated, next_obs = batch
            loss = self._q_loss(obs, action, reward, terminated, next_obs)
            losses.append(loss)
            task_gradients[task] = self._parameter_gradients(loss, parameters)
        self.q_optim.zero_grad(set_to_none=True)
        self._set_parameter_gradients(
            parameters, self.pcgrad_project(task_gradients.values())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.cfg.grad_clip_norm
        )
        self.q_optim.step()
        return (
            torch.stack([loss.detach() for loss in losses]).mean(),
            grad_norm.detach(),
            task_gradients,
        )

    def _pcgrad_pi_and_alpha_update(self, task_batches, **_kwargs):
        parameters = tuple(self.model.actor_parameters())
        losses = []
        task_gradients = {}
        task_info = []
        for task, batch in task_batches.items():
            pi_loss, info = self._pi_loss(batch[0])
            losses.append(pi_loss)
            task_info.append(info)
            task_gradients[task] = self._parameter_gradients(
                pi_loss, parameters
            )
        self.pi_optim.zero_grad(set_to_none=True)
        self._set_parameter_gradients(
            parameters, self.pcgrad_project(task_gradients.values())
        )
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        log_prob = torch.cat(
            [info["log_prob"].reshape(-1) for info in task_info]
        )
        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()
        entropy = torch.stack(
            [info["entropy"].detach().mean() for info in task_info]
        ).mean()
        return (
            {
                "pi_loss": torch.stack(
                    [loss.detach() for loss in losses]
                ).mean(),
                "pi_grad_norm": pi_grad_norm.detach(),
                "alpha_loss": alpha_loss.detach(),
                "alpha": self.alpha.detach(),
                "entropy": entropy,
            },
            task_gradients,
        )


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


class StaticTaskBuffer:
    supports_replay_profiling = True

    def __init__(self, task_batches):
        self.task_batches = task_batches

    def sample_task_batches(self, performance_profiler=None):
        return self.task_batches


def make_ring_graph(node_count: int, feature_dim: int, generator) -> Data:
    nodes = torch.arange(node_count, dtype=torch.long)
    next_nodes = nodes.roll(-1)
    return Data(
        x=torch.randn(node_count, feature_dim, generator=generator),
        edge_index=torch.stack(
            (
                torch.cat((nodes, next_nodes)),
                torch.cat((next_nodes, nodes)),
            ),
            dim=0,
        ),
        action_mask=torch.ones(node_count, dtype=torch.bool),
        rigidity=torch.tensor(0.75),
    )


def make_static_task_batches(
    cfg,
    *,
    task_count: int,
    node_counts: list[int],
    batch_size: int,
    device: torch.device,
    seed: int,
):
    if batch_size % task_count != 0:
        raise ValueError("Full-update batch size must be divisible by task count")
    per_task = batch_size // task_count
    generator = torch.Generator().manual_seed(seed)
    flags = graph_feature_flags(cfg)
    task_batches = {}
    for task_index in range(task_count):
        node_count = node_counts[task_index % len(node_counts)]
        observations = [
            prepare_graph(
                make_ring_graph(node_count, cfg.obs_dim, generator),
                use_virtual_node=bool(cfg.use_virtual_node),
                **flags,
            )
            for _ in range(per_task)
        ]
        next_observations = [
            prepare_graph(
                make_ring_graph(node_count, cfg.obs_dim, generator),
                use_virtual_node=bool(cfg.use_virtual_node),
                **flags,
            )
            for _ in range(per_task)
        ]
        observation_batch = Batch.from_data_list(observations).to(device)
        next_observation_batch = Batch.from_data_list(next_observations).to(
            device
        )
        action = torch.randn(
            per_task * node_count,
            cfg.action_dim,
            generator=generator,
        ).to(device)
        reward = torch.randn(per_task, generator=generator).to(device)
        terminated = torch.zeros(per_task, device=device)
        task_batches[f"benchmark:task-{task_index}"] = (
            observation_batch,
            action,
            reward,
            terminated,
            next_observation_batch,
        )
    return task_batches


def assert_nested_exact(first, second, path="state") -> None:
    if isinstance(first, torch.Tensor):
        if not torch.equal(first, second):
            difference = float((first - second).abs().max())
            raise RuntimeError(
                f"{path} differs; max_abs_difference={difference}"
            )
        return
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise RuntimeError(f"{path} has different keys")
        for key in first:
            assert_nested_exact(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            raise RuntimeError(f"{path} has different lengths")
        for index, (first_value, second_value) in enumerate(zip(first, second)):
            assert_nested_exact(
                first_value, second_value, f"{path}[{index}]"
            )
        return
    if first != second:
        raise RuntimeError(f"{path} differs: {first!r} != {second!r}")


def benchmark_full_update(
    cfg,
    *,
    task_count: int,
    node_counts: list[int],
    batch_size: int,
    warmup: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict:
    cfg.batch_size = batch_size
    cfg.pcgrad = True
    reference_agent = FrozenLegacyGNNSAC(cfg)
    optimized_agent = GNNSAC(cfg)
    optimized_agent.load_training_state_dict(
        reference_agent.training_state_dict()
    )
    buffer = StaticTaskBuffer(
        make_static_task_batches(
            cfg,
            task_count=task_count,
            node_counts=node_counts,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
    )
    samples = {"legacy": [], "optimized": []}
    agents = {"legacy": reference_agent, "optimized": optimized_agent}
    total_iterations = warmup + repeats
    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
        torch.cuda.reset_peak_memory_stats(device)

    for iteration in range(total_iterations):
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        )
        names = (
            ("legacy", "optimized")
            if iteration % 2 == 0
            else ("optimized", "legacy")
        )
        expected_cpu_rng_state = None
        expected_cuda_rng_state = None
        metrics = {}
        for position, name in enumerate(names):
            if position:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
            result = {}

            def run_update(agent=agents[name]):
                result.update(agent.update(buffer))

            elapsed = timed_call(run_update, device)
            metrics[name] = {
                key: value.detach().clone()
                for key, value in result.items()
            }
            if position == 0:
                expected_cpu_rng_state = torch.random.get_rng_state()
                expected_cuda_rng_state = (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None
                )
            if iteration >= warmup:
                samples[name].append(elapsed)

        if not torch.equal(
            expected_cpu_rng_state, torch.random.get_rng_state()
        ):
            raise RuntimeError("Full update changed CPU RNG consumption")
        if expected_cuda_rng_state is not None:
            for expected, actual in zip(
                expected_cuda_rng_state, torch.cuda.get_rng_state_all()
            ):
                if not torch.equal(expected, actual):
                    raise RuntimeError("Full update changed CUDA RNG consumption")
        assert_nested_exact(
            reference_agent.training_state_dict(),
            optimized_agent.training_state_dict(),
        )
        assert_nested_exact(metrics["legacy"], metrics["optimized"], "metrics")
        torch.random.set_rng_state(expected_cpu_rng_state)
        if expected_cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(expected_cuda_rng_state)

    legacy = statistics_for(samples["legacy"])
    optimized = statistics_for(samples["optimized"])
    return {
        "task_count": task_count,
        "batch_size": batch_size,
        "node_counts": node_counts,
        "legacy": legacy,
        "optimized": optimized,
        "speedup": legacy["mean_ms"] / optimized["mean_ms"],
        "mean_ms_saved": legacy["mean_ms"] - optimized["mean_ms"],
        "exact_match": True,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task-counts", default="2,3,5")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-update-repeats", type=int, default=0)
    parser.add_argument("--full-update-warmup", type=int, default=5)
    parser.add_argument("--full-update-task-count", type=int, default=3)
    parser.add_argument("--full-update-batch-size", type=int, default=256)
    parser.add_argument("--node-counts", default="6,4,8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    task_counts = [int(value) for value in args.task_counts.split(",")]
    node_counts = [int(value) for value in args.node_counts.split(",")]
    if not task_counts or any(value <= 0 for value in task_counts):
        raise ValueError("--task-counts must contain positive integers")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")
    if not node_counts or any(value <= 0 for value in node_counts):
        raise ValueError("--node-counts must contain positive integers")
    if args.full_update_repeats < 0 or args.full_update_warmup < 0:
        raise ValueError("Full-update repeat counts must be nonnegative")

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
    if args.full_update_repeats:
        results["full_update"] = benchmark_full_update(
            cfg,
            task_count=args.full_update_task_count,
            node_counts=node_counts,
            batch_size=args.full_update_batch_size,
            warmup=args.full_update_warmup,
            repeats=args.full_update_repeats,
            seed=args.seed + 900_000,
            device=device,
        )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
