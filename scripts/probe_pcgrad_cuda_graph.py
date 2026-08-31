"""Probe exact CUDA Graph capture for one production-sized PCGrad task."""

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
from benchmark_pcgrad import make_static_task_batches


def synchronize() -> None:
    torch.cuda.synchronize()


def summarize(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": 1000.0 * float(values.mean()),
        "median_ms": 1000.0 * float(np.median(values)),
        "p95_ms": 1000.0 * float(np.percentile(values, 95)),
    }


def timed(call) -> float:
    synchronize()
    started_at = time.perf_counter()
    call()
    synchronize()
    return time.perf_counter() - started_at


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    synchronize()

    outputs = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call()
    synchronize()
    return graph, outputs


def assert_tensors_exact(expected, actual, name: str) -> None:
    for index, (first, second) in enumerate(zip(expected, actual)):
        if not torch.equal(first, second):
            difference = float((first - second).abs().max())
            raise RuntimeError(
                f"{name}[{index}] differs; max_abs_difference={difference}"
            )


def probe_objective(agent, batch, objective: str, warmup: int, repeats: int):
    if objective == "critic":
        parameters = agent._q_parameters

        def call():
            loss = agent._q_loss(*batch)
            gradients = agent._parameter_gradients(loss, parameters)
            return (loss.detach(), *gradients)

    else:
        parameters = agent._actor_parameters

        def call():
            loss, _ = agent._pi_loss(batch[0])
            gradients = agent._parameter_gradients(loss, parameters)
            return (loss.detach(), *gradients)

    graph, graph_outputs = capture(call)
    initial_rng = torch.cuda.get_rng_state_all()
    eager_outputs = call()
    synchronize()
    expected_rng = torch.cuda.get_rng_state_all()
    torch.cuda.set_rng_state_all(initial_rng)
    graph.replay()
    synchronize()
    actual_rng = torch.cuda.get_rng_state_all()
    assert_tensors_exact(eager_outputs, graph_outputs, objective)
    assert_tensors_exact(expected_rng, actual_rng, f"{objective}_rng")

    for _ in range(warmup):
        call()
        graph.replay()
    eager_samples = []
    graph_samples = []
    for iteration in range(repeats):
        modes = (("eager", call), ("graph", graph.replay))
        if iteration % 2:
            modes = tuple(reversed(modes))
        for name, implementation in modes:
            sample = timed(implementation)
            (eager_samples if name == "eager" else graph_samples).append(sample)
    eager = summarize(eager_samples)
    captured = summarize(graph_samples)
    return {
        "exact_match": True,
        "eager": eager,
        "cuda_graph": captured,
        "speedup": eager["mean_ms"] / captured["mean_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=85)
    parser.add_argument("--node-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=319)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["sac_backend=gnn", "device=cuda"],
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    agent = GNNSAC(cfg)
    task_batches = make_static_task_batches(
        cfg,
        task_count=1,
        node_counts=[args.node_count],
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        seed=args.seed + 1,
    )
    batch = next(iter(task_batches.values()))
    results = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "node_count": args.node_count,
        "critic": probe_objective(
            agent, batch, "critic", args.warmup, args.repeats
        ),
        "actor": probe_objective(
            agent, batch, "actor", args.warmup, args.repeats
        ),
    }
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
