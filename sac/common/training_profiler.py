"""Bounded, opt-in phase profiling for online training."""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class TrainingProfiler:
    """Collect synchronized phase timings over a bounded vector-step window."""

    EVALUATION_PHASE = "evaluation"

    def __init__(
        self,
        *,
        enabled: bool,
        warmup_vector_steps: int,
        active_vector_steps: int,
        trace_enabled: bool,
        output_dir: str | Path,
        logger: Any,
        metadata: Mapping[str, Any],
        device: str | torch.device,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ):
        if warmup_vector_steps < 0:
            raise ValueError("profiling.warmup_vector_steps must be non-negative")
        if active_vector_steps <= 0:
            raise ValueError("profiling.active_vector_steps must be positive")

        self.enabled = bool(enabled)
        self.warmup_vector_steps = int(warmup_vector_steps)
        self.active_vector_steps = int(active_vector_steps)
        self.trace_enabled = bool(trace_enabled)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.metadata = dict(metadata)
        self.device = torch.device(device)
        self.clock = clock
        self.synchronize = synchronize or self._default_synchronize

        self._samples: dict[str, list[float]] = defaultdict(list)
        self._subsamples: dict[str, list[float]] = defaultdict(list)
        self._vector_steps_seen = 0
        self._measured_vector_steps = 0
        self._measured_transitions = 0
        self._measured_optimizer_updates = 0
        self._active = False
        self._active_step_started_at = None
        self._measured_step_seconds: list[float] = []
        self._start_global_step = None
        self._finalized = False
        self._summary = None
        self._torch_profiler = None
        self._trace_path = None

    @classmethod
    def from_config(cls, cfg: Any, logger: Any) -> "TrainingProfiler":
        profile_cfg = _get(cfg, "profiling", {})
        output_dir = _get(profile_cfg, "output_dir", "profiling")
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(_get(cfg, "work_dir", ".")) / output_dir
        backend = str(_get(cfg, "mujoco_backend", "mujoco"))
        task = str(_get(cfg, "task", ""))
        multitask = bool(_get(cfg, "multitask", False))
        truss_topologies = list(_get(cfg, "truss_topologies", None) or [])
        if backend.lower() == "mjx" and len(truss_topologies) > 1:
            base_task = task.split(":", 1)[0]
            task_names = [f"{base_task}:{topology}" for topology in truss_topologies]
        elif multitask:
            task_names = [str(name) for name in (_get(cfg, "tasks", None) or [])]
        else:
            task_names = [task]
        task_names = list(dict.fromkeys(task_names))
        return cls(
            enabled=bool(_get(profile_cfg, "enabled", False)),
            warmup_vector_steps=int(_get(profile_cfg, "warmup_vector_steps", 10)),
            active_vector_steps=int(_get(profile_cfg, "active_vector_steps", 100)),
            trace_enabled=bool(_get(profile_cfg, "trace_enabled", False)),
            output_dir=output_dir,
            logger=logger,
            metadata={
                "device": str(_get(cfg, "device", "cpu")),
                "backend": backend,
                "num_envs": int(_get(cfg, "num_envs", 1)),
                "batch_size": int(_get(cfg, "batch_size", 0)),
                "replay_ratio": float(_get(cfg, "replay_ratio", 0.0)),
                "update_every_vector_steps": int(
                    _get(cfg, "update_every_vector_steps", 1)
                ),
                "task": task,
                "multitask": multitask,
                "task_count": len(task_names),
                "task_names": task_names,
                "truss_topologies": truss_topologies,
            },
            device=_get(cfg, "device", "cpu"),
        )

    def _default_synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def finalized(self) -> bool:
        return self._finalized

    def begin_vector_step(self, *, global_step: int | None = None) -> bool:
        if not self.enabled or self._finalized:
            self._active = False
            return False
        window_end = self.warmup_vector_steps + self.active_vector_steps
        self._active = self.warmup_vector_steps <= self._vector_steps_seen < window_end
        if self._active and self._torch_profiler is None and self.trace_enabled:
            self._start_trace()
        if self._active:
            if self._start_global_step is None and global_step is not None:
                self._start_global_step = int(global_step)
            self.synchronize()
            self._active_step_started_at = self.clock()
        return self._active

    def end_vector_step(
        self,
        *,
        transitions: int,
        optimizer_updates: int,
        global_step: int,
    ) -> None:
        if not self.enabled or self._finalized:
            return
        if self._active:
            self.synchronize()
            self._measured_step_seconds.append(
                self.clock() - self._active_step_started_at
            )
            self._measured_vector_steps += 1
            self._measured_transitions += int(transitions)
            self._measured_optimizer_updates += int(optimizer_updates)
        self._vector_steps_seen += 1
        self._active = False
        self._active_step_started_at = None
        if self._measured_vector_steps >= self.active_vector_steps:
            self.finalize(global_step=global_step)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self._active:
            yield
            return
        with ExitStack() as stack:
            if self._torch_profiler is not None:
                stack.enter_context(torch.profiler.record_function(f"training/{name}"))
            self.synchronize()
            started_at = self.clock()
            try:
                yield
            finally:
                self.synchronize()
                self._samples[name].append(self.clock() - started_at)

    @contextmanager
    def subphase(self, name: str) -> Iterator[None]:
        """Time a replay subphase without double-counting the hot path."""
        if not self._active:
            yield
            return
        with ExitStack() as stack:
            if self._torch_profiler is not None:
                stack.enter_context(
                    torch.profiler.record_function(f"training/replay_sampling/{name}")
                )
            self.synchronize()
            started_at = self.clock()
            try:
                yield
            finally:
                self.synchronize()
                self._subsamples[name].append(self.clock() - started_at)

    def _start_trace(self) -> None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._torch_profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        )
        self._torch_profiler.start()

    @staticmethod
    def _phase_statistics(
        samples: list[float],
        hot_path_total: float,
        *,
        included_in_hot_path: bool,
    ) -> dict[str, float | int]:
        values = np.asarray(samples, dtype=np.float64)
        total = float(values.sum())
        return {
            "count": int(values.size),
            "total_seconds": total,
            "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)),
            "max_seconds": float(values.max()),
            "hot_path_percent": (
                100.0 * total / hot_path_total
                if included_in_hot_path and hot_path_total > 0.0
                else 0.0
            ),
        }

    def summary(self, *, global_step: int) -> dict[str, Any]:
        named_hot_path_total = sum(
            sum(samples)
            for name, samples in self._samples.items()
            if name != self.EVALUATION_PHASE
        )
        measured_window_total = sum(self._measured_step_seconds)
        evaluation_total = sum(self._samples.get(self.EVALUATION_PHASE, ()))
        hot_path_total = max(measured_window_total - evaluation_total, 0.0)
        phases = {
            name: self._phase_statistics(
                samples,
                hot_path_total,
                included_in_hot_path=name != self.EVALUATION_PHASE,
            )
            for name, samples in sorted(self._samples.items())
            if samples
        }
        replay_sampling_total = sum(self._samples.get("replay_sampling", ()))
        replay_subphases = {
            name: {
                **self._phase_statistics(
                    samples,
                    replay_sampling_total,
                    included_in_hot_path=True,
                ),
                "parent_phase": "replay_sampling",
            }
            for name, samples in sorted(self._subsamples.items())
            if samples
        }
        for stats in replay_subphases.values():
            stats["parent_phase_percent"] = stats.pop("hot_path_percent")
        throughput = {
            "environment_transitions_per_second": (
                self._measured_transitions / hot_path_total if hot_path_total > 0.0 else 0.0
            ),
            "vector_steps_per_second": (
                self._measured_vector_steps / hot_path_total if hot_path_total > 0.0 else 0.0
            ),
            "optimizer_updates_per_second": (
                self._measured_optimizer_updates / hot_path_total
                if hot_path_total > 0.0
                else 0.0
            ),
            "seconds_per_optimizer_update": (
                (
                    phases.get("replay_sampling", {}).get("total_seconds", 0.0)
                    + phases.get("optimization", {}).get("total_seconds", 0.0)
                )
                / self._measured_optimizer_updates
                if self._measured_optimizer_updates > 0
                else 0.0
            ),
            "optimization_seconds_per_update": (
                phases.get("optimization", {}).get("total_seconds", 0.0)
                / self._measured_optimizer_updates
                if self._measured_optimizer_updates > 0
                else 0.0
            ),
        }
        return {
            "format_version": 1,
            "metadata": self.metadata,
            "window": {
                "warmup_vector_steps": self.warmup_vector_steps,
                "requested_active_vector_steps": self.active_vector_steps,
                "measured_vector_steps": self._measured_vector_steps,
                "measured_environment_transitions": self._measured_transitions,
                "measured_optimizer_updates": self._measured_optimizer_updates,
                "start_global_step": self._start_global_step,
                "end_global_step": int(global_step),
            },
            "measured_window_total_seconds": measured_window_total,
            "hot_path_total_seconds": hot_path_total,
            "unattributed_hot_path_seconds": max(
                hot_path_total - named_hot_path_total,
                0.0,
            ),
            "phases": phases,
            "replay_subphases": replay_subphases,
            "throughput": throughput,
            "trace_path": str(self._trace_path) if self._trace_path is not None else None,
        }

    def _flat_metrics(self, summary: Mapping[str, Any]) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "step": int(summary["window"]["end_global_step"]),
            "measured_window_total_seconds": float(
                summary["measured_window_total_seconds"]
            ),
            "hot_path_total_seconds": float(summary["hot_path_total_seconds"]),
            "unattributed_hot_path_seconds": float(
                summary["unattributed_hot_path_seconds"]
            ),
        }
        for name, stats in summary["phases"].items():
            for key, value in stats.items():
                metrics[f"phase/{name}/{key}"] = value
        for name, stats in summary["replay_subphases"].items():
            for key, value in stats.items():
                if isinstance(value, (float, int)):
                    metrics[f"replay_subphase/{name}/{key}"] = value
        for key, value in summary["throughput"].items():
            metrics[f"throughput/{key}"] = value
        return metrics

    def finalize(self, *, global_step: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self._finalized:
            return self._summary
        if self._torch_profiler is not None:
            self._torch_profiler.stop()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._trace_path = self.output_dir / "training_trace.json"
            self._torch_profiler.export_chrome_trace(str(self._trace_path))
            self._torch_profiler = None
        self._summary = self.summary(global_step=global_step)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_dir / "profiling_summary.json"
        summary_path.write_text(json.dumps(self._summary, indent=2, sort_keys=True) + "\n")
        self.logger.log(self._flat_metrics(self._summary), "profiling")
        self._finalized = True
        return self._summary
