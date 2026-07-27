from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from common.training_profiler import TrainingProfiler
from gnn_sac import GNNSAC


class SequenceClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class RecordingLogger:
    def __init__(self):
        self.rows = []

    def log(self, metrics, category):
        self.rows.append((category, dict(metrics)))


class TrainingProfilerTest(unittest.TestCase):
    def make_profiler(self, output_dir, *, warmup=1, active=2, clock=None):
        return TrainingProfiler(
            enabled=True,
            warmup_vector_steps=warmup,
            active_vector_steps=active,
            trace_enabled=False,
            output_dir=output_dir,
            logger=RecordingLogger(),
            metadata={"backend": "mjx", "num_envs": 2},
            device="cpu",
            clock=clock or SequenceClock([0.0, 1.0]),
            synchronize=lambda: None,
        )

    def test_aggregates_window_statistics_and_writes_json(self):
        with TemporaryDirectory() as tmp:
            profiler = self.make_profiler(
                tmp,
                clock=SequenceClock([0.0, 0.0, 1.0, 2.0, 5.0, 5.0, 10.0, 10.0, 12.0, 12.0]),
            )

            profiler.begin_vector_step(global_step=0)
            with profiler.phase("environment_step"):
                pass
            profiler.end_vector_step(transitions=2, optimizer_updates=0, global_step=2)

            profiler.begin_vector_step(global_step=2)
            with profiler.phase("environment_step"):
                pass
            with profiler.phase("evaluation"):
                pass
            profiler.end_vector_step(transitions=2, optimizer_updates=1, global_step=4)

            profiler.begin_vector_step(global_step=4)
            with profiler.phase("environment_step"):
                pass
            profiler.end_vector_step(transitions=1, optimizer_updates=2, global_step=5)

            summary_path = Path(tmp) / "profiling_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text())
            env_stats = summary["phases"]["environment_step"]
            self.assertEqual(env_stats["count"], 2)
            self.assertEqual(env_stats["total_seconds"], 3.0)
            self.assertEqual(env_stats["mean_seconds"], 1.5)
            self.assertEqual(env_stats["median_seconds"], 1.5)
            self.assertAlmostEqual(env_stats["p95_seconds"], 1.95)
            self.assertEqual(env_stats["max_seconds"], 2.0)
            self.assertEqual(env_stats["hot_path_percent"], 75.0)
            self.assertEqual(summary["phases"]["evaluation"]["hot_path_percent"], 0.0)
            self.assertEqual(summary["window"]["measured_vector_steps"], 2)
            self.assertEqual(summary["window"]["measured_environment_transitions"], 3)
            self.assertEqual(summary["window"]["measured_optimizer_updates"], 3)
            self.assertEqual(summary["window"]["start_global_step"], 2)
            self.assertEqual(summary["measured_window_total_seconds"], 7.0)
            self.assertEqual(summary["hot_path_total_seconds"], 4.0)
            self.assertEqual(summary["unattributed_hot_path_seconds"], 1.0)
            self.assertEqual(summary["throughput"]["environment_transitions_per_second"], 0.75)
            self.assertEqual(profiler.logger.rows[0][0], "profiling")

    def test_short_run_after_warmup_emits_empty_valid_summary(self):
        with TemporaryDirectory() as tmp:
            profiler = self.make_profiler(tmp, warmup=3, active=2)
            for global_step in (2, 4):
                profiler.begin_vector_step()
                profiler.end_vector_step(
                    transitions=2,
                    optimizer_updates=0,
                    global_step=global_step,
                )
            summary = profiler.finalize(global_step=4)

            self.assertEqual(summary["window"]["measured_vector_steps"], 0)
            self.assertEqual(summary["phases"], {})
            self.assertEqual(
                summary["throughput"]["environment_transitions_per_second"],
                0.0,
            )

    def test_disabled_profiler_has_no_side_effects(self):
        with TemporaryDirectory() as tmp:
            logger = RecordingLogger()
            profiler = TrainingProfiler(
                enabled=False,
                warmup_vector_steps=0,
                active_vector_steps=1,
                trace_enabled=False,
                output_dir=tmp,
                logger=logger,
                metadata={},
                device="cpu",
                clock=SequenceClock([]),
                synchronize=lambda: self.fail("disabled profiling synchronized the device"),
            )
            profiler.begin_vector_step()
            with profiler.phase("environment_step"):
                pass
            profiler.end_vector_step(
                transitions=1,
                optimizer_updates=0,
                global_step=1,
            )

            self.assertIsNone(profiler.finalize(global_step=1))
            self.assertEqual(logger.rows, [])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_from_config_resolves_output_under_work_dir(self):
        with TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                work_dir=tmp,
                device="cpu",
                profiling=SimpleNamespace(
                    enabled=True,
                    warmup_vector_steps=0,
                    active_vector_steps=1,
                    trace_enabled=False,
                    output_dir="baseline",
                ),
            )
            profiler = TrainingProfiler.from_config(cfg, RecordingLogger())
            self.assertEqual(profiler.output_dir, Path(tmp) / "baseline")

    def test_from_config_records_multitask_metadata(self):
        cfg = SimpleNamespace(
            work_dir=".",
            device="cpu",
            mujoco_backend="mjx",
            task="truss-graph",
            multitask=True,
            tasks=["legacy-a", "legacy-b"],
            truss_topologies=["tetrahedron", "octahedron"],
            profiling=SimpleNamespace(
                enabled=True,
                warmup_vector_steps=0,
                active_vector_steps=1,
                trace_enabled=False,
                output_dir="profiling",
            ),
        )

        profiler = TrainingProfiler.from_config(cfg, RecordingLogger())

        self.assertTrue(profiler.metadata["multitask"])
        self.assertEqual(profiler.metadata["task_count"], 2)
        self.assertEqual(
            profiler.metadata["task_names"],
            ["truss-graph:tetrahedron", "truss-graph:octahedron"],
        )

    def test_replay_subphases_do_not_double_count_hot_path(self):
        with TemporaryDirectory() as tmp:
            profiler = self.make_profiler(
                tmp,
                warmup=0,
                active=1,
                clock=SequenceClock([0.0, 0.0, 2.0, 5.0]),
            )

            profiler.begin_vector_step(global_step=0)
            with profiler.subphase("balanced_gather"):
                pass
            profiler.end_vector_step(
                transitions=1,
                optimizer_updates=0,
                global_step=1,
            )

            summary = profiler.summary(global_step=1)
            stats = summary["replay_subphases"]["balanced_gather"]
            self.assertEqual(stats["total_seconds"], 2.0)
            self.assertEqual(stats["parent_phase"], "replay_sampling")
            self.assertEqual(summary["unattributed_hot_path_seconds"], 5.0)

    def test_optional_trace_is_exported(self):
        with TemporaryDirectory() as tmp:
            profiler = TrainingProfiler(
                enabled=True,
                warmup_vector_steps=0,
                active_vector_steps=1,
                trace_enabled=True,
                output_dir=tmp,
                logger=RecordingLogger(),
                metadata={},
                device="cpu",
                synchronize=lambda: None,
            )
            profiler.begin_vector_step()
            with profiler.phase("action_selection"):
                torch.ones(2) + 1
            profiler.end_vector_step(
                transitions=1,
                optimizer_updates=0,
                global_step=1,
            )

            trace_path = Path(tmp) / "training_trace.json"
            self.assertTrue(trace_path.exists())
            self.assertIn("training/action_selection", trace_path.read_text())

    def test_gnn_update_separates_replay_sampling_from_optimization(self):
        class PhaseRecorder:
            def __init__(self):
                self.names = []

            @contextmanager
            def phase(self, name):
                self.names.append(name)
                yield

        class Buffer:
            def sample(self):
                return (None, None, None, None, None)

        class Model:
            def train(self):
                return

            def soft_update_target_Q(self):
                return

            def eval(self):
                return

        agent = SimpleNamespace(
            cfg=SimpleNamespace(pcgrad=False),
            model=Model(),
            update_q=lambda *args: (torch.tensor(1.0), torch.tensor(2.0)),
            update_pi_and_alpha=lambda obs: {"pi_loss": torch.tensor(3.0)},
        )
        profiler = PhaseRecorder()

        metrics = GNNSAC.update(agent, Buffer(), performance_profiler=profiler)

        self.assertEqual(profiler.names, ["replay_sampling", "optimization"])
        self.assertEqual(metrics["value_loss"], torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
