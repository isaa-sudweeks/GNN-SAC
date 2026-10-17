"""Regression coverage for performance-triggered teacher release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))
from common.distillation import Distillation
from common.distillation_schedule import PerformanceDecay
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer
from tests.test_distillation import setup, replay_buffer


def options():
    return dict(distance_ratio=0.9, survival_tolerance=0.05, patience=2,
                release_steps=10, references={
                    "a": dict(distance=10.0, survival=1.0, horizon=100, teacher_sha256="a"),
                    "b": dict(distance=20.0, survival=1.0, horizon=100, teacher_sha256="b")})


class PerformanceDecayTest(unittest.TestCase):
    def schedule(self):
        return PerformanceDecay(options(), {"task:a": "a", "task:b": "b"},
                                {"task:a": {"sha256": "a"}, "task:b": {"sha256": "b"}})

    def test_patience_survival_horizon_and_irreversible_release(self):
        schedule = self.schedule()
        schedule.observe("task:a", 1, 9, 1, 100)
        schedule.observe("task:a", 1, 9, 1, 100)  # Duplicate evaluation cannot qualify.
        self.assertEqual(schedule.multiplier("task:a", 3), 1)
        schedule.observe("task:a", 2, 9, 0.8, 100)
        schedule.observe("task:a", 3, 9, 1, 99)
        schedule.observe("task:a", 4, float("nan"), 1, 100)
        schedule.observe("task:a", 5, 9, 1, 100)
        schedule.observe("task:a", 6, 9, 1, 100)
        self.assertEqual(schedule.multiplier("task:a", 11), 0.5)
        schedule.observe("task:a", 12, 0, 0, 100)
        self.assertEqual(schedule.multiplier("task:a", 16), 0)
        self.assertEqual(schedule.multiplier("task:b", 16), 1)
        schedule.observe("heldout", 16, 100, 1, 100)
        self.assertNotIn("heldout", schedule.progress)

    def test_resume_and_invalid_references(self):
        schedule = self.schedule()
        schedule.observe("task:a", 1, 10, 1, 100)
        restored = self.schedule()
        restored.load_state_dict(schedule.state_dict())
        restored.observe("task:a", 2, 10, 1, 100)
        self.assertEqual(restored.multiplier("task:a", 7), 0.5)
        for field, value in (("distance", 0), ("distance", float("nan")),
                             ("survival", 0), ("survival", 2), ("horizon", 0),
                             ("teacher_sha256", "wrong")):
            with self.subTest(field=field, value=value):
                opts = options()
                opts["references"]["a"][field] = value
                with self.assertRaises(ValueError):
                    PerformanceDecay(opts, {"task:a": "a"}, {"task:a": {"sha256": "a"}})
        state = schedule.state_dict()
        state["task:a"]["release_start"] = 100
        with self.assertRaises(ValueError):
            schedule.load_state_dict(state)

    def test_evaluation_excludes_heldout_and_uses_evaluation_horizon(self):
        observer = Mock()
        trainer = SimpleNamespace(
            cfg=SimpleNamespace(task="truss-graph", eval_episodes=1, save_video=False,
                                episode_length=100, episode_lengths=[100]),
            _evaluation_cfg=SimpleNamespace(episode_length=2, episode_lengths=[2, 3]),
            _eval_training_topologies=["a"], _eval_heldout_topologies=["heldout"],
            _metric_key=lambda value: value,
            _observe_distillation_eval=observer,
            _eval_one=lambda **kwargs: dict(episode_reward=1, episode_success=1,
                                           episode_length=2, episode_distance=10,
                                           episode_survival=1, episode_horizon=2),
        )
        metrics = OnlineTrainer._eval_topologies(trainer, {"a": 0, "heldout": 1})
        observer.assert_called_once()
        self.assertEqual(observer.call_args.args[0], "a")
        self.assertEqual(metrics["heldout_episode_distance"], 10)
        trainer.eval_env = SimpleNamespace(reset=lambda **kwargs: None)
        remaining = [1, 2]
        trainer.eval_env.step = lambda action: (None, 1, remaining.pop(0) == 2,
                                               {"success": 1, "com_delta_x": 1})
        trainer.agent = SimpleNamespace(act=lambda *args, **kwargs: None)
        trainer._scalar_value = float
        metrics = OnlineTrainer._eval_one(trainer, task_idx=0)
        self.assertEqual(metrics["episode_horizon"], 2)
        self.assertEqual(metrics["episode_survival"], 1)

    def test_single_teacher_key_and_eval_override_identity(self):
        observe = Mock()
        distill = SimpleNamespace(performance_decay=True,
                                 task_topologies={"truss-graph": "a"},
                                 observe_evaluation=observe)
        trainer = SimpleNamespace(cfg=SimpleNamespace(task="truss-graph"),
                                  distillation=distill)
        metrics = dict(episode_distance=10, episode_survival=1, episode_horizon=100)
        OnlineTrainer._observe_distillation_eval(trainer, "a", metrics)
        observe.assert_called_once_with("truss-graph", 10.0, 1.0, 100)
        observe.reset_mock()
        OnlineTrainer._observe_distillation_eval(trainer, "heldout", metrics)
        observe.assert_not_called()
        trainer._evaluation_cfg = SimpleNamespace(task="other-eval-task")
        OnlineTrainer._observe_distillation_eval(trainer, "a", metrics)
        observe.assert_not_called()
        trainer._evaluation_cfg.task = "truss-graph:heldout"
        OnlineTrainer._observe_distillation_eval(trainer, "a", metrics)
        observe.assert_not_called()
        trainer._evaluation_cfg.task = "truss-graph:a"
        OnlineTrainer._observe_distillation_eval(trainer, "a", metrics)
        observe.assert_called_once_with("truss-graph", 10.0, 1.0, 100)

    def test_actor_paths_use_topology_weights_and_report_actual_weighted_kl(self):
        for pcgrad in (False, True):
            with self.subTest(pcgrad=pcgrad), tempfile.TemporaryDirectory() as tmp:
                cfg, baseline = setup(tmp, pcgrad=pcgrad)
                opts = options()
                opts["enabled"] = True
                for task, source in baseline.sources.items():
                    opts["references"][task.split(":")[1]]["teacher_sha256"] = source["sha256"]
                cfg.distillation["performance_decay"] = opts
                distill = Distillation(cfg, cfg.tasks)
                distill.stage = "online"
                distill.completed_updates = distill.pretrain_updates
                for step in (1, 2):
                    distill.step = step
                    distill.observe_evaluation(cfg.tasks[0], 10, 1, 100)
                distill.step = 12
                self.assertEqual(distill.weight_for(cfg.tasks[0]), 0)
                self.assertGreater(distill.weight_for(cfg.tasks[1]), 0)
                self.assertEqual(distill.weight_for("unknown"), 0)
                agent = GNNSAC(cfg)
                agent.distillation = distill
                replay = replay_buffer(cfg)
                distill.bind_replay(replay)
                with patch.object(distill, "loss", wraps=distill.loss) as spy:
                    info = agent.update(replay)
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(spy.call_args.args[1], cfg.tasks[1])
                expected = info[f"distillation/kl/{cfg.tasks[1]}"] * distill.weight_for(cfg.tasks[1]) / 2
                torch.testing.assert_close(info["distillation/weighted_kl"], expected)
                state = deepcopy(distill.state_dict())
                restored = Distillation(cfg, cfg.tasks)
                restored.load_state_dict(state)
                restored.step = distill.step
                self.assertEqual(restored.weight_for(cfg.tasks[0]), 0)
                cfg.distillation["performance_decay"]["enabled"] = False
                disabled = Distillation(cfg, cfg.tasks)
                with self.assertRaises(ValueError):
                    disabled.load_state_dict(state)
                cfg.distillation["performance_decay"]["enabled"] = True
                cfg.distillation["performance_decay"]["patience"] = 3
                changed = Distillation(cfg, cfg.tasks)
                with self.assertRaises(ValueError):
                    changed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
