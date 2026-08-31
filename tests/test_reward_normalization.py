from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.reward_normalizer import TaskRewardNormalizer
from trainer.online_trainer import OnlineTrainer


class TaskRewardNormalizerTest(unittest.TestCase):
    def test_uses_discounted_return_population_variance_without_centering(self):
        normalizer = TaskRewardNormalizer(gamma=1.0, epsilon=1e-8, clip=100.0)

        first = normalizer.normalize(torch.tensor(1.0), task="easy", stream=0)
        second = normalizer.normalize(torch.tensor(1.0), task="easy", stream=0)

        self.assertAlmostEqual(first.item(), 100.0)
        self.assertAlmostEqual(second.item(), 2.0, places=5)
        metrics = normalizer.metrics()["easy"]
        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["return_mean"], 1.5)
        self.assertAlmostEqual(metrics["return_std"], 0.5)

    def test_task_scales_are_independent(self):
        normalizer = TaskRewardNormalizer(gamma=1.0, epsilon=1e-8, clip=100.0)
        for reward in (1.0, 1.0):
            small = normalizer.normalize(reward, task="small", stream="small")
        for reward in (100.0, 100.0):
            large = normalizer.normalize(reward, task="large", stream="large")

        self.assertAlmostEqual(small, large, places=5)
        self.assertAlmostEqual(normalizer.metrics()["small"]["return_std"], 0.5)
        self.assertAlmostEqual(normalizer.metrics()["large"]["return_std"], 50.0)

    def test_streams_accumulate_and_reset_independently(self):
        normalizer = TaskRewardNormalizer(gamma=1.0, epsilon=1e-8, clip=10.0)
        normalizer.normalize(1.0, task="task", stream=0)
        normalizer.normalize(5.0, task="task", stream=1)
        normalizer.normalize(1.0, task="task", stream=0, done=True)

        self.assertNotIn(0, normalizer._returns)
        self.assertEqual(normalizer._returns[1], 5.0)
        normalizer.normalize(1.0, task="task", stream=0)
        self.assertEqual(normalizer._returns[0], 1.0)

    def test_explicit_stream_reset_preserves_statistics_and_other_streams(self):
        normalizer = TaskRewardNormalizer(gamma=0.9, epsilon=1e-8, clip=10.0)
        normalizer.normalize(2.0, task="task", stream=0)
        normalizer.normalize(5.0, task="task", stream=1)
        metrics_before_reset = normalizer.metrics()

        normalizer.reset_stream(0)

        self.assertNotIn(0, normalizer._returns)
        self.assertNotIn(0, normalizer._stream_tasks)
        self.assertEqual(normalizer._returns[1], 5.0)
        self.assertEqual(normalizer._stream_tasks[1], "task")
        self.assertEqual(normalizer.metrics(), metrics_before_reset)

    def test_rejects_invalid_settings_unknown_tasks_and_mid_episode_task_changes(self):
        with self.assertRaisesRegex(ValueError, "gamma"):
            TaskRewardNormalizer(gamma=1.1)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            TaskRewardNormalizer(gamma=0.99, epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "clip"):
            TaskRewardNormalizer(gamma=0.99, clip=0.0)

        normalizer = TaskRewardNormalizer(gamma=0.99, allowed_tasks=["a", "b"])
        with self.assertRaises(KeyError):
            normalizer.normalize(1.0, task="c", stream=0)
        normalizer.normalize(1.0, task="a", stream=0)
        with self.assertRaisesRegex(ValueError, "changed task"):
            normalizer.normalize(1.0, task="b", stream=0)

    def test_state_dict_restores_statistics_and_active_returns(self):
        original = TaskRewardNormalizer(
            gamma=0.9,
            epsilon=1e-6,
            clip=5.0,
            allowed_tasks=["a", "b"],
        )
        original.normalize(2.0, task="a", stream=3)
        original.normalize(4.0, task="b", stream=4, done=True)

        restored = TaskRewardNormalizer(
            gamma=0.9,
            epsilon=1e-6,
            clip=5.0,
            allowed_tasks=["a", "b"],
        )
        restored.load_state_dict(original.state_dict())

        self.assertEqual(restored.state_dict(), original.state_dict())
        self.assertEqual(restored._returns, {3: 2.0})


class OnlineTrainerRewardNormalizationTest(unittest.TestCase):
    def _trainer(self, *, task_names=("only",), multitask=False):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(task="only", multitask=multitask)
        trainer.buffer = SimpleNamespace(task_names=list(task_names))
        trainer.env = SimpleNamespace(rand_act=lambda: torch.zeros(1))
        return trainer

    def test_single_task_fallback_and_multitask_missing_task_error(self):
        trainer = self._trainer()
        self.assertEqual(trainer._normalization_task({}), "only")

        trainer = self._trainer(task_names=("a", "b"), multitask=True)
        with self.assertRaisesRegex(ValueError, r"info\['task'\]"):
            trainer._normalization_task({})

    def test_transition_keeps_raw_reward_separate_from_learning_reward(self):
        trainer = self._trainer()
        td = trainer.to_td(
            torch.zeros(2),
            action=torch.ones(1),
            reward=torch.tensor(2.0),
            raw_reward=torch.tensor(20.0),
            terminated=torch.tensor(0.0),
        )

        self.assertEqual(td["reward"].item(), 2.0)
        self.assertEqual(td["raw_reward"].item(), 20.0)

    def test_environment_reset_discards_restored_active_returns(self):
        original = TaskRewardNormalizer(gamma=0.9, allowed_tasks=["only"])
        original.normalize(2.0, task="only", stream=0)

        trainer = self._trainer()
        trainer.reward_normalizer = TaskRewardNormalizer(
            gamma=0.9,
            allowed_tasks=["only"],
        )
        trainer.reward_normalizer.load_state_dict(original.state_dict())
        metrics_before_reset = trainer.reward_normalizer.metrics()

        trainer._reset_reward_normalizer_streams([0])
        trainer.reward_normalizer.normalize(1.0, task="only", stream=0)

        self.assertEqual(trainer.reward_normalizer._returns[0], 1.0)
        self.assertEqual(
            trainer.reward_normalizer.metrics()["only"]["count"],
            metrics_before_reset["only"]["count"] + 1,
        )

    def test_single_env_training_reset_clears_restored_stream(self):
        class ResetReached(Exception):
            pass

        class Profiler:
            def begin_vector_step(self, **kwargs):
                return None

            def phase(self, name):
                return nullcontext()

        trainer = self._trainer()
        trainer.cfg.steps = 1
        trainer.cfg.num_envs = 1
        trainer.cfg.seed_steps = 0
        trainer.cfg.pretrain_steps = 0
        trainer.cfg.eval_freq = 1000
        trainer._step = 0
        trainer._optimizer_updates = 0
        trainer.performance_profiler = Profiler()
        trainer.reward_normalizer = TaskRewardNormalizer(
            gamma=0.9,
            allowed_tasks=["only"],
        )
        trainer.reward_normalizer.normalize(2.0, task="only", stream=0)
        trainer._evaluate_and_log = lambda: None

        def reset():
            self.assertNotIn(0, trainer.reward_normalizer._returns)
            self.assertNotIn(0, trainer.reward_normalizer._stream_tasks)
            raise ResetReached

        trainer.env = SimpleNamespace(num_envs=1, reset=reset)

        with self.assertRaises(ResetReached):
            trainer.train()

    def test_multi_env_training_reset_clears_all_restored_streams(self):
        class ResetReached(Exception):
            pass

        class Profiler:
            def begin_vector_step(self, **kwargs):
                return None

            def phase(self, name):
                return nullcontext()

        trainer = self._trainer()
        trainer.cfg.steps = 2
        trainer.cfg.seed_steps = 0
        trainer.cfg.pretrain_steps = 0
        trainer.cfg.eval_freq = 1000
        trainer._step = 0
        trainer._optimizer_updates = 0
        trainer._episode_reward_components = {}
        trainer.performance_profiler = Profiler()
        trainer.reward_normalizer = TaskRewardNormalizer(
            gamma=0.9,
            allowed_tasks=["only"],
        )
        trainer.reward_normalizer.normalize(2.0, task="only", stream=0)
        trainer.reward_normalizer.normalize(3.0, task="only", stream=1)
        trainer._evaluate_and_log = lambda: None

        def reset_many(env_indices):
            self.assertEqual(list(env_indices), [0, 1])
            self.assertEqual(trainer.reward_normalizer._returns, {})
            self.assertEqual(trainer.reward_normalizer._stream_tasks, {})
            raise ResetReached

        trainer.env = SimpleNamespace(num_envs=2, reset_many=reset_many)
        trainer.eval_env = trainer.env

        with self.assertRaises(ResetReached):
            trainer._train_multi_env(num_envs=2)


if __name__ == "__main__":
    unittest.main()
