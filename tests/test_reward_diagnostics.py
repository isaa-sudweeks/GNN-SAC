import unittest
from types import SimpleNamespace

import numpy as np

from trainer.online_trainer import OnlineTrainer


class _Agent:
    def act(self, obs, t0=False, eval_mode=False):
        return 0.0


class _EvalEnv:
    def __init__(self):
        self.step_index = 0

    def reset(self, task_idx=None):
        self.step_index = 0
        return 0.0

    def step(self, action):
        self.step_index += 1
        done = self.step_index == 2
        critical_eig = 0.4 if self.step_index == 1 else 0.2
        return (
            0.0,
            1.0,
            done,
            {
                "success": float(done),
                "terminated": False,
                "truncated": done,
                "forward": 2.0,
                "energy": -0.5,
                "slip": -0.25,
                "critical_eig": critical_eig,
            },
        )


class RewardDiagnosticsTest(unittest.TestCase):
    def make_trainer(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(
            energy_weight=0.1,
            slip_weight=0.25,
            eval_episodes=1,
            save_video=False,
        )
        trainer._episode_reward_components = {}
        trainer.agent = _Agent()
        trainer.eval_env = _EvalEnv()
        return trainer

    def test_accumulation_tracks_raw_penalties_minimum_and_per_step_values(self):
        trainer = self.make_trainer()
        trainer._accumulate_reward_components(
            {"energy": -0.5, "slip": -0.25, "critical_eig": 0.4}
        )
        trainer._accumulate_reward_components(
            {"energy": -0.25, "slip": -0.5, "critical_eig": 0.2}
        )

        metrics = trainer._finalize_reward_components(
            trainer._episode_reward_components, 2
        )

        self.assertAlmostEqual(metrics["energy_penalty_raw"], 7.5)
        self.assertAlmostEqual(metrics["slip_penalty_raw"], 3.0)
        self.assertAlmostEqual(metrics["episode_min_critical_eig"], 0.2)
        self.assertAlmostEqual(metrics["energy_per_step"], -0.375)
        self.assertAlmostEqual(metrics["critical_eig_per_step"], 0.3)

    def test_eval_reports_reward_breakdown_and_rigidity_minimum(self):
        trainer = self.make_trainer()

        metrics = trainer._eval_one()

        self.assertAlmostEqual(metrics["episode_reward"], 2.0)
        self.assertAlmostEqual(metrics["reward/forward"], 4.0)
        self.assertAlmostEqual(metrics["reward/forward_per_step"], 2.0)
        self.assertAlmostEqual(metrics["reward/energy_penalty_raw"], 10.0)
        self.assertAlmostEqual(metrics["reward/slip_penalty_raw"], 2.0)
        self.assertAlmostEqual(metrics["reward/episode_min_critical_eig"], 0.2)
        self.assertAlmostEqual(metrics["reward/critical_eig_per_step"], 0.3)
        self.assertEqual(trainer._episode_reward_components, {})

    def test_grouped_eval_metrics_keep_group_and_global_component_means(self):
        metrics = {}
        aggregate = {}
        OnlineTrainer._merge_grouped_eval_metrics(
            metrics,
            {
                "episode_reward": 1.0,
                "episode_success": 1.0,
                "episode_length": 2.0,
                "reward/forward_per_step": 3.0,
            },
            "octahedron",
            aggregate,
        )

        self.assertEqual(metrics["octahedron/reward/forward_per_step"], 3.0)
        self.assertEqual(np.nanmean(aggregate["reward/forward_per_step"]), 3.0)


if __name__ == "__main__":
    unittest.main()
