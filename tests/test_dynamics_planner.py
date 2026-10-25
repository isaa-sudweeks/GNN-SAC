import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))
from dynamics.planner import CEMConfig, cem_plan, interpolate_knots, planning_score


class DynamicsPlannerTests(unittest.TestCase):
    def test_interpolation_preserves_action_axis(self):
        result = interpolate_knots(np.array([[[0, 1], [1, -1]]]), 3)
        np.testing.assert_allclose(result, [[[0, 1], [0.5, 0], [1, -1]]])

    def test_finds_known_optimum_reproducibly_with_mask(self):
        cfg = CEMConfig(candidates=128, iterations=5, horizon=8, knots=3)
        def evaluate(actions):
            self.assertTrue(np.all(actions[:, :, 1] == 0))
            self.assertLessEqual(np.abs(actions).max(), 1)
            return -np.mean((actions[:, :, 0] - 0.6) ** 2, axis=1)
        first = cem_plan(evaluate, 2, cfg, np.random.default_rng(4), action_mask=[1, 0])
        second = cem_plan(evaluate, 2, cfg, np.random.default_rng(4), action_mask=[1, 0])
        np.testing.assert_array_equal(first.actions, second.actions)
        self.assertGreater(first.score, -0.005)
        self.assertTrue(np.all(np.diff(first.iteration_best) >= 0))
        self.assertEqual(first.sequences_scored, 640)

    def test_warm_start_is_shifted_and_evaluated(self):
        cfg = CEMConfig(horizon=3, knots=3, candidates=1, iterations=1)
        previous = np.array([[0.1], [0.3], [0.7]])
        result = cem_plan(lambda actions: np.zeros(len(actions)), 1, cfg,
                          np.random.default_rng(0), previous_plan=previous)
        np.testing.assert_allclose(result.actions[:, 0], [0.3, 0.7, 0.7])

    def test_nonfinite_scores_and_wrong_shape_fail(self):
        cfg = CEMConfig(candidates=2)
        with self.assertRaises(RuntimeError):
            cem_plan(lambda x: np.full(2, np.nan), 1, cfg, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            cem_plan(lambda x: np.zeros((2, 1)), 1, cfg, np.random.default_rng(0))

    def test_failure_penalty_increases_with_remaining_time(self):
        kwargs = dict(initial_diagonal=2, squared_command_integral=0.1,
                      effort_weight=0.2, failure_cost=1, failure_cost_per_second=2)
        alive = planning_score(0.4, collapsed=False, remaining_seconds=0, **kwargs)
        late = planning_score(0.4, collapsed=True, remaining_seconds=0, **kwargs)
        early = planning_score(0.4, collapsed=True, remaining_seconds=1, **kwargs)
        self.assertAlmostEqual(alive, 0.18)
        self.assertAlmostEqual(late, -0.82)
        self.assertAlmostEqual(early, -2.82)

    def test_invalid_settings_rejected(self):
        for kwargs in ({"knots": 11}, {"minimum_std": 0}, {"iterations": 0},
                       {"elite_fraction": 0}, {"initial_std": float("inf")}):
            with self.assertRaises(ValueError):
                CEMConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
