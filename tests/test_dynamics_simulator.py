import copy
import pickle
from pathlib import Path
import sys
import unittest

import numpy as np
from omegaconf import OmegaConf

from env.mujoco import make_env
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))
from dynamics.simulator import SimulatorBranch


def make_source_env(max_steps=1000):
    """Small source-only nominal fixture; no W&B, saved artifacts or rendering."""
    cfg = OmegaConf.create({
        "task": "truss-graph:tetrahedron", "obs": "state", "save_video": False,
        "visualize": False, "max_steps": max_steps, "nsubsteps": 100,
        "speed": 0.05, "use_control_graph": True, "truss_realistic": False,
        "domain_randomization": False, "normalize_observations": True, "seed": 4101,
        "enable_wandb": False, "save_model": False,
    })
    env = make_env(cfg)
    env.reset()
    return env


class SimulatorBranchTests(unittest.TestCase):
    def setUp(self):
        self.live = make_source_env()
        self.planning = make_source_env()
        self.branch = SimulatorBranch(self.live, self.planning)
        shape = self.live.action_space.shape
        self.actions = np.random.default_rng(97).uniform(-0.8, 0.8, (8, *shape))

    def tearDown(self):
        self.live.close()
        self.planning.close()

    def test_repeated_candidates_and_order_leave_live_untouched(self):
        snapshot = self.branch.capture()
        before = self.live.unwrapped.mj_model.data.qpos.copy()
        first = self.branch.rollout(snapshot, self.actions)
        self.branch.rollout(snapshot, -self.actions)
        repeated = self.branch.rollout(snapshot, self.actions)
        np.testing.assert_array_equal(first.rewards, repeated.rewards)
        np.testing.assert_array_equal(first.com_positions, repeated.com_positions)
        np.testing.assert_array_equal(before, self.live.unwrapped.mj_model.data.qpos)
        self.assertEqual(self.live.unwrapped.steps, 0)
        self.assertEqual(self.live._t, 0)

    def test_restored_branch_matches_uninterrupted_live(self):
        for action in self.actions[:3]:
            self.live.step(action)
        snapshot = self.branch.capture()
        expected = [self.live.step(action) for action in self.actions[3:]]
        actual = self.branch.rollout(snapshot, self.actions[3:])
        np.testing.assert_array_equal(actual.rewards, [item[1] for item in expected])
        for obs, item in zip(actual.observations, expected):
            for key in obs:
                np.testing.assert_array_equal(obs[key], item[0][key])
        np.testing.assert_array_equal(self.live.unwrapped.mj_model.data.qpos,
                                      self.planning.unwrapped.mj_model.data.qpos)
        self.assertEqual(self.live._t, self.planning._t)
        self.assertEqual(self.live.env._cumulative_reward,
                         self.planning.env._cumulative_reward)

    def test_timeout_and_rng_are_restored(self):
        self.live._t = self.live.max_episode_steps - 1
        self.planning.unwrapped.np_random.random(23)
        snapshot = self.branch.capture()
        expected_rng = copy.deepcopy(self.live.unwrapped.np_random)
        result = self.branch.rollout(snapshot, self.actions)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.rewards), 1)
        self.assertEqual(expected_rng.random(), self.planning.unwrapped.np_random.random())

    def test_rejects_same_env_and_nonfinite_actions(self):
        with self.assertRaises(ValueError):
            SimulatorBranch(self.live, self.live)
        with self.assertRaises(ValueError):
            self.branch.rollout(self.branch.capture(), np.full_like(self.actions, np.nan))

    def test_serialized_full_snapshot_preserves_raw_reward(self):
        for action in self.actions[:3]:
            self.live.step(action)
        snapshot = pickle.loads(pickle.dumps(self.branch.capture()))
        expected = self.live.step(self.actions[3])
        actual = self.branch.rollout(snapshot, self.actions[3:4])
        self.assertEqual(float(actual.rewards[0]), float(expected[1]))
        np.testing.assert_array_equal(actual.com_positions[-1], self.live.unwrapped._center_of_mass())


if __name__ == "__main__":
    unittest.main()
