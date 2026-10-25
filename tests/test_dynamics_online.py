"""Regression checks for online real replay, TD bootstrapping and geometry gating."""
import unittest
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "sac")]

import numpy as np
from omegaconf import OmegaConf
import torch

from dynamics.online import OnlineLearner, OnlineReplay, td_target
from dynamics.rigidity import normalized_rigidity
from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state


class TestOnlineTargets(unittest.TestCase):
    def test_true_termination_stops_bootstrap(self):
        result = td_target(torch.tensor([1., 1.]), torch.tensor([True, False]),
                           torch.tensor([10., 10.]), .9)
        torch.testing.assert_close(result, torch.tensor([1., 10.]))

    def test_sequences_do_not_cross_reset_or_truncation(self):
        replay = OnlineReplay(8, 2)
        for i in range(6):
            replay.append(state=np.zeros((2, 6)), next_state=np.zeros((2, 6)),
                action=np.zeros((2, 1)), progress=np.zeros(2), reward=0., base_reward=0.,
                terminated=False, truncated=i == 2, episode=i // 3)
        self.assertEqual(replay.starts(3).tolist(), [0, 3])
        self.assertEqual(replay.starts(4).tolist(), [])
        self.assertEqual(replay.size, 6)
        with self.assertRaises(ValueError):
            replay.starts(0)


class TestOnlineNativeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = OmegaConf.to_container(OmegaConf.load('config/dynamics_online.yaml'))
        cls.env = make_training_env('tetrahedron', OmegaConf.create(cls.cfg))
        cls.env.reset()
        obs, _ = cls.env.unwrapped.reset(seed=789100)
        cls.metadata = graph_metadata(cls.env.unwrapped, obs)
        cls.metadata['control_interval_seconds'] = .2
        cls.mapping = np.asarray(cls.metadata['node_to_physical'])
        cls.state, _ = observable_state(cls.env.unwrapped, cls.mapping)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_vectorized_rigidity_matches_audited_numpy(self):
        agent = OnlineLearner(self.metadata, self.state, self.cfg, 4, True)
        state = np.stack([self.state, self.state.copy(), self.state.copy()])
        state[1, :, :3] *= [1., 1., .001]
        state[2, :, :3] *= [1., 1., .00000001]
        order = agent.graph['physical_order_index'].numpy()
        expected = [normalized_rigidity(x[order, :3], np.asarray(self.metadata['structural_edges']),
            self.metadata['initial_rigidity_reference'], axis_indices=tuple(self.metadata['axis_indices']),
            weighted=self.metadata['wcrm']) < self.metadata['critical_eig_threshold'] for x in state]
        self.assertEqual(agent.collapsed(torch.tensor(state, dtype=torch.float32)).tolist(), expected)

    def test_real_replay_update_and_model_only_planning(self):
        from scripts.run_online_dynamics_pilot import actual_step
        for use_value in (False, True):
            self.env.reset(); self.env.unwrapped.reset(seed=789100)
            agent = OnlineLearner(self.metadata, self.state, self.cfg, 4, use_value)
            replay = OnlineReplay(16, len(self.mapping))
            for i in range(12):
                row, _, done = actual_step(self.env, self.metadata, self.mapping,
                    np.zeros((len(self.mapping), 1)), self.cfg)
                replay.append(**row, episode=0)
                self.assertFalse(done)
            agent.normalize_from_warmup(replay)
            before = [p.detach().clone() for p in agent.model.dynamics.parameters()]
            for _ in range(2):
                losses = agent.update(replay)
                self.assertTrue(all(np.isfinite(v) for v in losses.values()))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.model.dynamics.parameters())))
            state, _ = observable_state(self.env.unwrapped, self.mapping)
            sim_time = float(self.env.unwrapped.mj_model.data.time)
            plan = agent.plan(state, np.random.default_rng(8))
            self.assertEqual(plan.actions.shape, (10, len(self.mapping)))
            self.assertLessEqual(np.abs(plan.actions).max(), 1.)
            self.assertEqual(float(self.env.unwrapped.mj_model.data.time), sim_time)
            self.assertEqual(replay.size, 12)


if __name__ == '__main__':
    unittest.main()
