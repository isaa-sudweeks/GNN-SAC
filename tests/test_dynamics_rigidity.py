"""Regression checks for geometry-only rigidity and source simulator agreement."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))

from dynamics.rigidity import critical_eigenvalue, normalized_rigidity
from scripts.run_simulator_mpc_pilot import make_source_env


class DynamicsRigidityTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.edges = np.asarray(list(itertools.combinations(range(4), 2)))

    def test_translation_permutation_and_weighted_normalization(self):
        value = critical_eigenvalue(self.positions, self.edges)
        self.assertGreater(value, 0)
        self.assertAlmostEqual(value, critical_eigenvalue(self.positions + [5, 2, -4], self.edges))
        permutation = np.asarray([2, 0, 3, 1])
        inverse = np.argsort(permutation)
        self.assertAlmostEqual(value, critical_eigenvalue(self.positions[permutation], inverse[self.edges]))
        self.assertAlmostEqual(value / 12, critical_eigenvalue(self.positions, self.edges, weighted=True))
        self.assertAlmostEqual(1, normalized_rigidity(self.positions, self.edges, value))

    def test_failure_inputs_and_insufficient_bars(self):
        self.assertEqual(0, critical_eigenvalue(self.positions, self.edges[:2]))
        self.assertTrue(np.isnan(critical_eigenvalue(self.positions * np.nan, self.edges)))
        with self.assertRaises(ValueError):
            normalized_rigidity(self.positions, self.edges, 0)
        with self.assertRaises(ValueError):
            critical_eigenvalue(self.positions, np.asarray([[0, 9]]))

    def test_matches_nominal_source_simulator_on_normal_trajectory(self):
        env = make_source_env(OmegaConf.create({'topology': 'tetrahedron'}))
        try:
            env.reset()
            base = env.unwrapped
            base.reset(seed=9200)
            wrapper = base.mj_model
            mapping = {name: i for i, name in enumerate(wrapper.node_names)}
            edges = np.asarray([[mapping[a], mapping[b]] for a, b in wrapper.structural_edges])
            rng = np.random.default_rng(9201)
            for _ in range(12):
                positions = wrapper.get_node_position_matrix()
                predicted = critical_eigenvalue(positions, edges,
                    axis_indices=tuple(wrapper.axis_indices), weighted=wrapper.wcrm)
                self.assertAlmostEqual(predicted, wrapper._critical_eig(), places=12)
                _, _, done, _ = env.step(rng.uniform(-0.5, 0.5, env.action_space.shape))
                if done:
                    break
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
