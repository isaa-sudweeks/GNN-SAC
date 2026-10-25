import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sac"))

from dynamics.state import (
    center_xy, geometry_features, integrate_centered_positions,
    mass_weighted_com, observation_to_physical,
)


class DynamicsStateTests(unittest.TestCase):
    def test_normalization_roundtrip(self):
        physical = np.arange(18).reshape(3, 6) / 10
        bbox = np.array([2., 3., 4.])
        np.testing.assert_allclose(observation_to_physical(
            physical / np.tile(bbox, 2), bbox, normalized=True), physical)

    def test_centering_preserves_height_and_translation_progress(self):
        p = np.array([[1., 2., 3.], [3., 4., 5.]])
        centered, origin = center_xy(p)
        np.testing.assert_allclose(origin, [2, 3])
        np.testing.assert_allclose(centered[:, 2], p[:, 2])
        result, delta = integrate_centered_positions(centered, np.ones_like(p), .2)
        np.testing.assert_allclose(delta, [.2, .2])
        np.testing.assert_allclose(result[:, :2], centered[:, :2])
        np.testing.assert_allclose(result[:, 2], p[:, 2] + .2)

    def test_geometry_translation_permutation_batching_and_refresh(self):
        p = np.array([[0., 0., 1.], [3., 4., 1.], [0., 0., 3.]])
        edges = np.array([[0, 1, 0], [1, 2, 2]])
        relative, distance = geometry_features(p, edges)
        shifted = geometry_features(p + [9., -7., 0.], edges)
        np.testing.assert_allclose(shifted[0], relative)
        permutation = np.array([2, 0, 1])
        inverse = np.argsort(permutation)
        reordered = geometry_features(p[permutation], inverse[edges])
        np.testing.assert_allclose(reordered[0], relative)
        batched = geometry_features(np.concatenate([p, p + 8]),
                                    np.concatenate([edges, edges + 3], axis=1))
        np.testing.assert_allclose(batched[1], np.concatenate([distance, distance]))
        moved = p.copy()
        moved[1, 0] += 1
        self.assertNotEqual(geometry_features(moved, edges)[1][0, 0], distance[0, 0])

    def test_com_is_mass_weighted_not_node_mean(self):
        np.testing.assert_allclose(mass_weighted_com(
            np.array([[0, 0, 0], [4, 0, 0]]), np.array([3, 1])), [1, 0, 0])

    def test_invalid_fields_fail(self):
        with self.assertRaises(ValueError):
            observation_to_physical(np.zeros((2, 6)), [0, 1, 1], normalized=True)
        with self.assertRaises(ValueError):
            geometry_features(np.zeros((2, 3)), np.array([[0], [2]]))


if __name__ == "__main__":
    unittest.main()
