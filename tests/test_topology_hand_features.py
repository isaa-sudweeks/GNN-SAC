import unittest

from mujoco_truss_gen import MujocoModel, get_mujoco_spec

from figures.topology_hand_features import compute_hand_features


class TopologyHandFeaturesTest(unittest.TestCase):
    def test_tetrahedron_node_and_edge_counts(self):
        features = compute_hand_features("tetrahedron")
        self.assertEqual(features["n_nodes"], 4)
        # A tetrahedron is fully connected: 4 choose 2 structural edges.
        self.assertEqual(features["n_edges"], 6)
        self.assertEqual(features["family"], "tetrahedron")

    def test_octahedron_node_count(self):
        features = compute_hand_features("octahedron")
        self.assertEqual(features["n_nodes"], 6)
        self.assertEqual(features["family"], "octahedron")

    def test_features_are_finite_and_sane(self):
        for topology in ("tetrahedron", "octahedron"):
            features = compute_hand_features(topology)
            self.assertGreater(features["static_rigidity"], 0.0)
            self.assertGreater(features["algebraic_connectivity"], 0.0)
            self.assertGreaterEqual(features["diameter"], 1.0)
            self.assertGreaterEqual(features["n_articulation_points"], 0)
            self.assertGreaterEqual(features["n_bridges"], 0)
            self.assertGreater(features["spectral_radius"], 0.0)

    def test_active_and_passive_node_counts_partition_control_nodes(self):
        for topology in ("tetrahedron", "octahedron"):
            features = compute_hand_features(topology)
            control_node_count = len(
                MujocoModel(get_mujoco_spec(topology)).control_graph.control_node_names
            )
            self.assertEqual(
                features["n_active_nodes"] + features["n_passive_nodes"],
                control_node_count,
            )
            self.assertGreaterEqual(features["active_node_fraction"], 0.0)
            self.assertLessEqual(features["active_node_fraction"], 1.0)
            self.assertGreaterEqual(features["active_dispersion"], 0.0)
            self.assertGreaterEqual(features["passive_dispersion"], 0.0)
            self.assertGreaterEqual(features["active_passive_separation"], 0.0)

    def test_tube_and_connector_counts_are_non_negative(self):
        features = compute_hand_features("henneberg_n6_2tube_1")
        self.assertGreaterEqual(features["n_tubes"], 0)
        self.assertGreaterEqual(features["n_connectors"], 0)
        self.assertEqual(features["family"], "henneberg")


if __name__ == "__main__":
    unittest.main()
