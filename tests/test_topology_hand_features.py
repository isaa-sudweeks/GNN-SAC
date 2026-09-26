import re
import unittest

import numpy as np
from mujoco_truss_gen import MujocoModel, get_mujoco_spec

from figures import _topology_catalog
from figures.topology_hand_features import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    _clean,
    compute_all_hand_features,
    compute_hand_features,
)

_HENNEBERG_TUBES = re.compile(r"^henneberg_n(\d+)_(\d+)tube_\d+$")


class TopologyHandFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One pass over the whole catalog (~40 s), shared by the catalog-wide tests.
        cls.features = compute_all_hand_features(_topology_catalog.all_topologies())

    def _row(self, topology):
        return self.features.set_index("topology").loc[topology]

    def test_output_columns_match_declared_feature_list(self):
        self.assertEqual(list(self.features.columns), [*ID_COLUMNS, *FEATURE_COLUMNS])
        self.assertFalse(self.features[list(FEATURE_COLUMNS)].isna().any().any())

    def test_tetrahedron_node_and_edge_counts(self):
        row = self._row("tetrahedron")
        self.assertEqual(row["n_nodes"], 4)
        # A tetrahedron is fully connected: 4 choose 2 structural edges.
        self.assertEqual(row["n_edges"], 6)
        self.assertEqual(row["family"], "tetrahedron")

    def test_octahedron_node_count(self):
        row = self._row("octahedron")
        self.assertEqual(row["n_nodes"], 6)
        self.assertEqual(row["family"], "octahedron")

    def test_henneberg_tube_count_matches_name(self):
        henneberg = [name for name in self.features["topology"] if _HENNEBERG_TUBES.match(name)]
        self.assertGreaterEqual(len(henneberg), 5)
        for name in henneberg:
            n_nodes, n_tubes = map(int, _HENNEBERG_TUBES.match(name).groups())
            row = self._row(name)
            with self.subTest(topology=name):
                self.assertEqual(row["n_nodes"], n_nodes)
                self.assertEqual(row["n_tubes"], n_tubes)
                # Routed tubes are open paths that partition the members.
                self.assertEqual(row["closed_tube_fraction"], 0.0)
                self.assertAlmostEqual(row["tube_length_mean"] * n_tubes, row["n_edges"])

    def test_non_henneberg_tube_counts(self):
        # Tetrahedron: two open routed tubes; octahedron / usevitch: one closed
        # perimeter-constrained triangle per tube.
        expected = {
            "tetrahedron": (2, 0.0, 3.0),
            "octahedron": (4, 1.0, 3.0),
            "usevitch_1514879": (5, 1.0, 3.0),
            "usevitch_210272254_p1": (6, 1.0, 3.0),
            "usevitch_60243677150_p1": (7, 1.0, 3.0),
        }
        for name, (n_tubes, closed_fraction, length) in expected.items():
            row = self._row(name)
            with self.subTest(topology=name):
                self.assertEqual(row["n_tubes"], n_tubes)
                self.assertEqual(row["closed_tube_fraction"], closed_fraction)
                self.assertEqual(row["tube_length_mean"], length)

    def test_no_other_feature_duplicates_n_edges(self):
        n_edges = self.features["n_edges"].to_numpy(dtype=float)
        for column in FEATURE_COLUMNS:
            if column == "n_edges":
                continue
            with self.subTest(column=column):
                self.assertFalse(
                    np.array_equal(self.features[column].to_numpy(dtype=float), n_edges)
                )

    def test_features_are_finite_and_sane(self):
        for topology in self.features["topology"]:
            row = self._row(topology)
            with self.subTest(topology=topology):
                self.assertGreater(row["static_rigidity"], 0.0)
                self.assertGreater(row["algebraic_connectivity"], 0.0)
                self.assertGreaterEqual(row["diameter"], 1.0)
                self.assertGreater(row["spectral_radius"], 0.0)
                self.assertGreaterEqual(row["n_tubes"], 1)
                self.assertGreaterEqual(row["node_multiplicity_mean"], 1.0)

    def test_floating_point_dust_is_cleaned(self):
        self.assertEqual(_clean(1.7e-12), 0.0)
        self.assertEqual(_clean(3.9999999999999964), 4.0)
        self.assertEqual(_clean(3), 3)
        # Exactly-zero-by-symmetry offsets come out as exact zeros.
        self.assertEqual(self._row("tetrahedron")["active_passive_separation"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
