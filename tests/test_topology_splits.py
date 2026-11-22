from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.topology_splits import FEATURE_NAMES, farthest_point_partition, morphology_features
from scripts.launch_cross_validation import build_launch, load_definition
from scripts.make_farthest_point_cross_validation import build_definition, render_definition


class FarthestPointPartitionTest(unittest.TestCase):
    def test_separated_clusters_and_order_independent_prototypes(self):
        features = {"a": [0.0], "b": [0.1], "c": [5.0], "d": [9.9], "e": [10.0]}
        groups, prototypes = farthest_point_partition(features, 3)
        self.assertEqual(prototypes, ["a", "e", "c"])
        self.assertEqual(list(groups.values()), [["a", "b"], ["d", "e"], ["c"]])
        self.assertEqual(
            farthest_point_partition(dict(reversed(list(features.items()))), 3),
            (groups, prototypes),
        )

    def test_unit_changes_and_constant_columns_do_not_change_split(self):
        features = {str(i): [float(i), float(i * i), 7.0] for i in range(10)}
        scaled = {name: [v[0] * 1000 + 42, v[1] * 0.01, 1.0] for name, v in features.items()}
        self.assertEqual(farthest_point_partition(features, 3), farthest_point_partition(scaled, 3))

    def test_identical_descriptors_still_have_nonempty_disjoint_groups(self):
        features = {name: [1.0, 2.0] for name in "abcde"}
        for count in (2, 3, 5):
            groups, prototypes = farthest_point_partition(features, count)
            self.assertTrue(all(groups.values()))
            self.assertEqual(len(set(prototypes)), count)
            self.assertEqual(sorted(name for group in groups.values() for name in group), list("abcde"))

    def test_invalid_vectors_and_fold_counts(self):
        for features in ({"a": [], "b": []}, {"a": [1], "b": [1, 2]},
                         {"a": [np.nan], "b": [1]}, {"a": [np.inf], "b": [1]}):
            with self.assertRaisesRegex(ValueError, "finite vectors"):
                farthest_point_partition(features, 2)
        for count in (1, 4):
            with self.assertRaisesRegex(ValueError, "num_folds"):
                farthest_point_partition({"a": [0], "b": [1], "c": [2]}, count)

    def test_committed_split_covers_pool_and_preserves_final_test(self):
        path = ROOT / "config/cross_validation/farthest_point_5fold.yaml"
        definition = OmegaConf.to_container(OmegaConf.load(path).cross_validation)
        split = definition["split"]
        source = load_definition(split["source"])
        groups, prototypes = farthest_point_partition(split["features"], split["num_folds"])
        self.assertEqual(definition["groups"], groups)
        self.assertEqual(split["prototypes"], prototypes)
        self.assertEqual(split["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(path.read_text(), render_definition(definition))
        pool = {name for group in source["groups"].values() for name in group}
        self.assertEqual(set(split["features"]), pool)
        self.assertEqual(definition["final_test"], source["final_test"])
        self.assertFalse(pool & set(definition["final_test"]))
        _, jobs = build_launch(config_name=definition["name"], spec=load_definition(definition["name"]),
                               seeds=[1, 2], shuffle_seed=17, overrides=[])
        self.assertEqual(len(jobs), 10)
        for job in jobs:
            self.assertEqual(set(job["training_topologies"]) | set(job["heldout_topologies"]), pool)
            self.assertFalse(set(job["training_topologies"]) & set(job["heldout_topologies"]))

    def test_generator_never_describes_final_test(self):
        import mujoco_truss_gen

        nodes = {"a": [0, 0, 0], "b": [1, 0, 0]}
        shapes = {"path": {"route": ["a", "b"], "active_edges": [["a", "b"]]}}
        source = {"groups": {"one": ["first"], "two": ["second"]}, "final_test": ["reserved"]}
        with patch("scripts.make_farthest_point_cross_validation.load_definition", return_value=source), \
             patch.object(mujoco_truss_gen, "PRESETS", {"first": lambda: (nodes, shapes), "second": lambda: (nodes, shapes)}):
            result = build_definition(source="example", num_folds=2, name="example_fps")
        self.assertEqual(result["final_test"], ["reserved"])
        self.assertNotIn("reserved", result["split"]["features"])


class MorphologyFeatureTest(unittest.TestCase):
    nodes = {"a": [0, 0, 0], "b": [1, 0, 0], "c": [0, 1, 0], "d": [0, 0, 1]}
    shapes = {"triangle": ["a", "b", "c", "a"],
              "path": {"route": ["b", "d", "a"], "active_edges": [["b", "d"]]}}

    def test_counts_and_active_edges_follow_route_contract(self):
        result = dict(zip(FEATURE_NAMES, morphology_features(self.nodes, self.shapes)))
        self.assertEqual(result["nodes"], 4)
        self.assertEqual(result["edges"], 5)
        self.assertEqual(result["tubes"], 2)
        self.assertAlmostEqual(result["active_edge_fraction"], 4 / 5)
        self.assertEqual(result["diameter"], 2)

    def test_geometry_and_relabeling_invariance(self):
        renames = {"a": "z", "b": "y", "c": "x", "d": "w"}
        nodes = {renames[k]: [5 + 3 * v[1], 7 - 3 * v[0], -2 + 3 * v[2]] for k, v in self.nodes.items()}
        shapes = {"triangle": [renames[k] for k in self.shapes["triangle"]],
                  "path": {"route": [renames[k] for k in self.shapes["path"]["route"]],
                           "active_edges": [[renames[k] for k in pair] for pair in self.shapes["path"]["active_edges"]]}}
        np.testing.assert_allclose(morphology_features(nodes, shapes), morphology_features(self.nodes, self.shapes), atol=1e-11)

    def test_disconnected_or_invalid_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "connected"):
            morphology_features(self.nodes, {"one": ["a", "b"], "two": ["c", "d"]})
        with self.assertRaisesRegex(ValueError, "positive length|nonzero extent"):
            morphology_features({"a": [0, 0, 0], "b": [0, 0, 0]}, {"path": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
