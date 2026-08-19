import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_robot_infer import RealRobotObservationBuilder, TrackerLayout
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg


class RealRobotObservationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.serial_path = root / "serial.json"
        self.layout_path = root / "layout.json"
        self.serial_path.write_text(json.dumps({"serial_to_tracker_id": {"S0": "T0", "S1": "T1", "S2": "T2"}}))
        self.layout_path.write_text(json.dumps({
            "node_names": ["n0", "n1", "n2"],
            "tracker_id_to_node": {"T2": "n2", "T0": "n0", "T1": "n1"},
            "actuated_nodes": ["n0", "n2"],
            "edges": [["n0", "n1", "tube"], ["n1", "n2", "connector"]],
        }))
        self.layout = TrackerLayout.from_files(str(self.serial_path), str(self.layout_path))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_maps_serials_to_stable_node_order(self):
        positions = self.layout.ordered_positions({"S2": [2, 4, 6], "S0": [0, 0, 0], "S1": [1, 2, 3]})
        np.testing.assert_array_equal(positions, [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_equal(self.layout.action_mask, [True, False, True])
        np.testing.assert_array_equal(self.layout.edge_role, [0, 0, 1, 1])

    def test_builds_com_relative_normalized_position_and_velocity(self):
        builder = RealRobotObservationBuilder(self.layout, True, 1.0, (True, True, True))
        initial = np.asarray([[0, 0, 0], [1, 2, 3], [2, 4, 6]], dtype=float)
        first = builder.build(initial, 10.0)
        np.testing.assert_allclose(first.x[:, :3], [[-0.5, -0.5, -0.5], [0, 0, 0], [0.5, 0.5, 0.5]])
        np.testing.assert_array_equal(first.x[:, 3:], 0)
        self.assertTrue(np.isfinite(float(first.rigidity)))
        second = builder.build(initial + [1, 2, 3], 12.0)
        np.testing.assert_allclose(second.x[:, :3], first.x[:, :3])
        np.testing.assert_allclose(second.x[:, 3:], 0.25)

    def test_missing_tracker_skips_frame(self):
        with self.assertRaisesRegex(RuntimeError, "n1"):
            self.layout.ordered_positions({"S0": [0, 0, 0], "S2": [2, 4, 6]})

    def test_automatic_assignment_matches_nearest_centered_preset_nodes(self):
        automatic = TrackerLayout(
            node_names=("left", "top", "right"),
            serial_to_tracker_id={"S0": "T0", "S1": "T1", "S2": "T2"},
            tracker_id_to_node={},
            edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]]),
            action_mask=np.ones(3, dtype=bool),
            edge_role=None,
            steamvr_to_policy_matrix=np.eye(3),
            reference_positions=np.asarray([[-1, 0, 0], [0, 2, 0], [1, 0, 0]], dtype=float),
            assignment_mode="automatic",
        )
        ordered = automatic.ordered_positions({
            "S0": [11, 5, 0],
            "S1": [9, 5, 0],
            "S2": [10, 7, 0],
        })
        np.testing.assert_array_equal(
            ordered,
            [[9, 5, 0], [10, 7, 0], [11, 5, 0]],
        )
        self.assertEqual(
            automatic.tracker_id_to_node,
            {"T0": "right", "T1": "left", "T2": "top"},
        )

    def test_preset_supplies_graph_metadata_and_nominal_positions(self):
        cfg = graph_test_cfg(
            truss_topology="octahedron",
            domain_randomization=False,
            graph_features={
                "node_roles": False,
                "edge_roles": True,
                "edge_distance": False,
            },
        )
        cfg.tracker_assignment = "automatic"
        cfg.steamvr_to_policy_matrix = np.eye(3).tolist()
        preset = TrackerLayout.from_preset(cfg, str(self.serial_path), None)
        self.assertEqual(preset.reference_positions.shape, (len(preset.node_names), 3))
        self.assertEqual(preset.edge_index.shape[0], 2)
        self.assertEqual(preset.edge_role.shape[0], preset.edge_index.shape[1])
        self.assertEqual(preset.action_mask.shape, (len(preset.node_names),))


if __name__ == "__main__":
    unittest.main()
