import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_robot_infer import (
    RealRobotObservationBuilder,
    SerialVelocityCommandFormatter,
    TrackerLayout,
    TrackerMount,
    TrackerPose,
    _run_print_only_control_loop,
)
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

    def test_coordinate_matrix_swaps_steamvr_y_and_z_for_policy(self):
        self.layout.steamvr_to_policy_matrix = np.asarray(
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float
        )
        positions = self.layout.ordered_positions(
            {"S0": [1, 2, 3], "S1": [4, 5, 6], "S2": [7, 8, 9]}
        )
        np.testing.assert_array_equal(
            positions,
            [[1, 3, 2], [4, 6, 5], [7, 9, 8]],
        )

    def test_complete_hand_authored_graph_supports_triangle_reconstruction(self):
        self.layout_path.write_text(json.dumps({
            "tracker_assignment": "triangle_planes",
            "node_names": ["n0", "n1", "n2"],
            "actuated_nodes": ["n0", "n2"],
            "edges": [["n0", "n1", "tube"], ["n1", "n2", "connector"]],
            "tracker_mounts": {
                "T0": {
                    "triangle": "tri_0",
                    "abstract_node": "n0",
                    "joint_triangles": ["tri_0", "tri_1"],
                },
                "T1": {
                    "triangle": "tri_1",
                    "abstract_node": "n1",
                    "joint_triangles": ["tri_1", "tri_2"],
                },
                "T2": {
                    "triangle": "tri_2",
                    "abstract_node": "n2",
                    "joint_triangles": ["tri_2", "tri_0"],
                },
            },
        }))
        layout = TrackerLayout.from_files(str(self.serial_path), str(self.layout_path))
        self.assertTrue(layout.requires_orientations)
        self.assertEqual(set(layout.tracker_mounts), {"T0", "T1", "T2"})
        np.testing.assert_array_equal(layout.action_mask, [True, False, True])

    def test_triangle_definition_generates_mujoco_style_control_graph(self):
        self.serial_path.write_text(json.dumps({
            "serial_to_tracker_id": {
                "S1": "B11", "S2": "B12", "S3": "B13",
                "S4": "B14", "S5": "B15", "S6": "B16",
            }
        }))
        self.layout_path.write_text(json.dumps({
            "triangles": [
                {
                    "name": "triangle_1",
                    "nodes": ["node_1", "node_2", "node_4"],
                    "passive_node": "node_1",
                    "trackers": {"node_1": "B11", "node_2": "B12"},
                    "rollers": {"node_2": "04", "node_4": "02"},
                },
                {
                    "name": "triangle_2",
                    "nodes": ["node_1", "node_5", "node_3"],
                    "passive_node": "node_1",
                    "trackers": {"node_3": "B13", "node_5": "B15"},
                    "rollers": {"node_5": "08", "node_3": "07"},
                },
                {
                    "name": "triangle_3",
                    "nodes": ["node_3", "node_6", "node_2"],
                    "passive_node": "node_6",
                    "trackers": {"node_6": "B16"},
                    "rollers": {"node_3": "06", "node_2": "05"},
                },
                {
                    "name": "triangle_4",
                    "nodes": ["node_4", "node_6", "node_5"],
                    "passive_node": "node_6",
                    "trackers": {"node_4": "B14"},
                    "rollers": {"node_4": "03", "node_5": "01"},
                },
            ]
        }))

        layout = TrackerLayout.from_files(str(self.serial_path), str(self.layout_path))

        self.assertEqual(layout.node_names, (
            "node_1", "node_2", "node_4",
            "node_1_tri_triangle_2", "node_5", "node_3",
            "node_3_tri_triangle_3", "node_6", "node_2_tri_triangle_3",
            "node_4_tri_triangle_4", "node_6_tri_triangle_4",
            "node_5_tri_triangle_4",
        ))
        np.testing.assert_array_equal(
            layout.action_mask,
            [False, True, True, False, True, True, True, False, True, True, False, True],
        )
        self.assertEqual(layout.edge_index.shape, (2, 36))
        self.assertEqual(np.count_nonzero(layout.edge_role == 0), 24)
        self.assertEqual(np.count_nonzero(layout.edge_role == 1), 12)
        self.assertEqual(
            layout.tracker_mounts["B14"].joint_triangles,
            ("triangle_1", "triangle_4"),
        )
        self.assertEqual(layout.tracker_mounts["B14"].abstract_node, "node_4")
        self.assertEqual(layout.serial_node_order, (
            "node_5_tri_triangle_4", "node_4", "node_4_tri_triangle_4", "node_2",
            "node_2_tri_triangle_3", "node_3_tri_triangle_3", "node_3", "node_5",
        ))
        formatter = SerialVelocityCommandFormatter(
            layout.node_names,
            layout.action_mask,
            layout.serial_node_order,
            max_velocity_ticks_per_second=100,
            duration_seconds=0.2,
        )
        normalized = np.arange(len(layout.node_names), dtype=float) / 20.0
        self.assertEqual(
            formatter.velocity_command(normalized),
            "VEL_DUR:55,10,45,5,40,30,25,20:0.2",
        )

    def test_triangle_rollers_require_every_actuated_occurrence(self):
        with self.assertRaisesRegex(ValueError, "every actuated triangle node"):
            TrackerLayout._expand_triangle_definition({
                "triangles": [{
                    "name": "triangle_1",
                    "nodes": ["node_1", "node_2", "node_3"],
                    "passive_node": "node_2",
                    "trackers": {},
                    "rollers": {"node_1": "02"},
                }]
            })

    def test_triangle_rollers_reject_duplicate_numeric_ids(self):
        with self.assertRaisesRegex(ValueError, "Roller numbers must be unique"):
            TrackerLayout._expand_triangle_definition({
                "triangles": [{
                    "name": "triangle_1",
                    "nodes": ["node_1", "node_2", "node_3"],
                    "passive_node": "node_2",
                    "trackers": {},
                    "rollers": {"node_1": "02", "node_3": "2"},
                }]
            })

    def test_multiple_trackers_on_one_triangle_fuse_their_plane_estimates(self):
        rotation_local_z_to_x = np.asarray(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float
        )
        rotation_local_z_to_y = np.asarray(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float
        )
        mounts = {
            "T0": TrackerMount("T0", "tri_a", "n0", ("tri_a", "tri_b"), np.asarray([0, 0, 1]), np.zeros(3)),
            "T1": TrackerMount("T1", "tri_a", "n1", ("tri_a", "tri_c"), np.asarray([0, 0, 1]), np.zeros(3)),
            "T2": TrackerMount("T2", "tri_b", "n2", ("tri_b", "tri_a"), np.asarray([0, 0, 1]), np.zeros(3)),
            "T3": TrackerMount("T3", "tri_c", "n3", ("tri_c", "tri_a"), np.asarray([0, 0, 1]), np.zeros(3)),
        }
        layout = TrackerLayout(
            node_names=("n0", "n1", "n2", "n3"),
            serial_to_tracker_id={"S0": "T0", "S1": "T1", "S2": "T2", "S3": "T3"},
            tracker_id_to_node={},
            edge_index=np.empty((2, 0), dtype=np.int64),
            action_mask=np.ones(4, dtype=bool),
            edge_role=None,
            steamvr_to_policy_matrix=np.eye(3),
            assignment_mode="triangle_planes",
            tracker_mounts=mounts,
        )
        layout._validate_tracker_mounts()
        planes, _ = layout._triangle_planes({
            "S0": TrackerPose(np.asarray([0.0, 0.0, 0.0]), rotation_local_z_to_x),
            "S1": TrackerPose(np.asarray([0.2, 2.0, 0.0]), rotation_local_z_to_x),
            "S2": TrackerPose(np.asarray([0.0, 0.0, 0.0]), rotation_local_z_to_y),
            "S3": TrackerPose(np.asarray([0.0, 0.0, 0.0]), np.eye(3)),
        })
        plane_point, plane_normal = planes["tri_a"]
        np.testing.assert_allclose(plane_normal, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(plane_point, [0.1, 0.0, 0.0], atol=1e-12)

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

    def test_triangle_plane_layout_accepts_one_mount_per_abstract_preset_node(self):
        cfg = graph_test_cfg(
            truss_topology="octahedron",
            domain_randomization=False,
        )
        cfg.tracker_assignment = "automatic"
        cfg.steamvr_to_policy_matrix = np.eye(3).tolist()
        preset = TrackerLayout.from_preset(cfg, str(self.serial_path), None)
        abstract_nodes = sorted(
            {TrackerLayout._abstract_node_name(name) for name in preset.node_names}
        )
        serial_map = {
            f"S{index}": f"T{index}" for index in range(len(abstract_nodes))
        }
        self.serial_path.write_text(json.dumps({"serial_to_tracker_id": serial_map}))
        tracker_mounts = {}
        for index, abstract_node in enumerate(abstract_nodes):
            triangle = f"triangle_{index}"
            next_triangle = f"triangle_{(index + 1) % len(abstract_nodes)}"
            tracker_mounts[f"T{index}"] = {
                "triangle": triangle,
                "abstract_node": abstract_node,
                "joint_triangles": [triangle, next_triangle],
                "local_plane_normal": [0, 0, 1],
            }
        self.layout_path.write_text(json.dumps({"tracker_mounts": tracker_mounts}))
        cfg.tracker_assignment = "triangle_planes"
        reconstructed = TrackerLayout.from_preset(
            cfg, str(self.serial_path), str(self.layout_path)
        )
        self.assertEqual(len(reconstructed.tracker_mounts), len(abstract_nodes))
        self.assertEqual(len(abstract_nodes), 6)
        self.assertEqual(len(reconstructed.node_names), 12)
        self.assertTrue(reconstructed.requires_orientations)

    def test_reconstructs_joint_and_expands_it_to_control_graph_duplicates(self):
        rotation_local_z_to_x = np.asarray(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float
        )
        rotation_local_z_to_y = np.asarray(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float
        )
        mounts = {
            "T0": TrackerMount(
                tracker_id="T0",
                triangle="tri_a",
                abstract_node="node_0",
                joint_triangles=("tri_a", "tri_b"),
                local_plane_normal=np.asarray([0, 0, 1], dtype=float),
                local_plane_point_offset=np.zeros(3),
            ),
            "T1": TrackerMount(
                tracker_id="T1",
                triangle="tri_b",
                abstract_node="node_1",
                joint_triangles=("tri_b", "tri_a"),
                local_plane_normal=np.asarray([0, 0, 1], dtype=float),
                local_plane_point_offset=np.zeros(3),
            ),
        }
        layout = TrackerLayout(
            node_names=("node_0_tri_a", "node_0_tri_b", "node_1_tri_a"),
            serial_to_tracker_id={"S0": "T0", "S1": "T1"},
            tracker_id_to_node={},
            edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            action_mask=np.ones(3, dtype=bool),
            edge_role=None,
            steamvr_to_policy_matrix=np.eye(3),
            assignment_mode="triangle_planes",
            tracker_mounts=mounts,
        )
        layout._validate_tracker_mounts()
        positions = layout.reconstructed_positions(
            {
                "S0": TrackerPose(
                    position=np.asarray([0, 2, 3], dtype=float),
                    rotation_matrix=rotation_local_z_to_x,
                ),
                "S1": TrackerPose(
                    position=np.asarray([4, 0, 5], dtype=float),
                    rotation_matrix=rotation_local_z_to_y,
                ),
            }
        )
        np.testing.assert_allclose(
            positions,
            [[0, 0, 3], [0, 0, 3], [0, 0, 5]],
            atol=1e-12,
        )

    def test_reconstruction_rejects_parallel_triangle_planes(self):
        mounts = {
            tracker_id: TrackerMount(
                tracker_id=tracker_id,
                triangle=triangle,
                abstract_node=abstract_node,
                joint_triangles=("tri_a", "tri_b"),
                local_plane_normal=np.asarray([0, 0, 1], dtype=float),
                local_plane_point_offset=np.zeros(3),
            )
            for tracker_id, triangle, abstract_node in (
                ("T0", "tri_a", "node_0"),
                ("T1", "tri_b", "node_1"),
            )
        }
        layout = TrackerLayout(
            node_names=("node_0", "node_1"),
            serial_to_tracker_id={"S0": "T0", "S1": "T1"},
            tracker_id_to_node={},
            edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            action_mask=np.ones(2, dtype=bool),
            edge_role=None,
            steamvr_to_policy_matrix=np.eye(3),
            assignment_mode="triangle_planes",
            tracker_mounts=mounts,
        )
        poses = {
            "S0": TrackerPose(np.zeros(3), np.eye(3)),
            "S1": TrackerPose(np.ones(3), np.eye(3)),
        }
        with self.assertRaisesRegex(RuntimeError, "parallel"):
            layout.reconstructed_positions(poses)


class SerialVelocityCommandFormatterTest(unittest.TestCase):
    def test_formats_firmware_velocity_command_in_transmitter_order(self):
        formatter = SerialVelocityCommandFormatter(
            ("n0", "n1", "n2"),
            None,
            ("n2", "n0", "n1"),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        command = formatter.velocity_command(np.asarray([[0.25], [-0.5], [2.0]]))
        self.assertEqual(command, "VEL_DUR:1800,450,-900:0.2")

    def test_emergency_stop_has_zero_for_every_transmitter_channel(self):
        formatter = SerialVelocityCommandFormatter(
            ("n0", "n1", "n2", "n3"),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        self.assertEqual(formatter.emergency_stop_command(), "VEL_DUR:0,0,0,0:0")

    def test_ctrl_c_prints_emergency_stop(self):
        class InterruptedSource:
            def positions_by_serial(self):
                raise KeyboardInterrupt

        formatter = SerialVelocityCommandFormatter(
            ("n0", "n1", "n2"),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        cfg = SimpleNamespace(
            control_frequency_hz=10.0,
            control_steps=None,
            deterministic=True,
            speed=0.05,
        )
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            _run_print_only_control_loop(
                cfg,
                InterruptedSource(),
                layout=SimpleNamespace(ordered_positions=lambda positions: positions),
                builder=None,
                agent=None,
                formatter=formatter,
            )
        self.assertEqual(stdout.getvalue().strip(), "VEL_DUR:0,0,0:0")
        self.assertIn("Emergency stop", stderr.getvalue())

    def test_passive_graph_nodes_are_omitted_from_transmitter_fields(self):
        formatter = SerialVelocityCommandFormatter(
            ("n0", "passive", "n2"),
            (True, False, True),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        self.assertEqual(
            formatter.velocity_command(np.asarray([0.5, 0.0, -0.25])),
            "VEL_DUR:900,-450:0.2",
        )
        self.assertEqual(formatter.emergency_stop_command(), "VEL_DUR:0,0:0")


if __name__ == "__main__":
    unittest.main()
