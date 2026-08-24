import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_robot_infer import (
    PrintCommandTransport,
    RealRobotObservationBuilder,
    SerialAcknowledgmentTimeout,
    SerialCommandTransport,
    SerialVelocityCommandFormatter,
    TrackerLayout,
    TrackerMount,
    TrackerPose,
    _make_command_transport,
    _run_control_loop,
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


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))


class _FakeSerial:
    def __init__(self, clock, lines=()):
        self.clock = clock
        self.lines = list(lines)
        self.writes = []
        self.flush_count = 0
        self.reset_input_buffer_count = 0
        self.is_open = True
        self.constructor_kwargs = None

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        self.flush_count += 1

    def reset_input_buffer(self):
        self.reset_input_buffer_count += 1

    def readline(self):
        self.clock.sleep(0.01 if self.lines else 0.1)
        if not self.lines:
            return b""
        line = self.lines.pop(0)
        if callable(line):
            line = line(self)
        return line.encode("utf-8") + b"\n"

    def close(self):
        self.is_open = False


class SerialCommandTransportTest(unittest.TestCase):
    def _make_transport(self, lines=(), **overrides):
        clock = _FakeClock()
        fake_serial = _FakeSerial(clock, lines)

        def serial_factory(**kwargs):
            fake_serial.constructor_kwargs = kwargs
            return fake_serial

        options = {
            "ack_timeout_seconds": 0.5,
            "startup_delay_seconds": 0.0,
            "serial_factory": serial_factory,
            "monotonic": clock.monotonic,
            "sleep": clock.sleep,
        }
        options.update(overrides)
        transport = SerialCommandTransport(
            "/dev/fake",
            115200,
            ("roller_a", "roller_b", "roller_c"),
            **options,
        )
        return transport, fake_serial, clock

    def test_uses_configured_serial_settings_and_startup_delay(self):
        transport, fake_serial, clock = self._make_transport(
            startup_delay_seconds=2.0
        )
        self.assertEqual(
            fake_serial.constructor_kwargs,
            {
                "port": "/dev/fake",
                "baudrate": 115200,
                "timeout": 0.1,
                "write_timeout": 0.5,
            },
        )
        self.assertEqual(clock.now, 2.0)
        self.assertEqual(fake_serial.reset_input_buffer_count, 1)
        transport.close()
        self.assertFalse(fake_serial.is_open)

    def test_waits_for_completion_and_reports_node_delivery_metrics(self):
        def assert_only_one_command_in_flight(fake_serial):
            self.assertEqual(fake_serial.writes, [b"VEL_DUR:1,2,3:0.2\n"])
            return "Transmitter ready."

        transport, fake_serial, _ = self._make_transport(
            lines=(
                assert_only_one_command_in_flight,
                "FAILED to send to node 2 after 3 attempts",
                "[10,20,30]",
                "VEL_DUR command completed in 37 ms",
            )
        )
        stdout = StringIO()
        with redirect_stdout(stdout):
            transport.send("VEL_DUR:1,2,3:0.2")

        output = stdout.getvalue()
        self.assertIn("serial rx: Transmitter ready.", output)
        self.assertIn("firmware_ms=37", output)
        self.assertIn("commands_with_failures=1/1", output)
        self.assertIn("node_delivery_success=66.67%", output)
        self.assertIn("current_failed_nodes=[2:roller_b(attempts=3)]", output)
        self.assertEqual(transport.node_drop_counts, {2: 1})
        self.assertEqual(transport.confirmed_commands, 1)

    def test_unrelated_lines_do_not_acknowledge_next_command(self):
        transport, fake_serial, _ = self._make_transport(
            lines=(
                "[1,2,3]",
                "VEL_DUR command completed in 10 ms",
                "[4,5,6]",
                "VEL_DUR command completed in 11 ms",
            )
        )
        transport.send("VEL_DUR:1,2,3:0.2")
        transport.send("VEL_DUR:4,5,6:0.2")
        self.assertEqual(
            fake_serial.writes,
            [b"VEL_DUR:1,2,3:0.2\n", b"VEL_DUR:4,5,6:0.2\n"],
        )
        self.assertEqual(transport.confirmed_commands, 2)

    def test_timeout_reports_stats_and_best_effort_stop_is_written_once(self):
        transport, fake_serial, _ = self._make_transport(
            ack_timeout_seconds=0.25
        )
        with self.assertRaises(SerialAcknowledgmentTimeout) as caught:
            transport.send("VEL_DUR:1,2,3:0.2")
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            transport.report_timeout(caught.exception)
            transport.emergency_stop("VEL_DUR:0,0,0:0", reason="acknowledgment timeout")
        self.assertEqual(
            fake_serial.writes,
            [b"VEL_DUR:1,2,3:0.2\n", b"VEL_DUR:0,0,0:0\n"],
        )
        self.assertIn("ack_timeouts=1", stdout.getvalue())
        self.assertIn("acknowledgment timeout", stderr.getvalue())

    def test_rejects_invalid_serial_configuration_before_opening_port(self):
        with self.assertRaisesRegex(ValueError, "serial_port"):
            SerialCommandTransport(
                "",
                115200,
                ("roller",),
                ack_timeout_seconds=1.0,
                startup_delay_seconds=0.0,
                serial_factory=lambda **kwargs: None,
            )

        invalid_options = (
            ({"baud_rate": 0}, "serial_baud_rate"),
            ({"ack_timeout_seconds": 0.0}, "serial_ack_timeout_s"),
            ({"startup_delay_seconds": -1.0}, "serial_startup_delay_s"),
        )
        for overrides, message in invalid_options:
            options = {
                "baud_rate": 115200,
                "ack_timeout_seconds": 1.0,
                "startup_delay_seconds": 0.0,
            }
            options.update(overrides)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                SerialCommandTransport(
                    "/dev/fake",
                    options["baud_rate"],
                    ("roller",),
                    ack_timeout_seconds=options["ack_timeout_seconds"],
                    startup_delay_seconds=options["startup_delay_seconds"],
                    serial_factory=lambda **kwargs: None,
                )

    def test_command_transport_mode_defaults_to_print_and_rejects_unknown(self):
        formatter = SerialVelocityCommandFormatter(
            ("node",),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        self.assertIsInstance(
            _make_command_transport(SimpleNamespace(), formatter),
            PrintCommandTransport,
        )
        with self.assertRaisesRegex(ValueError, "command_transport"):
            _make_command_transport(
                SimpleNamespace(command_transport="socket"), formatter
            )


class ControlLoopPacingTest(unittest.TestCase):
    @staticmethod
    def _loop_fixture():
        class Source:
            def positions_by_serial(self):
                return {"tracker": np.zeros(3)}

        class Builder:
            def build(self, positions, timestamp):
                return timestamp

        class Action:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray([0.0])

        class Agent:
            def act(self, observation, **kwargs):
                return Action()

        formatter = SerialVelocityCommandFormatter(
            ("node",),
            max_velocity_ticks_per_second=1800,
            duration_seconds=0.2,
        )
        cfg = SimpleNamespace(
            control_frequency_hz=10.0,
            control_steps=3,
            deterministic=True,
        )
        layout = SimpleNamespace(
            ordered_positions=lambda positions: np.zeros((1, 3)),
            requires_orientations=False,
        )
        return cfg, Source(), layout, Builder(), Agent(), formatter

    def test_delayed_ack_skips_missed_slot_without_catch_up_burst(self):
        clock = _FakeClock()

        class DelayedTransport(PrintCommandTransport):
            acknowledgment_gated = True

            def __init__(self):
                self.send_times = []
                self.delays = iter((0.15, 0.0, 0.0))

            def send(self, command):
                self.send_times.append(clock.monotonic())
                clock.sleep(next(self.delays))

        cfg, source, layout, builder, agent, formatter = self._loop_fixture()
        transport = DelayedTransport()
        with patch("real_robot_infer.time.monotonic", clock.monotonic), patch(
            "real_robot_infer.time.sleep", clock.sleep
        ):
            _run_control_loop(
                cfg,
                source,
                layout,
                builder,
                agent,
                formatter,
                transport,
            )
        np.testing.assert_allclose(transport.send_times, [0.0, 0.15, 0.25])

    def test_ack_timeout_stops_loop_and_writes_one_emergency_command(self):
        clock = _FakeClock()
        fake_serial = _FakeSerial(clock)
        transport = SerialCommandTransport(
            "/dev/fake",
            115200,
            ("node",),
            ack_timeout_seconds=0.25,
            startup_delay_seconds=0.0,
            serial_factory=lambda **kwargs: fake_serial,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        cfg, source, layout, builder, agent, formatter = self._loop_fixture()
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch(
            "real_robot_infer.time.monotonic", clock.monotonic
        ), patch("real_robot_infer.time.sleep", clock.sleep):
            _run_control_loop(
                cfg, source, layout, builder, agent, formatter, transport
            )
        self.assertEqual(
            fake_serial.writes,
            [b"VEL_DUR:0:0.2\n", b"VEL_DUR:0:0\n"],
        )
        self.assertIn("ack_timeouts=1", stdout.getvalue())
        self.assertIn("Best-effort emergency stop", stderr.getvalue())

    def test_ctrl_c_writes_only_one_emergency_command(self):
        clock = _FakeClock()
        fake_serial = _FakeSerial(clock)
        transport = SerialCommandTransport(
            "/dev/fake",
            115200,
            ("node",),
            ack_timeout_seconds=0.25,
            startup_delay_seconds=0.0,
            serial_factory=lambda **kwargs: fake_serial,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        cfg, _, layout, builder, agent, formatter = self._loop_fixture()

        class InterruptedSource:
            def positions_by_serial(self):
                raise KeyboardInterrupt

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()), patch(
            "real_robot_infer.time.monotonic", clock.monotonic
        ), patch("real_robot_infer.time.sleep", clock.sleep):
            _run_control_loop(
                cfg,
                InterruptedSource(),
                layout,
                builder,
                agent,
                formatter,
                transport,
            )
        self.assertEqual(fake_serial.writes, [b"VEL_DUR:0:0\n"])


if __name__ == "__main__":
    unittest.main()
