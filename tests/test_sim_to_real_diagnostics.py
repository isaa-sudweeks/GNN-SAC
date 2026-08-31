import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
SCRIPTS_ROOT = ROOT / "scripts"
for path in (ROOT, SAC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_sim_to_real import (
    _real_capture,
    _recording_sha256,
    _simulation_capture,
    _plot_comparison,
    _write_3d_video,
    kabsch_transform,
    paired_metrics,
    repeated_trial_metrics,
    tracker_health_metrics,
)
from replay_real_robot_commands import (
    ScheduledCommand,
    _assert_graph_parity,
    _command_action,
    _latest_started_index,
    _load_commands,
    _schedule_end_time,
    run_replay,
)
from real_robot_infer import (
    MissingTrackerFrame,
    PrintCommandTransport,
    RealRobotSessionRecorder,
    SerialVelocityCommandFormatter,
    _run_control_loop,
)
from sim_to_real_io import (
    JsonlWriter,
    parse_firmware_command,
    read_jsonl,
    resolve_triangle_graph_definition,
)
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg


def triangle_graph():
    return {
        "triangles": [
            {"name": "triangle_1", "nodes": ["node_1", "node_2", "node_4"], "passive_node": "node_1", "rollers": {"node_2": "02", "node_4": "04"}},
            {"name": "triangle_2", "nodes": ["node_1", "node_5", "node_3"], "passive_node": "node_1", "rollers": {"node_5": "06", "node_3": "07"}},
            {"name": "triangle_3", "nodes": ["node_3", "node_6", "node_2"], "passive_node": "node_6", "rollers": {"node_3": "08", "node_2": "10"}},
            {"name": "triangle_4", "nodes": ["node_4", "node_6", "node_5"], "passive_node": "node_6", "rollers": {"node_4": "11", "node_5": "12"}},
        ]
    }


class SharedContractTest(unittest.TestCase):
    def test_graph_resolves_passive_routing_and_serial_order(self):
        graph = resolve_triangle_graph_definition(triangle_graph())
        self.assertEqual(len(graph.control_node_names), 12)
        self.assertEqual(
            graph.passive_control_node_names,
            ("node_1", "node_1_tri_triangle_2", "node_6", "node_6_tri_triangle_4"),
        )
        self.assertEqual(len(graph.actuator_edges), 8)
        self.assertEqual(graph.serial_node_order[0], "node_2")
        self.assertEqual(graph.triangle_dict["triangle_3"][-1], "node_6")
        self.assertEqual(len(graph.sha256), 64)

    def test_firmware_parser_rejects_invalid_zero_and_range(self):
        command = parse_firmware_command("VEL_DUR:1800,-900:0.2")
        self.assertEqual(command.ticks, (1800, -900))
        self.assertFalse(command.emergency_stop)
        self.assertTrue(parse_firmware_command("VEL_DUR:0,0:0").emergency_stop)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            parse_firmware_command("VEL_DUR:1,0:0")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            parse_firmware_command("VEL_DUR:1801,0:0.2")

    def test_jsonl_reader_recovers_prior_complete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            with JsonlWriter(path) as writer:
                writer.write("session", value=1)
                writer.write("tracker_frame", value=2)
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"partial"')
            records = read_jsonl(path)
            self.assertEqual([record["type"] for record in records], ["session", "tracker_frame"])

    def test_replay_rejects_recorded_graph_hash_mismatch(self):
        graph = resolve_triangle_graph_definition(triangle_graph())
        metadata = SimpleNamespace(
            control_node_names=graph.control_node_names,
            passive_control_node_names=graph.passive_control_node_names,
            edges=[
                SimpleNamespace(
                    from_node=source,
                    to_node=target,
                    type="actuated" if role == "tube" else "connector",
                )
                for source, target, role in graph.edges
            ],
            actuator_edges=[
                SimpleNamespace(from_node=source, to_node=target)
                for source, target in graph.actuator_edges
            ],
        )
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            _assert_graph_parity(
                graph,
                metadata,
                {"graph_definition_sha256": "0" * 64},
            )


class ReplaySmokeTest(unittest.TestCase):
    def test_recorded_schedule_rejects_decreasing_times_and_inconsistent_fields(self):
        cases = (
            ({"ticks": [99], "duration_s": 0.1, "serial_node_order": ["node"]}, "ticks"),
            ({"ticks": [100], "duration_s": 0.2, "serial_node_order": ["node"]}, "duration_s"),
            ({"ticks": [100], "duration_s": 0.1, "serial_node_order": ["other"]}, "serial_node_order"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (fields, message) in enumerate(cases):
                path = root / f"inconsistent-{index}.jsonl"
                with JsonlWriter(path) as writer:
                    writer.write(
                        "session",
                        max_velocity_ticks_per_second=1800,
                        serial_node_order=["node"],
                    )
                    writer.write(
                        "command_attempt",
                        step=0,
                        relative_time_s=1.0,
                        command="VEL_DUR:100:0.1",
                        **fields,
                    )
                with self.assertRaisesRegex(ValueError, message):
                    _load_commands(path, limit=1800)

            decreasing = root / "decreasing.jsonl"
            with JsonlWriter(decreasing) as writer:
                writer.write("session", max_velocity_ticks_per_second=1800)
                writer.write(
                    "command_attempt", step=0, relative_time_s=1.0,
                    command="VEL_DUR:100:0.1",
                )
                writer.write(
                    "command_attempt", step=1, relative_time_s=0.9,
                    command="VEL_DUR:200:0.1",
                )
            with self.assertRaisesRegex(ValueError, "nondecreasing"):
                _load_commands(decreasing, limit=1800)

    def test_recorded_schedule_rejects_invalid_schema_or_session_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            future = root / "future.jsonl"
            future.write_text(
                json.dumps({
                    "schema_version": 2,
                    "type": "session",
                    "max_velocity_ticks_per_second": 1800,
                })
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported schema version"):
                _load_commands(future, limit=1800)

            duplicate = root / "duplicate.jsonl"
            with JsonlWriter(duplicate) as writer:
                writer.write("session", max_velocity_ticks_per_second=1800)
                writer.write("session", max_velocity_ticks_per_second=1800)
            with self.assertRaisesRegex(ValueError, "exactly one session"):
                _load_commands(duplicate, limit=1800)

    def test_recorded_schedule_preserves_durations_gaps_and_emergency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded.jsonl"
            with JsonlWriter(path) as writer:
                writer.write("session", max_velocity_ticks_per_second=900)
                writer.write(
                    "command_attempt", step=0, relative_time_s=1.0,
                    command="VEL_DUR:900:0.1",
                )
                writer.write(
                    "command_result", step=0, relative_time_s=1.01,
                    command="VEL_DUR:900:0.1", status="acknowledged",
                )
                writer.write(
                    "command_attempt", step=1, relative_time_s=1.5,
                    command="VEL_DUR:-450:0.2",
                )
                writer.write(
                    "command_result", step=1, relative_time_s=1.51,
                    command="VEL_DUR:-450:0.2", status="acknowledged",
                )
                writer.write(
                    "emergency_command", step=2, relative_time_s=1.75,
                    command="VEL_DUR:0:0", reason="Ctrl-C",
                )
            scheduled, _, limit = _load_commands(path, limit=1800)
            self.assertEqual(limit, 900)
            self.assertEqual(
                [(item.start_time_s, item.duration_s) for item in scheduled],
                [(0.0, 0.1), (0.5, 0.2), (0.75, 0.0)],
            )
            self.assertTrue(scheduled[-1].command.emergency_stop)

    def test_recorded_schedule_marks_uncertain_command_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded.jsonl"
            with JsonlWriter(path) as writer:
                writer.write("session", max_velocity_ticks_per_second=1800)
                writer.write(
                    "command_attempt", step=0, relative_time_s=0.0,
                    command="VEL_DUR:100:0.1",
                )
                writer.write(
                    "command_result", step=0, relative_time_s=0.1,
                    command="VEL_DUR:100:0.1", status="acknowledgment_timeout",
                )
                writer.write(
                    "emergency_command", step=0, relative_time_s=0.1,
                    command="VEL_DUR:0:0", reason="acknowledgment timeout",
                )
            scheduled, _, _ = _load_commands(path, limit=1800)
            self.assertEqual(scheduled[0].delivery_status, "acknowledgment_timeout")
            self.assertTrue(scheduled[0].delivery_uncertain)
            self.assertFalse(scheduled[1].delivery_uncertain)

    def test_emergency_only_recording_is_a_valid_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded.jsonl"
            with JsonlWriter(path) as writer:
                writer.write("session", max_velocity_ticks_per_second=1800)
                writer.write(
                    "emergency_command", step=0, relative_time_s=0.002,
                    command="VEL_DUR:0:0", reason="Ctrl-C",
                )
            scheduled, _, _ = _load_commands(path, limit=1800)
            self.assertEqual(len(scheduled), 1)
            self.assertTrue(scheduled[0].command.emergency_stop)

    def test_emergency_only_recording_ends_replay_without_stepping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_graph = triangle_graph()
            graph = resolve_triangle_graph_definition(raw_graph)
            graph_path = root / "graph.json"
            input_path = root / "recorded.jsonl"
            output_path = root / "replay.jsonl"
            graph_path.write_text(json.dumps(raw_graph), encoding="utf-8")
            with JsonlWriter(input_path) as writer:
                writer.write(
                    "session",
                    graph_definition_sha256=graph.sha256,
                    node_order=list(graph.control_node_names),
                    control_node_mapping=[
                        {"triangle": triangle, "logical_node": logical, "control_node": control}
                        for triangle, logical, control in graph.control_node_occurrences
                    ],
                    action_mask=[
                        name not in graph.passive_control_node_names
                        for name in graph.control_node_names
                    ],
                    passive_nodes=list(graph.passive_control_node_names),
                    edges=list(graph.edges),
                    routed_actuator_edges=list(graph.actuator_edges),
                    serial_node_order=list(graph.serial_node_order),
                    max_velocity_ticks_per_second=1800,
                )
                writer.write(
                    "emergency_command", step=0, relative_time_s=0.002,
                    command="VEL_DUR:0,0,0,0,0,0,0,0:0", reason="Ctrl-C",
                )
            cfg = graph_test_cfg(
                task="truss-graph", truss_topology="octahedron", truss_realistic=True,
                use_control_graph=True, nsubsteps=1, max_steps=10,
                domain_randomization=False,
            )
            cfg.input_file = str(input_path)
            cfg.graph_definition_file = str(graph_path)
            cfg.output_file = str(output_path)
            cfg.visualize = False
            cfg.max_velocity_ticks_per_second = 1800
            run_replay(cfg)
            records = read_jsonl(output_path)
            frames = [record for record in records if record["type"] == "simulation_frame"]
            endings = [record for record in records if record["type"] == "replay_end"]
            self.assertEqual(len(frames), 1)
            self.assertEqual(endings[-1]["reason"], "emergency_stop")
            self.assertEqual(endings[-1]["simulation_time_s"], 0.0)

    def test_latest_overlapping_command_wins_and_does_not_resume(self):
        scheduled = [
            ScheduledCommand(0.0, 1.0, parse_firmware_command("VEL_DUR:100:1")),
            ScheduledCommand(0.25, 0.1, parse_firmware_command("VEL_DUR:200:0.1")),
        ]
        self.assertEqual(_latest_started_index(scheduled, 0.2), 0)
        self.assertEqual(_latest_started_index(scheduled, 0.3), 1)
        latest = scheduled[_latest_started_index(scheduled, 0.5)]
        self.assertLessEqual(latest.start_time_s + latest.duration_s, 0.5)
        self.assertAlmostEqual(_schedule_end_time(scheduled), 0.35)

    def test_recorded_serial_override_and_limit_define_normalized_action(self):
        graph = resolve_triangle_graph_definition(triangle_graph())
        override = tuple(reversed(graph.serial_node_order))
        ticks = [0] * len(override)
        ticks[0] = 900
        command = parse_firmware_command(
            f"VEL_DUR:{','.join(str(value) for value in ticks)}:0.1", limit=900
        )
        action = _command_action(command, graph, 900, serial_order=override)
        self.assertEqual(float(action[graph.control_node_names.index(override[0]), 0]), 1.0)

        metadata = SimpleNamespace(
            control_node_names=graph.control_node_names,
            passive_control_node_names=graph.passive_control_node_names,
            edges=[
                SimpleNamespace(
                    from_node=source, to_node=target,
                    type="actuated" if role == "tube" else "connector",
                )
                for source, target, role in graph.edges
            ],
            actuator_edges=[
                SimpleNamespace(from_node=source, to_node=target)
                for source, target in graph.actuator_edges
            ],
        )
        session = {
            "graph_definition_sha256": graph.sha256,
            "node_order": list(graph.control_node_names),
            "control_node_mapping": [
                {"triangle": triangle, "logical_node": logical, "control_node": control}
                for triangle, logical, control in graph.control_node_occurrences
            ],
            "action_mask": [
                name not in graph.passive_control_node_names
                for name in graph.control_node_names
            ],
            "passive_nodes": list(graph.passive_control_node_names),
            "edges": list(graph.edges),
            "routed_actuator_edges": list(graph.actuator_edges),
            "serial_node_order": list(override),
        }
        _assert_graph_parity(graph, metadata, session)
        session["serial_node_order"][-1] = session["serial_node_order"][0]
        with self.assertRaisesRegex(ValueError, "every actuated graph node exactly once"):
            _assert_graph_parity(graph, metadata, session)

    def test_replay_builds_graph_defined_mujoco_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "graph.json"
            command_path = root / "commands.txt"
            output_path = root / "replay.jsonl"
            graph_path.write_text(json.dumps(triangle_graph()), encoding="utf-8")
            command_path.write_text("VEL_DUR:100,0,0,0,0,0,0,0:0.002\n", encoding="utf-8")
            cfg = graph_test_cfg(
                task="truss-graph",
                truss_topology="octahedron",
                truss_realistic=True,
                use_control_graph=True,
                nsubsteps=1,
                max_steps=10,
                domain_randomization=False,
            )
            cfg.input_file = str(command_path)
            cfg.graph_definition_file = str(graph_path)
            cfg.output_file = str(output_path)
            cfg.visualize = False
            cfg.max_velocity_ticks_per_second = 1800
            run_replay(cfg)
            records = read_jsonl(output_path)
            session = records[0]
            self.assertEqual(session["passive_nodes"], [
                "node_1", "node_1_tri_triangle_2", "node_6", "node_6_tri_triangle_4"
            ])
            frames = [record for record in records if record["type"] == "simulation_frame"]
            self.assertGreaterEqual(len(frames), 2)
            self.assertEqual(len(frames[-1]["routed_actuator_ctrl"]), 8)

    def test_recorded_session_replays_and_emergency_command_ends_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_graph = triangle_graph()
            graph = resolve_triangle_graph_definition(raw_graph)
            graph_path = root / "graph.json"
            input_path = root / "recorded.jsonl"
            output_path = root / "replay.jsonl"
            graph_path.write_text(json.dumps(raw_graph), encoding="utf-8")
            with JsonlWriter(input_path) as writer:
                writer.write(
                    "session",
                    graph_definition_sha256=graph.sha256,
                    node_order=list(graph.control_node_names),
                    control_node_mapping=[
                        {"triangle": triangle, "logical_node": logical, "control_node": control}
                        for triangle, logical, control in graph.control_node_occurrences
                    ],
                    action_mask=[
                        name not in graph.passive_control_node_names
                        for name in graph.control_node_names
                    ],
                    passive_nodes=list(graph.passive_control_node_names),
                    edges=list(graph.edges),
                    routed_actuator_edges=list(graph.actuator_edges),
                    serial_node_order=list(graph.serial_node_order),
                    max_velocity_ticks_per_second=1800,
                )
                writer.write(
                    "command_attempt",
                    step=0,
                    relative_time_s=0.0,
                    command="VEL_DUR:100,0,0,0,0,0,0,0:0.002",
                )
                writer.write(
                    "command_result",
                    step=0,
                    relative_time_s=0.001,
                    command="VEL_DUR:100,0,0,0,0,0,0,0:0.002",
                    status="acknowledged",
                )
                writer.write(
                    "emergency_command",
                    step=1,
                    relative_time_s=0.006,
                    command="VEL_DUR:0,0,0,0,0,0,0,0:0",
                )
            cfg = graph_test_cfg(
                task="truss-graph", truss_topology="octahedron", truss_realistic=True,
                use_control_graph=True, nsubsteps=1, max_steps=10, domain_randomization=False,
            )
            cfg.input_file = str(input_path)
            cfg.graph_definition_file = str(graph_path)
            cfg.output_file = str(output_path)
            cfg.visualize = False
            cfg.max_velocity_ticks_per_second = 1800
            run_replay(cfg)
            records = read_jsonl(output_path)
            replay_session = records[0]
            self.assertEqual(replay_session["source_commands"], [
                {
                    "command": "VEL_DUR:100,0,0,0,0,0,0,0:0.002",
                    "start_time_s": 0.0,
                    "delivery_status": "acknowledged",
                    "delivery_uncertain": False,
                },
                {
                    "command": "VEL_DUR:0,0,0,0,0,0,0,0:0",
                    "start_time_s": 0.006,
                    "delivery_status": "emergency_stop",
                    "delivery_uncertain": False,
                },
            ])
            self.assertEqual(len(replay_session["source_recording_sha256"]), 64)
            end = [record for record in records if record["type"] == "replay_end"]
            self.assertEqual(end[-1]["reason"], "emergency_stop")
            frames = [
                record for record in records
                if record["type"] == "simulation_frame" and record["step"] > 0
            ]
            self.assertEqual(frames[0]["command"], "VEL_DUR:100,0,0,0,0,0,0,0:0.002")
            self.assertTrue(all(frame["command"] is None for frame in frames[1:]))
            self.assertTrue(all(np.allclose(frame["normalized_action"], 0.0) for frame in frames[1:]))


class RecordingTest(unittest.TestCase):
    def test_control_loop_records_skipped_raw_frames_actions_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "session.jsonl"

            class Source:
                calls = 0

                def positions_by_serial(self):
                    self.calls += 1
                    return {"S": np.asarray([self.calls, 0.0, 0.0])}

            class Layout:
                requires_orientations = False
                node_names = ("node",)

                def ordered_positions_with_diagnostics(self, frame):
                    if float(frame["S"][0]) == 1.0:
                        raise MissingTrackerFrame("synthetic dropout")
                    return np.asarray([[0.0, 0.0, 0.1]]), {"synthetic": True}

            class Builder:
                def build(self, positions, timestamp):
                    return SimpleNamespace(rigidity=torch.tensor([1.0]))

            class Agent:
                def act(self, observation, **kwargs):
                    return torch.tensor([[0.5]])

            cfg = SimpleNamespace(control_frequency_hz=1000.0, control_steps=1, deterministic=True)
            formatter = SerialVelocityCommandFormatter(
                ("node",), max_velocity_ticks_per_second=1800, duration_seconds=0.2
            )
            recorder = RealRobotSessionRecorder(output, {"node_order": ["node"]})
            with redirect_stdout(StringIO()):
                _run_control_loop(
                    cfg, Source(), Layout(), Builder(), Agent(), formatter,
                    PrintCommandTransport(), recorder,
                )
            recorder.close()
            records = read_jsonl(output)
            skipped = [record for record in records if record["type"] == "tracker_frame" and record["status"] == "skipped"]
            complete = [record for record in records if record["type"] == "tracker_frame" and record["status"] == "complete"]
            attempts = [record for record in records if record["type"] == "command_attempt"]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(len(complete), 1)
            self.assertEqual(attempts[0]["ticks"], [900])
            self.assertEqual(attempts[0]["serial_node_order"], ["node"])


class AnalysisMathTest(unittest.TestCase):
    @staticmethod
    def capture(positions, *, graph_hash="hash"):
        positions = np.asarray(positions, dtype=float)
        times = np.arange(len(positions), dtype=float)
        node_order = tuple(f"n{index}" for index in range(positions.shape[1]))
        commands = [{"command": "VEL_DUR:0,0,0:1"}]
        return {
            "path": Path("synthetic.jsonl"),
            "session": {"graph_definition_sha256": graph_hash, "edges": []},
            "frames": [{"relative_time_s": float(t), "status": "complete"} for t in times],
            "complete": [],
            "commands": commands,
            "node_order": node_order,
            "positions": positions,
            "times": times,
            "rigidity": np.ones(len(times)),
            "end_reason": "complete",
        }

    def test_kabsch_and_paired_metrics_remove_fixed_rigid_transform(self):
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sim = np.stack((base, base + [0.1, 0.0, 0.0], base + [0.2, 0.0, 0.0]))
        angle = np.pi / 3.0
        rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        real_positions = sim @ rotation.T + [2.0, -1.0, 0.5]
        real = self.capture(real_positions)
        simulation = self.capture(sim)
        result = paired_metrics(real, simulation)
        self.assertLess(result["summary"]["aggregate_rmse_m"], 1e-10)
        self.assertLess(np.max(result["com_error"]), 1e-10)
        self.assertFalse(result["summary"]["alignment_uses_scale_fitting"])
        self.assertAlmostEqual(result["summary"]["initial_scale_ratio_real_to_simulation"], 1.0)

    def test_paired_metrics_reports_scale_mismatch_without_fitting_it_away(self):
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        simulation = self.capture(np.stack((base, base)))
        real = self.capture(np.stack((1.05 * base + [2.0, -1.0, 0.5],) * 2))
        result = paired_metrics(real, simulation)
        self.assertAlmostEqual(
            result["summary"]["initial_scale_ratio_real_to_simulation"], 1.05
        )
        self.assertGreater(result["summary"]["initial_shape_rmse_after_rigid_alignment_m"], 0.0)

    @staticmethod
    def write_file_backed_pair(
        root: Path,
        *,
        real_hash: str = "graph-hash",
        simulation_hash: str = "graph-hash",
        real_command: str = "VEL_DUR:0,0,0,0:1",
        source_command: str = "VEL_DUR:0,0,0,0:1",
        source_start_time_s: float = 0.0,
        source_recording_sha256: str | None = None,
    ):
        base = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        node_order = ["n0", "n1", "n2", "n3"]
        real_path = root / "real.jsonl"
        with JsonlWriter(real_path) as writer:
            writer.write(
                "session", graph_definition_sha256=real_hash,
                node_order=node_order, edges=[],
            )
            writer.write(
                "command_attempt", relative_time_s=0.5, command=real_command,
            )
            writer.write(
                "tracker_frame", relative_time_s=0.5, status="complete",
                node_positions=base, node_order=node_order, rigidity=1.0,
            )
            writer.write(
                "tracker_frame", relative_time_s=1.5, status="complete",
                node_positions=base, node_order=node_order, rigidity=1.0,
            )
        if source_recording_sha256 is None:
            source_recording_sha256 = _recording_sha256(read_jsonl(real_path))

        simulation_path = root / "simulation.jsonl"
        with JsonlWriter(simulation_path) as writer:
            writer.write(
                "replay_session", graph_definition_sha256=simulation_hash,
                node_order=node_order, edges=[],
                source_recording_sha256=source_recording_sha256,
                source_commands=[{
                    "command": source_command,
                    "start_time_s": source_start_time_s,
                }],
            )
            writer.write(
                "simulation_frame", simulation_time_s=0.0,
                node_positions=base, rigidity=1.0,
            )
            writer.write(
                "simulation_frame", simulation_time_s=1.0,
                node_positions=base, rigidity=1.0,
            )
        return _real_capture(real_path), _simulation_capture(simulation_path)

    def test_file_backed_pair_rejects_graph_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            real, simulation = self.write_file_backed_pair(
                Path(directory), simulation_hash="different-graph-hash"
            )
            with self.assertRaisesRegex(ValueError, "graph hashes must match"):
                paired_metrics(real, simulation)

    def test_file_backed_pair_rejects_command_or_timing_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real, simulation = self.write_file_backed_pair(
                root, source_command="VEL_DUR:1,0,0,0:1"
            )
            with self.assertRaisesRegex(ValueError, "source commands differ"):
                paired_metrics(real, simulation)

            real, simulation = self.write_file_backed_pair(
                root, source_start_time_s=0.25
            )
            with self.assertRaisesRegex(ValueError, "command timing differs"):
                paired_metrics(real, simulation)

    def test_file_backed_pair_rejects_unrelated_source_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            real, simulation = self.write_file_backed_pair(
                Path(directory), source_recording_sha256="0" * 64
            )
            with self.assertRaisesRegex(ValueError, "not bound to this real capture"):
                paired_metrics(real, simulation)

    def test_capture_loaders_reject_invalid_schema_or_session_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_path = root / "real.jsonl"
            with JsonlWriter(real_path) as writer:
                writer.write(
                    "tracker_frame", relative_time_s=0.0, status="complete",
                    node_positions=[[[0.0, 0.0, 0.0]]],
                )
                writer.write("session", node_order=["node"])
            with self.assertRaisesRegex(ValueError, "begin with exactly one session"):
                _real_capture(real_path)

            replay_path = root / "replay.jsonl"
            replay_path.write_text(
                json.dumps({
                    "schema_version": 2,
                    "type": "replay_session",
                    "node_order": ["node"],
                })
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported schema version"):
                _simulation_capture(replay_path)

    def test_repeated_trial_variance_is_across_trials(self):
        base = np.stack((
            np.zeros((3, 3)),
            np.ones((3, 3)),
            np.full((3, 3), 2.0),
        ))
        first = self.capture(base)
        second = self.capture(base + 0.2)
        result = repeated_trial_metrics([first, second])
        np.testing.assert_allclose(result["position_variance_xyz"], 0.01)

    def test_stationary_health_reports_dropout_and_drift(self):
        positions = np.asarray([[[0, 0, 0]], [[0.01, 0, 0]]], dtype=float)
        capture = self.capture(positions)
        capture["complete"] = [
            {"relative_time_s": 0.0, "raw_poses_by_serial": {"S": {"position": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()}}},
            {"relative_time_s": 2.0, "raw_poses_by_serial": {"S": {"position": [0.01, 0, 0], "rotation_matrix": np.eye(3).tolist()}}},
        ]
        capture["frames"] = [
            {**capture["complete"][0], "status": "complete"},
            {"relative_time_s": 1.0, "status": "skipped", "raw_poses_by_serial": {}},
            {**capture["complete"][1], "status": "complete"},
        ]
        capture["node_order"] = ("n0",)
        metrics = tracker_health_metrics(capture)
        self.assertEqual(metrics["dropout_frames"], 1)
        self.assertAlmostEqual(metrics["trackers"]["S"]["drift_from_initial_m"], 0.01)

    def test_png_and_mp4_smoke(self):
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        positions = np.stack((base, base + [0.05, 0.0, 0.0]))
        real = self.capture(positions)
        simulation = self.capture(positions)
        result = paired_metrics(real, simulation)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _plot_comparison(result, root / "errors.png", root / "shape.png")
            _write_3d_video(
                root / "nodes.mp4",
                result["times"], result["node_order"], result["real_positions"],
                [(0, 1), (1, 2), (2, 0)],
                secondary=result["simulation_positions_aligned"], fps=5,
            )
            self.assertGreater((root / "errors.png").stat().st_size, 0)
            self.assertGreater((root / "shape.png").stat().st_size, 0)
            self.assertGreater((root / "nodes.mp4").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
