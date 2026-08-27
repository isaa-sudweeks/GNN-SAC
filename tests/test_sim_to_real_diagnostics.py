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
    _plot_comparison,
    _write_3d_video,
    kabsch_transform,
    paired_metrics,
    repeated_trial_metrics,
    tracker_health_metrics,
)
from replay_real_robot_commands import _assert_graph_parity, run_replay
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
                )
                writer.write(
                    "command_attempt",
                    relative_time_s=1.0,
                    command="VEL_DUR:100,0,0,0,0,0,0,0:0.002",
                )
                writer.write(
                    "command_attempt",
                    relative_time_s=1.002,
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
            end = [record for record in read_jsonl(output_path) if record["type"] == "replay_end"]
            self.assertEqual(end[-1]["reason"], "emergency_stop")


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
