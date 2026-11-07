"""Deterministic math, protocol, operator, and hardware-free control regressions."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]

from encoder_state import Pose, TwoTrackerFusion, roller_frame
from encoder_robot import build_encoder_geometry, EncoderViveEstimator, make_encoder_layout
from encoder_runtime import run_encoder_loop
from robot_operator import OperatorControls, OperatorEmergencyStop
from real_robot_infer import RealRobotObservationBuilder, RealRobotSessionRecorder, SerialVelocityCommandFormatter, TrackerPose
from sim_to_real_io import read_jsonl
from tests import test_real_robot_infer as legacy_tests
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg


def fusion():
    return TwoTrackerFusion(["B13", "B16"], window=30, position_variance_floor=1e-8,
                            rotation_variance_floor=1e-8, initial_position_variance=1e-4,
                            initial_rotation_variance=1e-4, reacquire_after_s=.5)


class FusionTest(unittest.TestCase):
    def test_mount_frame_follows_corner_bisector_as_triangle_deforms(self):
        pose = roller_frame(np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float))
        np.testing.assert_allclose(pose.rotation[:,0], [2**-.5,2**-.5,0],atol=1e-12)
        pose = roller_frame(np.array([[0,0,0],[1,0,0],[-.5,3**.5/2,0]],dtype=float))
        np.testing.assert_allclose(pose.rotation[:,0], [.5,3**.5/2,0],atol=1e-12)

    def test_full_pose_composition_uses_orientation_and_mount_offset(self):
        world = Pose([2, -1, .6], Rotation.from_euler("xyz", [.2, -.4, 1]).as_matrix())
        mount = Pose([.3, .1, -.2], Rotation.from_euler("xyz", [.7, .1, -.1]).as_matrix())
        recovered = world.compose(mount).compose(mount.inverse())
        np.testing.assert_allclose(recovered.position, world.position, atol=1e-12)
        np.testing.assert_allclose(recovered.rotation, world.rotation, atol=1e-12)

    def test_dropouts_hold_transform_and_reacquisition_resets_stale_variance(self):
        estimator = fusion()
        pose = Pose([1, 2, 3], np.eye(3))
        result, health = estimator.update({}, 0)
        self.assertIsNone(result)
        self.assertEqual(health["status"], "uninitialized")
        result, health = estimator.update({"B13": pose}, .1)
        self.assertEqual(health["status"], "one_tracker")
        result, health = estimator.update({}, .2)
        np.testing.assert_equal(result.position, pose.position)
        self.assertAlmostEqual(health["global_pose_age_s"], .1)
        self.assertFalse(health["global_pose_observed"])
        estimator.update({"B13": pose}, .3)
        estimator.update({"B13": pose}, .4)
        _, health = estimator.update({"B13": pose, "B16": pose}, 2)
        self.assertEqual(len(estimator.history["B13"]), 1)
        self.assertAlmostEqual(health["weights"]["B13"]["position"], .5)

    def test_noisy_tracker_gets_lower_position_and_orientation_weights(self):
        estimator = fusion()
        rng = np.random.default_rng(4)
        for i in range(120):
            common = np.array([i * .01, 0, 0])
            samples = {}
            for key, scale in (("B13", .0002), ("B16", .03)):
                samples[key] = Pose(common + rng.normal(0, scale, 3),
                                    Rotation.from_rotvec(np.array([0, 0, i * .005]) + rng.normal(0, scale, 3)).as_matrix())
            result, health = estimator.update(samples, i * .1)
        self.assertLess(health["weights"]["B16"]["position"], .01)
        self.assertLess(health["weights"]["B16"]["rotation"], .01)
        self.assertLess(np.linalg.norm(result.position - common), .003)
        np.testing.assert_allclose(result.rotation.T @ result.rotation, np.eye(3), atol=1e-12)

    def test_rotations_wrap_without_linear_angle_artifact(self):
        a = Pose([0, 0, 0], Rotation.from_euler("z", 179, degrees=True).as_matrix())
        b = Pose([0, 0, 0], Rotation.from_euler("z", -179, degrees=True).as_matrix())
        result, _ = fusion().update({"B13": a, "B16": b}, 0)
        self.assertAlmostEqual(Rotation.from_matrix(result.rotation).magnitude(), np.pi, places=8)

    def test_rejects_reflections_and_nonmonotonic_time(self):
        with self.assertRaises(ValueError):
            Pose([0, 0, 0], np.diag([1, 1, -1]))
        estimator = fusion()
        estimator.update({}, 1)
        with self.assertRaises(ValueError):
            estimator.update({}, 1)


class OperatorTest(unittest.TestCase):
    def test_ctrl_p_toggles_and_emergency_stop_is_latched(self):
        controls = OperatorControls(enabled=False)
        controls.feed(b"\x10")
        self.assertTrue(controls.poll())
        controls.feed(b"\x10")
        self.assertFalse(controls.poll())
        controls.feed(b"e\x10")
        with self.assertRaises(OperatorEmergencyStop):
            controls.poll()
        self.assertTrue(controls.stopped)
        self.assertTrue(controls.paused)

    def test_ctrl_c_remains_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            OperatorControls(enabled=False).feed(b"\x03")


class EncoderSerialTest(unittest.TestCase):
    _make_transport = legacy_tests.SerialCommandTransportTest._make_transport
    def test_waits_for_encoder_array_and_completion(self):
        transport, serial, _ = self._make_transport(lines=["VEL_DUR command completed in 20 ms", "[100,-20,30]"])
        transport.configure_encoder_feedback(["01", "02", "03"], ["01", "02", "03"])
        transport.send("VEL_DUR:1,2,3:0.2")
        self.assertEqual(transport.latest_encoder_sample["ticks"], [100, -20, 30])
        self.assertFalse(transport.latest_encoder_sample["freshness_verified"])

    def test_maps_wire_slots_and_unused_receiver_failures(self):
        transport, serial, _ = self._make_transport(lines=["FAILED to send to node 2 after 3 attempts", "VEL_DUR command completed in 20 ms", "[10,999,20,30]"])
        transport.configure_encoder_feedback(["01", "09", "02", "03"], ["03", "01", "02"])
        transport.send("VEL_DUR:1,2,3:0.2")
        self.assertEqual(serial.writes, [b"VEL_DUR:2,0,3,1:0.2\n"])
        self.assertEqual(transport.latest_encoder_sample["ticks"], [30, 10, 20])

    def test_failed_active_receiver_does_not_supply_a_valid_encoder_frame(self):
        transport, _, _ = self._make_transport(lines=["FAILED to send to node 1 after 3 attempts", "VEL_DUR command completed in 20 ms", "[10,20,30]"])
        transport.configure_encoder_feedback(["01", "02", "03"], ["01", "02", "03"])
        with self.assertRaisesRegex(ValueError, "delivery failure"):
            transport.send("VEL_DUR:1,2,3:0.2")
        self.assertIsNone(transport.latest_encoder_sample)

    def test_pause_during_ack_sends_zero_and_consumes_its_own_reply(self):
        controls = OperatorControls(enabled=False)
        def pause(_serial):
            controls.request("toggle_pause")
            return "VEL_DUR command completed in 20 ms"
        transport, serial, _ = self._make_transport(lines=[pause, "[1,2,3]", "VEL_DUR command completed in 10 ms", "[4,5,6]"])
        transport.configure_encoder_feedback(["01", "02", "03"], ["01", "02", "03"])
        transport.operator_controls = controls
        transport.pause_command = "VEL_DUR:0,0,0:0.2"
        transport.send("VEL_DUR:1,2,3:0.2")
        self.assertEqual(serial.writes, [b"VEL_DUR:1,2,3:0.2\n", b"VEL_DUR:0,0,0:0.2\n"])
        self.assertEqual(transport.latest_encoder_sample["ticks"], [4,5,6])
        self.assertIsNotNone(transport.last_pause_event)
        self.assertFalse(transport.command_in_flight)

    def test_missing_encoder_array_times_out_and_bad_channel_count_is_rejected(self):
        from real_robot_infer import SerialAcknowledgmentTimeout
        transport, _, _ = self._make_transport(lines=["VEL_DUR command completed in 20 ms"], ack_timeout_seconds=.2)
        transport.configure_encoder_feedback(["01", "02", "03"], ["01", "02", "03"])
        with self.assertRaises(SerialAcknowledgmentTimeout):
            transport.send("VEL_DUR:1,2,3:0.2")
        transport, _, _ = self._make_transport(lines=["[1,2]", "VEL_DUR command completed in 20 ms"])
        transport.configure_encoder_feedback(["01", "02", "03"], ["01", "02", "03"])
        with self.assertRaisesRegex(ValueError, "channel"):
            transport.send("VEL_DUR:1,2,3:0.2")


class OffsetEstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = graph_test_cfg(graph_definition_file=str(ROOT / "config/tracker_maps/robot_def.json"),
            encoder_solver_tolerance_m=1e-4, encoder_solver_max_evaluations=60,
            encoder_allow_unverified_feedback=True, encoder_max_age_s=1, encoder_max_state_latency_s=2, encoder_max_step_m=.05,
            steamvr_to_policy_matrix=np.eye(3).tolist(),
            control_node_observation_source="connector_ball", tracker_variance_window=30,
            tracker_position_variance_floor_m2=1e-6, tracker_rotation_variance_floor_rad2=1e-6,
            tracker_initial_position_variance_m2=1e-4, tracker_initial_rotation_variance_rad2=1e-4,
            tracker_reacquire_after_s=1, tracker_startup_min_visible=2, tracker_startup_timeout_s=30)
        cls.geometry = build_encoder_geometry(cls.cfg)

    def setUp(self):
        self.geometry.solver.current = self.geometry.solver.reference.copy()

    def calibration(self):
        triangles = list(self.geometry.graph.triangle_nodes)
        return {"reference_shape_confirmed": True,
                "serial_node_order": list(self.geometry.graph.serial_node_order), "ticks_per_inch": 1125,
                "encoder_signs": [1]*8, "encoder_zero_ticks": [0]*8,
                "tracker_mounts": {key: {"triangle": triangle, "node": self.geometry.graph.triangle_nodes[triangle][0],
                     "position_m": [.03,.01,.02], "rotation_matrix": Rotation.from_euler("x", .2).as_matrix().tolist()}
                    for key, triangle in zip(["B13", "B16"], triangles[:2])}}

    def make_estimator(self):
        return EncoderViveEstimator(self.cfg, self.geometry, self.calibration(), np.eye(3))

    def poses(self, estimator, transform, rollers=None):
        rollers = self.geometry.solver.initial_rollers if rollers is None else rollers
        return {key: TrackerPose(pose.position, pose.rotation)
                for key, (order, mount) in estimator.mounts.items()
                for pose in [transform.compose(roller_frame(rollers[order]).compose(mount))]}

    def test_zero_encoders_preserve_generated_offset_geometry(self):
        rollers, balls, diagnostics = self.geometry.solver.solve(np.zeros(8))
        np.testing.assert_allclose(rollers, self.geometry.solver.initial_rollers, atol=1e-7)
        np.testing.assert_allclose(balls, self.geometry.solver.initial_balls, atol=1e-7)
        self.assertLess(diagnostics["max_constraint_error_m"], 1e-7)

    def test_incidence_matches_generated_controller_and_conserves_tubes(self):
        g = self.geometry
        channels = {name: i for i, name in enumerate(g.graph.serial_node_order)}
        nodes = {name:i for i,name in enumerate(g.graph.control_node_names)}
        for a,b in g.graph.actuator_edges:
            ei = next(i for i,pair in enumerate(g.solver.edges) if set(pair) == {nodes[a],nodes[b]})
            expected = np.zeros(8)
            if a in channels: expected[channels[a]] = -1
            if b in channels: expected[channels[b]] = 1
            np.testing.assert_equal(g.solver.incidence[ei], expected)
        np.testing.assert_allclose(g.solver.incidence.reshape(4,3,8).sum(axis=1), 0)

    def test_moves_shape_and_rejects_impossible_encoders(self):
        rollers, _, diagnostics = self.geometry.solver.solve(np.array([.002,-.001,0,0,0,0,0,0]))
        self.assertGreater(np.linalg.norm(rollers-self.geometry.solver.initial_rollers), .0001)
        self.assertLess(diagnostics["max_constraint_error_m"], 1e-4)
        with self.assertRaises(ValueError):
            self.geometry.solver.solve(np.ones(8)*100)

    def test_estimation_compensates_moving_mounts_and_continues_on_dropout(self):
        estimator = self.make_estimator()
        world = Pose([2,-1,.4], Rotation.from_euler("xyz", [.2,.4,-.3]).as_matrix())
        raw = self.poses(estimator, world)
        sample = {"ticks":[0]*8,"sequence":1,"received_at":0,"freshness_verified":False}
        positions, health = estimator.sample(sample, raw, {key:key for key in raw}, 0)
        expected = world.apply(self.geometry.solver.initial_balls[self.geometry.control_ball_indices])
        np.testing.assert_allclose(positions, expected, atol=1e-7)
        sample.update(sequence=2,received_at=.1,ticks=[20,0,0,0,0,0,0,0])
        # Reconstruct first, then generate exact full poses for the changed shape.
        displacement = np.array(sample["ticks"])*.0254/1125
        rollers, balls, _ = self.geometry.solver.solve(displacement)
        raw = self.poses(estimator, world, rollers)
        positions, health = estimator.sample(sample, raw, {key:key for key in raw}, .1)
        np.testing.assert_allclose(positions, world.apply(balls[self.geometry.control_ball_indices]), atol=1e-6)
        sample["received_at"] = .2
        positions, health = estimator.sample(sample, {}, {}, .2)
        self.assertEqual(health["status"], "held_global_transform")
        np.testing.assert_allclose(positions, world.apply(balls[self.geometry.control_ball_indices]), atol=1e-6)

    def test_calibration_and_stale_encoder_inputs_fail_before_motion(self):
        calibration = self.calibration()
        calibration["reference_shape_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "Confirm"):
            EncoderViveEstimator(self.cfg,self.geometry,calibration,np.eye(3))
        with self.assertRaisesRegex(ValueError,"stale"):
            self.make_estimator().sample({"received_at":0}, {}, {}, 2)

    def test_proper_coordinate_change_and_invalid_tracker_degrade_to_one_anchor(self):
        matrix = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        estimator = EncoderViveEstimator(self.cfg, self.geometry, self.calibration(), matrix)
        world = Pose([1,2,3], Rotation.from_euler("z", .3).as_matrix())
        poses = self.poses(estimator, world)
        poses = {key: TrackerPose(matrix.T @ pose.position, matrix.T @ pose.rotation_matrix) for key,pose in poses.items()}
        sample = {"ticks":[0]*8,"sequence":1,"received_at":0,"freshness_verified":False}
        positions, _ = estimator.sample(sample, poses, {key:key for key in poses}, 0)
        np.testing.assert_allclose(positions, world.apply(self.geometry.solver.initial_balls[self.geometry.control_ball_indices]), atol=1e-7)
        poses["B13"] = TrackerPose(np.full(3,np.nan),np.eye(3))
        sample["received_at"] = .1
        with redirect_stderr(io.StringIO()):
            positions, health = estimator.sample(sample, poses, {key:key for key in poses}, .1)
        self.assertIsNotNone(positions)
        self.assertEqual(health["status"],"one_tracker")
        self.assertIn("B13",health["rejected_trackers"])

    def test_faults_send_immediate_stop_and_are_logged(self):
        class Source:
            def read_encoder_frame(self,now):
                raise ValueError("bad sensor frame")
        class Transport:
            acknowledgment_gated=False
            def __init__(self): self.stops=[]
            def emergency_stop(self,command,**kwargs): self.stops.append(command)
        cfg=SimpleNamespace(control_frequency_hz=5,control_steps=1)
        layout=SimpleNamespace(node_names=["node"])
        formatter=SerialVelocityCommandFormatter(["node"],max_velocity_ticks_per_second=1800,duration_seconds=.2)
        transport=Transport()
        with tempfile.TemporaryDirectory() as tmp:
            recorder=RealRobotSessionRecorder(Path(tmp)/"session.jsonl",{})
            with self.assertRaisesRegex(ValueError,"bad sensor"):
                run_encoder_loop(cfg,Source(),layout,None,None,formatter,transport,None,OperatorControls(enabled=False),recorder)
            recorder.close(reason="error")
            records=read_jsonl(recorder.path)
        self.assertEqual(transport.stops,["VEL_DUR:0:0"])
        self.assertTrue(any(record["type"]=="fault" for record in records))
        self.assertEqual(sum(record["type"]=="emergency_command" for record in records),1)

    def test_preview_draws_exact_policy_positions_and_health_status(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from robot_preview import draw_frame
        positions=self.geometry.solver.initial_balls[self.geometry.control_ball_indices]
        fig=plt.figure(figsize=(7,5))
        try:
            ax=fig.add_subplot(111,projection="3d")
            draw_frame(ax,list(self.geometry.graph.control_node_names),self.geometry.solver.edges,
                       {"positions":positions,"paused":True,"health":{"status":"held_global_transform","global_pose_age_s":2,"encoder_freshness_verified":False}})
            self.assertIn("PAUSED",ax.get_title())
            self.assertIn("Global pose held",ax.get_title())
            self.assertIn("UNVERIFIED",ax.get_title())
            plotted=np.column_stack(ax.collections[0]._offsets3d)
            np.testing.assert_allclose(plotted,positions)
            with tempfile.TemporaryDirectory() as tmp:
                fig.savefig(Path(tmp)/"preview.png")
                self.assertGreater((Path(tmp)/"preview.png").stat().st_size,10000)
        finally:
            plt.close(fig)

    def test_cpu_loop_pause_keeps_sensor_logs_and_resumes_with_fresh_state(self):
        import torch
        estimator = self.make_estimator()
        layout = make_encoder_layout(self.cfg,self.geometry,{"B13":"B13","B16":"B16"})
        formatter = SerialVelocityCommandFormatter(layout.node_names,layout.action_mask,layout.serial_node_order,
                                                   max_velocity_ticks_per_second=1800,duration_seconds=.2)
        controls = OperatorControls(enabled=False)
        controls.request("toggle_pause")
        outer = self
        class Source:
            count=0
            def read_encoder_frame(self,now):
                self.count+=1
                if self.count==3: controls.request("toggle_pause")
                return outer.poses(estimator,Pose(np.zeros(3),np.eye(3))), {"ticks":[0]*8,"sequence":self.count,"received_at":now,"freshness_verified":False}
        class Agent:
            calls=0
            def act(self,*args,**kwargs):
                self.calls+=1
                return torch.ones((12,1))*.1
        class Transport:
            acknowledgment_gated=False
            def __init__(self): self.commands=[]; self.stops=[]
            def send(self,command): self.commands.append(command)
            def emergency_stop(self,command,**kwargs): self.stops.append(command)
        cfg = SimpleNamespace(control_frequency_hz=1000,control_steps=4,deterministic=True,
                              encoder_max_state_latency_s=2,tracker_startup_timeout_s=30)
        builder=RealRobotObservationBuilder(layout,False,1)
        source,agent,transport=Source(),Agent(),Transport()
        with tempfile.TemporaryDirectory() as tmp:
            recorder=RealRobotSessionRecorder(Path(tmp)/"session.jsonl",{
                "node_order":list(layout.node_names), "serial_node_order":list(formatter.serial_node_order),
                "max_velocity_ticks_per_second":1800, "serial_to_tracker_id":dict(layout.serial_to_tracker_id)})
            with redirect_stderr(io.StringIO()):
                reason=run_encoder_loop(cfg,source,layout,builder,agent,formatter,transport,estimator,controls,recorder)
            recorder.close(reason=reason)
            records=read_jsonl(recorder.path)
            from scripts.analyze_sim_to_real import _real_capture, tracker_health_metrics
            from replay_real_robot_commands import _load_commands
            report=tracker_health_metrics(_real_capture(recorder.path))
            self.assertEqual(report["state_estimation"]["paused_frames"],2)
            schedule,_,_=_load_commands(recorder.path,limit=1800)
            self.assertEqual(len(schedule),5)
            self.assertEqual(schedule[0].command.ticks,(0,)*8)
            self.assertEqual(schedule[1].command.ticks,(0,)*8)
            self.assertTrue(any(schedule[2].command.ticks))
        frames=[r for r in records if r["type"]=="tracker_frame"]
        self.assertEqual(len(frames),4)
        self.assertEqual(agent.calls,2)
        self.assertTrue(all("node_positions" in r for r in frames))
        self.assertEqual(transport.commands[:2],[formatter.velocity_command(np.zeros((12,1)))]*2)
        self.assertEqual(len(transport.stops),1)
        self.assertEqual(reason,"control_steps_complete")


if __name__ == "__main__":
    unittest.main()
