"""Calibrated adapter from Arduino encoders and two Vive poses to policy nodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from encoder_state import OffsetShapeSolver, Pose, TwoTrackerFusion, finite_array, roller_frame
from sim_to_real_io import canonical_json, load_triangle_graph_definition

if TYPE_CHECKING:
    from real_robot_infer import TrackerLayout


@dataclass
class EncoderGeometry:
    graph: object
    solver: OffsetShapeSolver
    triangle_indices: dict[str, list[int]]
    control_ball_indices: np.ndarray
    model_xml_sha256: str


def build_encoder_geometry(cfg: Any) -> EncoderGeometry:
    """Use the same model generator/configuration as replay, without stepping dynamics."""
    from mujoco_truss_gen import get_mujoco_spec, get_node_features, get_preset_definition, MujocoModel
    from env.mujoco_gen.topology_envs import _fixed_model_scale, _physical_parameters_from_config, resolve_truss_topology

    graph = load_triangle_graph_definition(str(cfg.graph_definition_file))
    node_dict, _ = get_preset_definition(resolve_truss_topology(cfg), scale=_fixed_model_scale(cfg))
    if set(node_dict) != set(graph.logical_node_names):
        raise ValueError("Graph node names must match the selected MuJoCo preset coordinates")
    spec = get_mujoco_spec(node_dict, graph.triangle_dict, realistic=True,
                           physical_params=_physical_parameters_from_config(cfg))
    model = MujocoModel(spec)
    metadata = model.control_graph
    if tuple(metadata.control_node_names) != graph.control_node_names:
        raise ValueError("Generated model node order does not match the robot graph")
    rollers = np.asarray(get_node_features(model, graph_view="control")[:, :3], dtype=float)
    ball_copies = np.asarray(get_node_features(model, graph_view="control", control_node_observation_source="connector_ball")[:, :3], dtype=float)
    logical_index = {name: i for i, name in enumerate(graph.logical_node_names)}
    ball_indices = np.array([logical_index[logical] for _, logical, _ in graph.control_node_occurrences])
    balls = np.array([ball_copies[np.flatnonzero(ball_indices == i)[0]] for i in range(len(logical_index))])
    index = {name: i for i, name in enumerate(graph.control_node_names)}
    triangles = {triangle: [index[graph.control_node_by_occurrence[triangle, node]] for node in nodes]
                 for triangle, nodes in graph.triangle_nodes.items()}
    if graph.serial_node_order is None:
        raise ValueError("Encoder reconstruction requires explicit roller numbers")
    channels = {name: i for i, name in enumerate(graph.serial_node_order)}
    incidence = np.zeros((3 * len(triangles), len(channels)))
    # Match the generator's directed actuator incidence, then conserve each tube.
    for ti, controls in enumerate(triangles.values()):
        tube_edges = [(controls[i], controls[(i + 1) % 3]) for i in range(3)]
        measured = set()
        for edge in metadata.actuator_edges:
            a, b = index[edge.from_node], index[edge.to_node]
            for ei, pair in enumerate(tube_edges):
                if set(pair) != {a, b}:
                    continue
                row = 3 * ti + ei
                if edge.from_node in channels:
                    incidence[row, channels[edge.from_node]] = -1
                if edge.to_node in channels:
                    incidence[row, channels[edge.to_node]] = 1
                measured.add(ei)
        if len(measured) != 2:
            raise ValueError("Each triangle must have exactly two independently actuated segments")
        missing = (set(range(3)) - measured).pop()
        incidence[3 * ti + missing] = -sum(incidence[3 * ti + ei] for ei in measured)
    solver = OffsetShapeSolver(rollers, balls, list(triangles.values()), ball_indices, incidence,
                              tolerance_m=float(cfg.encoder_solver_tolerance_m),
                              max_evaluations=int(cfg.encoder_solver_max_evaluations))
    return EncoderGeometry(graph, solver, triangles, ball_indices,
                           hashlib.sha256(spec.to_xml().encode()).hexdigest())


def make_encoder_layout(cfg: Any, geometry: EncoderGeometry, serial_map: dict[str, str]) -> TrackerLayout:
    """Reuse the exact graph/action contract without the old six-tracker constraint."""
    from real_robot_infer import TrackerLayout

    graph = geometry.graph
    indices = {name: i for i, name in enumerate(graph.control_node_names)}
    edges, roles = [], []
    for a, b, role in graph.edges:
        edges.extend(((indices[a], indices[b]), (indices[b], indices[a])))
        roles.extend([0 if role == "tube" else 1] * 2)
    coordinate = finite_array(cfg.steamvr_to_policy_matrix, (3, 3), "steamvr_to_policy_matrix")
    if not np.allclose(coordinate.T @ coordinate, np.eye(3), atol=1e-6) or np.linalg.det(coordinate) < 0:
        raise ValueError("Full-pose alignment requires a proper SteamVR-to-policy rotation (det +1)")
    return TrackerLayout(node_names=graph.control_node_names, serial_to_tracker_id=serial_map,
                         tracker_id_to_node={}, edge_index=np.asarray(edges, dtype=np.int64).T,
                         action_mask=np.array([name not in graph.passive_control_node_names for name in graph.control_node_names]),
                         edge_role=np.array(roles), steamvr_to_policy_matrix=coordinate,
                         assignment_mode="encoder_vive", serial_node_order=graph.serial_node_order,
                         graph_definition=graph.raw, graph_definition_sha256=graph.sha256,
                         routed_actuator_edges=graph.actuator_edges)


class EncoderViveEstimator:
    """Fuse the latest encoder sample with calibrated, shape-dependent tracker mounts."""

    def __init__(self, cfg: Any, geometry: EncoderGeometry, calibration: dict, coordinate_matrix: np.ndarray) -> None:
        self.cfg, self.geometry = cfg, geometry
        self.coordinate = np.asarray(coordinate_matrix)
        self.calibration = json.loads(json.dumps(calibration))
        for key in ("encoder_max_age_s", "encoder_max_step_m", "encoder_max_state_latency_s", "tracker_startup_timeout_s"):
            if not np.isfinite(float(getattr(cfg, key))) or float(getattr(cfg, key)) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        channels = len(geometry.graph.serial_node_order)
        expected = list(geometry.graph.serial_node_order)
        if calibration.get("serial_node_order") != expected:
            raise ValueError(f"Calibration serial_node_order must be {expected}")
        if calibration.get("reference_shape_confirmed") is not True:
            raise ValueError("Confirm the robot matches the MuJoCo reference shape at encoder_zero_ticks")
        self.signs = finite_array(calibration.get("encoder_signs"), (channels,), "encoder_signs")
        if not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("Each encoder sign must be -1 or +1")
        self.zero = finite_array(calibration.get("encoder_zero_ticks"), (channels,), "encoder_zero_ticks")
        scale = calibration.get("ticks_per_inch", 1125)
        self.ticks_per_inch = np.broadcast_to(np.asarray(scale, dtype=float), (channels,)).copy()
        if not np.isfinite(self.ticks_per_inch).all() or np.any(self.ticks_per_inch <= 0):
            raise ValueError("ticks_per_inch must be positive, scalar or one value per channel")
        mounts = calibration.get("tracker_mounts", {})
        if len(mounts) != 2:
            raise ValueError("Exactly two measured tracker mounts are required")
        self.mounts = {}
        for tracker_id, mount in mounts.items():
            triangle = str(mount["triangle"])
            if triangle not in geometry.triangle_indices:
                raise ValueError(f"Unknown tracker triangle: {triangle}")
            node = str(mount["node"])
            if node not in geometry.graph.triangle_nodes[triangle]:
                raise ValueError(f"Tracker {tracker_id} node must belong to its configured triangle")
            order = geometry.triangle_indices[triangle]
            anchor = geometry.graph.triangle_nodes[triangle].index(node)
            order = order[anchor:] + order[:anchor]
            self.mounts[tracker_id] = (order, Pose(mount["position_m"], mount["rotation_matrix"]))
        self.fusion = TwoTrackerFusion(self.mounts, window=int(cfg.tracker_variance_window),
            position_variance_floor=float(cfg.tracker_position_variance_floor_m2),
            rotation_variance_floor=float(cfg.tracker_rotation_variance_floor_rad2),
            initial_position_variance=float(cfg.tracker_initial_position_variance_m2),
            initial_rotation_variance=float(cfg.tracker_initial_rotation_variance_rad2),
            reacquire_after_s=float(cfg.tracker_reacquire_after_s))
        self.last_encoder_sequence = None
        self.last_rollers = None
        self.last_balls = None
        self.shape_diagnostics = {}
        self.previous_status = None
        self.ready = False
        self.last_diagnostics = None
        self.last_displacement = None
        self.last_ticks = None

    def sample(self, encoder_sample: dict, raw_poses: dict, serial_map: dict, now: float) -> tuple[np.ndarray | None, dict]:
        """Return raw poses, policy node positions, and additive health diagnostics."""
        if encoder_sample is None:
            raise ValueError("No encoder sample received")
        age = now - float(encoder_sample["received_at"])
        if not np.isfinite(age) or age < 0 or age > float(self.cfg.encoder_max_age_s):
            raise ValueError(f"Encoder receive age is invalid or stale: {age:g}s")
        verified = encoder_sample.get("freshness_verified") is True
        if not verified and not bool(self.cfg.encoder_allow_unverified_feedback):
            raise ValueError("Arduino cached encoder arrays have no verified per-channel freshness; see calibration guide")
        sequence = encoder_sample["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)) or sequence < 1:
            raise ValueError("Encoder sequence must be a positive host-array integer")
        if self.last_encoder_sequence is not None and sequence < self.last_encoder_sequence:
            raise ValueError("Encoder sequence moved backwards")
        ticks = finite_array(encoder_sample["ticks"], self.zero.shape, "encoder ticks")
        if not np.equal(ticks, np.floor(ticks)).all():
            raise ValueError("Encoder ticks must be integral")
        if sequence == self.last_encoder_sequence and not np.array_equal(ticks, self.last_ticks):
            raise ValueError("Encoder contents changed without a new host-array sequence")
        if sequence != self.last_encoder_sequence:
            displacement = (ticks - self.zero) * self.signs * 0.0254 / self.ticks_per_inch
            maximum_step = float(self.cfg.encoder_max_step_m)
            if not np.isfinite(maximum_step) or maximum_step <= 0:
                raise ValueError("encoder_max_step_m must be positive and finite")
            if self.last_displacement is not None and np.max(np.abs(displacement - self.last_displacement)) > maximum_step:
                raise ValueError("Encoder jump exceeds encoder_max_step_m; possible reset, rollover, or channel error")
            self.last_rollers, self.last_balls, self.shape_diagnostics = self.geometry.solver.solve(displacement)
            self.last_encoder_sequence = sequence
            self.last_displacement = displacement.copy()
            self.last_ticks = ticks.copy()
        candidates, rejected = {}, {}
        for serial, raw in raw_poses.items():
            tracker_id = serial_map.get(serial)
            if tracker_id not in self.mounts:
                continue
            try:
                observed = Pose(self.coordinate @ raw.position, self.coordinate @ raw.rotation_matrix)
                order, mount = self.mounts[tracker_id]
                local = roller_frame(self.last_rollers[order]).compose(mount)
                candidates[tracker_id] = observed.compose(local.inverse())
            except (ValueError, AttributeError) as exc:
                rejected[tracker_id] = str(exc)
        transform, health = self.fusion.update(candidates, now)
        if not self.ready:
            required = int(self.cfg.tracker_startup_min_visible)
            if required not in (1, 2):
                raise ValueError("tracker_startup_min_visible must be 1 or 2")
            self.ready = len(candidates) >= required
        health.update({"encoder_ticks": ticks.tolist(), "encoder_receive_age_s": age,
                       "encoder_freshness_verified": verified, "encoder_sequence": sequence,
                       "shape": self.shape_diagnostics, "rejected_trackers": rejected,
                       "startup_ready": self.ready})
        health["raw_firmware_ticks"] = encoder_sample.get("raw_firmware_ticks")
        if not verified:
            health["encoder_warning"] = "Firmware may return cached encoder values; age is receive age only"
        status_key = (health["status"], tuple(health["missing_trackers"]))
        if status_key != self.previous_status:
            if health["warning"]:
                print(f"WARNING: {health['warning']}; {health['status']}", file=sys.stderr, flush=True)
            elif self.previous_status is not None:
                print("Tracker pose anchors recovered.", file=sys.stderr, flush=True)
            self.previous_status = status_key
        self.last_diagnostics = health
        if not self.ready or transform is None:
            return None, health
        health["world_from_shape"] = {"position": transform.position.tolist(), "rotation_matrix": transform.rotation.tolist()}
        source = str(self.cfg.control_node_observation_source)
        if source == "connector_ball":
            local_nodes = self.last_balls[self.geometry.control_ball_indices]
        elif source == "physical_node":
            local_nodes = self.last_rollers
        else:
            raise ValueError("Unknown control_node_observation_source")
        return transform.apply(local_nodes), health

    def metadata(self) -> dict:
        return {"estimator": "encoder_offset_two_vive", "calibration": self.calibration,
                "calibration_sha256": hashlib.sha256(canonical_json(self.calibration).encode()).hexdigest(),
                "model_xml_sha256": self.geometry.model_xml_sha256,
                "shape_model": "spatial_triangle_bisectors_fixed_connector_rods",
                "reference_offset_components_m": self.geometry.solver.offset_components.tolist(),
                "settings": {key: getattr(self.cfg, key) for key in (
                    "tracker_variance_window", "tracker_position_variance_floor_m2",
                    "tracker_rotation_variance_floor_rad2", "tracker_initial_position_variance_m2",
                    "tracker_initial_rotation_variance_rad2", "tracker_reacquire_after_s",
                    "tracker_startup_min_visible", "encoder_solver_tolerance_m",
                    "encoder_solver_max_evaluations", "encoder_max_age_s", "encoder_max_step_m",
                    "encoder_max_state_latency_s", "encoder_allow_unverified_feedback")},
                "firmware_receiver_order": list(self.cfg.firmware_receiver_order),
                "variance_model": "rolling_second_difference_jitter",
                "both_trackers_missing": "hold_last_global_transform"}


def load_calibration(path: str | Path) -> dict:
    value = json.loads(Path(path).expanduser().read_text())
    if not isinstance(value, dict):
        raise ValueError("Encoder calibration must be a JSON object")
    return value
