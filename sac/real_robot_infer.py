"""Closed-loop GNN inference using SteamVR trackers instead of MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
import torch
from omegaconf import open_dict
from torch_geometric.data import Data

from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env
from gnn_infer import load_agent_checkpoint, resolve_checkpoint
from gnn_sac import GNNSAC


class TrackerSource(Protocol):
    def positions_by_serial(self) -> Mapping[str, np.ndarray]: ...
    def close(self) -> None: ...


class MissingTrackerFrame(RuntimeError):
    """A transient frame that does not contain every configured tracker."""


def _critical_rigidity(positions: np.ndarray, edge_index: np.ndarray) -> float:
    """Return the first mode after the six free-body modes of R^T R."""
    node_count = positions.shape[0]
    undirected = {
        tuple(sorted((int(source), int(target))))
        for source, target in edge_index.T
        if source != target
    }
    rigidity_matrix = np.zeros((len(undirected), 3 * node_count), dtype=np.float64)
    for row, (source, target) in enumerate(sorted(undirected)):
        delta = positions[source] - positions[target]
        length = np.linalg.norm(delta)
        if length <= 1e-8:
            return 0.0
        direction = delta / length
        rigidity_matrix[row, 3 * source : 3 * source + 3] = direction
        rigidity_matrix[row, 3 * target : 3 * target + 3] = -direction
    eigenvalues = np.linalg.eigvalsh(rigidity_matrix.T @ rigidity_matrix)
    return float(max(eigenvalues[6], 0.0)) if eigenvalues.size > 6 else 0.0


class SteamVRTrackerSource:
    """Read valid tracker translations from OpenVR's standing coordinate frame."""

    def __init__(self):
        try:
            import openvr
        except ImportError as exc:
            raise RuntimeError(
                "SteamVR input requires the optional 'openvr' package: pip install openvr"
            ) from exc
        self.openvr = openvr
        openvr.init(openvr.VRApplication_Background)
        self.system = openvr.VRSystem()

    def positions_by_serial(self) -> dict[str, np.ndarray]:
        poses = self.system.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding,
            0.0,
            self.openvr.k_unMaxTrackedDeviceCount,
        )
        positions = {}
        for device_index, pose in enumerate(poses):
            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                continue
            device_class = self.system.getTrackedDeviceClass(device_index)
            if device_class != self.openvr.TrackedDeviceClass_GenericTracker:
                continue
            serial = self.system.getStringTrackedDeviceProperty(
                device_index, self.openvr.Prop_SerialNumber_String
            )
            matrix = pose.mDeviceToAbsoluteTracking
            positions[str(serial)] = np.asarray(
                [matrix[0][3], matrix[1][3], matrix[2][3]], dtype=np.float64
            )
        return positions

    def close(self) -> None:
        self.openvr.shutdown()


def _load_json(path_value: str) -> dict:
    path = Path(path_value).expanduser()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


@dataclass
class TrackerLayout:
    node_names: tuple[str, ...]
    serial_to_tracker_id: dict[str, str]
    tracker_id_to_node: dict[str, str]
    edge_index: np.ndarray
    action_mask: np.ndarray
    edge_role: np.ndarray | None
    steamvr_to_policy_matrix: np.ndarray
    reference_positions: np.ndarray | None = None
    assignment_mode: str = "manual"

    @classmethod
    def from_files(cls, serial_map_file: str, layout_file: str) -> "TrackerLayout":
        serial_data = _load_json(serial_map_file)
        serial_map = serial_data.get("serial_to_tracker_id", serial_data)
        layout = _load_json(layout_file)
        node_names = tuple(str(value) for value in layout["node_names"])
        tracker_map = {str(k): str(v) for k, v in layout["tracker_id_to_node"].items()}
        if len(set(node_names)) != len(node_names):
            raise ValueError("node_names must be unique")
        unknown_nodes = set(tracker_map.values()) - set(node_names)
        if unknown_nodes:
            raise ValueError(f"Tracker map refers to unknown nodes: {sorted(unknown_nodes)}")
        missing_nodes = set(node_names) - set(tracker_map.values())
        if missing_nodes:
            raise ValueError(f"Every node needs one tracker; missing: {sorted(missing_nodes)}")
        if len(set(tracker_map.values())) != len(tracker_map):
            raise ValueError("Only one tracker may map to each node")

        node_index = {name: idx for idx, name in enumerate(node_names)}
        directed_edges, directed_roles = [], []
        role_names = {"tube": 0, "connector": 1}
        for edge in layout["edges"]:
            source, target = str(edge[0]), str(edge[1])
            if source not in node_index or target not in node_index:
                raise ValueError(f"Edge refers to an unknown node: {edge}")
            role = role_names[str(edge[2])] if len(edge) > 2 else None
            directed_edges.extend(((node_index[source], node_index[target]), (node_index[target], node_index[source])))
            if role is not None:
                directed_roles.extend((role, role))
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        actuated = set(str(value) for value in layout.get("actuated_nodes", node_names))
        if not actuated <= set(node_names):
            raise ValueError(f"actuated_nodes contains unknown nodes: {sorted(actuated - set(node_names))}")
        edge_role = np.asarray(directed_roles, dtype=np.int64) if directed_roles else None
        if edge_role is not None and edge_role.size != edge_index.shape[1]:
            raise ValueError("Either all edges or no edges must specify a role")
        coordinate_matrix = np.asarray(
            layout.get("steamvr_to_policy_matrix", np.eye(3)), dtype=np.float64
        )
        if coordinate_matrix.shape != (3, 3) or not np.isfinite(coordinate_matrix).all():
            raise ValueError("steamvr_to_policy_matrix must be a finite 3x3 matrix")
        if abs(np.linalg.det(coordinate_matrix)) <= 1e-8:
            raise ValueError("steamvr_to_policy_matrix must be invertible")
        return cls(
            node_names=node_names,
            serial_to_tracker_id={str(k): str(v) for k, v in serial_map.items()},
            tracker_id_to_node=tracker_map,
            edge_index=edge_index,
            action_mask=np.asarray([name in actuated for name in node_names], dtype=bool),
            edge_role=edge_role,
            steamvr_to_policy_matrix=coordinate_matrix,
        )

    @classmethod
    def from_preset(
        cls,
        cfg,
        serial_map_file: str,
        tracker_layout_file: str | None,
    ) -> "TrackerLayout":
        """Load graph metadata and nominal geometry from a generated preset."""
        serial_data = _load_json(serial_map_file)
        serial_map = serial_data.get("serial_to_tracker_id", serial_data)
        layout_data = _load_json(tracker_layout_file) if tracker_layout_file else {}
        assignment_mode = str(getattr(cfg, "tracker_assignment", "automatic")).lower()
        if assignment_mode not in {"automatic", "manual"}:
            raise ValueError("tracker_assignment must be 'automatic' or 'manual'")
        tracker_map = {
            str(k): str(v)
            for k, v in layout_data.get("tracker_id_to_node", {}).items()
        }
        coordinate_matrix = np.asarray(
            layout_data.get(
                "steamvr_to_policy_matrix",
                getattr(cfg, "steamvr_to_policy_matrix", np.eye(3)),
            ),
            dtype=np.float64,
        )
        if coordinate_matrix.shape != (3, 3) or not np.isfinite(coordinate_matrix).all():
            raise ValueError("steamvr_to_policy_matrix must be a finite 3x3 matrix")
        if abs(np.linalg.det(coordinate_matrix)) <= 1e-8:
            raise ValueError("steamvr_to_policy_matrix must be invertible")

        env = make_env(cfg)
        try:
            observation = env.reset()
            preset_env = env.unwrapped
            node_names = tuple(str(name) for name in preset_env.graph_node_names)
            from mujoco_truss_gen import get_node_features

            reference_positions = np.asarray(
                get_node_features(
                    preset_env.mj_model,
                    graph_view=preset_env._graph_view(),
                    aggregation="connector_ball",
                )[:, :3],
                dtype=np.float64,
            )
            edge_index = observation.edge_index.detach().cpu().numpy().astype(np.int64)
            edge_role = (
                observation.edge_role.detach().cpu().numpy().astype(np.int64)
                if hasattr(observation, "edge_role")
                else None
            )
            result = cls(
                node_names=node_names,
                serial_to_tracker_id={str(k): str(v) for k, v in serial_map.items()},
                tracker_id_to_node=tracker_map,
                edge_index=edge_index,
                action_mask=observation.action_mask.detach().cpu().numpy().astype(bool),
                edge_role=edge_role,
                steamvr_to_policy_matrix=coordinate_matrix,
                reference_positions=reference_positions,
                assignment_mode=assignment_mode,
            )
            result._validate_tracker_map()
            return result
        finally:
            env.close()

    def _validate_tracker_map(self) -> None:
        if self.assignment_mode == "automatic":
            return
        unknown_trackers = set(self.tracker_id_to_node) - set(self.serial_to_tracker_id.values())
        if unknown_trackers:
            raise ValueError(
                "Manual layout refers to tracker IDs absent from the serial map: "
                f"{sorted(unknown_trackers)}"
            )
        unknown_nodes = set(self.tracker_id_to_node.values()) - set(self.node_names)
        missing_nodes = set(self.node_names) - set(self.tracker_id_to_node.values())
        if unknown_nodes:
            raise ValueError(f"Tracker map refers to unknown preset nodes: {sorted(unknown_nodes)}")
        if missing_nodes:
            raise ValueError(f"Every preset node needs one tracker; missing: {sorted(missing_nodes)}")
        if len(set(self.tracker_id_to_node.values())) != len(self.tracker_id_to_node):
            raise ValueError("Only one tracker may map to each node")

    @staticmethod
    def _minimum_cost_assignment(cost: np.ndarray) -> np.ndarray:
        """Assign each row to a unique column exactly with dynamic programming."""
        row_count, column_count = cost.shape
        if row_count > column_count:
            raise ValueError("Assignment requires at least as many columns as rows")
        states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
        for row in range(row_count):
            next_states = {}
            for mask, (total, columns) in states.items():
                for column in range(column_count):
                    bit = 1 << column
                    if mask & bit:
                        continue
                    candidate = (total + float(cost[row, column]), columns + (column,))
                    new_mask = mask | bit
                    if new_mask not in next_states or candidate[0] < next_states[new_mask][0]:
                        next_states[new_mask] = candidate
            states = next_states
        return np.asarray(min(states.values(), key=lambda value: value[0])[1], dtype=np.int64)

    def _automatic_tracker_map(self, positions_by_serial: Mapping[str, np.ndarray]) -> dict[str, str]:
        if self.reference_positions is None:
            raise ValueError("Automatic tracker assignment requires preset reference positions")
        tracker_ids, tracker_positions = [], []
        for serial, tracker_id in self.serial_to_tracker_id.items():
            if serial in positions_by_serial:
                tracker_ids.append(tracker_id)
                tracker_positions.append(np.asarray(positions_by_serial[serial], dtype=np.float64))
        if len(tracker_positions) < len(self.node_names):
            raise MissingTrackerFrame(
                f"Automatic assignment needs at least {len(self.node_names)} valid configured trackers; "
                f"received {len(tracker_positions)}"
            )
        measured = np.stack(tracker_positions) @ self.steamvr_to_policy_matrix.T
        measured_centered = measured - measured.mean(axis=0, keepdims=True)
        reference_centered = self.reference_positions - self.reference_positions.mean(
            axis=0, keepdims=True
        )
        cost = np.linalg.norm(
            reference_centered[:, None, :] - measured_centered[None, :, :], axis=-1
        )
        assignment = self._minimum_cost_assignment(cost)
        mapping = {
            tracker_ids[tracker_index]: node_name
            for node_name, tracker_index in zip(self.node_names, assignment)
        }
        print(
            "Automatic tracker assignment: "
            + json.dumps(mapping, sort_keys=True)
            + f" (RMS error={np.sqrt(np.mean(cost[np.arange(len(assignment)), assignment] ** 2)):.6f} m)",
            flush=True,
        )
        return mapping

    def ordered_positions(self, positions_by_serial: Mapping[str, np.ndarray]) -> np.ndarray:
        tracker_map = self.tracker_id_to_node
        if self.assignment_mode == "automatic" and not tracker_map:
            tracker_map = self._automatic_tracker_map(positions_by_serial)
            self.tracker_id_to_node = tracker_map
        by_node = {}
        for serial, position in positions_by_serial.items():
            tracker_id = self.serial_to_tracker_id.get(str(serial))
            node_name = tracker_map.get(tracker_id) if tracker_id is not None else None
            if node_name is not None:
                by_node[node_name] = np.asarray(position, dtype=np.float64)
        missing = [name for name in self.node_names if name not in by_node]
        if missing:
            raise MissingTrackerFrame(f"Missing valid SteamVR poses for nodes: {missing}")
        positions = np.stack([by_node[name] for name in self.node_names])
        return positions @ self.steamvr_to_policy_matrix.T


class RealRobotObservationBuilder:
    """Build simulation-compatible graph observations from tracker positions."""

    def __init__(
        self,
        layout: TrackerLayout,
        normalize_observations: bool,
        velocity_filter_alpha: float,
        com_relative_axes=(True, True, False),
        initial_bbox_dimensions: np.ndarray | None = None,
        initial_rigidity: float | None = None,
    ):
        if not 0.0 < velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        self.layout = layout
        self.normalize_observations = normalize_observations
        self.velocity_filter_alpha = velocity_filter_alpha
        self.com_relative_axes = np.asarray(com_relative_axes, dtype=bool)
        if self.com_relative_axes.shape != (3,):
            raise ValueError("com_relative_axes must contain three booleans")
        self.initial_bbox_dimensions = (
            None
            if initial_bbox_dimensions is None
            else np.asarray(initial_bbox_dimensions, dtype=np.float64).copy()
        )
        self.previous_positions: np.ndarray | None = None
        self.previous_time: float | None = None
        self.filtered_velocity: np.ndarray | None = None
        self.initial_rigidity = initial_rigidity

    def _critical_rigidity(self, positions: np.ndarray) -> float:
        return _critical_rigidity(positions, self.layout.edge_index)

    def build(self, positions: np.ndarray, timestamp: float) -> Data:
        positions = np.asarray(positions, dtype=np.float64)
        expected = (len(self.layout.node_names), 3)
        if positions.shape != expected or not np.isfinite(positions).all():
            raise ValueError(f"Expected finite tracker positions with shape {expected}")
        if self.initial_bbox_dimensions is None:
            dimensions = np.ptp(positions, axis=0)
            if np.any(dimensions <= 1e-8):
                raise ValueError(f"Initial bounding-box dimensions must be nonzero; got {dimensions.tolist()}")
            self.initial_bbox_dimensions = dimensions
            self.initial_rigidity = max(self._critical_rigidity(positions), 1e-8)
        elif np.any(self.initial_bbox_dimensions <= 1e-8):
            raise ValueError(
                "Preset initial bounding-box dimensions must all be nonzero; "
                f"got {self.initial_bbox_dimensions.tolist()}"
            )
        if self.initial_rigidity is None:
            self.initial_rigidity = max(self._critical_rigidity(positions), 1e-8)

        velocity = np.zeros_like(positions)
        if self.previous_positions is not None and self.previous_time is not None:
            dt = timestamp - self.previous_time
            if dt <= 0.0:
                raise ValueError("Tracker timestamps must increase")
            measured = (positions - self.previous_positions) / dt
            previous = self.filtered_velocity if self.filtered_velocity is not None else measured
            alpha = self.velocity_filter_alpha
            velocity = alpha * measured + (1.0 - alpha) * previous
        self.previous_positions = positions.copy()
        self.previous_time = float(timestamp)
        self.filtered_velocity = velocity.copy()

        relative = positions.copy()
        relative[:, self.com_relative_axes] -= positions[:, self.com_relative_axes].mean(
            axis=0, keepdims=True
        )
        if self.normalize_observations:
            relative = relative / self.initial_bbox_dimensions
            velocity = velocity / self.initial_bbox_dimensions
        graph = Data(
            x=torch.from_numpy(np.concatenate((relative, velocity), axis=1).astype(np.float32)),
            edge_index=torch.from_numpy(self.layout.edge_index.copy()).long(),
            action_mask=torch.from_numpy(self.layout.action_mask.copy()).bool(),
            rigidity=torch.tensor(
                [self._critical_rigidity(positions) / self.initial_rigidity],
                dtype=torch.float32,
            ),
        )
        if self.layout.edge_role is not None:
            graph.edge_role = torch.from_numpy(self.layout.edge_role.copy()).long()
        return graph


def run_real_robot_inference(cfg, source: TrackerSource | None = None) -> None:
    with open_dict(cfg):
        cfg.enable_wandb = False
        cfg.save_agent = False
        cfg.save_csv = False
        cfg.mujoco_backend = "mujoco"  # Keep returned actions on CPU; no simulator is constructed.
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    layout_file = getattr(cfg, "tracker_layout_file", None)
    if layout_file in (None, "", "null"):
        layout_file = None
    layout = TrackerLayout.from_preset(
        cfg,
        str(cfg.serial_map_file),
        str(layout_file) if layout_file is not None else None,
    )
    builder = RealRobotObservationBuilder(
        layout,
        normalize_observations=bool(cfg.normalize_observations),
        velocity_filter_alpha=float(cfg.velocity_filter_alpha),
        com_relative_axes=tuple(bool(value) for value in cfg.com_relative_axes),
    )
    expected_actions = int(layout.action_mask.sum())
    if int(cfg.num_policy_actions) != expected_actions:
        raise ValueError(
            f"Configuration expects {cfg.num_policy_actions} policy actions but the tracker "
            f"layout marks {expected_actions} nodes as actuated."
        )
    checkpoint = resolve_checkpoint(str(cfg.model), cfg)
    agent = GNNSAC(cfg)
    load_agent_checkpoint(agent, checkpoint)
    agent.model.eval()
    source = source or SteamVRTrackerSource()
    frequency_hz = float(cfg.control_frequency_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("control_frequency_hz must be positive and finite")
    period = 1.0 / frequency_hz
    next_tick = time.monotonic()
    step = 0
    try:
        while cfg.control_steps is None or step < int(cfg.control_steps):
            now = time.monotonic()
            try:
                positions = layout.ordered_positions(source.positions_by_serial())
                observation = builder.build(positions, now)
                action = agent.act(observation, t0=step == 0, eval_mode=bool(cfg.deterministic))
                command = action.detach().cpu().numpy() * float(cfg.speed)
                print(json.dumps({
                    "step": step,
                    "timestamp": now,
                    "velocity_command_by_node": {
                        name: np.asarray(value).reshape(-1).astype(float).tolist()
                        for name, value in zip(layout.node_names, command)
                    },
                }), flush=True)
                step += 1
            except MissingTrackerFrame as exc:
                print(f"tracker frame skipped: {exc}", file=sys.stderr, flush=True)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        source.close()


@hydra.main(config_name="inference/real_robot", config_path="../config", version_base=None)
def infer(cfg):
    run_real_robot_inference(cfg)


if __name__ == "__main__":
    infer()
