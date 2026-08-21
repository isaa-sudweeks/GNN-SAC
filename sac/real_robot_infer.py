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


@dataclass(frozen=True)
class TrackerPose:
    """One tracker pose, expressed in the SteamVR standing frame."""

    position: np.ndarray
    rotation_matrix: np.ndarray


class TrackerSource(Protocol):
    def positions_by_serial(self) -> Mapping[str, np.ndarray]: ...
    def poses_by_serial(self) -> Mapping[str, TrackerPose]: ...
    def close(self) -> None: ...


class MissingTrackerFrame(RuntimeError):
    """A transient frame that does not contain every configured tracker."""


class SerialVelocityCommandFormatter:
    """Format commands accepted by the TrussRobotFirmware USB transmitter."""

    def __init__(
        self,
        graph_node_names,
        action_mask=None,
        serial_node_order=None,
        *,
        max_velocity_ticks_per_second: int,
        duration_seconds: float,
    ):
        self.graph_node_names = tuple(str(name) for name in graph_node_names)
        if action_mask is None:
            action_mask = np.ones(len(self.graph_node_names), dtype=bool)
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (len(self.graph_node_names),):
            raise ValueError("action_mask must contain one entry per preset graph node")
        self.command_node_names = tuple(
            name for name, active in zip(self.graph_node_names, action_mask) if active
        )
        self.serial_node_order = (
            self.command_node_names
            if serial_node_order is None
            else tuple(str(name) for name in serial_node_order)
        )
        if (
            len(self.serial_node_order) != len(self.command_node_names)
            or set(self.serial_node_order) != set(self.command_node_names)
        ):
            raise ValueError(
                "serial_node_order must contain every actuated preset node exactly once"
            )
        self._graph_index = {
            name: index for index, name in enumerate(self.graph_node_names)
        }
        self.max_velocity_ticks_per_second = int(max_velocity_ticks_per_second)
        self.duration_seconds = float(duration_seconds)
        if self.max_velocity_ticks_per_second <= 0:
            raise ValueError("serial_max_velocity_ticks_per_second must be positive")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("serial_command_duration_s must be positive and finite")

    def velocity_command(self, normalized_node_velocity: np.ndarray) -> str:
        normalized = np.asarray(normalized_node_velocity, dtype=np.float64)
        if normalized.shape == (len(self.graph_node_names), 1):
            normalized = normalized[:, 0]
        if normalized.shape != (len(self.graph_node_names),):
            raise ValueError(
                "Serial firmware supports one scalar velocity per graph node; "
                f"received shape {normalized.shape}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError("Serial velocity commands must be finite")
        limit = self.max_velocity_ticks_per_second
        ticks = np.rint(np.clip(normalized, -1.0, 1.0) * limit).astype(np.int64)
        ordered = [int(ticks[self._graph_index[name]]) for name in self.serial_node_order]
        duration = f"{self.duration_seconds:g}"
        return f"VEL_DUR:{','.join(str(value) for value in ordered)}:{duration}"

    def emergency_stop_command(self) -> str:
        zeros = ",".join("0" for _ in self.serial_node_order)
        return f"VEL_DUR:{zeros}:0"


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
    """Read valid tracker poses from OpenVR's standing coordinate frame."""

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

    def poses_by_serial(self) -> dict[str, TrackerPose]:
        poses = self.system.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding,
            0.0,
            self.openvr.k_unMaxTrackedDeviceCount,
        )
        tracker_poses = {}
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
            tracker_poses[str(serial)] = TrackerPose(
                position=np.asarray(
                    [matrix[0][3], matrix[1][3], matrix[2][3]], dtype=np.float64
                ),
                rotation_matrix=np.asarray(
                    [[matrix[row][column] for column in range(3)] for row in range(3)],
                    dtype=np.float64,
                ),
            )
        return tracker_poses

    def positions_by_serial(self) -> dict[str, np.ndarray]:
        """Retain the translation-only interface used by direct tracker layouts."""
        return {
            serial: pose.position
            for serial, pose in self.poses_by_serial().items()
        }

    def close(self) -> None:
        self.openvr.shutdown()


def _load_json(path_value: str) -> dict:
    path = Path(path_value).expanduser()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


@dataclass(frozen=True)
class TrackerMount:
    """Calibration needed to recover one abstract joint from triangle planes."""

    tracker_id: str
    triangle: str
    abstract_node: str
    joint_triangles: tuple[str, str]
    local_plane_normal: np.ndarray
    local_plane_point_offset: np.ndarray


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
    tracker_mounts: dict[str, TrackerMount] | None = None
    plane_parallel_tolerance: float = 1e-6

    @staticmethod
    def _expand_triangle_definition(layout: Mapping[str, object]) -> dict:
        """Generate a MuJoCo-style control graph from logical triangles."""
        raw_triangles = layout.get("triangles")
        if not isinstance(raw_triangles, list) or not raw_triangles:
            raise ValueError("triangles must be a nonempty list")
        conflicting = {"node_names", "edges", "actuated_nodes", "tracker_mounts"} & set(
            layout
        )
        if conflicting:
            raise ValueError(
                "Triangle definitions generate graph fields automatically; remove: "
                f"{sorted(conflicting)}"
            )

        triangle_nodes: dict[str, tuple[str, str, str]] = {}
        passive_nodes: dict[str, str] = {}
        tracker_assignments: list[tuple[str, str, str]] = []
        tracker_ids = set()
        tracked_nodes = set()
        calibration = layout.get("tracker_calibration", {})
        if not isinstance(calibration, dict):
            raise ValueError("tracker_calibration must be an object")

        for index, raw_triangle in enumerate(raw_triangles):
            if not isinstance(raw_triangle, dict):
                raise ValueError(f"triangles[{index}] must be an object")
            missing = {"name", "nodes", "passive_node"} - set(raw_triangle)
            if missing:
                raise ValueError(
                    f"triangles[{index}] is missing fields: {sorted(missing)}"
                )
            name = str(raw_triangle["name"])
            if name in triangle_nodes:
                raise ValueError(f"Triangle names must be unique; duplicate: {name!r}")
            raw_nodes = raw_triangle["nodes"]
            if not isinstance(raw_nodes, list) or len(raw_nodes) != 3:
                raise ValueError(f"Triangle {name!r} must contain exactly three nodes")
            nodes = tuple(str(node) for node in raw_nodes)
            if len(set(nodes)) != 3:
                raise ValueError(f"Triangle {name!r} nodes must be unique")
            passive_node = str(raw_triangle["passive_node"])
            if passive_node not in nodes:
                raise ValueError(
                    f"Triangle {name!r} passive_node must be one of its nodes"
                )
            triangle_nodes[name] = nodes
            passive_nodes[name] = passive_node

            raw_trackers = raw_triangle.get(
                "trackers",
                raw_triangle.get("tracker_nodes", raw_triangle.get("tracker", [])),
            )
            if isinstance(raw_trackers, str):
                raw_trackers = [raw_trackers]
            if isinstance(raw_trackers, dict):
                tracker_items = [
                    (str(node), str(tracker_id))
                    for node, tracker_id in raw_trackers.items()
                ]
            elif isinstance(raw_trackers, list):
                tracker_items = []
                for tracker_index, raw_tracker in enumerate(raw_trackers):
                    if isinstance(raw_tracker, str):
                        tracker_items.append((raw_tracker, raw_tracker))
                    elif isinstance(raw_tracker, dict) and "node" in raw_tracker:
                        node = str(raw_tracker["node"])
                        tracker_id = str(
                            raw_tracker.get("id", raw_tracker.get("tracker_id", node))
                        )
                        tracker_items.append((node, tracker_id))
                    else:
                        raise ValueError(
                            f"Triangle {name!r} tracker {tracker_index} must be a node "
                            "name or an object containing node and optional id"
                        )
            else:
                raise ValueError(
                    f"Triangle {name!r} trackers must be a node-to-tracker object or list"
                )
            for logical_node, tracker_id in tracker_items:
                if logical_node not in nodes:
                    raise ValueError(
                        f"Tracker {tracker_id!r} is mounted at {logical_node!r}, which "
                        f"is not a node of triangle {name!r}"
                    )
                if tracker_id in tracker_ids:
                    raise ValueError(f"Tracker IDs must be unique; duplicate: {tracker_id!r}")
                if logical_node in tracked_nodes:
                    raise ValueError(
                        f"Logical node {logical_node!r} has more than one tracker"
                    )
                tracker_ids.add(tracker_id)
                tracked_nodes.add(logical_node)
                tracker_assignments.append((tracker_id, logical_node, name))

        node_incidence: dict[str, list[str]] = {}
        for triangle_name, nodes in triangle_nodes.items():
            for logical_node in nodes:
                node_incidence.setdefault(logical_node, []).append(triangle_name)
        missing_trackers = set(node_incidence) - tracked_nodes
        if missing_trackers:
            raise ValueError(
                "Every logical node needs exactly one tracker; missing: "
                f"{sorted(missing_trackers)}"
            )
        unknown_calibrations = set(calibration) - tracker_ids - tracked_nodes
        if unknown_calibrations:
            raise ValueError(
                "tracker_calibration contains unknown tracker or node IDs: "
                f"{sorted(unknown_calibrations)}"
            )
        untracked_triangles = set(triangle_nodes) - {
            triangle for _, _, triangle in tracker_assignments
        }
        if untracked_triangles:
            raise ValueError(
                "Every triangle plane needs at least one tracker; missing: "
                f"{sorted(untracked_triangles)}"
            )
        ambiguous_nodes = {
            node: triangles
            for node, triangles in node_incidence.items()
            if len(triangles) != 2
        }
        if ambiguous_nodes:
            details = ", ".join(
                f"{node}={triangles}" for node, triangles in sorted(ambiguous_nodes.items())
            )
            raise ValueError(
                "Triangle-plane reconstruction currently requires every logical node "
                f"to belong to exactly two triangles; got {details}"
            )

        control_node_names = []
        control_triangles: dict[str, list[str]] = {}
        control_by_logical: dict[str, list[str]] = {}
        owned_nodes = set()
        for triangle_name, nodes in triangle_nodes.items():
            control_nodes = []
            for logical_node in nodes:
                control_node = (
                    logical_node
                    if logical_node not in owned_nodes
                    else f"{logical_node}_tri_{triangle_name}"
                )
                owned_nodes.add(logical_node)
                control_node_names.append(control_node)
                control_nodes.append(control_node)
                control_by_logical.setdefault(logical_node, []).append(control_node)
            control_triangles[triangle_name] = control_nodes

        edges = []
        passive_control_nodes = set()
        for triangle_name, nodes in triangle_nodes.items():
            controls = control_triangles[triangle_name]
            for edge_index in range(3):
                edges.append(
                    [controls[edge_index], controls[(edge_index + 1) % 3], "tube"]
                )
            passive_index = nodes.index(passive_nodes[triangle_name])
            passive_control_nodes.add(controls[passive_index])
        for controls in control_by_logical.values():
            for index, source in enumerate(controls):
                for target in controls[index + 1 :]:
                    edges.append([source, target, "connector"])

        tracker_mounts = {}
        for tracker_id, logical_node, triangle_name in tracker_assignments:
            raw_calibration = calibration.get(
                tracker_id, calibration.get(logical_node, {})
            )
            if not isinstance(raw_calibration, dict):
                raise ValueError(
                    f"tracker_calibration[{tracker_id!r}] must be an object"
                )
            tracker_mounts[tracker_id] = {
                "triangle": triangle_name,
                "abstract_node": logical_node,
                "joint_triangles": node_incidence[logical_node],
                "local_plane_normal": raw_calibration.get(
                    "local_plane_normal", [0.0, 0.0, 1.0]
                ),
                "local_plane_point_offset": raw_calibration.get(
                    "local_plane_point_offset", [0.0, 0.0, 0.0]
                ),
            }

        expanded = dict(layout)
        expanded.update(
            tracker_assignment="triangle_planes",
            node_names=control_node_names,
            actuated_nodes=[
                node for node in control_node_names if node not in passive_control_nodes
            ],
            edges=edges,
            tracker_mounts=tracker_mounts,
        )
        return expanded

    @classmethod
    def from_files(cls, serial_map_file: str, layout_file: str) -> "TrackerLayout":
        """Load a complete hand-authored graph and tracker definition."""
        serial_data = _load_json(serial_map_file)
        serial_map = serial_data.get("serial_to_tracker_id", serial_data)
        layout = _load_json(layout_file)
        if "triangles" in layout:
            layout = cls._expand_triangle_definition(layout)
        assignment_mode = str(
            layout.get("tracker_assignment", layout.get("assignment_mode", "manual"))
        ).lower()
        if assignment_mode not in {"manual", "triangle_planes"}:
            raise ValueError(
                "Hand-authored layouts require tracker_assignment 'manual' or "
                "'triangle_planes'"
            )
        node_names = tuple(str(value) for value in layout["node_names"])
        tracker_map = {
            str(k): str(v)
            for k, v in layout.get("tracker_id_to_node", {}).items()
        }
        if len(set(node_names)) != len(node_names):
            raise ValueError("node_names must be unique")

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
        tracker_mounts = cls._parse_tracker_mounts(layout, assignment_mode)
        plane_parallel_tolerance = float(layout.get("plane_parallel_tolerance", 1e-6))
        if not np.isfinite(plane_parallel_tolerance) or plane_parallel_tolerance <= 0.0:
            raise ValueError("plane_parallel_tolerance must be positive and finite")
        result = cls(
            node_names=node_names,
            serial_to_tracker_id={str(k): str(v) for k, v in serial_map.items()},
            tracker_id_to_node=tracker_map,
            edge_index=edge_index,
            action_mask=np.asarray([name in actuated for name in node_names], dtype=bool),
            edge_role=edge_role,
            steamvr_to_policy_matrix=coordinate_matrix,
            assignment_mode=assignment_mode,
            tracker_mounts=tracker_mounts,
            plane_parallel_tolerance=plane_parallel_tolerance,
        )
        result._validate_tracker_map()
        return result

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
        if assignment_mode not in {"automatic", "manual", "triangle_planes"}:
            raise ValueError(
                "tracker_assignment must be 'automatic', 'manual', or 'triangle_planes'"
            )
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
        tracker_mounts = cls._parse_tracker_mounts(layout_data, assignment_mode)
        plane_parallel_tolerance = float(
            layout_data.get(
                "plane_parallel_tolerance",
                getattr(cfg, "plane_parallel_tolerance", 1e-6),
            )
        )
        if not np.isfinite(plane_parallel_tolerance) or plane_parallel_tolerance <= 0.0:
            raise ValueError("plane_parallel_tolerance must be positive and finite")

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
                tracker_mounts=tracker_mounts,
                plane_parallel_tolerance=plane_parallel_tolerance,
            )
            result._validate_tracker_map()
            return result
        finally:
            env.close()

    def _validate_tracker_map(self) -> None:
        if self.assignment_mode == "automatic":
            return
        if self.assignment_mode == "triangle_planes":
            self._validate_tracker_mounts()
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
            raise ValueError(f"Tracker map refers to unknown graph nodes: {sorted(unknown_nodes)}")
        if missing_nodes:
            raise ValueError(f"Every graph node needs one tracker; missing: {sorted(missing_nodes)}")
        if len(set(self.tracker_id_to_node.values())) != len(self.tracker_id_to_node):
            raise ValueError("Only one tracker may map to each node")

    @staticmethod
    def _parse_tracker_mounts(
        layout_data: Mapping[str, object], assignment_mode: str
    ) -> dict[str, TrackerMount] | None:
        if assignment_mode != "triangle_planes":
            return None
        raw_mounts = layout_data.get("tracker_mounts")
        if not isinstance(raw_mounts, dict) or not raw_mounts:
            raise ValueError(
                "triangle_planes assignment requires a nonempty tracker_mounts object"
            )
        mounts = {}
        for raw_tracker_id, raw_mount in raw_mounts.items():
            tracker_id = str(raw_tracker_id)
            if not isinstance(raw_mount, dict):
                raise ValueError(f"tracker_mounts[{tracker_id!r}] must be an object")
            missing_fields = {
                field
                for field in ("triangle", "abstract_node")
                if field not in raw_mount
            }
            if missing_fields:
                raise ValueError(
                    f"tracker_mounts[{tracker_id!r}] is missing fields: "
                    f"{sorted(missing_fields)}"
                )
            triangles = raw_mount.get("joint_triangles")
            if not isinstance(triangles, list) or len(triangles) != 2:
                raise ValueError(
                    f"tracker_mounts[{tracker_id!r}].joint_triangles must contain two triangles"
                )
            if str(triangles[0]) == str(triangles[1]):
                raise ValueError(
                    f"tracker_mounts[{tracker_id!r}].joint_triangles must be distinct"
                )
            normal = np.asarray(
                raw_mount.get("local_plane_normal", [0.0, 0.0, 1.0]),
                dtype=np.float64,
            )
            offset = np.asarray(
                raw_mount.get("local_plane_point_offset", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-8:
                raise ValueError(
                    f"tracker_mounts[{tracker_id!r}].local_plane_normal must be a finite nonzero 3-vector"
                )
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(
                    f"tracker_mounts[{tracker_id!r}].local_plane_point_offset must be a finite 3-vector"
                )
            mounts[tracker_id] = TrackerMount(
                tracker_id=tracker_id,
                triangle=str(raw_mount["triangle"]),
                abstract_node=str(raw_mount["abstract_node"]),
                joint_triangles=(str(triangles[0]), str(triangles[1])),
                local_plane_normal=normal / np.linalg.norm(normal),
                local_plane_point_offset=offset,
            )
        return mounts

    def _validate_tracker_mounts(self) -> None:
        mounts = self.tracker_mounts or {}
        known_tracker_ids = set(self.serial_to_tracker_id.values())
        unknown_trackers = set(mounts) - known_tracker_ids
        if unknown_trackers:
            raise ValueError(
                "Tracker mounts refer to tracker IDs absent from the serial map: "
                f"{sorted(unknown_trackers)}"
            )
        tracked_triangles = set()
        abstract_nodes = set()
        for mount in mounts.values():
            tracked_triangles.add(mount.triangle)
            if mount.abstract_node in abstract_nodes:
                raise ValueError(
                    f"Abstract node {mount.abstract_node!r} has more than one rigidly connected tracker"
                )
            abstract_nodes.add(mount.abstract_node)
        missing_triangles = {
            triangle
            for mount in mounts.values()
            for triangle in mount.joint_triangles
            if triangle not in tracked_triangles
        }
        if missing_triangles:
            raise ValueError(
                "Joint reconstruction refers to triangles without tracker mounts: "
                f"{sorted(missing_triangles)}"
            )
        for mount in mounts.values():
            if mount.triangle not in mount.joint_triangles:
                raise ValueError(
                    f"Tracker {mount.tracker_id!r} is mounted on {mount.triangle!r}, which must "
                    "also appear in its joint_triangles"
                )
        graph_abstract_nodes = {
            self._abstract_node_name(node_name) for node_name in self.node_names
        }
        missing_abstract_nodes = graph_abstract_nodes - abstract_nodes
        unknown_abstract_nodes = abstract_nodes - graph_abstract_nodes
        if unknown_abstract_nodes:
            raise ValueError(
                "Tracker mounts refer to unknown abstract nodes: "
                f"{sorted(unknown_abstract_nodes)}"
            )
        if missing_abstract_nodes:
            raise ValueError(
                "Every graph abstract node needs one tracker mount; missing: "
                f"{sorted(missing_abstract_nodes)}"
            )

    @staticmethod
    def _abstract_node_name(node_name: str) -> str:
        return str(node_name).split("_tri_", 1)[0]

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

    @property
    def requires_orientations(self) -> bool:
        return self.assignment_mode == "triangle_planes"

    @staticmethod
    def _coerce_tracker_pose(value: object, serial: str) -> TrackerPose:
        if isinstance(value, TrackerPose):
            position = np.asarray(value.position, dtype=np.float64)
            rotation = np.asarray(value.rotation_matrix, dtype=np.float64)
        elif isinstance(value, Mapping):
            position = np.asarray(value.get("position"), dtype=np.float64)
            rotation_value = value.get("rotation_matrix", value.get("rotation"))
            rotation = np.asarray(rotation_value, dtype=np.float64)
        else:
            raise ValueError(
                f"Tracker {serial!r} pose must provide position and rotation_matrix"
            )
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Tracker {serial!r} position must be a finite 3-vector")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(f"Tracker {serial!r} rotation_matrix must be a finite 3x3 matrix")
        orthogonality_error = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
        if orthogonality_error > 1e-4 or np.linalg.det(rotation) < 0.0:
            raise ValueError(f"Tracker {serial!r} rotation_matrix must be a proper rotation")
        return TrackerPose(position=position, rotation_matrix=rotation)

    def _triangle_planes(
        self, poses_by_serial: Mapping[str, object]
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
        mounts = self.tracker_mounts or {}
        pose_by_tracker = {}
        for serial, raw_pose in poses_by_serial.items():
            tracker_id = self.serial_to_tracker_id.get(str(serial))
            if tracker_id in mounts:
                pose_by_tracker[tracker_id] = self._coerce_tracker_pose(raw_pose, str(serial))
        missing = sorted(set(mounts) - set(pose_by_tracker))
        if missing:
            raise MissingTrackerFrame(
                f"Missing valid SteamVR poses for tracker mounts: {missing}"
            )

        plane_estimates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        anchor_positions = {}
        coordinate_matrix = self.steamvr_to_policy_matrix
        for tracker_id, mount in mounts.items():
            pose = pose_by_tracker[tracker_id]
            position = coordinate_matrix @ pose.position
            normal = np.linalg.solve(
                coordinate_matrix.T,
                pose.rotation_matrix @ mount.local_plane_normal,
            )
            normal_length = np.linalg.norm(normal)
            if normal_length <= 1e-8:
                raise ValueError(
                    f"Tracker {tracker_id!r} produced a zero triangle-plane normal"
                )
            plane_point = position + coordinate_matrix @ (
                pose.rotation_matrix @ mount.local_plane_point_offset
            )
            plane_estimates.setdefault(mount.triangle, []).append(
                (plane_point, normal / normal_length)
            )
            anchor_positions[tracker_id] = position
        planes = {}
        for triangle, estimates in plane_estimates.items():
            reference_normal = estimates[0][1]
            aligned_normals = [
                normal if normal @ reference_normal >= 0.0 else -normal
                for _, normal in estimates
            ]
            fused_normal = np.mean(aligned_normals, axis=0)
            fused_length = np.linalg.norm(fused_normal)
            if fused_length <= 1e-8:
                raise MissingTrackerFrame(
                    f"Tracker plane normals for triangle {triangle!r} cancel each other"
                )
            fused_normal /= fused_length
            fused_offset = float(
                np.mean([fused_normal @ point for point, _ in estimates])
            )
            planes[triangle] = (fused_normal * fused_offset, fused_normal)
        return planes, anchor_positions

    def reconstructed_positions(self, poses_by_serial: Mapping[str, object]) -> np.ndarray:
        """Recover abstract joints and expand them into preset control-graph order."""
        if not self.requires_orientations:
            raise ValueError("Joint reconstruction requires triangle_planes assignment")
        planes, anchor_positions = self._triangle_planes(poses_by_serial)
        positions_by_abstract_node = {}
        for mount in (self.tracker_mounts or {}).values():
            triangle_i, triangle_j = mount.joint_triangles
            point_i, normal_i = planes[triangle_i]
            point_j, normal_j = planes[triangle_j]
            line_vector = np.cross(normal_i, normal_j)
            line_scale = np.linalg.norm(line_vector)
            if line_scale <= self.plane_parallel_tolerance:
                raise MissingTrackerFrame(
                    f"Triangle planes {triangle_i!r} and {triangle_j!r} are parallel or "
                    f"too nearly parallel (cross-normal magnitude={line_scale:.3e})"
                )
            line_direction = line_vector / line_scale
            constraints = np.stack((normal_i, normal_j), axis=0)
            right_hand_side = np.asarray(
                [normal_i @ point_i, normal_j @ point_j], dtype=np.float64
            )
            line_point = np.linalg.lstsq(
                constraints, right_hand_side, rcond=None
            )[0]
            anchor = anchor_positions[mount.tracker_id]
            joint_position = line_point + line_direction * (
                line_direction @ (anchor - line_point)
            )
            positions_by_abstract_node[mount.abstract_node] = joint_position

        return np.stack(
            [
                positions_by_abstract_node[self._abstract_node_name(node_name)]
                for node_name in self.node_names
            ]
        )

    def ordered_positions(self, positions_by_serial: Mapping[str, object]) -> np.ndarray:
        if self.requires_orientations:
            return self.reconstructed_positions(positions_by_serial)
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


def _run_print_only_control_loop(cfg, source, layout, builder, agent, formatter) -> None:
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
                if getattr(layout, "requires_orientations", False):
                    poses_method = getattr(source, "poses_by_serial", None)
                    if poses_method is None:
                        raise RuntimeError(
                            "triangle_planes assignment requires a tracker source that provides orientations"
                        )
                    tracker_frame = poses_method()
                else:
                    tracker_frame = source.positions_by_serial()
                positions = layout.ordered_positions(tracker_frame)
                observation = builder.build(positions, now)
                action = agent.act(
                    observation,
                    t0=step == 0,
                    eval_mode=bool(cfg.deterministic),
                )
                normalized_node_velocity = action.detach().cpu().numpy()
                print(formatter.velocity_command(normalized_node_velocity), flush=True)
                step += 1
            except MissingTrackerFrame as exc:
                print(f"tracker frame skipped: {exc}", file=sys.stderr, flush=True)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print(formatter.emergency_stop_command(), flush=True)
        print("Emergency stop command printed after Ctrl-C.", file=sys.stderr, flush=True)


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
    graph_definition_file = getattr(cfg, "graph_definition_file", None)
    if graph_definition_file in (None, "", "null"):
        graph_definition_file = None
    if graph_definition_file is not None:
        if layout_file is not None:
            raise ValueError(
                "Set either graph_definition_file or tracker_layout_file, not both"
            )
        layout = TrackerLayout.from_files(
            str(cfg.serial_map_file), str(graph_definition_file)
        )
    else:
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
    configured_order = getattr(cfg, "serial_node_order", None)
    if configured_order is None or (
        isinstance(configured_order, str) and configured_order in ("", "null")
    ):
        configured_order = None
    formatter = SerialVelocityCommandFormatter(
        layout.node_names,
        layout.action_mask,
        configured_order,
        max_velocity_ticks_per_second=int(cfg.serial_max_velocity_ticks_per_second),
        duration_seconds=float(cfg.serial_command_duration_s),
    )
    checkpoint = resolve_checkpoint(str(cfg.model), cfg)
    agent = GNNSAC(cfg)
    load_agent_checkpoint(agent, checkpoint)
    agent.model.eval()
    source = source or SteamVRTrackerSource()
    try:
        _run_print_only_control_loop(cfg, source, layout, builder, agent, formatter)
    finally:
        source.close()


@hydra.main(config_name="inference/real_robot", config_path="../config", version_base=None)
def infer(cfg):
    run_real_robot_inference(cfg)


if __name__ == "__main__":
    infer()
