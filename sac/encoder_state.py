"""Encoder shape reconstruction and two-pose alignment, independent of the policy.

Distances are metres; rotations map local column vectors into their parent frame.
No motor commands or learned-model changes belong in this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = finite_array(self.position, (3,), "position")
        rotation = finite_array(self.rotation, (3, 3), "rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
            raise ValueError("rotation must be a proper orthonormal rotation")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)

    def compose(self, other: Pose) -> Pose:
        return Pose(self.position + self.rotation @ other.position, self.rotation @ other.rotation)

    def inverse(self) -> Pose:
        return Pose(-self.rotation.T @ self.position, self.rotation.T)

    def apply(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions) @ self.rotation.T + self.position


def triangle_frame(points: np.ndarray) -> Pose:
    """Frame at first triangle node, x towards second, z from ordered cross product."""
    points = finite_array(points, (3, 3), "triangle points")
    x = points[1] - points[0]
    z = np.cross(x, points[2] - points[0])
    if min(np.linalg.norm(x), np.linalg.norm(z)) < 1e-9:
        raise ValueError("Degenerate triangle cannot define a tracker mount frame")
    x /= np.linalg.norm(x)
    z /= np.linalg.norm(z)
    return Pose(points[0], np.column_stack((x, np.cross(z, x), z)))


def roller_frame(points: np.ndarray) -> Pose:
    """Node housing frame: x along the interior corner bisector, z triangle normal.

    A roller housing follows the bisector as its corner angle changes; using the
    next tube edge as its x axis would introduce a false rotation during deformation.
    """
    frame = triangle_frame(points)
    a, b = np.asarray(points[1]) - points[0], np.asarray(points[2]) - points[0]
    x = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("Degenerate corner cannot define a roller mount frame")
    x /= np.linalg.norm(x)
    z = frame.rotation[:, 2]
    return Pose(frame.position, np.column_stack((x, np.cross(z, x), z)))


def fit_mount_yaw_from_point(
    tracker_world_pose: Pose,
    shared_point_world: np.ndarray,
    shared_point_housing: np.ndarray,
    tracker_offset_housing: np.ndarray,
    fixed_axis_alignment: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit R_housing_from_tracker = Rz(yaw) @ fixed_axis_alignment.

    One known point plus the fixed mounting-normal alignment determines yaw when
    the point has a nonzero in-plane baseline. Return point residuals explicitly:
    fitting yaw must not hide inconsistent geometry or manufacture roll/pitch.
    This computes a candidate, not a physical-calibration acceptance decision.
    """
    point_world = finite_array(shared_point_world, (3,), "shared world point")
    point_housing = finite_array(shared_point_housing, (3,), "shared housing point")
    offset = finite_array(tracker_offset_housing, (3,), "tracker housing offset")
    base = Pose(np.zeros(3), fixed_axis_alignment).rotation
    measured_tracker = tracker_world_pose.rotation.T @ (point_world - tracker_world_pose.position)
    measured = base @ measured_tracker
    target = point_housing - offset
    if min(np.linalg.norm(measured[:2]), np.linalg.norm(target[:2])) <= 1e-9:
        raise ValueError("Yaw is unobservable from a point on the mounting-normal axis")
    yaw = np.arctan2(target[1], target[0]) - np.arctan2(measured[1], measured[0])
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]) @ base
    residual = rotation @ measured_tracker - target
    return rotation, {"yaw_rad": yaw, "point_error_m": float(np.linalg.norm(residual)),
                      "normal_error_m": float(residual[2]),
                      "in_plane_length_error_m": float(np.linalg.norm(measured[:2]) - np.linalg.norm(target[:2]))}


class OffsetShapeSolver:
    """Solve 3-D tube lengths with fixed connector rods on triangle bisectors.

    Inspired by the offset constraints in Isoperimetric_Kinematics, with spatial
    triangles and full reference offsets from generated MuJoCo geometry.
    A six-coordinate gauge removes rigid motion;
    tracker alignment recovers that motion afterwards. No dynamics are integrated.
    """

    def __init__(self, roller_positions: np.ndarray, ball_positions: np.ndarray,
                 triangle_indices: Sequence[Sequence[int]], ball_indices: np.ndarray,
                 length_incidence: np.ndarray, *, tolerance_m: float, max_evaluations: int) -> None:
        self.initial_rollers = np.asarray(roller_positions, dtype=float).copy()
        self.initial_balls = np.asarray(ball_positions, dtype=float).copy()
        self.n = len(self.initial_rollers)
        self.b = len(self.initial_balls)
        finite_array(self.initial_rollers, (self.n, 3), "roller positions")
        finite_array(self.initial_balls, (self.b, 3), "ball positions")
        self.triangles = np.asarray(triangle_indices, dtype=int)
        self.ball_indices = np.asarray(ball_indices, dtype=int)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or self.ball_indices.shape != (self.n,):
            raise ValueError("Expected triangles and one logical ball index per roller")
        if sorted(self.triangles.ravel().tolist()) != list(range(self.n)):
            raise ValueError("Each control-node occurrence must belong to exactly one triangle")
        if np.any(self.ball_indices < 0) or np.any(self.ball_indices >= self.b):
            raise ValueError("Invalid ball index")
        self.edges = np.concatenate([np.column_stack((t, np.roll(t, -1))) for t in self.triangles])
        self.incidence = np.asarray(length_incidence, dtype=float)
        if self.incidence.ndim != 2 or self.incidence.shape[0] != len(self.edges) or not np.isfinite(self.incidence).all():
            raise ValueError("Invalid encoder-to-edge incidence matrix")
        self.reference_lengths = self._lengths(self.initial_rollers)
        self.rod_lengths = np.linalg.norm(self.initial_rollers - self.initial_balls[self.ball_indices], axis=1)
        if np.any(self.rod_lengths <= 0):
            raise ValueError("Offset reconstruction requires nonzero connector rods")
        # Preserve the generator's complete reference offset, including small
        # deviations from an ideal planar bisector. Otherwise zero encoders can
        # move the nominal reconstructed nodes by centimetres.
        self.offset_components = np.zeros((self.n, 3))
        for triangle in self.triangles:
            p = self.initial_rollers[triangle]
            for i, basis in enumerate(self._corner_bases(p)):
                node = triangle[i]
                self.offset_components[node] = basis.T @ (p[i] - self.initial_balls[self.ball_indices[node]])
        self.tolerance = float(tolerance_m)
        self.max_evaluations = int(max_evaluations)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0 or self.max_evaluations <= 0:
            raise ValueError("Solver tolerance and evaluation budget must be positive")
        # Use three noncollinear ball positions to establish a local coordinate gauge.
        self.gauge_indices = None
        for j in range(1, self.b):
            for k in range(j + 1, self.b):
                try:
                    self.gauge = triangle_frame(self.initial_balls[[0, j, k]])
                except ValueError:
                    continue
                self.gauge_indices = (0, j, k)
                break
            if self.gauge_indices is not None:
                break
        if self.gauge_indices is None:
            raise ValueError("Reference balls must contain three noncollinear points")
        self.reference = np.vstack((self.initial_rollers, self.initial_balls))
        self.reference = self.gauge.inverse().apply(self.reference)
        self.current = self.reference.copy()
        self.last_diagnostics = {}

    def _lengths(self, rollers):
        return np.linalg.norm(rollers[self.edges[:, 1]] - rollers[self.edges[:, 0]], axis=1)

    @staticmethod
    def _corner_bases(p):
        a = np.roll(p, -1, axis=-2) - p
        b = np.roll(p, 1, axis=-2) - p
        a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        bisector = a + b
        bisector /= np.maximum(np.linalg.norm(bisector, axis=-1, keepdims=True), 1e-12)
        normal = np.cross(a, b)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        tangent = np.cross(normal, bisector)
        return np.stack((bisector, tangent, normal), axis=-1)

    def _residual(self, flat, target_lengths):
        points = flat.reshape(-1, 3)
        rollers, balls = points[:self.n], points[self.n:]
        offsets = np.zeros_like(rollers)
        p = rollers[self.triangles]
        offset = np.einsum("...ij,...j->...i", self._corner_bases(p), self.offset_components[self.triangles])
        offsets[self.triangles] = p - offset - balls[self.ball_indices[self.triangles]]
        i, j, k = self.gauge_indices
        gauge = np.r_[balls[i], balls[j, 1:], balls[k, 2]]
        return np.r_[self._lengths(rollers) - target_lengths, offsets.ravel(), gauge]

    def _jacobian(self, flat: np.ndarray, target_lengths: np.ndarray) -> np.ndarray:
        """Exact derivatives of the same length, offset and gauge constraints.

        Numerical differencing previously evaluated every triangle for each of
        the 54 coordinates on each iteration, dominating live reconstruction.
        """
        rollers = flat.reshape(-1, 3)[:self.n]
        p = rollers[self.triangles]
        a, b = np.roll(p, -1, axis=-2) - p, np.roll(p, 1, axis=-2) - p
        identity = np.eye(3)

        def normalize_with_jacobian(vector):
            length = np.linalg.norm(vector, axis=-1, keepdims=True)
            denominator = np.maximum(length, 1e-12)
            unit = vector / denominator
            projection = identity - unit[..., :, None] * unit[..., None, :]
            # Match the residual's clipped normalization at degenerate inputs.
            projection = np.where((length > 1e-12)[..., None], projection, identity)
            return unit, projection / denominator[..., None]

        def cross_matrix(vector):
            result = np.zeros((*vector.shape[:-1], 3, 3))
            result[..., 0, 1], result[..., 0, 2] = -vector[..., 2], vector[..., 1]
            result[..., 1, 0], result[..., 1, 2] = vector[..., 2], -vector[..., 0]
            result[..., 2, 0], result[..., 2, 1] = -vector[..., 1], vector[..., 0]
            return result

        u, du = normalize_with_jacobian(a)
        v, dv = normalize_with_jacobian(b)
        bisector, db = normalize_with_jacobian(u + v)
        normal, dn = normalize_with_jacobian(np.cross(u, v))
        components = self.offset_components[self.triangles]
        c0, c1, c2 = (components[..., i, None, None] for i in range(3))
        # d(normal x bisector) = -[bisector]x dnormal + [normal]x dbisector.
        along_bisector = c0 * identity + c1 * cross_matrix(normal)
        along_normal = c2 * identity - c1 * cross_matrix(bisector)
        d_offset_a = along_bisector @ db @ du - along_normal @ dn @ cross_matrix(v) @ du
        d_offset_b = along_bisector @ db @ dv + along_normal @ dn @ cross_matrix(u) @ dv
        edge_count = len(self.edges)
        jacobian = np.zeros((edge_count + 3 * self.n + 6, 3 * (self.n + self.b)))
        delta = rollers[self.edges[:, 1]] - rollers[self.edges[:, 0]]
        directions = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        for row, (first, second) in enumerate(self.edges):
            jacobian[row, 3 * first:3 * first + 3] = -directions[row]
            jacobian[row, 3 * second:3 * second + 3] = directions[row]
        for t, triangle in enumerate(self.triangles):
            for corner, node in enumerate(triangle):
                rows = slice(edge_count + 3 * node, edge_count + 3 * node + 3)
                other_a, other_b = triangle[(corner + 1) % 3], triangle[(corner - 1) % 3]
                jacobian[rows, 3 * node:3 * node + 3] = identity + d_offset_a[t, corner] + d_offset_b[t, corner]
                jacobian[rows, 3 * other_a:3 * other_a + 3] = -d_offset_a[t, corner]
                jacobian[rows, 3 * other_b:3 * other_b + 3] = -d_offset_b[t, corner]
                ball_column = 3 * (self.n + self.ball_indices[node])
                jacobian[rows, ball_column:ball_column + 3] = -identity
        i, j, k = self.gauge_indices
        columns = [3 * (self.n + i) + axis for axis in range(3)]
        columns += [3 * (self.n + j) + 1, 3 * (self.n + j) + 2, 3 * (self.n + k) + 2]
        jacobian[np.arange(edge_count + 3 * self.n, len(jacobian)), columns] = 1
        return jacobian

    def solve(self, encoder_displacements_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        displacement = finite_array(encoder_displacements_m, (self.incidence.shape[1],), "encoder displacement")
        target = self.reference_lengths + self.incidence @ displacement
        if np.any(target <= 1e-6):
            raise ValueError("Encoder data implies a nonpositive tube segment")
        for triangle_lengths in target.reshape(-1, 3):
            if 2 * max(triangle_lengths) >= sum(triangle_lengths) - 1e-8:
                raise ValueError("Encoder data violates a triangle inequality")
        started = time.monotonic()
        solution = least_squares(self._residual, self.current.ravel(), args=(target,),
                                 jac=self._jacobian,
                                 max_nfev=self.max_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = float(np.max(np.abs(solution.fun)))
        singular_values = np.linalg.svd(solution.jac, compute_uv=False)
        condition = float(singular_values[0] / max(singular_values[-1], 1e-15))
        diagnostics = {"max_constraint_error_m": error, "condition_number": condition,
                       "evaluations": int(solution.nfev), "jacobian_evaluations": int(solution.njev or 0),
                       "jacobian": "analytic", "solve_time_s": time.monotonic() - started}
        if not solution.success or error > self.tolerance or condition > 1e10:
            raise ValueError(f"Offset shape solve rejected: {diagnostics}")
        self.current = solution.x.reshape(-1, 3).copy()
        result = self.gauge.apply(self.current)
        self.last_diagnostics = diagnostics
        return result[:self.n], result[self.n:], diagnostics


class TwoTrackerFusion:
    """Inverse-variance fusion of shape-compensated poses, with explicit hold-last.

    Translation and rotation use separate weights (m² and rad²). Variance is
    measured on second differences to remove constant-velocity robot motion.
    This estimates jitter, not absolute calibration error or physical covariance.
    """

    def __init__(self, tracker_ids: Sequence[str], *, window: int, position_variance_floor: float,
                 rotation_variance_floor: float, initial_position_variance: float,
                 initial_rotation_variance: float, reacquire_after_s: float) -> None:
        self.ids = tuple(tracker_ids)
        if len(self.ids) != 2 or len(set(self.ids)) != 2:
            raise ValueError("Exactly two distinct tracker IDs are required")
        self.window = int(window)
        if self.window < 3:
            raise ValueError("Variance window must contain at least three frames")
        self.pfloor, self.rfloor = float(position_variance_floor), float(rotation_variance_floor)
        self.pinitial, self.rinitial = float(initial_position_variance), float(initial_rotation_variance)
        self.reacquire_after = float(reacquire_after_s)
        if any(not np.isfinite(x) or x <= 0 for x in (self.pfloor, self.rfloor, self.pinitial, self.rinitial, self.reacquire_after)):
            raise ValueError("Variance parameters and reacquisition interval must be positive")
        self.history = {key: deque(maxlen=self.window) for key in self.ids}
        self.last_seen = {}
        self.last_pose = None
        self.last_observed = None
        self.previous_time = None

    def update(self, candidates: Mapping[str, Pose], timestamp: float) -> tuple[Pose | None, dict]:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("Fusion timestamps must be finite and strictly increasing")
        self.previous_time = timestamp
        if set(candidates) - set(self.ids):
            raise ValueError("Unexpected tracker ID in fusion input")
        variances = {}
        for key, pose in candidates.items():
            history = self.history[key]
            if key in self.last_seen and timestamp - self.last_seen[key] > self.reacquire_after:
                history.clear()
            history.append((timestamp, pose))
            self.last_seen[key] = timestamp
            vp, vr = self.pinitial, self.rinitial
            if len(history) >= 3:
                position_innovations, rotation_innovations = [], []
                for a, b, c in zip(list(history)[:-2], list(history)[1:-1], list(history)[2:]):
                    ratio = (c[0] - b[0]) / (b[0] - a[0])
                    noise_gain = 1 + (1 + ratio)**2 + ratio**2
                    predicted = b[1].position + ratio * (b[1].position - a[1].position)
                    position_innovations.append(np.sum((c[1].position - predicted)**2) / (3 * noise_gain))
                    ab = Rotation.from_matrix(b[1].rotation @ a[1].rotation.T).as_rotvec()
                    bc = Rotation.from_matrix(c[1].rotation @ b[1].rotation.T).as_rotvec()
                    rotation_innovations.append(np.sum((bc - ratio * ab)**2) / (3 * noise_gain))
                vp, vr = float(np.mean(position_innovations)), float(np.mean(rotation_innovations))
            variances[key] = {"position_m2": max(self.pfloor, vp), "rotation_rad2": max(self.rfloor, vr)}
        visible = [key for key in self.ids if key in candidates]
        status = "two_trackers" if len(visible) == 2 else "one_tracker" if visible else "held_global_transform"
        weights = {}
        if visible:
            wp = np.array([1 / variances[key]["position_m2"] for key in visible])
            wr = np.array([1 / variances[key]["rotation_rad2"] for key in visible])
            wp /= sum(wp)
            wr /= sum(wr)
            position = sum(weight * candidates[key].position for key, weight in zip(visible, wp))
            # Rotation.mean uses a quaternion eigensystem: sign-safe and on SO(3).
            rotation = Rotation.from_matrix(np.stack([candidates[key].rotation for key in visible])).mean(weights=wr).as_matrix()
            self.last_pose = Pose(position, rotation)
            self.last_observed = timestamp
            weights = {key: {"position": float(p), "rotation": float(r)} for key, p, r in zip(visible, wp, wr)}
        if self.last_pose is None:
            status = "uninitialized"
        missing = [key for key in self.ids if key not in candidates]
        diagnostics = {"status": status, "visible_trackers": visible, "missing_trackers": missing,
                       "weights": weights, "variance_estimates": variances,
                       "global_pose_age_s": None if self.last_observed is None else timestamp - self.last_observed,
                       "global_pose_observed": bool(visible),
                       "warning": f"Missing trackers: {missing}" if missing else None}
        if len(visible) == 2:
            a, b = (candidates[key] for key in visible)
            diagnostics["position_disagreement_m"] = float(np.linalg.norm(a.position - b.position))
            diagnostics["rotation_disagreement_rad"] = float(Rotation.from_matrix(a.rotation @ b.rotation.T).magnitude())
        return self.last_pose, diagnostics
