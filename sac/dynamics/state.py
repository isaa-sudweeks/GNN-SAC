"""Physical-unit coordinate helpers; these do not assert Markov sufficiency."""

from __future__ import annotations

import numpy as np


def observation_to_physical(x: np.ndarray, bbox: np.ndarray, *, normalized: bool) -> np.ndarray:
    """Undo per-axis policy scaling, retaining centered xy and absolute z."""
    values = np.asarray(x, dtype=np.float64)
    dimensions = np.asarray(bbox, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected physical-node observations with shape (N, 6).")
    if dimensions.shape != (3,) or np.any(dimensions <= 0) or not np.isfinite(dimensions).all():
        raise ValueError("Bounding-box dimensions must be finite positive xyz values.")
    if not np.isfinite(values).all():
        raise ValueError("Observation contains nonfinite values.")
    return values * np.tile(dimensions, 2) if normalized else values.copy()


def center_xy(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center node xy without removing height; return removed xy translation."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("Expected nonempty positions with shape (N, 3).")
    centered = positions.copy()
    origin = centered[:, :2].mean(axis=0)
    centered[:, :2] -= origin
    return centered, origin


def integrate_centered_positions(
    positions: np.ndarray, velocities: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity baseline with separately accumulated mean xy movement."""
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative.")
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.shape != velocities.shape:
        raise ValueError("Position and velocity shapes must match.")
    centered, old_origin = center_xy(positions)
    next_centered, movement = center_xy(centered + velocities * dt)
    return next_centered, movement


def geometry_features(positions: np.ndarray, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recompute directed displacement and distance from current imagined geometry."""
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edge_index)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (N, 3).")
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must be an integer array with shape (2, E).")
    if edges.size and (edges.min() < 0 or edges.max() >= len(positions)):
        raise ValueError("Edge index points outside the physical nodes.")
    relative = positions[edges[1]] - positions[edges[0]]
    return relative, np.linalg.norm(relative, axis=-1, keepdims=True)


def mass_weighted_com(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute body mass COM; callers must supply body inertial-center positions."""
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.shape != (len(masses), 3) or masses.ndim != 1:
        raise ValueError("Masses must match positions.")
    if not np.isfinite(masses).all() or np.any(masses < 0) or masses.sum() <= 0:
        raise ValueError("Masses must be finite, nonnegative, and have positive total.")
    return np.sum(positions * masses[:, None], axis=0) / masses.sum()
