"""Geometry-only rigidity for nominal abstract trusses, matching the simulator.

This helper must receive physical nodes and structural bars, not control-graph
alias edges. It needs no simulator state or future labels during prediction.
"""
from __future__ import annotations

import numpy as np


def critical_eigenvalue(
    positions: np.ndarray,
    structural_edges: np.ndarray,
    *,
    axis_indices: tuple[int, ...] = (0, 1, 2),
    weighted: bool = False,
) -> float:
    """Return the first non-rigid eigenvalue using the simulator's rank rule.

    Edges have shape (E, 2), contain each physical bar once, and use physical
    node indices. Degenerate bars shorter than 1e-8 metres are omitted, matching
    mujoco-truss-gen. Nonfinite imagined positions produce NaN for failure checks.
    """
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(structural_edges)
    axes = tuple(axis_indices)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("Expected physical positions with shape (N, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Expected integer structural edges with shape (E, 2)")
    if not axes or len(set(axes)) != len(axes) or any(a not in (0, 1, 2) for a in axes):
        raise ValueError("axis_indices must be distinct xyz indices")
    if edges.size and (edges.min() < 0 or edges.max() >= len(positions)):
        raise ValueError("Structural edge points outside physical nodes")
    if not np.isfinite(positions).all():
        return float("nan")
    dims = len(axes)
    rows = []
    for first, second in edges:
        delta = positions[second, list(axes)] - positions[first, list(axes)]
        length = np.linalg.norm(delta)
        if length < 1e-8:
            continue
        row = np.zeros((len(positions), dims))
        row[first] = -delta / length
        row[second] = delta / length
        rows.append(row.ravel())
    if not rows:
        return 0.0
    matrix = np.stack(rows)
    rank = matrix.shape[1] - dims - dims * (dims - 1) // 2
    if matrix.shape[0] < rank:
        return 0.0
    self_stress_modes = matrix.shape[0] - rank
    eigenvalues = np.linalg.eigvalsh(matrix @ matrix.T)
    if eigenvalues.size <= self_stress_modes:
        return 0.0
    norm = float(np.square(matrix).sum()) if weighted else 1.0
    return float(max(eigenvalues[self_stress_modes] / norm, 0.0))


def normalized_rigidity(
    positions: np.ndarray,
    structural_edges: np.ndarray,
    initial_reference: float,
    *,
    axis_indices: tuple[int, ...] = (0, 1, 2),
    weighted: bool = False,
) -> float:
    """Normalize predicted-geometry rigidity using a fixed initial reference."""
    if not np.isfinite(initial_reference) or initial_reference <= 0:
        raise ValueError("Initial rigidity reference must be finite and positive")
    return critical_eigenvalue(positions, structural_edges,
                              axis_indices=axis_indices, weighted=weighted) / initial_reference
