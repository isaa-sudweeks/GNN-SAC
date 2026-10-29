"""Reward-independent morphology descriptors and farthest-point CV groups."""

from collections.abc import Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "nodes", "edges", "tubes", "degree_std", "clustering",
    "mean_shortest_path", "diameter", "active_edge_fraction",
    "route_length_cv", "edge_length_cv", "shape_middle_ratio", "shape_small_ratio",
)


def morphology_features(nodes: Mapping, shapes: Mapping) -> list[float]:
    """Describe the physical graph, routing, and scale-free initial geometry.

    Supports both triangle routes and routed shapes from mujoco-truss-gen.
    Node labels, translation, rotation, and uniform scale do not affect features.
    """
    labels = sorted(nodes)
    if len(labels) < 2 or not shapes:
        raise ValueError("Morphology requires at least two nodes and one tube.")
    indices = {label: index for index, label in enumerate(labels)}
    edges, active = set(), set()
    route_lengths = []
    for shape in shapes.values():
        route = shape["route"] if isinstance(shape, Mapping) else shape
        pairs = list(zip(route, route[1:]))
        route_lengths.append(len(pairs))
        edges.update(tuple(sorted((indices[a], indices[b]))) for a, b in pairs)
        active_pairs = shape.get("active_edges", pairs) if isinstance(shape, Mapping) else pairs
        active.update(tuple(sorted((indices[a], indices[b]))) for a, b in active_pairs)
    if not edges or any(a == b for a, b in edges) or not active <= edges:
        raise ValueError("Morphology routes require valid edges and active-edge subsets.")
    n = len(labels)
    adjacency = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        adjacency[a, b] = adjacency[b, a] = True
    degree = adjacency.sum(axis=1)
    clustering = []
    for neighbors in adjacency:
        ids = np.flatnonzero(neighbors)
        count = len(ids)
        clustering.append(
            adjacency[np.ix_(ids, ids)].sum() / (count * (count - 1)) if count > 1 else 0.0
        )
    distances = np.where(adjacency, 1.0, np.inf)
    np.fill_diagonal(distances, 0.0)
    for k in range(n):
        distances = np.minimum(distances, distances[:, k, None] + distances[None, k, :])
    if not np.isfinite(distances).all():
        raise ValueError("Morphology graph must be connected.")
    positions = np.asarray([nodes[label] for label in labels], dtype=float)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError("Morphology positions must be finite 3D coordinates.")
    centered = positions - positions.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / n), 0.0)
    if eigenvalues[-1] <= 0:
        raise ValueError("Morphology geometry must have nonzero extent.")
    lengths = np.asarray([np.linalg.norm(positions[a] - positions[b]) for a, b in sorted(edges)])
    if np.any(lengths <= 0):
        raise ValueError("Morphology edges must have positive length.")
    routes = np.asarray(route_lengths, dtype=float)
    features = [
        n, len(edges), len(shapes), np.std(degree), np.mean(clustering),
        np.mean(distances[np.triu_indices(n, 1)]), np.max(distances),
        len(active) / len(edges), np.std(routes) / np.mean(routes),
        np.std(lengths) / np.mean(lengths), eigenvalues[1] / eigenvalues[2],
        eigenvalues[0] / eigenvalues[2],
    ]
    return [round(float(value), 12) for value in features]


def farthest_point_partition(
    features: Mapping[str, Sequence[float]], num_folds: int,
) -> tuple[dict[str, list[str]], list[str]]:
    """Choose distant prototypes, then hold out their nearest-neighbor clusters.

    Euclidean distances use development-pool z-scores (constant columns become
    zero). Start farthest from the centroid and break ties by topology name.
    Prototype ownership ensures nonempty groups even for identical descriptors.
    Fold sizes deliberately follow morphology clusters rather than a quota.
    """
    names = sorted(features)
    if not 2 <= num_folds <= len(names):
        raise ValueError(f"num_folds must be between 2 and {len(names)}.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Features require nonempty topology names.")
    try:
        values = np.asarray([features[name] for name in names], dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Features must be equal-width finite vectors.") from error
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Features must be equal-width finite vectors.")
    scale = values.std(axis=0)
    values = (values - values.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)
    selected = [int(np.argmax(np.sum(values * values, axis=1)))]
    distances = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
    while len(selected) < num_folds:
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -1.0
        selected.append(int(np.argmax(nearest)))
    groups = {f"fold_{index}": [] for index in range(num_folds)}
    assignments = np.argmin(distances[:, selected], axis=1)
    for fold, prototype in enumerate(selected):
        assignments[prototype] = fold
    for index, name in enumerate(names):
        groups[f"fold_{assignments[index]}"].append(name)
    return groups, [names[index] for index in selected]
