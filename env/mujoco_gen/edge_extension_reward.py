"""One-sided squared extension of physical control-graph edges."""

import math
from collections.abc import Mapping

import numpy as np
from mujoco_truss_gen import get_edge_index


def edge_extension_weight(config: object) -> float:
    value = (
        config.get("edge_extension_weight", 0.0)
        if isinstance(config, Mapping)
        else getattr(config, "edge_extension_weight", 0.0)
    )
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("edge_extension_weight must be finite and nonnegative.")
    return weight


def nominal_edge_lengths(
    model: object, observation_source: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return one pair of body IDs and a nominal length in meters per edge."""
    edges = get_edge_index(model, graph_view="control").T
    pairs = np.unique(np.sort(edges, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not pairs.size:
        raise ValueError("The edge extension penalty requires a physical control graph.")
    bodies = np.asarray(model.get_control_node_body_ids(observation_source))[pairs]
    lengths = edge_lengths(model.data.xpos, bodies, np)
    return bodies, lengths


def edge_lengths(positions, bodies, xp):
    displacement = positions[..., bodies[:, 0], :] - positions[..., bodies[:, 1], :]
    return xp.linalg.norm(displacement, axis=-1)


def squared_edge_extension(positions, bodies, nominal_lengths, xp):
    """Sum squared positive extensions over edges, retaining any batch axes."""
    extension = xp.maximum(edge_lengths(positions, bodies, xp) - nominal_lengths, 0.0)
    return xp.sum(xp.square(extension), axis=-1)
