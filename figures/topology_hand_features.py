"""Compute a hand-crafted graph/rigidity/actuation feature vector per topology.

Every feature here is intrinsic to the named topology alone: it is derived
from a single deterministic ``mj_forward`` at rest pose via
``mujoco_truss_gen``'s static model-building API, with no gymnasium
environment, domain randomization, or physics stepping involved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from mujoco_truss_gen import MujocoModel, get_edge_index, get_edge_types, get_mujoco_spec

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _topology_catalog

DEFAULT_OUTPUT_CSV = Path(__file__).with_name("topology_hand_features.csv")

_FAMILY_PATTERNS = (
    ("tetrahedron", re.compile(r"^tetrahedron$")),
    ("octahedron", re.compile(r"^octahedron$")),
    ("henneberg", re.compile(r"^henneberg_")),
    ("usevitch", re.compile(r"^usevitch_")),
)


def _family(topology: str) -> str:
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.match(topology):
            return family
    return "other"


def _physical_graph(model: MujocoModel) -> nx.Graph:
    edge_index = get_edge_index(model, graph_view="physical")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(model.node_names)))
    graph.add_edges_from(edge_index.T.tolist())
    return graph


def _degree_stats(graph: nx.Graph) -> dict[str, float]:
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    return {
        "degree_mean": float(degrees.mean()),
        "degree_max": float(degrees.max()),
        "degree_min": float(degrees.min()),
        "degree_std": float(degrees.std()),
    }


def _spectral_radius(graph: nx.Graph) -> float:
    adjacency = nx.to_numpy_array(graph)
    eigenvalues = np.linalg.eigvalsh(adjacency)
    return float(np.max(np.abs(eigenvalues)))


def _algebraic_connectivity(graph: nx.Graph) -> float:
    # These graphs are tiny (4-9 nodes), so a direct dense eigendecomposition
    # is both faster and fully deterministic compared to networkx's default
    # iterative (ARPACK-based) solver, which targets large sparse graphs and
    # can occasionally take a very long time to converge on graphs this small.
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(eigenvalues[1])


def _tube_connector_counts(model: MujocoModel) -> dict[str, float]:
    edge_types = get_edge_types(model, graph_view="control")
    n_tubes = int(np.sum(edge_types == "actuated")) // 2
    n_connectors = int(np.sum(edge_types == "connector")) // 2
    n_control_edges = n_tubes + n_connectors
    return {
        "n_tubes": n_tubes,
        "n_connectors": n_connectors,
        "tube_fraction": n_tubes / n_control_edges if n_control_edges else 0.0,
    }


def _active_passive_node_stats(model: MujocoModel) -> dict[str, float]:
    """Active vs. passive counts/positions in the control graph.

    "Active" and "passive" are a per-control-node-instance distinction
    (``model.control_graph.passive_control_node_names`` marks the passive
    subset, e.g. a route's endpoints or a triangle's non-driven vertex), not a
    per-physical-node one: the same physical node commonly serves as an active
    endpoint for one tube and a passive connector for another, so counting
    unique physical nodes here would double-count or miscategorize them.
    Positions come straight from the control graph
    (``get_control_node_position_matrix``), one row per control-node
    instance, in the same order as ``control_node_names``.

    Instance counts vary across topologies, so raw per-node positions can't
    go directly into a fixed-length feature vector; they're folded in as
    bounding-box-diagonal-normalized summary statistics instead, the same way
    ``_degree_stats`` summarizes a variable-length degree sequence.
    """
    control_graph = model.control_graph
    control_node_names = control_graph.control_node_names
    passive_names = set(control_graph.passive_control_node_names)
    is_passive = np.array([name in passive_names for name in control_node_names])
    positions = model.get_control_node_position_matrix("physical_node")
    bbox_diagonal = max(float(model.initial_bounding_box_diagonal), 1e-8)

    active_positions = positions[~is_passive]
    passive_positions = positions[is_passive]
    all_centroid = positions.mean(axis=0)

    def _dispersion(subset_positions: np.ndarray) -> float:
        if len(subset_positions) < 2:
            return 0.0
        centroid = subset_positions.mean(axis=0)
        return float(
            np.sqrt(np.mean(np.sum((subset_positions - centroid) ** 2, axis=1)))
        ) / bbox_diagonal

    active_centroid_offset = 0.0
    if len(active_positions):
        active_centroid_offset = (
            float(np.linalg.norm(active_positions.mean(axis=0) - all_centroid)) / bbox_diagonal
        )

    active_passive_separation = 0.0
    if len(active_positions) and len(passive_positions):
        active_passive_separation = (
            float(np.linalg.norm(active_positions.mean(axis=0) - passive_positions.mean(axis=0)))
            / bbox_diagonal
        )

    n_active = int((~is_passive).sum())
    n_passive = int(is_passive.sum())
    n_control_nodes = len(control_node_names)
    return {
        "n_active_nodes": n_active,
        "n_passive_nodes": n_passive,
        "active_node_fraction": n_active / n_control_nodes if n_control_nodes else 0.0,
        "active_centroid_offset": active_centroid_offset,
        "active_dispersion": _dispersion(active_positions),
        "passive_dispersion": _dispersion(passive_positions),
        "active_passive_separation": active_passive_separation,
    }


def compute_hand_features(topology: str) -> dict[str, Any]:
    """Compute the full hand-crafted feature vector for one named topology."""
    model = MujocoModel(get_mujoco_spec(topology))
    graph = _physical_graph(model)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    features: dict[str, Any] = {
        "topology": topology,
        "family": _family(topology),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "redundancy": n_edges - (3 * n_nodes - 6),
        "algebraic_connectivity": _algebraic_connectivity(graph),
        "diameter": float(nx.diameter(graph)),
        "n_articulation_points": len(list(nx.articulation_points(graph))),
        "n_bridges": len(list(nx.bridges(graph))),
        "spectral_radius": _spectral_radius(graph),
        "static_rigidity": float(model.initial_critical_eig),
    }
    features.update(_degree_stats(graph))
    features.update(_tube_connector_counts(model))
    features.update(_active_passive_node_stats(model))
    return features


def compute_all_hand_features(topologies: list[str]) -> pd.DataFrame:
    return pd.DataFrame([compute_hand_features(topology) for topology in topologies])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Where to write the per-topology feature CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topologies = _topology_catalog.all_topologies()
    features = compute_all_hand_features(topologies)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(features)} topology feature rows to {args.output_csv}")
    print(features.to_string(index=False))


if __name__ == "__main__":
    main()
