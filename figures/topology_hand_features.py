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

import mujoco
import networkx as nx
import numpy as np
import pandas as pd
from mujoco_truss_gen import MujocoModel, get_edge_index, get_mujoco_spec

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _topology_catalog

DEFAULT_OUTPUT_CSV = Path(__file__).with_name("topology_hand_features.csv")

# Non-feature identifier columns in the output table.
ID_COLUMNS = ("topology", "family")

# Numeric feature columns, in output order. Some are constant on the current
# catalog (redundancy, diameter, articulation points, bridges); they are kept
# as structural sanity checks and dropped from distance spaces by
# ``_feature_space.select_distance_features``.
FEATURE_COLUMNS = (
    "n_nodes",
    "n_edges",
    "redundancy",
    "algebraic_connectivity",
    "diameter",
    "n_articulation_points",
    "n_bridges",
    "spectral_radius",
    "static_rigidity",
    "degree_mean",
    "degree_max",
    "degree_min",
    "degree_std",
    "n_tubes",
    "tube_length_mean",
    "tube_length_max",
    "closed_tube_fraction",
    "n_connectors",
    "node_multiplicity_mean",
    "node_multiplicity_max",
    "n_active_nodes",
    "n_passive_nodes",
    "active_node_fraction",
    "active_centroid_offset",
    "active_dispersion",
    "passive_dispersion",
    "active_passive_separation",
)

# Eigen-solver / centroid round-off (e.g. 3.9999999999999964, 1.7e-12) is
# snapped away so exact structural ties stay exact ties in the CSV.
_ZERO_ATOL = 1e-8
_ROUND_DECIMALS = 10

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


_TUBE_CONSTRAINT_PATTERN = re.compile(r"^(Route_Length_Constraint_|Perimeter_Constraint_)")


def _tube_components(model: MujocoModel) -> list[nx.Graph]:
    """Split the control graph's actuated edges into one subgraph per tube.

    ``mujoco_truss_gen`` builds each tube (a routed cable, or a closed
    perimeter-constrained triangle) from its own control-node *instances*: a
    physical node visited by several tubes, or several times by one route,
    gets a separate instance per visit (``node_3_route_path_2_1``,
    ``node_1_tri_triangle_2``, ...), and instances are joined only by
    ``"connector"`` edges. The ``"actuated"`` control edges therefore never
    cross between tubes, so each connected component of the actuated-edge
    subgraph is exactly one tube: a simple path for a routed tube, a 3-cycle
    for a triangle. Components are returned in a deterministic order (sorted
    by their sorted node names).
    """
    actuated = nx.Graph()
    actuated.add_edges_from(
        (edge.from_node, edge.to_node)
        for edge in model.control_graph.edges
        if edge.type == "actuated"
    )
    components = [
        actuated.subgraph(nodes).copy() for nodes in nx.connected_components(actuated)
    ]
    return sorted(components, key=lambda component: sorted(component.nodes))


def _tube_constraint_count(model: MujocoModel) -> int:
    """Number of per-tube length equalities (route length or triangle perimeter)."""
    names = (
        mujoco.mj_id2name(model.model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
        for index in range(model.model.neq)
    )
    return sum(1 for name in names if name and _TUBE_CONSTRAINT_PATTERN.match(name))


def _tube_stats(model: MujocoModel) -> dict[str, float]:
    """True tube count and per-tube shape, from the control graph.

    ``n_tubes`` counts routed tubes / triangles (the "Ktube" in henneberg
    names), cross-checked against the model's one-per-tube length/perimeter
    equality constraints. Tube length is measured in actuated edges (physical
    truss members the tube runs along). Connector stats describe how many
    control-node instances share each physical node (a node on ``m`` tube
    visits has multiplicity ``m`` and ``m choose 2`` connector edges).
    """
    components = _tube_components(model)
    n_tubes = len(components)
    constraint_count = _tube_constraint_count(model)
    if constraint_count and constraint_count != n_tubes:
        raise ValueError(
            f"Actuated control-edge components ({n_tubes}) disagree with tube "
            f"length/perimeter constraints ({constraint_count})."
        )
    tube_lengths = np.array([component.number_of_edges() for component in components], dtype=float)
    n_closed = sum(
        1 for component in components
        if component.number_of_edges() == component.number_of_nodes()
    )

    control_graph = model.control_graph
    multiplicity: dict[str, int] = {}
    for name in control_graph.control_node_names:
        logical = control_graph.control_node_to_logical_node[name]
        multiplicity[logical] = multiplicity.get(logical, 0) + 1
    multiplicities = np.array(list(multiplicity.values()), dtype=float)
    n_connectors = sum(1 for edge in control_graph.edges if edge.type == "connector")
    return {
        "n_tubes": n_tubes,
        "tube_length_mean": float(tube_lengths.mean()) if n_tubes else 0.0,
        "tube_length_max": float(tube_lengths.max()) if n_tubes else 0.0,
        "closed_tube_fraction": n_closed / n_tubes if n_tubes else 0.0,
        "n_connectors": n_connectors,
        "node_multiplicity_mean": float(multiplicities.mean()) if len(multiplicities) else 0.0,
        "node_multiplicity_max": float(multiplicities.max()) if len(multiplicities) else 0.0,
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


def _clean(value: Any) -> Any:
    """Snap floating-point dust: |x| < 1e-8 -> 0, else round to 10 decimals."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        return value
    value = float(value)
    if abs(value) < _ZERO_ATOL:
        return 0.0
    return round(value, _ROUND_DECIMALS)


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
    features.update(_tube_stats(model))
    features.update(_active_passive_node_stats(model))
    return {key: _clean(value) for key, value in features.items()}


def compute_all_hand_features(topologies: list[str]) -> pd.DataFrame:
    rows = [compute_hand_features(topology) for topology in topologies]
    return pd.DataFrame(rows, columns=[*ID_COLUMNS, *FEATURE_COLUMNS])


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
