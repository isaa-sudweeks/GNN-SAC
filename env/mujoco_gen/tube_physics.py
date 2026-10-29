"""Validated route-length metadata for reversible six-channel observations."""
from collections import Counter

import mujoco
import numpy as np


TUBE_PHYSICS_FIELDS = ("tube_position_scale", "tube_reference_length", "tube_segment_weight")


def tube_physics_enabled(config: object) -> bool:
    features = config.get("graph_features", {}) if hasattr(config, "get") else getattr(config, "graph_features", {})
    return bool(features.get("tube_physics", False))


def build_tube_physics_metadata(source: object, edge_index: np.ndarray,
                               edge_role: np.ndarray, *, normalized: bool) -> dict[str, np.ndarray]:
    """Match structural components to compiled, fixed-reference route tendons.

    Supports site-at-body-origin abstract models. Realistic connector-ball
    observations cannot reconstruct tendon lengths and are rejected explicitly.
    Both directions receive half the route occurrence weight, preserving repeated
    segments without counting reverse message edges twice.
    """
    model, metadata = source.model, source.control_graph
    if not metadata.enabled or getattr(source, "realistic", False):
        raise ValueError("Tube physics requires an abstract control graph.")
    names = metadata.control_node_names
    physical = [metadata.control_node_to_physical_node[n] for n in names]
    edges, roles = np.asarray(edge_index), np.asarray(edge_role)
    adjacency = [set() for _ in names]
    for a, b in edges[:, roles == 0].T:
        adjacency[a].add(int(b)); adjacency[b].add(int(a))
    groups, unseen = [], set(range(len(names)))
    while unseen:
        stack, group = [min(unseen)], set()
        while stack:
            node = stack.pop()
            if node not in group:
                group.add(node); stack.extend(adjacency[node] - group)
        unseen -= group
        groups.append(sorted(group))
    routes = []
    for eq in range(model.neq):
        if int(model.eq_type[eq]) != int(mujoco.mjtEq.mjEQ_TENDON):
            continue
        tendon, second = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        perimeter = model.equality(eq).name.startswith("Perimeter_Constraint_")
        if not model.tendon(tendon).name.startswith("route_") and not perimeter:
            continue
        expected = [0, -1, 0, 0, 0] if perimeter else [0, 0, 0, 0, 0]
        if (not model.eq_active0[eq] or not np.allclose(model.eq_data[eq, :5], expected)
                or (perimeter and second < 0) or (not perimeter and second != -1)):
            raise ValueError("Unsupported route equality reference contract.")
        pairs, reference = Counter(), 0.0
        for tendon_id in ([tendon, second] if perimeter else [tendon]):
            start, count = int(model.tendon_adr[tendon_id]), int(model.tendon_num[tendon_id])
            if not np.all(model.wrap_type[start:start + count] == int(mujoco.mjtWrap.mjWRAP_SITE)):
                raise ValueError("Tube physics requires site-only route tendons.")
            sites = model.wrap_objid[start:start + count]
            if not np.allclose(model.site_pos[sites], 0):
                raise ValueError("Tube physics cannot recover offset tendon sites from node positions.")
            site_names = [model.site(int(i)).name for i in sites]
            pairs.update(tuple(sorted((a, b))) for a, b in zip(site_names, site_names[1:]))
            reference += float(model.tendon_length0[tendon_id])
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Route reference length must be finite and positive.")
        routes.append((pairs, reference))
    refs, weights = np.zeros(len(names), np.float32), np.zeros(edges.shape[1], np.float32)
    for group in groups:
        selected = [i for i, (a, b) in enumerate(edges.T) if roles[i] == 0 and int(a) in group and a < b]
        pairs = Counter(tuple(sorted((physical[edges[0, i]], physical[edges[1, i]]))) for i in selected)
        # Message graphs may deduplicate a segment traversed repeatedly by the
        # physical route. Recover multiplicity from the compiled tendon, not the
        # graph degree or number of unique members.
        matches = [i for i, (route, _) in enumerate(routes) if set(route) == set(pairs)]
        if not matches:
            raise ValueError("Structural tube does not match a compiled constrained route.")
        if any(routes[i] != routes[matches[0]] for i in matches):
            raise ValueError("Ambiguous physical route match for structural tube.")
        route, reference = routes.pop(matches[0])
        refs[group] = reference
        for i, (a, b) in enumerate(edges.T):
            if roles[i] == 0 and int(a) in group:
                pair = tuple(sorted((physical[a], physical[b])))
                weights[i] = route[pair] / (2 * pairs[pair])
    if routes:
        raise ValueError("Unmatched compiled route constraints.")
    scale = np.asarray(source.initial_bounding_box_dimensions if normalized else np.ones(3), dtype=np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid inverse position normalization scale.")
    return dict(tube_position_scale=scale.reshape(1, 3),
                tube_reference_length=refs, tube_segment_weight=weights)
