import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import VirtualNode

from env.mujoco_gen.tube_physics import TUBE_PHYSICS_FIELDS

_VIRTUAL_NODE = VirtualNode()
VIRTUAL_NODE_CONTEXT_DIM = 2
NODE_ROLE_DIM = 2
EDGE_ROLE_NAMES = ("tube", "connector", "virtual")
EDGE_ROLE_DIM = len(EDGE_ROLE_NAMES)
VIRTUAL_EDGE_ROLE = EDGE_ROLE_NAMES.index("virtual")


def _cfg_get(config, name: str, default=None):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def graph_feature_flags(config) -> dict[str, bool]:
    """Return normalized graph feature switches from a training configuration."""
    features = _cfg_get(config, "graph_features", {})
    return {
        "use_node_roles": bool(_cfg_get(features, "node_roles", False)),
        "use_edge_roles": bool(_cfg_get(features, "edge_roles", False)),
        "use_edge_distance": bool(_cfg_get(features, "edge_distance", False)),
        "use_tube_nodes": bool(_cfg_get(features, "tube_nodes", False)),
        "use_tube_physics": bool(_cfg_get(features, "tube_physics", False)),
        "use_edge_direction": bool(_cfg_get(features, "edge_direction", False)),
    }


def graph_feature_schema(config) -> dict[str, object]:
    """Return the checkpointed feature contract for a GNN policy."""
    flags = graph_feature_flags(config)
    schema = {
        "node_roles": flags["use_node_roles"],
        "edge_roles": flags["use_edge_roles"],
        "edge_distance": flags["use_edge_distance"],
        "edge_role_vocabulary": list(EDGE_ROLE_NAMES),
    }
    # Preserve the exact legacy schema for checkpoints without signed routing.
    if flags["use_edge_direction"]:
        schema["edge_direction"] = "controller_incidence_v1"
    if flags["use_tube_nodes"]:
        schema["tube_nodes"] = "structural_components_v1"
        schema["edge_role_vocabulary"] = [*EDGE_ROLE_NAMES, "membership"]
    if flags["use_tube_physics"]:
        schema["tube_physics"] = "route_segments_reference_m_ratio_relative_residual_v1"
    return schema


def graph_input_dim(
    node_feature_dim: int,
    *,
    use_virtual_node: bool,
    use_node_roles: bool = False,
    use_tube_nodes: bool = False,
    use_tube_physics: bool = False,
) -> int:
    """Return the GNN input width after optional virtual-node context."""
    return (
        int(node_feature_dim)
        + int(use_tube_nodes)
        + 4 * int(use_tube_physics)
        + (NODE_ROLE_DIM if use_node_roles else 0)
        + (VIRTUAL_NODE_CONTEXT_DIM if use_virtual_node else 0)
    )


def graph_edge_input_dim(*, use_edge_roles: bool, use_edge_distance: bool,
                         use_edge_direction: bool = False, use_tube_nodes: bool = False) -> int:
    """Return the configured message-edge feature width."""
    return ((EDGE_ROLE_DIM + int(use_tube_nodes)) if use_edge_roles else 0) + int(use_edge_direction) + int(use_edge_distance)


def _physical_edge_features(
    graph: Data,
    *,
    use_edge_roles: bool,
    use_edge_distance: bool,
    use_edge_direction: bool = False,
    use_tube_nodes: bool = False,
) -> torch.Tensor | None:
    edge_count = int(graph.edge_index.size(1))
    features = []
    if use_edge_roles:
        roles = getattr(graph, "edge_role", None)
        if roles is None:
            raise ValueError("graph_features.edge_roles=true requires graph.edge_role metadata.")
        roles = torch.as_tensor(roles, device=graph.edge_index.device).reshape(-1)
        if roles.numel() != edge_count:
            raise ValueError(
                f"Graph has {edge_count} directed edges but {roles.numel()} edge-role labels."
            )
        if roles.is_floating_point() and not torch.equal(roles, roles.round()):
            raise ValueError("Edge-role labels must be integer indices.")
        roles = roles.long()
        if roles.numel() and (int(roles.min()) < 0 or int(roles.max()) >= EDGE_ROLE_DIM - 1):
            raise ValueError("Raw edge roles must be tube=0 or connector=1; virtual edges are added internally.")
        features.append(F.one_hot(roles, num_classes=EDGE_ROLE_DIM + int(use_tube_nodes)).to(dtype=graph.x.dtype))

    if use_edge_direction:
        direction = getattr(graph, "edge_direction", None)
        if direction is None:
            raise ValueError("graph_features.edge_direction=true requires graph.edge_direction metadata.")
        direction = torch.as_tensor(direction, device=graph.x.device).reshape(-1)
        if direction.numel() != edge_count or not torch.isfinite(direction).all():
            raise ValueError("Invalid edge-direction metadata length or nonfinite values.")
        if not ((direction == -1) | (direction == 0) | (direction == 1)).all():
            raise ValueError("Edge directions must be -1, 0, or +1.")
        features.append(direction.to(graph.x.dtype).unsqueeze(-1))

    if use_edge_distance:
        if graph.x.ndim != 2 or graph.x.size(1) < 3:
            raise ValueError("Edge distance requires at least three xyz node features.")
        source, target = graph.edge_index
        distance = torch.linalg.vector_norm(
            graph.x[source, :3] - graph.x[target, :3], dim=-1, keepdim=True
        )
        features.append(distance)

    if not features:
        return None
    return torch.cat(features, dim=-1) if len(features) > 1 else features[0]


def tube_components(graph: Data) -> list[list[int]]:
    """Recover control instances per tube from all structural edges, including passive ones.

    Connector edges couple distinct instances and must never merge their tubes.
    Component labels are internal only and are never passed as numeric features.
    """
    roles = getattr(graph, "edge_role", None)
    if roles is None or roles.numel() != graph.edge_index.size(1):
        raise ValueError("Tube nodes require one raw edge_role per directed control edge.")
    if not bool(((roles == 0) | (roles == 1)).all()):
        raise ValueError("Tube membership requires raw tube/connector roles.")
    parents = list(range(graph.num_nodes))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for a, b in graph.edge_index[:, roles == 0].detach().cpu().t().tolist():
        parents[root(b)] = root(a)
    groups = {}
    for i in range(graph.num_nodes):
        groups.setdefault(root(i), []).append(i)
    result = list(groups.values())
    if any(len(group) < 2 for group in result):
        raise ValueError("Control nodes without a structural tube cannot receive a tube hub.")
    return result


def tube_physics_features(x: torch.Tensor, edge_index: torch.Tensor,
                          groups: list[list[int]], metadata: Data | dict) -> torch.Tensor:
    """Compute [..., tubes, 4] features after undoing xyz normalization.

    Reference length uses a fixed one-metre scale; residual is (L-L0)/L0.
    Static route weights count each segment once across bidirectional edges.
    """
    scale, reference, weights = [torch.as_tensor(metadata[k], device=x.device, dtype=x.dtype)
                                 for k in TUBE_PHYSICS_FIELDS]
    if (scale.shape != (1, 3) or reference.shape != (x.size(-2),)
            or weights.shape != (edge_index.size(1),)):
        raise ValueError("Invalid tube physics metadata shape.")
    if (not all(bool(torch.isfinite(v).all()) for v in (scale, reference, weights))
            or bool((scale <= 0).any()) or bool((reference <= 0).any())
            or bool((weights < 0).any())):
        raise ValueError("Invalid tube physics metadata values.")
    position = x[..., :3] * scale
    lengths = torch.linalg.vector_norm(position[..., edge_index[0], :] - position[..., edge_index[1], :], dim=-1)
    result = []
    for group in groups:
        ref = reference[group[0]]
        if not torch.allclose(reference[group], ref.expand(len(group))):
            raise ValueError("Tube members disagree on reference length.")
        members = torch.zeros(x.size(-2), dtype=torch.bool, device=x.device)
        members[group] = True
        mask = members[edge_index[0]] & members[edge_index[1]] & (weights > 0)
        count = weights[mask].sum()
        if count <= 0:
            raise ValueError("Tube has no route segments.")
        current = (lengths[..., mask] * weights[mask]).sum(-1)
        ratio = current / ref
        result.append(torch.stack((torch.ones_like(ratio) * count,
                                   torch.ones_like(ratio) * ref, ratio, ratio - 1), -1))
    return torch.stack(result, dim=-2)


def prepare_graph(
    graph: Data,
    *,
    use_virtual_node: bool,
    use_node_roles: bool = False,
    use_edge_roles: bool = False,
    use_edge_distance: bool = False,
    use_edge_direction: bool = False,
    use_tube_nodes: bool = False,
    use_tube_physics: bool = False,
) -> Data:
    """Add global context and optional typed tube hubs to a raw physical graph.

    The global node carries rigidity and only connects raw physical instances.
    Tube hubs connect their own structural component, excluding connectors.
    Architectural edges carry no geometric distance or controller incidence.
    Legacy feature widths and schemas remain unchanged when tube hubs are off.
    """
    if not (use_virtual_node or use_node_roles or use_edge_roles or use_edge_distance or use_edge_direction or use_tube_nodes or use_tube_physics):
        return graph

    if use_tube_physics and not use_tube_nodes:
        raise ValueError("Tube physics requires graph_features.tube_nodes=true.")
    if use_tube_nodes and not (use_virtual_node and use_edge_roles):
        raise ValueError("Tube nodes require use_virtual_node=true and graph_features.edge_roles=true.")
    groups = tube_components(graph) if use_tube_nodes else []
    physics = None
    if use_tube_physics:
        if any(getattr(graph, k, None) is None for k in TUBE_PHYSICS_FIELDS):
            raise ValueError("Tube physics requires validated simulator route metadata.")
        physics = tube_physics_features(graph.x, graph.edge_index, groups, graph)
    prepared = graph.clone()
    num_nodes = prepared.num_nodes
    if num_nodes is None:
        raise ValueError("Graph feature augmentation requires a known graph node count.")
    if prepared.edge_index.device != prepared.x.device:
        prepared.edge_index = prepared.edge_index.to(prepared.x.device)
    if getattr(prepared, "action_mask", None) is None:
        prepared.action_mask = torch.ones(
            num_nodes, dtype=torch.bool, device=prepared.x.device
        )
    else:
        prepared.action_mask = prepared.action_mask.to(
            device=prepared.x.device, dtype=torch.bool
        )
    if use_node_roles:
        action_mask = prepared.action_mask.bool()
        if action_mask.numel() != num_nodes:
            raise ValueError(
                f"Graph has {num_nodes} nodes but {action_mask.numel()} action-mask entries."
            )
        node_roles = torch.stack((action_mask, ~action_mask), dim=-1).to(prepared.x.dtype)
        prepared.x = torch.cat((prepared.x, node_roles), dim=-1)

    edge_attr = _physical_edge_features(
        prepared,
        use_edge_roles=use_edge_roles,
        use_edge_distance=use_edge_distance,
        use_edge_direction=use_edge_direction,
        use_tube_nodes=use_tube_nodes,
    )
    if edge_attr is not None:
        prepared.edge_attr = edge_attr
    for field in ("edge_role", "edge_direction", *TUBE_PHYSICS_FIELDS):
        if field in prepared:
            del prepared[field]

    if use_tube_nodes:
        if use_tube_physics:
            prepared.x = torch.cat((prepared.x, prepared.x.new_zeros((num_nodes, 4))), dim=-1)
        prepared.x = torch.cat((prepared.x, prepared.x.new_zeros((num_nodes, 1))), dim=-1)

    if not use_virtual_node:
        return prepared

    prepared.physical_node_mask = torch.ones(
        num_nodes, dtype=torch.bool, device=prepared.x.device
    )
    context = prepared.x.new_zeros((num_nodes, VIRTUAL_NODE_CONTEXT_DIM))
    prepared.x = torch.cat([prepared.x, context], dim=-1)
    physical_edge_count = int(prepared.edge_index.size(1))
    prepared = _VIRTUAL_NODE(prepared)

    if use_edge_roles:
        prepared.edge_attr[physical_edge_count:, VIRTUAL_EDGE_ROLE] = 1.0

    rigidity = getattr(graph, "rigidity", None)
    if rigidity is None:
        rigidity_value = prepared.x.new_zeros(())
    else:
        rigidity_value = torch.as_tensor(
            rigidity, dtype=prepared.x.dtype, device=prepared.x.device
        ).reshape(-1)
        if rigidity_value.numel() != 1:
            raise ValueError(
                "Virtual-node rigidity must contain exactly one scalar per graph."
            )
        rigidity_value = rigidity_value[0]
    prepared.x[-1, -2] = 1.0
    prepared.x[-1, -1] = rigidity_value
    if use_tube_nodes:
        # Append hubs AFTER the global node so it only connects physical instances.
        global_index = num_nodes
        prepared.global_node_mask = torch.arange(prepared.num_nodes, device=prepared.x.device) == global_index
        hubs = prepared.x.new_zeros((len(groups), prepared.x.size(1)))
        hubs[:, -3] = 1.0  # dedicated tube-type channel, before global context
        if physics is not None:
            hubs[:, -7:-3] = physics
        prepared.x = torch.cat((prepared.x, hubs), dim=0)
        for field in ("action_mask", "physical_node_mask", "global_node_mask"):
            prepared[field] = torch.cat((prepared[field], torch.zeros(len(groups), dtype=torch.bool, device=prepared.x.device)))
        members, targets = [], []
        for tube, group in enumerate(groups):
            members.extend(group)
            targets.extend([num_nodes + 1 + tube] * len(group))
        membership = torch.tensor([members, targets], dtype=torch.long, device=prepared.x.device)
        membership = torch.cat((membership, membership.flip(0)), dim=1)
        prepared.edge_index = torch.cat((prepared.edge_index, membership), dim=1)
        attr = prepared.edge_attr.new_zeros((membership.size(1), prepared.edge_attr.size(1)))
        attr[:, EDGE_ROLE_DIM] = 1.0
        prepared.edge_attr = torch.cat((prepared.edge_attr, attr), dim=0)
        if "edge_type" in prepared:
            prepared.edge_type = torch.cat((prepared.edge_type, prepared.edge_type.new_full((membership.size(1),), int(prepared.edge_type.max()) + 1)))
        prepared.num_nodes = prepared.x.size(0)
    return prepared


def physical_node_mask(graph: Data) -> torch.Tensor:
    """Return a mask that excludes architectural (virtual) nodes."""
    mask = getattr(graph, "physical_node_mask", None)
    if mask is None:
        return torch.ones(graph.x.size(0), dtype=torch.bool, device=graph.x.device)
    return mask.bool()


def policy_action_mask(graph: Data) -> torch.Tensor:
    """Return the actuated-node mask used for policy actions and entropy."""
    mask = getattr(graph, "action_mask", None)
    if mask is None:
        return physical_node_mask(graph)
    return mask.bool()


def graph_structure_signature(graph: Data) -> dict:
    """Return the topology and policy action-order contract for one graph."""
    mask = policy_action_mask(graph)
    if mask.ndim != 1 or mask.numel() != graph.x.size(0) or not mask.any():
        raise ValueError("Invalid teacher action mask.")
    signature = {
        "shape": list(graph.x.shape),
        "edges": graph.edge_index.detach().cpu().tolist(),
        "mask": mask.detach().cpu().tolist(),
    }
    if getattr(graph, "edge_direction", None) is not None:
        signature["edge_direction"] = graph.edge_direction.detach().cpu().tolist()
    for field in TUBE_PHYSICS_FIELDS:
        if getattr(graph, field, None) is not None:
            signature[field] = graph[field].detach().cpu().tolist()
    return signature


def graph_signature_compatible(
    candidate: dict,
    reference: dict,
    *,
    allow_action_subset: bool = False,
) -> bool:
    """Return whether a graph signature preserves a reference action contract.

    Broken-node randomization may turn base-active nodes off without changing
    topology or action row ordering. It must never turn a reference-passive
    node on.
    """
    if candidate.get("shape") != reference.get("shape"):
        return False
    if candidate.get("edge_direction") != reference.get("edge_direction"):
        return False
    if any(candidate.get(k) != reference.get(k) for k in TUBE_PHYSICS_FIELDS):
        return False
    if candidate.get("edges") != reference.get("edges"):
        return False
    candidate_mask = torch.as_tensor(candidate.get("mask", ()), dtype=torch.bool)
    reference_mask = torch.as_tensor(reference.get("mask", ()), dtype=torch.bool)
    if candidate_mask.shape != reference_mask.shape or not bool(candidate_mask.any()):
        return False
    if allow_action_subset:
        return not bool((candidate_mask & ~reference_mask).any())
    return torch.equal(candidate_mask, reference_mask)
