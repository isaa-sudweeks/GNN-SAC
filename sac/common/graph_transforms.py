import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import VirtualNode


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
    }


def graph_feature_schema(config) -> dict[str, object]:
    """Return the checkpointed feature contract for a GNN policy."""
    flags = graph_feature_flags(config)
    return {
        "node_roles": flags["use_node_roles"],
        "edge_roles": flags["use_edge_roles"],
        "edge_distance": flags["use_edge_distance"],
        "edge_role_vocabulary": list(EDGE_ROLE_NAMES),
    }


def graph_input_dim(
    node_feature_dim: int,
    *,
    use_virtual_node: bool,
    use_node_roles: bool = False,
) -> int:
    """Return the GNN input width after optional virtual-node context."""
    return (
        int(node_feature_dim)
        + (NODE_ROLE_DIM if use_node_roles else 0)
        + (VIRTUAL_NODE_CONTEXT_DIM if use_virtual_node else 0)
    )


def graph_edge_input_dim(*, use_edge_roles: bool, use_edge_distance: bool) -> int:
    """Return the configured message-edge feature width."""
    return (EDGE_ROLE_DIM if use_edge_roles else 0) + int(use_edge_distance)


def _physical_edge_features(
    graph: Data,
    *,
    use_edge_roles: bool,
    use_edge_distance: bool,
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
        features.append(F.one_hot(roles, num_classes=EDGE_ROLE_DIM).to(dtype=graph.x.dtype))

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


def prepare_graph(
    graph: Data,
    *,
    use_virtual_node: bool,
    use_node_roles: bool = False,
    use_edge_roles: bool = False,
    use_edge_distance: bool = False,
) -> Data:
    """Clone a graph and optionally append one rigidity-aware virtual node.

    Physical nodes receive two zero-valued context channels. The appended
    virtual node receives an ``is_virtual`` flag and the graph's normalized
    rigidity ratio, allowing global structural health to reach every physical
    node through one message-passing layer.
    """
    if not (use_virtual_node or use_node_roles or use_edge_roles or use_edge_distance):
        return graph

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
    )
    if edge_attr is not None:
        prepared.edge_attr = edge_attr
    if "edge_role" in prepared:
        del prepared.edge_role

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
