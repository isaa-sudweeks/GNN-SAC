import torch
from torch_geometric.data import Data
from torch_geometric.transforms import VirtualNode


_VIRTUAL_NODE = VirtualNode()
VIRTUAL_NODE_CONTEXT_DIM = 2


def graph_input_dim(node_feature_dim: int, *, use_virtual_node: bool) -> int:
    """Return the GNN input width after optional virtual-node context."""
    return int(node_feature_dim) + (VIRTUAL_NODE_CONTEXT_DIM if use_virtual_node else 0)


def prepare_graph(graph: Data, *, use_virtual_node: bool) -> Data:
    """Clone a graph and optionally append one rigidity-aware virtual node.

    Physical nodes receive two zero-valued context channels. The appended
    virtual node receives an ``is_virtual`` flag and the graph's normalized
    rigidity ratio, allowing global structural health to reach every physical
    node through one message-passing layer.
    """
    if not use_virtual_node:
        return graph

    prepared = graph.clone()
    num_nodes = prepared.num_nodes
    if num_nodes is None:
        raise ValueError("Virtual-node augmentation requires a known graph node count.")
    if getattr(prepared, "action_mask", None) is None:
        prepared.action_mask = torch.ones(
            num_nodes, dtype=torch.bool, device=prepared.x.device
        )
    prepared.physical_node_mask = torch.ones(
        num_nodes, dtype=torch.bool, device=prepared.x.device
    )
    context = prepared.x.new_zeros((num_nodes, VIRTUAL_NODE_CONTEXT_DIM))
    prepared.x = torch.cat([prepared.x, context], dim=-1)
    prepared = _VIRTUAL_NODE(prepared)

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
