import torch
from torch_geometric.data import Data
from torch_geometric.transforms import VirtualNode


_VIRTUAL_NODE = VirtualNode()


def prepare_graph(graph: Data, *, use_virtual_node: bool) -> Data:
    """Clone a graph and optionally append one masked PyG virtual node."""
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
    return _VIRTUAL_NODE(prepared)


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
