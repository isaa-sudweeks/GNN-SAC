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
    prepared.action_mask = torch.ones(num_nodes, dtype=torch.bool, device=prepared.x.device)
    return _VIRTUAL_NODE(prepared)


def physical_node_mask(graph: Data) -> torch.Tensor:
    """Return the node mask used for actions, entropy, and critic pooling."""
    mask = getattr(graph, "action_mask", None)
    if mask is None:
        return torch.ones(graph.x.size(0), dtype=torch.bool, device=graph.x.device)
    return mask.bool()
