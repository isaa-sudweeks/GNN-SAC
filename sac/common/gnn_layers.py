from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

from common.mlp_layers import mlp


def _validate_dims(name: str, dims: Sequence[int], *, allow_empty: bool) -> list[int]:
    """Return a validated copy of an architecture dimension sequence."""
    if isinstance(dims, (str, bytes)) or not isinstance(dims, Sequence):
        raise ValueError(f"{name} must be a sequence of positive integers")
    values = list(dims)
    if not allow_empty and not values:
        raise ValueError(f"{name} must contain at least one dimension")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in values):
        raise ValueError(f"{name} must contain only positive integers")
    return values


def _message_features(x_i, x_j, edge_attr, edge_channels: int):
    if edge_channels == 0:
        if edge_attr is not None and edge_attr.size(-1) != 0:
            raise ValueError("Received edge features for a GNN configured with edge_feature_dim=0.")
        return torch.cat([x_i, x_j], dim=1)
    if edge_attr is None:
        raise ValueError(f"GNN expects {edge_channels} edge features, but edge_attr is missing.")
    if edge_attr.ndim != 2 or edge_attr.size(0) != x_i.size(0):
        raise ValueError("edge_attr must have one row per directed message edge.")
    if edge_attr.size(1) != edge_channels:
        raise ValueError(
            f"GNN expects {edge_channels} edge features, got {edge_attr.size(1)}."
        )
    return torch.cat([x_i, x_j, edge_attr], dim=1)


class _MessagePassingLayer(MessagePassing):
    """One independently parameterized message-and-update block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        dropout: float,
        edge_channels: int,
        message_attention: bool,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.message_attention = bool(message_attention)
        self.phi = mlp(
            in_channels * 2 + edge_channels,
            hidden_channels,
            out_channels,
            dropout=dropout,
        )
        self.gamma = mlp(in_channels + out_channels, hidden_channels, out_channels, dropout=dropout)
        self.attention_score = (
            nn.Linear(in_channels * 2, 1, bias=False) if self.message_attention else None
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        pair = torch.cat([x_i, x_j], dim=1)
        message = self.phi(_message_features(x_i, x_j, edge_attr, self.edge_channels))
        if self.attention_score is None:
            return message
        logits = F.leaky_relu(self.attention_score(pair).squeeze(-1), negative_slope=0.2)
        weights = softmax(logits, index, ptr, num_nodes=size_i)
        return weights.unsqueeze(-1) * message

    def update(self, aggr_out, x):
        return self.gamma(torch.cat([x, aggr_out], dim=1))


class GNN(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: Sequence[int] = (),
        dropout: float = 0.0,
        *,
        mpl_dims: Sequence[int] | None = None,
        skip_connections: bool = True,
        edge_channels: int = 0,
        message_attention: bool = False,
    ):
        super().__init__(aggr="add")
        if mpl_dims is None:
            if out_channels is None:
                raise ValueError("out_channels is required when mpl_dims is not provided")
            mpl_dims = [out_channels]
        self.mpl_dims = _validate_dims("mpl_dims", mpl_dims, allow_empty=False)
        self.hidden_channels = _validate_dims("hidden_channels", hidden_channels, allow_empty=True)
        self.in_channels = in_channels
        self.out_channels = self.mpl_dims[-1]
        self.dropout = dropout
        self.skip_connections = bool(skip_connections)
        self.edge_channels = int(edge_channels)
        if self.edge_channels < 0:
            raise ValueError("edge_channels must be nonnegative")
        self.message_attention = bool(message_attention)

        # Keep these names for compatibility with existing one-layer checkpoints.
        self.phi = mlp(in_channels * 2 + self.edge_channels, self.hidden_channels, self.mpl_dims[0], dropout=dropout)
        self.gamma = mlp(in_channels + self.mpl_dims[0], self.hidden_channels, self.mpl_dims[0], dropout=dropout)
        self.attention_score = (
            nn.Linear(in_channels * 2, 1, bias=False) if self.message_attention else None
        )
        self.extra_mpls = nn.ModuleList(
            _MessagePassingLayer(
                input_dim,
                output_dim,
                self.hidden_channels,
                dropout,
                self.edge_channels,
                self.message_attention,
            )
            for input_dim, output_dim in zip(self.mpl_dims, self.mpl_dims[1:])
        )
        self.skip_projections = nn.ModuleList()
        if self.skip_connections:
            self.skip_projections.extend(
                nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in zip(self.mpl_dims, self.mpl_dims[1:])
            )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        for index, layer in enumerate(self.extra_mpls):
            updated = layer(x, edge_index, edge_attr)
            x = updated + self.skip_projections[index](x) if self.skip_connections else updated
        return x

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        pair = torch.cat([x_i, x_j], dim=1)
        message = self.phi(_message_features(x_i, x_j, edge_attr, self.edge_channels))
        if self.attention_score is None:
            return message
        logits = F.leaky_relu(self.attention_score(pair).squeeze(-1), negative_slope=0.2)
        weights = softmax(logits, index, ptr, num_nodes=size_i)
        return weights.unsqueeze(-1) * message

    def update(self, aggr_out, x):
        return self.gamma(torch.cat([x, aggr_out], dim=1))

    def __repr__(self):
        return (
            f"GNN(in_channels={self.in_channels}, mpl_dims={self.mpl_dims}, "
            f"message_hidden_dims={self.hidden_channels}, dropout={self.dropout}, "
            f"skip_connections={self.skip_connections}, "
            f"edge_channels={self.edge_channels}, "
            f"message_attention={self.message_attention})\n"
            f"Phi(Message Passing):\t{self.phi}\n"
            f"Gamma(Update):\t{self.gamma}\n"
            f"Additional MPLs:\t{self.extra_mpls}"
        )

class Q_GNN(GNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: Sequence[int] = (),
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        *,
        mpl_dims: Sequence[int] | None = None,
        skip_connections: bool = True,
        edge_channels: int = 0,
        critic_readout: str = "physical_mean",
        message_attention: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            hidden_channels,
            dropout,
            mpl_dims=mpl_dims,
            skip_connections=skip_connections,
            edge_channels=edge_channels,
            message_attention=message_attention,
        )
        valid_readouts = {
            "physical_mean",
            "virtual_node",
            "physical_mean_virtual_node",
        }
        if critic_readout not in valid_readouts:
            raise ValueError(
                f"critic_readout must be one of {sorted(valid_readouts)}, "
                f"got {critic_readout!r}"
            )
        self.critic_readout = critic_readout
        self.head_hidden_dims = _validate_dims(
            "head_hidden_dims", head_hidden_dims, allow_empty=True
        )
        head_input_dim = self.out_channels * (
            2 if critic_readout == "physical_mean_virtual_node" else 1
        )
        self.head = mlp(head_input_dim, self.head_hidden_dims, 1)

    def forward(
        self,
        x,
        edge_index,
        batch=None,
        physical_mask=None,
        edge_attr=None,
        num_graphs=None,
    ):
        x = super().forward(x, edge_index, edge_attr)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if physical_mask is None:
            if self.critic_readout != "physical_mean":
                raise ValueError(
                    f"critic_readout={self.critic_readout!r} requires a physical-node mask"
                )
            readout = global_mean_pool(x, batch, size=num_graphs)
        else:
            physical_mask = physical_mask.bool()
            if physical_mask.dim() != 1 or physical_mask.numel() != x.size(0):
                raise ValueError(
                    "The physical-node mask must contain one value per graph node."
                )
            physical_mean = global_mean_pool(
                x[physical_mask], batch[physical_mask], size=num_graphs
            )
            if self.critic_readout == "physical_mean":
                readout = physical_mean
            else:
                # ``prepare_graph`` appends exactly one virtual node to every
                # graph. Batched boolean indexing therefore returns virtual
                # embeddings in graph order without a pooling reduction or a
                # GPU-to-host synchronization in this training hot path.
                virtual_node = x[~physical_mask]
                if self.critic_readout == "virtual_node":
                    readout = virtual_node
                else:
                    readout = torch.cat([physical_mean, virtual_node], dim=-1)
        return self.head(readout).squeeze(-1)

    def __repr__(self):
        return (
            super().__repr__().replace("GNN(", "Q_GNN(", 1)
            + f"\nReadout:\t{self.critic_readout}\nHead:\t{self.head}"
        )
