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


class _MessagePassingLayer(MessagePassing):
    """One independently parameterized message-and-update block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        dropout: float,
        message_attention: bool,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_attention = bool(message_attention)
        self.phi = mlp(in_channels * 2, hidden_channels, out_channels, dropout=dropout)
        self.gamma = mlp(in_channels + out_channels, hidden_channels, out_channels, dropout=dropout)
        self.attention_score = (
            nn.Linear(in_channels * 2, 1, bias=False) if self.message_attention else None
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index, ptr, size_i):
        pair = torch.cat([x_i, x_j], dim=1)
        message = self.phi(pair)
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
        self.message_attention = bool(message_attention)

        # Keep these names for compatibility with existing one-layer checkpoints.
        self.phi = mlp(in_channels * 2, self.hidden_channels, self.mpl_dims[0], dropout=dropout)
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

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        for index, layer in enumerate(self.extra_mpls):
            updated = layer(x, edge_index)
            x = updated + self.skip_projections[index](x) if self.skip_connections else updated
        return x

    def message(self, x_i, x_j, index, ptr, size_i):
        pair = torch.cat([x_i, x_j], dim=1)
        message = self.phi(pair)
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
        message_attention: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            hidden_channels,
            dropout,
            mpl_dims=mpl_dims,
            skip_connections=skip_connections,
            message_attention=message_attention,
        )
        self.head_hidden_dims = _validate_dims("head_hidden_dims", head_hidden_dims, allow_empty=True)
        self.head = mlp(self.out_channels, self.head_hidden_dims, 1)

    def forward(self, x, edge_index, batch=None, action_mask=None):
        x = super().forward(x, edge_index)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if action_mask is not None:
            x = x[action_mask]
            batch = batch[action_mask]
        return self.head(global_mean_pool(x, batch)).squeeze(-1)

    def __repr__(self):
        return (
            super().__repr__().replace("GNN(", "Q_GNN(", 1)
            + f"\nPooling:\tglobal_mean_pool\nHead:\t{self.head}"
        )
