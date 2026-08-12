from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

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

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: list[int], dropout: float):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.phi = mlp(in_channels * 2, hidden_channels, out_channels, dropout=dropout)
        self.gamma = mlp(in_channels + out_channels, hidden_channels, out_channels, dropout=dropout)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.phi(torch.cat([x_i, x_j], dim=1))

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

        # Keep these names for compatibility with existing one-layer checkpoints.
        self.phi = mlp(in_channels * 2, self.hidden_channels, self.mpl_dims[0], dropout=dropout)
        self.gamma = mlp(in_channels + self.mpl_dims[0], self.hidden_channels, self.mpl_dims[0], dropout=dropout)
        self.extra_mpls = nn.ModuleList(
            _MessagePassingLayer(input_dim, output_dim, self.hidden_channels, dropout)
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

    def message(self, x_i, x_j):
        return self.phi(torch.cat([x_i, x_j], dim=1))

    def update(self, aggr_out, x):
        return self.gamma(torch.cat([x, aggr_out], dim=1))

    def __repr__(self):
        return (
            f"GNN(in_channels={self.in_channels}, mpl_dims={self.mpl_dims}, "
            f"message_hidden_dims={self.hidden_channels}, dropout={self.dropout}, "
            f"skip_connections={self.skip_connections})\n"
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
        critic_readout: str = "physical_mean",
    ):
        super().__init__(
            in_channels,
            out_channels,
            hidden_channels,
            dropout,
            mpl_dims=mpl_dims,
            skip_connections=skip_connections,
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

    def forward(self, x, edge_index, batch=None, physical_mask=None):
        x = super().forward(x, edge_index)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if physical_mask is None:
            if self.critic_readout != "physical_mean":
                raise ValueError(
                    f"critic_readout={self.critic_readout!r} requires a physical-node mask"
                )
            readout = global_mean_pool(x, batch)
        else:
            physical_mask = physical_mask.bool()
            if physical_mask.dim() != 1 or physical_mask.numel() != x.size(0):
                raise ValueError(
                    "The physical-node mask must contain one value per graph node."
                )
            physical_mean = global_mean_pool(
                x[physical_mask], batch[physical_mask]
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
