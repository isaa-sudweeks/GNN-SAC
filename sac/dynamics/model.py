"""Observable-input graph dynamics with explicit centered-xy translation tracking."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


def _mlp(in_features: int, out_features: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, width), nn.SiLU(), nn.Linear(width, out_features))


class ObservableDynamicsGNN(nn.Module):
    """Predict physical state changes, refreshing geometry at every recursive step.

    Graph nodes may alias the same physical node. Loss and translation use one
    representative per physical node; predicted aliases are refreshed from their
    representative before the next step. No actuator activation is consumed.
    """

    def __init__(self, width: int = 128, layers: int = 3) -> None:
        super().__init__()
        if width <= 0 or layers <= 0:
            raise ValueError("width and layers must be positive")
        self.width, self.layers = width, layers
        # x/v, known command, action availability, alias multiplicity.
        self.encoder = _mlp(12, width, width)
        # relative xyz, length, static role and initial length.
        self.messages = nn.ModuleList(_mlp(2 * width + 6, width, width) for _ in range(layers))
        self.updates = nn.ModuleList(_mlp(2 * width, width, width) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.decoder = _mlp(width, 6, width)
        self.register_buffer("state_mean", torch.zeros(6))
        self.register_buffer("state_scale", torch.ones(6))
        self.register_buffer("delta_scale", torch.ones(6))
        self.register_buffer("geometry_scale", torch.ones(()))

    def configure_normalization(self, mean: torch.Tensor, scale: torch.Tensor,
                                delta_scale: torch.Tensor, geometry_scale: float) -> None:
        """Install fitting-only physical-unit statistics, frozen for evaluation."""
        if min(float(scale.min()), float(delta_scale.min()), geometry_scale) <= 0:
            raise ValueError("normalization scales must be positive")
        with torch.no_grad():
            self.state_mean.copy_(mean)
            self.state_scale.copy_(scale)
            self.delta_scale.copy_(delta_scale)
            self.geometry_scale.fill_(geometry_scale)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return per-node displacement and velocity delta in physical units.

        `state` is (B,N,6), centered in xy; position deltas include global progress.
        Static graph attributes are unchanged, while edge geometry uses `state`.
        """
        return self.decoder(self.node_features(state, action, graph)) * self.delta_scale

    def node_features(self, state: torch.Tensor, action: torch.Tensor,
                      graph: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode observable geometry and known commands for task prediction heads."""
        if state.ndim != 3 or state.shape[-1] != 6 or action.shape != (*state.shape[:2], 1):
            raise ValueError("Expected state (B,N,6) and action (B,N,1)")
        edge_index = graph["edge_index"]
        src, dst = edge_index[0], edge_index[1]
        batch, nodes = state.shape[:2]
        mask = graph["action_mask"].to(state).reshape(1, nodes, 1).expand(batch, -1, -1)
        multiplicity = graph.get("multiplicity", torch.ones(nodes, device=state.device)).to(state)
        alias = multiplicity.reciprocal().reshape(1, nodes, 1).expand(batch, -1, -1)
        known = action * mask
        incidence = graph.get("routing_incidence")
        if incidence is not None:
            route = incidence.to(state)
            speed = graph["speed"].to(state)
            commands = (known[..., 0] @ route.T) * speed
            ranges = graph["ctrlrange"].to(state)
            commands = torch.maximum(torch.minimum(commands, ranges[:, 1]), ranges[:, 0]) / speed
            weights = route.abs()
            count = weights.sum(0).clamp_min(1)
            routed = commands @ weights / count
            magnitude = commands.abs() @ weights / count
            degree_feature = weights.sum(0)[None].expand(batch, -1)
            route_features = torch.stack((routed, magnitude, degree_feature), dim=-1)
        else:
            route_features = torch.zeros((batch, nodes, 3), device=state.device, dtype=state.dtype)
        h = self.encoder(torch.cat(((state - self.state_mean) / self.state_scale,
                                    known, mask, alias, route_features), dim=-1))
        relative = (state[:, dst, :3] - state[:, src, :3]) / self.geometry_scale
        length = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        edge_role = graph.get("edge_role", torch.zeros(len(src), device=state.device)).to(state)
        initial_length = graph.get("initial_edge_length", torch.zeros(len(src), device=state.device)).to(state)
        fixed = torch.stack((edge_role, initial_length / self.geometry_scale), dim=-1)
        edge = torch.cat((relative, length, fixed[None].expand(batch, -1, -1)), dim=-1)
        degree = torch.bincount(dst, minlength=nodes).clamp_min(1).to(state).reshape(1, nodes, 1)
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            messages = message(torch.cat((h[:, src], h[:, dst], edge), dim=-1))
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(1, dst, messages)
            h = norm(h + update(torch.cat((h, aggregated / degree), dim=-1)))
        return h

    def step(self, state: torch.Tensor, action: torch.Tensor,
             graph: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance imagined state and return physical-node-mean xy progress."""
        delta = self(state, action, graph)
        result = state + delta
        physical = graph["physical_mask"].bool()
        progress = result[:, physical, :2].mean(dim=1)
        centered = result.clone()
        centered[:, :, :2] -= progress[:, None]
        representatives = graph.get("representative_index")
        if representatives is not None:
            centered = centered[:, representatives.long()]
        return centered, progress


def physical_graph_loss(prediction: torch.Tensor, target: torch.Tensor,
                        physical_mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Average features/nodes inside each graph, then average graphs equally."""
    if not bool(physical_mask.any()):
        raise ValueError("No physical nodes in graph")
    return (((prediction - target) / scale)[:, physical_mask.bool()] ** 2).mean(dim=(1, 2)).mean()
