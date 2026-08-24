from copy import deepcopy

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from common import math
from common import mlp_layers as layers
from common.graph_transforms import graph_feature_flags


class PaddedMLPActorCritic(nn.Module):
    """Fixed-width MLP actor and critics over padded graph node observations.

    PyG ``Data`` and ``Batch`` objects are used only to carry variable-sized
    observations. Connectivity is intentionally ignored so this remains a flat
    MLP baseline rather than a graph network.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if bool(getattr(cfg, "use_virtual_node", False)):
            raise ValueError("padded_mlp requires use_virtual_node=false; rigidity is packed directly.")
        enabled_graph_features = [
            name for name, enabled in graph_feature_flags(cfg).items() if enabled
        ]
        if enabled_graph_features:
            raise ValueError(
                "padded_mlp requires graph node/edge features to be disabled; got "
                + ", ".join(enabled_graph_features)
            )
        self.max_nodes = int(cfg.padded_mlp_max_nodes)
        self.node_feature_dim = int(cfg.obs_dim)
        self.node_action_dim = int(cfg.node_action_dim)
        if self.max_nodes <= 0:
            raise ValueError("padded_mlp_max_nodes must be positive.")
        if self.node_feature_dim <= 0 or self.node_action_dim <= 0:
            raise ValueError("Padded MLP node feature and action dimensions must be positive.")

        hidden = list(getattr(cfg, "padded_mlp_hidden_dims", 2 * [cfg.mlp_dim]))
        self.observation_dim = self.max_nodes * (self.node_feature_dim + 2) + 1
        self.padded_action_dim = self.max_nodes * self.node_action_dim

        self._pi = layers.mlp(
            self.observation_dim,
            hidden,
            2 * self.padded_action_dim,
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    self.observation_dim + self.padded_action_dim,
                    hidden,
                    1,
                    dropout=cfg.dropout,
                )
                for _ in range(int(cfg.num_q))
            ]
        )
        self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min)
        )

    @property
    def total_params(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def actor_parameters(self):
        return tuple(self._pi.parameters())

    def __repr__(self):
        return (
            "Padded MLP Soft Actor Critic Network\n"
            f"Actor: {self._pi}\n"
            f"Critics: {self._Qs}\n"
            f"Learnable parameters: {self.total_params:,}"
        )

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    @torch.no_grad()
    def soft_update_target_Q(self):
        for parameter, target_parameter in zip(
            self._Qs.parameters(), self._target_Qs.parameters()
        ):
            target_parameter.data.lerp_(parameter.data, self.cfg.tau)

    def _batch_index(self, obs):
        batch = getattr(obs, "batch", None)
        if batch is None:
            return torch.zeros(obs.x.size(0), dtype=torch.long, device=obs.x.device)
        return batch

    def _masks(self, obs):
        batch = self._batch_index(obs)
        counts = torch.bincount(batch)
        if counts.numel() == 0:
            raise ValueError("Padded MLP observations must contain at least one node.")
        actual_max = int(counts.max())
        if actual_max > self.max_nodes:
            raise ValueError(
                f"Graph has {actual_max} nodes but padded_mlp_max_nodes={self.max_nodes}."
            )

        _, physical_mask = to_dense_batch(
            obs.x.new_ones((obs.x.size(0), 1)),
            batch,
            max_num_nodes=self.max_nodes,
        )
        action_mask = getattr(obs, "action_mask", None)
        if action_mask is None:
            action_mask = torch.ones(obs.x.size(0), dtype=torch.bool, device=obs.x.device)
        else:
            action_mask = action_mask.to(device=obs.x.device, dtype=torch.bool)
        if action_mask.numel() != obs.x.size(0):
            raise ValueError(
                f"Graph has {obs.x.size(0)} nodes but {action_mask.numel()} action-mask entries."
            )
        dense_action_mask, _ = to_dense_batch(
            action_mask,
            batch,
            max_num_nodes=self.max_nodes,
            fill_value=False,
        )
        return batch, physical_mask, dense_action_mask.bool(), action_mask

    def encode_observation(self, obs):
        """Return flat state, existence mask, action mask, and rigidity."""
        if obs.x.ndim != 2 or obs.x.size(1) != self.node_feature_dim:
            raise ValueError(
                f"Expected node features [N, {self.node_feature_dim}], got {tuple(obs.x.shape)}."
            )
        batch, physical_mask, dense_action_mask, action_mask = self._masks(obs)
        dense_x, _ = to_dense_batch(
            obs.x,
            batch,
            max_num_nodes=self.max_nodes,
        )
        batch_size = dense_x.size(0)
        rigidity = getattr(obs, "rigidity", None)
        if rigidity is None:
            raise ValueError("Padded MLP observations require one normalized rigidity value per graph.")
        rigidity = torch.as_tensor(
            rigidity, dtype=obs.x.dtype, device=obs.x.device
        ).reshape(batch_size, -1)
        if rigidity.size(1) != 1:
            raise ValueError("Padded MLP observations require exactly one rigidity value per graph.")

        encoded = torch.cat(
            [
                dense_x.reshape(batch_size, -1),
                physical_mask.to(obs.x.dtype),
                dense_action_mask.to(obs.x.dtype),
                rigidity,
            ],
            dim=-1,
        )
        return encoded, dense_action_mask, action_mask

    def _distribution(self, obs):
        encoded, dense_action_mask, action_mask = self.encode_observation(obs)
        mean, log_std = self._pi(encoded).view(
            encoded.size(0), self.max_nodes, 2 * self.node_action_dim
        ).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        return mean, log_std, dense_action_mask, action_mask

    def pi(self, obs):
        mean, log_std, dense_action_mask, _ = self._distribution(obs)
        eps = torch.randn_like(mean)
        log_prob = math.gaussian_logprob(eps, log_std)
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        mask = dense_action_mask.unsqueeze(-1)
        active_count = dense_action_mask.sum(dim=1).clamp_min(1).to(log_prob.dtype)
        graph_log_prob = (log_prob * mask).sum(dim=(1, 2)) / active_count
        return action[mask.expand_as(action)].view(-1, self.node_action_dim), {
            "mean": mean[mask.expand_as(mean)].view(-1, self.node_action_dim),
            "log_std": log_std[mask.expand_as(log_std)].view(-1, self.node_action_dim),
            "log_prob": graph_log_prob,
            "entropy": -graph_log_prob,
        }

    def pi_mean(self, obs):
        mean, _, dense_action_mask, _ = self._distribution(obs)
        squashed_mean = torch.tanh(mean)
        mask = dense_action_mask.unsqueeze(-1).expand_as(squashed_mean)
        return squashed_mean[mask].view(-1, self.node_action_dim)

    def _padded_action(self, obs, action, dense_action_mask, action_mask):
        action = action.reshape(-1, self.node_action_dim)
        node_count = obs.x.size(0)
        active_count = int(action_mask.sum())
        node_action = action.new_zeros((node_count, self.node_action_dim))
        if action.size(0) == node_count:
            node_action[action_mask] = action[action_mask]
        elif action.size(0) == active_count:
            node_action[action_mask] = action
        else:
            raise ValueError(
                f"Got {action.size(0)} node actions for {active_count} active and "
                f"{node_count} physical nodes."
            )
        dense_action, _ = to_dense_batch(
            node_action,
            self._batch_index(obs),
            max_num_nodes=self.max_nodes,
        )
        return dense_action * dense_action_mask.unsqueeze(-1).to(dense_action.dtype)

    def Q(self, obs, action, return_type="min", target=False):
        if return_type not in {"min", "avg", "all"}:
            raise ValueError(f"Unsupported critic return_type={return_type!r}.")
        encoded, dense_action_mask, action_mask = self.encode_observation(obs)
        padded_action = self._padded_action(
            obs, action, dense_action_mask, action_mask
        ).reshape(encoded.size(0), -1)
        qnet = self._target_Qs if target else self._Qs
        q_values = qnet(torch.cat([encoded, padded_action], dim=-1)).squeeze(-1)
        if return_type == "all":
            return q_values
        if return_type == "min":
            return q_values.min(0).values
        return q_values.mean(0)
