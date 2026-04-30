from copy import deepcopy

import torch
import torch.nn as nn

from common import math
from common import mlp_layers as layers


class ActorCritic(nn.Module):
    """Actor and twin Q critics for continuous-control SAC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden = 2 * [cfg.mlp_dim]

        self._pi = layers.mlp(cfg.obs_dim, hidden, 2 * cfg.action_dim)
        self._Qs = layers.Ensemble(
            [
                layers.mlp(cfg.obs_dim + cfg.action_dim, hidden, 1, dropout=cfg.dropout)
                for _ in range(int(cfg.num_q))
            ]
        )
        self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min))

    def __repr__(self):
        repr_str = "SAC Actor Critic Network\n"
        repr_str += f"Actor: {self._pi}\n"
        repr_str += f"Critics: {self._Qs}\n"
        repr_str += "Learnable parameters: {:,}".format(self.total_params)
        return repr_str

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    @torch.no_grad()
    def soft_update_target_Q(self):
        for param, target_param in zip(self._Qs.parameters(), self._target_Qs.parameters()):
            target_param.data.lerp_(param.data, self.cfg.tau)

    def pi(self, obs):
        mean, log_std = self._pi(obs).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)
        log_prob = math.gaussian_logprob(eps, log_std)

        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        entropy = -log_prob
        return action, {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def Q(self, obs, action, return_type="min", target=False):
        assert return_type in {"min", "avg", "all"}
        qnet = self._target_Qs if target else self._Qs
        q_values = qnet(torch.cat([obs, action], dim=-1))
        if return_type == "all":
            return q_values
        if return_type == "min":
            return q_values.min(0).values
        return q_values.mean(0)
