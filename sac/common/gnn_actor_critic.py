from copy import deepcopy 

import torch 
import torch.nn as nn 
from torch_geometric.nn import global_add_pool, global_mean_pool

from common import math 
from common import gnn_layers
from common import mlp_layers as layers 

class GNNActorCritic(nn.Module):
    """
    GNN Actor Critic Network for continuous-control SAC.

    This class combines a graph-based policy network (Actor) and graph-based Q-function
    networks (Critics) to learn an optimal policy for tasks where observations are represented
    as graphs.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        hidden = 2 * [cfg.mlp_dim] # I don't really know what this is about 

        self._pi = gnn_layers.GNN(
            cfg.obs_dim, cfg.embedding_dim, hidden,
            dropout=cfg.dropout
        )

        self._action_head = layers.mlp(
            cfg.embedding_dim, hidden, 2*cfg.action_dim,
            dropout=cfg.dropout
        )

        self._Qs = layers.Ensemble(
            [gnn_layers.Q_GNN(
                cfg.obs_dim + cfg.action_dim, cfg.Q_output_dim, hidden, cfg.head_hidden_dims,
                dropout=cfg.dropout
            ) for _ in range(int(cfg.num_q))]
        )

        self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)

        # Putting the log std min and log std diff in the buffer so that we can use them when updating without having to move stuff to devices
        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max - cfg.log_std_min))

    def __repr__(self):
        repr_str = "Graph Neural Network based Soft Actor Critic Network \n"
        repr_str += f"Actor: {self._pi}\n"
        repr_str += f"Critics: {self._Qs}\n"
        repr_str += "Total Learnable Parameters: {:,}".format(self.total_params)
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
        """
        Compute the action, entropy and log probability. 

        Args:
            obs:Observation data of a graph in the torch geometric Data format
        """
        mean, log_std = self._action_head(self._pi(obs.x, obs.edge_index)).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        
        eps = torch.randn_like(mean)
        log_prob = math.gaussian_logprob(eps, log_std)
        
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        batch = getattr(obs, "batch", None)
        if batch is None:
            batch = log_prob.new_zeros(log_prob.size(0), dtype=torch.long)
        log_prob = global_mean_pool(log_prob, batch).squeeze(-1) # Using a global mean pool means that the entropy then becomes normalized by the number of nodes.
        entropy = -log_prob
        return action, {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": entropy
        }

    def pi_mean(self, obs):
        """Compute deterministic actions without sampling policy noise or statistics."""
        mean, _ = self._action_head(self._pi(obs.x, obs.edge_index)).chunk(2, dim=-1)
        return torch.tanh(mean)
    
    def Q(self, obs, action, return_type="min", target=False):
        """
        Compute the Q-value(s).
        
        Args:
            obs: Observation data of a graph in the torch geometric Data format
            action: Action to take TODO: Maybe consider making this also a torch geometric Data object
            return_type: Type of Q-value to return ("min", "avg", or "all")
            target: Whether to use the target Q-network
        """
        assert return_type in {"min", "avg", "all"}
        qnet = self._target_Qs if target else self._Qs
        q_values = qnet(torch.cat([obs.x, action], dim=-1), obs.edge_index, getattr(obs, "batch", None)) # TODO: I am not sure if this works well or correctly because the dimensionality between the observation and actions seem off.

        if return_type == "all":
            return q_values
        if return_type == "min":
            return q_values.min(0).values
        return q_values.mean(0)
