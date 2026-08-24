from copy import deepcopy 

import torch 
import torch.nn as nn 
from torch_geometric.nn import global_add_pool, global_mean_pool

from common import math 
from common import gnn_layers
from common.graph_transforms import (
    graph_edge_input_dim,
    graph_feature_flags,
    graph_input_dim,
    physical_node_mask,
    policy_action_mask,
)
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
        legacy_hidden = 2 * [cfg.mlp_dim]
        message_hidden = getattr(cfg, "message_hidden_dims", legacy_hidden)
        action_head_hidden = getattr(cfg, "action_head_hidden_dims", legacy_hidden)
        shared_mpl_dims = getattr(cfg, "mpl_dims", None)
        actor_mpl_dims = [cfg.embedding_dim] if shared_mpl_dims is None else shared_mpl_dims
        critic_mpl_dims = [cfg.Q_output_dim] if shared_mpl_dims is None else shared_mpl_dims
        skip_connections = getattr(cfg, "mpl_skip_connections", True)
        feature_flags = graph_feature_flags(cfg)
        use_virtual_node = bool(getattr(cfg, "use_virtual_node", False))
        critic_readout = getattr(cfg, "critic_readout", "physical_mean")
        if (
            critic_readout in {"virtual_node", "physical_mean_virtual_node"}
            and not use_virtual_node
        ):
            raise ValueError(
                f"critic_readout={critic_readout!r} requires use_virtual_node=true"
            )
        message_attention = bool(getattr(cfg, "message_attention", False))
        gnn_obs_dim = graph_input_dim(
            cfg.obs_dim,
            use_virtual_node=use_virtual_node,
            use_node_roles=feature_flags["use_node_roles"],
        )
        edge_channels = graph_edge_input_dim(
            use_edge_roles=feature_flags["use_edge_roles"],
            use_edge_distance=feature_flags["use_edge_distance"],
        )

        self._pi = gnn_layers.GNN(
            gnn_obs_dim, hidden_channels=message_hidden, mpl_dims=actor_mpl_dims,
            dropout=cfg.dropout, skip_connections=skip_connections,
            edge_channels=edge_channels,
            message_attention=message_attention,
        )

        self._action_head = layers.mlp(
            actor_mpl_dims[-1], action_head_hidden, 2*cfg.action_dim,
            dropout=cfg.dropout
        )

        self._Qs = layers.Ensemble(
            [gnn_layers.Q_GNN(
                gnn_obs_dim + cfg.action_dim, hidden_channels=message_hidden,
                head_hidden_dims=cfg.head_hidden_dims, mpl_dims=critic_mpl_dims,
                dropout=cfg.dropout, skip_connections=skip_connections,
                edge_channels=edge_channels,
                critic_readout=critic_readout,
                message_attention=message_attention,
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

    def actor_parameters(self):
        """Return every trainable actor parameter, including the action head."""
        return tuple(self._pi.parameters()) + tuple(self._action_head.parameters())

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
        action_mask = policy_action_mask(obs)
        embeddings = self._pi(obs.x, obs.edge_index, getattr(obs, "edge_attr", None))
        mean, log_std = self._action_head(embeddings[action_mask]).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        
        eps = torch.randn_like(mean)
        log_prob = math.gaussian_logprob(eps, log_std)
        
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        batch = getattr(obs, "batch", None)
        if batch is None:
            batch = log_prob.new_zeros(log_prob.size(0), dtype=torch.long)
        else:
            batch = batch[action_mask]
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
        action_mask = policy_action_mask(obs)
        embeddings = self._pi(obs.x, obs.edge_index, getattr(obs, "edge_attr", None))
        mean, _ = self._action_head(embeddings[action_mask]).chunk(2, dim=-1)
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
        action_mask = policy_action_mask(obs)
        pool_mask = physical_node_mask(obs)
        node_action = action.new_zeros((obs.x.size(0), action.size(-1)))
        if action.size(0) == obs.x.size(0):
            node_action[action_mask] = action[action_mask]
        elif action.size(0) == int(pool_mask.sum()):
            node_action[action_mask] = action[action_mask[pool_mask]]
        elif action.size(0) == int(action_mask.sum()):
            node_action[action_mask] = action
        else:
            raise ValueError(
                f"Got {action.size(0)} node actions for {int(action_mask.sum())} "
                f"actuated nodes, {int(pool_mask.sum())} physical nodes, and "
                f"{obs.x.size(0)} total nodes."
            )
        q_values = qnet(
            torch.cat([obs.x, node_action], dim=-1),
            obs.edge_index,
            getattr(obs, "batch", None),
            pool_mask,
            getattr(obs, "edge_attr", None),
        )

        if return_type == "all":
            return q_values
        if return_type == "min":
            return q_values.min(0).values
        return q_values.mean(0)
