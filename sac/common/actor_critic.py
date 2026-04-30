from copy import deepcopy 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from common import layers 
from tensordict import TensorDict 
from tensordict.nn import TensorDictParams 

class ActorCritic(nn.Module):
    """
    Actor-Critic network for SAC.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Initialize networks 
        self._pi = layers.mlp(cfg.obs_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim) # Actor or policy 
        # Standard SAC Q-networks output a single scalar value for the state-action pair
        self.Qs = layers.Ensemble([layers.mls(cfg.obs_dim + cfg.action_dim, 2*[cfg.mlp_dim], 1, dropout=cfg.dropout) for _ in range(int(cfg.num_q))])

        # Okay I don't really know if we need this 
        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_def", torch.tensor(cfg.log_std_max) - self.log_std_min)


    def init(self):
        # Create params 
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules 
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules 
        # We do this strange assignment to avoid having duplicated tensors in the state-dict
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params
    
    def __repr__(self):
        repr = "SAC Actor Critic Network\n"
        modules = ['Actor', 'Critics']
        for i, m in enumerate([self._pi, self._Qs]):
            repr += f'{modules[i]}: {m}\n'
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr 

    @property 
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding the 'train' method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self
    
    def soft_update_target_Q(self):
        """
        Soft update the target Q-networks using Polyak averaging.
        Not really sure if I need this.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def pi(self, obs):
        """
        Get action from policy.
        """
        
        # Guassian Policy Prior 
        mean, log_std = self._pi(obs).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_def)
        eps = torch.randn_like(mean)
        action_dims = None
        log_prob = math.gaussian_log_prob(eps, log_std)

        # Scale log probability by action dimensions 
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick 
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        # Add entropy bonus 
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std" : log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, obs, a, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        'return_type' can be one of the following: ['min', 'avg', 'all']:
            'min': returns the minimum Q-value across all Q-networks
            'avg': returns the average Q-value across all Q-networks
            'all': returns all Q-values
        'target' determines whether to use the target Q-networks
        'detach' determines whether to detach the Q-networks
        """
        assert return_type in {'min', 'avg', 'all'}

        # Combine input
        combined_input = torch.cat([obs, a], dim=-1)
        
        # Set target networks
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        
        out = qnet(combined_input)

        if return_type == 'all':
            return out
        
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = out[qidx] # Use standard regression output directly
        if return_type == 'min':
            return Q.min(0).values 
        return Q.sum(0) / 2