import torch 
import torch.nn.functional as F 

from common import math 
from common.scale import RunningScale 
from common.actor_critic import ActorCritic 
from tensordict import TensorDict 



class SAC(torch.nn.Module):
    """
    SAC agent. Implements training + inference.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, 'device', 'cuda'))
        self.model = ActorCritic(cfg).to(self.device)
        capturable = self.device.type in {"cuda", "xpu", "hpu", "privateuseone", "xla"}
        # I understand this for the most part but I need to figure out the mechanics a bit more
        self.optim = torch.optim.Adam([
            {'params': self.model._Qs.parameters()},
        ], lr=self.cfg.lr, capturable=capturable)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable) # What the heck is this eps value doing?
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces but TODO: I still don't know what this is doing or why we need this
        self.discount = self._get_discount(cfg.episode_length)

        print('Episode length:', cfg.episode_length)
        print('Discount factor:', self.discount)
        self.register_buffer("_prev_mean", torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))


    def _safe_action(self, action):
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    
    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.
        
        Args:
            episode_length (int): Length of the episode.
        
        Returns:
            float: Discount factor.
        """
        frac = episode_length/self.cfg.discount_denom 
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save the state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Loaded a saved state dict from filepath (or dictionary) into current agent

        Args:
            fp (str or dict): Filepath to save the state dict to, or the state dict itself.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        self.model.load_state_dict(state_dict)
        return 
    
    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        """
        Select an action using the actor network.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distributions.

        Returns:
            torch.Tensor: Action to take in the environment.
        """

        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        action, info = self.model.pi(obs)
        if eval_mode:
            action = info['mean']
        return self._safe_action(action[0]).cpu()
            


    def update_pi(self, obs):
        """
        Update the policy using a sequence of latent states. 

        Args:
            obs (torch.Tensor): Sequence of latent states.
        
        Returns:
            TensorDict: Dictionary containing loss and other information.
        """
        self.model.train() # Not sure if this is needed 
        action, info = self.model.pi(obs)
        qs = self.model.Q(obs, action, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)


        # Loss is a weighted sum of Q-values 
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device)) # I don't really know what rho is doing
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean() # We want to maximize the reward + entropy or in otherwords minimize the negative reward - entropy
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm) # This is clipping the gradient inplace by taking the norm of all the gradients and then comparing that to some threshold
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none = True)



        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale" : self.scale.value,
        })
        return info

    @torch.no_grad()
    def _td_target(self, next_obs, reward, terminated):
        """
        Compute the TD-target from a reward and the observatin at the following time step

        Args:
            next_obs (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            terminated (torch.Tensor): Whether the episode terminated at the following time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        

        action, _ = self.model.pi(next_obs)
        discount = self.discount
        return reward + discount * (1-terminated) * self.model.Q(next_obs, action, return_type='min', target=True)
    
    def _update(self, obs, action, reward, terminated):
        # For standard SAC, we just need current and next states.
        # Assuming obs has a time dimension (horizon+1, batch, ...),
        # we can treat all transitions in the sequence as a big batch.
        _obs = obs[:-1]
        _next_obs = obs[1:]

        # Compute TD targets 
        with torch.no_grad():
            td_targets = self._td_target(_next_obs, reward, terminated)
        
        # Prep for update 
        self.model.train()

        # Get Q-value predictions for all Q-networks in the ensemble
        qs = self.model.Q(_obs, action, return_type='all')
        
        # Compute value loss (Critic loss)
        value_loss = 0
        for qs_pred in qs.unbind(0):
            # Assuming standard regression (num_bins=0). If using two-hot, this would be soft_ce.
            value_loss += F.mse_loss(qs_pred, td_targets)
        
        value_loss = value_loss / self.cfg.num_q

        # Update Q-functions (Critic)
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy (Actor)
        # Detach _obs so gradients don't flow back into the encoder/Q-networks from the policy loss
        pi_info = self.update_pi(_obs.detach())

        # Update target Q-functions 
        self.model.soft_update_target_Q()

        # Return training stats 
        self.model.eval()
        info = TensorDict({
            "value_loss": value_loss,
            "grad_norm": grad_norm,
        })

        info.update(pi_info)
        return info.detach().mean()
    
    def update(self, buffer):
        """
        Main update function. corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer. 

        Return 
            dict: Dictionary of training stats.
        """
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        if self.device.type == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)

        
            

            