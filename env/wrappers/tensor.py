from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env, graph_observations=False):
		super().__init__(env)
		self.graph_observations = graph_observations
	
	def rand_act(self, env_idx=None):
		if env_idx is not None and hasattr(self.env, "set_active_env"):
			self.env.set_active_env(env_idx)
		return torch.from_numpy(self.env.action_space.sample().astype(np.float32))

	def set_active_env(self, env_idx):
		if not hasattr(self.env, "set_active_env"):
			raise AttributeError("Wrapped environment does not support multiple active envs")
		self.env.set_active_env(env_idx)

	@property
	def num_envs(self):
		return int(getattr(self.env, "num_envs", 1))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
			if self.graph_observations:
				return Data(
					x=obs["x"].float(),
					edge_index=obs["edge_index"].long(),
				)
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		if task_idx is not None:
			return self._obs_to_tensor(self.env.reset(task_idx=task_idx))
		return self._obs_to_tensor(self.env.reset())

	def reset_many(self, env_indices=None):
		if not hasattr(self.env, "reset_many"):
			raise AttributeError("Wrapped environment does not support batched reset")
		return [self._obs_to_tensor(obs) for obs in self.env.reset_many(env_indices=env_indices)]

	def step(self, action):
		obs, reward, done, info = self.env.step(self._action_to_numpy(action))
		return self._step_to_tensor(obs, reward, done, info)

	def step_many(self, actions, env_indices=None):
		if not hasattr(self.env, "step_many"):
			raise AttributeError("Wrapped environment does not support batched step")
		np_actions = [self._action_to_numpy(action) for action in actions]
		results = self.env.step_many(np_actions, env_indices=env_indices)
		return [self._step_to_tensor(obs, reward, done, info) for obs, reward, done, info in results]

	def _action_to_numpy(self, action):
		if isinstance(action, torch.Tensor):
			return action.detach().cpu().numpy()
		return action

	def _step_to_tensor(self, obs, reward, done, info):
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		info['terminated'] = torch.tensor(float(info['terminated']))
		info['truncated'] = torch.tensor(float(info['truncated']))
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
