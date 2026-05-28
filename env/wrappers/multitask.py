from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultitaskWrapper(gym.Env):
    def __init__(self, cfg, make_env_fns):
        self.cfg = cfg
        self.envs = []
        if getattr(cfg, "multitask", False):
            self.tasks = list(getattr(cfg, "tasks", [cfg.task]))
        else:
            self.tasks = [cfg.task] * int(getattr(cfg, "num_envs", 1))
        if not self.tasks:
            raise ValueError("MultitaskWrapper requires at least one environment")
        
        for task in self.tasks:
            env_cfg = deepcopy(cfg)
            env_cfg.task = task
            env = None 
            errors = []
            for fn in make_env_fns:
                try:
                    env = fn(env_cfg)
                except ValueError as exc:
                    errors.append(str(exc))
                    pass 
            if env is None:
                details = '; '.join(errors)
                raise ValueError(f'Failed to make environment "{task}": {details}')
            self.envs.append(env)

        self._validate_spaces()
        
        self.active_env_idx = 0
        self.env = self.envs[0]
        self.num_envs = len(self.envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def set_active_env(self, env_idx):
        if env_idx < 0 or env_idx >= len(self.envs):
            raise IndexError(f"Environment index {env_idx} is out of range for {len(self.envs)} envs")
        self.active_env_idx = env_idx
        self.env = self.envs[self.active_env_idx]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(len(self.envs))
        self.set_active_env(task_idx)
        return self.env.reset()

    def reset_many(self, env_indices=None):
        env_indices = self._normalize_indices(env_indices)
        return [self._reset_one(env_idx) for env_idx in env_indices]

    def _reset_one(self, env_idx):
        self.set_active_env(env_idx)
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._annotate_info(info, self.active_env_idx)
        return obs, reward, done, info

    def step_many(self, actions, env_indices=None):
        env_indices = self._normalize_indices(env_indices)
        if len(actions) != len(env_indices):
            raise ValueError(f"Got {len(actions)} actions for {len(env_indices)} environments")
        return [self._step_one(env_idx, action) for env_idx, action in zip(env_indices, actions)]

    def _step_one(self, env_idx, action):
        self.set_active_env(env_idx)
        obs, reward, done, info = self.env.step(action)
        self._annotate_info(info, env_idx)
        return obs, reward, done, info

    def _annotate_info(self, info, env_idx):
        info["task"] = self.tasks[env_idx]
        info["env_idx"] = env_idx
        if getattr(self.cfg, "multitask", False):
            info["task_idx"] = env_idx

    def _normalize_indices(self, env_indices):
        if env_indices is None:
            return list(range(len(self.envs)))
        return list(env_indices)

    @property
    def unwrapped(self):
        return self.env.unwrapped
        
    def close(self):
        for e in self.envs:
            e.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def _validate_spaces(self):
        reference_obs = self.envs[0].observation_space
        reference_action = self.envs[0].action_space
        for task, env in zip(self.tasks[1:], self.envs[1:]):
            if env.observation_space != reference_obs and not self._compatible_graph_spaces(
                reference_obs, env.observation_space
            ):
                raise ValueError(
                    f'Environment "{task}" has observation space {env.observation_space}, '
                    f"but the first task uses {reference_obs}. Shared-policy multitask runs "
                    "currently require matching observation spaces."
                )
            if env.action_space != reference_action and not self._compatible_node_action_spaces(
                reference_action, env.action_space
            ):
                raise ValueError(
                    f'Environment "{task}" has action space {env.action_space}, '
                    f"but the first task uses {reference_action}. Shared-policy multitask runs "
                    "currently require matching action spaces."
                )

    def _compatible_graph_spaces(self, reference_obs, candidate_obs):
        if not (isinstance(reference_obs, spaces.Dict) and isinstance(candidate_obs, spaces.Dict)):
            return False
        if set(reference_obs.spaces) != {"x", "edge_index"} or set(candidate_obs.spaces) != {"x", "edge_index"}:
            return False
        reference_x = reference_obs.spaces["x"]
        candidate_x = candidate_obs.spaces["x"]
        reference_edges = reference_obs.spaces["edge_index"]
        candidate_edges = candidate_obs.spaces["edge_index"]
        return (
            len(reference_x.shape) == 2
            and len(candidate_x.shape) == 2
            and reference_x.shape[1] == candidate_x.shape[1]
            and reference_edges.shape[0] == candidate_edges.shape[0] == 2
        )

    def _compatible_node_action_spaces(self, reference_action, candidate_action):
        if not (isinstance(reference_action, spaces.Box) and isinstance(candidate_action, spaces.Box)):
            return False
        return (
            len(reference_action.shape) == 2
            and len(candidate_action.shape) == 2
            and reference_action.shape[1] == candidate_action.shape[1]
            and np.allclose(reference_action.low, reference_action.low[0, 0])
            and np.allclose(candidate_action.low, candidate_action.low[0, 0])
            and np.allclose(reference_action.high, reference_action.high[0, 0])
            and np.allclose(candidate_action.high, candidate_action.high[0, 0])
            and np.isclose(reference_action.low[0, 0], candidate_action.low[0, 0])
            and np.isclose(reference_action.high[0, 0], candidate_action.high[0, 0])
        )
