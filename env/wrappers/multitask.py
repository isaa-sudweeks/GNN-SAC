from copy import deepcopy

import gymnasium as gym
import numpy as np


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

    def reset(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(len(self.envs))
        self.set_active_env(task_idx)
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["task"] = self.tasks[self.active_env_idx]
        info["env_idx"] = self.active_env_idx
        if getattr(self.cfg, "multitask", False):
            info["task_idx"] = self.active_env_idx
        return obs, reward, done, info

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
            if env.observation_space != reference_obs:
                raise ValueError(
                    f'Environment "{task}" has observation space {env.observation_space}, '
                    f"but the first task uses {reference_obs}. Shared-policy multitask runs "
                    "currently require matching observation spaces."
                )
            if env.action_space != reference_action:
                raise ValueError(
                    f'Environment "{task}" has action space {env.action_space}, '
                    f"but the first task uses {reference_action}. Shared-policy multitask runs "
                    "currently require matching action spaces."
                )
