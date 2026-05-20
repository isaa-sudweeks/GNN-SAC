from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym


class RepeatedEnvWrapper(gym.Env):
    """
    Run independent copies of one task through a vector-style API.
    """

    def __init__(self, cfg, make_env_fns):
        self.cfg = cfg
        self.task = cfg.task
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        if self.num_envs < 1:
            raise ValueError("RepeatedEnvWrapper requires at least one environment")

        self.envs = [self._make_env(cfg, make_env_fns) for _ in range(self.num_envs)]
        self._validate_spaces()

        self.active_env_idx = 0
        self.env = self.envs[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def _make_env(self, cfg, make_env_fns):
        env_cfg = deepcopy(cfg)
        env_cfg.num_envs = 1
        errors = []
        for fn in make_env_fns:
            try:
                return fn(env_cfg)
            except ValueError as exc:
                errors.append(str(exc))
        details = "; ".join(errors)
        raise ValueError(f'Failed to make environment "{cfg.task}": {details}')

    def set_active_env(self, env_idx):
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"Environment index {env_idx} is out of range for {self.num_envs} envs")
        self.active_env_idx = env_idx
        self.env = self.envs[self.active_env_idx]

    def reset(self, task_idx=None):
        if task_idx is None:
            task_idx = self.active_env_idx
        self.set_active_env(task_idx)
        return self.env.reset()

    def reset_many(self, env_indices=None):
        env_indices = self._normalize_indices(env_indices)
        return list(self._executor.map(lambda idx: self._reset_one(idx), env_indices))

    def _reset_one(self, env_idx):
        obs = self.envs[env_idx].reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._annotate_info(info, self.active_env_idx)
        return obs, reward, done, info

    def step_many(self, actions, env_indices=None):
        env_indices = self._normalize_indices(env_indices)
        if len(actions) != len(env_indices):
            raise ValueError(f"Got {len(actions)} actions for {len(env_indices)} environments")
        return list(self._executor.map(lambda item: self._step_one(*item), zip(env_indices, actions)))

    def _step_one(self, env_idx, action):
        obs, reward, done, info = self.envs[env_idx].step(action)
        self._annotate_info(info, env_idx)
        return obs, reward, done, info

    def _annotate_info(self, info, env_idx):
        info["task"] = self.task
        info["env_idx"] = env_idx

    def _normalize_indices(self, env_indices):
        if env_indices is None:
            return list(range(self.num_envs))
        return list(env_indices)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def _validate_spaces(self):
        reference_obs = self.envs[0].observation_space
        reference_action = self.envs[0].action_space
        for env_idx, env in enumerate(self.envs[1:], start=1):
            if env.observation_space != reference_obs:
                raise ValueError(
                    f"Repeated env {env_idx} has observation space {env.observation_space}, "
                    f"but env 0 uses {reference_obs}."
                )
            if env.action_space != reference_action:
                raise ValueError(
                    f"Repeated env {env_idx} has action space {env.action_space}, "
                    f"but env 0 uses {reference_action}."
                )
