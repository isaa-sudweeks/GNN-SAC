from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym

from env.mujoco_gen.topology_envs import (
    broken_node_regime_fraction,
    broken_node_regime_slots,
    broken_node_schedule,
)


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

        regime_fraction = broken_node_regime_fraction(cfg)
        if (
            broken_node_schedule(cfg) == "interleaved"
            and regime_fraction > 0.0
            and self.num_envs > 1
        ):
            self._broken_regime_slots = broken_node_regime_slots(self.num_envs, regime_fraction)
        else:
            # None (rather than an all-False array) preserves the exact
            # legacy behavior when no regime split is configured -- including
            # the single-env case, which cannot represent both regimes at
            # once and must stay eligible for the plain per-node Bernoulli
            # draw rather than being silently locked to standard.
            self._broken_regime_slots = None

        self.envs = [
            self._make_env(cfg, make_env_fns, env_idx)
            for env_idx in range(self.num_envs)
        ]
        self._validate_spaces()

        self.active_env_idx = 0
        self.env = self.envs[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def _make_env(self, cfg, make_env_fns, env_idx):
        env_cfg = deepcopy(cfg)
        env_cfg.num_envs = 1
        if getattr(env_cfg, "seed", None) is not None:
            env_cfg.seed = int(env_cfg.seed) + int(env_idx)
        if self._broken_regime_slots is not None and not self._broken_regime_slots[env_idx]:
            self._lock_to_standard_regime(env_cfg)
        errors = []
        for fn in make_env_fns:
            try:
                return fn(env_cfg)
            except ValueError as exc:
                errors.append(str(exc))
        details = "; ".join(errors)
        raise ValueError(f'Failed to make environment "{cfg.task}": {details}')

    @staticmethod
    def _cfg_get(config, name, default=None):
        if hasattr(config, "get"):
            return config.get(name, default)
        return getattr(config, name, default)

    @classmethod
    def _lock_to_standard_regime(cls, env_cfg):
        """Force this slot's copy to the nominal (all-active) topology.

        Broken-node sampling is a per-episode reset-time draw; disabling it
        here keeps this slot's episodes teacher-compatible for the lifetime
        of the env, guaranteeing a "clean" standard-regime data source.

        A parsed config's nested sections may be plain dicts rather than
        attribute-style namespaces (see common.parser.Config), so both the
        read and the write below need to support either shape.
        """
        params = cls._cfg_get(env_cfg, "domain_randomization_params", None)
        if params is None:
            return
        broken_nodes = cls._cfg_get(params, "broken_nodes", None)
        if broken_nodes is None:
            return
        if isinstance(broken_nodes, dict):
            broken_nodes["enabled"] = False
        else:
            broken_nodes.enabled = False

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
        if self._broken_regime_slots is not None:
            info["regime"] = "broken" if self._broken_regime_slots[env_idx] else "standard"

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
