from __future__ import annotations

from collections.abc import Iterable, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from env.mujoco_gen.topology_envs import (
    make_truss_env_config,
    resolve_truss_topology,
)


class MjxVectorGraphEnv(gym.Env):
    """Stateful adapter around mujoco-truss-gen's batch-native MJX environment.

    The public API matches the repository's repeated-environment wrapper while
    keeping physics state, observations, and policy actions on the accelerator.
    One instance supports one fixed topology and one fixed batch size.
    """

    accepts_torch_actions = True
    node_action_dim = 1
    node_feature_dim = 6

    def __init__(self, cfg):
        if bool(getattr(cfg, "visualize", False)):
            raise ValueError("The MJX vector backend does not support interactive rendering.")
        eval_backend = str(getattr(cfg, "eval_backend", "mujoco")).lower()
        if bool(getattr(cfg, "save_video", False)) and eval_backend == "mjx":
            raise ValueError(
                "MJX evaluation does not support video capture; use eval_backend=mujoco."
            )

        try:
            import jax
            import jax.numpy as jnp
            from mujoco_truss_gen import MjxNodeVelocityEnv, get_edge_index
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "mujoco_backend=mjx requires mujoco-truss-gen>=0.11.0b0 with "
                "MjxNodeVelocityEnv and a working JAX/MJX installation."
            ) from exc

        self.cfg = cfg
        self.task = str(getattr(cfg, "task", "truss-graph"))
        self.topology = resolve_truss_topology(cfg)
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least one.")

        self._jax = jax
        self._jnp = jnp
        truss_config = make_truss_env_config(cfg)
        if (
            truss_config.domain_randomization is not None
            and truss_config.domain_randomization.model_factory is not None
        ):
            raise ValueError(
                "The MJX vector backend requires fixed-shape domain randomization. "
                "Disable domain_randomization_params.length_scale and "
                "domain_randomization_params.physical_parameters for MJX runs, or use "
                "mujoco_backend=mujoco for model-level randomization."
            )
        self._core = MjxNodeVelocityEnv(truss_config)
        self.mj_model = self._core.mujoco_model
        self.max_episode_steps = int(self._core.config.max_steps)
        self.nsubsteps = int(self._core.config.nsubsteps)
        self.speed = float(self._core.config.speed)
        self.graph_node_names = list(self._core._controller.node_names)
        self.passive_node_names = [
            node_name
            for node_name, is_passive in zip(
                self.graph_node_names,
                np.asarray(self._core._controller.passive_node_mask, dtype=bool),
            )
            if is_passive
        ]
        self.num_external_actuators = int(len(self.mj_model.external_actuator_ids))

        node_count = int(self._core.action_size)
        if node_count != len(self.graph_node_names):
            raise RuntimeError("MJX control graph action size does not match its node metadata.")
        edge_index = get_edge_index(self.mj_model, graph_view="control")
        self._edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(node_count, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(node_count - 1, 0),
                    shape=edge_index.shape,
                    dtype=np.int64,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(node_count, self.node_action_dim),
            dtype=np.float32,
        )

        seed = int(getattr(cfg, "seed", 0))
        self._key = jax.random.key(seed)
        self._state = None
        self.active_env_idx = 0
        self._reset_compiled = jax.jit(self._core.reset)
        self._reset_where_compiled = jax.jit(self._core.reset_where)
        self._step_compiled = jax.jit(self._core.step)
        self._step_masked_compiled = jax.jit(self._step_masked)

    @property
    def unwrapped(self):
        return self

    @property
    def action_device(self) -> torch.device:
        platform = self._jax.devices()[0].platform
        if platform in {"gpu", "cuda", "rocm"} and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def set_active_env(self, env_idx: int) -> None:
        self._validate_index(env_idx)
        self.active_env_idx = int(env_idx)

    def reset(self, task_idx: int | None = None):
        env_idx = self.active_env_idx if task_idx is None else int(task_idx)
        self.set_active_env(env_idx)
        return self.reset_many([env_idx])[0]

    def reset_many(self, env_indices: Iterable[int] | None = None):
        """Reset selected slots in the fixed-size MJX state batch.

        ``None`` resets every environment. On the first call the entire batch
        must be initialized because the JIT-compiled MJX state has a fixed
        leading dimension. Later calls use ``reset_where`` for a subset so
        completed episodes can restart without disturbing environments whose
        episodes are still running. A fresh JAX random key is generated for
        every batch slot, and observations are returned only for the requested
        indices, preserving their input order.
        """
        indices = self._normalize_indices(env_indices)
        keys = self._next_keys()
        if self._state is None:
            flat_obs, self._state = self._reset_compiled(keys)
        elif len(indices) == self.num_envs and indices == list(range(self.num_envs)):
            flat_obs, self._state = self._reset_compiled(keys)
        else:
            mask = self._index_mask(indices)
            flat_obs, self._state = self._reset_where_compiled(keys, self._state, mask)
        return self._graph_observations(flat_obs, indices)

    def step(self, action):
        result = self.step_many([action], [self.active_env_idx])
        return result[0]

    def step_many(self, actions: Sequence, env_indices: Iterable[int] | None = None):
        if self._state is None:
            raise RuntimeError("reset or reset_many must be called before step_many.")
        indices = self._normalize_indices(env_indices)
        if len(actions) != len(indices):
            raise ValueError(f"Got {len(actions)} actions for {len(indices)} environments.")

        full_actions = torch.zeros(
            (self.num_envs, self._core.action_size),
            dtype=torch.float32,
            device=self.action_device,
        )
        for env_idx, action in zip(indices, actions):
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.action_device)
            expected_shape = tuple(self.action_space.shape)
            if tuple(action_tensor.shape) != expected_shape and action_tensor.numel() != self._core.action_size:
                raise ValueError(
                    f"Action for environment {env_idx} must have shape {expected_shape}; "
                    f"got {tuple(action_tensor.shape)}."
                )
            full_actions[env_idx] = action_tensor.reshape(-1)

        normalized_actions = full_actions.clamp(-1.0, 1.0)
        physical_actions = normalized_actions * self.speed
        jax_actions = self._jnp.from_dlpack(physical_actions.contiguous())
        keys = self._next_keys()
        mask = self._index_mask(indices)
        if len(indices) == self.num_envs and indices == list(range(self.num_envs)):
            flat_obs, self._state, reward, done, info = self._step_compiled(
                keys, self._state, jax_actions
            )
        else:
            flat_obs, self._state, reward, done, info = self._step_masked_compiled(
                keys, self._state, jax_actions, mask
            )

        observations = self._graph_observations(flat_obs, indices)
        rewards = self._to_torch(reward)
        dones = np.asarray(self._jax.device_get(done), dtype=bool)
        torch_info = {key: self._to_torch(value) for key, value in info.items()}
        results = []
        for result_idx, env_idx in enumerate(indices):
            env_info = {key: value[env_idx] for key, value in torch_info.items()}
            env_info["task"] = self.task
            env_info["env_idx"] = env_idx
            env_info["success"] = float(env_info.get("success", 0.0))
            results.append(
                (
                    observations[result_idx],
                    rewards[env_idx],
                    bool(dones[env_idx]),
                    env_info,
                )
            )
        return results

    def _step_masked(self, keys, state, actions, mask):
        flat_obs, stepped_state, reward, done, info = self._core.step(keys, state, actions)
        batch_size = self.num_envs

        def select(new_value, old_value):
            expanded_mask = mask.reshape((batch_size,) + (1,) * (new_value.ndim - 1))
            return self._jnp.where(expanded_mask, new_value, old_value)

        merged_state = self._jax.tree.map(select, stepped_state, state)
        flat_obs = self._core._get_obs(merged_state)
        reward = self._jnp.where(mask, reward, 0.0)
        done = self._jnp.where(mask, done, False)
        info = {key: self._jnp.where(mask, value, 0) for key, value in info.items()}
        return flat_obs, merged_state, reward, done, info

    def _graph_observations(self, flat_obs, indices: Sequence[int]):
        obs = self._to_torch(flat_obs)
        node_count = self._core.action_size
        positions = obs[:, : 3 * node_count].reshape(self.num_envs, node_count, 3)
        velocities = obs[:, 3 * node_count : 6 * node_count].reshape(
            self.num_envs, node_count, 3
        )
        features = torch.cat((positions, velocities), dim=-1)
        return [
            {"x": features[env_idx], "edge_index": self._edge_index}
            for env_idx in indices
        ]

    def _to_torch(self, value) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(value)

    def _next_keys(self):
        keys = self._jax.random.split(self._key, self.num_envs + 1)
        self._key = keys[0]
        return keys[1:]

    def _index_mask(self, indices: Sequence[int]):
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[indices] = True
        return self._jnp.asarray(mask)

    def _normalize_indices(self, env_indices: Iterable[int] | None) -> list[int]:
        if env_indices is None:
            return list(range(self.num_envs))
        indices = [int(index) for index in env_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("Environment indices must be unique.")
        for index in indices:
            self._validate_index(index)
        return indices

    def _validate_index(self, env_idx: int) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(
                f"Environment index {env_idx} is out of range for {self.num_envs} environments."
            )

    def close(self):
        self._state = None

    def render(self, **kwargs):
        raise RuntimeError("The MJX vector backend does not support rendering.")
