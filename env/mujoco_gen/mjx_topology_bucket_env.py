from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy

import gymnasium as gym

from env.mujoco_gen.mjx_vector_env import MjxVectorGraphEnv


class MjxTopologyBucketEnv(gym.Env):
    """Split a total vector batch evenly across fixed-topology MJX buckets.

    MJX cannot place different compiled models in one array batch. This wrapper
    therefore owns one :class:`MjxVectorGraphEnv` per topology and presents the
    buckets as one global vector environment. ``cfg.num_envs`` is interpreted
    as the total exposed environment count and must divide evenly across the
    requested topologies. Global indices are interleaved by topology so any
    leading subset remains approximately topology-balanced.
    """

    accepts_torch_actions = True
    is_topology_bucket = True
    node_action_dim = 1
    node_feature_dim = 6

    def __init__(self, cfg, topologies: Sequence[str]):
        self.cfg = cfg
        self.task = str(getattr(cfg, "task", "truss-graph")).split(":", 1)[0]
        self.topologies = [str(topology) for topology in topologies]
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        self._validate_configuration()

        bucket_size = self.num_envs // len(self.topologies)
        self.envs_per_topology = bucket_size
        self.bucket_sizes = [bucket_size] * len(self.topologies)
        self.buckets = []
        try:
            for bucket_idx, topology in enumerate(self.topologies):
                self.buckets.append(
                    self._make_bucket(cfg, topology, bucket_size, bucket_idx)
                )
        except Exception:
            for bucket in self.buckets:
                bucket.close()
            raise
        self.envs = self.buckets

        self._global_to_bucket: list[tuple[int, int]] = []
        self._bucket_to_global: list[list[int]] = [
            [-1] * bucket_size for _ in self.topologies
        ]
        for local_idx in range(bucket_size):
            for bucket_idx in range(len(self.buckets)):
                global_idx = len(self._global_to_bucket)
                self._global_to_bucket.append((bucket_idx, local_idx))
                self._bucket_to_global[bucket_idx][local_idx] = global_idx

        self.max_episode_steps = max(bucket.max_episode_steps for bucket in self.buckets)
        self.nsubsteps = self.buckets[0].nsubsteps
        self.active_env_idx = 0
        self.active_bucket_idx = 0
        self.env = self.buckets[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def action_device(self):
        return self.env.action_device

    @property
    def topology_allocations(self) -> dict[str, int]:
        return dict(zip(self.topologies, self.bucket_sizes))

    @property
    def topology_representative_indices(self) -> dict[str, int]:
        return {
            topology: self._bucket_to_global[bucket_idx][0]
            for bucket_idx, topology in enumerate(self.topologies)
        }

    def topology_for_env(self, env_idx: int) -> str:
        bucket_idx, _ = self._mapping(env_idx)
        return self.topologies[bucket_idx]

    def set_active_env(self, env_idx: int) -> None:
        bucket_idx, local_idx = self._mapping(env_idx)
        self.active_env_idx = int(env_idx)
        self.active_bucket_idx = bucket_idx
        self.env = self.buckets[bucket_idx]
        self.env.set_active_env(local_idx)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, task_idx: int | None = None):
        env_idx = self.active_env_idx if task_idx is None else int(task_idx)
        self.set_active_env(env_idx)
        _, local_idx = self._mapping(env_idx)
        return self.env.reset(task_idx=local_idx)

    def reset_many(self, env_indices: Iterable[int] | None = None):
        indices = self._normalize_indices(env_indices)
        grouped = self._group_indices(indices)
        observations = {}
        for bucket_idx, entries in grouped.items():
            local_indices = [local_idx for _, local_idx in entries]
            bucket_observations = self.buckets[bucket_idx].reset_many(local_indices)
            for (global_idx, _), observation in zip(entries, bucket_observations):
                observations[global_idx] = observation
        return [observations[env_idx] for env_idx in indices]

    def step(self, action):
        return self.step_many([action], [self.active_env_idx])[0]

    def step_many(self, actions: Sequence, env_indices: Iterable[int] | None = None):
        indices = self._normalize_indices(env_indices)
        if len(actions) != len(indices):
            raise ValueError(f"Got {len(actions)} actions for {len(indices)} environments.")

        grouped = defaultdict(list)
        for action_idx, (global_idx, action) in enumerate(zip(indices, actions)):
            bucket_idx, local_idx = self._mapping(global_idx)
            grouped[bucket_idx].append((action_idx, global_idx, local_idx, action))

        results = [None] * len(indices)
        for bucket_idx, entries in grouped.items():
            bucket_actions = [entry[3] for entry in entries]
            local_indices = [entry[2] for entry in entries]
            bucket_results = self.buckets[bucket_idx].step_many(
                bucket_actions,
                env_indices=local_indices,
            )
            topology = self.topologies[bucket_idx]
            for entry, result in zip(entries, bucket_results):
                action_idx, global_idx, local_idx, _ = entry
                observation, reward, done, info = result
                info["task"] = f"{self.task}:{topology}"
                info["topology"] = topology
                info["topology_idx"] = bucket_idx
                info["bucket_env_idx"] = local_idx
                info["env_idx"] = global_idx
                results[action_idx] = (observation, reward, done, info)
        return results

    def close(self):
        for bucket in self.buckets:
            bucket.close()

    def render(self, **kwargs):
        raise RuntimeError("The MJX topology-bucket backend does not support rendering.")

    def _make_bucket(self, cfg, topology: str, bucket_size: int, bucket_idx: int):
        bucket_cfg = deepcopy(cfg)
        bucket_cfg.num_envs = bucket_size
        bucket_cfg.truss_topology = topology
        bucket_cfg.truss_topologies = None
        bucket_cfg.task = f"{self.task}:{topology}"
        bucket_cfg.seed = int(getattr(cfg, "seed", 0)) + bucket_idx
        return MjxVectorGraphEnv(bucket_cfg)

    def _validate_configuration(self) -> None:
        if len(self.topologies) < 2:
            raise ValueError("MjxTopologyBucketEnv requires at least two topologies.")
        if len(set(self.topologies)) != len(self.topologies):
            raise ValueError("MJX topology buckets require unique topology names.")
        topology_count = len(self.topologies)
        if self.num_envs < topology_count:
            raise ValueError(
                "num_envs must allocate at least one environment per topology."
            )
        if self.num_envs % topology_count != 0:
            raise ValueError(
                "num_envs must be divisible by the number of truss_topologies "
                "for balanced MJX topology buckets."
            )

    def _mapping(self, env_idx: int) -> tuple[int, int]:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(
                f"Environment index {env_idx} is out of range for {self.num_envs} environments."
            )
        return self._global_to_bucket[env_idx]

    def _normalize_indices(self, env_indices: Iterable[int] | None) -> list[int]:
        indices = list(range(self.num_envs)) if env_indices is None else [
            int(index) for index in env_indices
        ]
        if len(set(indices)) != len(indices):
            raise ValueError("Environment indices must be unique.")
        for env_idx in indices:
            self._mapping(env_idx)
        return indices

    def _group_indices(self, indices: Sequence[int]):
        grouped = defaultdict(list)
        for global_idx in indices:
            bucket_idx, local_idx = self._mapping(global_idx)
            grouped[bucket_idx].append((global_idx, local_idx))
        return grouped
