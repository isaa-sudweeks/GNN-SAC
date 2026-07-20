from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch_geometric.data import Batch, Data


@dataclass(frozen=True)
class ReplayBatch:
    """One balanced learner batch plus its task-specific constituent batches."""

    combined: tuple
    by_task: Mapping[str, tuple]


class _GNNTaskBuffer:
    """Circular replay buffer for graph observations and node-aligned actions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device(getattr(cfg, "device", "cuda"))
        self._capacity = int(min(cfg.buffer_size, cfg.steps))
        self._batch_size = int(cfg.batch_size)
        self._num_eps = 0
        self._size = 0
        self._idx = 0
        self._storage_device = torch.device("cpu")

        self._obs = [None] * self._capacity
        self._next_obs = [None] * self._capacity
        self._action = [None] * self._capacity
        self._reward = [None] * self._capacity
        self._terminated = [None] * self._capacity

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps

    @property
    def size(self):
        return self._size

    def add(self, td, count_episode=True):
        if isinstance(td, list):
            obs_seq = [step["obs"] for step in td]
            actions = [step["action"].squeeze(0) for step in td][1:]
            rewards = [step["reward"].squeeze(0) for step in td][1:]
            terminated = [step["terminated"].squeeze(0) for step in td][1:]
        else:
            obs_seq = self._graph_sequence(td["obs"])
            actions = self._transition_sequence(td["action"])[1:]
            rewards = self._transition_sequence(td["reward"])[1:]
            terminated = self._transition_sequence(td["terminated"])[1:]

        obs = obs_seq[:-1]
        next_obs = obs_seq[1:]
        n = len(obs)

        if not (len(actions) == len(rewards) == len(terminated) == n):
            raise ValueError(
                "Graph replay episode fields must contain one initial observation "
                "and one action/reward/terminated entry per transition."
            )

        if n > self._capacity:
            obs = obs[-self._capacity:]
            next_obs = next_obs[-self._capacity:]
            actions = actions[-self._capacity:]
            rewards = rewards[-self._capacity:]
            terminated = terminated[-self._capacity:]
            n = self._capacity

        for i in range(n):
            write_idx = (self._idx + i) % self._capacity
            action = self._clone_tensor(actions[i]).float()
            self._validate_action(obs[i], action)
            self._obs[write_idx] = self._clone_graph(obs[i])
            self._next_obs[write_idx] = self._clone_graph(next_obs[i])
            self._action[write_idx] = action
            self._reward[write_idx] = self._scalar_tensor(rewards[i])
            self._terminated[write_idx] = self._scalar_tensor(terminated[i])

        self._idx = (self._idx + n) % self._capacity
        self._size = min(self._size + n, self._capacity)
        self._num_eps += int(bool(count_episode))
        return self._num_eps

    def sample(self):
        if self._size < self._batch_size:
            raise ValueError(f"Replay buffer has {self._size} transitions, need batch_size={self._batch_size}.")

        idx = torch.randint(self._size, (self._batch_size,), device=self._storage_device).tolist()
        obs = Batch.from_data_list([self._obs[i] for i in idx]).to(self._device, non_blocking=True)
        next_obs = Batch.from_data_list([self._next_obs[i] for i in idx]).to(self._device, non_blocking=True)
        action = torch.cat([self._action[i] for i in idx], dim=0).to(self._device, non_blocking=True)
        reward = torch.stack([self._reward[i] for i in idx], dim=0).to(self._device, non_blocking=True)
        terminated = torch.stack([self._terminated[i] for i in idx], dim=0).to(self._device, non_blocking=True)
        return obs, action, reward, terminated, next_obs

    def _graph_sequence(self, obs):
        if isinstance(obs, Batch):
            return obs.to_data_list()
        if isinstance(obs, Data):
            raise ValueError("Graph replay add() needs an episode of observations, not a single Data object.")
        if isinstance(obs, (list, tuple)) and all(isinstance(item, Data) for item in obs):
            return list(obs)
        raise TypeError(f"Unsupported graph observation sequence type: {type(obs)!r}")

    def _transition_sequence(self, value):
        if isinstance(value, torch.Tensor):
            return list(value.unbind(0))
        if isinstance(value, (list, tuple)):
            return list(value)
        raise TypeError(f"Unsupported transition field type: {type(value)!r}")

    def _clone_graph(self, graph):
        return graph.clone().detach().cpu()

    def _clone_tensor(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.detach().cpu().clone()

    def _scalar_tensor(self, value):
        return self._clone_tensor(value).float().view(1)

    def _validate_action(self, obs, action):
        num_nodes = obs.num_nodes
        if num_nodes is not None and action.shape[0] != num_nodes:
            raise ValueError(
                f"Node action count ({action.shape[0]}) must match graph node count ({num_nodes})."
            )

    def state_dict(self):
        return {
            "capacity": self._capacity,
            "batch_size": self._batch_size,
            "num_eps": self._num_eps,
            "size": self._size,
            "idx": self._idx,
            "obs": self._obs,
            "next_obs": self._next_obs,
            "action": self._action,
            "reward": self._reward,
            "terminated": self._terminated,
        }

    def load_state_dict(self, state_dict):
        saved_capacity = int(state_dict["capacity"])
        if saved_capacity != self._capacity:
            raise ValueError(
                f"Checkpoint replay capacity ({saved_capacity}) does not match current capacity ({self._capacity}). "
                "Resume with the same buffer_size and steps, or start a fresh run."
            )
        self._batch_size = int(state_dict.get("batch_size", self._batch_size))
        self._num_eps = int(state_dict["num_eps"])
        self._size = int(state_dict["size"])
        self._idx = int(state_dict["idx"])
        self._obs = state_dict["obs"]
        self._next_obs = state_dict["next_obs"]
        self._action = state_dict["action"]
        self._reward = state_dict["reward"]
        self._terminated = state_dict["terminated"]


class GNNBuffer:
    """Task-balanced replay for graph SAC.

    ``buffer_size`` remains the total replay capacity. Multi-task batches draw
    exactly the same number of transitions from every distinct task.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.task_names = self._task_names(cfg)
        self._task_count = len(self.task_names)
        total_capacity = int(cfg.buffer_size)
        batch_size = int(cfg.batch_size)
        if total_capacity % self._task_count != 0:
            raise ValueError(
                f"buffer_size={total_capacity} must be divisible by the number of tasks "
                f"({self._task_count})."
            )
        if batch_size % self._task_count != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by the number of tasks "
                f"({self._task_count})."
            )

        self._batch_size = batch_size
        self._batch_size_per_task = batch_size // self._task_count
        self._capacity_per_task = total_capacity // self._task_count
        self._buffers = {}
        for task_name in self.task_names:
            task_cfg = deepcopy(cfg)
            task_cfg.buffer_size = self._capacity_per_task
            task_cfg.batch_size = self._batch_size_per_task
            # _GNNTaskBuffer caps capacity by steps. Each task can receive no
            # more than the total run length, so this preserves the requested
            # per-task capacity even for small test configurations.
            task_cfg.steps = max(int(getattr(cfg, "steps", total_capacity)), self._capacity_per_task)
            self._buffers[task_name] = _GNNTaskBuffer(task_cfg)

    @staticmethod
    def _task_names(cfg):
        multitask = bool(getattr(cfg, "multitask", False))
        backend = str(getattr(cfg, "mujoco_backend", "mujoco")).lower()
        topologies = getattr(cfg, "truss_topologies", None)
        if backend == "mjx" and topologies and len(topologies) > 1:
            base_task = str(getattr(cfg, "task", "truss-graph")).split(":", 1)[0]
            candidates = [f"{base_task}:{topology}" for topology in topologies]
        elif multitask:
            candidates = [str(task) for task in getattr(cfg, "tasks", [])]
        else:
            candidates = [str(getattr(cfg, "task", "task"))]
        task_names = list(dict.fromkeys(candidates))
        if not task_names:
            raise ValueError("Task-balanced replay requires at least one task.")
        return task_names

    @property
    def capacity(self):
        return sum(buffer.capacity for buffer in self._buffers.values())

    @property
    def num_eps(self):
        return sum(buffer.num_eps for buffer in self._buffers.values())

    @property
    def size(self):
        return sum(buffer.size for buffer in self._buffers.values())

    @property
    def sizes_by_task(self):
        return {task: buffer.size for task, buffer in self._buffers.items()}

    @property
    def ready(self):
        return all(buffer.size >= self._batch_size_per_task for buffer in self._buffers.values())

    def add(self, td, count_episode=True, *, task=None):
        if task is None:
            if self._task_count != 1:
                raise ValueError("Multi-task replay insertion requires an explicit task name.")
            task = self.task_names[0]
        task = str(task)
        if task not in self._buffers:
            raise KeyError(f"Unknown replay task {task!r}; expected one of {self.task_names!r}.")
        self._buffers[task].add(td, count_episode=count_episode)
        return self.num_eps

    def sample_task_batches(self):
        if not self.ready:
            sizes = ", ".join(f"{task}={size}" for task, size in self.sizes_by_task.items())
            raise ValueError(
                f"Every task replay buffer needs {self._batch_size_per_task} transitions; got {sizes}."
            )
        return {task: self._buffers[task].sample() for task in self.task_names}

    @staticmethod
    def combine_task_batches(by_task):
        batches = list(by_task.values())
        if not batches:
            raise ValueError("Cannot combine an empty set of task batches.")
        observations = Batch.from_data_list(
            [graph for batch in batches for graph in batch[0].to_data_list()]
        )
        next_observations = Batch.from_data_list(
            [graph for batch in batches for graph in batch[4].to_data_list()]
        )
        actions = torch.cat([batch[1] for batch in batches], dim=0)
        rewards = torch.cat([batch[2] for batch in batches], dim=0)
        terminated = torch.cat([batch[3] for batch in batches], dim=0)
        return observations, actions, rewards, terminated, next_observations

    def sample_with_tasks(self):
        by_task = self.sample_task_batches()
        return ReplayBatch(combined=self.combine_task_batches(by_task), by_task=by_task)

    def sample(self):
        return self.sample_with_tasks().combined

    def state_dict(self):
        return {
            "format_version": 2,
            "task_names": list(self.task_names),
            "batch_size": self._batch_size,
            "batch_size_per_task": self._batch_size_per_task,
            "capacity_per_task": self._capacity_per_task,
            "buffers": {
                task: buffer.state_dict() for task, buffer in self._buffers.items()
            },
        }

    def load_state_dict(self, state_dict):
        saved_tasks = list(state_dict.get("task_names", []))
        if saved_tasks != self.task_names:
            raise ValueError(
                f"Checkpoint replay tasks {saved_tasks!r} do not match configured tasks {self.task_names!r}."
            )
        expected_layout = (
            self._batch_size,
            self._batch_size_per_task,
            self._capacity_per_task,
        )
        saved_layout = (
            int(state_dict.get("batch_size", -1)),
            int(state_dict.get("batch_size_per_task", -1)),
            int(state_dict.get("capacity_per_task", -1)),
        )
        if saved_layout != expected_layout:
            raise ValueError(
                f"Checkpoint replay layout {saved_layout!r} does not match configured layout "
                f"{expected_layout!r}."
            )
        saved_buffers = state_dict.get("buffers", {})
        if list(saved_buffers) != self.task_names:
            raise ValueError("Checkpoint replay buffer ordering does not match the configured task ordering.")
        for task in self.task_names:
            self._buffers[task].load_state_dict(saved_buffers[task])


Buffer = GNNBuffer
