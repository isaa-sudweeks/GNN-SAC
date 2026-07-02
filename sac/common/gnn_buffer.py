import torch
from torch_geometric.data import Batch, Data


class GNNBuffer:
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


Buffer = GNNBuffer
