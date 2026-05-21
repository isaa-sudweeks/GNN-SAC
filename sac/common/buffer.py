import torch


class Buffer:
    """Circular transition replay buffer for SAC."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device(getattr(cfg, "device", "cuda"))
        self._capacity = int(min(cfg.buffer_size, cfg.steps))
        self._batch_size = int(cfg.batch_size)
        self._num_eps = 0
        self._size = 0
        self._idx = 0
        self._storage_device = torch.device("cpu")
        self._obs = None
        self._next_obs = None
        self._action = None
        self._reward = None
        self._terminated = None

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps

    @property
    def size(self):
        return self._size

    def _init(self, obs, action):
        print(f"Buffer capacity: {self.capacity:,}")
        self._obs = torch.empty((self._capacity, *obs.shape[1:]), dtype=obs.dtype, device=self._storage_device)
        self._next_obs = torch.empty_like(self._obs)
        self._action = torch.empty((self._capacity, *action.shape[1:]), dtype=action.dtype, device=self._storage_device)
        self._reward = torch.empty((self._capacity, 1), dtype=torch.float32, device=self._storage_device)
        self._terminated = torch.empty((self._capacity, 1), dtype=torch.float32, device=self._storage_device)

        bytes_per_step = (
            self._obs[0].numel() * self._obs.element_size()
            + self._next_obs[0].numel() * self._next_obs.element_size()
            + self._action[0].numel() * self._action.element_size()
            + self._reward[0].numel() * self._reward.element_size()
            + self._terminated[0].numel() * self._terminated.element_size()
        )
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes / 1e9:.2f} GB")
        print(f"Using {self._storage_device} memory")

    def add(self, td):
        if isinstance(td["obs"], dict):
            raise NotImplementedError("Graph/dict observations need a graph replay buffer.")

        obs = td["obs"][:-1].float().cpu()
        next_obs = td["obs"][1:].float().cpu()
        action = td["action"][1:].float().cpu()
        reward = td["reward"][1:].float().view(-1, 1).cpu()
        terminated = td["terminated"][1:].float().view(-1, 1).cpu()

        if self._obs is None:
            self._init(obs, action)

        n = obs.shape[0]
        if n > self._capacity:
            obs = obs[-self._capacity:]
            next_obs = next_obs[-self._capacity:]
            action = action[-self._capacity:]
            reward = reward[-self._capacity:]
            terminated = terminated[-self._capacity:]
            n = self._capacity

        first = min(n, self._capacity - self._idx)
        second = n - first
        self._write_slice(self._idx, self._idx + first, obs[:first], next_obs[:first], action[:first], reward[:first], terminated[:first])
        if second:
            self._write_slice(0, second, obs[first:], next_obs[first:], action[first:], reward[first:], terminated[first:])

        self._idx = (self._idx + n) % self._capacity
        self._size = min(self._size + n, self._capacity)
        self._num_eps += 1
        return self._num_eps

    def _write_slice(self, start, end, obs, next_obs, action, reward, terminated):
        self._obs[start:end].copy_(obs)
        self._next_obs[start:end].copy_(next_obs)
        self._action[start:end].copy_(action)
        self._reward[start:end].copy_(reward)
        self._terminated[start:end].copy_(terminated)

    def sample(self):
        if self._size < self._batch_size:
            raise ValueError(f"Replay buffer has {self._size} transitions, need batch_size={self._batch_size}.")
        idx = torch.randint(self._size, (self._batch_size,), device=self._storage_device)
        return (
            self._obs[idx].to(self._device, non_blocking=True),
            self._action[idx].to(self._device, non_blocking=True),
            self._reward[idx].to(self._device, non_blocking=True),
            self._terminated[idx].to(self._device, non_blocking=True),
            self._next_obs[idx].to(self._device, non_blocking=True),
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
