"""
Code adapted from https://github.com/nicklashansen/tdmpc2
"""
import torch 
from tensordict.tensordict import TensorDict 
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler #What are the other sampler options?

class Buffer():
    """
    Replay buffer for SAC. Based on torchrl.
    """
    def __init__(self, cfg):
        self.cfg = cfg 
        self._device = torch.device(getattr(cfg, 'device', 'cuda'))
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None, # Seems weird to me that this is None
            strict_length=True,
            cache_values=cfg.multitask
        )
        self._batch_size = cfg.batch_size
        self._num_eps = 0
    
    @property
    def capacity(self):
        """Returns the capacity of the replay buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Returns the number of episodes stored in the replay buffer."""
        return self._num_eps
    
    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        # I do not know what pin_memory or prefetch do.
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size
        )
    
    def _init(self, tds):
        """
        Initialize the replay buffer. Use the first episode to estimate storage requirements.
        """
        print(f'Buffer capacity: {self.capacity:,}')

        bytes_per_step = sum([
            (v.numel()*v.element_size() if not isinstance(v, TensorDict) else sum([x.numel()*x.element_size() for x in v.values()])) for v in tds.values()
        ]) / len(tds)

        total_bytes = bytes_per_step * self._capcity 
        print(f'Storage required: {total_bytes/1e-9:.2f} GB')

        # Decide wheter to use CUDA or CPU memory 
        if self._device.type == 'cuda':
            mem_free, _ = torch.cuda.mem_get_info()
            storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
        else:
            storage_device = 'cpu'
        
        print(f'Using {storage_device} memory')
        self._storage_device = torch.device(storage_device)

        return self._reserve_buffer(LazyTensorStorage(self.capacity, device=self._storage_device))

    def load(self, td):
        """
        Load a batch of episodes into the buffer. 
        """
        num_new_eps = len(td)
        episode_idx = torch.arange(self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64)
        td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[-1])
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        td = td.reshape(td.shape[0]*td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps 
        return self._num_eps
    
    def add(self, td):
        """
        Add an episode to the buffer.
        """
        td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps 

    def _prepare_batch(self, td: TensorDict):
        """
        Prepare a batch of data for training.
        Expects 'td' to be a TensorDict with size TxB.
        """
        task = td.get('task', None)
        if task is not None:
            task = task[0].contiguous().to('cpu', dtype=torch.long)
            if (task < 0).any() or (task >= len(self.cfg.tasks)).any():
                raise ValueError(f'Sampled invalid task ids: {task.tolist()}')
            if self._device.type != 'mps':
                task = task.to(self._device, non_blocking=True)
        td = td.select("obs", "action", "reward", strict=False).to(self._device, non_blocking=True)
        obs = td.get("obs").contiguous()
        action = td.get('action')[1:].contiguous()
        reward = td.get("reward")[1:].contiguous()

        return obs, action, reward, task 

    def sample(self):
        """
        Sample a batch of subsequences from the buffer.
        """

        td = self._buffer.sample().view(-1, ).permute(1,0) # Okay this is something different because we have no horizon here
        return self._prepare_batch(td)

            
        