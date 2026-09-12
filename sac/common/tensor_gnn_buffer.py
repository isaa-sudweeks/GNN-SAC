"""Tensorized, topology-balanced graph replay backed by TorchRL storage."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import torch
from tensordict import TensorDict
from torch_geometric.data import Batch, Data
from torchrl.data import LazyTensorStorage, ReplayBufferEnsemble, TensorDictReplayBuffer

from common.config_utils import round_to_nearest_multiple
from common.gnn_buffer import ReplayBatch
from common.graph_transforms import (
    graph_feature_flags,
    graph_structure_signature,
    physical_node_mask,
    policy_action_mask,
    prepare_graph,
)


_GIB = 1024 ** 3
_SUPPORTED_GRAPH_FIELDS = {"x", "edge_index", "action_mask", "rigidity", "edge_role"}


@dataclass(frozen=True)
class _PackedReplaySample:
    task: str
    values: TensorDict
    static: dict


def _task_names(cfg) -> list[str]:
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
    result = list(dict.fromkeys(candidates))
    if not result:
        raise ValueError("Task-balanced replay requires at least one task.")
    return result


def _move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).contiguous().clone()
    if isinstance(value, Mapping):
        return type(value)((key, _move_tree(item, device)) for key, item in value.items())
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    return deepcopy(value)


class _TensorTaskBuffer:
    def __init__(self, cfg, task: str, capacity: int, batch_size: int, storage_device):
        self.cfg = cfg
        self.task = task
        self._device = torch.device(getattr(cfg, "device", "cuda"))
        self._storage_device = torch.device(storage_device)
        self._capacity = int(capacity)
        self._batch_size = int(batch_size)
        self._num_eps = 0
        self._size = 0
        self._idx = 0
        self._graph_signature = None
        self._static = None
        self._buffer = None

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return self._size

    @property
    def num_eps(self):
        return self._num_eps

    @property
    def storage_device(self):
        return self._storage_device

    def allocated_bytes(self):
        if self._buffer is None:
            return 0
        storage = self._buffer.storage._storage
        return sum(value.numel() * value.element_size() for value in storage.values())

    @staticmethod
    def _graph_sequence(obs):
        if isinstance(obs, Batch):
            return obs.to_data_list()
        if isinstance(obs, Data):
            raise ValueError("Graph replay add() needs an episode of observations, not a single Data object.")
        if isinstance(obs, (list, tuple)) and all(isinstance(item, Data) for item in obs):
            return list(obs)
        raise TypeError(f"Unsupported graph observation sequence type: {type(obs)!r}")

    @staticmethod
    def _transition_sequence(value):
        if isinstance(value, torch.Tensor):
            return list(value.unbind(0))
        if isinstance(value, (list, tuple)):
            return list(value)
        raise TypeError(f"Unsupported transition field type: {type(value)!r}")

    def _initialize(self, graph: Data, action: torch.Tensor) -> None:
        extra = set(graph.keys()) - _SUPPORTED_GRAPH_FIELDS
        if extra:
            raise ValueError(f"Tensor replay does not support graph fields: {sorted(extra)!r}.")
        mask = getattr(graph, "action_mask", None)
        role = getattr(graph, "edge_role", None)
        rigidity = getattr(graph, "rigidity", None)
        template = Data(x=torch.zeros_like(graph.x), edge_index=graph.edge_index.detach().clone())
        if mask is not None:
            template.action_mask = mask.detach().clone().bool()
        if role is not None:
            template.edge_role = role.detach().clone().long()
        if rigidity is not None:
            template.rigidity = torch.zeros_like(torch.as_tensor(rigidity).reshape(1)).float()
        prepared = prepare_graph(
            template,
            use_virtual_node=bool(getattr(self.cfg, "use_virtual_node", False)),
            **graph_feature_flags(self.cfg),
        )
        self._static = {
            "raw_node_count": int(graph.x.size(0)),
            "raw_feature_dim": int(graph.x.size(1)),
            "action_shape": list(action.shape),
            "edge_index": graph.edge_index.detach().cpu().long().contiguous(),
            "action_mask": None if mask is None else mask.detach().cpu().bool().contiguous(),
            "edge_role": None if role is None else role.detach().cpu().long().contiguous(),
            "has_rigidity": rigidity is not None,
            "prepared_edge_index": prepared.edge_index.detach().cpu().long().contiguous(),
            "prepared_action_mask": (
                prepared.action_mask.detach().cpu().bool().contiguous()
                if "action_mask" in prepared else None
            ),
            "prepared_physical_node_mask": (
                prepared.physical_node_mask.detach().cpu().bool().contiguous()
                if "physical_node_mask" in prepared else None
            ),
            "prepared_x_template": prepared.x.detach().cpu().contiguous(),
            "prepared_edge_attr_template": (
                prepared.edge_attr.detach().cpu().contiguous() if "edge_attr" in prepared else None
            ),
            "prepared_edge_role": (
                prepared.edge_role.detach().cpu().long().contiguous() if "edge_role" in prepared else None
            ),
            "prepared_edge_type": (
                prepared.edge_type.detach().cpu().long().contiguous() if "edge_type" in prepared else None
            ),
            "raw_edge_count": int(graph.edge_index.size(1)),
            "signature": graph_structure_signature(graph),
        }
        self._buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self._capacity, device=self._storage_device),
            batch_size=self._batch_size,
            pin_memory=self._storage_device.type == "cpu" and self._device.type == "cuda",
        )

    def _validate_graph(self, graph: Data) -> None:
        if self._static is None:
            return
        if graph_structure_signature(graph) != self._static["signature"]:
            raise ValueError("Replay observation topology or action ordering changed within a task.")
        if (getattr(graph, "rigidity", None) is not None) != self._static["has_rigidity"]:
            raise ValueError("Replay observation rigidity field presence changed within a task.")
        role = getattr(graph, "edge_role", None)
        saved_role = self._static["edge_role"]
        if (role is None) != (saved_role is None):
            raise ValueError("Replay observation edge-role field presence changed within a task.")
        if role is not None and not torch.equal(role.detach().cpu().long(), saved_role):
            raise ValueError("Replay observation edge roles changed within a task.")

    def add(self, td, count_episode=True):
        if isinstance(td, list):
            observations = [step["obs"] for step in td]
            actions = [step["action"].squeeze(0) for step in td][1:]
            rewards = [step["reward"].squeeze(0) for step in td][1:]
            terminated = [step["terminated"].squeeze(0) for step in td][1:]
        else:
            observations = self._graph_sequence(td["obs"])
            actions = self._transition_sequence(td["action"])[1:]
            rewards = self._transition_sequence(td["reward"])[1:]
            terminated = self._transition_sequence(td["terminated"])[1:]
        current, following = observations[:-1], observations[1:]
        count = len(current)
        if not (len(actions) == len(rewards) == len(terminated) == count):
            raise ValueError("Graph replay transition fields have inconsistent lengths.")
        if count > self._capacity:
            current, following = current[-self._capacity:], following[-self._capacity:]
            actions, rewards, terminated = actions[-self._capacity:], rewards[-self._capacity:], terminated[-self._capacity:]
            count = self._capacity
        if not count:
            return self._num_eps
        first_action = torch.as_tensor(actions[0]).float()
        if self._buffer is None:
            self._initialize(current[0], first_action)
        for graph in (*current, *following):
            self._validate_graph(graph)
        values = self._build_values(current, following, actions, rewards, terminated)
        self._buffer.extend(values)
        self._idx = (self._idx + count) % self._capacity
        self._size = min(self._size + count, self._capacity)
        self._num_eps += int(bool(count_episode))
        return self._num_eps

    def _build_values(self, current, following, actions, rewards, terminated):
        count = len(current)
        action_values = torch.stack([torch.as_tensor(value).detach().float() for value in actions])
        if list(action_values.shape[1:]) != self._static["action_shape"]:
            raise ValueError("Replay action shape changed within a task.")
        def rigidity_values(graphs):
            if not self._static["has_rigidity"]:
                return torch.zeros((len(graphs), 1), dtype=torch.float32)
            return torch.stack([
                torch.as_tensor(graph.rigidity).detach().float().reshape(1) for graph in graphs
            ])
        return TensorDict(
            {
                "obs_x": torch.stack([graph.x.detach().float() for graph in current]),
                "next_obs_x": torch.stack([graph.x.detach().float() for graph in following]),
                "obs_rigidity": rigidity_values(current),
                "next_obs_rigidity": rigidity_values(following),
                "action": action_values,
                "reward": torch.stack([
                    torch.as_tensor(value).detach().float().reshape(1) for value in rewards
                ]),
                "terminated": torch.stack([
                    torch.as_tensor(value).detach().float().reshape(1) for value in terminated
                ]),
            },
            batch_size=[count],
        ).to(self._storage_device)

    def load_legacy_state_dict(self, state, *, chunk_size=4096):
        if int(state["capacity"]) != self._capacity:
            raise ValueError("Legacy replay capacity does not match tensor replay capacity.")
        size, index = int(state["size"]), int(state["idx"])
        if not size:
            self._num_eps, self._size, self._idx = int(state["num_eps"]), 0, index
            return
        first = 0
        self._initialize(state["obs"][first], torch.as_tensor(state["action"][first]).float())
        limit = self._capacity if size == self._capacity else size
        for start in range(0, limit, int(chunk_size)):
            stop = min(start + int(chunk_size), limit)
            current = state["obs"][start:stop]
            following = state["next_obs"][start:stop]
            for graph in (*current, *following):
                self._validate_graph(graph)
            values = self._build_values(
                current, following, state["action"][start:stop],
                state["reward"][start:stop], state["terminated"][start:stop],
            )
            self._buffer.extend(values)
        self._num_eps, self._size, self._idx = int(state["num_eps"]), size, index
        self._buffer.writer._cursor_value.value = index

    def set_graph_signature(self, signature):
        self._graph_signature = signature
        if self._static is not None and self._static["signature"] != signature:
            raise ValueError("Replay observation topology or action ordering differs from teacher.")

    def sample_raw(self, performance_profiler=None):
        if self._size < self._batch_size:
            raise ValueError(f"Replay buffer has {self._size} transitions, need batch_size={self._batch_size}.")
        subphase = performance_profiler.subphase if performance_profiler is not None else None
        context = subphase("balanced_gather") if subphase is not None else torch.no_grad()
        with context:
            indices = torch.randint(self._size, (self._batch_size,), device="cpu")
            values = self._buffer[indices.to(self._storage_device)]
        return _PackedReplaySample(self.task, values, self._static)

    def state_dict(self):
        replay = None if self._buffer is None else _move_tree(self._buffer.state_dict(), "cpu")
        return {
            "format_version": 3,
            "capacity": self._capacity,
            "batch_size": self._batch_size,
            "num_eps": self._num_eps,
            "size": self._size,
            "idx": self._idx,
            "storage_device": str(self._storage_device),
            "static": deepcopy(self._static),
            "replay_buffer": replay,
        }

    def load_state_dict(self, state):
        if int(state.get("format_version", 0)) != 3:
            raise ValueError("Tensor replay requires format_version=3 task state.")
        if int(state["capacity"]) != self._capacity:
            raise ValueError("Checkpoint replay capacity does not match current capacity.")
        self._num_eps, self._size, self._idx = int(state["num_eps"]), int(state["size"]), int(state["idx"])
        self._static = deepcopy(state["static"])
        if self._graph_signature is not None and self._static is not None:
            if self._static["signature"] != self._graph_signature:
                raise ValueError("Checkpoint replay topology differs from teacher.")
        if state["replay_buffer"] is not None:
            self._buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(self._capacity, device=self._storage_device),
                batch_size=self._batch_size,
                pin_memory=self._storage_device.type == "cpu" and self._device.type == "cuda",
            )
            saved_fields = state["replay_buffer"]["_storage"]["_storage"]
            fields = {
                key: value[:self._size].to(self._storage_device)
                for key, value in saved_fields.items()
                if key != "index"
            }
            self._buffer.extend(TensorDict(fields, batch_size=[self._size], device=self._storage_device))
            # A full ring may have a cursor other than zero. TorchRL owns the
            # circular writer, but this compatibility cursor is checkpointed by
            # GNN-SAC and must be restored exactly.
            self._buffer.writer._cursor_value.value = self._idx


class TensorGNNBuffer:
    """Task-balanced tensor replay retaining the public GNNBuffer contract."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.task_names = _task_names(cfg)
        self._task_count = len(self.task_names)
        total_capacity = round_to_nearest_multiple(cfg.buffer_size, self._task_count, name="buffer_size")
        batch_size = round_to_nearest_multiple(cfg.batch_size, self._task_count, name="batch_size")
        cfg.buffer_size, cfg.batch_size = total_capacity, batch_size
        self._batch_size = batch_size
        self._batch_size_per_task = batch_size // self._task_count
        self._capacity_per_task = total_capacity // self._task_count
        self._placements, self.placement_metadata = self._plan_placements()
        self._buffers = {
            task: _TensorTaskBuffer(
                cfg, task, self._capacity_per_task, self._batch_size_per_task,
                self._placements[task],
            )
            for task in self.task_names
        }
        self._ensemble = None
        placements = ", ".join(f"{task}={device}" for task, device in self._placements.items())
        print(f"Tensor replay placement: {placements}", flush=True)

    def _estimated_task_bytes(self, node_count: int) -> int:
        obs_dim = int(getattr(self.cfg, "obs_dim", 6))
        action_dim = int(getattr(self.cfg, "action_dim", 1))
        scalars = 2 * node_count * obs_dim + node_count * action_dim + 4
        float_bytes = scalars * torch.tensor([], dtype=torch.float32).element_size()
        index_bytes = torch.tensor([], dtype=torch.long).element_size()
        return self._capacity_per_task * (float_bytes + index_bytes)

    def _plan_placements(self):
        mode = str(getattr(self.cfg, "replay_storage", "auto")).lower()
        if mode not in {"cpu_pinned", "cuda", "auto"}:
            raise ValueError("replay_storage must be cpu_pinned, cuda, or auto.")
        node_counts = list(getattr(self.cfg, "node_counts", []) or [])
        if len(node_counts) != self._task_count:
            node_counts = [int(getattr(self.cfg, "num_nodes", 1))] * self._task_count
        estimates = {task: self._estimated_task_bytes(int(nodes)) for task, nodes in zip(self.task_names, node_counts)}
        cuda_available = torch.cuda.is_available() and torch.device(getattr(self.cfg, "device", "cpu")).type == "cuda"
        total_memory = free_memory = 0
        if cuda_available:
            free_memory, total_memory = torch.cuda.mem_get_info()
        fraction = float(getattr(self.cfg, "replay_gpu_fraction", 0.20))
        max_bytes = float(getattr(self.cfg, "replay_gpu_max_gb", 8.0)) * _GIB
        reserve = float(getattr(self.cfg, "replay_gpu_reserve_gb", 12.0)) * _GIB
        if not 0 <= fraction <= 1:
            raise ValueError("replay_gpu_fraction must be between 0 and 1.")
        if max_bytes < 0 or reserve < 0:
            raise ValueError("Replay GPU size and reserve limits must be non-negative.")
        budget = max(0, int(min(total_memory * fraction, max_bytes, max(0, free_memory - reserve)))) if cuda_available else 0
        placements = {task: "cpu" for task in self.task_names}
        used = 0
        if mode == "cuda":
            if not cuda_available:
                raise ValueError("CUDA replay requested but CUDA is unavailable.")
            required = sum(estimates.values())
            if required > budget:
                raise MemoryError(f"CUDA replay needs {required / _GIB:.2f} GiB but budget is {budget / _GIB:.2f} GiB.")
            placements = {task: "cuda" for task in self.task_names}
            used = required
        elif mode == "auto" and cuda_available:
            for task in sorted(self.task_names, key=lambda name: (estimates[name], name)):
                if used + estimates[task] <= budget:
                    placements[task] = "cuda"
                    used += estimates[task]
        metadata = {
            "mode": mode,
            "estimated_bytes": estimates,
            "cuda_budget_bytes": budget,
            "estimated_cuda_bytes": used,
            "free_cuda_bytes_at_init": free_memory,
            "total_cuda_bytes": total_memory,
            "placements": dict(placements),
            "remaining_cuda_headroom_bytes": max(0, free_memory - used),
            "fallback_reasons": {
                task: (
                    "cuda_unavailable" if not cuda_available
                    else "storage_mode_cpu_pinned" if mode == "cpu_pinned"
                    else "gpu_budget_exhausted"
                )
                for task, placement in placements.items() if placement == "cpu"
            },
        }
        return placements, metadata

    def runtime_storage_metadata(self):
        metadata = deepcopy(self.placement_metadata)
        metadata["actual_allocated_bytes"] = {
            task: buffer.allocated_bytes() for task, buffer in self._buffers.items()
        }
        if torch.cuda.is_available() and torch.device(getattr(self.cfg, "device", "cpu")).type == "cuda":
            free, total = torch.cuda.mem_get_info()
            metadata["current_free_cuda_bytes"] = int(free)
            metadata["current_total_cuda_bytes"] = int(total)
        return metadata

    @property
    def capacity(self):
        return sum(buffer.capacity for buffer in self._buffers.values())

    @property
    def size(self):
        return sum(buffer.size for buffer in self._buffers.values())

    @property
    def num_eps(self):
        return sum(buffer.num_eps for buffer in self._buffers.values())

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
        if str(task) not in self._buffers:
            raise KeyError(f"Unknown replay task {task!r}; expected one of {self.task_names!r}.")
        self._buffers[str(task)].add(td, count_episode=count_episode)
        return self.num_eps

    def set_task_graph_signatures(self, signatures):
        if set(signatures) != set(self.task_names):
            raise ValueError("Replay signature tasks do not match configured tasks.")
        for task, signature in signatures.items():
            self._buffers[task].set_graph_signature(signature)

    supports_replay_profiling = True

    def _sample_raw_by_task(self, performance_profiler=None):
        if not self.ready:
            raise ValueError("Every task replay buffer must contain one full sub-batch.")
        return {
            task: self._buffers[task].sample_raw(performance_profiler=performance_profiler)
            for task in self.task_names
        }

    def _sample_raw_by_task_ensemble(self):
        if not self.ready:
            raise ValueError("Every task replay buffer must contain one full sub-batch.")
        devices = {buffer.storage_device for buffer in self._buffers.values()}
        if len(devices) != 1:
            raise ValueError("ReplayBufferEnsemble evaluation requires one common storage device.")
        if self._ensemble is None:
            self._ensemble = ReplayBufferEnsemble(
                *(buffer._buffer for buffer in self._buffers.values()),
                sample_from_all=True,
                batch_size=self._batch_size,
            )
        values = self._ensemble.sample()
        return {
            task: _PackedReplaySample(task, values[index], self._buffers[task]._static)
            for index, task in enumerate(self.task_names)
        }

    def _prepared_dense(self, sample: _PackedReplaySample, key: str):
        values, static = sample.values, sample.static
        raw_x = values[key]
        rigidity_key = "obs_rigidity" if key == "obs_x" else "next_obs_rigidity"
        count, raw_nodes, raw_features = raw_x.shape
        device = raw_x.device
        template_x = static["prepared_x_template"].to(device)
        x = template_x.unsqueeze(0).expand(count, -1, -1).clone()
        x[:, :raw_nodes, :raw_features] = raw_x
        if bool(getattr(self.cfg, "use_virtual_node", False)):
            x[:, -1, -1] = values[rigidity_key].reshape(-1)
        template_edge_attr = static["prepared_edge_attr_template"]
        edge_attr = None
        if template_edge_attr is not None:
            edge_attr = template_edge_attr.to(device).unsqueeze(0).expand(count, -1, -1).clone()
            if graph_feature_flags(self.cfg)["use_edge_distance"]:
                edge_index = static["edge_index"].to(device)
                distance = torch.linalg.vector_norm(
                    raw_x[:, edge_index[0], :3] - raw_x[:, edge_index[1], :3], dim=-1
                )
                edge_attr[:, :static["raw_edge_count"], -1] = distance
        return x, edge_attr

    def _collate(self, samples, *, performance_profiler=None, subphase_name="combined_collation_transfer"):
        samples = list(samples)
        context = (
            performance_profiler.subphase(subphase_name)
            if performance_profiler is not None else torch.no_grad()
        )
        with context:
            target = torch.device(getattr(self.cfg, "device", "cpu"))
            x_parts, next_x_parts, edge_parts, next_edge_parts = [], [], [], []
            mask_parts, physical_parts, graph_ids, ptr = [], [], [], [0]
            action_parts, reward_parts, terminated_parts = [], [], []
            edge_attr_parts, next_edge_attr_parts, role_parts, type_parts = [], [], [], []
            graph_offset = node_offset = 0
            include_mask = include_physical = include_edge_attr = include_role = include_type = False
            rigidity_parts, next_rigidity_parts = [], []
            include_rigidity = False
            for sample in samples:
                values = sample.values
                if values.device != target:
                    values = values.to(target, non_blocking=values.device.type == "cpu" and target.type == "cuda")
                    sample = _PackedReplaySample(sample.task, values, sample.static)
                x, edge_attr = self._prepared_dense(sample, "obs_x")
                next_x, next_edge_attr = self._prepared_dense(sample, "next_obs_x")
                count, nodes = int(x.size(0)), int(x.size(1))
                static_edge = sample.static["prepared_edge_index"].to(target)
                edges = int(static_edge.size(1))
                offsets = torch.arange(count, device=target).view(-1, 1, 1) * nodes + node_offset
                packed_edge = (static_edge.view(1, 2, edges) + offsets).permute(1, 0, 2).reshape(2, -1)
                x_parts.append(x.flatten(0, 1))
                next_x_parts.append(next_x.flatten(0, 1))
                edge_parts.append(packed_edge)
                next_edge_parts.append(packed_edge)
                prepared_mask = sample.static["prepared_action_mask"]
                if prepared_mask is not None:
                    include_mask = True
                    mask_parts.append(prepared_mask.to(target).repeat(count))
                physical = sample.static["prepared_physical_node_mask"]
                if physical is not None:
                    include_physical = True
                    physical_parts.append(physical.to(target).repeat(count))
                graph_ids.append(torch.arange(graph_offset, graph_offset + count, device=target).repeat_interleave(nodes))
                ptr.extend(node_offset + i * nodes for i in range(1, count + 1))
                action_parts.append(values["action"].flatten(0, 1))
                reward_parts.append(values["reward"])
                terminated_parts.append(values["terminated"])
                if edge_attr is not None:
                    include_edge_attr = True
                    edge_attr_parts.append(edge_attr.flatten(0, 1))
                    next_edge_attr_parts.append(next_edge_attr.flatten(0, 1))
                prepared_role = sample.static["prepared_edge_role"]
                if prepared_role is not None:
                    include_role = True
                    role_parts.append(prepared_role.to(target).repeat(count))
                prepared_type = sample.static["prepared_edge_type"]
                if prepared_type is not None:
                    include_type = True
                    type_parts.append(prepared_type.to(target).repeat(count))
                if sample.static["has_rigidity"]:
                    include_rigidity = True
                    rigidity_parts.append(values["obs_rigidity"].reshape(-1))
                    next_rigidity_parts.append(values["next_obs_rigidity"].reshape(-1))
                graph_offset += count
                node_offset += count * nodes
            def make_batch(x_values, edges, attrs, rigidities):
                kwargs = {
                    "x": torch.cat(x_values), "edge_index": torch.cat(edges, dim=1),
                    "batch": torch.cat(graph_ids),
                    "ptr": torch.tensor(ptr, device=target, dtype=torch.long),
                }
                if include_mask:
                    kwargs["action_mask"] = torch.cat(mask_parts)
                if include_physical:
                    kwargs["physical_node_mask"] = torch.cat(physical_parts)
                if include_edge_attr:
                    kwargs["edge_attr"] = torch.cat(attrs)
                if include_role:
                    kwargs["edge_role"] = torch.cat(role_parts)
                if include_type:
                    kwargs["edge_type"] = torch.cat(type_parts)
                if include_rigidity:
                    kwargs["rigidity"] = torch.cat(rigidities)
                batch = Batch(**kwargs)
                batch._num_graphs = graph_offset
                object.__setattr__(batch, "_physical_node_count_cache", int(physical_node_mask(batch).sum()))
                object.__setattr__(batch, "_policy_action_count_cache", int(policy_action_mask(batch).sum()))
                return batch
            return (
                make_batch(x_parts, edge_parts, edge_attr_parts, rigidity_parts),
                torch.cat(action_parts), torch.cat(reward_parts), torch.cat(terminated_parts),
                make_batch(next_x_parts, next_edge_parts, next_edge_attr_parts, next_rigidity_parts),
            )

    def sample(self, performance_profiler=None):
        return self._collate(self._sample_raw_by_task(performance_profiler).values(), performance_profiler=performance_profiler)

    def sample_task_batches(self, performance_profiler=None):
        raw = self._sample_raw_by_task(performance_profiler)
        return {
            task: self._collate([sample], performance_profiler=performance_profiler, subphase_name="task_collation_transfer")
            for task, sample in raw.items()
        }

    def sample_with_tasks(self, performance_profiler=None):
        raw = self._sample_raw_by_task(performance_profiler)
        combined = self._collate(raw.values(), performance_profiler=performance_profiler)
        by_task = {
            task: self._collate([sample], performance_profiler=performance_profiler, subphase_name="task_collation_transfer")
            for task, sample in raw.items()
        }
        return ReplayBatch(combined=combined, by_task=by_task)

    def sample_ensemble(self):
        """Benchmark ReplayBufferEnsemble without changing the training path."""
        return self._collate(self._sample_raw_by_task_ensemble().values())

    def state_dict(self):
        return {
            "format_version": 3,
            "task_names": list(self.task_names),
            "batch_size": self._batch_size,
            "batch_size_per_task": self._batch_size_per_task,
            "capacity_per_task": self._capacity_per_task,
            "placement_metadata": self.runtime_storage_metadata(),
            "buffers": {task: buffer.state_dict() for task, buffer in self._buffers.items()},
        }

    def load_state_dict(self, state):
        if int(state.get("format_version", 0)) != 3:
            raise ValueError("Tensor replay requires a format_version=3 checkpoint.")
        if list(state.get("task_names", [])) != self.task_names:
            raise ValueError("Checkpoint replay tasks do not match configured tasks.")
        saved_layout = (int(state["batch_size"]), int(state["batch_size_per_task"]), int(state["capacity_per_task"]))
        current_layout = (self._batch_size, self._batch_size_per_task, self._capacity_per_task)
        if saved_layout != current_layout:
            raise ValueError("Checkpoint replay layout does not match configured layout.")
        for task in self.task_names:
            self._buffers[task].load_state_dict(state["buffers"][task])

    def load_legacy_state_dict(self, state, *, chunk_size=4096):
        if int(state.get("format_version", 0)) != 2:
            raise ValueError("Legacy conversion requires GNN replay format_version=2.")
        if list(state.get("task_names", [])) != self.task_names:
            raise ValueError("Legacy checkpoint replay tasks do not match configured tasks.")
        for task in self.task_names:
            self._buffers[task].load_legacy_state_dict(state["buffers"][task], chunk_size=chunk_size)


def make_gnn_buffer(cfg):
    backend = str(getattr(cfg, "replay_backend", "legacy")).lower()
    if backend == "legacy":
        from common.gnn_buffer import GNNBuffer
        return GNNBuffer(cfg)
    if backend == "torchrl_tensor":
        return TensorGNNBuffer(cfg)
    raise ValueError("replay_backend must be legacy or torchrl_tensor.")
