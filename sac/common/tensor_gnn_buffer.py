"""Tensorized, topology-balanced graph replay backed by TorchRL storage."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import torch
from tensordict import TensorDict
from torch_geometric.data import Batch, Data
from torchrl.data import LazyTensorStorage, ReplayBufferEnsemble, TensorDictReplayBuffer

from env.mujoco_gen.topology_envs import fan_out_broken_regime_tasks
from env.mujoco_gen.tube_physics import TUBE_PHYSICS_FIELDS

from common.config_utils import round_to_nearest_multiple
from common.finite_checks import require_finite
from common.gnn_buffer import ReplayBatch
from common.graph_transforms import (
    graph_feature_flags,
    graph_signature_compatible,
    graph_structure_signature,
    physical_node_mask,
    policy_action_mask,
    prepare_graph,
    tube_components,
    tube_physics_features,
)


_GIB = 1024 ** 3
_SUPPORTED_GRAPH_FIELDS = {"x", "edge_index", "action_mask", "rigidity", "edge_role", "edge_direction", *TUBE_PHYSICS_FIELDS}


@dataclass(frozen=True)
class _PackedReplaySample:
    task: str
    values: TensorDict
    static: dict


def _task_names(cfg) -> list[str]:
    multitask = bool(getattr(cfg, "multitask", False))
    backend = str(getattr(cfg, "mujoco_backend", "mujoco")).lower()
    topologies = getattr(cfg, "truss_topologies", None)
    num_envs = int(getattr(cfg, "num_envs", 1))
    is_mjx_multi_topology = backend == "mjx" and topologies and len(topologies) > 1
    if is_mjx_multi_topology:
        base_task = str(getattr(cfg, "task", "truss-graph")).split(":", 1)[0]
        candidates = [f"{base_task}:{topology}" for topology in topologies]
        # MjxTopologyBucketEnv splits num_envs evenly across topologies, so
        # each task's actual parallel-slot count is the bucket size, not the
        # aggregate num_envs -- registering a broken sibling task for a
        # bucket too small to ever populate it would leave that task's
        # buffer permanently empty and stall sampling.
        slots_per_task = num_envs // len(topologies)
    elif multitask:
        # MultitaskWrapper has no per-slot regime-locking implementation;
        # never fan out a broken sibling task here (it would never receive data).
        candidates = [str(task) for task in getattr(cfg, "tasks", [])]
        slots_per_task = 1
    else:
        candidates = [str(getattr(cfg, "task", "task"))]
        slots_per_task = num_envs
    result = list(dict.fromkeys(candidates))
    if not result:
        raise ValueError("Task-balanced replay requires at least one task.")
    return fan_out_broken_regime_tasks(result, cfg, slots_per_task)


@dataclass(frozen=True)
class DenseGraphGroup:
    """Same-topology graphs stored as dense tensors plus their replay static metadata."""

    static: dict
    x: torch.Tensor
    rigidity: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None


def build_graph_static(cfg, graph: Data, action: torch.Tensor) -> dict:
    """Return the per-topology template used to rebuild prepared graphs from dense rows."""
    extra = set(graph.keys()) - _SUPPORTED_GRAPH_FIELDS
    if extra:
        raise ValueError(f"Tensor replay does not support graph fields: {sorted(extra)!r}.")
    mask = getattr(graph, "action_mask", None)
    role = getattr(graph, "edge_role", None)
    direction = getattr(graph, "edge_direction", None)
    rigidity = getattr(graph, "rigidity", None)
    template = Data(x=torch.zeros_like(graph.x), edge_index=graph.edge_index.detach().clone())
    if mask is not None:
        template.action_mask = mask.detach().clone().bool()
    if role is not None:
        template.edge_role = role.detach().clone().long()
    if direction is not None:
        template.edge_direction = direction.detach().clone().float()
    physics_metadata = {k: graph[k].detach().clone().float() for k in TUBE_PHYSICS_FIELDS if k in graph}
    for key, value in physics_metadata.items():
        template[key] = value
    if rigidity is not None:
        template.rigidity = torch.zeros_like(torch.as_tensor(rigidity).reshape(1)).float()
    prepared = prepare_graph(
        template,
        use_virtual_node=bool(getattr(cfg, "use_virtual_node", False)),
        **graph_feature_flags(cfg),
    )
    return {
        "raw_node_count": int(graph.x.size(0)),
        "raw_feature_dim": int(graph.x.size(1)),
        "action_shape": list(action.shape),
        "edge_index": graph.edge_index.detach().cpu().long().contiguous(),
        "action_mask": None if mask is None else mask.detach().cpu().bool().contiguous(),
        "edge_role": None if role is None else role.detach().cpu().long().contiguous(),
        "edge_direction": None if direction is None else direction.detach().cpu().float().contiguous(),
        "tube_physics_metadata": {k: v.cpu().contiguous() for k, v in physics_metadata.items()},
        "tube_groups": tube_components(graph) if graph_feature_flags(cfg)["use_tube_physics"] else None,
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
        "prepared_global_node_mask": (
            prepared.global_node_mask.detach().cpu().bool().contiguous()
            if "global_node_mask" in prepared else None
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


def _dense_prepared_features(cfg, group: DenseGraphGroup, device):
    """Apply ``prepare_graph`` feature augmentation to every dense row at once."""
    static, raw_x = group.static, group.x
    count, raw_nodes, raw_features = raw_x.shape
    flags = graph_feature_flags(cfg)
    template_x = static["prepared_x_template"].to(device)
    x = template_x.unsqueeze(0).expand(count, -1, -1).clone()
    x[:, :raw_nodes, :raw_features] = raw_x
    if flags["use_node_roles"]:
        if group.action_mask is not None:
            action_mask = group.action_mask.bool()
        else:
            action_mask = static["prepared_action_mask"].to(device)[:raw_nodes]
            action_mask = action_mask.unsqueeze(0).expand(count, -1)
        node_roles = torch.stack((action_mask, ~action_mask), dim=-1).to(x.dtype)
        x[:, :raw_nodes, raw_features:raw_features + 2] = node_roles
    if flags["use_tube_physics"]:
        x[:, raw_nodes + 1:, -7:-3] = tube_physics_features(
            raw_x, static["edge_index"].to(device), static["tube_groups"], static["tube_physics_metadata"])
    if bool(getattr(cfg, "use_virtual_node", False)):
        rigidity = (
            torch.zeros(count, device=device, dtype=x.dtype)
            if group.rigidity is None else group.rigidity.reshape(-1)
        )
        global_mask = static.get("prepared_global_node_mask")
        global_index = -1 if global_mask is None else int(global_mask.nonzero()[0])
        x[:, global_index, -1] = rigidity
    template_edge_attr = static["prepared_edge_attr_template"]
    edge_attr = None
    if template_edge_attr is not None:
        edge_attr = template_edge_attr.to(device).unsqueeze(0).expand(count, -1, -1).clone()
        if flags["use_edge_distance"]:
            edge_index = static["edge_index"].to(device)
            distance = torch.linalg.vector_norm(
                raw_x[:, edge_index[0], :3] - raw_x[:, edge_index[1], :3], dim=-1
            )
            edge_attr[:, :static["raw_edge_count"], -1] = distance
    return x, edge_attr


def _dense_prepared_action_mask(group: DenseGraphGroup, device):
    prepared = group.static["prepared_action_mask"]
    if prepared is None:
        return None
    count = int(group.x.size(0))
    result = prepared.to(device).unsqueeze(0).expand(count, -1).clone()
    if group.action_mask is not None:
        result[:, :group.static["raw_node_count"]] = group.action_mask.bool()
    return result


def dense_graph_batch(cfg, groups, device) -> Batch:
    """Build one prepared PyG ``Batch`` from dense same-topology graph groups.

    This is equivalent to ``Batch.from_data_list([prepare_graph(g) for g in graphs])``
    but costs a fixed number of tensor operations per topology instead of Python
    work per graph.
    """
    device = torch.device(device)
    x_parts, edge_parts, attr_parts, mask_parts, physical_parts = [], [], [], [], []
    role_parts, type_parts, rigidity_parts, graph_ids, ptr_parts = [], [], [], [], []
    global_parts = []
    ptr_parts.append(torch.zeros(1, dtype=torch.long, device=device))
    graph_offset = node_offset = 0
    for group in groups:
        # Rollout tensors may live on the simulator device (e.g. MJX on CUDA)
        # while the learner runs elsewhere; build the whole batch on `device`.
        group = replace(
            group,
            x=group.x.to(device),
            rigidity=None if group.rigidity is None else group.rigidity.to(device),
            action_mask=None if group.action_mask is None else group.action_mask.to(device),
        )
        static = group.static
        x, edge_attr = _dense_prepared_features(cfg, group, device)
        count, nodes = int(x.size(0)), int(x.size(1))
        static_edge = static["prepared_edge_index"].to(device)
        edges = int(static_edge.size(1))
        offsets = torch.arange(count, device=device).view(-1, 1, 1) * nodes + node_offset
        edge_parts.append((static_edge.view(1, 2, edges) + offsets).permute(1, 0, 2).reshape(2, -1))
        x_parts.append(x.flatten(0, 1))
        mask = _dense_prepared_action_mask(group, device)
        if mask is not None:
            mask_parts.append(mask.flatten())
        global_mask = static.get("prepared_global_node_mask")
        if global_mask is not None:
            global_parts.append(global_mask.to(device).repeat(count))
        physical = static["prepared_physical_node_mask"]
        if physical is not None:
            physical_parts.append(physical.to(device).repeat(count))
        graph_ids.append(
            torch.arange(graph_offset, graph_offset + count, device=device).repeat_interleave(nodes)
        )
        ptr_parts.append(node_offset + torch.arange(1, count + 1, device=device) * nodes)
        if edge_attr is not None:
            attr_parts.append(edge_attr.flatten(0, 1))
        if static["prepared_edge_role"] is not None:
            role_parts.append(static["prepared_edge_role"].to(device).repeat(count))
        if static["prepared_edge_type"] is not None:
            type_parts.append(static["prepared_edge_type"].to(device).repeat(count))
        if static["has_rigidity"]:
            rigidity_parts.append(group.rigidity.reshape(-1))
        graph_offset += count
        node_offset += count * nodes
    kwargs = {
        "x": torch.cat(x_parts),
        "edge_index": torch.cat(edge_parts, dim=1),
        "batch": torch.cat(graph_ids),
        "ptr": torch.cat(ptr_parts),
    }
    if mask_parts:
        kwargs["action_mask"] = torch.cat(mask_parts)
    if global_parts:
        kwargs["global_node_mask"] = torch.cat(global_parts)
    if physical_parts:
        kwargs["physical_node_mask"] = torch.cat(physical_parts)
    if attr_parts:
        kwargs["edge_attr"] = torch.cat(attr_parts)
    if role_parts:
        kwargs["edge_role"] = torch.cat(role_parts)
    if type_parts:
        kwargs["edge_type"] = torch.cat(type_parts)
    if rigidity_parts:
        kwargs["rigidity"] = torch.cat(rigidity_parts)
    batch = Batch(**kwargs)
    batch._num_graphs = graph_offset
    object.__setattr__(batch, "_physical_node_count_cache", int(physical_node_mask(batch).sum()))
    object.__setattr__(batch, "_policy_action_count_cache", int(policy_action_mask(batch).sum()))
    return batch


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
        self._static = build_graph_static(self.cfg, graph, action)
        self._buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self._capacity, device=self._storage_device),
            batch_size=self._batch_size,
            pin_memory=self._storage_device.type == "cpu" and self._device.type == "cuda",
        )

    def _validate_graph(self, graph: Data) -> None:
        if self._static is None:
            return
        signature = graph_structure_signature(graph)
        static_signature = self._static["signature"]
        if (
            signature["shape"] != static_signature["shape"]
            or signature["edges"] != static_signature["edges"]
        ):
            raise ValueError("Replay observation topology or action ordering changed within a task.")
        mask = getattr(graph, "action_mask", None)
        saved_mask = self._static["action_mask"]
        if (mask is None) != (saved_mask is None):
            raise ValueError("Replay observation action-mask presence changed within a task.")
        if mask is not None:
            mask = mask.detach().bool().reshape(-1)
            if mask.numel() != self._static["raw_node_count"] or not bool(mask.any()):
                raise ValueError("Replay observation has an invalid action mask.")
        if self._graph_signature is not None and not graph_signature_compatible(
            signature,
            self._graph_signature,
            allow_action_subset=True,
        ):
            raise ValueError("Replay observation topology or action ordering differs from teacher.")
        if (getattr(graph, "rigidity", None) is not None) != self._static["has_rigidity"]:
            raise ValueError("Replay observation rigidity field presence changed within a task.")
        role = getattr(graph, "edge_role", None)
        saved_role = self._static["edge_role"]
        if (role is None) != (saved_role is None):
            raise ValueError("Replay observation edge-role field presence changed within a task.")
        if role is not None and not torch.equal(role.detach().cpu().long(), saved_role):
            raise ValueError("Replay observation edge roles changed within a task.")
        direction = getattr(graph, "edge_direction", None)
        saved_direction = self._static.get("edge_direction")
        if (direction is None) != (saved_direction is None):
            raise ValueError("Replay edge-direction field presence changed within a task.")
        if direction is not None and not torch.equal(direction.detach().cpu().float(), saved_direction):
            raise ValueError("Replay edge directions changed within a task.")
        for key in TUBE_PHYSICS_FIELDS:
            value, saved = getattr(graph, key, None), self._static.get("tube_physics_metadata", {}).get(key)
            if (value is None) != (saved is None) or (value is not None and not torch.equal(value.detach().cpu().float(), saved)):
                raise ValueError("Replay tube physics metadata changed within a task.")

    def _require_finite(self, label, values) -> None:
        if bool(getattr(self.cfg, "finite_checks", True)):
            require_finite(label, values)

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
        self._require_finite("replay insertion", values)
        self._buffer.extend(values)
        self._idx = (self._idx + count) % self._capacity
        self._size = min(self._size + count, self._capacity)
        self._num_eps += int(bool(count_episode))
        return self._num_eps

    def add_dense(self, transitions: Mapping[str, torch.Tensor], *, completed_episodes=0,
                  edge_index=None, edge_role=None, edge_direction=None, tube_physics_metadata=None):
        """Insert a batch of same-topology transitions stored as dense tensors.

        ``transitions`` holds ``obs_x``/``next_obs_x`` ``[B, N, F]``,
        ``obs_rigidity``/``next_obs_rigidity`` ``[B, 1]``, ``action`` ``[B, N, A]``,
        ``reward``/``terminated`` ``[B]`` and ``obs_action_mask``/``next_obs_action_mask``
        ``[B, N]``. The topology is fixed per task, so it is validated once from
        ``edge_index`` rather than per graph.
        """
        count = int(transitions["obs_x"].size(0))
        if not count:
            return self._num_eps
        if count > self._capacity:
            transitions = {key: value[-self._capacity:] for key, value in transitions.items()}
            count = self._capacity
        if self._buffer is None:
            if edge_index is None:
                raise ValueError("The first dense replay insertion needs the task edge_index.")
            template = Data(
                x=transitions["obs_x"][0],
                edge_index=torch.as_tensor(edge_index),
                action_mask=transitions["obs_action_mask"][0],
                rigidity=transitions["obs_rigidity"][0],
            )
            if edge_direction is not None:
                template.edge_direction = torch.as_tensor(edge_direction)
            if edge_role is not None:
                template.edge_role = torch.as_tensor(edge_role)
            for key, value in (tube_physics_metadata or {}).items():
                template[key] = torch.as_tensor(value)
            self._initialize(template, transitions["action"][0])
        static = self._static
        provided = tube_physics_metadata or {}
        saved_physics = static.get("tube_physics_metadata", {})
        if set(provided) != set(saved_physics) or any(
                not torch.equal(torch.as_tensor(provided[k]).detach().cpu().float(), v)
                for k, v in saved_physics.items()):
            raise ValueError("Dense replay tube physics metadata changed.")
        saved_direction = static.get("edge_direction")
        if (edge_direction is None) != (saved_direction is None):
            raise ValueError("Dense replay edge-direction metadata changed.")
        if edge_direction is not None and not torch.equal(
            torch.as_tensor(edge_direction).detach().cpu().float(), saved_direction
        ):
            raise ValueError("Dense replay edge directions changed.")
        if list(transitions["obs_x"].shape[1:]) != [static["raw_node_count"], static["raw_feature_dim"]]:
            raise ValueError("Replay observation topology or action ordering changed within a task.")
        if self._graph_signature is not None and (
            static["signature"]["shape"] != self._graph_signature["shape"]
            or static["signature"]["edges"] != self._graph_signature["edges"]
        ):
            raise ValueError("Replay observation topology or action ordering differs from teacher.")
        if list(transitions["action"].shape[1:]) != static["action_shape"]:
            raise ValueError("Replay action shape changed within a task.")
        fields = {
            "obs_x": transitions["obs_x"].detach().float(),
            "next_obs_x": transitions["next_obs_x"].detach().float(),
            "obs_rigidity": transitions["obs_rigidity"].detach().float().reshape(count, 1),
            "next_obs_rigidity": transitions["next_obs_rigidity"].detach().float().reshape(count, 1),
            "action": transitions["action"].detach().float(),
            "reward": transitions["reward"].detach().float().reshape(count, 1),
            "terminated": transitions["terminated"].detach().float().reshape(count, 1),
        }
        if static["action_mask"] is not None:
            masks = torch.cat((
                transitions["obs_action_mask"].detach().bool(),
                transitions["next_obs_action_mask"].detach().bool(),
            ))
            invalid = ~masks.any(dim=1)
            if self._graph_signature is not None:
                reference = torch.as_tensor(
                    self._graph_signature["mask"], dtype=torch.bool, device=masks.device
                )
                invalid = invalid | (masks & ~reference).any(dim=1)
            if bool(invalid.any()):
                raise ValueError("Replay observation has an invalid action mask.")
            fields["obs_action_mask"] = masks[:count]
            fields["next_obs_action_mask"] = masks[count:]
        values = TensorDict(fields, batch_size=[count]).to(self._storage_device)
        self._require_finite("replay insertion", values)
        self._buffer.extend(values)
        self._idx = (self._idx + count) % self._capacity
        self._size = min(self._size + count, self._capacity)
        self._num_eps += int(completed_episodes)
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
        fields = {
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
            }
        if self._static["action_mask"] is not None:
            fields["obs_action_mask"] = torch.stack([
                graph.action_mask.detach().bool() for graph in current
            ])
            fields["next_obs_action_mask"] = torch.stack([
                graph.action_mask.detach().bool() for graph in following
            ])
        return TensorDict(
            fields,
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
            self._require_finite("loaded legacy replay checkpoint", values)
            self._buffer.extend(values)
        self._num_eps, self._size, self._idx = int(state["num_eps"]), size, index
        self._buffer.writer._cursor_value.value = index

    def set_graph_signature(self, signature):
        self._graph_signature = signature
        if self._static is not None and not graph_signature_compatible(
            self._static["signature"],
            signature,
            allow_action_subset=True,
        ):
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
            "format_version": 4,
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
        format_version = int(state.get("format_version", 0))
        if format_version not in {3, 4}:
            raise ValueError("Tensor replay requires format_version=3 or 4 task state.")
        if int(state["capacity"]) != self._capacity:
            raise ValueError("Checkpoint replay capacity does not match current capacity.")
        self._num_eps, self._size, self._idx = int(state["num_eps"]), int(state["size"]), int(state["idx"])
        self._static = deepcopy(state["static"])
        if self._graph_signature is not None and self._static is not None:
            if not graph_signature_compatible(
                self._static["signature"],
                self._graph_signature,
                allow_action_subset=True,
            ):
                raise ValueError("Checkpoint replay topology differs from teacher.")
        if state["replay_buffer"] is not None:
            self._buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(self._capacity, device=self._storage_device),
                batch_size=self._batch_size,
                pin_memory=self._storage_device.type == "cpu" and self._device.type == "cuda",
            )
            saved_fields = state["replay_buffer"]["_storage"]["_storage"]
            mask_fields = {"obs_action_mask", "next_obs_action_mask"}
            present_mask_fields = mask_fields.intersection(saved_fields)
            if present_mask_fields and present_mask_fields != mask_fields:
                raise ValueError("Tensor replay checkpoint has incomplete action-mask storage.")
            has_dynamic_masks = present_mask_fields == mask_fields
            if format_version == 4 and (self._static["action_mask"] is not None) != has_dynamic_masks:
                raise ValueError("Tensor replay v4 action-mask storage does not match its static schema.")
            fields = {
                key: value[:self._size].to(self._storage_device)
                for key, value in saved_fields.items()
                if key != "index"
            }
            if format_version == 3 and self._static["action_mask"] is not None:
                saved_mask = self._static["action_mask"].to(self._storage_device)
                fields["obs_action_mask"] = saved_mask.unsqueeze(0).expand(
                    self._size, -1
                ).clone()
                fields["next_obs_action_mask"] = saved_mask.unsqueeze(0).expand(
                    self._size, -1
                ).clone()
            values = TensorDict(fields, batch_size=[self._size], device=self._storage_device)
            self._require_finite("loaded replay checkpoint", values)
            self._buffer.extend(values)
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
        mask_bytes = 2 * node_count * torch.tensor([], dtype=torch.bool).element_size()
        return self._capacity_per_task * (float_bytes + mask_bytes + index_bytes)

    def _plan_placements(self):
        mode = str(getattr(self.cfg, "replay_storage", "auto")).lower()
        if mode not in {"cpu_pinned", "cuda", "auto"}:
            raise ValueError("replay_storage must be cpu_pinned, cuda, or auto.")
        node_counts = list(getattr(self.cfg, "node_counts", []) or [])
        if len(node_counts) != self._task_count:
            node_counts = [int(getattr(self.cfg, "num_nodes", 1))] * self._task_count
        estimates = {task: self._estimated_task_bytes(int(nodes)) for task, nodes in zip(self.task_names, node_counts)}
        learner_device = torch.device(getattr(self.cfg, "device", "cpu"))
        cuda_available = torch.cuda.is_available() and learner_device.type == "cuda"
        cuda_storage_device = str(learner_device)
        total_memory = free_memory = 0
        if cuda_available:
            free_memory, total_memory = torch.cuda.mem_get_info(learner_device)
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
            placements = {task: cuda_storage_device for task in self.task_names}
            used = required
        elif mode == "auto" and cuda_available:
            for task in sorted(self.task_names, key=lambda name: (estimates[name], name)):
                if used + estimates[task] <= budget:
                    placements[task] = cuda_storage_device
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
        learner_device = torch.device(getattr(self.cfg, "device", "cpu"))
        if torch.cuda.is_available() and learner_device.type == "cuda":
            free, total = torch.cuda.mem_get_info(learner_device)
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

    supports_dense_insertion = True

    def add_dense(self, task, transitions, *, completed_episodes=0, edge_index=None, edge_role=None, edge_direction=None, tube_physics_metadata=None):
        """Insert a dense batch of one task's transitions; see ``_TensorTaskBuffer.add_dense``."""
        if str(task) not in self._buffers:
            raise KeyError(f"Unknown replay task {task!r}; expected one of {self.task_names!r}.")
        self._buffers[str(task)].add_dense(
            transitions,
            completed_episodes=completed_episodes,
            edge_index=edge_index,
            edge_role=edge_role,
            edge_direction=edge_direction,
            tube_physics_metadata=tube_physics_metadata,
        )
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

    @staticmethod
    def _dense_group(sample: _PackedReplaySample, key: str) -> DenseGraphGroup:
        values = sample.values
        prefix = "obs" if key == "obs_x" else "next_obs"
        mask_key = f"{prefix}_action_mask"
        return DenseGraphGroup(
            static=sample.static,
            x=values[key],
            rigidity=values[f"{prefix}_rigidity"],
            action_mask=values[mask_key] if mask_key in values.keys() else None,
        )

    @staticmethod
    def _raw_observations(sample: _PackedReplaySample) -> list[Data]:
        """Reconstruct sampled raw graphs for alternate feature-schema views."""
        values, static = sample.values, sample.static
        raw_x = values["obs_x"]
        device = raw_x.device
        edge_index = static["edge_index"].to(device)
        action_mask = values.get("obs_action_mask", static["action_mask"])
        edge_role = static["edge_role"]
        edge_direction = static.get("edge_direction")
        if action_mask is not None:
            action_mask = action_mask.to(device)
        if edge_role is not None:
            edge_role = edge_role.to(device)

        observations = []
        for index, x in enumerate(raw_x):
            graph = Data(x=x, edge_index=edge_index)
            for key, value in static.get("tube_physics_metadata", {}).items():
                graph[key] = value.to(device)
            if action_mask is not None:
                graph.action_mask = action_mask[index] if action_mask.ndim == 2 else action_mask
            if edge_direction is not None:
                graph.edge_direction = edge_direction.to(device)
            if edge_role is not None:
                graph.edge_role = edge_role
            if static["has_rigidity"]:
                graph.rigidity = values["obs_rigidity"][index]
            observations.append(graph)
        return observations

    def _collate(self, samples, *, performance_profiler=None, subphase_name="combined_collation_transfer"):
        samples = list(samples)
        context = (
            performance_profiler.subphase(subphase_name)
            if performance_profiler is not None else torch.no_grad()
        )
        with context:
            target = torch.device(getattr(self.cfg, "device", "cpu"))
            moved = []
            for sample in samples:
                values = sample.values
                if values.device != target:
                    values = values.to(target, non_blocking=values.device.type == "cpu" and target.type == "cuda")
                    sample = _PackedReplaySample(sample.task, values, sample.static)
                moved.append(sample)
            obs = dense_graph_batch(
                self.cfg, [self._dense_group(sample, "obs_x") for sample in moved], target
            )
            next_obs = dense_graph_batch(
                self.cfg, [self._dense_group(sample, "next_obs_x") for sample in moved], target
            )
            return (
                obs,
                torch.cat([sample.values["action"].flatten(0, 1) for sample in moved]),
                torch.cat([sample.values["reward"] for sample in moved]),
                torch.cat([sample.values["terminated"] for sample in moved]),
                next_obs,
            )

    def sample(self, performance_profiler=None):
        return self._collate(self._sample_raw_by_task(performance_profiler).values(), performance_profiler=performance_profiler)

    def sample_task_batches(self, performance_profiler=None):
        raw = self._sample_raw_by_task(performance_profiler)
        return {
            task: self._collate([sample], performance_profiler=performance_profiler, subphase_name="task_collation_transfer")
            for task, sample in raw.items()
        }

    def sample_with_tasks(self, performance_profiler=None, *, combine=True):
        raw = self._sample_raw_by_task(performance_profiler)
        combined = (
            self._collate(raw.values(), performance_profiler=performance_profiler)
            if combine
            else None
        )
        by_task = {
            task: self._collate([sample], performance_profiler=performance_profiler, subphase_name="task_collation_transfer")
            for task, sample in raw.items()
        }
        return ReplayBatch(
            combined=combined,
            by_task=by_task,
            raw_observations_by_task={
                task: self._raw_observations(sample)
                for task, sample in raw.items()
            },
        )

    def sample_ensemble(self):
        """Benchmark ReplayBufferEnsemble without changing the training path."""
        return self._collate(self._sample_raw_by_task_ensemble().values())

    def state_dict(self):
        return {
            "format_version": 4,
            "task_names": list(self.task_names),
            "batch_size": self._batch_size,
            "batch_size_per_task": self._batch_size_per_task,
            "capacity_per_task": self._capacity_per_task,
            "placement_metadata": self.runtime_storage_metadata(),
            "buffers": {task: buffer.state_dict() for task, buffer in self._buffers.items()},
        }

    def load_state_dict(self, state):
        if int(state.get("format_version", 0)) not in {3, 4}:
            raise ValueError("Tensor replay requires a format_version=3 or 4 checkpoint.")
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
    backend = str(getattr(cfg, "replay_backend", "torchrl_tensor")).lower()
    if backend == "legacy":
        from common.gnn_buffer import GNNBuffer
        return GNNBuffer(cfg)
    if backend == "torchrl_tensor":
        return TensorGNNBuffer(cfg)
    raise ValueError("replay_backend must be legacy or torchrl_tensor.")
