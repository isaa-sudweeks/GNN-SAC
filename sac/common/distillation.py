"""Frozen topology teachers, target tensor shards, and forward Gaussian KL."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from common.gnn_actor_critic import GNNActorCritic
from common.graph_transforms import (
    graph_feature_flags, graph_feature_schema,
    graph_structure_signature as graph_signature,
    policy_action_mask, prepare_graph,
)


def _get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def enabled(cfg) -> bool:
    return bool(_get(_get(cfg, "distillation", {}), "enabled", False))


def gaussian_forward_kl(teacher_mean: torch.Tensor, teacher_log_std: torch.Tensor,
                        student_mean: torch.Tensor, student_log_std: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(teacher || student), without changing either variance."""
    values = (teacher_mean, teacher_log_std, student_mean, student_log_std)
    if any(value.shape != student_mean.shape for value in values):
        raise ValueError("Teacher and student distribution shapes differ.")
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite policy distribution parameters in distillation.")
    # Float64 avoids overflow for finite, narrow SAC Gaussians near log_std=-10.
    tm, tl, sm, sl = (value.double() for value in values)
    result = sl - tl + 0.5 * ((2 * (tl - sl)).exp() + (tm - sm).square() * (-2 * sl).exp() - 1)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite Gaussian KL in distillation.")
    return result


def graph_mean_kl(kl: torch.Tensor, obs: Data) -> torch.Tensor:
    """Sum coordinates, average active nodes per graph, then average graphs."""
    mask = policy_action_mask(obs)
    cached_action_count = getattr(obs, "_policy_action_count_cache", None)
    action_count = (
        int(cached_action_count)
        if cached_action_count is not None
        else int(mask.sum())
    )
    if kl.ndim != 2 or kl.size(0) != action_count:
        raise ValueError("KL rows do not match the graph action mask.")
    batch = getattr(obs, "batch", None)
    graph_count = int(obs.num_graphs) if batch is not None else 1
    ids = batch[mask] if batch is not None else mask.new_zeros(int(mask.sum()), dtype=torch.long)
    # Replay/cache construction already validates one or more active actions per
    # graph. Preserve the standalone helper's defensive check without forcing a
    # CUDA synchronization in every online/offline distillation loss.
    if cached_action_count is None and torch.bincount(ids, minlength=graph_count).eq(0).any():
        raise ValueError("Distillation requires at least one active node per graph.")
    return global_mean_pool(kl.sum(-1, keepdim=True), ids, size=graph_count).mean()


def replay_observations(replay: dict):
    """Yield every valid ring-buffer observation, oldest first, without copying."""
    capacity, size, index = (int(replay[key]) for key in ("capacity", "size", "idx"))
    observations = replay["obs"]
    if not (capacity > 0 and 0 < size <= capacity and 0 <= index < capacity):
        raise ValueError("Empty or invalid teacher replay layout.")
    if len(observations) != capacity or (size < capacity and index != size):
        raise ValueError("Inconsistent teacher replay capacity/index.")
    start = index if size == capacity else 0
    for offset in range(size):
        graph = observations[(start + offset) % capacity]
        if not isinstance(graph, Data):
            raise ValueError("Teacher replay contains an invalid observation.")
        yield graph


TARGET_CACHE_FORMAT = "gnn-sac-distillation-targets-v2"


class ObservationShards:
    """Versioned prepared-observation/teacher-target tensor shards.

    Select a shard proportional to its size, then draw a minibatch uniformly
    inside it. At most the current and one prefetched shard are resident. Sample
    specifications are separate from loading so lookahead remains checkpointable.
    """

    def __init__(self, directory: Path, expected_contract: dict,
                 *, pin_memory: bool = False, prefetch: bool = True):
        self.directory = directory
        self.metadata = json.loads((directory / "manifest.json").read_text())
        if self.metadata.get("format") != TARGET_CACHE_FORMAT:
            raise ValueError("Cached distillation data is not a compatible teacher-target cache.")
        if self.metadata.get("contract") != expected_contract:
            raise ValueError("Cached teacher targets have a changed schema or topology signature.")
        self.sizes = torch.tensor(self.metadata["sizes"], dtype=torch.long)
        if not len(self.sizes) or (self.sizes <= 0).any():
            raise ValueError("Invalid teacher-target shard manifest.")
        self.static = torch.load(directory / "static.pt", map_location="cpu", weights_only=True)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.prefetch_enabled = bool(prefetch)
        self._index = None
        self._shard = None
        self._future = None
        self._future_index = None
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetch_enabled else None

    def draw(self, count: int, generator: torch.Generator) -> dict:
        draw = torch.randint(int(self.sizes.sum()), (1,), generator=generator)
        index = int(torch.searchsorted(self.sizes.cumsum(0), draw, right=True))
        indices = torch.randint(int(self.sizes[index]), (count,), generator=generator).tolist()
        return {"shard": index, "indices": indices}

    def _load(self, index: int) -> dict:
        return torch.load(self.directory / f"{index}.pt", map_location="cpu", weights_only=True)

    def prefetch(self, specification: dict) -> None:
        index = int(specification["shard"])
        if not self.prefetch_enabled or index == self._index:
            return
        if self._future is not None and self._future_index == index:
            return
        if self._future is not None:
            self._future.result()
        self._future_index = index
        self._future = self._executor.submit(self._load, index)

    def _get_shard(self, index: int) -> dict:
        if index == self._index:
            return self._shard
        if self._future is not None and self._future_index == index:
            shard = self._future.result()
            self._future = self._future_index = None
        else:
            shard = self._load(index)
        self._index, self._shard = index, shard
        return shard

    def resolve(self, specification: dict, device: torch.device) -> tuple[Batch, torch.Tensor, torch.Tensor]:
        index = int(specification["shard"])
        indices = torch.tensor(specification["indices"], dtype=torch.long)
        shard = self._get_shard(index)
        count = int(indices.numel())
        x = shard["x"].index_select(0, indices).contiguous()
        teacher_mean = shard["teacher_mean"].index_select(0, indices).flatten(0, 1).contiguous()
        teacher_log_std = shard["teacher_log_std"].index_select(0, indices).flatten(0, 1).contiguous()
        nodes = int(x.size(1))
        edges = int(self.static["edge_index"].size(1))
        offsets = torch.arange(count).view(-1, 1, 1) * nodes
        edge_index = (self.static["edge_index"].view(1, 2, edges) + offsets).permute(1, 0, 2).reshape(2, -1)
        kwargs = dict(
            x=x.flatten(0, 1), edge_index=edge_index,
            action_mask=self.static["action_mask"].repeat(count),
            batch=torch.arange(count).repeat_interleave(nodes),
            ptr=torch.arange(0, (count + 1) * nodes, nodes),
        )
        if "physical_node_mask" in self.static:
            kwargs["physical_node_mask"] = self.static["physical_node_mask"].repeat(count)
        if "edge_attr" in shard:
            kwargs["edge_attr"] = shard["edge_attr"].index_select(0, indices).flatten(0, 1).contiguous()
        batch = Batch(**kwargs)
        batch._num_graphs = count
        object.__setattr__(batch, "_policy_action_count_cache", int(teacher_mean.size(0)))
        if self.pin_memory:
            batch = batch.pin_memory()
            teacher_mean = teacher_mean.pin_memory()
            teacher_log_std = teacher_log_std.pin_memory()
        non_blocking = self.pin_memory and device.type == "cuda"
        return (batch.to(device, non_blocking=non_blocking),
                teacher_mean.to(device, non_blocking=non_blocking),
                teacher_log_std.to(device, non_blocking=non_blocking))

    def sample(self, count: int, generator: torch.Generator, device="cpu"):
        """Synchronous convenience wrapper, primarily for diagnostics/tests."""
        return self.resolve(self.draw(count, generator), torch.device(device))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def prepare(cls, directory: Path, observations, shard_size: int, contract: dict,
                teacher, prepare_graph_fn, device: torch.device, target_batch_size: int,
                *, pin_memory: bool = False, prefetch: bool = True):
        if (directory / "manifest.json").exists():
            return cls(directory, contract, pin_memory=pin_memory, prefetch=prefetch)
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete cache. Separate temporary directories also make
        # simultaneous jobs extracting the same teacher safe.
        with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
            root = Path(temporary)
            sizes, shard, static = [], [], None

            def write_shard(graphs):
                nonlocal static
                prepared = [prepare_graph_fn(graph) for graph in graphs]
                if any(graph_signature(graph) != contract["prepared_signature"] for graph in prepared):
                    raise ValueError("Prepared teacher replay changes topology or action ordering.")
                first = prepared[0]
                current_static = {
                    "edge_index": first.edge_index.cpu().contiguous(),
                    "action_mask": policy_action_mask(first).cpu().contiguous(),
                }
                physical = getattr(first, "physical_node_mask", None)
                if physical is not None:
                    current_static["physical_node_mask"] = physical.cpu().contiguous()
                if static is None:
                    static = current_static
                    torch.save(static, root / "static.pt")
                elif (set(static) != set(current_static)
                      or any(not torch.equal(value, current_static[key]) for key, value in static.items())):
                    raise ValueError("Prepared teacher replay has changing structural tensors.")
                means, log_stds = [], []
                for start in range(0, len(prepared), target_batch_size):
                    batch = Batch.from_data_list(prepared[start:start + target_batch_size]).to(device)
                    with torch.no_grad():
                        mean, log_std = teacher.policy_distribution(batch)
                    if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
                        raise ValueError("Nonfinite teacher distribution parameters while building cache.")
                    active = int(current_static["action_mask"].sum())
                    means.append(mean.reshape(-1, active, mean.size(-1)).cpu())
                    log_stds.append(log_std.reshape(-1, active, log_std.size(-1)).cpu())
                payload = {
                    "x": torch.stack([graph.x.cpu() for graph in prepared]).contiguous(),
                    "teacher_mean": torch.cat(means).contiguous(),
                    "teacher_log_std": torch.cat(log_stds).contiguous(),
                }
                edge_attrs = [getattr(graph, "edge_attr", None) for graph in prepared]
                if any(value is not None for value in edge_attrs):
                    if any(value is None for value in edge_attrs):
                        raise ValueError("Prepared replay inconsistently provides edge features.")
                    payload["edge_attr"] = torch.stack([value.cpu() for value in edge_attrs]).contiguous()
                torch.save(payload, root / f"{len(sizes)}.pt")
                sizes.append(len(prepared))

            for graph in observations:
                if graph_signature(graph) != contract["replay_signature"]:
                    raise ValueError("Teacher replay changes topology or action ordering.")
                shard.append(graph.clone().cpu())
                if len(shard) == shard_size:
                    write_shard(shard)
                    shard = []
            if shard:
                write_shard(shard)
            (root / "manifest.json").write_text(json.dumps({
                "format": TARGET_CACHE_FORMAT, "sizes": sizes, "contract": contract,
            }, sort_keys=True))
            try:
                root.rename(directory)
            except OSError:
                if not (directory / "manifest.json").exists():
                    raise
        return cls(directory, contract, pin_memory=pin_memory, prefetch=prefetch)


class Distillation:
    """Training-only state; teachers are deliberately not student submodules."""

    def __init__(self, cfg, task_names: list[str], device=None, *, defer_target_cache=False):
        self.cfg = cfg
        self.device = torch.device(device if device is not None else _get(cfg, "device", "cuda"))
        options = _get(cfg, "distillation", {})
        if _get(cfg, "sac_backend", "gnn") != "gnn":
            raise ValueError("Distillation currently requires a GNN student.")
        self.pretrain_updates = int(_get(options, "pretrain_updates", 0))
        self.batch_size = int(_get(options, "batch_size", 256))
        self.initial_weight = float(_get(options, "initial_weight", 1.0))
        duration = _get(options, "decay_steps")
        self.decay_steps = float(duration if duration is not None else float(_get(options, "decay_fraction", 0.5)) * cfg.steps)
        self.checkpoint_freq = int(_get(options, "checkpoint_freq", 1000))
        self.log_freq = int(_get(options, "log_freq", 100))
        shard_size = int(_get(options, "shard_size", 4096))
        target_batch_size = int(_get(options, "target_batch_size", self.batch_size))
        self.pin_memory = bool(_get(options, "pin_memory", True))
        self.prefetch = bool(_get(options, "prefetch", True))
        if (self.pretrain_updates < 0 or self.batch_size <= 0 or shard_size <= 0
                or target_batch_size <= 0
                or self.checkpoint_freq < 0 or self.log_freq <= 0
                or not math.isfinite(self.initial_weight) or self.initial_weight < 0
                or not math.isfinite(self.decay_steps) or self.decay_steps <= 0):
            raise ValueError("Invalid distillation counts, weight, or decay duration.")
        # Only settings that affect optimization/sampling belong in checkpoint
        # compatibility. Cache-build batch size, pinning, and prefetch are
        # performance choices and may safely change across a resume.
        self.settings = dict(pretrain_updates=self.pretrain_updates, batch_size=self.batch_size,
                             initial_weight=self.initial_weight, decay_steps=self.decay_steps,
                             shard_size=shard_size)
        self.completed_updates = 0
        self.stage = "offline" if self.pretrain_updates else "online"
        self.step = 0
        self.generator = torch.Generator().manual_seed(int(_get(cfg, "seed", 0)))
        self.teachers, self.datasets, self.sources = {}, {}, {}
        self._dataset_plans = {}
        self.replay_signatures = {}
        self.pending_samples = {}
        self.metrics = {}
        mapping = _get(options, "teachers", {})
        selected = {}
        for task in task_names:
            topology = task.split(":", 1)[1] if ":" in task else str(cfg.truss_topology)
            if topology not in mapping:
                raise ValueError(f"Missing distillation teacher for training topology {topology!r}.")
            selected[task] = (topology, Path(mapping[topology]).expanduser().resolve())
        cache = Path(_get(options, "cache_dir") or Path(cfg.work_dir) / "distillation_cache").expanduser()
        # Teacher initialization must not consume the student's sampling RNG.
        with torch.random.fork_rng(devices=[]):
            for task, (topology, path) in selected.items():
                print(f"Loading distillation teacher for {topology}: {path}", flush=True)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    stream.seek(0)
                    state = torch.load(stream, map_location="cpu", weights_only=False)
                self.sources[task] = {"path": str(path), "sha256": digest}
                if not all(key in state for key in ("config", "agent", "buffer")):
                    raise ValueError("Distillation teachers require full trainer checkpoints with config and replay.")
                teacher_cfg = SimpleNamespace(**state["config"])
                self._validate_config(teacher_cfg, topology, state["agent"])
                teacher = GNNActorCritic(teacher_cfg)
                teacher.load_state_dict(state["agent"]["model"])
                teacher.requires_grad_(False).eval()
                # Keep the module's train/eval contract while releasing critics.
                teacher._Qs = torch.nn.ModuleList()
                teacher._target_Qs = torch.nn.ModuleList()
                teacher.to(self.device)
                self.teachers[task] = teacher
                replay = state["buffer"]
                if "buffers" in replay:
                    matching = [key for key in replay["buffers"]
                                if key == task or key.split(":")[-1] == topology]
                    if not matching and len(replay["buffers"]) == 1:
                        matching = list(replay["buffers"])
                    if len(matching) != 1:
                        raise ValueError(f"Cannot identify replay for {topology!r}.")
                    replay = replay["buffers"][matching[0]]
                first = next(replay_observations(replay))
                signature = graph_signature(first)
                if first.x.size(1) != int(cfg.obs_dim):
                    raise ValueError("Teacher replay observation width differs from student.")
                self.replay_signatures[task] = signature
                if self.pretrain_updates:
                    contract = {
                        "format": TARGET_CACHE_FORMAT,
                        "source_sha256": digest,
                        "replay_signature": signature,
                        "prepared_signature": graph_signature(self.prepare(first)),
                        "graph_feature_schema": graph_feature_schema(self.cfg),
                        "use_virtual_node": bool(_get(self.cfg, "use_virtual_node", False)),
                    }
                    contract_hash = hashlib.sha256(
                        json.dumps(contract, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    directory = cache / f"{digest}-targets-v2-{contract_hash}-s{shard_size}"
                    replay_key = None
                    if "buffers" in state["buffer"]:
                        replay_key = matching[0]
                    self._dataset_plans[task] = {
                        "directory": directory,
                        "contract": contract,
                        "source_path": path,
                        "replay_key": replay_key,
                        "shard_size": shard_size,
                        "target_batch_size": target_batch_size,
                    }
                    if not defer_target_cache:
                        self.datasets[task] = self._prepare_dataset(
                            task, replay_observations(replay)
                        )
                del first, replay, state

    def _prepare_dataset(self, task: str, observations) -> ObservationShards:
        plan = self._dataset_plans[task]
        return ObservationShards.prepare(
            plan["directory"], observations, plan["shard_size"], plan["contract"],
            self.teachers[task], self.prepare, self.device, plan["target_batch_size"],
            pin_memory=self.pin_memory and self.device.type == "cuda",
            prefetch=self.prefetch,
        )

    def prepare_offline_datasets(self) -> None:
        """Build or open target caches only after resume establishes an offline stage."""
        if self.stage != "offline":
            return
        for task, plan in self._dataset_plans.items():
            if task in self.datasets:
                continue
            if (plan["directory"] / "manifest.json").exists():
                self.datasets[task] = self._prepare_dataset(task, iter(()))
                continue
            with plan["source_path"].open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != self.sources[task]["sha256"]:
                    raise ValueError(f"Distillation teacher source changed while preparing {task!r}.")
                stream.seek(0)
                state = torch.load(stream, map_location="cpu", weights_only=False)
            replay = state["buffer"]
            if plan["replay_key"] is not None:
                replay = replay["buffers"][plan["replay_key"]]
            self.datasets[task] = self._prepare_dataset(
                task, replay_observations(replay)
            )
            del replay, state
        for task, specification in self.pending_samples.items():
            self.datasets[task].prefetch(specification)

    def _validate_config(self, teacher, topology: str, agent_state: dict) -> None:
        if _get(teacher, "sac_backend", "gnn") != "gnn":
            raise ValueError("Distillation currently requires GNN teachers.")
        teacher_topologies = _get(teacher, "truss_topologies") or [_get(teacher, "truss_topology")]
        if isinstance(teacher_topologies, str):
            teacher_topologies = [teacher_topologies]
        if list(teacher_topologies) != [topology]:
            raise ValueError(f"Teacher topology does not match {topology!r}.")
        if graph_feature_schema(teacher) != graph_feature_schema(self.cfg):
            raise ValueError("Teacher and student graph feature schemas differ.")
        saved = agent_state.get("graph_feature_schema")
        if saved is not None and saved != graph_feature_schema(teacher):
            raise ValueError("Teacher checkpoint graph schema differs from its config.")
        if saved is None and any(graph_feature_flags(teacher).values()):
            raise ValueError("Teacher checkpoint is missing its graph feature schema.")
        defaults = dict(obs_dim=None, action_dim=None, use_virtual_node=False,
                        use_control_graph=False, truss_realistic=False, truss_graph_view="auto",
                        obs_norm=True,
                        normalize_observations=True, speed=0.05, action_low=-1.0, action_high=1.0)
        for key, default in defaults.items():
            if _get(teacher, key, default) != _get(self.cfg, key, default):
                raise ValueError(f"Teacher and student observation/control convention differs: {key}.")
        if _get(self.cfg, "truss_realistic", False):
            if (_get(teacher, "control_node_observation_source", "physical_node")
                    != _get(self.cfg, "control_node_observation_source", "physical_node")):
                raise ValueError("Teacher and student control_node_observation_source differs.")
        if str(_get(teacher, "task", "truss-graph")).split(":")[0] != str(self.cfg.task).split(":")[0]:
            raise ValueError("Teacher and student task conventions differ.")

    def prepare(self, graph: Data) -> Data:
        return prepare_graph(graph, use_virtual_node=bool(_get(self.cfg, "use_virtual_node", False)),
                             **graph_feature_flags(self.cfg))

    def bind_replay(self, replay) -> None:
        """Install teacher topology contracts at the replay's cold write/load boundary."""
        setter = getattr(replay, "set_task_graph_signatures", None)
        if setter is None:
            raise TypeError("Distillation requires replay topology validation support.")
        setter(self.replay_signatures)

    @property
    def weight(self) -> float:
        return self.initial_weight * max(0.0, 1.0 - self.step / self.decay_steps)

    def loss(self, student, task: str, obs: Data) -> torch.Tensor:
        if task not in self.teachers:
            raise ValueError(f"No teacher for replay task {task!r}.")
        teacher = self.teachers[task]
        with torch.no_grad():
            tm, tl = teacher.policy_distribution(obs)
        sm, sl = student.policy_distribution(obs)
        return graph_mean_kl(gaussian_forward_kl(tm, tl, sm, sl), obs)

    def target_loss(self, student, obs: Data, teacher_mean: torch.Tensor,
                    teacher_log_std: torch.Tensor) -> torch.Tensor:
        """Offline student loss against precomputed, pre-tanh teacher targets."""
        sm, sl = student.policy_distribution(obs)
        return graph_mean_kl(
            gaussian_forward_kl(teacher_mean, teacher_log_std, sm, sl), obs
        )

    def offline_update(self, agent) -> dict:
        self.prepare_offline_datasets()
        agent.model.eval() # Deterministic policy parameters, while retaining gradients.
        agent.pi_optim.zero_grad(set_to_none=True)
        metrics = {}
        current_samples = {}
        for task, dataset in self.datasets.items():
            specification = self.pending_samples.pop(task, None)
            if specification is None:
                specification = dataset.draw(self.batch_size, self.generator)
            current_samples[task] = specification
        for task, dataset in self.datasets.items():
            specification = current_samples[task]
            batch, teacher_mean, teacher_log_std = dataset.resolve(specification, agent.device)
            if self.completed_updates + 1 < self.pretrain_updates:
                next_specification = dataset.draw(self.batch_size, self.generator)
                self.pending_samples[task] = next_specification
                dataset.prefetch(next_specification)
            loss = self.target_loss(agent.model, batch, teacher_mean, teacher_log_std)
            (loss / len(self.datasets)).backward()
            metrics[f"kl/{task}"] = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(agent.model.actor_parameters(), agent.cfg.grad_clip_norm, error_if_nonfinite=True)
        agent.pi_optim.step()
        self.completed_updates += 1
        return {"kl": sum(metrics.values()) / len(metrics), **metrics,
                "offline_updates": self.completed_updates, "stage": "offline"}

    def finish_pretraining(self, agent) -> None:
        if self.stage != "offline":
            return
        if self.completed_updates != self.pretrain_updates:
            raise ValueError("Cannot transition before offline distillation completes.")
        # Clear Adam moments without changing any optimizer settings or weights.
        agent.pi_optim.state.clear()
        agent.pi_optim.zero_grad(set_to_none=True)
        self.stage = "online"
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()
        self.pending_samples.clear()

    def state_dict(self) -> dict:
        return dict(stage=self.stage, completed_updates=self.completed_updates,
                    sources=self.sources, settings=self.settings, generator=self.generator.get_state(),
                    pending_samples=_copy_sample_specs(self.pending_samples))

    def load_state_dict(self, state: dict) -> None:
        saved_settings = state.get("settings", {})
        saved_semantic_settings = {
            key: saved_settings.get(key) for key in self.settings
        }
        if state["sources"] != self.sources or saved_semantic_settings != self.settings:
            raise ValueError("Distillation teacher sources or schedule/settings changed during resume.")
        stage, completed = state["stage"], int(state["completed_updates"])
        if stage not in {"offline", "online"} or not 0 <= completed <= self.pretrain_updates:
            raise ValueError("Invalid distillation checkpoint stage/progress.")
        if stage == "online" and completed != self.pretrain_updates:
            raise ValueError("Online checkpoint has incomplete offline distillation.")
        self.stage, self.completed_updates = stage, completed
        self.generator.set_state(state["generator"])
        self.pending_samples = _copy_sample_specs(state.get("pending_samples", {}))
        if stage == "online":
            for dataset in self.datasets.values():
                dataset.close()
            self.datasets.clear()
            self.pending_samples.clear()
        elif self.datasets:
            for task, specification in self.pending_samples.items():
                self.datasets[task].prefetch(specification)


def _copy_sample_specs(specifications: dict) -> dict:
    """Copy JSON-like lookahead state without sharing mutable checkpoint values."""
    return {
        task: {"shard": int(value["shard"]), "indices": list(value["indices"])}
        for task, value in specifications.items()
    }
