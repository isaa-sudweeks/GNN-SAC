"""Frozen topology teachers, observation shards, and forward Gaussian KL."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from common.gnn_actor_critic import GNNActorCritic
from common.graph_transforms import (
    graph_feature_flags, graph_feature_schema, policy_action_mask, prepare_graph,
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
    if kl.ndim != 2 or kl.size(0) != int(mask.sum()):
        raise ValueError("KL rows do not match the graph action mask.")
    batch = getattr(obs, "batch", None)
    graph_count = int(obs.num_graphs) if batch is not None else 1
    ids = batch[mask] if batch is not None else mask.new_zeros(int(mask.sum()), dtype=torch.long)
    if torch.bincount(ids, minlength=graph_count).eq(0).any():
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


def graph_signature(graph: Data) -> dict:
    """Structural action-order contract; excludes changing node/edge features."""
    mask = policy_action_mask(graph)
    if mask.ndim != 1 or mask.numel() != graph.x.size(0) or not mask.any():
        raise ValueError("Invalid teacher action mask.")
    return {
        "shape": list(graph.x.shape),
        "edges": graph.edge_index.detach().cpu().tolist(),
        "mask": mask.detach().cpu().tolist(),
    }


class ObservationShards:
    """Uniform marginal observation sampling with one resident CPU shard.

    Select a shard proportional to its size, then draw a minibatch uniformly
    inside it. This gives an unbiased loss over the entire replay without
    deserializing hundreds of shards for each minibatch.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.metadata = json.loads((directory / "manifest.json").read_text())
        self.sizes = torch.tensor(self.metadata["sizes"], dtype=torch.long)
        if not len(self.sizes) or (self.sizes <= 0).any():
            raise ValueError("Invalid observation shard manifest.")
        self._index = None
        self._graphs = None

    def sample(self, count: int, generator: torch.Generator) -> list[Data]:
        draw = torch.randint(int(self.sizes.sum()), (1,), generator=generator)
        index = int(torch.searchsorted(self.sizes.cumsum(0), draw, right=True))
        if index != self._index:
            self._graphs = torch.load(self.directory / f"{index}.pt", map_location="cpu", weights_only=False)
            self._index = index
        indices = torch.randint(len(self._graphs), (count,), generator=generator).tolist()
        return [self._graphs[i] for i in indices]

    @classmethod
    def prepare(cls, directory: Path, observations, shard_size: int, signature: dict):
        if (directory / "manifest.json").exists():
            return cls(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete cache. Separate temporary directories also make
        # simultaneous jobs extracting the same teacher safe.
        with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
            root = Path(temporary)
            sizes, shard = [], []
            for graph in observations:
                if graph_signature(graph) != signature:
                    raise ValueError("Teacher replay changes topology or action ordering.")
                shard.append(graph.clone().cpu())
                if len(shard) == shard_size:
                    torch.save(shard, root / f"{len(sizes)}.pt")
                    sizes.append(len(shard))
                    shard = []
            if shard:
                torch.save(shard, root / f"{len(sizes)}.pt")
                sizes.append(len(shard))
            (root / "manifest.json").write_text(json.dumps({"sizes": sizes, "signature": signature}))
            try:
                root.rename(directory)
            except OSError:
                if not (directory / "manifest.json").exists():
                    raise
        return cls(directory)


class Distillation:
    """Training-only state; teachers are deliberately not student submodules."""

    def __init__(self, cfg, task_names: list[str]):
        self.cfg = cfg
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
        if (self.pretrain_updates < 0 or self.batch_size <= 0 or shard_size <= 0
                or self.checkpoint_freq < 0 or self.log_freq <= 0
                or not math.isfinite(self.initial_weight) or self.initial_weight < 0
                or not math.isfinite(self.decay_steps) or self.decay_steps <= 0):
            raise ValueError("Invalid distillation counts, weight, or decay duration.")
        self.settings = dict(pretrain_updates=self.pretrain_updates, batch_size=self.batch_size,
                             initial_weight=self.initial_weight, decay_steps=self.decay_steps,
                             shard_size=shard_size)
        self.completed_updates = 0
        self.stage = "offline" if self.pretrain_updates else "online"
        self.step = 0
        self.generator = torch.Generator().manual_seed(int(_get(cfg, "seed", 0)))
        self.teachers, self.datasets, self.sources, self.signatures = {}, {}, {}, {}
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
                self.signatures[task] = graph_signature(self.prepare(first))
                if self.pretrain_updates:
                    self.datasets[task] = ObservationShards.prepare(
                        cache / f"{digest}-{shard_size}", replay_observations(replay), shard_size, signature)
                del first, replay, state

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

    @property
    def weight(self) -> float:
        return self.initial_weight * max(0.0, 1.0 - self.step / self.decay_steps)

    def loss(self, student, task: str, obs: Data) -> torch.Tensor:
        if task not in self.teachers:
            raise ValueError(f"No teacher for replay task {task!r}.")
        graphs = obs.to_data_list() if isinstance(obs, Batch) else [obs]
        if any(graph_signature(graph) != self.signatures[task] for graph in graphs):
            raise ValueError(f"Observation topology/action ordering differs from teacher for {task}.")
        teacher = self.teachers[task].to(obs.x.device)
        try:
            with torch.no_grad():
                tm, tl = teacher.policy_distribution(obs)
        finally:
            teacher.cpu()
        sm, sl = student.policy_distribution(obs)
        return graph_mean_kl(gaussian_forward_kl(tm, tl, sm, sl), obs)

    def offline_update(self, agent) -> dict:
        agent.model.eval() # Deterministic policy parameters, while retaining gradients.
        agent.pi_optim.zero_grad(set_to_none=True)
        metrics = {}
        for task, dataset in self.datasets.items():
            graphs = dataset.sample(self.batch_size, self.generator)
            batch = Batch.from_data_list([self.prepare(graph) for graph in graphs]).to(agent.device)
            loss = self.loss(agent.model, task, batch)
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
        self.datasets.clear()

    def state_dict(self) -> dict:
        return dict(stage=self.stage, completed_updates=self.completed_updates,
                    sources=self.sources, settings=self.settings, generator=self.generator.get_state())

    def load_state_dict(self, state: dict) -> None:
        if state["sources"] != self.sources or state["settings"] != self.settings:
            raise ValueError("Distillation teacher sources or schedule/settings changed during resume.")
        stage, completed = state["stage"], int(state["completed_updates"])
        if stage not in {"offline", "online"} or not 0 <= completed <= self.pretrain_updates:
            raise ValueError("Invalid distillation checkpoint stage/progress.")
        if stage == "online" and completed != self.pretrain_updates:
            raise ValueError("Online checkpoint has incomplete offline distillation.")
        self.stage, self.completed_updates = stage, completed
        self.generator.set_state(state["generator"])
        if stage == "online":
            self.datasets.clear()
