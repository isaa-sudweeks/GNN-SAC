"""Fail early on unsupported policy/simulator contracts, without collecting data."""

from copy import deepcopy
from typing import Any

import numpy as np


def _get(cfg: Any, name: str, default=None):
    return cfg.get(name, default) if hasattr(cfg, "get") else getattr(cfg, name, default)


def _list(value) -> list[str]:
    if value is None or isinstance(value, str) and value == "null":
        return []
    return [value] if isinstance(value, str) else list(value)


def configured_topologies(cfg: Any, *, include_evaluation: bool = False) -> list[str]:
    """Return ordered, unique topology identifiers from a resolved run config."""
    topologies = _list(_get(cfg, "truss_topologies")) or _list(_get(cfg, "topologies"))
    if not topologies:
        topologies = [str(task).split(":", 1)[1] for task in _get(cfg, "tasks", []) if ":" in str(task)]
    if not topologies:
        topologies = _list(_get(cfg, "truss_topology"))
    if include_evaluation:
        topologies += _list(_get(cfg, "eval_extra_topologies"))
    return list(dict.fromkeys(topologies))


def validate_training_config(cfg: Any) -> None:
    """Reject incompatible resolved choices before constructing models or teachers."""
    backend = str(_get(cfg, "sac_backend", "mlp")).lower()
    tasks = _list(_get(cfg, "tasks")) or [_get(cfg, "task", "")]
    tasks = list(dict.fromkeys([str(_get(cfg, "task", "")), *map(str, tasks)]))
    graph = bool(_get(cfg, "use_graph_observations", False)) or any("graph" in task for task in tasks)
    topologies = configured_topologies(cfg)
    if backend == "mlp":
        if graph:
            raise ValueError("sac_backend=mlp requires flat observations; use padded_mlp for graph observations.")
        if len(topologies) > 1 or bool(_get(cfg, "multitask", False)) and len(set(tasks)) > 1:
            raise ValueError("Heterogeneous flat MLP training is unsupported; use sac_backend=padded_mlp.")
    elif backend in {"gnn", "padded_mlp"}:
        if not graph:
            raise ValueError(f"sac_backend={backend} requires graph observations and a graph task.")
    else:
        raise ValueError(f"Unsupported sac_backend={backend!r}.")

    if backend == "padded_mlp":
        from common.graph_transforms import graph_feature_flags

        if not bool(_get(cfg, "use_control_graph", False)):
            raise ValueError("padded_mlp requires use_control_graph=true.")
        if bool(_get(cfg, "use_virtual_node", False)) or any(graph_feature_flags(cfg).values()):
            raise ValueError("padded_mlp requires use_virtual_node=false and graph features disabled.")
        if int(_get(cfg, "padded_mlp_max_nodes", 0)) <= 0:
            raise ValueError("padded_mlp_max_nodes must be positive.")
        mode = _get(cfg, "mlp_training_mode")
        if mode == "specialist" and len(topologies) != 1:
            raise ValueError("The specialist preset requires exactly one morphology.")
        if mode == "generalist" and len(topologies) < 2:
            raise ValueError("The generalist preset requires at least two training morphologies.")

    simulator = str(_get(cfg, "mujoco_backend", "mujoco")).lower()
    if simulator != "mjx":
        return
    if backend not in {"gnn", "padded_mlp"} or any(task.split(":", 1)[0] != "truss-graph" for task in tasks):
        raise ValueError("sim_backend=mjx requires a graph backend with task=truss-graph.")
    if bool(_get(cfg, "visualize", False)):
        raise ValueError("MJX does not support interactive rendering; use sim_backend=mujoco.")
    if bool(_get(cfg, "save_video", False)) and str(_get(cfg, "eval_backend", "mujoco")) == "mjx":
        raise ValueError("MJX evaluation does not support video capture; use eval_backend=mujoco.")
    if bool(_get(cfg, "domain_randomization", False)):
        # Match the owning environment's default and runtime-alias semantics.
        from env.mujoco_gen.topology_envs import _has_enabled_physical_parameter_randomization

        params = _get(cfg, "domain_randomization_params", {})
        length_scale = bool(_get(_get(params, "length_scale", {}), "enabled", True))
        if length_scale or _has_enabled_physical_parameter_randomization(cfg):
            raise ValueError(
                "MJX requires fixed-shape domain randomization. Disable "
                "domain_randomization_params.length_scale.enabled and model-rebuilding "
                "physical_parameters, or use sim_backend=mujoco."
            )


def _components(env):
    """Visit leaf environments rather than just the first heterogeneous wrapper slot."""
    for name in ("envs", "buckets"):
        components = getattr(env, name, None)
        if components is not None:
            for component in components:
                yield from _components(component)
            return
    wrapped = getattr(env, "env", None)
    if wrapped is not None and wrapped is not env:
        yield from _components(wrapped)
    else:
        yield env


def padded_environment_contract(cfg: Any, env, *, owner: str = "training") -> dict:
    """Validate static spaces and physical action order without resetting the env."""
    spaces = getattr(env.observation_space, "spaces", {})
    if not {"x", "action_mask", "rigidity"}.issubset(spaces):
        raise ValueError(f"{owner}: padded MLP requires node state, action mask and rigidity spaces.")
    shape = tuple(spaces["x"].shape)
    expected_features = int(_get(cfg, "obs_dim", 6))
    if len(shape) != 2 or shape[1] != expected_features:
        raise ValueError(f"{owner}: node feature shape {shape} differs from width {expected_features}.")
    nodes = shape[0]
    if nodes < 1 or nodes > int(_get(cfg, "padded_mlp_max_nodes", 0)):
        raise ValueError(f"{owner}: {nodes} control nodes exceed padded_mlp_max_nodes={_get(cfg, 'padded_mlp_max_nodes', 0)}.")
    if tuple(spaces["action_mask"].shape) != (nodes,) or tuple(spaces["rigidity"].shape) != (1,):
        raise ValueError(f"{owner}: malformed action-mask or rigidity space.")
    action_dim = int(_get(cfg, "node_action_dim", 1))
    if tuple(env.action_space.shape) != (nodes, action_dim):
        raise ValueError(f"{owner}: action space must contain one action row per control node.")
    names = list(getattr(env, "graph_node_names", []))
    if len(names) != nodes or len(set(names)) != nodes:
        raise ValueError(f"{owner}: control-node identity/order metadata is missing or ambiguous.")
    mask_fn = getattr(env, "_policy_action_mask", None)
    if callable(mask_fn):
        mask = np.asarray(mask_fn(), dtype=bool)
    elif hasattr(env, "_base_passive_node_mask"):
        mask = ~np.asarray(env._base_passive_node_mask, dtype=bool)
    else:
        raise ValueError(f"{owner}: physical action-mask metadata is missing.")
    if mask.shape != (nodes,) or not mask.any():
        raise ValueError(f"{owner}: action mask must identify at least one active control node.")
    return dict(topology=str(getattr(env, "topology", owner)), control_nodes=nodes,
                node_features=shape[1], active_actions=int(mask.sum()) * action_dim,
                action_node_names=[name for name, active in zip(names, mask) if active])


def validate_padded_mlp_environment(cfg: Any, env) -> list[dict]:
    """Check all training slots and held-out native evaluation shapes before learning.

    Evaluation models are inspected without reset/step, teacher access, replay
    insertion or normalizer updates. The live training environment is untouched.
    """
    if str(_get(cfg, "sac_backend", "mlp")).lower() != "padded_mlp":
        return []
    records = [padded_environment_contract(cfg, component) for component in _components(env)]
    evaluated = {record["topology"] for record in records}
    from env import make_env

    eval_task = _get(cfg, "eval_task")
    if eval_task is not None and eval_task != _get(cfg, "task"):
        # Mirror OnlineTrainer's eval_task construction, including registered
        # aliases and any topology-list precedence in make_env. Inferring only
        # a topology suffix misses aliases with their own fixed model source.
        eval_cfg = deepcopy(cfg)
        eval_cfg.domain_randomization = False
        eval_cfg.mujoco_backend = str(_get(cfg, "eval_backend", "mujoco"))
        if str(_get(cfg, "mujoco_backend", "mujoco")) == "mjx" and eval_cfg.mujoco_backend == "mujoco":
            eval_cfg.num_envs = 1
        eval_cfg.task = eval_task
        eval_cfg.env_name = eval_task
        eval_cfg.tasks = [eval_task]
        eval_cfg.save_video = False
        eval_env = make_env(eval_cfg)
        try:
            records.extend(padded_environment_contract(cfg, component, owner=f"evaluation task {eval_task}")
                           for component in _components(eval_env))
        finally:
            eval_env.close()

    for topology in _list(_get(cfg, "eval_extra_topologies")):
        if topology in evaluated:
            continue
        eval_cfg = deepcopy(cfg)
        eval_cfg.task = str(_get(cfg, "task", "truss-graph")).split(":", 1)[0]
        eval_cfg.tasks = [eval_cfg.task]
        eval_cfg.truss_topology = topology
        eval_cfg.truss_topologies = None
        eval_cfg.topologies = None
        eval_cfg.multitask = False
        eval_cfg.num_envs = 1
        eval_cfg.mujoco_backend = "mujoco"
        eval_cfg.domain_randomization = False
        eval_cfg.save_video = False
        eval_env = make_env(eval_cfg)
        try:
            records.extend(padded_environment_contract(cfg, component, owner=f"evaluation {topology}")
                           for component in _components(eval_env))
        finally:
            eval_env.close()
        evaluated.add(topology)
    return records


def parameter_counts(model) -> dict[str, int]:
    """Report trainable actor/critic counts, excluding frozen target critics."""
    actor = sum(parameter.numel() for parameter in model.actor_parameters())
    critics = sum(parameter.numel() for parameter in model._Qs.parameters() if parameter.requires_grad)
    return dict(actor=actor, critics=critics, total=actor + critics)
