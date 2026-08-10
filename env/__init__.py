import warnings 

import gymnasium as gym
import numpy as np
from env.wrappers.tensor import TensorWrapper

def _missing_dependencies_factory(name, exc):
    def missing_dependencies(cfg):
        task = getattr(cfg, 'task', cfg)
        raise ValueError(
            f'Missing dependencies for {name} task "{task}"; original import error: {exc}'
        )
    return missing_dependencies

try:
	from env.dmcontrol import make_env as make_dm_control_env
except Exception as exc:
	make_dm_control_env = _missing_dependencies_factory('dmcontrol', exc)
try:
	from env.maniskill import make_env as make_maniskill_env
except Exception as exc:
	make_maniskill_env = _missing_dependencies_factory('maniskill', exc)
try:
	from env.metaworld import make_env as make_metaworld_env
except Exception as exc:
	make_metaworld_env = _missing_dependencies_factory('metaworld', exc)
try:
	from env.myosuite import make_env as make_myosuite_env
except Exception as exc:
	make_myosuite_env = _missing_dependencies_factory('myosuite', exc)
try:
	from env.mujoco import make_env as make_mujoco_env
except Exception as exc:
	make_mujoco_env = _missing_dependencies_factory('mujoco', exc)

warnings.filterwarnings('ignore', category=DeprecationWarning)

def _max_episode_steps(env):
    """
    Resolve max episode steps across wrapper stacks.
    """
    current = env
    while current is not None:
        steps = getattr(current, 'max_episode_steps', None)
        if steps is not None:
            return steps
        current = getattr(current, 'env', None)
    spec = getattr(env.unwrapped, 'spec', None)
    if spec is not None and getattr(spec, 'max_episode_steps', None) is not None:
        return spec.max_episode_steps
    raise AttributeError('Environment does not define max_episode_steps')

def _configure_episode_length(cfg, episode_length):
    """Record the environment horizon without overriding an explicit seed budget."""
    cfg.episode_length = int(episode_length)
    if getattr(cfg, "seed_steps", None) is None:
        cfg.seed_steps = int(max(1000, 5 * cfg.episode_length))

def _obs_shapes(env, cfg):
    try:
        return {k: v.shape for k, v in env.observation_space.spaces.items()}
    except AttributeError:
        shape = getattr(env.observation_space, "shape", None)
        if shape is None:
            raise AttributeError(f"Unsupported observation space: {env.observation_space}")
        return {cfg.get('obs', 'state'): shape}

def _is_graph_env(cfg):
    tasks = list(getattr(cfg, "tasks", [getattr(cfg, "task", "")]))
    return bool(getattr(cfg, "use_graph_observations", False)) or any("graph" in str(task) for task in tasks)


def _configure_graph_feature_dims(cfg):
    features = getattr(cfg, "graph_features", {})
    get_value = (
        features.get
        if hasattr(features, "get")
        else lambda name, default=False: getattr(features, name, default)
    )
    node_roles = bool(get_value("node_roles", False))
    edge_roles = bool(get_value("edge_roles", False))
    edge_distance = bool(get_value("edge_distance", False))
    use_virtual_node = bool(getattr(cfg, "use_virtual_node", False))
    cfg.effective_node_feature_dim = (
        int(cfg.node_feature_dim)
        + 2 * int(node_roles)
        + 2 * int(use_virtual_node)
    )
    cfg.edge_feature_dim = 3 * int(edge_roles) + int(edge_distance)

def _num_policy_actuators(env):
    mj_model = env.unwrapped.mj_model
    return int(len(getattr(mj_model, "external_actuator_ids", range(mj_model.model.nu))))

def _num_policy_actions(env):
    """Count node-action scalars that can affect an actuator."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "_policy_action_mask"):
        active_nodes = int(np.asarray(unwrapped._policy_action_mask(), dtype=bool).sum())
        return active_nodes * int(getattr(unwrapped, "node_action_dim", 1))
    passive_names = set(getattr(unwrapped, "passive_node_names", ()))
    graph_names = list(getattr(unwrapped, "graph_node_names", ()))
    if graph_names:
        active_nodes = sum(name not in passive_names for name in graph_names)
        return active_nodes * int(getattr(unwrapped, "node_action_dim", 1))
    return int(np.prod(env.action_space.shape))

def _is_mjx_vector_graph_run(cfg):
    task = str(getattr(cfg, "task", "")).split(":", 1)[0]
    return str(getattr(cfg, "mujoco_backend", "mujoco")).lower() == "mjx" and task == "truss-graph"

def _make_mjx_vector_graph_env(cfg):
    if bool(getattr(cfg, "multitask", False)):
        raise ValueError(
            "MJX topology bucketing splits num_envs across truss_topologies directly; "
            "set multitask=false."
        )

    cfg.use_control_graph = True
    topologies = _configured_truss_topologies(cfg)
    if len(topologies) > 1:
        from env.mujoco_gen.mjx_topology_bucket_env import MjxTopologyBucketEnv

        env = MjxTopologyBucketEnv(cfg, topologies)
        component_envs = env.buckets
        cfg.topology_allocations = env.topology_allocations
        cfg.envs_per_topology = env.envs_per_topology
        cfg.num_envs = env.num_envs
        cfg.tasks = [f"truss-graph:{topology}" for topology in topologies]
    else:
        from env.mujoco_gen.mjx_vector_env import MjxVectorGraphEnv

        if topologies:
            cfg.truss_topology = topologies[0]
        env = MjxVectorGraphEnv(cfg)
        component_envs = [env]

    cfg.obs_shape = {key: value.shape for key, value in env.observation_space.spaces.items()}
    cfg.obs_dim = env.node_feature_dim
    cfg.node_feature_dim = env.node_feature_dim
    _configure_graph_feature_dims(cfg)
    cfg.node_action_dim = env.node_action_dim
    cfg.action_dim = env.node_action_dim
    cfg.policy_action_counts = [
        _num_policy_actions(component) for component in component_envs
    ]
    cfg.node_counts = [int(component.action_space.shape[0]) for component in component_envs]
    cfg.num_nodes = int(max(cfg.node_counts))
    cfg.policy_actuator_counts = [
        _num_policy_actuators(component) for component in component_envs
    ]
    cfg.num_policy_actions = int(max(cfg.policy_action_counts))
    cfg.num_actuators = int(max(cfg.policy_actuator_counts))
    _configure_episode_length(cfg, env.max_episode_steps)
    return TensorWrapper(env, graph_observations=True)

def _configured_truss_topologies(cfg):
    topologies = getattr(cfg, "truss_topologies", None)
    if topologies in (None, "null"):
        return []
    if isinstance(topologies, str):
        return [topologies]
    return list(topologies)

def _apply_truss_topology_tasks(cfg):
    topologies = _configured_truss_topologies(cfg)
    if not topologies:
        return
    base_task = str(getattr(cfg, "task", "truss-graph")).split(":", 1)[0]
    if (
        str(getattr(cfg, "mujoco_backend", "mujoco")).lower() == "mjx"
        and base_task == "truss-graph"
    ):
        cfg.task = base_task
        cfg.tasks = [f"{base_task}:{topology}" for topology in topologies]
        if len(topologies) == 1:
            cfg.truss_topology = topologies[0]
        return
    if len(topologies) == 1:
        cfg.truss_topology = topologies[0]
        cfg.task = base_task
        cfg.tasks = [base_task]
        return
    if int(getattr(cfg, "num_envs", 1)) > 1 and not bool(getattr(cfg, "multitask", False)):
        raise ValueError("Use either truss_topologies for topology multitask runs or num_envs>1 for repeated same-env runs, not both.")
    cfg.multitask = True
    cfg.num_envs = 1
    cfg.task = base_task
    cfg.tasks = [f"{base_task}:{topology}" for topology in topologies]

def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    if hasattr(gym.logger, "set_level"):
        gym.logger.set_level(40)
    else:
        gym.logger.min_level = 40
    _apply_truss_topology_tasks(cfg)
    env = None
    num_envs = int(getattr(cfg, "num_envs", 1))
    multitask = bool(getattr(cfg, "multitask", False))
    if _is_mjx_vector_graph_run(cfg):
        return _make_mjx_vector_graph_env(cfg)
    if multitask and num_envs > 1:
        raise ValueError("Use either multitask=true with cfg.tasks or num_envs>1 for repeated same-task envs, not both.")
    if multitask or num_envs > 1:
        if multitask:
            from env.wrappers.multitask import MultitaskWrapper
            env = MultitaskWrapper(cfg, [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env])
        else:
            from env.wrappers.repeated import RepeatedEnvWrapper
            env = RepeatedEnvWrapper(cfg, [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env])
        cfg.obs_shapes = []
        cfg.action_dims = []
        cfg.policy_action_counts = []
        cfg.policy_actuator_counts = []
        cfg.episode_lengths = []
        for e in env.envs:
            cfg.obs_shapes.append(_obs_shapes(e, cfg))
            cfg.action_dims.append(int(e.action_space.shape[0]))
            cfg.policy_action_counts.append(_num_policy_actions(e))
            cfg.policy_actuator_counts.append(_num_policy_actuators(e))
            cfg.episode_lengths.append(int(_max_episode_steps(e)))
        cfg.action_dim = int(max(cfg.action_dims))
        _configure_episode_length(cfg, max(cfg.episode_lengths))
        cfg.obs_shape = {}
        for shape_dict in cfg.obs_shapes:
            for k, v in shape_dict.items():
                if k not in cfg.obs_shape:
                    cfg.obs_shape[k] = list(v)
                else:
                    cfg.obs_shape[k] = [max(a, b) for a, b in zip(cfg.obs_shape[k], v)]
        for k in cfg.obs_shape:
            cfg.obs_shape[k] = tuple(cfg.obs_shape[k])
        is_graph_env = _is_graph_env(cfg)
        env = TensorWrapper(env, graph_observations=is_graph_env)
        if is_graph_env:
            cfg.obs_dim = int(getattr(env.unwrapped, "node_feature_dim", cfg.get("node_feature_dim", 0)))
            cfg.node_feature_dim = cfg.obs_dim
            _configure_graph_feature_dims(cfg)
            cfg.node_action_dim = int(getattr(env.unwrapped, "node_action_dim", env.action_space.shape[-1]))
            cfg.action_dim = cfg.node_action_dim
            cfg.num_policy_actions = int(max(cfg.policy_action_counts))
            cfg.node_counts = list(cfg.action_dims)
            cfg.num_nodes = int(max(cfg.node_counts))
            cfg.num_actuators = int(max(cfg.policy_actuator_counts))
        return env
    else:
        errors = []
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env]:
            try:
                env = fn(cfg)
            except ValueError as exc:
                errors.append(str(exc))
                pass 
        if env is None:
            details = '; '.join(errors)
            raise ValueError(f'Failed to make environment "{cfg.task}": {details}')
        episode_length = _max_episode_steps(env)
        is_graph_env = _is_graph_env(cfg)
        env = TensorWrapper(env, graph_observations=is_graph_env)
        if is_graph_env:
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
            cfg.obs_dim = int(getattr(env.unwrapped, "node_feature_dim", cfg.get("node_feature_dim", 0)))
            cfg.node_feature_dim = cfg.obs_dim
            _configure_graph_feature_dims(cfg)
            cfg.node_action_dim = int(getattr(env.unwrapped, "node_action_dim", env.action_space.shape[-1]))
            cfg.action_dim = cfg.node_action_dim
            cfg.num_policy_actions = _num_policy_actions(env)
            cfg.node_counts = [int(env.action_space.shape[0])]
            cfg.num_nodes = cfg.node_counts[0]
            cfg.num_actuators = _num_policy_actuators(env)
            _configure_episode_length(cfg, episode_length)
            return env
        try: # Dict
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
        except: #Box 
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
            cfg.obs_dim = int(np.prod(env.observation_space.shape))
        cfg.action_dim = env.action_space.shape[0]
        _configure_episode_length(cfg, episode_length)
        # TODO: Add support for wrappers
        return env 
