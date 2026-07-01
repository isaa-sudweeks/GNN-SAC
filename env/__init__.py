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

def _num_policy_actuators(env):
    mj_model = env.unwrapped.mj_model
    return int(len(getattr(mj_model, "external_actuator_ids", range(mj_model.model.nu))))

def _is_mjx_vector_graph_run(cfg):
    task = str(getattr(cfg, "task", "")).split(":", 1)[0]
    return str(getattr(cfg, "mujoco_backend", "mujoco")).lower() == "mjx" and task == "truss-graph"

def _make_mjx_vector_graph_env(cfg):
    if bool(getattr(cfg, "multitask", False)):
        raise ValueError(
            "The MJX vector backend supports one fixed topology per run; "
            "use separate runs for truss_topologies or set mujoco_backend=mujoco."
        )
    from env.mujoco_gen.mjx_vector_env import MjxVectorGraphEnv

    cfg.use_control_graph = True
    env = MjxVectorGraphEnv(cfg)
    cfg.obs_shape = {key: value.shape for key, value in env.observation_space.spaces.items()}
    cfg.obs_dim = env.node_feature_dim
    cfg.node_feature_dim = env.node_feature_dim
    cfg.node_action_dim = env.node_action_dim
    cfg.action_dim = env.node_action_dim
    cfg.num_policy_actions = int(np.prod(env.action_space.shape))
    cfg.num_actuators = _num_policy_actuators(env)
    cfg.episode_length = int(env.max_episode_steps)
    cfg.seed_steps = int(max(1000, 5 * cfg.episode_length))
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
            cfg.policy_action_counts.append(int(np.prod(e.action_space.shape)))
            cfg.policy_actuator_counts.append(_num_policy_actuators(e))
            cfg.episode_lengths.append(int(_max_episode_steps(e)))
        cfg.action_dim = int(max(cfg.action_dims))
        cfg.episode_length = int(max(cfg.episode_lengths))
        cfg.seed_steps = int(max(1000, 5*cfg.episode_length))
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
            cfg.node_action_dim = int(getattr(env.unwrapped, "node_action_dim", env.action_space.shape[-1]))
            cfg.action_dim = cfg.node_action_dim
            cfg.num_policy_actions = int(max(cfg.policy_action_counts))
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
            cfg.node_action_dim = int(getattr(env.unwrapped, "node_action_dim", env.action_space.shape[-1]))
            cfg.action_dim = cfg.node_action_dim
            cfg.num_policy_actions = int(np.prod(env.action_space.shape))
            cfg.num_actuators = _num_policy_actuators(env)
            cfg.episode_length = episode_length
            cfg.seed_steps = max(1000, 5*cfg.episode_length)
            return env
        try: # Dict
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
        except: #Box 
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
            cfg.obs_dim = int(np.prod(env.observation_space.shape))
        cfg.action_dim = env.action_space.shape[0]
        cfg.episode_length = episode_length
        cfg.seed_steps = max(1000, 5*cfg.episode_length)
        # TODO: Add support for wrappers
        return env 
