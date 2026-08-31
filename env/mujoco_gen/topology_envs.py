import copy
from dataclasses import fields
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MujocoRelativeObsEnv,
    NodeVelocityController,
    PRESETS,
    TrussPhysicalParameters,
    TrussEnvConfig,
    get_edge_index,
    get_edge_types,
    get_mujoco_spec,
    get_node_features,
)

from env.mujoco_gen.rigidity_reward import FirstNonRigidEigenvalueRewardMixin


def _safe_register(env_id, entry_point):
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


def _cfg_get(config, name, default=None):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _edge_roles_enabled(config) -> bool:
    features = _cfg_get(config, "graph_features", {})
    return bool(_cfg_get(features, "edge_roles", False))


def _semantic_edge_roles(source, *, graph_view: str) -> np.ndarray:
    """Map upstream edge labels to tube=0 and connector=1."""
    upstream_roles = get_edge_types(source, graph_view=graph_view)
    role_map = {"actuated": 0, "structural": 0, "connector": 1}
    try:
        return np.asarray([role_map[str(role)] for role in upstream_roles], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unsupported mujoco-truss-gen edge type: {exc.args[0]!r}") from exc


_MISSING = object()


# User-facing Hydra names mapped to mujoco-truss-gen's dataclass field names.
# Keep this explicit so configuration remains stable if the upstream dataclass
# gains non-range bookkeeping fields.
_RUNTIME_DOMAIN_RANDOMIZATION_FIELDS = {
    "body_mass_multiplier": "body_mass_multiplier_range",
    "body_inertia_multiplier": "body_inertia_multiplier_range",
    "dof_damping_multiplier": "dof_damping_multiplier_range",
    "dof_armature": "dof_armature_range",
    "dof_frictionloss": "dof_frictionloss_range",
    "actuator_gain_multiplier": "actuator_gain_multiplier_range",
    "actuator_bias_multiplier": "actuator_bias_multiplier_range",
    "actuator_dynprm_multiplier": "actuator_dynprm_multiplier_range",
    "geom_friction_slide": "geom_friction_slide_range",
    "geom_friction_torsional": "geom_friction_torsional_range",
    "geom_friction_rolling": "geom_friction_rolling_range",
    "tendon_stiffness": "tendon_stiffness_range",
    "tendon_damping": "tendon_damping_range",
    "tendon_armature": "tendon_armature_range",
    "tendon_frictionloss": "tendon_frictionloss_range",
    "gravity_z": "gravity_z_range",
    "initial_translation_x": "initial_translation_x_range",
    "initial_translation_y": "initial_translation_y_range",
    "initial_yaw": "initial_yaw_range",
}


def _is_mapping_like(value):
    return hasattr(value, "items") or hasattr(value, "keys")


def _physical_parameter_field_names():
    return {field.name for field in fields(TrussPhysicalParameters)}


def _normalize_physical_parameter_value(name, value):
    default_value = getattr(TrussPhysicalParameters(), name)
    if value in (None, "null"):
        return None
    if isinstance(default_value, list):
        if isinstance(value, str):
            raise ValueError(f"physical_parameters.{name} must be a list of numbers, got {value!r}")
        try:
            normalized = [float(item) for item in value]
        except TypeError as exc:
            raise ValueError(f"physical_parameters.{name} must be a list of numbers") from exc
        if len(normalized) != len(default_value):
            raise ValueError(
                f"physical_parameters.{name} must have {len(default_value)} values; got {len(normalized)}"
            )
        return normalized
    return float(value)


def _physical_parameters_from_config(config, overrides=None):
    if not bool(_cfg_get(config, "physical_parameters_enabled", True)):
        return None
    params_cfg = _cfg_get(config, "physical_parameters", {})
    field_names = _physical_parameter_field_names()
    values = {}
    unknown_fields = []
    if _is_mapping_like(params_cfg):
        for name in params_cfg.keys():
            if name not in field_names:
                unknown_fields.append(str(name))
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown physical_parameters field(s): {unknown}")

    for name in field_names:
        value = _cfg_get(params_cfg, name, _MISSING)
        if value is not _MISSING:
            values[name] = _normalize_physical_parameter_value(name, value)
    for name, value in (overrides or {}).items():
        if name not in field_names:
            raise ValueError(f"Unknown physical parameter override: {name}")
        values[name] = _normalize_physical_parameter_value(name, value)
    return TrussPhysicalParameters(**values)


def _sample_uniform_range(rng, name, spec, base_value):
    low = _cfg_get(spec, "min", _cfg_get(spec, "low", _MISSING))
    high = _cfg_get(spec, "max", _cfg_get(spec, "high", _MISSING))
    if low is _MISSING or high is _MISSING:
        raise ValueError(
            f"domain_randomization_params.physical_parameters.{name} requires min and max values."
        )
    if base_value is None or not isinstance(base_value, list):
        return float(rng.uniform(float(low), float(high)))

    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.ndim == 0:
        low = np.full(len(base_value), float(low))
    if high.ndim == 0:
        high = np.full(len(base_value), float(high))
    if low.shape != (len(base_value),) or high.shape != (len(base_value),):
        raise ValueError(
            f"domain_randomization_params.physical_parameters.{name} min/max must be scalars "
            f"or lists with {len(base_value)} values."
        )
    return rng.uniform(low, high).astype(float).tolist()


def _randomized_physical_parameter_overrides(config, rng):
    if not bool(_cfg_get(config, "physical_parameters_enabled", True)):
        return {}
    if not bool(_cfg_get(config, "domain_randomization", False)):
        return {}
    params = _cfg_get(config, "domain_randomization_params", {})
    physical_randomization = _cfg_get(params, "physical_parameters", {})
    if physical_randomization in (None, "null"):
        return {}

    base_params = _physical_parameters_from_config(config)
    if base_params is None:
        return {}
    overrides = {}
    field_names = _physical_parameter_field_names()
    if _is_mapping_like(physical_randomization):
        for name in physical_randomization.keys():
            if name not in field_names:
                raise ValueError(f"Unknown domain-randomized physical parameter: {name}")

    for name in field_names:
        spec = _cfg_get(physical_randomization, name, None)
        if spec in (None, "null") or not bool(_cfg_get(spec, "enabled", False)):
            continue
        base_value = getattr(base_params, name)
        overrides[name] = _sample_uniform_range(rng, name, spec, base_value)
    return overrides


def _has_enabled_physical_parameter_randomization(config):
    if not bool(_cfg_get(config, "physical_parameters_enabled", True)):
        return False
    if not bool(_cfg_get(config, "domain_randomization", False)):
        return False
    params = _cfg_get(config, "domain_randomization_params", {})
    physical_randomization = _cfg_get(params, "physical_parameters", {})
    if physical_randomization in (None, "null"):
        return False
    field_names = _physical_parameter_field_names()
    if _is_mapping_like(physical_randomization):
        for name in physical_randomization.keys():
            if name not in field_names:
                raise ValueError(f"Unknown domain-randomized physical parameter: {name}")
        return any(
            bool(_cfg_get(_cfg_get(physical_randomization, name, None), "enabled", False))
            for name in field_names
        )
    return False


def _enabled_range(params, name):
    spec = _cfg_get(params, name, None)
    if spec in (None, "null"):
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        low, high = spec
    else:
        if not bool(_cfg_get(spec, "enabled", False)):
            return None
        low = _cfg_get(spec, "min", _cfg_get(spec, "low", _MISSING))
        high = _cfg_get(spec, "max", _cfg_get(spec, "high", _MISSING))
    if low is _MISSING or high is _MISSING:
        raise ValueError(f"domain_randomization_params.{name} requires min and max values.")
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(
            f"domain_randomization_params.{name} must contain finite values with min <= max."
        )
    return (low, high)


def _copy_config_with(config, **updates):
    copied = copy.copy(config)
    for key, value in updates.items():
        setattr(copied, key, value)
    return copied


def available_truss_topologies():
    return tuple(sorted(PRESETS))


def parse_truss_topology_spec(topology_spec):
    topology = str(topology_spec)
    realistic = None
    if topology.endswith("-generated"):
        topology = topology.removesuffix("-generated")
    if ":" in topology:
        topology, variant = topology.split(":", 1)
        if variant == "realistic":
            realistic = True
        elif variant in {"simple", "default", "physical"}:
            realistic = False
        else:
            raise ValueError(
                f"Unknown truss topology variant '{variant}' in '{topology_spec}'. "
                "Supported variants: realistic, simple."
            )
    aliases = {
        "octehedron": "octahedron",
        "tetrehedron": "tetrahedron",
    }
    topology = aliases.get(topology, topology)
    return topology, realistic


def resolve_truss_topology(config):
    topology, _ = parse_truss_topology_spec(
        _cfg_get(config, "truss_topology", _cfg_get(config, "topology_id", "octahedron"))
    )
    if topology not in PRESETS:
        known = ", ".join(available_truss_topologies())
        raise ValueError(f"Unknown truss topology '{topology}'. Known mujoco-truss-gen presets: {known}")
    return topology


def resolve_truss_realistic(config):
    _, topology_realistic = parse_truss_topology_spec(
        _cfg_get(config, "truss_topology", _cfg_get(config, "topology_id", "octahedron"))
    )
    if topology_realistic is not None:
        return topology_realistic
    return bool(_cfg_get(config, "truss_realistic", False))


def _domain_randomization(config, topology, realistic):
    if not bool(_cfg_get(config, "domain_randomization", False)):
        return None
    params = _cfg_get(config, "domain_randomization_params", {})
    length_scale = _cfg_get(params, "length_scale", {})
    scale_enabled = bool(_cfg_get(length_scale, "enabled", True))
    physical_randomization_enabled = _has_enabled_physical_parameter_randomization(config)

    randomization_kwargs = {}
    supported_randomization_fields = {
        field.name for field in fields(DomainRandomizationConfig)
    }
    for config_name, randomization_name in _RUNTIME_DOMAIN_RANDOMIZATION_FIELDS.items():
        value_range = _enabled_range(params, config_name)
        if value_range is None:
            continue
        if randomization_name not in supported_randomization_fields:
            raise ValueError(
                f"domain_randomization_params.{config_name} is enabled, but the installed "
                f"mujoco-truss-gen does not support {randomization_name}. Upgrade the package."
            )
        randomization_kwargs[randomization_name] = value_range

    if not scale_enabled and not physical_randomization_enabled:
        return DomainRandomizationConfig(**randomization_kwargs)

    def randomized_model(rng):
        if scale_enabled:
            scale = rng.uniform(
                float(
                    _cfg_get(
                        length_scale,
                        "min",
                        _cfg_get(params, "length_scale_min", _cfg_get(config, "length_scale_min", 1.0)),
                    )
                ),
                float(
                    _cfg_get(
                        length_scale,
                        "max",
                        _cfg_get(params, "length_scale_max", _cfg_get(config, "length_scale_max", 1.0)),
                    )
                ),
            )
        else:
            scale = _fixed_model_scale(config)
        physical_params = _physical_parameters_from_config(
            config,
            overrides=_randomized_physical_parameter_overrides(config, rng),
        )
        return get_mujoco_spec(topology, realistic=realistic, scale=scale, physical_params=physical_params)

    randomization_kwargs["model_factory"] = randomized_model
    return DomainRandomizationConfig(**randomization_kwargs)


def _fixed_model_scale(config):
    scale = _cfg_get(config, "scale", _cfg_get(config, "truss_scale", 1.0))
    if scale in (None, "null"):
        return 1.0
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a positive finite value; got {scale!r}")
    return scale


def make_truss_env_config(config, *, model_source=None):
    topology = resolve_truss_topology(config)
    realistic = resolve_truss_realistic(config)
    if model_source is None:
        model_source = get_mujoco_spec(
            topology,
            realistic=realistic,
            scale=_fixed_model_scale(config),
            physical_params=_physical_parameters_from_config(config),
        )
    config_values = {
        "model_source": model_source,
        "max_steps": int(_cfg_get(config, "max_steps", 10000)),
        "nsubsteps": int(_cfg_get(config, "nsubsteps", 1)),
        "speed": float(_cfg_get(config, "speed", 0.01)),
        "forward_weight": float(_cfg_get(config, "forward_weight", 5.0)),
        "energy_weight": float(_cfg_get(config, "energy_weight", 0.005)),
        "alive_bonus": float(_cfg_get(config, "alive_bonus", 0.1)),
        "rigidity_weight": float(_cfg_get(config, "rigidity_weight", 0.5)),
        "slip_weight": float(_cfg_get(config, "slip_weight", 0.1)),
        "critical_eig_threshold": float(_cfg_get(config, "critical_eig_threshold", 0.03)),
        "slip_height": float(_cfg_get(config, "slip_height", 0.2)),
        "domain_randomization": _domain_randomization(config, topology, realistic),
        "normalize_observations": bool(
            _cfg_get(config, "normalize_observations", _cfg_get(config, "obs_norm", False))
        ),
    }
    optional_fields = {
        "control_noise_std": float,
        "control_noise_relative": bool,
        "runtime_apply_control_noise": bool,
        "max_forward_velocity": lambda value: None if value in (None, "null") else float(value),
        "zero_positive_forward_reward_on_termination": bool,
        "collapse_penalty": float,
        "zero_alive_bonus_on_termination": bool,
        "zero_rigidity_reward_on_termination": bool,
        "zero_velocity_shaping_on_termination": bool,
    }
    supported_fields = {field.name for field in fields(TrussEnvConfig)}
    for name, converter in optional_fields.items():
        if name in supported_fields:
            value = _cfg_get(config, name, _MISSING)
            if value is not _MISSING:
                config_values[name] = converter(value)
    return TrussEnvConfig(**config_values)


class MujocoPresetMLPEnv(MujocoRelativeObsEnv):
    """Flat observation/action environment for any mujoco-truss-gen preset."""

    def __init__(self, config, render_mode=None, rank=0):
        self.topology = resolve_truss_topology(config)
        super().__init__(make_truss_env_config(config), render_mode=render_mode, rank=rank)


class MujocoPresetGraphEnv(FirstNonRigidEigenvalueRewardMixin, MujocoRelativeObsEnv):
    """Graph observation environment for any mujoco-truss-gen preset."""

    def __init__(self, config, render_mode=None, rank=0, model_source=None):
        self.source_config = config
        self.topology = resolve_truss_topology(config)
        self.node_action_dim = int(_cfg_get(config, "node_action_dim", 1))
        self.node_feature_dim = 6
        super().__init__(
            make_truss_env_config(config, model_source=model_source),
            render_mode=render_mode,
            rank=rank,
        )

    def _use_control_graph(self):
        return bool(_cfg_get(self.source_config, "use_control_graph", False))

    def _graph_view(self):
        if self._use_control_graph():
            return "control"
        configured = str(_cfg_get(self.source_config, "truss_graph_view", "auto"))
        if configured == "auto":
            return "logical" if resolve_truss_realistic(self.source_config) else "physical"
        if configured not in {"physical", "logical"}:
            raise ValueError("truss_graph_view must be 'auto', 'physical', or 'logical'.")
        return configured

    def _node_names(self):
        if self._use_control_graph():
            if hasattr(self, "node_velocity_controller"):
                return list(self.node_velocity_controller.node_names)
            return list(getattr(self.mj_model, "control_node_names", []))
        if self._graph_view() == "logical":
            return self._logical_node_names()
        return list(self.mj_model.node_names)

    def _define_action_space(self):
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._node_names()), self.node_action_dim),
            dtype=np.float32,
        )

    def _on_model_changed(self):
        if self._use_control_graph():
            self._initialize_node_velocity_controller()
        super()._on_model_changed()
        self.logical_node_names = self._logical_node_names()
        self.graph_node_names = self._node_names()
        self.node_feature_dim = 6
        self._node_to_idx = {name: idx for idx, name in enumerate(self.graph_node_names)}
        self._actuator_edges = [] if self._use_control_graph() else self._build_actuator_edges()
        self._define_action_space()
        self._define_observation_space()

    def _initialize_node_velocity_controller(self):
        if self.node_action_dim != 1:
            raise ValueError("use_control_graph requires node_action_dim == 1.")

        self.node_velocity_controller = NodeVelocityController(
            self.mj_model.model,
            getattr(self.mj_model, "xml", None),
            self.mj_model.node_names,
            getattr(self.mj_model, "site_to_node", {}),
            getattr(self.mj_model, "external_actuator_ids", range(self.mj_model.model.nu)),
        )
        if not self.node_velocity_controller.enabled:
            raise ValueError(
                "use_control_graph requires mujoco-truss-gen control graph metadata "
                "or routed node velocity control support."
            )

        actuator_ids = np.asarray(self.node_velocity_controller.actuator_ids, dtype=int)
        external_ids = np.asarray(
            getattr(self.mj_model, "external_actuator_ids", range(self.mj_model.model.nu)),
            dtype=int,
        )
        if actuator_ids.shape != external_ids.shape or not np.array_equal(actuator_ids, external_ids):
            raise ValueError(
                "use_control_graph requires NodeVelocityController actuator ids to match "
                "the environment external actuator ordering."
            )

    def _define_observation_space(self):
        edge_index = get_edge_index(self.mj_model, graph_view=self._graph_view())
        observation_spaces = {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self._node_names()), self.node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(len(self._node_names()) - 1, 0),
                    shape=edge_index.shape,
                    dtype=np.int64,
                ),
                "action_mask": spaces.MultiBinary(len(self._node_names())),
                "rigidity": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        if _edge_roles_enabled(self.source_config):
            observation_spaces["edge_role"] = spaces.Box(
                low=0,
                high=1,
                shape=(edge_index.shape[1],),
                dtype=np.int64,
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def _policy_action_mask(self):
        if self._use_control_graph():
            passive = np.asarray(
                self.node_velocity_controller.passive_node_mask, dtype=bool
            )
            return ~passive

        actuated_nodes = {
            node_name
            for edge in self._actuator_edges
            if edge is not None
            for node_name in edge
        }
        return np.asarray(
            [name in actuated_nodes for name in self.graph_node_names], dtype=bool
        )

    def _get_obs(self):
        graph_view = self._graph_view()
        edge_index = get_edge_index(self.mj_model, graph_view=graph_view)
        features = get_node_features(
            self.mj_model,
            graph_view=graph_view,
            aggregation="connector_ball",
        )

        normalize_observations = bool(self.config.normalize_observations)
        bbox_dimensions = self.mj_model.initial_bounding_box_dimensions
        com = np.mean(features[:, :3], axis=0) if features.size else np.zeros(3)
        pos_rel = features[:, :3].copy()
        if pos_rel.size:
            pos_rel[:, 0] -= com[0]
            pos_rel[:, 1] -= com[1]

        if normalize_observations:
            pos_rel = pos_rel / bbox_dimensions
            vel_norm = features[:, 3:] / bbox_dimensions
        else:
            vel_norm = features[:, 3:]

        observation = {
            "x": np.concatenate([pos_rel, vel_norm], axis=1).astype(np.float32),
            "edge_index": edge_index,
            "action_mask": self._policy_action_mask(),
            "rigidity": np.asarray(
                [self._current_observation_rigidity()], dtype=np.float32
            ),
        }
        if _edge_roles_enabled(self.source_config):
            observation["edge_role"] = _semantic_edge_roles(
                self.mj_model, graph_view=graph_view
            )
        return observation

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._use_control_graph():
            return self._step_control_graph_node_action(action)
        if action.shape == (self.num_external_actuators,) or action.size == self.num_external_actuators:
            return self._step_actuator_action(action.reshape(self.num_external_actuators))
        if action.shape == (self.mj_model.model.nu,) or action.size == self.mj_model.model.nu:
            external_action = action.reshape(self.mj_model.model.nu)[self.mj_model.external_actuator_ids]
            return self._step_actuator_action(external_action)
        return self._step_actuator_action(self._node_action_to_actuator_action(action))

    def _step_control_graph_node_action(self, action):
        normalized_node_action, ctrl = self._control_graph_node_action_to_actuator_ctrl(action)
        previous_com = self._center_of_mass()
        self._advance(ctrl)
        reward, info, terminated = self._compute_reward(normalized_node_action, previous_com)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _control_graph_node_action_to_actuator_ctrl(self, action):
        node_actions = np.asarray(action, dtype=np.float32)
        expected_size = len(self.graph_node_names) * self.node_action_dim
        if node_actions.size != expected_size:
            raise ValueError(
                "Control graph node action must have one scalar action per control node; "
                f"got shape {node_actions.shape} for {len(self.graph_node_names)} nodes."
            )

        normalized_node_action = np.clip(
            node_actions.reshape(len(self.graph_node_names), self.node_action_dim)[:, 0],
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)
        node_commands = normalized_node_action * float(self.config.speed)
        ctrl = self.node_velocity_controller.clipped_edge_commands(
            self.mj_model.model,
            node_commands,
        ).astype(np.float32, copy=False)
        if ctrl.size != self.num_external_actuators:
            raise ValueError(
                "NodeVelocityController produced an actuator command vector with "
                f"{ctrl.size} entries for {self.num_external_actuators} external actuators."
            )
        return normalized_node_action, ctrl

    def _step_actuator_action(self, actuator_action):
        actuator_action = np.asarray(actuator_action, dtype=np.float32)
        if actuator_action.size != self.num_external_actuators:
            raise ValueError(
                "Actuator action must target external tendon controls only; "
                f"got shape {actuator_action.shape} for {self.num_external_actuators} external actuators."
            )
        actuator_action = np.clip(actuator_action.reshape(self.num_external_actuators), -1.0, 1.0)
        ctrl = self.mj_model.get_external_ctrl() + actuator_action * self.config.speed
        ctrlrange = self.mj_model.get_external_ctrlrange()
        ctrl = np.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        previous_com = self._center_of_mass()
        self._advance(ctrl)
        reward, info, terminated = self._compute_reward(actuator_action, previous_com)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    @property
    def num_external_actuators(self):
        return int(len(getattr(self.mj_model, "external_actuator_ids", range(self.mj_model.model.nu))))

    def _logical_node_names(self):
        return sorted(
            {self._logical_node_name(node_name) for node_name in self.mj_model.node_names},
            key=self._node_sort_key,
        )

    @staticmethod
    def _logical_node_name(node_name):
        return node_name.split("_tri_", 1)[0]

    @staticmethod
    def _node_sort_key(node_name):
        suffix = node_name.removeprefix("node_")
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    def _build_actuator_edges(self):
        tendon_edges = {}
        for tendon_name, node_pair in self._tendon_node_pairs_from_xml().items():
            tendon_id = self._tendon_id(tendon_name)
            if tendon_id >= 0:
                tendon_edges[tendon_id] = node_pair

        actuator_edges = []
        for actuator_id in getattr(self.mj_model, "external_actuator_ids", range(self.mj_model.model.nu)):
            tendon_id = int(self.mj_model.model.actuator_trnid[actuator_id, 0])
            actuator_edges.append(tendon_edges.get(tendon_id))
        return actuator_edges

    def _tendon_id(self, tendon_name):
        for tendon_id in range(self.mj_model.model.ntendon):
            if self.mj_model.model.tendon(tendon_id).name == tendon_name:
                return tendon_id
        return -1

    def _tendon_node_pairs_from_xml(self):
        tendon_edges = {}
        xml = getattr(self.mj_model, "xml", None)
        site_to_node = getattr(self.mj_model, "site_to_node", {})
        if xml and site_to_node:
            root = ET.fromstring(xml)
            tendon_root = root.find("tendon")
            if tendon_root is not None:
                for spatial in tendon_root.findall("spatial"):
                    tendon_name = spatial.get("name")
                    sites = [site_ref.get("site") for site_ref in spatial.findall("site")]
                    sites = [site for site in sites if site]
                    if tendon_name is None or len(sites) != 2:
                        continue
                    node_pair = tuple(site_to_node.get(site) for site in sites)
                    if None in node_pair or node_pair[0] == node_pair[1]:
                        continue
                    graph_pair = self._to_graph_node_pair(node_pair)
                    if graph_pair is not None:
                        tendon_edges[tendon_name] = graph_pair
                return tendon_edges

        for tendon_id in range(self.mj_model.model.ntendon):
            tendon_name = self.mj_model.model.tendon(tendon_id).name
            node_pair = self._node_pair_from_tendon_name(tendon_name)
            if node_pair is not None:
                tendon_edges[tendon_name] = node_pair
        return tendon_edges

    def _to_graph_node_pair(self, node_pair):
        if self._graph_view() == "logical":
            node_pair = tuple(self._logical_node_name(node) for node in node_pair)
        if node_pair[0] == node_pair[1]:
            return None
        if node_pair[0] not in self._node_to_idx or node_pair[1] not in self._node_to_idx:
            return None
        return node_pair

    def _node_pair_from_tendon_name(self, tendon_name):
        if not tendon_name.startswith("tendon_"):
            return None
        node_suffixes = tendon_name.removeprefix("tendon_").split("_node_")
        if len(node_suffixes) != 2:
            return None
        node_a = node_suffixes[0] if node_suffixes[0].startswith("node_") else f"node_{node_suffixes[0]}"
        node_b = f"node_{node_suffixes[1]}"
        return self._to_graph_node_pair((node_a, node_b))

    def _node_action_to_actuator_action(self, action):
        node_actions = np.asarray(action, dtype=np.float32)
        expected_size = len(self.graph_node_names) * self.node_action_dim
        if node_actions.size != expected_size:
            raise ValueError(
                "Graph node action must have one scalar action per graph node; "
                f"got shape {node_actions.shape} for {len(self.graph_node_names)} nodes."
            )
        node_actions = node_actions.reshape(len(self.graph_node_names), self.node_action_dim)
        actuator_action = np.zeros(self.num_external_actuators, dtype=np.float32)
        for actuator_id, node_pair in enumerate(self._actuator_edges):
            if node_pair is None:
                continue
            node_a, node_b = node_pair
            actuator_action[actuator_id] = node_actions[self._node_to_idx[node_a], 0] + node_actions[
                self._node_to_idx[node_b], 0
            ]
        return np.clip(actuator_action, -1.0, 1.0).astype(np.float32, copy=False)


class MujocoOctahedronGraphEnvRight(MujocoPresetGraphEnv):
    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(_copy_config_with(config, truss_topology="octahedron", truss_realistic=False), render_mode, rank)


class MujocoOctahedronGraphEnvRightRealistic(MujocoPresetGraphEnv):
    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(_copy_config_with(config, truss_topology="octahedron", truss_realistic=True), render_mode, rank)


class MujocoTetrahedronGraphEnvRight(MujocoPresetGraphEnv):
    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(_copy_config_with(config, truss_topology="tetrahedron", truss_realistic=False), render_mode, rank)


_safe_register("MujocoPresetMLPEnv-v0", "env.mujoco_gen.topology_envs:MujocoPresetMLPEnv")
_safe_register("MujocoPresetGraphEnv-v0", "env.mujoco_gen.topology_envs:MujocoPresetGraphEnv")
_safe_register("MujocoOctahedronGraphEnvRight-v0", "env.mujoco_gen.topology_envs:MujocoOctahedronGraphEnvRight")
_safe_register(
    "MujocoOctahedronGraphEnvRightRealistic-v0",
    "env.mujoco_gen.topology_envs:MujocoOctahedronGraphEnvRightRealistic",
)
_safe_register("MujocoTetrahedronGraphEnvRight-v0", "env.mujoco_gen.topology_envs:MujocoTetrahedronGraphEnvRight")
