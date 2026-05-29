import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from mujoco_truss_gen import (
    MujocoRelativeObsEnv,
    TrussEnvConfig,
    get_mujoco_spec,
    get_octahedron_definition,
    DomainRandomizationConfig,
)


register(
    id="MujocoOctahedronGraphEnvRight-v0",
    entry_point="env.mujoco_gen.octehedron_graph_env_right:MujocoOctahedronGraphEnvRight",
)



class MujocoOctahedronGraphEnvRight(MujocoRelativeObsEnv):
    """
    Generated octahedron truss environment that emits graph dict observations.

    The policy still acts on nodes. The environment maps node actions to actuator
    commands by summing the two endpoint node actions for each actuated tendon.
    """

    def __init__(self, config, render_mode=None, rank=0):
        self.node_action_dim = 1
        node_dict, triangle_dict = get_octahedron_definition()
        model_source = get_mujoco_spec(node_dict, triangle_dict)

        if getattr(config, "domain_randomization", False):
            def randomized_model(rng: np.random.Generator):
                scale = rng.uniform(config.length_scale_min, config.length_scale_max)
                node_dict_rand, triangle_dict_rand = get_octahedron_definition(scale=scale)
                return get_mujoco_spec(node_dict_rand, triangle_dict_rand)

            domain_randomization = DomainRandomizationConfig(
                model_factory=randomized_model
            )
        else:
            domain_randomization = None

        truss_config = TrussEnvConfig(
            model_source=model_source,
            max_steps=int(config.max_steps),
            nsubsteps=int(config.nsubsteps),
            speed=float(config.speed),
            forward_weight=float(config.forward_weight),
            energy_weight=float(config.energy_weight),
            alive_bonus=float(config.alive_bonus),
            rigidity_weight=float(config.rigidity_weight),
            slip_weight=float(config.slip_weight),
            critical_eig_threshold=float(config.critical_eig_threshold),
            slip_height=float(config.slip_height),
            domain_randomization=domain_randomization,
        )
        super().__init__(truss_config, render_mode=render_mode, rank=rank)

    def _define_action_space(self):
        num_nodes = len(self.mj_model.node_names)
        node_action_dim = getattr(self, "node_action_dim", 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_nodes, node_action_dim),
            dtype=np.float32,
        )

    def _on_model_changed(self) -> None:
        super()._on_model_changed()
        self.node_feature_dim = 2 * len(self.mj_model.active_axes)
        self._node_to_idx = {name: idx for idx, name in enumerate(self.mj_model.node_names)}
        self._actuator_edges = self._build_actuator_edges()

    def _define_observation_space(self):
        num_nodes = len(self.mj_model.node_names)
        num_message_edges = 2 * len(self.mj_model.structural_edges)
        node_feature_dim = 2 * len(self.mj_model.active_axes)
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_nodes, node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(num_nodes - 1, 0),
                    shape=(2, num_message_edges),
                    dtype=np.int64,
                ),
            }
        )

    def _get_obs(self):
        node_positions = self.mj_model.get_node_position_dict()
        node_velocities = self.mj_model.get_node_velocity_linear_dict()
        com = np.mean(self.mj_model.get_node_position_matrix(), axis=0)
        active_axes = self.mj_model.active_axes

        features = []
        for node_name in self.mj_model.node_names:
            pos = node_positions[node_name]
            vel = node_velocities[node_name]
            node_features = []
            for axis in active_axes:
                axis_idx = "xyz".index(axis)
                node_features.append(pos[axis_idx] if axis == "z" else pos[axis_idx] - com[axis_idx])
            for axis in active_axes:
                node_features.append(vel["xyz".index(axis)])
            features.append(node_features)

        node_to_idx = {name: idx for idx, name in enumerate(self.mj_model.node_names)}
        directed_edges = []
        for node_a, node_b in self.mj_model.structural_edges:
            ia, ib = node_to_idx[node_a], node_to_idx[node_b]
            directed_edges.append((ia, ib))
            directed_edges.append((ib, ia))

        edge_index = np.array(directed_edges, dtype=np.int64).T
        if edge_index.size == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)

        return {
            "x": np.asarray(features, dtype=np.float32),
            "edge_index": edge_index,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (self.mj_model.model.nu,) or action.size == self.mj_model.model.nu:
            return self._step_actuator_action(action.reshape(self.mj_model.model.nu))
        actuator_action = self._node_action_to_actuator_action(action)
        return self._step_actuator_action(actuator_action)

    def _step_actuator_action(self, actuator_action):
        actuator_action = np.asarray(actuator_action, dtype=np.float32)
        actuator_action = np.clip(actuator_action, -1.0, 1.0)
        ctrl = self.mj_model.data.ctrl.copy() + actuator_action * self.config.speed
        ctrl_low = self.mj_model.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.mj_model.model.actuator_ctrlrange[:, 1]
        ctrl = np.clip(ctrl, ctrl_low, ctrl_high)
        self._advance(ctrl)
        reward, info, terminated = self._compute_reward(actuator_action)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _build_actuator_edges(self):
        tendon_edges = {}
        for tendon_id in range(self.mj_model.model.ntendon):
            tendon_name = self.mj_model.model.tendon(tendon_id).name
            node_pair = self._node_pair_from_tendon_name(tendon_name)
            if node_pair is not None:
                tendon_edges[tendon_id] = node_pair

        actuator_edges = []
        for actuator_id in range(self.mj_model.model.nu):
            tendon_id = int(self.mj_model.model.actuator_trnid[actuator_id, 0])
            actuator_edges.append(tendon_edges.get(tendon_id))
        return actuator_edges

    def _node_pair_from_tendon_name(self, tendon_name):
        if not tendon_name.startswith("tendon_"):
            return None
        node_suffixes = tendon_name.removeprefix("tendon_").split("_node_")
        if len(node_suffixes) != 2:
            return None
        node_a = node_suffixes[0] if node_suffixes[0].startswith("node_") else f"node_{node_suffixes[0]}"
        node_b = f"node_{node_suffixes[1]}"
        if node_a not in self._node_to_idx or node_b not in self._node_to_idx:
            return None
        return node_a, node_b

    def _node_action_to_actuator_action(self, action):
        node_actions = np.asarray(action, dtype=np.float32)
        if node_actions.size != len(self.mj_model.node_names) * self.node_action_dim:
            raise ValueError(
                "Graph node action must have one scalar action per node; "
                f"got shape {node_actions.shape} for {len(self.mj_model.node_names)} nodes."
            )
        node_actions = node_actions.reshape(len(self.mj_model.node_names), self.node_action_dim)

        actuator_action = np.zeros(self.mj_model.model.nu, dtype=np.float32)
        for actuator_id, node_pair in enumerate(self._actuator_edges):
            if node_pair is None:
                continue
            node_a, node_b = node_pair
            ia, ib = self._node_to_idx[node_a], self._node_to_idx[node_b]
            actuator_action[actuator_id] = node_actions[ia, 0] + node_actions[ib, 0]
        return np.clip(actuator_action, -1.0, 1.0).astype(np.float32, copy=False)



