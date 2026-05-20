import xml.etree.ElementTree as ET

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from mujoco_truss_gen import (
    MujocoRelativeObsEnv,
    TrussEnvConfig,
    get_mujoco_spec,
    get_edge_index,
    get_node_features,
)


register(
    id="MujocoOctahedronGraphEnvRightRealistic-v0",
    entry_point="env.mujoco_gen.octehedron_graph_env_right_realistic:MujocoOctahedronGraphEnvRightRealistic",
)


class MujocoOctahedronGraphEnvRightRealistic(MujocoRelativeObsEnv):
    """
    Generated octahedron truss environment that emits graph dict observations.

    The policy still acts on nodes. The environment maps node actions to actuator
    commands by summing the two endpoint node actions for each actuated tendon.
    """

    def __init__(self, config, render_mode=None, rank=0):
        model_source = get_mujoco_spec("octahedron", realistic=True)
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
        )
        super().__init__(truss_config, render_mode=render_mode, rank=rank)
        self.logical_node_names = self._logical_node_names()
        self.node_feature_dim = 6
        self.node_action_dim = 1
        self._node_to_idx = {name: idx for idx, name in enumerate(self.logical_node_names)}
        self._actuator_edges = self._build_actuator_edges()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.logical_node_names), self.node_action_dim),
            dtype=np.float32,
        )

    def _define_observation_space(self):
        num_nodes = len(self._logical_node_names())
        edge_index = get_edge_index(self.mj_model, graph_view="logical")
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_nodes, 6),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max(num_nodes - 1, 0),
                    shape=edge_index.shape,
                    dtype=np.int64,
                ),
            }
        )

    def _get_obs(self):
        edge_index = get_edge_index(self.mj_model, graph_view="logical")
        features = get_node_features(
            self.mj_model,
            graph_view="logical",
            aggregation="connector_ball",
        )

        return {
            "x": np.asarray(features, dtype=np.float32),
            "edge_index": edge_index,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape == (self.num_external_actuators,)
            or action.size == self.num_external_actuators
        ):
            return self._step_actuator_action(action.reshape(self.num_external_actuators))
        if action.shape == (self.mj_model.model.nu,) or action.size == self.mj_model.model.nu:
            external_action = action.reshape(self.mj_model.model.nu)[
                self.mj_model.external_actuator_ids
            ]
            return self._step_actuator_action(external_action)
        actuator_action = self._node_action_to_actuator_action(action)
        return self._step_actuator_action(actuator_action)

    def _step_actuator_action(self, actuator_action):
        actuator_action = np.asarray(actuator_action, dtype=np.float32)
        if actuator_action.size != self.num_external_actuators:
            raise ValueError(
                "Actuator action must target the external tendon controls only; "
                f"got shape {actuator_action.shape} for {self.num_external_actuators} external actuators."
            )
        actuator_action = actuator_action.reshape(self.num_external_actuators)
        actuator_action = np.clip(actuator_action, -1.0, 1.0)
        ctrl = self.mj_model.get_external_ctrl() + actuator_action * self.config.speed
        ctrlrange = self.mj_model.get_external_ctrlrange()
        ctrl_low = ctrlrange[:, 0]
        ctrl_high = ctrlrange[:, 1]
        ctrl = np.clip(ctrl, ctrl_low, ctrl_high)
        self._advance(ctrl)
        reward, info, terminated = self._compute_reward(actuator_action)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _build_actuator_edges(self):
        tendon_edges = {}
        for tendon_name, node_pair in self._tendon_node_pairs_from_xml().items():
            tendon_id = self._tendon_id(tendon_name)
            if tendon_id >= 0:
                tendon_edges[tendon_id] = node_pair

        external_actuator_ids = getattr(
            self.mj_model,
            "external_actuator_ids",
            np.arange(self.mj_model.model.nu, dtype=int),
        )
        actuator_edges = []
        for actuator_id in external_actuator_ids:
            tendon_id = int(self.mj_model.model.actuator_trnid[actuator_id, 0])
            actuator_edges.append(tendon_edges.get(tendon_id))
        return actuator_edges

    @property
    def num_external_actuators(self):
        external_ids = getattr(
            self.mj_model,
            "external_actuator_ids",
            range(self.mj_model.model.nu),
        )
        return int(len(external_ids))

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

    def _tendon_id(self, tendon_name):
        for tendon_id in range(self.mj_model.model.ntendon):
            if self.mj_model.model.tendon(tendon_id).name == tendon_name:
                return tendon_id
        return -1

    def _tendon_node_pairs_from_xml(self):
        tendon_edges = {}
        xml = getattr(self.mj_model, "xml", None)
        site_to_node = getattr(self.mj_model, "site_to_node", {})
        if not xml or not site_to_node:
            for tendon_id in range(self.mj_model.model.ntendon):
                tendon_name = self.mj_model.model.tendon(tendon_id).name
                node_pair = self._node_pair_from_tendon_name(tendon_name)
                if node_pair is not None:
                    tendon_edges[tendon_name] = node_pair
            return tendon_edges

        root = ET.fromstring(xml)
        tendon_root = root.find("tendon")
        if tendon_root is None:
            return tendon_edges

        for spatial in tendon_root.findall("spatial"):
            tendon_name = spatial.get("name")
            sites = [site_ref.get("site") for site_ref in spatial.findall("site")]
            sites = [site for site in sites if site]
            if tendon_name is None or len(sites) != 2:
                continue
            node_pair = tuple(site_to_node.get(site) for site in sites)
            if None in node_pair or node_pair[0] == node_pair[1]:
                continue
            logical_pair = tuple(self._logical_node_name(node) for node in node_pair)
            if logical_pair[0] != logical_pair[1]:
                tendon_edges[tendon_name] = logical_pair
        return tendon_edges

    def _node_pair_from_tendon_name(self, tendon_name):
        if not tendon_name.startswith("tendon_"):
            return None
        node_suffixes = tendon_name.removeprefix("tendon_").split("_node_")
        if len(node_suffixes) != 2:
            return None
        node_a = (
            node_suffixes[0]
            if node_suffixes[0].startswith("node_")
            else f"node_{node_suffixes[0]}"
        )
        node_b = f"node_{node_suffixes[1]}"
        node_a = self._logical_node_name(node_a)
        node_b = self._logical_node_name(node_b)
        if node_a not in self._node_to_idx or node_b not in self._node_to_idx:
            return None
        return node_a, node_b

    def _node_action_to_actuator_action(self, action):
        node_actions = np.asarray(action, dtype=np.float32)
        if node_actions.size != len(self.logical_node_names) * self.node_action_dim:
            raise ValueError(
                "Graph node action must have one scalar action per node; "
                f"got shape {node_actions.shape} for {len(self.logical_node_names)} logical nodes."
            )
        node_actions = node_actions.reshape(len(self.logical_node_names), self.node_action_dim)

        actuator_action = np.zeros(self.num_external_actuators, dtype=np.float32)
        for actuator_id, node_pair in enumerate(self._actuator_edges):
            if node_pair is None:
                continue
            node_a, node_b = node_pair
            ia, ib = self._node_to_idx[node_a], self._node_to_idx[node_b]
            actuator_action[actuator_id] = node_actions[ia, 0] + node_actions[ib, 0]
        return np.clip(actuator_action, -1.0, 1.0).astype(np.float32, copy=False)

