import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register
from torch_geometric.data import Data

from env.truss.relative_observation_env import MujocoRelativeObsEnv


register(
    id="MujocoOctahedronGraphEnvRight-v0",
    entry_point="env.mujoco_gen.octehedron_graph_env_right:MujocoOctahedronGraphEnvRight",
)


class MujocoOctahedronGraphEnvRight(MujocoRelativeObsEnv):
    """
    Octahedron truss environment that emits PyG graph observations.

    Actions are intentionally still unresolved for node-level graph policies. The
    flat actuator action path remains available for environment smoke tests.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(config, render_mode=render_mode, rank=rank)
        self.node_feature_dim = 2 * len(self.mj_model.active_axes)
        self.node_action_dim = 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.mj_model.node_names), self.node_action_dim),
            dtype=np.float32,
        )

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

        return Data(
            x=torch.as_tensor(np.asarray(features, dtype=np.float32)),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (self.mj_model.model.nu,) or action.size == self.mj_model.model.nu:
            return super().step(action.reshape(self.mj_model.model.nu))
        raise NotImplementedError(
            "Graph node actions must be translated to actuator controls before stepping this environment."
        )
