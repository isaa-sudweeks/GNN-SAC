import mujoco
import numpy as np 
from gymnasium import spaces
import torch
from mujoco_truss_gen import get_mujoco_spec, get_edge_index, get_node_features
from torch_geometric.data import Data




class MujocoOctahedronEnvRight(MujocoRelativeObsEnv):
    def __init__(self, cfg, spec):
        spec = get_mujoco_spec("octahedron", realistic=False)

        configuration = TrussEnvConfig(
            model_source=spec,
            max_steps=cfg.max_steps,
            nsubsteps=cfg.nsubsteps,
            speed=cfg.speed,
        )

        super().__init__(configuration)
    
    def _get_obs(self):
        edge_index = get_edge_index(self.mj_model)
        node_features = get_node_features(self.mj_model)

        edge_index_tensor = torch.from_numpy(edge_index)
        x_tensor = torch.from_numpy(node_features)

        data = Data(x=x_tensor, edge_index=edge_index_tensor)
        return data


    