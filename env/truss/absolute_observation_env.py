import numpy as np

from env.truss.base_env import MujocoTrussEnv


class MujocoAbsoluteObsEnv(MujocoTrussEnv):
    """
    MuJoCo truss environment with absolute node observations.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(
            config.xml_path,
            render_mode=render_mode,
            rank=rank,
            mujoco_backend=getattr(config, "mujoco_backend", "mjx"),
        )
        self.config = config
        self.max_steps = config.max_steps
        self.nsubsteps = config.nsubsteps

    def _get_obs(self):
        node_positions = self.mj_model.get_node_position_matrix().reshape(-1)
        node_velocities = self.mj_model.get_node_linear_velocity_matrix().reshape(-1)

        return np.concatenate([
            self.mj_model._data_array("ten_length"),
            self.mj_model._data_array("ten_velocity"),
            node_positions,
            node_velocities,
            self.mj_model.get_ctrl(),
        ]).astype(np.float32)
