import numpy as np
import mujoco
from gymnasium import spaces

from env.truss.base_env import MujocoTrussEnv


class MujocoRelativeObsEnv(MujocoTrussEnv):
    """
    Translationally invariant MuJoCo truss environment.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(config.xml_path, render_mode=render_mode, rank=rank)
        self.config = config
        self.max_steps = config.max_steps
        self.nsubsteps = config.nsubsteps
        self.metadata["render_fps"] = int(np.round(1.0 / (self.nsubsteps * self.mj_model.model.opt.timestep)))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.mj_model.model.nu,), dtype=np.float32)

    def _get_obs(self):
        # We need absolute Z position to know where the ground is.
        # We need absolute velocities to predict forward velocity rewards.
        # Horizontal coordinates are represented relative to the COM.
        node_positions = self.mj_model.get_node_position_dict()
        node_velocities = self.mj_model.get_node_velocity_linear_dict()
        com = np.mean(self.mj_model.get_node_position_matrix(), axis=0)
        active_axes = self.mj_model.active_axes
        
        relative_positions = []
        absolute_velocities = []
        
        for node_name in self.mj_model.node_names:
            pos = node_positions[node_name]
            vel = node_velocities[node_name]

            for axis in active_axes:
                axis_idx = "xyz".index(axis)
                if axis == "z":
                    relative_positions.append(pos[axis_idx])
                else:
                    relative_positions.append(pos[axis_idx] - com[axis_idx])
                absolute_velocities.append(vel[axis_idx])
            
        
        return np.concatenate([
            np.array(relative_positions),
            np.array(absolute_velocities),
            self.mj_model.data.ctrl.copy()
        ]).astype(np.float32)

    def step(self, action):
        current_ctrl = self._sanitize_ctrl(self.mj_model.data.ctrl.copy())
        
        # Action is interpreted as a delta: [-1, 1] means [-speed, speed]
        action = self._sanitize_action(action)
        speed = self.config.speed
        new_ctrl = current_ctrl + action * speed
        
        # Clip to hardware (global actuator) limits so we don't break simulation bounds.
        new_ctrl = self._sanitize_ctrl(new_ctrl)
        
        self.mj_model.data.ctrl[:] = new_ctrl

        for _ in range(self.nsubsteps):
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

        truncate = self.steps >= self.max_steps
        reward, reward_dict, terminate = self._compute_reward(action)
        obs = self._get_obs()
        
        return obs, reward, terminate, truncate, reward_dict

    def _compute_reward(self, action):
        # User Set Hyperparameters  
        forward_weight = self.config.forward_weight 
        energy_weight = self.config.energy_weight 
        alive_bonus = self.config.alive_bonus
        rigidity_weight = self.config.rigidity_weight
        slip_weight = self.config.slip_weight


        terminate = False
        critical_eig = self.mj_model.collapse_check()
        if critical_eig < self.config.critical_eig_threshold:
            terminate = True

        foward_vel = self.mj_model.get_forward_velocity_x()
        energy_penalty = float(np.sum(np.square(action)))
        critical_eig = float(critical_eig)

        # To incentiveze rolling I am going to do a slip penalty
        slip_penalty = float(self.mj_model.get_slip_penalty(height=self.config.slip_height))
        
        total_reward = forward_weight * foward_vel + alive_bonus - energy_weight * energy_penalty + rigidity_weight * critical_eig - slip_weight * slip_penalty
        reward_dict = {
            "forward": forward_weight * foward_vel,
            "alive": alive_bonus,
            "energy": -energy_weight * energy_penalty,
            "rigidity": rigidity_weight * critical_eig,
            "slip": -slip_weight * slip_penalty,
            "total_raw": total_reward,
        }

        return total_reward, reward_dict, terminate
