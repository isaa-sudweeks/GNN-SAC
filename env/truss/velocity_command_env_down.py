import mujoco
import numpy as np
from gymnasium import spaces

from env.truss.relative_observation_env import MujocoRelativeObsEnv
from gymnasium.envs.registration import register

register(
    id="MujocoVelocityCommandEnvDown-v0",
    entry_point="env.truss.velocity_command_env_down:MujocoVelocityCommandEnvDown",
)


class MujocoVelocityCommandEnvDown(MujocoRelativeObsEnv):
    """
    MuJoCo truss environment with down velocity commands.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(config, render_mode, rank)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.mj_model.model.nu,), dtype=np.float32)

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

        forward_vel = self.mj_model.get_forward_velocity_y()
        energy_penalty = float(np.sum(np.square(action)))
        critical_eig = float(critical_eig)

        # To incentivize rolling I am going to do a slip penalty
        slip_penalty = float(self.mj_model.get_slip_penalty(height=self.config.slip_height))
        
        # DOWN velocity is negative Y velocity
        total_reward = forward_weight * -forward_vel + alive_bonus - energy_weight * energy_penalty + rigidity_weight * critical_eig - slip_weight * slip_penalty
        reward_dict = {
            "forward": forward_weight * -forward_vel,
            "alive": alive_bonus,
            "energy": -energy_weight * energy_penalty,
            "rigidity": rigidity_weight * critical_eig,
            "slip": -slip_weight * slip_penalty,
            "total_raw": total_reward + alive_bonus - energy_weight - slip_weight + rigidity_weight 
        }

        return total_reward, reward_dict, terminate

    def step(self, action):
        action = self._sanitize_action(action)
        ctrl = self._sanitize_ctrl(action * self.config.speed)
        self.mj_model.data.ctrl[:] = ctrl

        for _ in range(self.nsubsteps):
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

        truncate = self.steps >= self.max_steps
        reward, reward_dict, terminate = self._compute_reward(ctrl)
        obs = self._get_obs()

        return obs, reward, terminate, truncate, reward_dict
