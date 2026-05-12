import numpy as np
from gymnasium import spaces

from env.truss.relative_observation_env import MujocoRelativeObsEnv
from gymnasium.envs.registration import register

register(
    id="MujocoVelocityCommandEnvRight-v0",
    entry_point="env.truss.velocity_command_env:MujocoVelocityCommandEnv",
)


class MujocoVelocityCommandEnv(MujocoRelativeObsEnv):
    """
    MuJoCo truss environment with direct velocity commands.
    """

    def __init__(self, config, render_mode=None, rank=0):
        super().__init__(config, render_mode, rank)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.mj_model.model.nu,), dtype=np.float32)

    def step(self, action):
        action = self._sanitize_action(action)
        ctrl = self._sanitize_ctrl(action * self.config.speed)
        self._advance(ctrl)

        self.steps += 1

        truncate = self.steps >= self.max_steps
        reward, reward_dict, terminate = self._compute_reward(ctrl)
        obs = self._get_obs()

        return obs, reward, terminate, truncate, reward_dict
