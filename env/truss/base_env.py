import gymnasium as gym
import mujoco
import numpy as np
import time
import warnings
from pathlib import Path
from gymnasium import spaces

from env.truss.model import MujocoModel

try:
    import mujoco.viewer as mujoco_viewer
except ImportError:
    mujoco_viewer = None


class MujocoTrussEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, xml_path, render_mode=None, rank=0):
        super().__init__()
        self.render_mode = render_mode
        self.rank = rank
        self.xml_path = self._resolve_xml_path(xml_path)
        self.mj_model = MujocoModel(self.xml_path)
        self._define_action_space()
        self._define_observation_space()

        self.viewer = None
        self.renderer = None
        self.cam = None
        self._rgb_render_failed = False
        self.last_render_time = time.time()
        self.steps = 0
        self.max_steps = 10_000
        self.nsubsteps = 1

    @staticmethod
    def _resolve_xml_path(xml_path):
        path = Path(xml_path).expanduser()
        if path.is_absolute() and path.exists():
            return str(path)

        repo_root = Path(__file__).resolve().parents[2]
        package_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path.cwd() / path,
            repo_root / path,
            package_root / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(f"Could not resolve MuJoCo XML path: {xml_path}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mj_model.reset(self.np_random)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate the COM of the robot
        node_positions = self.mj_model.get_node_position_matrix()
        node_velocities = self.mj_model.get_node_linear_velocity_matrix()
        com = np.mean(node_positions, axis=0)
        com_vel = np.mean(node_velocities, axis=0)

        return np.concatenate([
            self.mj_model.data.ten_length,
            self.mj_model.data.ten_velocity,
            [com[0], com[2]],
            [com_vel[0], com_vel[2]]
        ]).astype(np.float32)

    def _define_action_space(self):
        # Get the number of actuators and thier limits from the model
        self.action_space = spaces.Box(low=self.mj_model.model.actuator_ctrlrange[:, 0], high=self.mj_model.model.actuator_ctrlrange[:, 1], dtype=np.float32) # Not sure if this works 

    def _define_observation_space(self):
        # Dynamically set the space shape based on the actual returned observation
        dummy_obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32)

    def _sanitize_box_value(self, value, low, high):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != low.shape:
            value = np.reshape(value, low.shape)

        value = np.where(np.isnan(value), 0.0, value)
        value = np.where(np.isposinf(value), high, value)
        value = np.where(np.isneginf(value), low, value)
        return np.clip(value, low, high).astype(np.float32, copy=False)

    def _sanitize_action(self, action):
        return self._sanitize_box_value(action, self.action_space.low, self.action_space.high)

    def _sanitize_ctrl(self, ctrl):
        ctrl_low = self.mj_model.model.actuator_ctrlrange[:, 0].astype(np.float32, copy=False)
        ctrl_high = self.mj_model.model.actuator_ctrlrange[:, 1].astype(np.float32, copy=False)
        return self._sanitize_box_value(ctrl, ctrl_low, ctrl_high)

    def step(self, action):
        action = self._sanitize_ctrl(action)
        self.mj_model.data.ctrl[:] = action

        for _ in range(self.nsubsteps):
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

        truncate = self.steps >= self.max_steps
        reward, reward_dict, terminate = self._compute_reward(action)
        obs = self._get_obs()
        
        return obs, reward, terminate, truncate, reward_dict

    def render(self):
        if self.render_mode == "rgb_array":
            if self.rank != 0:
                # Return a minimal 0-array for ignored background ranks to not crash MacOS Driver limits
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if self._rgb_render_failed:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                if self.renderer is None:
                    renderer = mujoco.Renderer(self.mj_model.model, 480, 640)
                    cam = mujoco.MjvCamera()
                    mujoco.mjv_defaultFreeCamera(self.mj_model.model, cam)
                    cam.distance = self.mj_model.model.stat.extent * 1.5
                    self.renderer = renderer
                    self.cam = cam

                # Update camera lookat to track the robot's COM
                com = np.mean(self.mj_model.get_node_position_matrix(), axis=0)
                self.cam.lookat[:] = com

                self.renderer.update_scene(self.mj_model.data, camera=self.cam)
                return self.renderer.render()
            except Exception as exc:
                self._rgb_render_failed = True
                if self.renderer is not None:
                    try:
                        self.renderer.close()
                    except Exception:
                        pass
                    self.renderer = None
                warnings.warn(
                    f"MuJoCo rgb_array rendering failed ({exc}); returning blank frames instead."
                )
                return np.zeros((480, 640, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            if self.viewer is None:
                if mujoco_viewer is None:
                    raise RuntimeError(
                        "MuJoCo human viewer is unavailable in this Python environment. "
                        "Install a MuJoCo build that includes the viewer module."
                    )
                self.viewer = mujoco_viewer.launch_passive(self.mj_model.model, self.mj_model.data)
            
            # Update camera lookat to track the robot's COM in the viewer
            com = np.mean(self.mj_model.get_node_position_matrix(), axis=0)
            self.viewer.cam.lookat[:] = com
            self.viewer.sync()
            return None
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def _compute_reward(self, action):
        # User Set Hyperparameters  
        forward_weight = 5.0   # Heavily incentivize forward movement
        energy_weight = 0.005  # Decrease energy penalty
        alive_bonus = 0.1
        rigidity_weight = 0.5
        slip_weight = 0.1     # Decrease slip penalty so it isn't afraid to move


        terminate = False
        critical_eig = self.mj_model.collapse_check()
        if critical_eig < 0.03:
            terminate = True

        foward_vel = self.mj_model.get_forward_velocity_x()
        energy_penalty = float(np.sum(np.square(action)))
        critical_eig = float(critical_eig)

        # To incentiveze rolling I am going to do a slip penalty
        slip_penalty = float(self.mj_model.get_slip_penalty())
        
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
