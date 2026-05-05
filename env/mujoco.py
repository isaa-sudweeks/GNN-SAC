import numpy as np 
import gymnasium as gym 
from env.wrappers.timeout import Timeout


import env.truss.velocity_command_env
import env.truss.velocity_command_env_left
import env.truss.velocity_command_env_up
import env.truss.velocity_command_env_down

# How to add my own custom task
# Step 1: Add the class task definition to the tasks folder 
# Step 2: Register the environment 
# ```python 
# import gymnasium as gym 
# gym.register(
# id = "MyCustomTask-v0", entry_point="my_custom_env:MyCustomEnv", max_episode_steps=1000
# )
# ```
# Step 3: Add the task to the MUJOCO_TASKS dictionary 

MUJOCO_TASKS = {
    'mujoco-walker': 'Walker2d-v4',
    'mujoco-halfcheetah': 'HalfCheetah-v4',
    'bipedal-walker': 'BipedalWalker-v3',
    'lunarlander-continuous' : 'LunarLander-v2',
    'truss-velocity-command-right': 'MujocoVelocityCommandEnvRight-v0',
    'truss-velocity-command-left': 'MujocoVelocityCommandEnvLeft-v0',
    'truss-velocity-command-up': 'MujocoVelocityCommandEnvUp-v0',
    'truss-velocity-command-down': 'MujocoVelocityCommandEnvDown-v0',
    'octahedron-graph-right': 'MujocoOctahedronGraphEnvRight-v0',
    'octehedron-graph-right': 'MujocoOctahedronGraphEnvRight-v0',
    
}

CUSTOM_MUJOCO_TASKS = {
    'truss-velocity-command-right',
    'truss-velocity-command-left',
    'truss-velocity-command-up',
    'truss-velocity-command-down',
    'octahedron-graph-right',
    'octehedron-graph-right',
}

GRAPH_MUJOCO_TASKS = {
    'octahedron-graph-right',
    'octehedron-graph-right',
}

class MuJoCoWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg 
        self.env = env 
        self._cumulative_reward = 0


    def reset(self):
        self._cumulative_reward = 0
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        self._cumulative_reward += reward 
        done = terminated or truncated 
        info['terminated'] = terminated 
        info['truncated'] = truncated
        if self.cfg.task == 'lunarlander-continuous':
            info['success'] = self._cumulative_reward > 200
        return obs, reward, done, info 

    @property 
    def unwrapped(self):
        return self.env.unwrapped 

    def render(self, **kwargs):
        return self.env.render(**kwargs)

def make_env(cfg):
    """
        Make MuJoCo environment
    """

    if not cfg.task in MUJOCO_TASKS:
        raise ValueError(f'Task {cfg.task} not found in MuJoCo tasks')
    assert cfg.obs == 'state', 'MuJoCo envs only support state observations' 
    render_mode = 'rgb_array' if cfg.save_video else None
    env_kwargs = {'render_mode': render_mode}
    if cfg.task in GRAPH_MUJOCO_TASKS:
        import env.mujoco_gen.octehedron_graph_env_right
    if cfg.task in CUSTOM_MUJOCO_TASKS:
        env_kwargs['config'] = cfg
    if cfg. task == 'lunarlander-continuous':
        env = gym.make(MUJOCO_TASKS[cfg.task], continuous=True, **env_kwargs)
    else:
        env = gym.make(MUJOCO_TASKS[cfg.task], **env_kwargs)
    env = MuJoCoWrapper(env, cfg)
    env = Timeout(env, max_episode_steps={
        'lunarlander-continuous': 500,
        'bipedal-walker': 1600,
        'truss-velocity-command-right': cfg.max_steps,
        'truss-velocity-command-left': cfg.max_steps,
        'truss-velocity-command-up': cfg.max_steps,
        'truss-velocity-command-down': cfg.max_steps,
        'octahedron-graph-right': cfg.max_steps,
        'octehedron-graph-right': cfg.max_steps,
    }.get(cfg.task, 1000))
    cfg.discount_max = 0.99 
    cfg.rho = 0.7 # Increase this for tasks that are episodic #TODO
    return env 
