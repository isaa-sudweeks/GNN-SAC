import numpy as np

class MultitaskWrapper:
    def __init__(self, cfg, make_env_fns):
        self.cfg = cfg
        self.envs = []
        original_task = cfg.task
        
        for task in cfg.tasks:
            cfg.task = task
            env = None 
            errors = []
            for fn in make_env_fns:
                try:
                    env = fn(cfg)
                except ValueError as exc:
                    errors.append(str(exc))
                    pass 
            if env is None:
                details = '; '.join(errors)
                raise ValueError(f'Failed to make environment "{task}": {details}')
            self.envs.append(env)
        
        cfg.task = original_task
        
        self.active_env_idx = 0
        self.env = self.envs[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(len(self.envs))
        self.active_env_idx = task_idx
        self.env = self.envs[self.active_env_idx]
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    @property
    def unwrapped(self):
        return self.env.unwrapped
        
    def close(self):
        for e in self.envs:
            e.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)
