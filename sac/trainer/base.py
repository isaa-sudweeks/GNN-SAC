from copy import deepcopy
import optuna


class Trainer:
    """
    Base trainer class for TD-MPC2.
    """
    def __init__(self, cfg, env, agent, buffer, logger, trial=None):
        self.cfg = cfg 
        self.env = env 
        self.agent = agent
        self.buffer = buffer 
        self.logger = logger 
        self.trial = trial
        self._best_eval_metrics = None
        print('Architecture:' , self.agent.model)

    def eval(self):
        """
        Evaluate a TD-MPC2 agent.
        """
        raise NotImplementedError

    def train(self):
        """
        Train a TD-MPC2 agent.
        """
        raise NotImplementedError

        