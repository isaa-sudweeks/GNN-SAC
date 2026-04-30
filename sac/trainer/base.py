from copy import deepcopy
import optuna


class Trainer:
    """
    Base trainer class for SAC.
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

    def report_eval_metrics(self, metrics, step):
        if self.trial is not None:
            self.trial.report(float(metrics["episode_reward"]), step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        if (
            self._best_eval_metrics is None
            or metrics["episode_reward"] > self._best_eval_metrics["episode_reward"]
        ):
            self._best_eval_metrics = deepcopy(metrics)

    def best_objective(self):
        if self._best_eval_metrics is None:
            return float("-inf"), "episode_reward"
        return float(self._best_eval_metrics["episode_reward"]), "episode_reward"

    def eval(self):
        """
        Evaluate a SAC agent.
        """
        raise NotImplementedError

    def train(self):
        """
        Train a SAC agent.
        """
        raise NotImplementedError

        
