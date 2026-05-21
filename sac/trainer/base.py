from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import optuna
import torch

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - only used in minimal tooling envs.
    OmegaConf = None


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

    def checkpoint_dir(self):
        checkpoint_dir = getattr(self.cfg, "checkpoint_dir", None)
        if checkpoint_dir in (None, "", "???"):
            return Path(self.cfg.work_dir) / "checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = Path(self.cfg.work_dir) / checkpoint_dir
        return checkpoint_dir

    def resolve_checkpoint_path(self, checkpoint_ref):
        if checkpoint_ref in (None, "", "???", False):
            return None
        if str(checkpoint_ref).lower() == "latest":
            return self.checkpoint_dir() / "latest.pt"
        return Path(checkpoint_ref).expanduser()

    def _rng_state_dict(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_rng_state_dict(self, state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _config_state_dict(self):
        if OmegaConf is not None and OmegaConf.is_config(self.cfg):
            return OmegaConf.to_container(self.cfg, resolve=True)
        try:
            return dict(vars(self.cfg))
        except TypeError:
            return {}

    def checkpoint_state_dict(self):
        return {
            "format_version": 1,
            "trainer": {
                "step": getattr(self, "_step", 0),
                "episode": getattr(self, "_ep_idx", 0),
                "best_eval_metrics": self._best_eval_metrics,
            },
            "agent": self.agent.training_state_dict(),
            "buffer": self.buffer.state_dict(),
            "logger": self.logger.state_dict(),
            "rng": self._rng_state_dict(),
            "config": self._config_state_dict(),
        }

    def load_checkpoint_state_dict(self, state_dict):
        trainer_state = state_dict.get("trainer", {})
        self._step = int(trainer_state.get("step", 0))
        self._ep_idx = int(trainer_state.get("episode", 0))
        self._best_eval_metrics = trainer_state.get("best_eval_metrics")
        self.agent.load_training_state_dict(state_dict["agent"])
        self.buffer.load_state_dict(state_dict["buffer"])
        if "logger" in state_dict:
            self.logger.load_state_dict(state_dict["logger"])
        self._load_rng_state_dict(state_dict.get("rng"))

    def save_checkpoint(self, identifier=None):
        checkpoint_dir = self.checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step = getattr(self, "_step", 0)
        identifier = f"step_{step}" if identifier is None else str(identifier)
        path = checkpoint_dir / f"{identifier}.pt"
        tmp_path = checkpoint_dir / f".{identifier}.tmp"
        state_dict = self.checkpoint_state_dict()
        torch.save(state_dict, tmp_path)
        tmp_path.replace(path)

        latest_tmp = checkpoint_dir / ".latest.tmp"
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(state_dict, latest_tmp)
        latest_tmp.replace(latest_path)
        self._prune_old_checkpoints()
        return path

    def maybe_save_checkpoint(self, previous_step=None, force=False):
        checkpoint_freq = int(getattr(self.cfg, "checkpoint_freq", 0) or 0)
        if checkpoint_freq <= 0:
            return None
        current_step = int(getattr(self, "_step", 0))
        if not force:
            previous_step = current_step - 1 if previous_step is None else int(previous_step)
            crossed = previous_step // checkpoint_freq < current_step // checkpoint_freq
            if not crossed:
                return None
        path = self.save_checkpoint()
        print(f"Saved checkpoint: {path}")
        return path

    def maybe_load_checkpoint(self):
        checkpoint_path = self.resolve_checkpoint_path(getattr(self.cfg, "resume_from_checkpoint", None))
        if checkpoint_path is None:
            return None
        if not checkpoint_path.exists():
            if str(getattr(self.cfg, "resume_from_checkpoint", "")).lower() == "latest":
                print(f"No checkpoint found at {checkpoint_path}; starting a fresh run.")
                return None
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=getattr(self.agent, "device", "cpu"), weights_only=False)
        self.load_checkpoint_state_dict(state_dict)
        print(f"Resumed checkpoint: {checkpoint_path} at step {self._step}")
        return checkpoint_path

    def _prune_old_checkpoints(self):
        keep_last = int(getattr(self.cfg, "checkpoint_keep_last", 0) or 0)
        if keep_last <= 0:
            return
        checkpoint_dir = self.checkpoint_dir()
        checkpoints = sorted(
            checkpoint_dir.glob("step_*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_checkpoint in checkpoints[keep_last:]:
            old_checkpoint.unlink(missing_ok=True)

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

        
