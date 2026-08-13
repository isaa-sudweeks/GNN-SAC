from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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
        self._checkpoint_executor = None
        self._checkpoint_future = None
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
        def _cpu_byte_tensor(value):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value.detach().cpu().to(torch.uint8)

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(_cpu_byte_tensor(state["torch"]))
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([_cpu_byte_tensor(cuda_state) for cuda_state in state["cuda"]])

    def _config_state_dict(self):
        if OmegaConf is not None and OmegaConf.is_config(self.cfg):
            return OmegaConf.to_container(self.cfg, resolve=True)
        try:
            return dict(vars(self.cfg))
        except TypeError:
            return {}

    def checkpoint_state_dict(self):
        state = {
            "format_version": 3,
            "trainer": {
                "step": getattr(self, "_step", 0),
                "episode": getattr(self, "_ep_idx", 0),
                "best_eval_metrics": self._best_eval_metrics,
                "update_budget": float(getattr(self, "_update_budget", 0.0)),
                "pending_update_transitions": int(getattr(self, "_pending_update_transitions", 0)),
                "vector_steps_since_update": int(getattr(self, "_vector_steps_since_update", 0)),
                "pretrain_complete": bool(getattr(self, "_pretrain_complete", False)),
                "optimizer_updates": int(getattr(self, "_optimizer_updates", 0)),
                "last_eval_step": getattr(self, "_last_eval_step", None),
            },
            "agent": self.agent.training_state_dict(),
            "buffer": self.buffer.state_dict(),
            "logger": self.logger.state_dict(),
            "rng": self._rng_state_dict(),
            "config": self._config_state_dict(),
        }
        reward_normalizer = getattr(self, "reward_normalizer", None)
        if reward_normalizer is not None:
            state["reward_normalizer"] = reward_normalizer.state_dict()
        return state

    def _snapshot_checkpoint_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {key: self._snapshot_checkpoint_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._snapshot_checkpoint_value(item) for item in value)
        if isinstance(value, list):
            return [self._snapshot_checkpoint_value(item) for item in value]
        return deepcopy(value)

    def _snapshot_buffer_state_dict(self, state_dict):
        """Copy replay metadata while avoiding a full clone of immutable stored samples."""
        def snapshot_value(value):
            if isinstance(value, Mapping):
                return {key: snapshot_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return list(value)
            return self._snapshot_checkpoint_value(value)

        return snapshot_value(state_dict)

    def _checkpoint_state_snapshot(self):
        state_dict = self.checkpoint_state_dict()
        snapshot = {
            "format_version": state_dict["format_version"],
            "trainer": self._snapshot_checkpoint_value(state_dict["trainer"]),
            "agent": self._snapshot_checkpoint_value(state_dict["agent"]),
            "buffer": self._snapshot_buffer_state_dict(state_dict["buffer"]),
            "logger": self._snapshot_checkpoint_value(state_dict["logger"]),
            "rng": self._snapshot_checkpoint_value(state_dict["rng"]),
            "config": self._snapshot_checkpoint_value(state_dict["config"]),
        }
        if "reward_normalizer" in state_dict:
            snapshot["reward_normalizer"] = self._snapshot_checkpoint_value(
                state_dict["reward_normalizer"]
            )
        return snapshot

    def load_checkpoint_state_dict(self, state_dict):
        trainer_state = state_dict.get("trainer", {})
        self._step = int(trainer_state.get("step", 0))
        self._ep_idx = int(trainer_state.get("episode", 0))
        self._best_eval_metrics = trainer_state.get("best_eval_metrics")
        self.agent.load_training_state_dict(state_dict["agent"])
        self.buffer.load_state_dict(state_dict["buffer"])
        self._update_budget = float(trainer_state.get("update_budget", 0.0))
        self._pending_update_transitions = int(trainer_state.get("pending_update_transitions", 0))
        self._vector_steps_since_update = int(trainer_state.get("vector_steps_since_update", 0))
        pretrain_complete = trainer_state.get("pretrain_complete")
        if pretrain_complete is None:
            # Older checkpoints did not record this flag. If replay already
            # contains usable data after the seed phase, updates have started.
            buffer_size = int(getattr(self.buffer, "size", 0))
            batch_size = int(getattr(self.cfg, "batch_size", 1))
            seed_steps = int(getattr(self.cfg, "seed_steps", 0))
            pretrain_complete = self._step > seed_steps and buffer_size >= batch_size
        self._pretrain_complete = bool(pretrain_complete)
        self._optimizer_updates = int(trainer_state.get("optimizer_updates", 0))
        last_eval_step = trainer_state.get("last_eval_step")
        self._last_eval_step = None if last_eval_step is None else int(last_eval_step)
        if "logger" in state_dict:
            self.logger.load_state_dict(state_dict["logger"])
        saved_reward_normalizer = state_dict.get("reward_normalizer")
        reward_normalizer = getattr(self, "reward_normalizer", None)
        if saved_reward_normalizer is not None and reward_normalizer is None:
            raise ValueError(
                "Checkpoint contains reward-normalization state, but normalize_rewards is disabled."
            )
        if saved_reward_normalizer is not None:
            reward_normalizer.load_state_dict(saved_reward_normalizer)
        self._load_rng_state_dict(state_dict.get("rng"))

    @staticmethod
    def _agent_checkpoint_state_dict(state_dict):
        agent_state = state_dict["agent"]
        if isinstance(agent_state, Mapping) and "model" in agent_state:
            checkpoint = {"model": agent_state["model"]}
            for key in ("log_alpha", "graph_feature_schema"):
                if key in agent_state:
                    checkpoint[key] = agent_state[key]
            return checkpoint
        return agent_state

    @classmethod
    def _write_checkpoint_files(cls, checkpoint_dir, identifier, state_dict, keep_last, write_agent):
        path = checkpoint_dir / f"{identifier}.pt"
        tmp_path = checkpoint_dir / f".{identifier}.tmp"
        torch.save(state_dict, tmp_path)
        tmp_path.replace(path)

        latest_tmp = checkpoint_dir / ".latest.tmp"
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(state_dict, latest_tmp)
        latest_tmp.replace(latest_path)

        if write_agent:
            agent_state = cls._agent_checkpoint_state_dict(state_dict)
            agent_tmp = checkpoint_dir / f".{identifier}.agent.tmp"
            agent_path = checkpoint_dir / f"{identifier}.agent.pt"
            torch.save(agent_state, agent_tmp)
            agent_tmp.replace(agent_path)

            latest_agent_tmp = checkpoint_dir / ".latest.agent.tmp"
            latest_agent_path = checkpoint_dir / "latest.agent.pt"
            torch.save(agent_state, latest_agent_tmp)
            latest_agent_tmp.replace(latest_agent_path)

        cls._prune_old_checkpoints_in_dir(checkpoint_dir, keep_last)
        return path

    @staticmethod
    def _prune_old_checkpoints_in_dir(checkpoint_dir, keep_last):
        keep_last = int(keep_last or 0)
        if keep_last <= 0:
            return
        checkpoints = sorted(
            (
                path
                for path in checkpoint_dir.glob("step_*.pt")
                if not path.name.endswith(".agent.pt")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_checkpoint in checkpoints[keep_last:]:
            old_checkpoint.unlink(missing_ok=True)
            old_checkpoint.with_name(f"{old_checkpoint.stem}.agent.pt").unlink(missing_ok=True)

    def wait_for_async_checkpoint(self):
        if self._checkpoint_future is None:
            return None
        try:
            return self._checkpoint_future.result()
        finally:
            self._checkpoint_future = None
            if self._checkpoint_executor is not None:
                self._checkpoint_executor.shutdown(wait=True)
                self._checkpoint_executor = None

    def save_checkpoint(self, identifier=None, *, async_write=False):
        if self._checkpoint_future is not None:
            self.wait_for_async_checkpoint()
        checkpoint_dir = self.checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step = getattr(self, "_step", 0)
        identifier = f"step_{step}" if identifier is None else str(identifier)
        state_dict = self._checkpoint_state_snapshot() if async_write else self.checkpoint_state_dict()
        keep_last = int(getattr(self.cfg, "checkpoint_keep_last", 0) or 0)
        write_agent = hasattr(self.agent, "save")
        if async_write:
            self._checkpoint_executor = ThreadPoolExecutor(max_workers=1)
            self._checkpoint_future = self._checkpoint_executor.submit(
                self._write_checkpoint_files,
                checkpoint_dir,
                identifier,
                state_dict,
                keep_last,
                write_agent,
            )
            return checkpoint_dir / f"{identifier}.pt"
        path = self._write_checkpoint_files(checkpoint_dir, identifier, state_dict, keep_last, write_agent)
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
        async_write = bool(getattr(self.cfg, "checkpoint_async", False)) and not force
        path = self.save_checkpoint(async_write=async_write)
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
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.load_checkpoint_state_dict(state_dict)
        print(f"Resumed checkpoint: {checkpoint_path} at step {self._step}")
        return checkpoint_path

    def _prune_old_checkpoints(self):
        keep_last = int(getattr(self.cfg, "checkpoint_keep_last", 0) or 0)
        if keep_last <= 0:
            return
        checkpoint_dir = self.checkpoint_dir()
        checkpoints = sorted(
            (
                path
                for path in checkpoint_dir.glob("step_*.pt")
                if not path.name.endswith(".agent.pt")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_checkpoint in checkpoints[keep_last:]:
            old_checkpoint.unlink(missing_ok=True)
            old_checkpoint.with_name(f"{old_checkpoint.stem}.agent.pt").unlink(missing_ok=True)

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

        
