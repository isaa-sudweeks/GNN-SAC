"""Success Guided Sampling over training tasks (arXiv:2610.12465, §3.2/B.2)."""

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np


def _get(cfg, key, default):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def beta_probabilities(success_rates: Sequence[float], *, target: float = 0.66,
                       concentration: float = 5.0, temperature: float = 2.0,
                       epsilon: float = 1e-8) -> np.ndarray:
    """Paper beta kernel, weight floor, log epsilon, and temperature clamp.

    Beta(1 + k*t, 1 + k*(1-t)); its normalization cancels in the softmax.
    Compute in log space to avoid underflow for concentrated kernels.
    """
    p = np.asarray(success_rates, dtype=np.float64)
    params = np.asarray([target, concentration, temperature, epsilon])
    if not np.isfinite(params).all() or not 0 <= target <= 1 or concentration < 0 or temperature <= 0 or not 0 < epsilon <= 1:
        raise ValueError("SGS needs target in [0,1], concentration >= 0, temperature > 0, epsilon in (0,1].")
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("SGS success rates must be a nonempty finite vector in [0,1].")
    log_weight = concentration * (target * np.log(p + epsilon) + (1 - target) * np.log(1 - p + epsilon))
    logits = np.logaddexp(np.maximum(log_weight, np.log(epsilon)), np.log(epsilon)) / max(temperature, 1.0)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


class SuccessGuidedSampler:
    """Shared online success history with independent collection/replay RNGs."""

    def __init__(self, cfg, tasks: Sequence[str], *, seed: int = 0):
        self.mode = str(_get(cfg, "mode", "disabled"))
        if self.mode not in {"disabled", "collection", "replay", "both"}:
            raise ValueError("sgs.mode must be disabled, collection, replay, or both.")
        self.tasks = tuple(tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("SGS requires unique training tasks.")
        self.target = float(_get(cfg, "target", 0.66))
        self.concentration = float(_get(cfg, "concentration", 5.0))
        self.temperature = float(_get(cfg, "temperature", 2.0))
        self.epsilon = float(_get(cfg, "epsilon", 1e-8))
        beta_probabilities([0.5], target=self.target, concentration=self.concentration,
                           temperature=self.temperature, epsilon=self.epsilon)
        self.history_size = int(_get(cfg, "history_size", 100))
        self.min_episodes = int(_get(cfg, "min_episodes", 10))
        self.warmup_steps = int(_get(cfg, "warmup_steps", 5000))
        self.collection_block_steps = int(_get(cfg, "collection_block_steps", 16))
        self.uniform_mix = float(_get(cfg, "uniform_mix", 0.0))
        self.require_survival = bool(_get(cfg, "require_survival", False))
        self.success_key = str(_get(cfg, "success_key", "success"))
        self.success_metric = str(_get(cfg, "success_metric", "info"))
        survival_steps = _get(cfg, "min_survival_steps", 0)
        if isinstance(survival_steps, bool) or not np.isfinite(float(survival_steps)) or float(survival_steps) < 0 or not float(survival_steps).is_integer():
            raise ValueError("sgs.min_survival_steps must be a nonnegative integer.")
        self.min_survival_steps = int(survival_steps)
        self.distance_threshold = float(_get(cfg, "distance_threshold", 15.0))
        if self.success_metric not in {"info", "distance"} or not np.isfinite(self.distance_threshold) or self.distance_threshold < 0:
            raise ValueError("SGS needs success_metric=info|distance and a finite nonnegative distance_threshold.")
        if not 1 <= self.min_episodes <= self.history_size or self.warmup_steps < 0 or self.collection_block_steps < 1:
            raise ValueError("SGS requires 1 <= min_episodes <= history_size, warmup_steps >= 0, collection_block_steps >= 1.")
        if not np.isfinite(self.uniform_mix) or not 0 <= self.uniform_mix <= 1:
            raise ValueError("sgs.uniform_mix must be in [0,1].")
        self.histories = {task: deque(maxlen=self.history_size) for task in self.tasks}
        self.episodes = dict.fromkeys(self.tasks, 0)
        self.collected = dict.fromkeys(self.tasks, 0)
        self.replayed = dict.fromkeys(self.tasks, 0)
        self.step = 0
        self._collection_rng = np.random.default_rng(seed)
        self._replay_rng = np.random.default_rng(seed + 1)
        self._selected = None
        self._block_remaining = 0

    @property
    def adaptive_collection(self) -> bool:
        return self.mode in {"collection", "both"}

    @property
    def adaptive_replay(self) -> bool:
        return self.mode in {"replay", "both"}

    @property
    def warmed_up(self) -> bool:
        return self.step >= self.warmup_steps and all(len(h) >= self.min_episodes for h in self.histories.values())

    def probabilities(self) -> np.ndarray:
        if not self.warmed_up:
            return np.full(len(self.tasks), 1 / len(self.tasks))
        p = beta_probabilities([np.mean(self.histories[task]) for task in self.tasks],
                               target=self.target, concentration=self.concentration,
                               temperature=self.temperature, epsilon=self.epsilon)
        return (1 - self.uniform_mix) * p + self.uniform_mix / len(self.tasks)

    def record_episode(self, task: str, info: Mapping) -> None:
        """Record exactly once on completion; never use evaluation outcomes."""
        if task not in self.histories:
            raise ValueError(f"SGS received non-training task {task!r}.")
        if self.success_metric == "distance":
            if "sgs_distance" not in info or not np.isfinite(float(info["sgs_distance"])):
                raise ValueError("SGS distance success needs finite accumulated com_delta_x.")
            success = float(float(info["sgs_distance"]) >= self.distance_threshold)
        else:
            if self.success_key not in info:
                raise ValueError(f"SGS episode info lacks {self.success_key!r}.")
            success = float(info[self.success_key])
            if success not in (0.0, 1.0):
                raise ValueError("SGS requires Boolean episode success outcomes.")
        if self.min_survival_steps:
            if "sgs_episode_length" not in info or "terminated" not in info:
                raise ValueError("SGS survival steps need episode length and terminated in episode info.")
            length = float(info["sgs_episode_length"])
            if not np.isfinite(length) or length < 0 or not length.is_integer():
                raise ValueError("SGS episode length must be a finite nonnegative integer.")
            # A failure transition is not a survived step. Later failure does
            # not erase an already achieved minimum survival duration.
            success *= length - int(bool(info["terminated"])) >= self.min_survival_steps
        if self.require_survival:
            if "terminated" not in info:
                raise ValueError("SGS require_survival needs terminated in episode info.")
            success *= not bool(info["terminated"])
        self.histories[task].append(bool(success))
        self.episodes[task] += 1

    def record_collection(self, task: str, count: int) -> None:
        self.collected[task] += count
        self.step += count

    def collection_group(self, tasks: Sequence[str], sizes: Sequence[int]) -> int | None:
        """Select a fixed-shape batch; None means collect all during warm-up.

        Probabilities describe transitions, so divide by batch size when
        selecting batches. Paused simulator/episode state remains untouched.
        """
        if not self.adaptive_collection or not self.warmed_up:
            return None
        if set(tasks) != set(self.tasks) or len(tasks) != len(self.tasks):
            raise ValueError("SGS collection needs one homogeneous group per training task.")
        if self._block_remaining == 0:
            probabilities = dict(zip(self.tasks, self.probabilities()))
            weights = np.asarray([probabilities[task] / size for task, size in zip(tasks, sizes)])
            self._selected = int(self._collection_rng.choice(len(tasks), p=weights / weights.sum()))
            self._block_remaining = self.collection_block_steps
        self._block_remaining -= 1
        return self._selected

    def replay_counts(self, batch_size: int, balanced_count: int) -> dict[str, int]:
        if self.adaptive_replay and self.warmed_up:
            counts = self._replay_rng.multinomial(batch_size, self.probabilities())
        else:
            counts = np.full(len(self.tasks), balanced_count)
        result = {task: int(count) for task, count in zip(self.tasks, counts) if count}
        for task, count in result.items():
            self.replayed[task] += count
        return result

    def metrics(self) -> dict[str, float]:
        result = {"adaptive": float(self.warmed_up)}
        for task, probability in zip(self.tasks, self.probabilities()):
            result[f"probability/{task}"] = float(probability)
            result[f"success/{task}"] = float(np.mean(self.histories[task])) if self.histories[task] else 0.0
            for name in ("episodes", "collected", "replayed"):
                result[f"{name}/{task}"] = getattr(self, name)[task]
        return result

    def state_dict(self) -> dict:
        state = {key: deepcopy(value) for key, value in vars(self).items()
                 if key not in {"histories", "_collection_rng", "_replay_rng"}}
        state["histories"] = {task: list(h) for task, h in self.histories.items()}
        state["collection_rng"] = deepcopy(self._collection_rng.bit_generator.state)
        state["replay_rng"] = deepcopy(self._replay_rng.bit_generator.state)
        return state

    def load_state_dict(self, state: Mapping) -> None:
        settings = ("mode", "tasks", "target", "concentration", "temperature", "epsilon", "history_size",
                    "min_episodes", "warmup_steps", "collection_block_steps", "uniform_mix", "require_survival", "success_key", "success_metric", "distance_threshold", "min_survival_steps")
        for key in settings:
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume requires matching SGS {key}.")
        self.histories = {task: deque(state["histories"][task], maxlen=self.history_size) for task in self.tasks}
        for key in ("episodes", "collected", "replayed", "step", "_selected", "_block_remaining"):
            setattr(self, key, deepcopy(state[key]))
        self._collection_rng.bit_generator.state = deepcopy(state["collection_rng"])
        self._replay_rng.bit_generator.state = deepcopy(state["replay_rng"])
