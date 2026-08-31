from dataclasses import dataclass
import math
from typing import Hashable, Iterable

import torch


@dataclass
class RunningMeanVariance:
    """Numerically stable scalar running moments."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningMeanVariance":
        return cls(
            count=int(state["count"]),
            mean=float(state["mean"]),
            m2=float(state["m2"]),
        )


class TaskRewardNormalizer:
    """Normalize rewards using independent discounted-return variance per task."""

    def __init__(
        self,
        gamma: float,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        allowed_tasks: Iterable[str] | None = None,
    ):
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"reward_norm_gamma must be in [0, 1], got {self.gamma}.")
        if self.epsilon <= 0.0:
            raise ValueError(f"reward_norm_epsilon must be positive, got {self.epsilon}.")
        if self.clip <= 0.0:
            raise ValueError(f"reward_norm_clip must be positive, got {self.clip}.")

        self.allowed_tasks = None if allowed_tasks is None else tuple(str(task) for task in allowed_tasks)
        self._stats: dict[str, RunningMeanVariance] = {}
        self._returns: dict[Hashable, float] = {}
        self._stream_tasks: dict[Hashable, str] = {}
        self._last_rewards: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _scalar(reward) -> float:
        if isinstance(reward, torch.Tensor):
            if reward.numel() != 1:
                raise ValueError(f"Reward normalization requires a scalar reward, got shape {tuple(reward.shape)}.")
            value = float(reward.detach().cpu().item())
        else:
            value = float(reward)
        if not math.isfinite(value):
            raise ValueError(f"Reward normalization requires a finite reward, got {value}.")
        return value

    def normalize(self, reward, *, task: str, stream: Hashable, done: bool = False):
        task = str(task)
        if self.allowed_tasks is not None and task not in self.allowed_tasks:
            raise KeyError(f"Unknown reward-normalization task {task!r}; expected one of {list(self.allowed_tasks)!r}.")
        active_task = self._stream_tasks.get(stream)
        if active_task is not None and active_task != task:
            raise ValueError(
                f"Reward-normalization stream {stream!r} changed task from {active_task!r} "
                f"to {task!r} before the episode ended."
            )

        raw_value = self._scalar(reward)
        discounted_return = self.gamma * self._returns.get(stream, 0.0) + raw_value
        stats = self._stats.setdefault(task, RunningMeanVariance())
        stats.update(discounted_return)
        scale = math.sqrt(stats.variance + self.epsilon)
        normalized_value = max(-self.clip, min(self.clip, raw_value / scale))
        self._last_rewards[task] = (raw_value, normalized_value)

        if done:
            self._returns.pop(stream, None)
            self._stream_tasks.pop(stream, None)
        else:
            self._returns[stream] = discounted_return
            self._stream_tasks[stream] = task

        if isinstance(reward, torch.Tensor):
            return torch.as_tensor(normalized_value, dtype=reward.dtype, device=reward.device)
        return normalized_value

    def reset_stream(self, stream: Hashable) -> None:
        """Forget an unfinished return when its environment starts a new episode."""
        self._returns.pop(stream, None)
        self._stream_tasks.pop(stream, None)

    def metrics(self) -> dict[str, dict[str, float]]:
        return {
            task: {
                "return_mean": stats.mean,
                "return_std": stats.std,
                "count": stats.count,
                "raw_reward": self._last_rewards.get(task, (float("nan"), float("nan")))[0],
                "normalized_reward": self._last_rewards.get(task, (float("nan"), float("nan")))[1],
            }
            for task, stats in self._stats.items()
        }

    def state_dict(self) -> dict:
        return {
            "format_version": 1,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "allowed_tasks": self.allowed_tasks,
            "stats": {task: stats.state_dict() for task, stats in self._stats.items()},
            "returns": dict(self._returns),
            "stream_tasks": dict(self._stream_tasks),
            "last_rewards": dict(self._last_rewards),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_settings = (float(state["gamma"]), float(state["epsilon"]), float(state["clip"]))
        current_settings = (self.gamma, self.epsilon, self.clip)
        if saved_settings != current_settings:
            raise ValueError(
                f"Checkpoint reward-normalization settings {saved_settings!r} do not match "
                f"configured settings {current_settings!r}."
            )
        saved_tasks = state.get("allowed_tasks")
        if saved_tasks is not None and tuple(saved_tasks) != self.allowed_tasks:
            raise ValueError(
                f"Checkpoint reward-normalization tasks {tuple(saved_tasks)!r} do not match "
                f"configured tasks {self.allowed_tasks!r}."
            )
        self._stats = {
            str(task): RunningMeanVariance.from_state_dict(stats)
            for task, stats in state.get("stats", {}).items()
        }
        self._returns = dict(state.get("returns", {}))
        self._stream_tasks = {
            stream: str(task) for stream, task in state.get("stream_tasks", {}).items()
        }
        self._last_rewards = {
            str(task): (float(values[0]), float(values[1]))
            for task, values in state.get("last_rewards", {}).items()
        }
