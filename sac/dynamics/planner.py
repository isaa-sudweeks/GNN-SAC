"""Bounded smooth-action CEM, independent of the rollout implementation."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CEMConfig:
    """All proposal settings; horizon and knot spacing are control steps."""

    horizon: int = 10
    candidates: int = 256
    iterations: int = 3
    knots: int = 3
    elite_fraction: float = 0.1
    initial_std: float = 0.5
    minimum_std: float = 0.05
    update_fraction: float = 1.0

    def __post_init__(self):
        for name in ("horizon", "candidates", "iterations", "knots"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.knots > self.horizon:
            raise ValueError("knots cannot exceed horizon")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0 < self.update_fraction <= 1:
            raise ValueError("update_fraction must be in (0, 1]")
        if not 0 < self.minimum_std <= self.initial_std or not np.isfinite(self.initial_std):
            raise ValueError("standard deviations must be finite and 0 < minimum <= initial")


@dataclass
class PlanResult:
    actions: np.ndarray
    score: float
    iteration_best: list[float]
    sequences_scored: int


def interpolate_knots(knots: np.ndarray, horizon: int) -> np.ndarray:
    """Linearly interpolate (..., knots, actions) without mixing action axes."""
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim < 2 or knots.shape[-2] < 1 or horizon < 1:
        raise ValueError("Expected (..., positive knots, actions) and positive horizon")
    if knots.shape[-2] == 1:
        return np.repeat(knots, horizon, axis=-2)
    location = np.linspace(0, knots.shape[-2] - 1, horizon)
    left = np.floor(location).astype(int)
    right = np.minimum(left + 1, knots.shape[-2] - 1)
    weight = (location - left)[:, None]
    return knots[..., left, :] * (1 - weight) + knots[..., right, :] * weight


def cem_plan(
    evaluate: Callable[[np.ndarray], np.ndarray],
    action_size: int,
    config: CEMConfig,
    rng: np.random.Generator,
    *,
    action_mask: np.ndarray | None = None,
    previous_plan: np.ndarray | None = None,
) -> PlanResult:
    """Score bounded sequences and return the best actually evaluated sequence.

    Warm starts shift the previous sequence one step and repeat its last action.
    The evaluator handles collapse, objective, and simulator/model transitions.
    Nonfinite scores are rejected; all-invalid proposals raise explicitly.
    """
    if action_size < 1:
        raise ValueError("action_size must be positive")
    mask = np.ones(action_size) if action_mask is None else np.asarray(action_mask, dtype=float)
    if mask.shape != (action_size,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("action_mask must be a binary vector of action_size")
    mean = np.zeros((config.knots, action_size))
    if previous_plan is not None:
        previous = np.asarray(previous_plan, dtype=float)
        if previous.shape != (config.horizon, action_size) or not np.isfinite(previous).all():
            raise ValueError("previous_plan must be finite with shape (horizon, action_size)")
        shifted = np.concatenate([previous[1:], previous[-1:]], axis=0)
        indices = np.linspace(0, config.horizon - 1, config.knots)
        mean = np.stack([np.interp(indices, np.arange(config.horizon), shifted[:, i])
                         for i in range(action_size)], axis=-1)
    mean = np.clip(mean, -1, 1) * mask
    std = np.full_like(mean, config.initial_std)
    elite_count = max(1, int(np.ceil(config.candidates * config.elite_fraction)))
    best_score = -np.inf
    best_actions = None
    history = []
    for _ in range(config.iterations):
        proposal = np.clip(rng.normal(mean, std, (config.candidates,) + mean.shape), -1, 1) * mask
        # Include the mean so refinement cannot lose a useful warm-start proposal.
        proposal[0] = mean
        sequences = interpolate_knots(proposal, config.horizon)
        scores = np.asarray(evaluate(sequences), dtype=float)
        if scores.shape != (config.candidates,):
            raise ValueError("Evaluator must return one score per candidate")
        finite = np.flatnonzero(np.isfinite(scores))
        if not finite.size:
            raise RuntimeError("Every candidate received a nonfinite score")
        ordered = finite[np.argsort(scores[finite])]
        index = ordered[-1]
        if scores[index] > best_score:
            best_score = float(scores[index])
            best_actions = sequences[index].copy()
        history.append(best_score)
        elite = proposal[ordered[-elite_count:]]
        alpha = config.update_fraction
        mean = (1 - alpha) * mean + alpha * elite.mean(axis=0)
        std = np.maximum(config.minimum_std, (1 - alpha) * std + alpha * elite.std(axis=0))
    return PlanResult(best_actions, best_score, history, config.candidates * config.iterations)


def planning_score(
    forward_displacement: float,
    initial_diagonal: float,
    squared_command_integral: float,
    *,
    effort_weight: float,
    collapsed: bool,
    remaining_seconds: float,
    failure_cost: float,
    failure_cost_per_second: float,
) -> float:
    """Progress minus time-integrated routed-command effort and failure cost.

    Squared command effort is averaged over external actuators before integration.
    A failed branch pays a fixed cost plus a remaining-time cost, so skipping
    future steps does not erase its failure penalty. Weights need source tuning.
    """
    values = np.asarray([forward_displacement, initial_diagonal, squared_command_integral,
                         effort_weight, remaining_seconds, failure_cost, failure_cost_per_second])
    if not np.isfinite(values).all():
        return -np.inf
    if initial_diagonal <= 0 or np.any(values[2:] < 0):
        raise ValueError("Diagonal must be positive and costs/time nonnegative")
    failure = failure_cost + failure_cost_per_second * remaining_seconds if collapsed else 0.0
    return float(forward_displacement / initial_diagonal
                 - effort_weight * squared_command_integral - failure)
