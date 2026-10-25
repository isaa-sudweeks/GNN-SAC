"""Episode-safe replay strata and a sampling-only intervention for native updates."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def failure_band(arrays: dict[str, np.ndarray], distance: int) -> np.ndarray:
    """Mark transition endpoints zero through distance steps before true termination."""
    if distance < 0:
        raise ValueError('Failure distance must be nonnegative')
    mask = np.zeros(len(arrays['episode']), dtype=bool)
    for terminal in np.flatnonzero(arrays['terminated']):
        indices = np.arange(max(0, terminal - distance), terminal + 1)
        mask[indices[arrays['episode'][indices] == arrays['episode'][terminal]]] = True
    return mask


def safe_starts(arrays: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    starts = np.arange(max(0, len(arrays['episode']) - horizon + 1))
    valid = np.ones(len(starts), dtype=bool)
    for offset in range(horizon):
        valid &= arrays['episode'][starts] == arrays['episode'][starts + offset]
        if offset < horizon - 1:
            valid &= ~(arrays['terminated'][starts + offset] | arrays['truncated'][starts + offset])
    return starts[valid]


def restored_history(history: list[dict[str, Any]], checkpoint_round: int) -> list[dict[str, Any]]:
    """Discard progress written after the last durable restart checkpoint."""
    return [row for row in history if row['round'] <= checkpoint_round]


class ReplaySampler:
    """Replace only native update's two integers calls; retain exact uniform RNG behavior."""

    def __init__(self, arrays: dict[str, np.ndarray], horizon: int, arm: str,
                 rng_state: dict[str, Any], batch_size: int = 32,
                 distance: int = 10, fraction: float = .5) -> None:
        if arm not in ('uniform', 'balanced') or not 0 <= fraction <= 1:
            raise ValueError('Invalid sampling intervention')
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = deepcopy(rng_state)
        self.arm, self.batch_size, self.fraction = arm, batch_size, fraction
        self.phase = 0
        single = failure_band(arrays, distance)
        self.valid = safe_starts(arrays, horizon)
        if not len(self.valid):
            raise ValueError('No episode-safe recursive windows')
        self.risk = (single, single[self.valid + horizon - 1])
        self.highs = tuple(len(values) for values in self.risk)
        self.pools = tuple((np.flatnonzero(values), np.flatnonzero(~values)) for values in self.risk)
        self.counts = [dict(calls=0, draws=0, failure_draws=0, fallback_calls=0) for _ in range(2)]

    def integers(self, high: int, *, size: int) -> np.ndarray:
        phase = self.phase
        if int(high) != self.highs[phase] or size != self.batch_size:
            raise AssertionError('Native replay sampling call contract changed')
        risky, ordinary = self.pools[phase]
        fallback = self.arm == 'balanced' and (not len(risky) or not len(ordinary))
        if self.arm == 'uniform' or fallback:
            chosen = self.rng.integers(high, size=size)
        else:
            count = int(size * self.fraction)
            chosen = np.concatenate([self.rng.choice(risky, size=count, replace=True),
                                     self.rng.choice(ordinary, size=size - count, replace=True)])
            chosen = self.rng.permutation(chosen)
        counter = self.counts[phase]
        counter['calls'] += 1
        counter['draws'] += len(chosen)
        counter['failure_draws'] += int(self.risk[phase][chosen].sum())
        counter['fallback_calls'] += int(fallback)
        self.phase = 1 - phase
        return chosen

    def state_dict(self) -> dict[str, Any]:
        return dict(rng=deepcopy(self.rng.bit_generator.state), phase=self.phase,
                    counts=deepcopy(self.counts), arm=self.arm)

    def load_state_dict(self, saved: dict[str, Any]) -> None:
        if saved['arm'] != self.arm or saved['phase'] not in (0, 1):
            raise ValueError('Incompatible sampler state')
        self.rng.bit_generator.state = deepcopy(saved['rng'])
        self.phase, self.counts = saved['phase'], deepcopy(saved['counts'])

    def availability(self) -> dict[str, Any]:
        return {name: dict(failure=len(pair[0]), ordinary=len(pair[1]))
                for name, pair in zip(('one_step', 'recursive_positions'), self.pools)}
