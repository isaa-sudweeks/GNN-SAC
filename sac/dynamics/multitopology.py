"""Balanced source-topology views of a shared online dynamics optimizer."""
from __future__ import annotations

import torch

from .online import OnlineLearner, OnlineReplay
from .training import fitting_statistics


def normalize_warmups(learners: dict[str, OnlineLearner], replays: dict[str, OnlineReplay]) -> dict:
    """Freeze equal-topology physical-node moments from fitting warmups only."""
    episodes = [{**learner.metadata, **replays[name].arrays(), 'role': 'fitting'}
                for name, learner in learners.items()]
    statistics = fitting_statistics(episodes)
    for learner in learners.values():
        for encoder in (learner.model.dynamics, learner.actor.encoder):
            encoder.configure_normalization(torch.tensor(statistics['mean']),
                torch.tensor(statistics['scale']), torch.tensor(statistics['delta_scale']),
                statistics['geometry_scale'])
        learner.target.load_state_dict(learner.model.state_dict())
        learner.target_actor.load_state_dict(learner.actor.state_dict())
    return statistics


def share_dynamics(learners: dict[str, OnlineLearner]) -> None:
    """Keep graph/routing/RNG views distinct while sharing parameters and Adam state."""
    first = next(iter(learners.values()))
    for learner in learners.values():
        if learner.use_value:
            raise ValueError('This experiment permits dynamics-only MPC')
        learner.model, learner.optimizer, learner.target = first.model, first.optimizer, first.target


def update_round(learners: dict[str, OnlineLearner], replays: dict[str, OnlineReplay]) -> dict:
    """Exactly one homogeneous batch/update per topology, in declared order.

    Each update uses B one-step samples and B within-episode recursive windows.
    No padding, cross-topology windows, node-count weighting or pooled sampling.
    """
    horizon = int(next(iter(learners.values())).settings['training_horizon'])
    if any(not len(replay.starts(horizon)) for replay in replays.values()):
        raise ValueError('Every topology must have a valid recursive window')
    return {name: learner.update(replays[name]) for name, learner in learners.items()}
