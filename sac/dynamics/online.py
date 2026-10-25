"""Online observable GNN dynamics, optional reward/TD-value heads, and model MPC.

This is a TD-MPC2-inspired explicit-state experiment, not the TD-MPC2 algorithm.
Only actual environment transitions enter replay. Model rollouts score actions.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import ObservableDynamicsGNN, physical_graph_loss
from .planner import CEMConfig, cem_plan
from .training import tensor_graph


def td_target(reward: torch.Tensor, terminated: torch.Tensor, next_value: torch.Tensor,
              gamma: float) -> torch.Tensor:
    """True terminals stop bootstrap; time-limit truncations do not."""
    return reward + gamma * (~terminated.bool()).to(reward) * next_value


class OnlineReplay:
    """Fixed-capacity indexed real transitions; sequence sampling cannot cross resets."""

    def __init__(self, capacity: int, nodes: int) -> None:
        self.capacity, self.size = capacity, 0
        self.state = torch.empty(capacity, nodes, 6)
        self.next_state = torch.empty_like(self.state)
        self.action = torch.empty(capacity, nodes, 1)
        self.progress = torch.empty(capacity, 2)
        self.reward = torch.empty(capacity)
        self.base_reward = torch.empty(capacity)
        self.terminated = torch.empty(capacity, dtype=torch.bool)
        self.truncated = torch.empty(capacity, dtype=torch.bool)
        self.episode = torch.empty(capacity, dtype=torch.long)

    def append(self, *, state, next_state, action, progress, reward: float,
               base_reward: float, terminated: bool, truncated: bool, episode: int) -> None:
        if self.size >= self.capacity:
            raise ValueError('Replay capacity exhausted; protocol cannot silently discard data')
        i = self.size
        for name in ('state', 'next_state', 'action', 'progress'):
            value = torch.as_tensor(locals()[name], dtype=torch.float32)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite real transition: {name}')
            getattr(self, name)[i] = value
        for name in ('reward', 'base_reward', 'terminated', 'truncated', 'episode'):
            getattr(self, name)[i] = locals()[name]
        self.size += 1

    def starts(self, horizon: int) -> torch.Tensor:
        if horizon < 1:
            raise ValueError('horizon must be positive')
        if self.size < horizon:
            return torch.empty(0, dtype=torch.long)
        starts = torch.arange(self.size - horizon + 1)
        # Episode identity is authoritative; also guard terminal/truncation boundaries.
        good = self.episode[starts] == self.episode[starts + horizon - 1]
        for offset in range(horizon - 1):
            good &= ~(self.terminated[starts + offset] | self.truncated[starts + offset])
        return starts[good]

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name)[:self.size].numpy() for name in
                ('state', 'next_state', 'action', 'progress', 'reward', 'base_reward',
                 'terminated', 'truncated', 'episode')}


class TaskModel(nn.Module):
    """Physical dynamics and graph-pooled reward/twin action-value predictions."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.dynamics = ObservableDynamicsGNN(width, layers)
        self.reward = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.q1 = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.q2 = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def task(self, state: torch.Tensor, action: torch.Tensor, graph: Mapping):
        features = self.dynamics.node_features(state, action, graph)
        pooled = features[:, graph['physical_mask'].bool()].mean(1)
        return tuple(head(pooled).squeeze(-1) for head in (self.reward, self.q1, self.q2))


class GraphPolicy(nn.Module):
    """A TD actor used only for terminal value and TD targets."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.encoder = ObservableDynamicsGNN(width, layers)
        self.output = nn.Linear(width, 1)

    def forward(self, state: torch.Tensor, graph: Mapping) -> torch.Tensor:
        zeros = state.new_zeros((*state.shape[:2], 1))
        features = self.encoder.node_features(state, zeros, graph)
        return self.output(features).tanh() * graph['action_mask'][None, :, None]


class OnlineLearner:
    """One real-replay update and bounded learned-model MPC action selection."""

    def __init__(self, metadata: dict, initial: np.ndarray, settings: dict, seed: int,
                 use_value: bool) -> None:
        torch.manual_seed(seed)
        self.settings, self.metadata, self.use_value = settings, metadata, use_value
        self.graph = tensor_graph({**metadata, 'state': initial[None]})
        self.model = TaskModel(int(settings['width']), int(settings['layers']))
        self.actor = GraphPolicy(int(settings['width']), int(settings['layers']))
        self.target, self.target_actor = deepcopy(self.model), deepcopy(self.actor)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(settings['learning_rate']), weight_decay=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(settings['actor_learning_rate']))
        self.updates = 0
        self.cem = CEMConfig(**settings['cem'])
        self.edges = torch.as_tensor(metadata['structural_edges'], dtype=torch.long)
        self.rng = np.random.default_rng(seed + 200000)

    def normalize_from_warmup(self, replay: OnlineReplay) -> None:
        """Freeze shared normalization using only each run's initial real fitting data."""
        physical = self.graph['physical_mask'].bool()
        states = replay.state[:replay.size, physical].reshape(-1, 6)
        delta = replay.next_state[:replay.size] - replay.state[:replay.size]
        delta[:, :, :2] += replay.progress[:replay.size, None]
        scale = states.std(0, unbiased=False).clamp_min(.001)
        delta_scale = delta[:, physical].square().mean((0, 1)).sqrt().clamp_min(.001)
        for encoder in (self.model.dynamics, self.actor.encoder):
            encoder.configure_normalization(states.mean(0), scale, delta_scale,
                                            float(self.metadata['initial_diagonal']))
        self.target.load_state_dict(self.model.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def update(self, replay: OnlineReplay) -> dict[str, float]:
        cfg, graph, model = self.settings, self.graph, self.model
        batch = int(cfg['batch_size'])
        idx = torch.as_tensor(self.rng.integers(replay.size, size=batch))
        state, action = replay.state[idx], replay.action[idx]
        target_delta = replay.next_state[idx] - state
        target_delta[:, :, :2] += replay.progress[idx, None]
        loss = physical_graph_loss(model.dynamics(state, action, graph), target_delta,
                                   graph['physical_mask'], model.dynamics.delta_scale)
        one_step = loss.detach()
        horizon = int(cfg['training_horizon'])
        valid = replay.starts(horizon)
        rollout = loss.new_zeros(())
        if len(valid):
            starts = valid[torch.as_tensor(self.rng.integers(len(valid), size=batch))]
            imagined = replay.state[starts]
            progress = imagined.new_zeros((batch, 2))
            truth = progress.clone()
            for offset in range(horizon):
                imagined, move = model.dynamics.step(imagined, replay.action[starts + offset], graph)
                progress, truth = progress + move, truth + replay.progress[starts + offset]
                rollout = rollout + physical_graph_loss(imagined, replay.next_state[starts + offset],
                    graph['physical_mask'], model.dynamics.state_scale)
                rollout = rollout + ((progress - truth) / (model.dynamics.delta_scale[:2] * (offset + 1))).square().mean()
            rollout = rollout / horizon
            loss = loss + float(cfg['rollout_weight']) * rollout
        reward_loss, q_loss = loss.new_zeros(()), loss.new_zeros(())
        if self.use_value:
            scale = float(cfg['reward_scale'])
            reward, q1, q2 = model.task(state, action, graph)
            with torch.no_grad():
                next_action = self.target_actor(replay.next_state[idx], graph)
                noise = torch.randn_like(next_action).mul(.1).clamp(-.2, .2)
                next_action = (next_action + noise).clamp(-1, 1) * graph['action_mask'][None, :, None]
                _, n1, n2 = self.target.task(replay.next_state[idx], next_action, graph)
                target = td_target(replay.reward[idx] / scale, replay.terminated[idx],
                                   torch.minimum(n1, n2), float(cfg['gamma']))
            reward_loss = F.smooth_l1_loss(reward, replay.base_reward[idx] / scale)
            q_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
            loss = loss + reward_loss + q_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite update loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        actor_loss = 0.
        if self.use_value and self.updates % int(cfg['actor_delay']) == 0:
            for param in model.parameters():
                param.requires_grad_(False)
            proposed = self.actor(state, graph)
            _, q1, q2 = model.task(state, proposed, graph)
            objective = -torch.minimum(q1, q2).mean()
            self.actor_optimizer.zero_grad()
            objective.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_optimizer.step()
            actor_loss = float(objective.detach())
            for param in model.parameters():
                param.requires_grad_(True)
        with torch.no_grad():
            tau = float(cfg['tau'])
            for target_module, live in ((self.target, model), (self.target_actor, self.actor)):
                for destination, source in zip(target_module.parameters(), live.parameters()):
                    destination.lerp_(source, tau)
        return dict(loss=float(loss.detach()), one_step=float(one_step), rollout=float(rollout.detach()),
                    reward_loss=float(reward_loss.detach()), q_loss=float(q_loss.detach()), actor_loss=actor_loss)

    def collapsed(self, state: torch.Tensor) -> torch.Tensor:
        """Vectorized nominal geometry rigidity, using the audited bar/rank convention."""
        metadata, graph = self.metadata, self.graph
        positions = state[:, graph['physical_order_index'], :3]
        axes = metadata['axis_indices']
        delta = positions[:, self.edges[:, 1]][:, :, axes] - positions[:, self.edges[:, 0]][:, :, axes]
        length = delta.norm(dim=-1, keepdim=True)
        directions = torch.where(length >= 1e-8, delta / length.clamp_min(1e-8), torch.zeros_like(delta))
        batch, bars, dims = directions.shape
        incidence = state.new_zeros((bars, positions.shape[1]))
        incidence[torch.arange(bars), self.edges[:, 0]] = -1
        incidence[torch.arange(bars), self.edges[:, 1]] = 1
        rigidity = (directions[:, :, None, :] * incidence[None, :, :, None]).reshape(batch, bars, -1)
        matrix = rigidity @ rigidity.transpose(1, 2)
        finite = torch.isfinite(matrix).all((1, 2))
        matrix = torch.where(finite[:, None, None], matrix, torch.zeros_like(matrix))
        rank = positions.shape[1] * dims - dims - dims * (dims - 1) // 2
        if bars < rank:
            return torch.ones(batch, dtype=torch.bool)
        eigen = torch.linalg.eigvalsh(matrix)[:, bars - rank].clamp_min(0)
        if metadata['wcrm']:
            eigen /= rigidity.square().sum((1, 2)).clamp_min(1e-12)
        normalized = eigen / float(metadata['initial_rigidity_reference'])
        return ~finite | ~torch.isfinite(state).all((1, 2)) | (normalized < float(metadata['critical_eig_threshold']))

    @torch.no_grad()
    def score(self, initial: np.ndarray, sequences: np.ndarray) -> np.ndarray:
        """Discounted predicted reward plus optional surviving terminal actor/Q value."""
        cfg, graph = self.settings, self.graph
        actions = torch.as_tensor(sequences, dtype=torch.float32).reshape(len(sequences), self.cem.horizon, -1, 1)
        state = torch.as_tensor(initial, dtype=torch.float32)[None].expand(len(actions), -1, -1).clone()
        score = state.new_zeros(len(actions))
        alive = torch.ones(len(actions), dtype=torch.bool)
        gamma, scale = float(cfg['gamma']), float(cfg['reward_scale'])
        for step in range(self.cem.horizon):
            action = actions[:, step]
            predicted, progress = self.model.dynamics.step(state, action, graph)
            failure = self.collapsed(predicted)
            if self.use_value:
                reward = self.model.task(state, action, graph)[0] * scale
            else:
                commands = (action[..., 0] * graph['action_mask']) @ graph['routing_incidence'].T * graph['speed']
                bounds = graph['ctrlrange']
                commands = commands.clamp(bounds[:, 0], bounds[:, 1])
                reward = progress[:, 0] / float(self.metadata['initial_diagonal'])
                reward -= float(cfg['objective']['effort_weight']) * commands.square().mean(1) * float(self.metadata['control_interval_seconds'])
            reward -= float(cfg['objective']['failure_cost']) * failure
            finite = torch.isfinite(reward) & torch.isfinite(predicted).all((1, 2))
            score += (gamma ** step) * torch.where(alive, torch.nan_to_num(reward, nan=-1., posinf=-1., neginf=-1.), 0.)
            alive &= ~failure & finite
            state = torch.nan_to_num(predicted, nan=0., posinf=0., neginf=0.)
        if self.use_value:
            action = self.actor(state, graph)
            _, q1, q2 = self.model.task(state, action, graph)
            terminal = torch.minimum(q1, q2) * scale
            score += gamma ** self.cem.horizon * torch.where(alive, terminal, 0.)
        return score.numpy()

    def plan(self, state: np.ndarray, rng: np.random.Generator, previous=None):
        return cem_plan(lambda sequences: self.score(state, sequences), len(self.metadata['action_mask']),
                        self.cem, rng, action_mask=np.asarray(self.metadata['action_mask']), previous_plan=previous)
