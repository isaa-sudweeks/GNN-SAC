"""Nominal native-MuJoCo branches for source-only planner diagnostics."""

import copy
from dataclasses import dataclass, fields
import hashlib
from typing import Callable

import mujoco
import numpy as np


_RUNTIME_FIELDS = (
    "steps", "_t", "_elapsed_steps", "_cumulative_reward", "_has_reset",
    "_observation_rigidity", "_initial_critical_eig", "_broken_node_mask",
    "_broken_nodes_sampling_enabled", "_np_random", "_np_random_seed",
    "_has_reset", "_has_updated", "_disable_render_order_enforcing",
)


def _layers(env):
    layers = [env]
    while "env" in vars(layers[-1]):
        layers.append(layers[-1].env)
    return layers


def _signature(env):
    base = env.unwrapped
    model = base.mj_model.model
    config = {field.name: getattr(base.config, field.name) for field in fields(base.config)
              if field.name != "model_source"}
    payload = (str(base.mj_model.xml), model.opt.timestep, base.nsubsteps,
               config, str(base.source_config),
               tuple(type(layer).__name__ for layer in _layers(env)))
    return hashlib.sha256(repr(payload).encode()).hexdigest()


@dataclass
class SimulatorSnapshot:
    """Full MjData plus explicitly audited wrapper and environment runtime fields."""

    data: object
    runtime: list[dict]
    signature: str


@dataclass
class BranchRollout:
    observations: list
    rewards: np.ndarray
    infos: list[dict]
    com_positions: np.ndarray
    elapsed_times: np.ndarray
    terminated: bool
    truncated: bool
    nonfinite: bool
    simulator_steps: int


class SimulatorBranch:
    """Evaluate candidates on a distinct, identically configured nominal env.

    No reset, restore, or step operation is applied to the execution env.
    This contract is intentionally restricted to generated abstract robots,
    fixed physics and no rendering. New wrappers require a runtime-field audit.
    """

    def __init__(self, live_env, planning_env):
        if live_env.unwrapped is planning_env.unwrapped:
            raise ValueError("Planning and execution environments must be distinct")
        self.live_env = live_env
        self.planning_env = planning_env
        for env in (live_env, planning_env):
            base = env.unwrapped
            if base.config.domain_randomization is not None:
                raise ValueError("Branching requires fixed nominal physics")
            if bool(base.source_config.get("truss_realistic", False)):
                raise ValueError("Realistic controller restoration is not audited")
            if base.render_mode is not None:
                raise ValueError("Branching requires rendering disabled")
        if _signature(live_env) != _signature(planning_env):
            raise ValueError("Planning configuration/model/wrappers differ from execution")

    def capture(self) -> SimulatorSnapshot:
        """Capture the live environment without evaluating observations or stepping."""
        return SimulatorSnapshot(
            copy.copy(self.live_env.unwrapped.mj_model.data),
            [{key: copy.deepcopy(vars(layer)[key]) for key in _RUNTIME_FIELDS
              if key in vars(layer)} for layer in _layers(self.live_env)],
            _signature(self.live_env),
        )

    def restore(self, snapshot: SimulatorSnapshot) -> None:
        """Restore all MjData, including solver state and derived caches, verbatim."""
        if snapshot.signature != _signature(self.planning_env):
            raise ValueError("Snapshot is incompatible with planning environment")
        base = self.planning_env.unwrapped
        mujoco.mj_copyData(base.mj_model.data, base.mj_model.model, snapshot.data)
        for layer, runtime in zip(_layers(self.planning_env), snapshot.runtime, strict=True):
            for key, value in runtime.items():
                setattr(layer, key, copy.deepcopy(value))

    def rollout(self, snapshot: SimulatorSnapshot, actions: np.ndarray) -> BranchRollout:
        """Restore before every sequence; stop at collapse, timeout or nonfinite data."""
        actions = np.asarray(actions)
        if not np.all(np.isfinite(actions)):
            raise ValueError("Candidate actions must be finite")
        self.restore(snapshot)
        base = self.planning_env.unwrapped
        observations, rewards, infos = [], [], []
        positions = [base._center_of_mass().copy()]
        start_time = float(base.mj_model.data.time)
        times = [0.0]
        terminated = truncated = nonfinite = False
        steps = 0
        for action in actions:
            result = self.planning_env.step(action.copy())
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated = bool(info.get("terminated", done))
                truncated = bool(info.get("truncated", False))
            observations.append(copy.deepcopy(obs))
            rewards.append(float(reward))
            infos.append(copy.deepcopy(info))
            positions.append(base._center_of_mass().copy())
            times.append(float(base.mj_model.data.time) - start_time)
            steps += int(info.get("substeps_executed", base.nsubsteps))
            nonfinite = not (np.isfinite(reward) and
                             np.all(np.isfinite(base.mj_model.data.qpos)) and
                             np.all(np.isfinite(base.mj_model.data.qvel)))
            if terminated or truncated or nonfinite:
                break
        return BranchRollout(observations, np.asarray(rewards), infos,
                             np.asarray(positions), np.asarray(times),
                             bool(terminated), bool(truncated), nonfinite, steps)

    def evaluate(self, snapshot: SimulatorSnapshot, sequences: np.ndarray,
                 score: Callable[[BranchRollout, int], float]) -> np.ndarray:
        """Score each candidate with its requested horizon, including stopped tails."""
        return np.asarray([score(self.rollout(snapshot, sequence), len(sequence))
                           for sequence in sequences], dtype=float)
