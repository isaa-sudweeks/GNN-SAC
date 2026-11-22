"""Tensor-batched rollout state for accelerator-resident vector environments."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from common.tensor_gnn_buffer import DenseGraphGroup, build_graph_static

# Info entries that are bookkeeping rather than reward terms.
NON_REWARD_INFO_KEYS = frozenset({"success", "terminated", "truncated"})


class VectorBucket:
    """Rollout state for one fixed-topology environment batch.

    Observations, episode returns, lengths, and reward components live in
    tensors with one row per batch slot, so a vector step costs a fixed number
    of tensor operations regardless of the number of environments. Only the
    done flags are copied to the host each step, to schedule resets.
    """

    def __init__(self, env, global_indices: torch.Tensor):
        self.env = env
        self.size = int(env.num_envs)
        self.global_indices = global_indices.cpu().numpy()
        self.device = env.action_device
        slot_tasks = list(env.slot_tasks())
        self.task_slots = {}
        for task in dict.fromkeys(slot_tasks):
            slots = [index for index, slot_task in enumerate(slot_tasks) if slot_task == task]
            self.task_slots[task] = (
                None if len(slots) == self.size
                else torch.as_tensor(slots, dtype=torch.long, device=self.device)
            )
        self.obs = None
        self.done = np.ones(self.size, dtype=bool)
        self.episode_return = torch.zeros(self.size, device=self.device)
        self.episode_length = torch.zeros(self.size, dtype=torch.long, device=self.device)
        self.normalizer_returns = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        self.reward_components: dict[str, torch.Tensor] = {}
        self.last_info: dict[str, torch.Tensor] = {}
        self._static = None

    def policy_group(self, cfg) -> DenseGraphGroup:
        """Current observations as a dense actor input group."""
        if self._static is None:
            template = Data(
                x=self.obs["x"][0],
                edge_index=self.env.edge_index,
                action_mask=self.obs["action_mask"][0],
                rigidity=self.obs["rigidity"][0],
            )
            if getattr(self.env, "edge_direction", None) is not None:
                template.edge_direction = self.env.edge_direction
            if self.env.edge_role is not None:
                template.edge_role = self.env.edge_role
            for key, value in getattr(self.env, "tube_physics_metadata", {}).items():
                template[key] = value
            action = torch.zeros(self.obs["x"].size(1), 1)
            static = build_graph_static(cfg, template, action)
            self._static = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in static.items()
            }
        return DenseGraphGroup(
            static=self._static,
            x=self.obs["x"],
            rigidity=self.obs["rigidity"],
            action_mask=self.obs["action_mask"],
        )

    def reset_done(self, observation_noise) -> None:
        """Reset finished slots and clear their episode statistics."""
        indices = np.flatnonzero(self.done)
        if not indices.size:
            return
        observations = self.env.reset_batch(indices.tolist())
        observations["x"] = observation_noise(observations["x"])
        if self.obs is None or indices.size == self.size:
            self.obs = observations
        else:
            mask = torch.from_numpy(self.done).to(self.device)
            self.obs = {
                key: torch.where(
                    mask.view(-1, *([1] * (value.dim() - 1))), value, self.obs[key]
                )
                for key, value in observations.items()
            }
        mask = torch.from_numpy(self.done).to(self.device)
        self.episode_return.masked_fill_(mask, 0.0)
        self.episode_length.masked_fill_(mask, 0)
        self.normalizer_returns.masked_fill_(mask, 0.0)
        for value in self.reward_components.values():
            value.masked_fill_(mask, 0.0)
        self.done[:] = False

    def record_step(self, reward: torch.Tensor, done: torch.Tensor, info: dict) -> None:
        """Accumulate per-slot episode statistics for one vector step."""
        self.episode_return += reward.float()
        self.episode_length += 1
        for key, value in info.items():
            if key in NON_REWARD_INFO_KEYS or not isinstance(value, torch.Tensor):
                continue
            if value.shape != (self.size,):
                continue
            total = self.reward_components.get(key)
            if total is None:
                total = self.reward_components[key] = torch.zeros(self.size, device=self.device)
            total += value.float()
        self.last_info = {
            key: info[key].float() if key in info else torch.zeros(self.size, device=self.device)
            for key in ("success", "terminated", "truncated")
        }

    def finished_episodes(self) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return finished slot indices with their summary and reward-component values."""
        slots = np.flatnonzero(self.done)
        if not slots.size or not self.last_info:
            return slots[:0], {}, {}
        index = torch.as_tensor(slots, device=self.device)
        names = ["episode_reward", "episode_length", "episode_success",
                 "episode_terminated", "episode_truncated"]
        summary = torch.stack((
            self.episode_return[index],
            self.episode_length[index].float(),
            self.last_info["success"][index],
            self.last_info["terminated"][index],
            self.last_info["truncated"][index],
        )).cpu().numpy()
        component_names = list(self.reward_components)
        components = (
            torch.stack([self.reward_components[key][index] for key in component_names]).cpu().numpy()
            if component_names else np.zeros((0, slots.size))
        )
        keep = summary[1] > 0
        return (
            slots[keep],
            {name: row[keep] for name, row in zip(names, summary)},
            {name: row[keep] for name, row in zip(component_names, components)},
        )

    def task_groups(self, keep: np.ndarray | None):
        """Yield ``(task, slot_index_or_None)`` pairs restricted to kept slots."""
        for task, slots in self.task_slots.items():
            if keep is None:
                yield task, slots
                continue
            kept = np.flatnonzero(keep) if slots is None else np.intersect1d(
                slots.cpu().numpy(), np.flatnonzero(keep)
            )
            if kept.size:
                yield task, torch.as_tensor(kept, dtype=torch.long, device=self.device)


def select_rows(value: torch.Tensor, slots: torch.Tensor | None) -> torch.Tensor:
    return value if slots is None else value.index_select(0, slots)
