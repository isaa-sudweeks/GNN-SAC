import contextlib
from collections.abc import Sequence
import math
import re

import torch 
import torch.nn.functional as F 
from torch_geometric.data import Batch, Data

from common.gnn_actor_critic import GNNActorCritic 
from common.graph_transforms import policy_action_mask, prepare_graph

class GNNSAC(torch.nn.Module):
    """
    Soft Actor-Critic for GNN-based control.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cuda"))
        self.model = GNNActorCritic(cfg).to(self.device)
        capturable = self.device.type in {"cuda", "xpu", "hpu", "privateuseone", "xla"}

        self.q_optim = torch.optim.Adam(self.model._Qs.parameters(), lr=self.cfg.lr, capturable=capturable)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable) # What does eps do in this case 

        init_alpha = float(getattr(self.cfg, "entropy_coef", 0.2))
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(init_alpha, device=self.device)))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr, capturable=capturable)
        self.safety_enabled = bool(self._safety_cfg("enabled", False))
        self.safety_horizon = int(self._safety_cfg("horizon", 250))
        self._safety_tasks = self._configured_tasks()
        self._safety_task_index = {
            task: index for index, task in enumerate(self._safety_tasks)
        }
        self._pending_safety_outcomes = {task: [] for task in self._safety_tasks}
        self._resolved_safety_counts = {task: 0 for task in self._safety_tasks}
        self._last_safety_metrics = {}
        self.curriculum_enabled = bool(self._curriculum_cfg("enabled", False))
        initial_horizon = int(self._curriculum_cfg("initial_horizon", 50))
        self.active_horizon_by_task = {
            task: initial_horizon if self.curriculum_enabled else self.safety_horizon
            for task in self._safety_tasks
        }
        self._curriculum_pass_streaks = {task: 0 for task in self._safety_tasks}
        self._curriculum_promotion_counts = {task: 0 for task in self._safety_tasks}
        self._curriculum_stale_outcomes = {task: 0 for task in self._safety_tasks}
        self._last_cost_horizon_metrics = {}
        self._last_cost_target_metrics = {}
        self.actor_safety_penalty = str(
            self._safety_cfg("actor_penalty", "probability")
        ).lower()
        if self.safety_enabled:
            if self.safety_horizon <= 0:
                raise ValueError("safety_constraint.horizon must be positive")
            self._validate_curriculum_config()
            if self.actor_safety_penalty not in {
                "probability",
                "softplus_logit",
                "raw_logit",
            }:
                raise ValueError(
                    "safety_constraint.actor_penalty must be one of "
                    "probability, softplus_logit, or raw_logit"
                )
            lambda_init = float(self._safety_cfg("lambda_init", 0.1))
            lambda_max = float(self._safety_cfg("lambda_max", 100.0))
            lambda_batch_size = int(self._safety_cfg("lambda_batch_size", 32))
            if not (0.0 < lambda_init <= lambda_max):
                raise ValueError("safety lambda_init must be positive and no larger than lambda_max")
            if lambda_max <= 0.0 or lambda_batch_size <= 0:
                raise ValueError("safety lambda_max and lambda_batch_size must be positive")
            self.raw_lambdas = torch.nn.Parameter(
                torch.full(
                    (len(self._safety_tasks),),
                    self._inverse_softplus(lambda_init),
                    device=self.device,
                )
            )
            self.cost_q_optim = torch.optim.Adam(
                self.model._CostQs.parameters(),
                lr=float(self._safety_cfg("cost_critic_lr", self.cfg.lr)),
                capturable=capturable,
            )
            self.lambda_optim = torch.optim.Adam(
                [self.raw_lambdas],
                lr=float(self._safety_cfg("lambda_lr", 1e-3)),
                capturable=capturable,
            )
        target_entropy = getattr(self.cfg, "target_entropy", "auto")
        if target_entropy == "auto":
            entropy_dim = getattr(cfg, "num_policy_actions", cfg.action_dim)
            self.target_entropy = -float(entropy_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.model.eval()
        self.discount = float(getattr(self.cfg, "discount", self._get_discount(self.cfg.episode_length)))

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        print("Target entropy:", self.target_entropy)

    def _safety_cfg(self, name, default=None):
        safety_cfg = getattr(self.cfg, "safety_constraint", None)
        if safety_cfg is None:
            return default
        if hasattr(safety_cfg, "get"):
            return safety_cfg.get(name, default)
        return getattr(safety_cfg, name, default)

    def _curriculum_cfg(self, name, default=None):
        curriculum = self._safety_cfg("curriculum", None)
        if curriculum is None:
            return default
        if hasattr(curriculum, "get"):
            return curriculum.get(name, default)
        return getattr(curriculum, name, default)

    def _validate_curriculum_config(self):
        if not self.curriculum_enabled:
            return
        initial = int(self._curriculum_cfg("initial_horizon", 50))
        factor = float(self._curriculum_cfg("promotion_factor", 1.5))
        successes = int(self._curriculum_cfg("consecutive_success_windows", 3))
        boundary = float(self._curriculum_cfg("boundary_sample_probability", 0.5))
        upper = float(self._curriculum_cfg("upper_half_sample_probability", 0.25))
        if not 1 <= initial <= self.safety_horizon:
            raise ValueError(
                "safety curriculum initial_horizon must be in "
                f"[1, {self.safety_horizon}]"
            )
        if not math.isfinite(factor) or factor <= 1.0:
            raise ValueError("safety curriculum promotion_factor must be finite and greater than 1")
        if successes <= 0:
            raise ValueError("safety curriculum consecutive_success_windows must be positive")
        if not 0.0 <= boundary <= 1.0 or not 0.0 <= upper <= 1.0:
            raise ValueError("safety curriculum sampling probabilities must be in [0, 1]")
        if boundary + upper > 1.0:
            raise ValueError("safety curriculum sampling probabilities must sum to at most 1")

    def _configured_tasks(self):
        backend = str(getattr(self.cfg, "mujoco_backend", "mujoco")).lower()
        topologies = getattr(self.cfg, "truss_topologies", None)
        if backend == "mjx" and topologies and len(topologies) > 1:
            base_task = str(getattr(self.cfg, "task", "truss-graph")).split(":", 1)[0]
            tasks = [f"{base_task}:{topology}" for topology in topologies]
        elif bool(getattr(self.cfg, "multitask", False)):
            tasks = [str(task) for task in getattr(self.cfg, "tasks", [])]
        else:
            tasks = [str(getattr(self.cfg, "task", "task"))]
        return list(dict.fromkeys(tasks))

    @staticmethod
    def _inverse_softplus(value):
        value = float(value)
        return value if value > 20.0 else math.log(math.expm1(value))

    def _budget(self, task):
        overrides = self._safety_cfg("budgets_by_topology", {}) or {}
        topology = str(task).split(":", 1)[1] if ":" in str(task) else str(task)
        if hasattr(overrides, "get"):
            value = overrides.get(topology, overrides.get(str(task), None))
        else:
            value = None
        budget = float(self._safety_cfg("default_budget", 0.1) if value is None else value)
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"Safety budget for {task!r} must be in [0, 1]")
        return budget

    def safety_lambdas(self):
        if not self.safety_enabled:
            return torch.empty(0, device=self.device)
        return F.softplus(self.raw_lambdas).clamp_max(
            float(self._safety_cfg("lambda_max", 100.0))
        )

    def _safety_identity(self):
        if not self.safety_enabled:
            return {"enabled": False}
        identity = {
            "enabled": True,
            "horizon": self.safety_horizon,
            "tasks": list(self._safety_tasks),
            "budgets": {task: self._budget(task) for task in self._safety_tasks},
            "cost_critic_lr": float(self._safety_cfg("cost_critic_lr", self.cfg.lr)),
            "lambda_lr": float(self._safety_cfg("lambda_lr", 1e-3)),
            "lambda_init": float(self._safety_cfg("lambda_init", 0.1)),
            "lambda_max": float(self._safety_cfg("lambda_max", 100.0)),
            "lambda_batch_size": int(self._safety_cfg("lambda_batch_size", 32)),
            "actor_penalty": self.actor_safety_penalty,
        }
        if self.curriculum_enabled:
            identity["curriculum"] = {
                "enabled": True,
                "initial_horizon": int(self._curriculum_cfg("initial_horizon", 50)),
                "promotion_factor": float(self._curriculum_cfg("promotion_factor", 1.5)),
                "consecutive_success_windows": int(
                    self._curriculum_cfg("consecutive_success_windows", 3)
                ),
                "boundary_sample_probability": float(
                    self._curriculum_cfg("boundary_sample_probability", 0.5)
                ),
                "upper_half_sample_probability": float(
                    self._curriculum_cfg("upper_half_sample_probability", 0.25)
                ),
            }
        return identity

    def active_safety_horizon(self, task=None):
        if task is None:
            if len(self._safety_tasks) != 1:
                raise ValueError("A safety task is required for a multitask constrained run")
            task = self._safety_tasks[0]
        task = str(task)
        try:
            return int(self.active_horizon_by_task[task])
        except KeyError as exc:
            raise KeyError(
                f"Unknown safety task {task!r}; expected one of {self._safety_tasks!r}"
            ) from exc

    def safety_lambda(self, task):
        try:
            index = self._safety_task_index[str(task)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown safety task {task!r}; expected one of {self._safety_tasks!r}"
            ) from exc
        return self.safety_lambdas()[index]

    def observe_safety_outcome(self, task, outcome, horizon=None):
        """Consume one on-policy reset-window outcome and update its multiplier in batches."""
        if not self.safety_enabled:
            return {}
        task = str(task)
        if task not in self._pending_safety_outcomes:
            raise KeyError(f"Unknown safety outcome task {task!r}")
        outcome = float(outcome)
        if outcome not in {0.0, 1.0}:
            raise ValueError("Safety outcomes must be binary")
        active_horizon = self.active_safety_horizon(task)
        horizon = active_horizon if horizon is None else int(horizon)
        key = self._diagnostic_key(task)
        if horizon != active_horizon:
            self._curriculum_stale_outcomes[task] += 1
            metrics = self._curriculum_metrics(task, promoted=False)
            self._last_safety_metrics.update(metrics)
            return metrics
        pending = self._pending_safety_outcomes[task]
        pending.append(outcome)
        batch_size = int(self._safety_cfg("lambda_batch_size", 32))
        if len(pending) < batch_size:
            return {}
        batch = pending[:batch_size]
        del pending[:batch_size]
        self._resolved_safety_counts[task] += batch_size
        rate = torch.tensor(batch, dtype=torch.float32, device=self.device).mean()
        budget = self._budget(task)
        multiplier = self.safety_lambda(task)
        lambda_loss = -multiplier * (rate.detach() - budget)
        self.lambda_optim.zero_grad(set_to_none=True)
        lambda_loss.backward()
        self.lambda_optim.step()
        with torch.no_grad():
            maximum_raw = self._inverse_softplus(float(self._safety_cfg("lambda_max", 100.0)))
            self.raw_lambdas.clamp_(min=-20.0, max=maximum_raw)
        multiplier = self.safety_lambda(task).detach()
        metrics = {
            f"safety/lambda/{key}": multiplier,
            f"safety/budget/{key}": budget,
            f"safety/observed_horizon_collapse_rate/{key}": rate,
            f"safety/violation/{key}": rate - budget,
            f"safety/lambda_saturated/{key}": float(
                multiplier >= float(self._safety_cfg("lambda_max", 100.0)) - 1e-6
            ),
            f"safety/resolved_window_count/{key}": self._resolved_safety_counts[task],
        }
        promoted = False
        if self.curriculum_enabled:
            if float(rate) <= budget:
                self._curriculum_pass_streaks[task] += 1
            else:
                self._curriculum_pass_streaks[task] = 0
            required = int(self._curriculum_cfg("consecutive_success_windows", 3))
            if (
                self._curriculum_pass_streaks[task] >= required
                and active_horizon < self.safety_horizon
            ):
                factor = float(self._curriculum_cfg("promotion_factor", 1.5))
                promoted_horizon = min(
                    self.safety_horizon,
                    int(math.ceil(active_horizon * factor)),
                )
                if promoted_horizon > active_horizon:
                    self.active_horizon_by_task[task] = promoted_horizon
                    self._curriculum_promotion_counts[task] += 1
                    self._curriculum_pass_streaks[task] = 0
                    self._pending_safety_outcomes[task].clear()
                    promoted = True
        metrics.update(self._curriculum_metrics(task, promoted=promoted))
        self._last_safety_metrics.update(metrics)
        return metrics

    def _curriculum_metrics(self, task, *, promoted):
        key = self._diagnostic_key(task)
        metrics = {
            f"safety/active_horizon/{key}": self.active_safety_horizon(task),
            f"safety/max_horizon/{key}": self.safety_horizon,
            f"safety/curriculum_pass_streak/{key}": self._curriculum_pass_streaks[task],
            f"safety/curriculum_promotion_count/{key}": self._curriculum_promotion_counts[task],
            f"safety/curriculum_stale_outcomes/{key}": self._curriculum_stale_outcomes[task],
        }
        if promoted:
            metrics[f"safety/curriculum_promoted/{key}"] = 1.0
        return metrics

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _safe_action(self, action):
        """
        TODO: I think it would be benefitial to add some sort of checking to make sure that the GNN stuff is right for the actions.

        """
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        state = {
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "safety_enabled": self.safety_enabled,
        }
        if self.safety_enabled:
            state.update(
                raw_lambdas=self.raw_lambdas.detach().cpu(),
                safety_tasks=list(self._safety_tasks),
                safety_horizon=self.safety_horizon,
                safety_identity=self._safety_identity(),
            )
            if self.curriculum_enabled:
                state.update(self._curriculum_state_dict())
        torch.save(state, fp)
    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device, weights_only=False)
        saved_safety = bool(state_dict.get("safety_enabled", False)) if isinstance(state_dict, dict) else False
        if saved_safety != self.safety_enabled:
            raise ValueError(
                "Checkpoint constrained-safety setting does not match the current configuration; "
                "start a fresh constrained run or load with safety_constraint.enabled=false."
            )
        self.model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
        if isinstance(state_dict, dict) and "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        if self.safety_enabled:
            if self.curriculum_enabled and "active_horizon_by_task" not in state_dict:
                raise ValueError(
                    "Checkpoint is missing safety curriculum state; start a fresh curriculum run."
                )
            if state_dict.get("safety_identity") != self._safety_identity():
                raise ValueError(
                    "Checkpoint safety configuration does not match the current run configuration"
                )
            if list(state_dict.get("safety_tasks", [])) != self._safety_tasks:
                raise ValueError("Checkpoint safety tasks do not match configured topology tasks")
            if int(state_dict.get("safety_horizon", -1)) != self.safety_horizon:
                raise ValueError("Checkpoint safety horizon does not match configured horizon")
            self.raw_lambdas.data.copy_(state_dict["raw_lambdas"].to(self.device))
            if self.curriculum_enabled:
                self._load_curriculum_state_dict(state_dict)

    def training_state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "q_optim": self.q_optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "safety_enabled": self.safety_enabled,
        }
        if self.safety_enabled:
            state.update(
                raw_lambdas=self.raw_lambdas.detach().cpu(),
                safety_tasks=list(self._safety_tasks),
                safety_horizon=self.safety_horizon,
                safety_identity=self._safety_identity(),
                cost_q_optim=self.cost_q_optim.state_dict(),
                lambda_optim=self.lambda_optim.state_dict(),
                pending_safety_outcomes={
                    task: list(values)
                    for task, values in self._pending_safety_outcomes.items()
                },
                resolved_safety_counts=dict(self._resolved_safety_counts),
            )
            if self.curriculum_enabled:
                state.update(self._curriculum_state_dict())
        return state

    def _curriculum_state_dict(self):
        return {
            "active_horizon_by_task": dict(self.active_horizon_by_task),
            "curriculum_pass_streaks": dict(self._curriculum_pass_streaks),
            "curriculum_promotion_counts": dict(self._curriculum_promotion_counts),
            "curriculum_stale_outcomes": dict(self._curriculum_stale_outcomes),
        }

    def _load_curriculum_state_dict(self, state_dict):
        saved_horizons = state_dict.get("active_horizon_by_task", {})
        if list(saved_horizons) != self._safety_tasks:
            raise ValueError("Checkpoint curriculum tasks do not match configured topology tasks")
        horizons = {task: int(saved_horizons[task]) for task in self._safety_tasks}
        if any(not 1 <= horizon <= self.safety_horizon for horizon in horizons.values()):
            raise ValueError("Checkpoint contains an invalid active safety horizon")
        self.active_horizon_by_task = horizons
        self._curriculum_pass_streaks = {
            task: int(state_dict.get("curriculum_pass_streaks", {}).get(task, 0))
            for task in self._safety_tasks
        }
        self._curriculum_promotion_counts = {
            task: int(state_dict.get("curriculum_promotion_counts", {}).get(task, 0))
            for task in self._safety_tasks
        }
        self._curriculum_stale_outcomes = {
            task: int(state_dict.get("curriculum_stale_outcomes", {}).get(task, 0))
            for task in self._safety_tasks
        }

    def load_training_state_dict(self, state_dict):
        self.load(state_dict)
        if "q_optim" in state_dict:
            self.q_optim.load_state_dict(state_dict["q_optim"])
        if "pi_optim" in state_dict:
            self.pi_optim.load_state_dict(state_dict["pi_optim"])
        if "alpha_optim" in state_dict:
            self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        if self.safety_enabled:
            if "cost_q_optim" not in state_dict or "lambda_optim" not in state_dict:
                raise ValueError(
                    "Constrained checkpoint is missing safety optimizer state; start a fresh run."
                )
            self.cost_q_optim.load_state_dict(state_dict["cost_q_optim"])
            self.lambda_optim.load_state_dict(state_dict["lambda_optim"])
            saved_pending = state_dict.get("pending_safety_outcomes", {})
            self._pending_safety_outcomes = {
                task: list(saved_pending.get(task, [])) for task in self._safety_tasks
            }
            saved_counts = state_dict.get("resolved_safety_counts", {})
            self._resolved_safety_counts = {
                task: int(saved_counts.get(task, 0)) for task in self._safety_tasks
            }

    @torch.no_grad()
    def act(self, obs, t0 = False, eval_mode = False):
        """
        Right now this assumes that obs is coming in as a torch geometric Data object
        """
        return self.act_batch([obs], eval_mode=eval_mode)[0]

    @torch.no_grad()
    def act_batch(self, observations: Sequence[Data], eval_mode: bool = False) -> list[torch.Tensor]:
        """Compute actions for graph observations in one actor forward pass."""
        if not observations:
            return []

        node_counts = [int(obs.x.size(0)) for obs in observations]
        action_masks = [policy_action_mask(obs) for obs in observations]
        action_counts = [int(mask.sum()) for mask in action_masks]
        use_virtual_node = bool(getattr(self.cfg, "use_virtual_node", False))
        prepared = [
            prepare_graph(obs, use_virtual_node=use_virtual_node) for obs in observations
        ]
        obs_batch = Batch.from_data_list(prepared).to(self.device, non_blocking=True)
        if eval_mode:
            action = self.model.pi_mean(obs_batch)
        else:
            action, _ = self.model.pi(obs_batch)

        action = self._safe_action(action)
        keep_on_device = (
            str(getattr(self.cfg, "mujoco_backend", "mujoco")).lower() == "mjx"
            and bool(getattr(self.cfg, "mjx_zero_copy", True))
        )
        if not keep_on_device:
            action = action.cpu()
        if action.size(0) != sum(action_counts):
            raise RuntimeError(
                f"Actor produced {action.size(0)} node actions for "
                f"{sum(action_counts)} actuated observation nodes"
            )
        full_actions = []
        for node_count, mask, active_action in zip(
            node_counts, action_masks, action.split(action_counts, dim=0)
        ):
            full_action = action.new_zeros((node_count, action.size(-1)))
            full_action[mask.to(action.device)] = active_action
            full_actions.append(full_action)
        return full_actions

    @torch.no_grad()
    def predict_safety_risk(self, obs, task=None):
        """Predict deterministic-policy H-step collapse risk for one graph."""
        if not self.safety_enabled:
            raise RuntimeError("Safety risk prediction requires constrained safety")
        prepared = prepare_graph(
            obs, use_virtual_node=bool(getattr(self.cfg, "use_virtual_node", False))
        )
        batch = Batch.from_data_list([prepared]).to(self.device)
        action = self.model.pi_mean(batch)
        risk = self.model.cost_Q(
            batch,
            action,
            torch.tensor([self.active_safety_horizon(task)], device=self.device),
            return_type="max",
        )
        return float(risk.item())

    @torch.no_grad()
    def _td_target(self, next_obs, reward, terminated):
        """
        Again this assumes that next_obs is coming in as a torch geometric Data object.
        """
        reward = reward.view(-1)
        terminated = terminated.view(-1)
        next_action, next_info = self.model.pi(next_obs)
        target_q = self.model.Q(next_obs, next_action, return_type="min", target=True)
        target_v = target_q - self.alpha.detach() * next_info["log_prob"]
        return reward + self.discount * (1.0 - terminated) * target_v

    def update_q(self, obs, action, reward, terminated, next_obs):
        q_loss = self._q_loss(obs, action, reward, terminated, next_obs)

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._Qs.parameters(), self.cfg.grad_clip_norm)
        self.q_optim.step()
        return q_loss.detach(), q_grad_norm.detach()

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 7:
            return batch
        if len(batch) == 5:
            obs, action, reward, terminated, next_obs = batch
            zeros = None if terminated is None else torch.zeros_like(terminated)
            return obs, action, reward, zeros, terminated, terminated, next_obs
        raise ValueError(f"Expected a 5- or 7-field replay batch, got {len(batch)} fields")

    @torch.no_grad()
    def _cost_td_target(self, next_obs, collapse_cost, episode_end, horizon):
        collapse_cost = collapse_cost.view(-1).clamp(0.0, 1.0)
        episode_end = episode_end.view(-1).clamp(0.0, 1.0)
        horizon = horizon.view(-1).long()
        next_action, _ = self.model.pi(next_obs)
        continuation_horizon = torch.clamp(horizon - 1, min=1)
        next_risk = self.model.cost_Q(
            next_obs,
            next_action,
            continuation_horizon,
            return_type="max",
            target=True,
        )
        continuation = (horizon > 1).to(next_risk.dtype)
        target = collapse_cost + (
            (1.0 - collapse_cost)
            * (1.0 - episode_end)
            * continuation
            * next_risk
        )
        return target.clamp(0.0, 1.0)

    def _sample_cost_horizons(self, task, batch_size):
        if not self.curriculum_enabled:
            return torch.randint(
                1,
                self.safety_horizon + 1,
                (batch_size,),
                device=self.device,
            )
        active_horizon = self.active_safety_horizon(task)
        boundary_probability = float(
            self._curriculum_cfg("boundary_sample_probability", 0.5)
        )
        upper_probability = float(
            self._curriculum_cfg("upper_half_sample_probability", 0.25)
        )
        categories = torch.rand(batch_size, device=self.device)
        horizons = torch.randint(
            1,
            active_horizon + 1,
            (batch_size,),
            device=self.device,
        )
        upper_start = max(1, int(math.ceil(active_horizon / 2.0)))
        upper_mask = (
            (categories >= boundary_probability)
            & (categories < boundary_probability + upper_probability)
        )
        upper_count = int(upper_mask.sum().item())
        if upper_count:
            horizons[upper_mask] = torch.randint(
                upper_start,
                active_horizon + 1,
                (upper_count,),
                device=self.device,
            )
        horizons[categories < boundary_probability] = active_horizon
        return horizons

    def _record_cost_horizon_metrics(self, task, horizons):
        key = self._diagnostic_key(task)
        active_horizon = self.active_safety_horizon(task)
        self._last_cost_horizon_metrics.update(
            {
                f"safety/critic_sampled_horizon_mean/{key}": horizons.float().mean(),
                f"safety/critic_sampled_horizon_max/{key}": horizons.max(),
                f"safety/critic_sampled_boundary_fraction/{key}": (
                    horizons == active_horizon
                ).float().mean(),
            }
        )

    def _cost_q_loss(
        self, obs, action, collapse_cost, episode_end, next_obs, horizon=None, task=None
    ):
        batch_size = int(collapse_cost.numel())
        if horizon is None:
            horizon = self._sample_cost_horizons(task, batch_size)
        if task is not None:
            self._record_cost_horizon_metrics(task, horizon)
        target = self._cost_td_target(
            next_obs, collapse_cost, episode_end, horizon
        )
        if task is not None:
            key = self._diagnostic_key(task)
            detached_target = target.detach()
            self._last_cost_target_metrics.update(
                {
                    f"safety/cost_target_mean/{key}": detached_target.mean(),
                    f"safety/cost_target_std/{key}": detached_target.std(unbiased=False),
                    f"safety/cost_target_high_fraction/{key}": (
                        detached_target > 0.95
                    ).float().mean(),
                }
            )
        logits = self.model.cost_Q(
            obs,
            action,
            horizon,
            return_type="all",
            logits=True,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            target.unsqueeze(0).expand_as(logits),
        )

    def update_cost_q(self, obs, action, collapse_cost, episode_end, next_obs):
        cost_loss = self._cost_q_loss(
            obs, action, collapse_cost, episode_end, next_obs
        )
        self.cost_q_optim.zero_grad(set_to_none=True)
        cost_loss.backward()
        parameters = self.model._CostQs.parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip_norm)
        self.cost_q_optim.step()
        return cost_loss.detach(), grad_norm.detach()

    def update_cost_q_by_task(self, task_batches):
        losses = {}
        for task, batch in task_batches.items():
            obs, action, _, collapse_cost, _, episode_end, next_obs = self._unpack_batch(batch)
            losses[task] = self._cost_q_loss(
                obs, action, collapse_cost, episode_end, next_obs, task=task
            )
        cost_loss = torch.stack(list(losses.values())).mean()
        self.cost_q_optim.zero_grad(set_to_none=True)
        cost_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._CostQs.parameters(), self.cfg.grad_clip_norm
        )
        self.cost_q_optim.step()
        return (
            cost_loss.detach(),
            grad_norm.detach(),
            {task: loss.detach() for task, loss in losses.items()},
        )

    def update_pi_and_alpha(self, obs):
        pi_loss, info = self._pi_loss(obs)
        log_prob = info["log_prob"]

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        return {
            "pi_loss": pi_loss.detach(),
            "pi_grad_norm": pi_grad_norm.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "entropy": info["entropy"].detach().mean(),
        }
    def _q_loss(self, obs, action, reward, terminated, next_obs):
        td_target = self._td_target(next_obs, reward, terminated)
        qs = self.model.Q(obs, action, return_type="all")
        return F.mse_loss(qs, td_target.unsqueeze(0).expand_as(qs))

    def _actor_safety_values(self, logits_all):
        """Return interpretable risks and the configured actor safety surrogate."""
        risk_all = torch.sigmoid(logits_all)
        worst_logits = logits_all.max(0).values
        risk = torch.sigmoid(worst_logits)
        if self.actor_safety_penalty == "probability":
            surrogate = risk
        elif self.actor_safety_penalty == "softplus_logit":
            surrogate = F.softplus(worst_logits)
        else:
            surrogate = worst_logits
        return risk_all, risk, surrogate

    def _pi_loss(self, obs, task=None):
        action, info = self.model.pi(obs)
        q = self.model.Q(obs, action, return_type="min")
        objective = self.alpha.detach() * info["log_prob"] - q
        if self.safety_enabled:
            if task is None:
                raise ValueError("Constrained actor loss requires an explicit topology task")
            horizon = torch.full(
                (q.numel(),),
                self.active_safety_horizon(task),
                dtype=torch.long,
                device=self.device,
            )
            risk_logits_all = self.model.cost_Q(
                obs, action, horizon, return_type="all", logits=True
            )
            risk_all, risk, safety_surrogate = self._actor_safety_values(
                risk_logits_all
            )
            multiplier = self.safety_lambda(task).detach()
            objective = objective + multiplier * safety_surrogate
            info = dict(info)
            info.update(
                safety_risk=risk,
                safety_risk_all=risk_all,
                safety_risk_logits_all=risk_logits_all,
                safety_surrogate=safety_surrogate,
                safety_lambda=multiplier,
            )
        return objective.mean(), info

    def _constrained_pi_and_alpha_update(self, task_batches):
        losses = []
        task_info = []
        for task, batch in task_batches.items():
            obs, *_ = self._unpack_batch(batch)
            loss, info = self._pi_loss(obs, task=task)
            losses.append(loss)
            task_info.append((task, info))
        pi_loss = torch.stack(losses).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()

        log_prob = torch.cat([info["log_prob"].reshape(-1) for _, info in task_info])
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        metrics = {
            "pi_loss": pi_loss.detach(),
            "pi_grad_norm": pi_grad_norm.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "entropy": torch.stack(
                [info["entropy"].detach().mean() for _, info in task_info]
            ).mean(),
        }
        for task, info in task_info:
            key = self._diagnostic_key(task)
            risks = info["safety_risk"].detach()
            twin = info["safety_risk_all"].detach()
            metrics[f"safety/predicted_risk_mean/{key}"] = risks.mean()
            metrics[f"safety/predicted_risk_max/{key}"] = risks.max()
            metrics[f"safety/predicted_risk_saturated_high_fraction/{key}"] = (
                risks > 0.95
            ).float().mean()
            metrics[f"safety/predicted_risk_saturated_low_fraction/{key}"] = (
                risks < 0.05
            ).float().mean()
            metrics[f"safety/twin_disagreement/{key}"] = (
                twin.max(0).values - twin.min(0).values
            ).mean()
            metrics[f"safety/actor_surrogate_mean/{key}"] = info[
                "safety_surrogate"
            ].detach().mean()
            metrics[f"safety/lambda/{key}"] = info["safety_lambda"]
            metrics[f"safety/budget/{key}"] = self._budget(task)
            metrics.update(self._curriculum_metrics(task, promoted=False))
        return metrics

    def _safety_diagnostics(self, task_batches):
        """Measure cost-critic action coverage and the actor's safety gradient."""
        if not self.safety_enabled:
            return {}
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        parameters = tuple(self.model._pi.parameters())
        metrics = {}
        try:
            for task, batch in task_batches.items():
                obs, replay_action, *_ = self._unpack_batch(batch)
                key = self._diagnostic_key(task)
                num_graphs = int(getattr(obs, "num_graphs", 1))
                horizon = torch.full(
                    (num_graphs,),
                    self.active_safety_horizon(task),
                    dtype=torch.long,
                    device=self.device,
                )

                reward_action, _ = self.model.pi(obs)
                reward_term = -self.model.Q(
                    obs, reward_action, return_type="min"
                ).mean()
                reward_gradients = self._parameter_gradients(reward_term, parameters)

                policy_action, _ = self.model.pi(obs)
                policy_logits_all = self.model.cost_Q(
                    obs,
                    policy_action,
                    horizon,
                    return_type="all",
                    logits=True,
                )
                _, policy_risk, policy_surrogate = self._actor_safety_values(
                    policy_logits_all
                )
                policy_logit = policy_logits_all.max(0).values
                risk_action_gradient = torch.autograd.grad(
                    policy_risk.sum(),
                    policy_action,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                surrogate_action_gradient = torch.autograd.grad(
                    policy_surrogate.sum(),
                    policy_action,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                weighted_constraint = (
                    self.safety_lambda(task).detach() * policy_surrogate.mean()
                )
                constraint_gradients = self._parameter_gradients(
                    weighted_constraint, parameters
                )
                pair = self.gradient_pair_metrics(
                    reward_gradients, constraint_gradients
                )
                reward_norm = pair["first_norm"]
                constraint_norm = pair["second_norm"]

                replay_logits_all = self.model.cost_Q(
                    obs,
                    replay_action,
                    horizon,
                    return_type="all",
                    logits=True,
                )
                replay_logit = replay_logits_all.max(0).values
                replay_risk = torch.sigmoid(replay_logit)
                metrics.update(
                    {
                        f"safety/actor_reward_grad_norm/{key}": reward_norm,
                        f"safety/actor_weighted_constraint_grad_norm/{key}": constraint_norm,
                        f"safety/constraint_to_reward_grad_ratio/{key}": (
                            constraint_norm / (reward_norm + 1e-12)
                        ),
                        f"safety/reward_constraint_grad_cosine/{key}": pair["cosine"],
                        f"safety/policy_action_risk_mean/{key}": policy_risk.detach().mean(),
                        f"safety/replay_action_risk_mean/{key}": replay_risk.detach().mean(),
                        f"safety/policy_action_logit_mean/{key}": policy_logit.detach().mean(),
                        f"safety/replay_action_logit_mean/{key}": replay_logit.detach().mean(),
                        f"safety/policy_minus_replay_logit_mean/{key}": (
                            policy_logit.detach() - replay_logit.detach()
                        ).mean(),
                    }
                )
                if policy_action.shape == replay_action.shape:
                    action_distance = (
                        policy_action.detach() - replay_action.detach()
                    ).reshape(-1, policy_action.shape[-1]).norm(dim=-1)
                    metrics[f"safety/policy_replay_action_l2_mean/{key}"] = (
                        action_distance.mean()
                    )
                for label, gradient in (
                    ("risk", risk_action_gradient),
                    ("surrogate", surrogate_action_gradient),
                ):
                    if gradient is None:
                        continue
                    per_action_norm = gradient.detach().reshape(
                        gradient.shape[0], -1
                    ).norm(dim=-1)
                    metrics.update(
                        {
                            f"safety/{label}_action_grad_norm_mean/{key}": per_action_norm.mean(),
                            f"safety/{label}_action_grad_near_zero_fraction/{key}": (
                                per_action_norm < 1e-6
                            ).float().mean(),
                        }
                    )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return metrics

    @staticmethod
    def _parameter_gradients(loss, parameters):
        parameters = tuple(parameters)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        return tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        )

    @staticmethod
    def _gradient_norm(gradients):
        squared_norm = sum(torch.sum(gradient.double() ** 2) for gradient in gradients)
        return torch.sqrt(squared_norm)

    @staticmethod
    def _gradient_dot(first, second):
        return sum(
            torch.sum(first_gradient.double() * second_gradient.double())
            for first_gradient, second_gradient in zip(first, second)
        )

    @staticmethod
    def _gradient_dot_native(first, second):
        return sum(
            torch.sum(first_gradient * second_gradient)
            for first_gradient, second_gradient in zip(first, second)
        )

    @classmethod
    def pcgrad_project(cls, task_gradients):
        """Return the equally weighted PCGrad update for per-task gradients."""
        task_gradients = tuple(tuple(gradients) for gradients in task_gradients)
        if not task_gradients:
            raise ValueError("PCGrad requires at least one task gradient.")

        gradient_count = len(task_gradients[0])
        if any(len(gradients) != gradient_count for gradients in task_gradients):
            raise ValueError("All PCGrad task gradients must use the same parameters.")

        projected = [
            [gradient.detach().clone() for gradient in gradients]
            for gradients in task_gradients
        ]
        for task_idx, task_gradient in enumerate(projected):
            for other_idx in torch.randperm(len(task_gradients)).tolist():
                if other_idx == task_idx:
                    continue
                other_gradient = task_gradients[other_idx]
                other_norm_squared = cls._gradient_dot_native(
                    other_gradient,
                    other_gradient,
                )
                if float(other_norm_squared) == 0.0:
                    continue
                dot = cls._gradient_dot_native(task_gradient, other_gradient)
                if float(dot) >= 0.0:
                    continue
                coefficient = dot / other_norm_squared
                task_gradient[:] = [
                    gradient
                    - coefficient.to(device=gradient.device, dtype=gradient.dtype)
                    * other.to(device=gradient.device, dtype=gradient.dtype)
                    for gradient, other in zip(task_gradient, other_gradient)
                ]

        return tuple(
            sum(
                (task_gradient[param_idx] for task_gradient in projected),
                start=torch.zeros_like(projected[0][param_idx]),
            )
            / len(projected)
            for param_idx in range(gradient_count)
        )

    @staticmethod
    def _set_parameter_gradients(parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.detach().clone()

    @classmethod
    def gradient_pair_metrics(cls, first, second):
        first_norm = cls._gradient_norm(first)
        second_norm = cls._gradient_norm(second)
        dot = cls._gradient_dot(first, second)
        norm_product = first_norm * second_norm
        cosine = torch.where(
            norm_product > 0,
            dot / norm_product,
            torch.zeros_like(dot),
        )
        norm_denominator = first_norm.square() + second_norm.square()
        norm_agreement = torch.where(
            norm_denominator > 0,
            2.0 * norm_product / norm_denominator,
            torch.zeros_like(norm_denominator),
        )
        return {
            "cosine": cosine.float(),
            "norm_agreement": norm_agreement.float(),
            "first_norm": first_norm.float(),
            "second_norm": second_norm.float(),
        }

    @staticmethod
    def _diagnostic_key(task):
        return re.sub(r"[^0-9a-zA-Z_.-]+", "_", str(task)).strip("_") or "task"

    def _gradient_diagnostics(self, task_batches):
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            gradients = {"critic": {}, "actor": {}}
            if self.safety_enabled:
                gradients["cost_critic"] = {}
            q_parameters = tuple(self.model._Qs.parameters())
            pi_parameters = tuple(self.model._pi.parameters())
            cost_parameters = tuple(self.model._CostQs.parameters()) if self.safety_enabled else ()
            for task, batch in task_batches.items():
                (
                    obs,
                    action,
                    reward,
                    collapse_cost,
                    terminated,
                    episode_end,
                    next_obs,
                ) = self._unpack_batch(batch)
                gradients["critic"][task] = self._parameter_gradients(
                    self._q_loss(obs, action, reward, terminated, next_obs),
                    q_parameters,
                )
                if self.safety_enabled:
                    gradients["cost_critic"][task] = self._parameter_gradients(
                        self._cost_q_loss(
                            obs,
                            action,
                            collapse_cost,
                            episode_end,
                            next_obs,
                            task=task,
                        ),
                        cost_parameters,
                    )
                pi_loss, _ = self._pi_loss(
                    obs, task=task if self.safety_enabled else None
                )
                gradients["actor"][task] = self._parameter_gradients(pi_loss, pi_parameters)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        return self._gradient_metrics(gradients)

    def _gradient_metrics(self, gradients):
        metrics = {}
        for objective, task_gradients in gradients.items():
            task_names = list(task_gradients)
            for task in task_names:
                task_key = self._diagnostic_key(task)
                metrics[f"{objective}/norm/{task_key}"] = self._gradient_norm(
                    task_gradients[task]
                ).float()
            for first_idx, first_task in enumerate(task_names):
                for second_task in task_names[first_idx + 1:]:
                    pair_key = (
                        f"{self._diagnostic_key(first_task)}__"
                        f"{self._diagnostic_key(second_task)}"
                    )
                    pair = self.gradient_pair_metrics(
                        task_gradients[first_task],
                        task_gradients[second_task],
                    )
                    metrics[f"{objective}/cosine/{pair_key}"] = pair["cosine"]
                    metrics[f"{objective}/norm_agreement/{pair_key}"] = pair["norm_agreement"]
        return metrics

    def _pcgrad_q_update(self, task_batches):
        parameters = tuple(self.model._Qs.parameters())
        losses = []
        task_gradients = {}
        for task, batch in task_batches.items():
            obs, action, reward, _, terminated, _, next_obs = self._unpack_batch(batch)
            loss = self._q_loss(obs, action, reward, terminated, next_obs)
            losses.append(loss)
            task_gradients[task] = self._parameter_gradients(loss, parameters)

        self.q_optim.zero_grad(set_to_none=True)
        self._set_parameter_gradients(
            parameters,
            self.pcgrad_project(task_gradients.values()),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip_norm)
        self.q_optim.step()
        return (
            torch.stack([loss.detach() for loss in losses]).mean(),
            grad_norm.detach(),
            task_gradients,
        )

    def _pcgrad_cost_q_update(self, task_batches):
        parameters = tuple(self.model._CostQs.parameters())
        losses = []
        task_gradients = {}
        for task, batch in task_batches.items():
            obs, action, _, collapse_cost, _, episode_end, next_obs = self._unpack_batch(batch)
            loss = self._cost_q_loss(
                obs, action, collapse_cost, episode_end, next_obs, task=task
            )
            losses.append(loss)
            task_gradients[task] = self._parameter_gradients(loss, parameters)
        self.cost_q_optim.zero_grad(set_to_none=True)
        self._set_parameter_gradients(
            parameters, self.pcgrad_project(task_gradients.values())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip_norm)
        self.cost_q_optim.step()
        return (
            torch.stack([loss.detach() for loss in losses]).mean(),
            grad_norm.detach(),
            task_gradients,
            {
                task: loss.detach()
                for task, loss in zip(task_batches, losses)
            },
        )

    def _pcgrad_pi_and_alpha_update(self, task_batches):
        parameters = tuple(self.model._pi.parameters())
        losses = []
        task_gradients = {}
        task_info = []
        for task, batch in task_batches.items():
            pi_loss, info = self._pi_loss(
                batch[0], task=task if self.safety_enabled else None
            )
            losses.append(pi_loss)
            task_info.append(info)
            task_gradients[task] = self._parameter_gradients(pi_loss, parameters)

        self.pi_optim.zero_grad(set_to_none=True)
        self._set_parameter_gradients(
            parameters,
            self.pcgrad_project(task_gradients.values()),
        )
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip_norm)
        self.pi_optim.step()

        log_prob = torch.cat([info["log_prob"].reshape(-1) for info in task_info])
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        entropy = torch.stack(
            [info["entropy"].detach().mean() for info in task_info]
        ).mean()
        metrics = {
                "pi_loss": torch.stack([loss.detach() for loss in losses]).mean(),
                "pi_grad_norm": pi_grad_norm.detach(),
                "alpha_loss": alpha_loss.detach(),
                "alpha": self.alpha.detach(),
                "entropy": entropy,
            }
        if self.safety_enabled:
            for (task, _), info in zip(task_batches.items(), task_info):
                key = self._diagnostic_key(task)
                risks = info["safety_risk"].detach()
                twin = info["safety_risk_all"].detach()
                metrics[f"safety/predicted_risk_mean/{key}"] = risks.mean()
                metrics[f"safety/predicted_risk_max/{key}"] = risks.max()
                metrics[f"safety/predicted_risk_saturated_high_fraction/{key}"] = (
                    risks > 0.95
                ).float().mean()
                metrics[f"safety/predicted_risk_saturated_low_fraction/{key}"] = (
                    risks < 0.05
                ).float().mean()
                metrics[f"safety/twin_disagreement/{key}"] = (
                    twin.max(0).values - twin.min(0).values
                ).mean()
                metrics[f"safety/actor_surrogate_mean/{key}"] = info[
                    "safety_surrogate"
                ].detach().mean()
                metrics[f"safety/lambda/{key}"] = info["safety_lambda"]
                metrics[f"safety/budget/{key}"] = self._budget(task)
                metrics.update(self._curriculum_metrics(task, promoted=False))
        return metrics, task_gradients

    def update(
        self,
        buffer,
        compute_diagnostics=False,
        compute_safety_diagnostics=False,
        performance_profiler=None,
    ):
        pcgrad_enabled = bool(getattr(self.cfg, "pcgrad", False))
        safety_enabled = bool(getattr(self, "safety_enabled", False))
        combined_batch = None
        task_batches = None
        sampling_phase = (
            performance_profiler.phase("replay_sampling")
            if performance_profiler is not None
            else contextlib.nullcontext()
        )
        with sampling_phase:
            if getattr(buffer, "supports_replay_profiling", False):
                if pcgrad_enabled:
                    task_batches = buffer.sample_task_batches(
                        performance_profiler=performance_profiler
                    )
                elif compute_diagnostics or safety_enabled:
                    replay_batch = buffer.sample_with_tasks(
                        performance_profiler=performance_profiler
                    )
                    combined_batch = replay_batch.combined
                    task_batches = replay_batch.by_task
                else:
                    combined_batch = buffer.sample(
                        performance_profiler=performance_profiler
                    )
            elif hasattr(buffer, "sample_with_tasks"):
                replay_batch = buffer.sample_with_tasks()
                combined_batch = replay_batch.combined
                task_batches = replay_batch.by_task
            else:
                if pcgrad_enabled or safety_enabled:
                    raise ValueError(
                        "PCGrad and constrained safety require replay batches grouped by task."
                    )
                combined_batch = buffer.sample()
        if combined_batch is not None:
            (
                obs,
                action,
                reward,
                collapse_cost,
                terminated,
                episode_end,
                next_obs,
            ) = GNNSAC._unpack_batch(combined_batch)
       # if self.device.type == 'cuda':
       #     torch.compiler.cudagraph_mark_step_begin()
        
        optimization_phase = (
            performance_profiler.phase("optimization")
            if performance_profiler is not None
            else contextlib.nullcontext()
        )
        with optimization_phase:
            self.model.train()
            safety_diagnostics = (
                self._safety_diagnostics(task_batches)
                if compute_safety_diagnostics and safety_enabled
                else None
            )
            if pcgrad_enabled:
                q_loss, q_grad_norm, q_task_gradients = self._pcgrad_q_update(task_batches)
                if safety_enabled:
                    (
                        cost_loss,
                        cost_grad_norm,
                        cost_task_gradients,
                        cost_losses_by_task,
                    ) = self._pcgrad_cost_q_update(task_batches)
                pi_info, pi_task_gradients = self._pcgrad_pi_and_alpha_update(task_batches)
                diagnostic_gradients = {
                    "critic": q_task_gradients,
                    "actor": pi_task_gradients,
                }
                if safety_enabled:
                    diagnostic_gradients["cost_critic"] = cost_task_gradients
                diagnostics = (
                    self._gradient_metrics(diagnostic_gradients)
                    if compute_diagnostics
                    else None
                )
            else:
                diagnostics = (
                    self._gradient_diagnostics(task_batches)
                    if compute_diagnostics and task_batches is not None
                    else None
                )
                q_loss, q_grad_norm = self.update_q(obs, action, reward, terminated, next_obs)
                if safety_enabled:
                    (
                        cost_loss,
                        cost_grad_norm,
                        cost_losses_by_task,
                    ) = self.update_cost_q_by_task(
                        task_batches
                    )
                    pi_info = self._constrained_pi_and_alpha_update(task_batches)
                else:
                    pi_info = self.update_pi_and_alpha(obs)
            self.model.soft_update_target_Q()
            self.model.eval()

        info = {
            "value_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }
        if safety_enabled:
            info.update(
                **{
                    "safety/cost_value_loss": cost_loss,
                    "safety/cost_q_grad_norm": cost_grad_norm,
                }
            )
            for task, task_loss in cost_losses_by_task.items():
                info[f"safety/cost_loss/{self._diagnostic_key(task)}"] = task_loss
            info.update(self._last_cost_horizon_metrics)
            info.update(self._last_cost_target_metrics)
            if safety_diagnostics:
                info.update(safety_diagnostics)
        info.update(pi_info)
        if diagnostics is not None:
            info["gradient_diagnostics"] = diagnostics
        return info
