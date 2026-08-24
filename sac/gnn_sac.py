import contextlib
from collections.abc import Sequence
import re
import warnings

import torch 
import torch.nn.functional as F 
from torch_geometric.data import Batch, Data

from common.gnn_actor_critic import GNNActorCritic 
from common.graph_transforms import (
    graph_feature_flags,
    graph_feature_schema,
    policy_action_mask,
    prepare_graph,
)

class GNNSAC(torch.nn.Module):
    """
    Soft Actor-Critic for GNN-based control.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cuda"))
        self.model = self._make_model(cfg).to(self.device)
        capturable = self.device.type in {"cuda", "xpu", "hpu", "privateuseone", "xla"}

        self.q_optim = torch.optim.Adam(self.model._Qs.parameters(), lr=self.cfg.lr, capturable=capturable)
        self.pi_optim = torch.optim.Adam(self.model.actor_parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable)

        init_alpha = float(getattr(self.cfg, "entropy_coef", 0.2))
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(init_alpha, device=self.device)))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr, capturable=capturable)
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

    def _make_model(self, cfg):
        """Construct the actor-critic used by this graph-batch SAC backend."""
        return GNNActorCritic(cfg)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _safe_action(self, action):
        """
        TODO: I think it would be benefitial to add some sort of checking to make sure that the GNN stuff is right for the actions.

        """
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def project_action(self, obs, action):
        """Zero passive graph-action rows after seed sampling and action noise."""
        mask = policy_action_mask(obs)
        if isinstance(action, torch.Tensor):
            projected = action.clone()
            projected[~mask.to(projected.device)] = 0
            return projected
        projected = torch.as_tensor(action).clone()
        projected[~mask.cpu()] = 0
        return projected.numpy()

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        torch.save(
            {
                "model": self.model.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "graph_feature_schema": graph_feature_schema(self.cfg),
            },
            fp,
        )
    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device, weights_only=False)
        self._validate_graph_feature_schema(state_dict)
        self.model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
        if isinstance(state_dict, dict) and "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))

    def training_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "q_optim": self.q_optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "graph_feature_schema": graph_feature_schema(self.cfg),
        }

    def _validate_graph_feature_schema(self, state_dict):
        if not isinstance(state_dict, dict) or "model" not in state_dict:
            saved_schema = None
        else:
            saved_schema = state_dict.get("graph_feature_schema")
        expected_schema = graph_feature_schema(self.cfg)
        features_enabled = any(
            expected_schema[name]
            for name in ("node_roles", "edge_roles", "edge_distance")
        )
        if saved_schema is None:
            if features_enabled:
                raise ValueError(
                    "Checkpoint has no graph feature schema, but configurable graph features are enabled. "
                    "Use a checkpoint trained with the same graph_features configuration."
                )
            return
        if saved_schema != expected_schema:
            raise ValueError(
                "Checkpoint graph feature schema does not match this run: "
                f"saved={saved_schema}, current={expected_schema}."
            )

    def load_training_state_dict(self, state_dict):
        self.load(state_dict)
        if "q_optim" in state_dict:
            self.q_optim.load_state_dict(state_dict["q_optim"])
        if "pi_optim" in state_dict:
            self._load_actor_optimizer_state(state_dict["pi_optim"])
        if "alpha_optim" in state_dict:
            self.alpha_optim.load_state_dict(state_dict["alpha_optim"])

    def _load_actor_optimizer_state(self, saved_state):
        """Load actor Adam state, upgrading checkpoints that omitted the GNN head."""
        try:
            self.pi_optim.load_state_dict(saved_state)
            return
        except ValueError as exc:
            saved_groups = list(saved_state.get("param_groups", []))
            current_state = self.pi_optim.state_dict()
            current_groups = list(current_state.get("param_groups", []))
            legacy_parameter_count = len(tuple(self.model._pi.parameters()))
            is_legacy_gnn_state = (
                hasattr(self.model, "_action_head")
                and len(saved_groups) == len(current_groups) == 1
                and len(saved_groups[0].get("params", [])) == legacy_parameter_count
                and len(current_groups[0].get("params", [])) > legacy_parameter_count
            )
            if not is_legacy_gnn_state:
                raise exc

        saved_group = saved_groups[0]
        current_group = current_groups[0]
        saved_ids = list(saved_group["params"])
        current_ids = list(current_group["params"])
        upgraded_group = dict(current_group)
        upgraded_group.update(
            {key: value for key, value in saved_group.items() if key != "params"}
        )
        upgraded_group["params"] = current_ids
        upgraded_state = {
            current_ids[index]: saved_state.get("state", {}).get(saved_id, {})
            for index, saved_id in enumerate(saved_ids)
        }
        self.pi_optim.load_state_dict(
            {"state": upgraded_state, "param_groups": [upgraded_group]}
        )
        warnings.warn(
            "Loaded a legacy GNN actor optimizer checkpoint that omitted the action head; "
            "the restored encoder state was kept and fresh Adam state was initialized for "
            "the action head.",
            RuntimeWarning,
        )

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
        feature_flags = graph_feature_flags(self.cfg)
        prepared = [
            prepare_graph(obs, use_virtual_node=use_virtual_node, **feature_flags)
            for obs in observations
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

    def update_pi_and_alpha(self, obs):
        pi_loss, info = self._pi_loss(obs)
        log_prob = info["log_prob"]

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.actor_parameters(), self.cfg.grad_clip_norm
        )
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

    def _pi_loss(self, obs):
        action, info = self.model.pi(obs)
        q = self.model.Q(obs, action, return_type="min")
        return (self.alpha.detach() * info["log_prob"] - q).mean(), info

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
            q_parameters = tuple(self.model._Qs.parameters())
            pi_parameters = tuple(self.model.actor_parameters())
            for task, batch in task_batches.items():
                obs, action, reward, terminated, next_obs = batch
                gradients["critic"][task] = self._parameter_gradients(
                    self._q_loss(obs, action, reward, terminated, next_obs),
                    q_parameters,
                )
                pi_loss, _ = self._pi_loss(obs)
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
            obs, action, reward, terminated, next_obs = batch
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

    def _pcgrad_pi_and_alpha_update(self, task_batches):
        parameters = tuple(self.model.actor_parameters())
        losses = []
        task_gradients = {}
        task_info = []
        for task, batch in task_batches.items():
            pi_loss, info = self._pi_loss(batch[0])
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
        return (
            {
                "pi_loss": torch.stack([loss.detach() for loss in losses]).mean(),
                "pi_grad_norm": pi_grad_norm.detach(),
                "alpha_loss": alpha_loss.detach(),
                "alpha": self.alpha.detach(),
                "entropy": entropy,
            },
            task_gradients,
        )

    def update(self, buffer, compute_diagnostics=False, performance_profiler=None):
        pcgrad_enabled = bool(getattr(self.cfg, "pcgrad", False))
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
                    obs = action = reward = terminated = next_obs = None
                elif compute_diagnostics:
                    replay_batch = buffer.sample_with_tasks(
                        performance_profiler=performance_profiler
                    )
                    obs, action, reward, terminated, next_obs = replay_batch.combined
                    task_batches = replay_batch.by_task
                else:
                    obs, action, reward, terminated, next_obs = buffer.sample(
                        performance_profiler=performance_profiler
                    )
                    task_batches = None
            elif hasattr(buffer, "sample_with_tasks"):
                replay_batch = buffer.sample_with_tasks()
                obs, action, reward, terminated, next_obs = replay_batch.combined
                task_batches = replay_batch.by_task
            else:
                if pcgrad_enabled:
                    raise ValueError(
                        "pcgrad=true requires replay batches grouped by task via sample_with_tasks()."
                    )
                obs, action, reward, terminated, next_obs = buffer.sample()
                task_batches = None
       # if self.device.type == 'cuda':
       #     torch.compiler.cudagraph_mark_step_begin()
        
        optimization_phase = (
            performance_profiler.phase("optimization")
            if performance_profiler is not None
            else contextlib.nullcontext()
        )
        with optimization_phase:
            self.model.train()
            if pcgrad_enabled:
                q_loss, q_grad_norm, q_task_gradients = self._pcgrad_q_update(task_batches)
                pi_info, pi_task_gradients = self._pcgrad_pi_and_alpha_update(task_batches)
                diagnostics = (
                    self._gradient_metrics(
                        {"critic": q_task_gradients, "actor": pi_task_gradients}
                    )
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
                pi_info = self.update_pi_and_alpha(obs)
            self.model.soft_update_target_Q()
            self.model.eval()

        info = {
            "value_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }
        info.update(pi_info)
        if diagnostics is not None:
            info["gradient_diagnostics"] = diagnostics
        return info
