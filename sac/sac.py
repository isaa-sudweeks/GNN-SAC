import contextlib

import torch
import torch.nn.functional as F

from common.actor_critic import ActorCritic
from common.finite_checks import (
    NonFiniteTrainingError,
    require_finite,
    require_finite_gradients,
    require_finite_optimizer,
)


class SAC(torch.nn.Module):
    """Soft Actor-Critic agent for continuous actions."""

    def __init__(self, cfg):
        super().__init__()
        if bool(getattr(cfg, "pcgrad", False)):
            raise ValueError(
                "pcgrad=true requires sac_backend=gnn and task-aware GNN replay batches."
            )
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cuda"))
        self.model = ActorCritic(cfg).to(self.device)
        self._q_parameters = tuple(self.model._Qs.parameters())
        self._actor_parameters = tuple(self.model._pi.parameters())
        capturable = self.device.type in {"cuda", "xpu", "hpu", "privateuseone", "xla"}

        self.q_optim = torch.optim.Adam(self.model._Qs.parameters(), lr=self.cfg.lr, capturable=capturable)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable)

        init_alpha = float(getattr(cfg, "entropy_coef", 0.2))
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(init_alpha, device=self.device)))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr, capturable=capturable)
        target_entropy = getattr(cfg, "target_entropy", "auto")
        self.target_entropy = -float(cfg.action_dim) if target_entropy == "auto" else float(target_entropy)

        self.model.eval()
        self.discount = float(getattr(cfg, "discount", self._get_discount(cfg.episode_length)))

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        print("Target entropy:", self.target_entropy)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def finite_checks_enabled(self):
        return bool(getattr(self.cfg, "finite_checks", True))

    def _require_finite(self, label, value):
        if self.finite_checks_enabled:
            require_finite(label, value)

    def _validate_finite_training_state(self, label):
        self._require_finite(f"{label} model state", self.model.state_dict())
        self._validate_finite_temperature(label)
        if self.finite_checks_enabled:
            require_finite_optimizer(f"{label} critic optimizer", self.q_optim)
            require_finite_optimizer(f"{label} actor optimizer", self.pi_optim)
            require_finite_optimizer(f"{label} alpha optimizer", self.alpha_optim)

    def _validate_finite_temperature(self, label):
        self._require_finite(
            f"{label} entropy temperature",
            {"log_alpha": self.log_alpha, "alpha": self.alpha},
        )

    def _safe_action(self, action):
        if self.finite_checks_enabled:
            require_finite("policy action", action)
            return action.clamp(-1, 1)
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        self._require_finite("agent save model state", self.model.state_dict())
        self._validate_finite_temperature("agent save")
        torch.save(
            {
                "model": self.model.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            fp,
        )

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
        if isinstance(state_dict, dict) and "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self._require_finite("loaded checkpoint model state", self.model.state_dict())
        self._validate_finite_temperature("loaded checkpoint")

    def training_state_dict(self):
        self._validate_finite_training_state("trainer checkpoint save")
        return {
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "q_optim": self.q_optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_training_state_dict(self, state_dict):
        self.load(state_dict)
        if "q_optim" in state_dict:
            self.q_optim.load_state_dict(state_dict["q_optim"])
        if "pi_optim" in state_dict:
            self.pi_optim.load_state_dict(state_dict["pi_optim"])
        if "alpha_optim" in state_dict:
            self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        self._validate_finite_training_state("loaded trainer checkpoint")

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        action, info = self.model.pi(obs)
        if eval_mode:
            action = info["mean"]
        return self._safe_action(action[0]).cpu()

    @torch.no_grad()
    def _td_target(self, next_obs, reward, terminated):
        next_action, next_info = self.model.pi(next_obs)
        target_q = self.model.Q(next_obs, next_action, return_type="min", target=True)
        target_v = target_q - self.alpha.detach() * next_info["log_prob"]
        return reward + self.discount * (1.0 - terminated) * target_v

    def update_q(self, obs, action, reward, terminated, next_obs):
        td_target = self._td_target(next_obs, reward, terminated)
        qs = self.model.Q(obs, action, return_type="all")
        q_loss = F.mse_loss(qs, td_target.unsqueeze(0).expand_as(qs))
        self._require_finite("critic loss", q_loss)

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.finite_checks_enabled:
            require_finite_gradients(
                "critic gradients before clipping", self.model._Qs.named_parameters()
            )
        try:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._q_parameters,
                self.cfg.grad_clip_norm,
                error_if_nonfinite=self.finite_checks_enabled,
            )
        except RuntimeError as exc:
            if not self.finite_checks_enabled:
                raise
            raise NonFiniteTrainingError("critic gradient norm became non-finite.") from exc
        self.q_optim.step()
        self._require_finite("critic parameters after optimizer step", self._q_parameters)
        return q_loss.detach(), q_grad_norm.detach()

    def update_pi_and_alpha(self, obs):
        action, info = self.model.pi(obs)
        q = self.model.Q(obs, action, return_type="min")
        log_prob = info["log_prob"]
        pi_loss = (self.alpha.detach() * log_prob - q).mean()
        self._require_finite("actor loss", pi_loss)
        self._require_finite("actor distribution statistics", info)

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        if self.finite_checks_enabled:
            require_finite_gradients(
                "actor gradients before clipping", self.model._pi.named_parameters()
            )
        try:
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._actor_parameters,
                self.cfg.grad_clip_norm,
                error_if_nonfinite=self.finite_checks_enabled,
            )
        except RuntimeError as exc:
            if not self.finite_checks_enabled:
                raise
            raise NonFiniteTrainingError("actor gradient norm became non-finite.") from exc
        self.pi_optim.step()
        self._require_finite("actor parameters after optimizer step", self._actor_parameters)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self._require_finite("entropy temperature loss", alpha_loss)
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        if self.finite_checks_enabled:
            require_finite_gradients(
                "entropy temperature gradients", (("log_alpha", self.log_alpha),)
            )
        self.alpha_optim.step()
        self._validate_finite_temperature("after optimizer step")

        return {
            "pi_loss": pi_loss.detach(),
            "pi_grad_norm": pi_grad_norm.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "entropy": info["entropy"].detach().mean(),
        }

    def update(self, buffer, performance_profiler=None):
        sampling_phase = (
            performance_profiler.phase("replay_sampling")
            if performance_profiler is not None
            else contextlib.nullcontext()
        )
        with sampling_phase:
            obs, action, reward, terminated, next_obs = buffer.sample()
        self._require_finite(
            "replay batch", (obs, action, reward, terminated, next_obs)
        )

        optimization_phase = (
            performance_profiler.phase("optimization")
            if performance_profiler is not None
            else contextlib.nullcontext()
        )
        with optimization_phase:
            if self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            self.model.train()
            q_loss, q_grad_norm = self.update_q(obs, action, reward, terminated, next_obs)
            pi_info = self.update_pi_and_alpha(obs)
            self.model.soft_update_target_Q()
            self._require_finite(
                "target critic after soft update", self.model._target_Qs.state_dict()
            )
            self.model.eval()

        info = {
            "value_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }
        info.update(pi_info)
        return info
