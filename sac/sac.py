import torch
import torch.nn.functional as F

from common.actor_critic import ActorCritic


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

    def _safe_action(self, action):
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
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

    def training_state_dict(self):
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

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._Qs.parameters(), self.cfg.grad_clip_norm)
        self.q_optim.step()
        return q_loss.detach(), q_grad_norm.detach()

    def update_pi_and_alpha(self, obs):
        action, info = self.model.pi(obs)
        q = self.model.Q(obs, action, return_type="min")
        log_prob = info["log_prob"]
        pi_loss = (self.alpha.detach() * log_prob - q).mean()

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

    def update(self, buffer):
        obs, action, reward, terminated, next_obs = buffer.sample()
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()

        self.model.train()
        q_loss, q_grad_norm = self.update_q(obs, action, reward, terminated, next_obs)
        pi_info = self.update_pi_and_alpha(obs)
        self.model.soft_update_target_Q()
        self.model.eval()

        info = {
            "value_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }
        info.update(pi_info)
        return info
