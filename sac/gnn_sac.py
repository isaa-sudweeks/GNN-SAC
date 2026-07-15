from collections.abc import Sequence
import re

import torch 
import torch.nn.functional as F 
from torch_geometric.data import Batch, Data

from common.gnn_actor_critic import GNNActorCritic 
from common.graph_transforms import prepare_graph

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
        if action.size(0) != sum(node_counts):
            raise RuntimeError(
                f"Actor produced {action.size(0)} node actions for {sum(node_counts)} observation nodes"
            )
        return list(action.split(node_counts, dim=0))

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

    @classmethod
    def gradient_pair_metrics(cls, first, second):
        first_norm = cls._gradient_norm(first)
        second_norm = cls._gradient_norm(second)
        dot = sum(
            torch.sum(first_gradient.double() * second_gradient.double())
            for first_gradient, second_gradient in zip(first, second)
        )
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
            pi_parameters = tuple(self.model._pi.parameters())
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

        metrics = {}
        task_names = list(task_batches)
        for objective, task_gradients in gradients.items():
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

    def update(self, buffer, compute_diagnostics=False):
        if hasattr(buffer, "sample_with_tasks"):
            replay_batch = buffer.sample_with_tasks()
            obs, action, reward, terminated, next_obs = replay_batch.combined
            task_batches = replay_batch.by_task
        else:
            obs, action, reward, terminated, next_obs = buffer.sample()
            task_batches = None
       # if self.device.type == 'cuda':
       #     torch.compiler.cudagraph_mark_step_begin()
        
        self.model.train()
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
