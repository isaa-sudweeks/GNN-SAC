import torch

from common.padded_mlp_actor_critic import PaddedMLPActorCritic
from gnn_sac import GNNSAC


class PaddedMLPSAC(GNNSAC):
    """Graph-batch SAC whose policy and critics are fixed-width flat MLPs."""

    SCHEMA_KEY = "padded_mlp_schema"

    def _make_model(self, cfg):
        return PaddedMLPActorCritic(cfg)

    def _schema(self):
        return {
            "version": 1,
            "max_nodes": int(self.cfg.padded_mlp_max_nodes),
            "node_feature_dim": int(self.cfg.obs_dim),
            "node_action_dim": int(self.cfg.node_action_dim),
            "physical_mask": True,
            "action_mask": True,
            "rigidity": True,
        }

    def save(self, fp):
        torch.save(
            {
                "model": self.model.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                self.SCHEMA_KEY: self._schema(),
            },
            fp,
        )

    def training_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "q_optim": self.q_optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
            self.SCHEMA_KEY: self._schema(),
        }

    def _validate_graph_feature_schema(self, state_dict):
        if not isinstance(state_dict, dict) or "model" not in state_dict:
            saved_schema = None
        else:
            saved_schema = state_dict.get(self.SCHEMA_KEY)
        expected_schema = self._schema()
        if saved_schema != expected_schema:
            raise ValueError(
                "Checkpoint padded MLP schema does not match this run: "
                f"saved={saved_schema}, current={expected_schema}."
            )
