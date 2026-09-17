from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import unittest

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.finite_checks import NonFiniteTrainingError, require_finite
from common.gnn_buffer import GNNBuffer
from gnn_sac import GNNSAC


def agent_cfg(**overrides):
    values = dict(
        device="cpu",
        obs_dim=3,
        embedding_dim=16,
        mlp_dim=16,
        dropout=0.0,
        action_dim=1,
        Q_output_dim=16,
        head_hidden_dims=[16],
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy="auto",
        num_policy_actions=3,
        episode_length=100,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        tau=0.005,
        grad_clip_norm=10.0,
        pcgrad=False,
        finite_checks=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def graph(node_count=3, *, nonfinite=False):
    nodes = torch.arange(node_count, dtype=torch.long)
    edge_index = torch.stack(
        [
            torch.cat([nodes, nodes.roll(-1)]),
            torch.cat([nodes.roll(-1), nodes]),
        ]
    )
    x = torch.ones(node_count, 3)
    if nonfinite:
        x[1, 2] = float("nan")
    return Data(x=x, edge_index=edge_index)


class _FiniteForwardNaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.shape = value.shape
        return value.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, float("nan"), device=grad_output.device)


class FiniteTrainingCheckTest(unittest.TestCase):
    def test_nested_replay_tensor_reports_exact_path(self):
        with self.assertRaisesRegex(
            NonFiniteTrainingError,
            r"replay batch\.task\.obs\.x contains 1 non-finite",
        ):
            require_finite(
                "replay batch",
                {"task": {"obs": graph(nonfinite=True)}},
            )

    def test_replay_check_precedes_any_optimizer_step(self):
        agent = GNNSAC(agent_cfg())

        class Buffer:
            @staticmethod
            def sample():
                return (
                    graph(nonfinite=True),
                    torch.zeros(3, 1),
                    torch.zeros(1),
                    torch.zeros(1),
                    graph(),
                )

        with mock.patch.object(agent.q_optim, "step") as q_step:
            with self.assertRaisesRegex(NonFiniteTrainingError, "replay batch"):
                agent.update(Buffer())
        q_step.assert_not_called()

    def test_nonfinite_transition_is_rejected_before_replay_insertion(self):
        config = agent_cfg(
            task="truss-graph",
            tasks=["truss-graph"],
            multitask=False,
            mujoco_backend="mujoco",
            truss_topologies=None,
            buffer_size=8,
            batch_size=2,
            steps=8,
        )
        buffer = GNNBuffer(config)
        action = torch.zeros(1, 3, 1)
        trajectory = [
            {
                "obs": graph(),
                "action": torch.full_like(action, float("nan")),
                "reward": torch.tensor([float("nan")]),
                "terminated": torch.tensor([float("nan")]),
            },
            {
                "obs": graph(nonfinite=True),
                "action": action,
                "reward": torch.zeros(1),
                "terminated": torch.zeros(1),
            },
        ]
        with self.assertRaisesRegex(
            NonFiniteTrainingError,
            r"replay insertion\.next_obs\[0\]\.x",
        ):
            buffer.add(trajectory)
        self.assertEqual(buffer.size, 0)

    def test_nonfinite_loss_is_rejected_before_backward(self):
        agent = GNNSAC(agent_cfg())
        parameter = agent._q_parameters[0]
        bad_loss = parameter.sum() * float("nan")
        with mock.patch.object(agent, "_q_loss", return_value=bad_loss), mock.patch.object(
            agent.q_optim, "step"
        ) as q_step:
            with self.assertRaisesRegex(NonFiniteTrainingError, "critic loss"):
                agent.update_q(None, None, None, None, None)
        q_step.assert_not_called()

    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self):
        agent = GNNSAC(agent_cfg())
        parameter = agent._q_parameters[0]
        loss = _FiniteForwardNaNBackward.apply(parameter)
        with mock.patch.object(agent, "_q_loss", return_value=loss), mock.patch.object(
            agent.q_optim, "step"
        ) as q_step:
            with self.assertRaisesRegex(
                NonFiniteTrainingError, "critic gradients before clipping"
            ):
                agent.update_q(None, None, None, None, None)
        q_step.assert_not_called()

    def test_nonfinite_policy_action_is_not_silently_zeroed(self):
        agent = GNNSAC(agent_cfg())
        observation = graph()
        action = torch.full((observation.num_nodes, 1), float("nan"))
        with mock.patch.object(agent.model, "pi", return_value=(action, {})):
            with self.assertRaisesRegex(NonFiniteTrainingError, "policy action"):
                agent.act_batch([observation])

    def test_checkpoint_save_rejects_nonfinite_parameters_and_optimizer_state(self):
        agent = GNNSAC(agent_cfg())
        parameter = agent._q_parameters[0]
        with torch.no_grad():
            parameter.view(-1)[0] = float("inf")
        with self.assertRaisesRegex(NonFiniteTrainingError, "model state"):
            agent.training_state_dict()

        agent = GNNSAC(agent_cfg())
        parameter = agent._q_parameters[0]
        agent.q_optim.state[parameter]["exp_avg"] = torch.full_like(
            parameter, float("nan")
        )
        with self.assertRaisesRegex(NonFiniteTrainingError, "critic optimizer"):
            agent.training_state_dict()

    def test_checkpoint_load_rejects_nonfinite_model_state(self):
        source = GNNSAC(agent_cfg())
        state = deepcopy(source.training_state_dict())
        first_key = next(iter(state["model"]))
        state["model"][first_key].view(-1)[0] = float("nan")

        restored = GNNSAC(agent_cfg())
        with self.assertRaisesRegex(NonFiniteTrainingError, "loaded checkpoint"):
            restored.load_training_state_dict(state)

    def test_explicit_disable_preserves_legacy_action_sanitization(self):
        agent = GNNSAC(agent_cfg(finite_checks=False))
        action = torch.tensor([float("nan"), float("inf"), float("-inf")])
        torch.testing.assert_close(
            agent._safe_action(action),
            torch.tensor([0.0, 1.0, -1.0]),
        )


if __name__ == "__main__":
    unittest.main()
