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

from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer


def agent_cfg():
    return SimpleNamespace(
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
        num_policy_actions=6,
        episode_length=100,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        tau=0.005,
        grad_clip_norm=10.0,
    )


def graph(num_nodes: int, offset: float = 0.0) -> Data:
    source = torch.arange(num_nodes, dtype=torch.long)
    target = source.roll(-1)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])],
        dim=0,
    )
    x = torch.arange(num_nodes * 3, dtype=torch.float32).view(num_nodes, 3)
    return Data(x=x / max(num_nodes, 1) + offset, edge_index=edge_index)


class GNNBatchedInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = GNNSAC(agent_cfg())

    def test_empty_batch_does_not_invoke_actor(self):
        with mock.patch.object(self.agent.model, "pi") as pi, mock.patch.object(
            self.agent.model, "pi_mean"
        ) as pi_mean:
            self.assertEqual(self.agent.act_batch([]), [])
        pi.assert_not_called()
        pi_mean.assert_not_called()

    def test_mixed_size_deterministic_batch_matches_single_actions(self):
        observations = [graph(3), graph(5, offset=0.5), graph(4, offset=-0.25)]

        with mock.patch.object(
            self.agent.model,
            "pi_mean",
            wraps=self.agent.model.pi_mean,
        ) as pi_mean:
            batched = self.agent.act_batch(observations, eval_mode=True)
        pi_mean.assert_called_once()
        serialized = [self.agent.act(obs, eval_mode=True) for obs in observations]

        self.assertEqual(len(batched), len(observations))
        for action, expected, obs in zip(batched, serialized, observations):
            self.assertEqual(action.shape, (obs.num_nodes, 1))
            self.assertEqual(action.device.type, "cpu")
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue((action >= -1.0).all())
            self.assertTrue((action <= 1.0).all())
            torch.testing.assert_close(action, expected)

    def test_deterministic_batch_skips_stochastic_policy_path(self):
        observations = [graph(3), graph(4)]
        with mock.patch.object(
            self.agent.model,
            "pi",
            side_effect=AssertionError("stochastic policy path should not run"),
        ), mock.patch("torch.randn_like", side_effect=AssertionError("noise should not be sampled")):
            actions = self.agent.act_batch(observations, eval_mode=True)

        self.assertEqual([action.shape for action in actions], [(3, 1), (4, 1)])

    def test_stochastic_batch_returns_independent_valid_actions(self):
        observation = graph(5)
        first, second = self.agent.act_batch([observation, observation])

        self.assertEqual(first.shape, (5, 1))
        self.assertEqual(second.shape, (5, 1))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        self.assertFalse(torch.equal(first, second))


class MultiEnvActionSelectionTest(unittest.TestCase):
    def make_trainer(self, agent, step=3, seed_steps=1):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.agent = agent
        trainer.env = mock.Mock()
        trainer.cfg = SimpleNamespace(seed_steps=seed_steps, domain_randomization=False)
        trainer._step = step
        return trainer

    def test_partial_vector_step_uses_one_batched_call(self):
        class BatchedAgent:
            def __init__(self):
                self.calls = []

            def act_batch(self, observations):
                self.calls.append(observations)
                return [torch.full((obs.num_nodes, 1), float(i)) for i, obs in enumerate(observations)]

        agent = BatchedAgent()
        trainer = self.make_trainer(agent)
        observations = [graph(3), graph(4), graph(5)]
        episode_tds = [[object()], [object(), object()], [object()]]

        actions = trainer._select_multi_env_actions(observations, episode_tds, [0, 2])

        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.calls[0], [observations[0], observations[2]])
        self.assertEqual([action.shape for action in actions], [(3, 1), (5, 1)])

    def test_agent_without_batch_api_uses_serialized_fallback(self):
        class SerializedAgent:
            def __init__(self):
                self.calls = []

            def act(self, observation, t0=False):
                self.calls.append((observation, t0))
                return torch.zeros(observation.num_nodes, 1)

        agent = SerializedAgent()
        trainer = self.make_trainer(agent)
        observations = [graph(3), graph(4)]
        episode_tds = [[object()], [object(), object()]]

        actions = trainer._select_multi_env_actions(observations, episode_tds, [0, 1])

        self.assertEqual(agent.calls, [(observations[0], True), (observations[1], False)])
        self.assertEqual([action.shape for action in actions], [(3, 1), (4, 1)])

    def test_seed_collection_keeps_per_environment_random_actions(self):
        agent = mock.Mock()
        trainer = self.make_trainer(agent, step=1, seed_steps=1)
        trainer.env.rand_act.side_effect = lambda env_idx: torch.tensor([float(env_idx)])

        actions = trainer._select_multi_env_actions(
            [graph(3), graph(4), graph(5)],
            [[object()], [object()], [object()]],
            [2, 0],
        )

        agent.act_batch.assert_not_called()
        self.assertEqual(trainer.env.rand_act.call_args_list, [mock.call(env_idx=2), mock.call(env_idx=0)])
        torch.testing.assert_close(actions[0], torch.tensor([2.0]))
        torch.testing.assert_close(actions[1], torch.tensor([0.0]))


if __name__ == "__main__":
    unittest.main()
