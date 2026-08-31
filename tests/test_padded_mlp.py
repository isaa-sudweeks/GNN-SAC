from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from common.padded_mlp_actor_critic import PaddedMLPActorCritic
from gnn_sac import GNNSAC
from gnn_infer import _make_agent
from padded_mlp_sac import PaddedMLPSAC
from trainer.online_trainer import OnlineTrainer


def cfg(**overrides):
    values = dict(
        device="cpu",
        task="truss-graph",
        tasks=["truss-graph:a", "truss-graph:b"],
        multitask=True,
        mujoco_backend="mujoco",
        truss_topologies=None,
        buffer_size=8,
        batch_size=4,
        steps=20,
        obs_dim=6,
        node_action_dim=1,
        padded_mlp_max_nodes=5,
        padded_mlp_hidden_dims=[16, 16],
        mlp_dim=16,
        dropout=0.0,
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        tau=0.005,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy=-1,
        num_policy_actions=5,
        episode_length=100,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        grad_clip_norm=10.0,
        pcgrad=False,
        use_virtual_node=False,
        graph_features={},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def graph(marker=0.0, node_count=3, action_mask=None, rigidity=0.75):
    x = torch.arange(node_count * 6, dtype=torch.float32).view(node_count, 6)
    x = x / 10.0 + marker
    nodes = torch.arange(node_count, dtype=torch.long)
    edge_index = torch.stack(
        [
            torch.cat([nodes, nodes.roll(-1)]),
            torch.cat([nodes.roll(-1), nodes]),
        ],
        dim=0,
    )
    if action_mask is None:
        action_mask = torch.ones(node_count, dtype=torch.bool)
    return Data(
        x=x,
        edge_index=edge_index,
        action_mask=torch.as_tensor(action_mask, dtype=torch.bool),
        rigidity=torch.tensor([rigidity], dtype=torch.float32),
    )


def transition(marker, task_node_count, action_mask=None):
    first = graph(marker, task_node_count, action_mask=action_mask)
    second = graph(marker + 0.1, task_node_count, action_mask=action_mask)
    action = torch.full((1, task_node_count, 1), marker)
    return [
        {
            "obs": first,
            "action": action,
            "reward": torch.tensor([marker]),
            "terminated": torch.tensor([0.0]),
        },
        {
            "obs": second,
            "action": action,
            "reward": torch.tensor([marker]),
            "terminated": torch.tensor([0.0]),
        },
    ]


class PaddedMLPActorCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PaddedMLPActorCritic(cfg())

    def test_packs_features_existence_action_mask_and_rigidity(self):
        observation = graph(
            node_count=3,
            action_mask=[True, False, True],
            rigidity=0.25,
        )
        encoded, dense_action_mask, _ = self.model.encode_observation(observation)

        self.assertEqual(encoded.shape, (1, 5 * 8 + 1))
        torch.testing.assert_close(encoded[0, :18], observation.x.flatten())
        torch.testing.assert_close(
            encoded[0, 30:35], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            encoded[0, 35:40], torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(float(encoded[0, -1]), 0.25)
        self.assertEqual(int(dense_action_mask.sum()), 2)

    def test_rejects_graph_larger_than_capacity(self):
        with self.assertRaisesRegex(ValueError, "padded_mlp_max_nodes=5"):
            self.model.encode_observation(graph(node_count=6))

    def test_actor_ignores_edges_and_returns_only_active_actions(self):
        first = graph(node_count=4, action_mask=[True, False, True, False])
        second = first.clone()
        second.edge_index = second.edge_index.flip(0)

        first_action = self.model.pi_mean(first)
        second_action = self.model.pi_mean(second)

        self.assertEqual(first_action.shape, (2, 1))
        torch.testing.assert_close(first_action, second_action)

    def test_critic_ignores_passive_replay_action_values(self):
        observation = graph(node_count=4, action_mask=[True, False, True, False])
        baseline_action = torch.tensor([[0.2], [0.0], [-0.4], [0.0]])
        passive_changed = torch.tensor([[0.2], [99.0], [-0.4], [-73.0]])

        baseline_q = self.model.Q(observation, baseline_action, return_type="all")
        changed_q = self.model.Q(observation, passive_changed, return_type="all")

        torch.testing.assert_close(baseline_q, changed_q)

    def test_batched_and_serialized_deterministic_actions_match(self):
        observations = [
            graph(0.0, 3, [True, False, True]),
            graph(1.0, 5, [True, True, False, True, False]),
        ]
        batched = self.model.pi_mean(Batch.from_data_list(observations))
        serialized = torch.cat([self.model.pi_mean(item) for item in observations])
        torch.testing.assert_close(batched, serialized)

    def test_log_probability_is_unchanged_by_batch_padding(self):
        small = graph(0.0, 3, [True, False, True])
        large = graph(1.0, 5, [True, True, False, True, False])
        with mock.patch("torch.randn_like", side_effect=lambda value: torch.zeros_like(value)):
            _, serialized_info = self.model.pi(small)
        with mock.patch("torch.randn_like", side_effect=lambda value: torch.zeros_like(value)):
            _, batched_info = self.model.pi(Batch.from_data_list([small, large]))
        torch.testing.assert_close(
            serialized_info["log_prob"].reshape(()),
            batched_info["log_prob"][0],
        )

    def test_incompatible_graph_augmentation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "use_virtual_node=false"):
            PaddedMLPActorCritic(cfg(use_virtual_node=True))
        with self.assertRaisesRegex(ValueError, "graph node/edge features"):
            PaddedMLPActorCritic(cfg(graph_features={"node_roles": True}))


class PaddedMLPSACTest(unittest.TestCase):
    def test_graph_inference_dispatches_padded_backend(self):
        agent = _make_agent(
            cfg(
                sac_backend="padded_mlp",
                multitask=False,
                tasks=["truss-graph"],
            )
        )
        self.assertIsInstance(agent, PaddedMLPSAC)

    def test_agent_returns_full_node_action_with_passive_rows_zero(self):
        agent = PaddedMLPSAC(cfg(multitask=False, tasks=["truss-graph"]))
        observation = graph(node_count=4, action_mask=[True, False, True, False])
        action = agent.act(observation, eval_mode=True)

        self.assertEqual(action.shape, (4, 1))
        torch.testing.assert_close(action[~observation.action_mask], torch.zeros(2, 1))

    def test_graph_action_projection_removes_seed_and_noise_values(self):
        agent = PaddedMLPSAC(cfg(multitask=False, tasks=["truss-graph"]))
        observation = graph(node_count=4, action_mask=[True, False, True, False])
        action = torch.ones(4, 1)
        projected = agent.project_action(observation, action)
        torch.testing.assert_close(projected[:, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))

    def test_checkpoint_schema_rejects_different_capacity(self):
        first = PaddedMLPSAC(cfg(multitask=False, tasks=["truss-graph"]))
        state = first.training_state_dict()
        second = PaddedMLPSAC(
            cfg(multitask=False, tasks=["truss-graph"], padded_mlp_max_nodes=6)
        )
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            second.load_training_state_dict(state)

    def test_heterogeneous_task_balanced_replay_update(self):
        config = cfg()
        agent = PaddedMLPSAC(config)
        buffer = GNNBuffer(config)
        for task, node_count, mask, markers in (
            (config.tasks[0], 3, [True, False, True], [0.1, 0.2]),
            (config.tasks[1], 5, [True, True, False, True, False], [1.1, 1.2]),
        ):
            for marker in markers:
                buffer.add(
                    transition(marker, node_count, action_mask=mask),
                    task=task,
                )

        metrics = agent.update(buffer)
        self.assertIn("value_loss", metrics)
        self.assertIn("pi_loss", metrics)
        self.assertTrue(torch.isfinite(metrics["value_loss"]))
        self.assertTrue(torch.isfinite(metrics["pi_loss"]))

    def test_heterogeneous_pcgrad_update_is_supported(self):
        config = cfg(pcgrad=True)
        agent = PaddedMLPSAC(config)
        buffer = GNNBuffer(config)
        for task, node_count, mask, markers in (
            (config.tasks[0], 3, [True, False, True], [0.1, 0.2]),
            (config.tasks[1], 5, [True, True, False, True, False], [1.1, 1.2]),
        ):
            for marker in markers:
                buffer.add(
                    transition(marker, node_count, action_mask=mask),
                    task=task,
                )
        metrics = agent.update(buffer)
        self.assertTrue(torch.isfinite(metrics["value_loss"]))
        self.assertTrue(torch.isfinite(metrics["pi_loss"]))


class GraphActionProjectionTrainerTest(unittest.TestCase):
    def test_multi_env_projection_happens_after_action_noise(self):
        observation = graph(node_count=4, action_mask=[True, False, True, False])

        class Agent:
            def act_batch(self, observations):
                return [torch.ones(4, 1) for _ in observations]

            def project_action(self, obs, action):
                projected = action.clone()
                projected[~obs.action_mask] = 0
                return projected

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.agent = Agent()
        trainer.env = SimpleNamespace()
        trainer.cfg = SimpleNamespace(
            seed_steps=0,
            domain_randomization=True,
            action_low=-1.0,
            action_high=1.0,
            domain_randomization_params={
                "action_noise": {"enabled": True, "std": 0.2}
            },
        )
        trainer._step = 1
        actions = trainer._select_multi_env_actions(
            [observation], [[None, None]], [0]
        )
        torch.testing.assert_close(actions[0][~observation.action_mask], torch.zeros(2, 1))


class GNNActorOptimizationRegressionTest(unittest.TestCase):
    def _config(self):
        return cfg(
            multitask=False,
            tasks=["truss-graph"],
            embedding_dim=16,
            Q_output_dim=16,
            mpl_dims=[16],
            message_hidden_dims=[16],
            action_head_hidden_dims=[16],
            head_hidden_dims=[16],
            mpl_skip_connections=True,
            message_attention=False,
            action_dim=1,
            critic_readout="physical_mean",
        )

    def test_actor_optimizer_includes_action_head(self):
        agent = GNNSAC(self._config())
        optimized_ids = {
            id(parameter)
            for group in agent.pi_optim.param_groups
            for parameter in group["params"]
        }
        action_head_ids = {id(parameter) for parameter in agent.model._action_head.parameters()}
        self.assertTrue(action_head_ids <= optimized_ids)

    def test_actor_update_steps_action_head(self):
        agent = GNNSAC(self._config())
        observation = graph(node_count=4, action_mask=[True, False, True, True])
        before = agent.model._action_head[-1].weight.detach().clone()
        original_clip = torch.nn.utils.clip_grad_norm_
        clipped_parameter_ids = set()

        def capture_clip(parameters, *args, **kwargs):
            parameters = tuple(parameters)
            clipped_parameter_ids.update(id(parameter) for parameter in parameters)
            return original_clip(parameters, *args, **kwargs)

        with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=capture_clip):
            agent.update_pi_and_alpha(observation)
        after = agent.model._action_head[-1].weight.detach()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(
            {id(parameter) for parameter in agent.model._action_head.parameters()}
            <= clipped_parameter_ids
        )

    def test_legacy_actor_optimizer_checkpoint_is_upgraded(self):
        source = GNNSAC(self._config())
        source.update_pi_and_alpha(
            graph(node_count=4, action_mask=[True, False, True, True])
        )
        state = source.training_state_dict()
        legacy_count = len(tuple(source.model._pi.parameters()))
        legacy_optimizer = deepcopy(state["pi_optim"])
        group = legacy_optimizer["param_groups"][0]
        removed_ids = set(group["params"][legacy_count:])
        group["params"] = group["params"][:legacy_count]
        legacy_optimizer["state"] = {
            key: value
            for key, value in legacy_optimizer["state"].items()
            if key not in removed_ids
        }
        state["pi_optim"] = legacy_optimizer

        restored = GNNSAC(self._config())
        with self.assertWarnsRegex(RuntimeWarning, "legacy GNN actor optimizer"):
            restored.load_training_state_dict(state)
        self.assertEqual(
            len(restored.pi_optim.param_groups[0]["params"]),
            len(tuple(restored.model.actor_parameters())),
        )


if __name__ == "__main__":
    unittest.main()
