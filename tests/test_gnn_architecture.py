from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_actor_critic import GNNActorCritic
from common.gnn_layers import GNN, Q_GNN
from common.mlp_layers import NormedLinear


def graph(num_nodes=4):
    source = torch.arange(num_nodes, dtype=torch.long)
    target = source.roll(-1)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])], dim=0
    )
    return Data(x=torch.randn(num_nodes, 3), edge_index=edge_index)


def cfg(**overrides):
    values = dict(
        obs_dim=3,
        action_dim=1,
        mpl_dims=[8, 6],
        message_hidden_dims=[10, 9],
        action_head_hidden_dims=[],
        mpl_skip_connections=True,
        head_hidden_dims=[5],
        mlp_dim=12,
        dropout=0.0,
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        tau=0.005,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class GNNArchitectureTest(unittest.TestCase):
    def test_variable_width_stack_and_projected_residual(self):
        model = GNN(3, hidden_channels=[11, 7], mpl_dims=[8, 8, 5])
        self.assertEqual(len(model.extra_mpls), 2)
        self.assertIsInstance(model.skip_projections[0], torch.nn.Identity)
        self.assertIsInstance(model.skip_projections[1], torch.nn.Linear)
        self.assertEqual(model.skip_projections[1].in_features, 8)
        self.assertEqual(model.skip_projections[1].out_features, 5)
        self.assertEqual(model(graph().x, graph().edge_index).shape, (4, 5))
        self.assertEqual(sum(isinstance(layer, NormedLinear) for layer in model.phi), 2)
        self.assertEqual(sum(isinstance(layer, NormedLinear) for layer in model.extra_mpls[0].phi), 2)

    def test_disabled_skips_create_no_projection_modules(self):
        model = GNN(3, hidden_channels=[], mpl_dims=[8, 5], skip_connections=False)
        self.assertEqual(len(model.skip_projections), 0)
        self.assertEqual(model(graph().x, graph().edge_index).shape, (4, 5))

    def test_single_layer_preserves_first_layer_parameter_names(self):
        model = GNN(3, 8, [10, 10])
        keys = set(model.state_dict())
        self.assertTrue(any(key.startswith("phi.") for key in keys))
        self.assertTrue(any(key.startswith("gamma.") for key in keys))
        self.assertFalse(any(key.startswith("extra_mpls.") for key in keys))
        self.assertFalse(any("attention_score" in key for key in keys))

    def test_attention_normalizes_messages_over_incoming_edges(self):
        model = GNN(3, hidden_channels=[], mpl_dims=[5], message_attention=True)
        model.eval()
        center = torch.tensor([[0.2, -0.3, 0.7]])
        neighbor = torch.tensor([[1.1, 0.4, -0.2]])
        single_x = torch.cat([center, neighbor], dim=0)
        repeated_x = torch.cat([center, neighbor.repeat(3, 1)], dim=0)
        single_edge = torch.tensor([[1], [0]], dtype=torch.long)
        repeated_edges = torch.tensor(
            [[1, 2, 3], [0, 0, 0]], dtype=torch.long
        )

        single_output = model(single_x, single_edge)[0]
        repeated_output = model(repeated_x, repeated_edges)[0]

        torch.testing.assert_close(single_output, repeated_output)

    def test_attention_is_single_head_and_backpropagates_through_every_layer(self):
        model = GNN(
            3,
            hidden_channels=[11],
            mpl_dims=[8, 6],
            message_attention=True,
        )
        output = model(graph().x, graph().edge_index)
        output.sum().backward()

        self.assertEqual(model.attention_score.out_features, 1)
        self.assertIsNotNone(model.attention_score.weight.grad)
        self.assertEqual(model.extra_mpls[0].attention_score.out_features, 1)
        self.assertIsNotNone(model.extra_mpls[0].attention_score.weight.grad)
    def test_disabled_edge_features_preserve_first_layer_shapes(self):
        legacy = GNN(3, 8, [10, 10])
        explicit_disabled = GNN(3, 8, [10, 10], edge_channels=0)
        self.assertEqual(legacy.state_dict().keys(), explicit_disabled.state_dict().keys())
        for key, value in legacy.state_dict().items():
            self.assertEqual(value.shape, explicit_disabled.state_dict()[key].shape)

    def test_critic_handles_batched_graphs_and_backward(self):
        batch = Batch.from_data_list([graph(3), graph(5)])
        action = torch.randn(batch.x.size(0), 1)
        critic = Q_GNN(4, hidden_channels=[9], head_hidden_dims=[], mpl_dims=[7, 5])
        values = critic(torch.cat([batch.x, action], dim=-1), batch.edge_index, batch.batch)
        self.assertEqual(values.shape, (2,))
        values.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in critic.parameters()))

    def test_actor_critic_uses_linear_action_projection(self):
        model = GNNActorCritic(cfg())
        self.assertEqual(model._pi.mpl_dims, [8, 6])
        self.assertEqual(model._pi.hidden_channels, [10, 9])
        self.assertEqual(len(model._action_head), 1)
        self.assertIsInstance(model._action_head[0], torch.nn.Linear)
        self.assertEqual(model._action_head[0].in_features, 6)
        self.assertEqual(model._action_head[0].out_features, 2)
        actor_parameter_ids = {id(parameter) for parameter in model.actor_parameters()}
        self.assertTrue(
            {id(parameter) for parameter in model._action_head.parameters()}
            <= actor_parameter_ids
        )
        for critic in model._Qs.modules_list:
            self.assertEqual(critic.mpl_dims, [8, 6])
            self.assertEqual(critic.head[0].out_features, 5)
        self.assertEqual(model._target_Qs.modules_list[0].mpl_dims, [8, 6])

    def test_actor_critic_supports_hidden_action_head_override(self):
        model = GNNActorCritic(cfg(action_head_hidden_dims=[7, 5]))

        self.assertEqual(len(model._action_head), 3)
        self.assertEqual(model._action_head[0].in_features, 6)
        self.assertEqual(model._action_head[0].out_features, 7)
        self.assertEqual(model._action_head[1].out_features, 5)
        self.assertEqual(model._action_head[2].out_features, 2)

    def test_actor_critic_enables_attention_for_actor_critics_and_targets(self):
        model = GNNActorCritic(cfg(message_attention=True))

        self.assertTrue(model._pi.message_attention)
        self.assertIsNotNone(model._pi.attention_score)
        for critic in model._Qs.modules_list:
            self.assertTrue(critic.message_attention)
            self.assertIsNotNone(critic.attention_score)
        for critic in model._target_Qs.modules_list:
            self.assertTrue(critic.message_attention)
            self.assertIsNotNone(critic.attention_score)

    def test_legacy_config_uses_actor_and_critic_output_widths(self):
        legacy = cfg(embedding_dim=13, Q_output_dim=17)
        del legacy.mpl_dims
        del legacy.message_hidden_dims
        del legacy.action_head_hidden_dims
        del legacy.mpl_skip_connections
        model = GNNActorCritic(legacy)
        self.assertEqual(model._pi.mpl_dims, [13])
        self.assertEqual(model._Qs.modules_list[0].mpl_dims, [17])
        self.assertEqual(model._pi.hidden_channels, [12, 12])

    def test_dimension_validation(self):
        invalid = ([], [0], [-1], [True], [3.5], "8")
        for dims in invalid:
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    GNN(3, hidden_channels=[], mpl_dims=dims)
        with self.assertRaises(ValueError):
            GNN(3, hidden_channels=[False], mpl_dims=[8])
        with self.assertRaisesRegex(ValueError, "critic_readout must be one of"):
            Q_GNN(3, mpl_dims=[8], critic_readout="unknown")


if __name__ == "__main__":
    unittest.main()
