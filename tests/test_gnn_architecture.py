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
        action_head_hidden_dims=[7],
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

    def test_critic_handles_batched_graphs_and_backward(self):
        batch = Batch.from_data_list([graph(3), graph(5)])
        action = torch.randn(batch.x.size(0), 1)
        critic = Q_GNN(4, hidden_channels=[9], head_hidden_dims=[], mpl_dims=[7, 5])
        values = critic(torch.cat([batch.x, action], dim=-1), batch.edge_index, batch.batch)
        self.assertEqual(values.shape, (2,))
        values.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in critic.parameters()))

    def test_actor_critic_uses_independent_head_and_message_widths(self):
        model = GNNActorCritic(cfg())
        self.assertEqual(model._pi.mpl_dims, [8, 6])
        self.assertEqual(model._pi.hidden_channels, [10, 9])
        self.assertEqual(model._action_head[0].out_features, 7)
        for critic in model._Qs.modules_list:
            self.assertEqual(critic.mpl_dims, [8, 6])
            self.assertEqual(critic.head[0].out_features, 5)
        self.assertEqual(model._target_Qs.modules_list[0].mpl_dims, [8, 6])

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


if __name__ == "__main__":
    unittest.main()
