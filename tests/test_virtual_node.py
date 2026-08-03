from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_actor_critic import GNNActorCritic
from common.gnn_buffer import _GNNTaskBuffer
from common.gnn_layers import GNN, Q_GNN
from common.graph_transforms import physical_node_mask, prepare_graph
from gnn_sac import GNNSAC


def graph(num_nodes: int) -> Data:
    source = torch.arange(num_nodes, dtype=torch.long)
    target = source.roll(-1)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])], dim=0
    )
    return Data(x=torch.randn(num_nodes, 3), edge_index=edge_index)


def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        obs_dim=3,
        action_dim=1,
        mpl_dims=[8, 8],
        message_hidden_dims=[10],
        action_head_hidden_dims=[7],
        mpl_skip_connections=True,
        head_hidden_dims=[5],
        mlp_dim=12,
        dropout=0.0,
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        tau=0.005,
        use_virtual_node=True,
        critic_readout="physical_mean",
    )


class VirtualNodeTest(unittest.TestCase):
    def test_pyg_transform_adds_rigidity_virtual_node_without_mutating_source(self):
        original = graph(4)
        original.rigidity = torch.tensor([0.625])
        prepared = prepare_graph(original, use_virtual_node=True)

        self.assertEqual(original.num_nodes, 4)
        self.assertFalse(hasattr(original, "action_mask"))
        self.assertEqual(prepared.num_nodes, 5)
        self.assertEqual(prepared.edge_index.size(1), original.edge_index.size(1) + 8)
        self.assertTrue(torch.equal(prepared.action_mask, torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)))
        self.assertTrue(torch.equal(prepared.physical_node_mask, torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)))
        self.assertEqual(tuple(original.x.shape), (4, 3))
        self.assertEqual(tuple(prepared.x.shape), (5, 5))
        self.assertTrue(torch.equal(prepared.x[:-1, -2:], torch.zeros(4, 2)))
        self.assertTrue(
            torch.equal(prepared.x[-1], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.625]))
        )

    def test_batch_has_one_isolated_virtual_node_per_graph(self):
        prepared = [prepare_graph(graph(count), use_virtual_node=True) for count in (3, 5)]
        batch = Batch.from_data_list(prepared)
        virtual_indices = (~batch.action_mask).nonzero(as_tuple=False).flatten()

        self.assertEqual(virtual_indices.tolist(), [3, 9])
        self.assertEqual(batch.batch[virtual_indices].tolist(), [0, 1])
        self.assertTrue(torch.all(batch.batch[batch.edge_index[0]] == batch.batch[batch.edge_index[1]]))

    def test_replay_stores_physical_graphs_and_augments_samples(self):
        replay_cfg = SimpleNamespace(
            device="cpu",
            buffer_size=4,
            steps=4,
            batch_size=1,
            use_virtual_node=True,
        )
        replay = _GNNTaskBuffer(replay_cfg)
        observations = [graph(3), graph(3)]
        replay.add([
            {
                "obs": observation,
                "action": torch.zeros(1, 3, 1),
                "reward": torch.zeros(1),
                "terminated": torch.zeros(1),
            }
            for observation in observations
        ])

        self.assertEqual(replay._obs[0].num_nodes, 3)
        self.assertFalse(hasattr(replay._obs[0], "action_mask"))
        sampled_obs, sampled_action, _, _, sampled_next_obs = replay.sample()
        self.assertEqual(sampled_obs.num_nodes, 4)
        self.assertEqual(sampled_next_obs.num_nodes, 4)
        self.assertEqual(int(sampled_obs.action_mask.sum()), 3)
        self.assertEqual(sampled_action.shape, (3, 1))

    def test_actor_excludes_virtual_nodes_from_actions_and_entropy(self):
        torch.manual_seed(7)
        batch = Batch.from_data_list([
            prepare_graph(graph(3), use_virtual_node=True),
            prepare_graph(graph(5), use_virtual_node=True),
        ])
        model = GNNActorCritic(cfg())

        action, info = model.pi(batch)
        deterministic_action = model.pi_mean(batch)

        self.assertEqual(action.shape, (8, 1))
        self.assertEqual(deterministic_action.shape, (8, 1))
        self.assertEqual(info["mean"].shape, (8, 1))
        self.assertEqual(info["log_prob"].shape, (2,))

    def test_actor_excludes_passive_nodes_and_act_batch_zero_pads_them(self):
        observation = graph(4)
        observation.action_mask = torch.tensor([True, False, True, False])
        agent_cfg = cfg()
        agent_cfg.device = "cpu"
        agent_cfg.lr = 1e-3
        agent_cfg.entropy_coef = 0.2
        agent_cfg.target_entropy = -1
        agent_cfg.episode_length = 100
        agent_cfg.discount_denom = 5
        agent_cfg.discount_min = 0.95
        agent_cfg.discount_max = 0.999
        agent_cfg.use_virtual_node = True
        agent_cfg.mujoco_backend = "mujoco"

        agent = GNNSAC(agent_cfg)
        prepared = prepare_graph(observation, use_virtual_node=True)
        sampled_action, info = agent.model.pi(prepared)
        env_action = agent.act(observation)

        self.assertEqual(sampled_action.shape, (2, 1))
        self.assertEqual(info["mean"].shape, (2, 1))
        self.assertEqual(info["log_prob"].shape, (1,))
        self.assertEqual(env_action.shape, (4, 1))
        self.assertTrue(torch.equal(env_action[~observation.action_mask], torch.zeros(2, 1)))

    def test_critic_zero_pads_virtual_actions_and_pools_physical_nodes(self):
        first = graph(3)
        first.action_mask = torch.tensor([True, False, True])
        second = graph(4)
        second.action_mask = torch.tensor([False, True, True, True])
        batch = Batch.from_data_list([
            prepare_graph(first, use_virtual_node=True),
            prepare_graph(second, use_virtual_node=True),
        ])
        model = GNNActorCritic(cfg())
        action = torch.randn(5, 1)
        captured_input = []
        captured_pool = []
        critic = model._Qs.modules_list[0]
        input_hook = critic.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0].detach().clone())
        )
        head_hook = critic.head.register_forward_pre_hook(
            lambda module, args: captured_pool.append(args[0].detach().clone())
        )
        try:
            values = model.Q(batch, action, return_type="all")
        finally:
            input_hook.remove()
            head_hook.remove()

        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(torch.equal(captured_input[0][~batch.action_mask, -1], torch.zeros(4)))
        encoded = GNN.forward(critic, captured_input[0], batch.edge_index)
        pool_mask = physical_node_mask(batch)
        expected_pool = global_mean_pool(
            encoded[pool_mask], batch.batch[pool_mask]
        )
        self.assertTrue(torch.allclose(captured_pool[0], expected_pool))

    def test_critic_accepts_physical_node_replay_actions_with_virtual_nodes(self):
        first = graph(3)
        first.action_mask = torch.tensor([True, False, True])
        second = graph(4)
        second.action_mask = torch.tensor([False, True, True, True])
        batch = Batch.from_data_list([
            prepare_graph(first, use_virtual_node=True),
            prepare_graph(second, use_virtual_node=True),
        ])
        model = GNNActorCritic(cfg())
        action = torch.arange(1, 8, dtype=torch.float32).unsqueeze(-1)
        captured_input = []
        critic = model._Qs.modules_list[0]
        input_hook = critic.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0].detach().clone())
        )
        try:
            values = model.Q(batch, action, return_type="all")
        finally:
            input_hook.remove()

        self.assertEqual(values.shape, (2, 2))
        expected = torch.tensor([1, 0, 3, 0, 0, 5, 6, 7, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(captured_input[0][:, -1], expected))

    def test_critic_can_read_out_virtual_node_embedding(self):
        batch = Batch.from_data_list([
            prepare_graph(graph(3), use_virtual_node=True),
            prepare_graph(graph(5), use_virtual_node=True),
        ])
        config = cfg()
        config.critic_readout = "virtual_node"
        model = GNNActorCritic(config)
        action = torch.randn(8, 1)
        captured_input = []
        captured_readout = []
        critic = model._Qs.modules_list[0]
        input_hook = critic.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0].detach().clone())
        )
        head_hook = critic.head.register_forward_pre_hook(
            lambda module, args: captured_readout.append(args[0].detach().clone())
        )
        try:
            values = model.Q(batch, action, return_type="all")
        finally:
            input_hook.remove()
            head_hook.remove()

        encoded = GNN.forward(critic, captured_input[0], batch.edge_index)
        expected = encoded[~physical_node_mask(batch)]
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(torch.allclose(captured_readout[0], expected))
        values.sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model._Qs.parameters())
        )

    def test_critic_can_concatenate_physical_mean_and_virtual_node(self):
        batch = Batch.from_data_list([
            prepare_graph(graph(3), use_virtual_node=True),
            prepare_graph(graph(5), use_virtual_node=True),
        ])
        config = cfg()
        config.critic_readout = "physical_mean_virtual_node"
        model = GNNActorCritic(config)
        action = torch.randn(8, 1)
        captured_input = []
        captured_readout = []
        critic = model._Qs.modules_list[0]
        input_hook = critic.register_forward_pre_hook(
            lambda module, args: captured_input.append(args[0].detach().clone())
        )
        head_hook = critic.head.register_forward_pre_hook(
            lambda module, args: captured_readout.append(args[0].detach().clone())
        )
        try:
            values = model.Q(batch, action, return_type="all")
        finally:
            input_hook.remove()
            head_hook.remove()

        encoded = GNN.forward(critic, captured_input[0], batch.edge_index)
        pool_mask = physical_node_mask(batch)
        expected = torch.cat([
            global_mean_pool(encoded[pool_mask], batch.batch[pool_mask]),
            encoded[~pool_mask],
        ], dim=-1)
        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(critic.head[0].in_features, 16)
        self.assertTrue(torch.allclose(captured_readout[0], expected))
        values.sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model._Qs.parameters())
        )

    def test_virtual_critic_readouts_require_virtual_nodes(self):
        for readout in ("virtual_node", "physical_mean_virtual_node"):
            with self.subTest(readout=readout):
                config = cfg()
                config.use_virtual_node = False
                config.critic_readout = readout
                with self.assertRaisesRegex(ValueError, "requires use_virtual_node=true"):
                    GNNActorCritic(config)

    def test_two_layers_create_physical_virtual_physical_dependency(self):
        base = Data(
            x=torch.randn(2, 3),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        prepared = prepare_graph(base, use_virtual_node=True)

        def cross_node_gradient(depth: int) -> torch.Tensor:
            x = prepared.x.clone().requires_grad_(True)
            model = GNN(5, hidden_channels=[6], mpl_dims=[5] * depth, skip_connections=False)
            output = model(x, prepared.edge_index)
            return torch.autograd.grad(output[1].sum(), x)[0][0, :3]

        self.assertTrue(torch.equal(cross_node_gradient(1), torch.zeros(3)))
        self.assertGreater(float(cross_node_gradient(2).abs().sum()), 0.0)

    def test_one_layer_broadcasts_virtual_rigidity_to_physical_nodes(self):
        base = graph(3)
        base.rigidity = torch.tensor([0.75])
        prepared = prepare_graph(base, use_virtual_node=True)
        x = prepared.x.clone().requires_grad_(True)
        model = GNN(5, hidden_channels=[6], mpl_dims=[5], skip_connections=False)

        physical_output = model(x, prepared.edge_index)[:-1].sum()
        gradient = torch.autograd.grad(physical_output, x)[0]

        self.assertGreater(float(gradient[-1, -1].abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
