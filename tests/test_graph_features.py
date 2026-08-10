from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import _GNNTaskBuffer
from common.gnn_actor_critic import GNNActorCritic
from common.gnn_layers import GNN
from common.graph_transforms import (
    EDGE_ROLE_NAMES,
    graph_edge_input_dim,
    graph_feature_schema,
    graph_input_dim,
    prepare_graph,
)
from env.mujoco_gen.topology_envs import _semantic_edge_roles
from env.wrappers.tensor import TensorWrapper
from gnn_sac import GNNSAC


def graph() -> Data:
    return Data(
        x=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 12.0],
            ]
        ),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_role=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        action_mask=torch.tensor([True, False, True]),
    )


def agent_cfg(**feature_overrides) -> SimpleNamespace:
    return SimpleNamespace(
        obs_dim=3,
        action_dim=1,
        mpl_dims=[8],
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
        use_virtual_node=False,
        graph_features=SimpleNamespace(
            node_roles=feature_overrides.get("node_roles", False),
            edge_roles=feature_overrides.get("edge_roles", False),
            edge_distance=feature_overrides.get("edge_distance", False),
        ),
        device="cpu",
        lr=1e-3,
        entropy_coef=0.2,
        target_entropy=-1,
        episode_length=100,
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.999,
        mujoco_backend="mujoco",
    )


class GraphFeatureTransformTest(unittest.TestCase):
    def test_all_feature_switch_combinations(self):
        source = graph()
        for node_roles, edge_roles, edge_distance in product((False, True), repeat=3):
            with self.subTest(
                node_roles=node_roles,
                edge_roles=edge_roles,
                edge_distance=edge_distance,
            ):
                prepared = prepare_graph(
                    source,
                    use_virtual_node=False,
                    use_node_roles=node_roles,
                    use_edge_roles=edge_roles,
                    use_edge_distance=edge_distance,
                )
                if not (node_roles or edge_roles or edge_distance):
                    self.assertIs(prepared, source)
                    continue
                self.assertEqual(prepared.x.shape, (3, 3 + 2 * int(node_roles)))
                edge_width = 3 * int(edge_roles) + int(edge_distance)
                if edge_width:
                    self.assertEqual(prepared.edge_attr.shape, (4, edge_width))
                if node_roles:
                    expected = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
                    self.assertTrue(torch.equal(prepared.x[:, -2:], expected))
                if edge_roles:
                    expected = torch.tensor(
                        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
                        dtype=torch.float32,
                    )
                    self.assertTrue(torch.equal(prepared.edge_attr[:, :3], expected))
                if edge_distance:
                    self.assertTrue(
                        torch.allclose(
                            prepared.edge_attr[:, -1],
                            torch.tensor([5.0, 5.0, 13.0, 13.0]),
                        )
                    )

    def test_distance_uses_observed_positions_and_is_translation_invariant(self):
        original = graph()
        translated = graph()
        translated.x[:, :3] += torch.tensor([17.0, -2.0, 9.0])
        noisy = graph()
        noisy.x[1, :3] = torch.tensor([0.0, 0.0, 2.0])

        kwargs = dict(use_virtual_node=False, use_edge_distance=True)
        original_distance = prepare_graph(original, **kwargs).edge_attr
        translated_distance = prepare_graph(translated, **kwargs).edge_attr
        noisy_distance = prepare_graph(noisy, **kwargs).edge_attr

        self.assertTrue(torch.allclose(original_distance, translated_distance))
        self.assertFalse(torch.allclose(original_distance, noisy_distance))
        self.assertTrue(torch.allclose(noisy_distance[:2, 0], torch.tensor([2.0, 2.0])))

    def test_virtual_edges_have_virtual_role_and_zero_distance(self):
        prepared = prepare_graph(
            graph(),
            use_virtual_node=True,
            use_node_roles=True,
            use_edge_roles=True,
            use_edge_distance=True,
        )
        physical_edge_count = 4
        virtual_edges = prepared.edge_attr[physical_edge_count:]

        self.assertEqual(prepared.x.shape, (4, 7))
        self.assertTrue(torch.equal(prepared.x[-1, 3:5], torch.zeros(2)))
        self.assertTrue(torch.equal(virtual_edges[:, :2], torch.zeros(6, 2)))
        self.assertTrue(torch.equal(virtual_edges[:, 2], torch.ones(6)))
        self.assertTrue(torch.equal(virtual_edges[:, 3], torch.zeros(6)))

    def test_invalid_or_missing_edge_roles_fail_clearly(self):
        missing = graph()
        del missing.edge_role
        with self.assertRaisesRegex(ValueError, "requires graph.edge_role"):
            prepare_graph(missing, use_virtual_node=False, use_edge_roles=True)

        invalid = graph()
        invalid.edge_role[0] = 2
        with self.assertRaisesRegex(ValueError, "tube=0 or connector=1"):
            prepare_graph(invalid, use_virtual_node=False, use_edge_roles=True)

    def test_dimension_helpers_match_feature_contract(self):
        self.assertEqual(EDGE_ROLE_NAMES, ("tube", "connector", "virtual"))
        self.assertEqual(
            graph_input_dim(6, use_virtual_node=True, use_node_roles=True), 10
        )
        self.assertEqual(
            graph_edge_input_dim(use_edge_roles=True, use_edge_distance=True), 4
        )


class GraphFeatureIntegrationTest(unittest.TestCase):
    def test_upstream_edge_types_map_to_semantic_roles(self):
        with patch(
            "env.mujoco_gen.topology_envs.get_edge_types",
            return_value=np.asarray(["actuated", "connector", "structural"], dtype=object),
        ):
            roles = _semantic_edge_roles(object(), graph_view="control")
        self.assertTrue(np.array_equal(roles, np.asarray([0, 1, 0], dtype=np.int64)))

    def test_tensor_wrapper_preserves_edge_roles(self):
        wrapper = TensorWrapper.__new__(TensorWrapper)
        wrapper.graph_observations = True
        observation = {
            "x": np.zeros((2, 3), dtype=np.float32),
            "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            "action_mask": np.asarray([True, False]),
            "edge_role": np.asarray([0, 1], dtype=np.int64),
        }
        converted = wrapper._obs_to_tensor(observation)
        self.assertTrue(torch.equal(converted.edge_role, torch.tensor([0, 1])))

    def test_replay_preserves_roles_and_constructs_edge_features(self):
        replay_cfg = SimpleNamespace(
            device="cpu",
            buffer_size=4,
            steps=4,
            batch_size=1,
            use_virtual_node=False,
            graph_features=SimpleNamespace(
                node_roles=True,
                edge_roles=True,
                edge_distance=True,
            ),
        )
        replay = _GNNTaskBuffer(replay_cfg)
        observations = [graph(), graph()]
        replay.add(
            [
                {
                    "obs": observation,
                    "action": torch.zeros(1, 3, 1),
                    "reward": torch.zeros(1),
                    "terminated": torch.zeros(1),
                }
                for observation in observations
            ]
        )
        sampled, _, _, _, sampled_next = replay.sample()
        self.assertEqual(sampled.x.shape, (3, 5))
        self.assertEqual(sampled.edge_attr.shape, (4, 4))
        self.assertEqual(sampled_next.edge_attr.shape, (4, 4))

    def test_edge_conditioned_gnn_has_edge_gradients(self):
        prepared = prepare_graph(
            graph(),
            use_virtual_node=False,
            use_edge_roles=True,
            use_edge_distance=True,
        )
        edge_attr = prepared.edge_attr.clone().requires_grad_(True)
        model = GNN(3, hidden_channels=[7], mpl_dims=[5], edge_channels=4)
        output = model(prepared.x, prepared.edge_index, edge_attr)
        gradient = torch.autograd.grad(output.sum(), edge_attr)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_actor_equivariance_and_critic_invariance_under_permutation(self):
        original = graph()
        original.action_mask = torch.ones(3, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        permuted = Data(
            x=original.x[permutation],
            edge_index=inverse[original.edge_index],
            edge_role=original.edge_role.clone(),
            action_mask=original.action_mask[permutation],
        )
        kwargs = dict(
            use_virtual_node=False,
            use_node_roles=True,
            use_edge_roles=True,
            use_edge_distance=True,
        )
        original = prepare_graph(original, **kwargs)
        permuted = prepare_graph(permuted, **kwargs)
        model = GNNActorCritic(agent_cfg(node_roles=True, edge_roles=True, edge_distance=True))
        model.eval()

        original_action = model.pi_mean(original)
        permuted_action = model.pi_mean(permuted)
        self.assertTrue(torch.allclose(permuted_action, original_action[permutation]))

        q_original = model.Q(original, original_action, return_type="all")
        q_permuted = model.Q(permuted, permuted_action, return_type="all")
        self.assertTrue(torch.allclose(q_original, q_permuted, atol=1e-6, rtol=1e-6))

    def test_checkpoint_schema_accepts_match_and_rejects_mismatch(self):
        agent = GNNSAC(agent_cfg(node_roles=True, edge_roles=True, edge_distance=True))
        state = {
            "model": agent.model.state_dict(),
            "graph_feature_schema": graph_feature_schema(agent.cfg),
        }
        agent.load(state)

        missing_schema = {"model": agent.model.state_dict()}
        with self.assertRaisesRegex(ValueError, "no graph feature schema"):
            agent.load(missing_schema)

        mismatch = dict(state)
        mismatch["graph_feature_schema"] = graph_feature_schema(agent_cfg())
        with self.assertRaisesRegex(ValueError, "does not match"):
            agent.load(mismatch)


if __name__ == "__main__":
    unittest.main()
