from pathlib import Path
import hashlib
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from common.graph_transforms import graph_structure_signature
from common.tensor_gnn_buffer import TensorGNNBuffer, _task_names, make_gnn_buffer
from common.distillation import replay_observations
from gnn_sac import GNNSAC
from tests.test_task_balanced_replay import (
    agent_cfg as legacy_agent_cfg,
    assert_nested_equal,
    populate_buffer as populate_legacy_fixture,
    transition as legacy_transition,
)
from scripts.convert_gnn_replay_checkpoint import convert


def config(**overrides):
    values = dict(
        device="cpu", task="graph", tasks=["graph:a", "graph:b"], multitask=True,
        mujoco_backend="mujoco", truss_topologies=None, buffer_size=16, batch_size=4,
        steps=32, obs_dim=6, action_dim=1, node_counts=[3, 5], num_nodes=5,
        use_virtual_node=False, graph_features={}, replay_backend="torchrl_tensor",
        replay_storage="cpu_pinned", replay_gpu_fraction=.2, replay_gpu_max_gb=8.,
        replay_gpu_reserve_gb=12.,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def graph(marker, nodes, *, metadata=False):
    x = torch.arange(nodes * 6, dtype=torch.float32).reshape(nodes, 6) + marker
    ids = torch.arange(nodes)
    edge_index = torch.stack((torch.cat((ids, ids.roll(-1))), torch.cat((ids.roll(-1), ids))))
    result = Data(x=x, edge_index=edge_index)
    if metadata:
        result.action_mask = ids.remainder(2).eq(0)
        result.rigidity = torch.tensor([marker / 10])
        result.edge_role = torch.arange(edge_index.size(1)).remainder(2)
    return result


def transition(marker, nodes, *, metadata=False):
    action = torch.full((1, nodes, 1), float(marker))
    reward, terminated = torch.tensor([float(marker)]), torch.tensor([0.])
    return [
        dict(obs=graph(marker, nodes, metadata=metadata), action=action, reward=reward, terminated=terminated),
        dict(obs=graph(marker + .5, nodes, metadata=metadata), action=action, reward=reward, terminated=terminated),
    ]


def masked_transition(marker, current_mask, next_mask=None):
    current_mask = torch.as_tensor(current_mask, dtype=torch.bool)
    next_mask = current_mask if next_mask is None else torch.as_tensor(next_mask, dtype=torch.bool)
    nodes = int(current_mask.numel())
    values = transition(marker, nodes, metadata=True)
    values[0]["obs"].action_mask = current_mask.clone()
    values[1]["obs"].action_mask = next_mask.clone()
    return values


def populate(buffer, *, metadata=False, count=6):
    for task, nodes, base in (("graph:a", 3, 0), ("graph:b", 5, 10)):
        for offset in range(count):
            buffer.add(transition(base + offset, nodes, metadata=metadata), task=task)


def assert_batch_equal(test, expected, actual):
    for left, right in zip(expected, actual):
        if hasattr(left, "to_dict"):
            test.assertEqual(left.to_dict().keys(), right.to_dict().keys())
            for key in left.to_dict():
                torch.testing.assert_close(left.to_dict()[key], right.to_dict()[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def assert_tensor_task_state_equal(test, expected, actual):
    """Compare the ring metadata and dense storage that affect future writes."""
    for key in ("capacity", "batch_size", "num_eps", "size", "idx", "static"):
        assert_nested_equal(test, expected[key], actual[key])
    expected_storage = expected["replay_buffer"]["_storage"]["_storage"]
    actual_storage = actual["replay_buffer"]["_storage"]["_storage"]
    test.assertEqual(expected_storage.keys(), actual_storage.keys())
    for key in expected_storage:
        torch.testing.assert_close(expected_storage[key], actual_storage[key], rtol=0, atol=0)


class TensorGNNBufferTest(unittest.TestCase):
    def test_factory_defaults_to_tensor_and_keeps_legacy_override(self):
        self.assertIsInstance(make_gnn_buffer(config(replay_backend="legacy")), GNNBuffer)
        cfg = config()
        del cfg.replay_backend
        self.assertIsInstance(make_gnn_buffer(cfg), TensorGNNBuffer)

    def test_direct_batches_match_legacy_for_mixed_node_counts(self):
        legacy, tensor = GNNBuffer(config()), TensorGNNBuffer(config())
        populate(legacy)
        populate(tensor)
        torch.manual_seed(91)
        expected = legacy.sample_with_tasks()
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(91)
        actual = tensor.sample_with_tasks()
        actual_rng = torch.random.get_rng_state()
        assert_batch_equal(self, expected.combined, actual.combined)
        for task in tensor.task_names:
            assert_batch_equal(self, expected.by_task[task], actual.by_task[task])
            expected_raw = expected.raw_observations_by_task[task]
            actual_raw = actual.raw_observations_by_task[task]
            self.assertEqual(len(expected_raw), len(actual_raw))
            for expected_graph, actual_graph in zip(expected_raw, actual_raw):
                torch.testing.assert_close(expected_graph.x, actual_graph.x)
                torch.testing.assert_close(
                    expected_graph.edge_index, actual_graph.edge_index
                )
                self.assertEqual(
                    "action_mask" in expected_graph, "action_mask" in actual_graph
                )
                if "action_mask" in expected_graph:
                    torch.testing.assert_close(
                        expected_graph.action_mask, actual_graph.action_mask
                    )
        self.assertTrue(torch.equal(expected_rng, actual_rng))

    def test_task_only_sampling_skips_combined_batch_and_keeps_raw_graphs(self):
        tensor = TensorGNNBuffer(config())
        populate(tensor, metadata=True)

        sampled = tensor.sample_with_tasks(combine=False)

        self.assertIsNone(sampled.combined)
        self.assertEqual(set(sampled.by_task), set(tensor.task_names))
        self.assertEqual(
            set(sampled.raw_observations_by_task), set(tensor.task_names)
        )
        for graphs in sampled.raw_observations_by_task.values():
            self.assertTrue(graphs)
            self.assertTrue(all(hasattr(graph, "edge_role") for graph in graphs))

    def test_graph_features_and_virtual_nodes_match_exactly(self):
        cfg = config(
            use_virtual_node=True,
            graph_features={"node_roles": True, "edge_roles": True, "edge_distance": True},
        )
        legacy, tensor = GNNBuffer(cfg), TensorGNNBuffer(config(
            use_virtual_node=True,
            graph_features={"node_roles": True, "edge_roles": True, "edge_distance": True},
        ))
        populate(legacy, metadata=True)
        populate(tensor, metadata=True)
        torch.manual_seed(12)
        expected = legacy.sample()
        torch.manual_seed(12)
        actual = tensor.sample()
        assert_batch_equal(self, expected, actual)

    def test_dynamic_action_masks_are_stored_and_reconstructed_per_transition(self):
        cfg = config(
            multitask=False, task="graph", tasks=[], node_counts=[3], num_nodes=3,
            buffer_size=4, batch_size=2, graph_features={"node_roles": True},
        )
        tensor = TensorGNNBuffer(cfg)
        first_current = torch.tensor([True, True, False])
        first_next = torch.tensor([True, False, False])
        second_current = torch.tensor([False, True, True])
        second_next = torch.tensor([False, True, False])
        tensor.add(masked_transition(1, first_current, first_next))
        tensor.add(masked_transition(2, second_current, second_next))

        state = tensor.state_dict()
        self.assertEqual(state["format_version"], 4)
        task_state = state["buffers"]["graph"]
        fields = task_state["replay_buffer"]["_storage"]["_storage"]
        torch.testing.assert_close(
            fields["obs_action_mask"][:2], torch.stack((first_current, second_current))
        )
        torch.testing.assert_close(
            fields["next_obs_action_mask"][:2], torch.stack((first_next, second_next))
        )

        with patch("torch.randint", return_value=torch.tensor([0, 1])):
            observations, _, _, _, next_observations = tensor.sample()
        torch.testing.assert_close(
            observations.action_mask.reshape(2, 3),
            torch.stack((first_current, second_current)),
        )
        torch.testing.assert_close(
            next_observations.action_mask.reshape(2, 3),
            torch.stack((first_next, second_next)),
        )
        torch.testing.assert_close(
            observations.x.reshape(2, 3, -1)[:, :, 6:8],
            torch.stack((
                torch.stack((first_current, ~first_current), dim=-1),
                torch.stack((second_current, ~second_current), dim=-1),
            )).float(),
        )
        self.assertEqual(observations._policy_action_count_cache, 4)
        self.assertEqual(next_observations._policy_action_count_cache, 2)

        with patch("torch.randint", return_value=torch.tensor([0, 1])):
            raw = tensor.sample_with_tasks(combine=False).raw_observations_by_task["graph"]
        torch.testing.assert_close(raw[0].action_mask, first_current)
        torch.testing.assert_close(raw[1].action_mask, second_current)

    def test_teacher_signature_accepts_only_dynamic_active_subsets(self):
        cfg = config(
            multitask=False, task="graph", tasks=[], node_counts=[3], num_nodes=3,
            buffer_size=4, batch_size=1, graph_features={"node_roles": True},
        )
        tensor = TensorGNNBuffer(cfg)
        base = masked_transition(0, [True, True, False])[0]["obs"]
        tensor.set_task_graph_signatures({"graph": graph_structure_signature(base)})
        tensor.add(masked_transition(1, [True, False, False]))
        with self.assertRaisesRegex(ValueError, "differs from teacher"):
            tensor.add(masked_transition(2, [True, False, True]))
        with self.assertRaisesRegex(ValueError, "Invalid teacher action mask"):
            tensor.add(masked_transition(3, [False, False, False]))
        changed_edges = masked_transition(4, [True, False, False])
        changed_edges[0]["obs"].edge_index = changed_edges[0]["obs"].edge_index.roll(1, 1)
        with self.assertRaisesRegex(ValueError, "topology or action ordering"):
            tensor.add(changed_edges)

    def test_v3_checkpoint_upgrades_masks_and_accepts_dynamic_writes(self):
        cfg = config(
            multitask=False, task="graph", tasks=[], node_counts=[3], num_nodes=3,
            buffer_size=4, batch_size=2, graph_features={"node_roles": True},
        )
        original = TensorGNNBuffer(cfg)
        base_mask = torch.tensor([True, True, False])
        original.add(masked_transition(1, base_mask))
        original.add(masked_transition(2, base_mask))
        v3 = original.state_dict()
        v3["format_version"] = 3
        for task_state in v3["buffers"].values():
            task_state["format_version"] = 3
            storage = task_state["replay_buffer"]["_storage"]["_storage"]
            del storage["obs_action_mask"]
            del storage["next_obs_action_mask"]

        restored = TensorGNNBuffer(config(
            multitask=False, task="graph", tasks=[], node_counts=[3], num_nodes=3,
            buffer_size=4, batch_size=2, graph_features={"node_roles": True},
        ))
        restored.load_state_dict(v3)
        upgraded = restored.state_dict()
        self.assertEqual(upgraded["format_version"], 4)
        upgraded_fields = upgraded["buffers"]["graph"]["replay_buffer"]["_storage"]["_storage"]
        torch.testing.assert_close(
            upgraded_fields["obs_action_mask"][:2], base_mask.repeat(2, 1)
        )
        torch.testing.assert_close(
            upgraded_fields["next_obs_action_mask"][:2], base_mask.repeat(2, 1)
        )

        dynamic_mask = torch.tensor([True, False, False])
        restored.add(masked_transition(3, dynamic_mask))
        v4 = restored.state_dict()
        reloaded = TensorGNNBuffer(config(
            multitask=False, task="graph", tasks=[], node_counts=[3], num_nodes=3,
            buffer_size=4, batch_size=2, graph_features={"node_roles": True},
        ))
        reloaded.load_state_dict(v4)
        assert_tensor_task_state_equal(
            self, v4["buffers"]["graph"], reloaded.state_dict()["buffers"]["graph"]
        )
        replayed = list(replay_observations(v4["buffers"]["graph"]))
        torch.testing.assert_close(replayed[-1].action_mask, dynamic_mask)

    def test_full_ring_nonzero_cursor_round_trip_preserves_future_overwrites(self):
        tensor = TensorGNNBuffer(config(buffer_size=8))
        populate(tensor, count=7)
        state = tensor.state_dict()
        for task, expected_rewards in (
            ("graph:a", [4., 5., 6., 3.]),
            ("graph:b", [14., 15., 16., 13.]),
        ):
            task_state = state["buffers"][task]
            self.assertEqual(
                (task_state["capacity"], task_state["size"], task_state["idx"]),
                (4, 4, 3),
            )
            rewards = task_state["replay_buffer"]["_storage"]["_storage"]["reward"]
            self.assertEqual(rewards.flatten().tolist(), expected_rewards)

        restored = TensorGNNBuffer(config(buffer_size=8))
        restored.load_state_dict(state)
        self.assertEqual(restored.sizes_by_task, tensor.sizes_by_task)
        restored_state = restored.state_dict()
        for task in tensor.task_names:
            assert_tensor_task_state_equal(
                self, state["buffers"][task], restored_state["buffers"][task]
            )

        for buffer in (tensor, restored):
            buffer.add(transition(100, 3), task="graph:a")
            buffer.add(transition(200, 5), task="graph:b")
        expected_after_write = tensor.state_dict()
        actual_after_write = restored.state_dict()
        for task, expected_rewards in (
            ("graph:a", [4., 5., 6., 100.]),
            ("graph:b", [14., 15., 16., 200.]),
        ):
            expected_task = expected_after_write["buffers"][task]
            actual_task = actual_after_write["buffers"][task]
            self.assertEqual(expected_task["idx"], 0)
            rewards = actual_task["replay_buffer"]["_storage"]["_storage"]["reward"]
            self.assertEqual(rewards.flatten().tolist(), expected_rewards)
            assert_tensor_task_state_equal(self, expected_task, actual_task)

        torch.manual_seed(44)
        expected = tensor.sample()
        torch.manual_seed(44)
        actual = restored.sample()
        assert_batch_equal(self, expected, actual)

    def test_ensemble_matches_direct_and_rng(self):
        tensor = TensorGNNBuffer(config())
        populate(tensor)
        torch.manual_seed(7)
        direct = tensor.sample()
        direct_rng = torch.random.get_rng_state()
        torch.manual_seed(7)
        ensemble = tensor.sample_ensemble()
        ensemble_rng = torch.random.get_rng_state()
        assert_batch_equal(self, direct, ensemble)
        self.assertTrue(torch.equal(direct_rng, ensemble_rng))

    def test_repeated_agent_updates_match_legacy_exactly(self):
        cfg = legacy_agent_cfg(buffer_size=16)
        cfg.replay_storage = "cpu_pinned"
        legacy_agent, tensor_agent = GNNSAC(cfg), GNNSAC(cfg)
        tensor_agent.load_training_state_dict(legacy_agent.training_state_dict())
        legacy = populate_legacy_fixture(cfg)
        tensor = TensorGNNBuffer(cfg)
        for task, markers in ((cfg.tasks[0], (1, 2, 3, 4)), (cfg.tasks[1], (11, 12, 13, 14))):
            for marker in markers:
                tensor.add(legacy_transition(marker), task=task)
        torch.manual_seed(123)
        for _ in range(3):
            state = torch.random.get_rng_state()
            expected = legacy_agent.update(legacy)
            expected_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            actual = tensor_agent.update(tensor)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            assert_nested_equal(self, legacy_agent.training_state_dict(), tensor_agent.training_state_dict())
            torch.random.set_rng_state(expected_rng)

    def test_pcgrad_update_matches_legacy_exactly(self):
        cfg = legacy_agent_cfg(buffer_size=16, pcgrad=True)
        cfg.replay_storage = "cpu_pinned"
        legacy_agent, tensor_agent = GNNSAC(cfg), GNNSAC(cfg)
        tensor_agent.load_training_state_dict(legacy_agent.training_state_dict())
        legacy = populate_legacy_fixture(cfg)
        tensor = TensorGNNBuffer(cfg)
        for task, markers in ((cfg.tasks[0], (1, 2, 3, 4)), (cfg.tasks[1], (11, 12, 13, 14))):
            for marker in markers:
                tensor.add(legacy_transition(marker), task=task)
        torch.manual_seed(321)
        expected = legacy_agent.update(legacy, compute_diagnostics=True)
        torch.manual_seed(321)
        actual = tensor_agent.update(tensor, compute_diagnostics=True)
        assert_nested_equal(self, expected, actual)
        assert_nested_equal(self, legacy_agent.training_state_dict(), tensor_agent.training_state_dict())

    def test_auto_falls_back_to_cpu_with_zero_budget(self):
        cfg = config(device="cuda", replay_storage="auto", replay_gpu_fraction=0.)
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.mem_get_info", return_value=(16 * 1024 ** 3, 24 * 1024 ** 3)
        ):
            tensor = TensorGNNBuffer(cfg)
        self.assertEqual(set(tensor.placement_metadata["placements"].values()), {"cpu"})

    def test_auto_preserves_indexed_cuda_device_for_budget_and_placement(self):
        cfg = config(
            device="cuda:1", replay_storage="auto", replay_gpu_fraction=1.,
            replay_gpu_max_gb=100., replay_gpu_reserve_gb=0.,
        )
        configured_device = torch.device("cuda:1")
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.mem_get_info", return_value=(16 * 1024 ** 3, 24 * 1024 ** 3)
        ) as mem_get_info:
            tensor = TensorGNNBuffer(cfg)
        mem_get_info.assert_called_once_with(configured_device)
        self.assertEqual(
            set(tensor.placement_metadata["placements"].values()),
            {str(configured_device)},
        )

    def test_legacy_checkpoint_conversion_is_non_destructive_and_exact(self):
        legacy = GNNBuffer(config(replay_backend="legacy"))
        populate(legacy)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.pt"
            destination = Path(directory) / "tensor.pt"
            torch.save({"buffer": legacy.state_dict(), "config": vars(config())}, source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            report = convert(source, destination, chunk_size=2)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertTrue(destination.exists())
            self.assertTrue(destination.with_suffix(".pt.conversion.json").exists())
            self.assertEqual(report["source_sha256"], before)
            converted_state = torch.load(destination, map_location="cpu", weights_only=False)
            restored = TensorGNNBuffer(config())
            restored.load_state_dict(converted_state["buffer"])
            torch.manual_seed(19)
            expected = legacy.sample()
            torch.manual_seed(19)
            actual = restored.sample()
            assert_batch_equal(self, expected, actual)

    def test_legacy_wrapped_nonzero_cursor_conversion_preserves_future_overwrites(self):
        cfg = config(buffer_size=8, replay_backend="legacy")
        legacy = GNNBuffer(cfg)
        populate(legacy, count=7)
        legacy_state = legacy.state_dict()
        for task, expected_rewards in (
            ("graph:a", [4., 5., 6., 3.]),
            ("graph:b", [14., 15., 16., 13.]),
        ):
            task_state = legacy_state["buffers"][task]
            self.assertEqual(
                (task_state["capacity"], task_state["size"], task_state["idx"]),
                (4, 4, 3),
            )
            self.assertEqual(
                [reward.item() for reward in task_state["reward"]],
                expected_rewards,
            )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy-wrapped.pt"
            destination = Path(directory) / "tensor-wrapped.pt"
            torch.save({"buffer": legacy_state, "config": vars(cfg)}, source)
            convert(source, destination, chunk_size=2)
            converted_state = torch.load(destination, map_location="cpu", weights_only=False)
            converted = TensorGNNBuffer(config(buffer_size=8))
            converted.load_state_dict(converted_state["buffer"])

            for task, expected_rewards in (
                ("graph:a", [4., 5., 6., 3.]),
                ("graph:b", [14., 15., 16., 13.]),
            ):
                task_state = converted.state_dict()["buffers"][task]
                self.assertEqual(
                    (task_state["capacity"], task_state["size"], task_state["idx"]),
                    (4, 4, 3),
                )
                rewards = task_state["replay_buffer"]["_storage"]["_storage"]["reward"]
                self.assertEqual(rewards.flatten().tolist(), expected_rewards)

            legacy.add(transition(100, 3), task="graph:a")
            legacy.add(transition(200, 5), task="graph:b")
            converted.add(transition(100, 3), task="graph:a")
            converted.add(transition(200, 5), task="graph:b")
            converted_after_write = converted.state_dict()
            for task, expected_rewards in (
                ("graph:a", [4., 5., 6., 100.]),
                ("graph:b", [14., 15., 16., 200.]),
            ):
                task_state = converted_after_write["buffers"][task]
                self.assertEqual(task_state["idx"], 0)
                rewards = task_state["replay_buffer"]["_storage"]["_storage"]["reward"]
                self.assertEqual(rewards.flatten().tolist(), expected_rewards)

            torch.manual_seed(73)
            expected = legacy.sample()
            expected_rng = torch.random.get_rng_state()
            torch.manual_seed(73)
            actual = converted.sample()
            actual_rng = torch.random.get_rng_state()
            assert_batch_equal(self, expected, actual)
            self.assertTrue(torch.equal(expected_rng, actual_rng))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_storage_returns_exact_batch(self):
        cuda_cfg = config(device="cuda", replay_storage="cuda", replay_gpu_fraction=1.,
                          replay_gpu_max_gb=100., replay_gpu_reserve_gb=0.)
        legacy, tensor = GNNBuffer(cuda_cfg), TensorGNNBuffer(cuda_cfg)
        populate(legacy)
        populate(tensor)
        torch.manual_seed(101)
        expected = legacy.sample()
        torch.manual_seed(101)
        actual = tensor.sample()
        assert_batch_equal(self, expected, actual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_auto_mixed_cuda_cpu_placement_is_exact(self):
        first_estimate = 8 * ((2 * 3 * 6 + 3 + 4) * 4 + 8)
        mixed_cfg = config(
            device="cuda", replay_storage="auto", replay_gpu_fraction=1.,
            replay_gpu_max_gb=(first_estimate + 1) / 1024 ** 3,
            replay_gpu_reserve_gb=0.,
        )
        legacy = GNNBuffer(mixed_cfg)
        tensor = TensorGNNBuffer(config(
            device="cuda", replay_storage="auto", replay_gpu_fraction=1.,
            replay_gpu_max_gb=(first_estimate + 1) / 1024 ** 3,
            replay_gpu_reserve_gb=0.,
        ))
        self.assertEqual(list(tensor.placement_metadata["placements"].values()), ["cuda", "cpu"])
        populate(legacy)
        populate(tensor)
        torch.manual_seed(27)
        expected = legacy.sample()
        torch.manual_seed(27)
        actual = tensor.sample()
        assert_batch_equal(self, expected, actual)

    def test_task_names_fans_out_broken_regime_sibling_when_interleaved(self):
        base_cfg = config(
            task="graph", tasks=None, multitask=False, mujoco_backend="mujoco",
            truss_topologies=None, num_envs=4,
            domain_randomization=True,
            domain_randomization_params={
                "broken_nodes": {
                    "enabled": True, "probability": 0.2, "regime_fraction": 0.5,
                },
            },
            graph_features={"node_roles": True}, use_control_graph=True,
        )
        self.assertEqual(_task_names(base_cfg), ["graph", "graph__broken"])

        # Disabling either the fraction, the schedule, or num_envs>1 must
        # leave today's task registry untouched.
        self.assertEqual(
            _task_names(SimpleNamespace(**{**vars(base_cfg), "num_envs": 1})),
            ["graph"],
        )
        staged = SimpleNamespace(**{**vars(base_cfg)})
        staged.domain_randomization_params = dict(base_cfg.domain_randomization_params)
        staged.domain_randomization_params["broken_nodes"] = {
            **base_cfg.domain_randomization_params["broken_nodes"], "schedule": "staged",
        }
        self.assertEqual(_task_names(staged), ["graph"])
        self.assertEqual(
            _task_names(config(task="graph", tasks=None, multitask=False, num_envs=4)),
            ["graph"],
        )

        # regime_fraction=1.0 makes every slot broken-eligible, so a standard
        # sibling task would never receive transitions and stall replay
        # readiness forever -- only the broken task should be registered.
        all_broken = SimpleNamespace(**{**vars(base_cfg)})
        all_broken.domain_randomization_params = dict(base_cfg.domain_randomization_params)
        all_broken.domain_randomization_params["broken_nodes"] = {
            **base_cfg.domain_randomization_params["broken_nodes"], "regime_fraction": 1.0,
        }
        self.assertEqual(_task_names(all_broken), ["graph__broken"])


if __name__ == "__main__":
    unittest.main()
