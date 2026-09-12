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
from common.tensor_gnn_buffer import TensorGNNBuffer, make_gnn_buffer
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


class TensorGNNBufferTest(unittest.TestCase):
    def test_factory_keeps_legacy_default(self):
        self.assertIsInstance(make_gnn_buffer(config(replay_backend="legacy")), GNNBuffer)
        self.assertIsInstance(make_gnn_buffer(config()), TensorGNNBuffer)

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
        self.assertTrue(torch.equal(expected_rng, actual_rng))

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

    def test_ring_wrap_and_checkpoint_round_trip(self):
        tensor = TensorGNNBuffer(config(buffer_size=8))
        populate(tensor, count=7)
        state = tensor.state_dict()
        restored = TensorGNNBuffer(config(buffer_size=8))
        restored.load_state_dict(state)
        self.assertEqual(restored.sizes_by_task, tensor.sizes_by_task)
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


if __name__ == "__main__":
    unittest.main()
