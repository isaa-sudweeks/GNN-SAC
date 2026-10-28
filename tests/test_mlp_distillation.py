"""Cross-backend distillation contracts and padded-policy regression tests."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sac"))

from common.distillation import (
    Distillation, _validate_action_alignment, gaussian_forward_kl, graph_mean_kl,
)
from common.gnn_buffer import GNNBuffer
from common.tensor_gnn_buffer import TensorGNNBuffer
from common.graph_transforms import prepare_graph
from gnn_sac import GNNSAC
from padded_mlp_sac import PaddedMLPSAC
from tests.test_distillation import config
from tests.test_gnn_batched_inference import graph


def observation(nodes, marker=0.0):
    raw = graph(nodes, offset=marker)
    raw.action_mask = torch.arange(nodes) != nodes - 1
    raw.rigidity = torch.tensor([0.6 + marker / 10])
    return raw


def student_config(directory, **overrides):
    values = dict(
        sac_backend="padded_mlp", use_virtual_node=False,
        padded_mlp_max_nodes=5, padded_mlp_hidden_dims=[16, 16],
        node_action_dim=1, use_control_graph=True, graph_features={},
    )
    values.update(overrides)
    return config(directory, **values)


def teacher_file(directory, topology, nodes, backend="gnn", **overrides):
    values = dict(
        sac_backend=backend, truss_topology=topology, truss_topologies=None,
        multitask=False, tasks=["truss-graph"],
        use_virtual_node=(backend == "gnn"),
    )
    values.update(overrides)
    cfg = student_config(directory, **values)
    agent = GNNSAC(cfg) if backend == "gnn" else PaddedMLPSAC(cfg)
    observations = [observation(nodes, marker=i / 10) for i in range(5)]
    for raw in observations:
        raw.edge_role = torch.arange(raw.edge_index.size(1)).remainder(2)
    path = Path(directory) / f"{topology}-{backend}.pt"
    torch.save(dict(config=vars(cfg), agent=agent.training_state_dict(),
                    buffer=dict(capacity=5, size=5, idx=0, obs=observations)), path)
    return str(path), observations


def setup(directory, backend="gnn", **student_overrides):
    cfg = student_config(directory, **student_overrides)
    raw = {}
    for topology, nodes in (("a", 3), ("b", 5)):
        path, observations = teacher_file(directory, topology, nodes, backend)
        cfg.distillation["teachers"][topology] = path
        raw[f"truss-graph:{topology}"] = observations
    cfg.distillation["teachers"]["heldout"] = "/teacher/must/not/be/opened.pt"
    distill = Distillation(cfg, cfg.tasks)
    return cfg, distill, raw


class PaddedDistributionTest(unittest.TestCase):
    def test_gaussian_interface_preserves_rng_mode_and_active_action_order(self):
        agent = PaddedMLPSAC(student_config("/tmp"))
        agent.model.train()
        batch = Batch.from_data_list([observation(3), observation(5)])
        rng = torch.random.get_rng_state().clone()
        mean, log_std = agent.model.policy_distribution(batch)
        torch.testing.assert_close(rng, torch.random.get_rng_state())
        self.assertTrue(agent.model.training)
        self.assertEqual(mean.shape, (6, 1))
        self.assertTrue(mean.requires_grad)
        self.assertTrue(log_std.requires_grad)
        torch.testing.assert_close(mean.tanh(), agent.model.pi_mean(batch))
        serial = [agent.model.policy_distribution(raw) for raw in batch.to_data_list()]
        torch.testing.assert_close(mean, torch.cat([value[0] for value in serial]))
        torch.testing.assert_close(log_std, torch.cat([value[1] for value in serial]))

    def test_virtual_architecture_preserves_physical_order_but_rejects_active_virtual(self):
        raw = observation(3)
        virtual = prepare_graph(raw, use_virtual_node=True)
        _validate_action_alignment(virtual, raw)
        virtual.action_mask[-1] = True
        with self.assertRaisesRegex(ValueError, "action ordering"):
            _validate_action_alignment(virtual, raw)


class MLPDistillationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_virtual_teacher_cached_targets_equal_live_teacher_and_online_kl(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill, observations = setup(tmp)
            self.addCleanup(lambda: [dataset.close() for dataset in distill.datasets.values()])
            student = PaddedMLPSAC(cfg)
            for task, dataset in distill.datasets.items():
                batch, cached_mean, cached_log_std = dataset.resolve(
                    {"shard": 0, "indices": [0, 1]}, torch.device("cpu")
                )
                raw = observations[task][:2]
                self.assertIsNotNone(batch.rigidity)
                self.assertEqual(batch.rigidity.shape, (2, 1))
                teacher_batch = Batch.from_data_list([distill.prepare_teacher(task, item) for item in raw])
                with torch.no_grad():
                    live_mean, live_log_std = distill.teachers[task].policy_distribution(teacher_batch)
                torch.testing.assert_close(cached_mean, live_mean, atol=0, rtol=0)
                torch.testing.assert_close(cached_log_std, live_log_std, atol=0, rtol=0)
                online = distill.loss(student.model, task, batch, raw)
                offline = distill.target_loss(student.model, batch, cached_mean, cached_log_std)
                torch.testing.assert_close(online, offline)
                with self.assertRaisesRegex(ValueError, "raw"):
                    distill.loss(student.model, task, batch)

    def test_equal_padded_views_reuse_batch_without_raw_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill, _ = setup(tmp, backend="padded_mlp")
            self.addCleanup(lambda: [dataset.close() for dataset in distill.datasets.values()])
            task = cfg.tasks[0]
            batch, tm, tl = distill.datasets[task].resolve(
                {"shard": 0, "indices": [0, 1]}, torch.device("cpu")
            )
            student = PaddedMLPSAC(cfg)
            with patch.object(distill, "prepare_teacher", side_effect=AssertionError("view rebuilt")):
                online = distill.loss(student.model, task, batch)
            torch.testing.assert_close(online, distill.target_loss(student.model, batch, tm, tl))

    def test_bound_replay_online_loss_skips_per_graph_cpu_contract_checks(self):
        for buffer_type in (GNNBuffer, TensorGNNBuffer):
            with self.subTest(buffer=buffer_type.__name__), tempfile.TemporaryDirectory() as tmp:
                cfg, distill, observations = setup(tmp)
                self.addCleanup(lambda d=distill: [dataset.close() for dataset in d.datasets.values()])
                replay = buffer_type(cfg)
                distill.bind_replay(replay)
                for task, raw in observations.items():
                    replay.add([
                        dict(obs=item, action=torch.zeros(item.num_nodes, 1),
                             reward=torch.tensor(0.), terminated=torch.tensor(False))
                        for item in raw
                    ], task=task)
                agent = PaddedMLPSAC(cfg)
                agent.distillation = distill
                with patch("common.distillation._validate_action_alignment", side_effect=AssertionError("hot alignment")), \
                     patch("common.distillation.graph_signature", side_effect=AssertionError("hot signature")), \
                     patch.object(distill, "prepare_student", side_effect=AssertionError("student view rebuilt")):
                    metrics = agent.update(replay)
                self.assertGreater(float(metrics["distillation/kl"]), 0)
                # Skipping learner-side checks must not bypass the cold write
                # boundary that makes this optimization safe.
                wrong = observation(5)
                with self.assertRaises(ValueError):
                    replay.add([
                        dict(obs=wrong, action=torch.zeros(5, 1), reward=torch.tensor(0.),
                             terminated=torch.tensor(False))
                    ] * 3, task=cfg.tasks[0])

    def test_unbound_online_loss_rejects_raw_graph_from_another_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill, observations = setup(tmp)
            self.addCleanup(lambda: [dataset.close() for dataset in distill.datasets.values()])
            agent = PaddedMLPSAC(cfg)
            task = cfg.tasks[0]
            batch = Batch.from_data_list(observations[task][:2])
            with self.assertRaisesRegex(ValueError, "teacher task contract"):
                distill.loss(agent.model, task, batch, observations[cfg.tasks[1]][:2])

    def test_richer_teacher_features_do_not_expand_padded_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = student_config(tmp, truss_topologies=["a"], tasks=["truss-graph:a"])
            path, raw = teacher_file(tmp, "a", 3, graph_features={
                "node_roles": True, "edge_roles": True, "edge_distance": True,
            })
            cfg.distillation["teachers"] = {"a": path}
            distill = Distillation(cfg, cfg.tasks)
            self.addCleanup(lambda: [dataset.close() for dataset in distill.datasets.values()])
            batch, tm, tl = distill.datasets[cfg.tasks[0]].resolve(
                {"shard": 0, "indices": [0, 1]}, torch.device("cpu")
            )
            self.assertEqual(batch.x.size(1), cfg.obs_dim)
            self.assertIsNone(getattr(batch, "edge_attr", None))
            student = PaddedMLPSAC(cfg)
            torch.testing.assert_close(
                distill.loss(student.model, cfg.tasks[0], batch, raw[:2]),
                distill.target_loss(student.model, batch, tm, tl),
            )

    def test_offline_kl_decreases_without_changing_critics_alpha_or_teachers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill, _ = setup(tmp, backend="padded_mlp", lr=2e-3)
            self.addCleanup(lambda: [dataset.close() for dataset in distill.datasets.values()])
            distill.pretrain_updates = 40
            agent = PaddedMLPSAC(cfg)
            before_critics = deepcopy(agent.model._Qs.state_dict())
            before_alpha = agent.log_alpha.detach().clone()
            before_teachers = {task: deepcopy(model.state_dict()) for task, model in distill.teachers.items()}
            task = cfg.tasks[0]
            fixture = distill.datasets[task].resolve({"shard": 0, "indices": [0, 1]}, torch.device("cpu"))
            before = float(distill.target_loss(agent.model, *fixture).detach())
            for _ in range(40):
                distill.offline_update(agent)
            after = float(distill.target_loss(agent.model, *fixture).detach())
            self.assertLess(after, before)
            for key, value in before_critics.items():
                torch.testing.assert_close(agent.model._Qs.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(agent.log_alpha, before_alpha, atol=0, rtol=0)
            for task, teacher in distill.teachers.items():
                for key, value in before_teachers[task].items():
                    torch.testing.assert_close(teacher.state_dict()[key], value, atol=0, rtol=0)
                self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_online_updates_and_zero_weight_equivalence_for_both_replays_and_optimizers(self):
        for buffer_type in (GNNBuffer, TensorGNNBuffer):
            for pcgrad in (False, True):
                with self.subTest(buffer=buffer_type.__name__, pcgrad=pcgrad), tempfile.TemporaryDirectory() as tmp:
                    cfg, distill, observations = setup(tmp, pcgrad=pcgrad)
                    self.addCleanup(lambda d=distill: [dataset.close() for dataset in d.datasets.values()])
                    buffer = buffer_type(cfg)
                    for task, raw in observations.items():
                        transitions = [dict(obs=item, action=torch.zeros(item.num_nodes, 1),
                                            reward=torch.tensor(0.), terminated=torch.tensor(False))
                                       for item in raw]
                        buffer.add(transitions, task=task)
                    distill.bind_replay(buffer)
                    agent = PaddedMLPSAC(cfg)
                    agent.distillation = distill
                    result = agent.update(buffer)
                    self.assertTrue(torch.isfinite(result["pi_loss"]))
                    self.assertGreater(float(result["distillation/kl"]), 0)
                    distill.step = cfg.steps
                    baseline = PaddedMLPSAC(cfg)
                    baseline.load_training_state_dict(deepcopy(agent.training_state_dict()))
                    rng = torch.random.get_rng_state()
                    with patch.object(distill, "loss", side_effect=AssertionError("teacher called at zero weight")):
                        agent.update(buffer)
                    torch.random.set_rng_state(rng)
                    baseline.update(buffer)
                    for key, value in baseline.model.state_dict().items():
                        torch.testing.assert_close(value, agent.model.state_dict()[key], rtol=0, atol=0)

    def test_padded_teacher_schema_is_required_and_checked(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                cfg = student_config(tmp, truss_topologies=["a"], tasks=["truss-graph:a"])
                path, _ = teacher_file(tmp, "a", 3, backend="padded_mlp")
                state = torch.load(path, weights_only=False)
                if missing:
                    del state["agent"]["padded_mlp_schema"]
                else:
                    state["agent"]["padded_mlp_schema"]["max_nodes"] += 1
                torch.save(state, path)
                cfg.distillation["teachers"] = {"a": path}
                with self.assertRaisesRegex(ValueError, "padded MLP schema"):
                    Distillation(cfg, cfg.tasks)

    def test_capacity_changes_separate_cache_and_reject_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, first, _ = setup(tmp)
            self.addCleanup(lambda: [dataset.close() for dataset in first.datasets.values()])
            changed = deepcopy(cfg)
            changed.padded_mlp_max_nodes += 1
            second = Distillation(changed, changed.tasks)
            self.addCleanup(lambda: [dataset.close() for dataset in second.datasets.values()])
            for task in first.datasets:
                self.assertNotEqual(first.datasets[task].directory, second.datasets[task].directory)
            with self.assertRaisesRegex(ValueError, "preparation contracts"):
                second.load_state_dict(first.state_dict())
            legacy = first.state_dict()
            del legacy["policy_contracts"]
            with self.assertRaisesRegex(ValueError, "Legacy"):
                first.load_state_dict(legacy)

    def test_exact_offline_resume_restores_next_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, first, _ = setup(tmp)
            self.addCleanup(lambda: [dataset.close() for dataset in first.datasets.values()])
            agent = PaddedMLPSAC(cfg)
            first.offline_update(agent)
            snapshot = deepcopy(first.state_dict())
            agent_state = deepcopy(agent.training_state_dict())
            expected = first.offline_update(agent)
            resumed = Distillation(cfg, cfg.tasks, defer_target_cache=True)
            self.addCleanup(lambda: [dataset.close() for dataset in resumed.datasets.values()])
            restored = PaddedMLPSAC(cfg)
            restored.load_training_state_dict(agent_state)
            resumed.load_state_dict(snapshot)
            actual = resumed.offline_update(restored)
            self.assertEqual(actual, expected)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[key], value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
