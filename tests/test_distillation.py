from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sac"))

from common.distillation import (
    TARGET_CACHE_FORMAT, Distillation, ObservationShards, _validate_action_alignment,
    gaussian_forward_kl, graph_mean_kl, graph_signature, replay_observations,
)
from common.gnn_actor_critic import GNNActorCritic
from common.gnn_buffer import GNNBuffer
from common.tensor_gnn_buffer import TensorGNNBuffer
from common.graph_transforms import prepare_graph
from common.logger import Logger, VideoRecorder
from gnn_sac import GNNSAC
from trainer.base import Trainer
from trainer.online_trainer import OnlineTrainer
from tests.test_gnn_batched_inference import agent_cfg, graph
from tests.test_checkpointing import DummyLogger


def config(directory, **overrides):
    values = vars(agent_cfg()).copy()
    values.update(sac_backend="gnn", task="truss-graph", truss_topology="a",
                  truss_topologies=["a", "b"], tasks=["truss-graph:a", "truss-graph:b"],
                  multitask=True, steps=100, buffer_size=16, batch_size=4, seed=3,
                  work_dir=str(directory), checkpoint_freq=0, use_virtual_node=True,
                  target_entropy=-1, log_std_min=-2.0, log_std_max=0.0,
                  distillation=dict(enabled=True, teachers={}, pretrain_updates=3,
                                    batch_size=4, initial_weight=1., decay_fraction=.5,
                                    shard_size=2, checkpoint_freq=0, log_freq=1))
    values.update(overrides)
    return SimpleNamespace(**values)


def raw_graph(nodes):
    obs = graph(nodes)
    obs.action_mask = torch.arange(nodes) != nodes - 1
    return obs


def teacher_checkpoint(directory, topology, nodes, *, raw_edge_roles=None, **overrides):
    cfg = config(directory, truss_topology=topology, truss_topologies=None,
                 multitask=False, tasks=["truss-graph"], **overrides)
    agent = GNNSAC(cfg)
    observations = [raw_graph(nodes) for _ in range(5)]
    features = getattr(cfg, "graph_features", {})
    edge_roles = features.get("edge_roles", False) if hasattr(features, "get") else False
    for index, observation in enumerate(observations):
        # Exercise dynamic node-derived edge features across cache entries.
        observation.x.mul_(1.0 + 0.1 * index)
        if edge_roles if raw_edge_roles is None else raw_edge_roles:
            observation.edge_role = torch.arange(
                observation.edge_index.size(1), dtype=torch.long
            ).remainder(2)
    replay = dict(capacity=5, size=5, idx=2, obs=observations)
    path = Path(directory) / f"{topology}.pt"
    torch.save(dict(config=vars(cfg), agent=agent.training_state_dict(), buffer=replay), path)
    return str(path)


def setup(directory, **overrides):
    cfg = config(directory, **overrides)
    cfg.distillation["teachers"] = {
        "a": teacher_checkpoint(directory, "a", 3, embedding_dim=8),
        "b": teacher_checkpoint(directory, "b", 5, embedding_dim=12),
        "heldout": "/does/not/exist.pt",
    }
    return cfg, Distillation(cfg, cfg.tasks)


def replay_buffer(cfg, buffer_type=GNNBuffer):
    buffer = buffer_type(cfg)
    for task, nodes in zip(cfg.tasks, (3, 5)):
        obs = raw_graph(nodes)
        item = dict(obs=obs, action=torch.zeros(nodes, 1), reward=torch.tensor(0.), terminated=torch.tensor(False))
        buffer.add([item] * 5, task=task)
    return buffer


class GaussianKLTest(unittest.TestCase):
    def test_matches_pytorch_and_trains_mean_and_variance(self):
        tm = torch.tensor([[0.1, -.2]], dtype=torch.double)
        tl = torch.tensor([[-1., -.5]], dtype=torch.double)
        sm = torch.tensor([[.3, .1]], dtype=torch.double, requires_grad=True)
        sl = torch.tensor([[-.7, -.2]], dtype=torch.double, requires_grad=True)
        actual = gaussian_forward_kl(tm, tl, sm, sl)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(tm, tl.exp()), torch.distributions.Normal(sm, sl.exp()))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(sm.grad.ne(0).all())
        self.assertTrue(sl.grad.ne(0).all())
        torch.testing.assert_close(gaussian_forward_kl(tm, tl, tm, tl), torch.zeros_like(tm))
        self.assertTrue((gaussian_forward_kl(tm, tl, tm, tl + .2) > 0).all())

    def test_invalid_distributions_and_node_normalization(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            gaussian_forward_kl(torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        batch = Batch.from_data_list([prepare_graph(raw_graph(n), use_virtual_node=True) for n in (3, 5)])
        # 2 active nodes at KL=2, 4 at KL=6 => graph mean 4, not node mean 14/3.
        kl = torch.tensor([[2.]] * 2 + [[6.]] * 4)
        self.assertEqual(float(graph_mean_kl(kl, batch)), 4.)

    def test_distribution_interface_preserves_policy_and_rng(self):
        cfg = config("/tmp")
        agent = GNNSAC(cfg)
        obs = prepare_graph(raw_graph(3), use_virtual_node=True)
        rng = torch.random.get_rng_state()
        mean, log_std = agent.model.policy_distribution(obs)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(mean.tanh(), agent.model.pi_mean(obs))
        _, info = agent.model.pi(obs)
        torch.testing.assert_close(info["mean"], mean.tanh())
        torch.testing.assert_close(info["log_std"], log_std)

    def test_distribution_interface_disables_dropout_and_restores_mode(self):
        agent = GNNSAC(config("/tmp", dropout=.5))
        agent.model.train()
        obs = prepare_graph(raw_graph(3), use_virtual_node=True)
        rng = torch.random.get_rng_state()
        first = agent.model.policy_distribution(obs)
        second = agent.model.policy_distribution(obs)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(agent.model._pi.training)
        self.assertTrue(agent.model._action_head.training)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)


class TeacherReplayTest(unittest.TestCase):
    def test_teacher_student_views_require_identical_action_ordering(self):
        teacher = prepare_graph(raw_graph(3), use_virtual_node=True)
        student = teacher.clone()
        student.action_mask = student.action_mask.roll(1)
        with self.assertRaisesRegex(ValueError, "action ordering"):
            _validate_action_alignment(teacher, student)

    def test_hydra_preset_and_teacher_mapping(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            cfg = compose(config_name="archieved/gnn_config", overrides=[
                "distillation=kl", "+distillation.teachers={a:/tmp/a.pt,b:/tmp/b.pt}"])
        self.assertTrue(cfg.distillation.enabled)
        self.assertEqual(cfg.distillation.pretrain_updates, 10000)
        self.assertEqual(cfg.distillation.batch_size, 256)
        self.assertEqual(cfg.distillation.teachers.a, "/tmp/a.pt")

    def test_partial_and_wrapped_replay_and_shard_reuse(self):
        values = [raw_graph(3) for _ in range(5)]
        for i, value in enumerate(values):
            value.x.fill_(i)
        partial = dict(capacity=5, size=3, idx=3, obs=values[:3] + [None, None])
        self.assertEqual([int(g.x[0, 0]) for g in replay_observations(partial)], [0, 1, 2])
        full = dict(capacity=5, size=5, idx=2, obs=values)
        self.assertEqual([int(g.x[0, 0]) for g in replay_observations(full)], [2, 3, 4, 0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp)
            teacher = GNNSAC(cfg).model.eval()
            prepare = lambda value: prepare_graph(value, use_virtual_node=True)
            contract = {
                "format": TARGET_CACHE_FORMAT,
                "replay_signature": graph_signature(values[0]),
                "prepared_signature": graph_signature(prepare(values[0])),
            }
            path = Path(tmp) / "target-shards"
            dataset = ObservationShards.prepare(
                path, replay_observations(full), 2, contract, teacher, prepare,
                torch.device("cpu"), 2, prefetch=False,
            )
            self.assertEqual(dataset.sizes.tolist(), [2, 2, 1])
            rng = torch.Generator().manual_seed(5)
            counts = torch.zeros(5)
            for _ in range(1000):
                batch, _, _ = dataset.sample(4, rng)
                for value in batch.x[::4, 0]:
                    counts[int(value)] += 1
            self.assertTrue((counts > 600).all(), counts)
            self.assertTrue((counts < 1000).all(), counts)
            self.assertEqual(
                ObservationShards.prepare(
                    path, iter(()), 2, contract, teacher, prepare,
                    torch.device("cpu"), 2, prefetch=False,
                ).sizes.tolist(),
                [2, 2, 1],
            )
            changed = deepcopy(contract)
            changed["prepared_signature"] = graph_signature(prepare(raw_graph(4)))
            with self.assertRaisesRegex(ValueError, "schema|signature"):
                ObservationShards(
                    path, changed, prefetch=False,
                )
            old = Path(tmp) / "observation-only"
            old.mkdir()
            (old / "manifest.json").write_text(json.dumps({"sizes": [1], "signature": {}}))
            with self.assertRaisesRegex(ValueError, "teacher-target"):
                ObservationShards(old, contract, prefetch=False)
        with self.assertRaises(ValueError):
            list(replay_observations(dict(full, size=0)))

    def test_tensor_v3_and_v4_replay_observations_preserve_ring_order_and_masks(self):
        cfg = config("/tmp", replay_backend="torchrl_tensor", replay_storage="cpu_pinned", buffer_size=8,
                     node_counts=[3, 5], obs_dim=6, action_dim=1, graph_features={})
        replay = TensorGNNBuffer(cfg)
        for marker in range(6):
            obs = raw_graph(3)
            obs.x.fill_(marker)
            following = raw_graph(3)
            following.x.fill_(marker + .5)
            if marker % 2:
                obs.action_mask[1] = False
            if not marker % 2:
                following.action_mask[1] = False
            item = lambda graph: dict(
                obs=graph, action=torch.zeros(1, 3, 1), reward=torch.zeros(1),
                terminated=torch.zeros(1),
            )
            replay.add([item(obs), item(following)], task=cfg.tasks[0])
        v4 = replay.state_dict()["buffers"][cfg.tasks[0]]
        v3 = deepcopy(v4)
        v3["format_version"] = 3
        storage = v3["replay_buffer"]["_storage"]["_storage"]
        del storage["obs_action_mask"]
        del storage["next_obs_action_mask"]

        for version, task_state in ((3, v3), (4, v4)):
            with self.subTest(version=version):
                observations = list(replay_observations(task_state))
                self.assertEqual([int(graph.x[0, 0]) for graph in observations], [2, 3, 4, 5])
                expected_masks = (
                    [task_state["static"]["action_mask"]] * 4
                    if version == 3
                    else [
                        torch.tensor([True, True, False]),
                        torch.tensor([True, False, False]),
                        torch.tensor([True, True, False]),
                        torch.tensor([True, False, False]),
                    ]
                )
                for graph, expected_mask in zip(observations, expected_masks):
                    torch.testing.assert_close(graph.action_mask, expected_mask)

    def test_frozen_teachers_routing_and_split_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            self.assertEqual(set(distill.teachers), set(cfg.tasks))
            self.assertTrue(all(not p.requires_grad for t in distill.teachers.values() for p in t.parameters()))
            self.assertTrue(all(not t.training for t in distill.teachers.values()))
            self.assertTrue(all(
                p.device == torch.device(cfg.device)
                for teacher in distill.teachers.values()
                for p in teacher.parameters()
            ))
            agent = GNNSAC(cfg)
            obs = Batch.from_data_list([distill.prepare(raw_graph(3))])
            teacher = distill.teachers[cfg.tasks[0]]
            with patch.object(teacher, "to", side_effect=AssertionError("teacher moved in hot loop")), \
                    patch("common.distillation.graph_signature",
                          side_effect=AssertionError("signature checked in hot loop")):
                distill.loss(agent.model, cfg.tasks[0], obs).backward()
            self.assertTrue(all(p.grad is None for t in distill.teachers.values() for p in t.parameters()))
            optimizer_ids = {
                id(parameter)
                for group in agent.pi_optim.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(all(
                id(parameter) not in optimizer_ids
                for teacher in distill.teachers.values()
                for parameter in teacher.parameters()
            ))
            with self.assertRaisesRegex(ValueError, "Missing"):
                Distillation(cfg, ["truss-graph:missing"])
            cfg.obs_dim = 4
            with self.assertRaisesRegex(ValueError, "convention"):
                Distillation(cfg, cfg.tasks)

    def test_bad_topology_schema_and_control_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = setup(tmp)
            path = cfg.distillation["teachers"]["a"]
            original = torch.load(path, weights_only=False)
            for key, value in (("truss_topology", "wrong"), ("speed", 99),
                               ("use_virtual_node", False)):
                bad = deepcopy(original)
                bad["config"][key] = value
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    Distillation(cfg, cfg.tasks)
            torch.save(original, path)

    def test_online_replay_validates_teacher_structure_at_write_and_load_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            buffer = GNNBuffer(cfg)
            distill.bind_replay(buffer)
            bad = raw_graph(3)
            item = dict(obs=bad, action=torch.zeros(3, 1), reward=torch.tensor(0.),
                        terminated=torch.tensor(False))
            with self.assertRaisesRegex(ValueError, "ordering"):
                buffer.add([item] * 2, task=cfg.tasks[1])

            valid = replay_buffer(cfg)
            state = valid.state_dict()
            state["buffers"][cfg.tasks[1]]["obs"][0] = raw_graph(3)
            empty = GNNBuffer(cfg)
            distill.bind_replay(empty)
            with self.assertRaisesRegex(ValueError, "ordering"):
                empty.load_state_dict(state)


class DistillationTrainingTest(unittest.TestCase):
    def test_node_role_schema_mismatch_offline_and_online_for_both_actor_paths(self):
        for pcgrad in (False, True):
            with self.subTest(pcgrad=pcgrad), tempfile.TemporaryDirectory() as tmp:
                student_features = dict(
                    node_roles=True, edge_roles=False, edge_distance=False
                )
                cfg = config(tmp, graph_features=student_features, pcgrad=pcgrad)
                cfg.distillation["teachers"] = {
                    topology: teacher_checkpoint(
                        tmp,
                        topology,
                        nodes,
                        embedding_dim=embedding_dim,
                        graph_features={},
                    )
                    for topology, nodes, embedding_dim in (
                        ("a", 3, 8),
                        ("b", 5, 12),
                    )
                }
                distill = Distillation(cfg, cfg.tasks)
                self.assertTrue(all(
                    pair["teacher"] != pair["student"]
                    for pair in distill.schema_pairs.values()
                ))

                task = cfg.tasks[0]
                dataset = distill.datasets[task]
                contract = dataset.metadata["contract"]
                self.assertIn("teacher_graph_feature_schema", contract)
                self.assertIn("student_graph_feature_schema", contract)
                batch, cached_mean, cached_log_std = dataset.resolve(
                    {"shard": 0, "indices": [0, 1]}, torch.device("cpu")
                )
                with open(cfg.distillation["teachers"]["a"], "rb") as stream:
                    replay = torch.load(stream, weights_only=False)["buffer"]
                raw = list(replay_observations(replay))[:2]
                teacher_batch = Batch.from_data_list([
                    distill.prepare_teacher(task, graph) for graph in raw
                ])
                with torch.no_grad():
                    live_mean, live_log_std = distill.teachers[task].policy_distribution(
                        teacher_batch
                    )
                torch.testing.assert_close(cached_mean, live_mean, rtol=0, atol=0)
                torch.testing.assert_close(cached_log_std, live_log_std, rtol=0, atol=0)
                self.assertEqual(batch.x.size(1), int(cfg.obs_dim) + 4)

                agent = GNNSAC(cfg)
                agent.distillation = distill
                buffer = replay_buffer(cfg)
                distill.bind_replay(buffer)
                info = agent.update(buffer)
                self.assertGreater(float(info["distillation/kl"]), 0)

                tensor_agent = GNNSAC(cfg)
                tensor_agent.distillation = distill
                tensor_buffer = replay_buffer(cfg, TensorGNNBuffer)
                distill.bind_replay(tensor_buffer)
                tensor_info = tensor_agent.update(tensor_buffer)
                self.assertGreater(float(tensor_info["distillation/kl"]), 0)

                legacy_state = distill.state_dict()
                legacy_state.pop("schema_pairs")
                with self.assertRaisesRegex(ValueError, "Legacy.*matching"):
                    distill.load_state_dict(legacy_state)

    def test_all_graph_feature_flags_can_differ_when_raw_metadata_is_available(self):
        cases = (
            ({"node_roles": True}, {}),
            ({}, {"node_roles": True}),
            ({"edge_distance": True}, {}),
            ({}, {"edge_distance": True}),
            ({"edge_roles": True}, {}),
            ({}, {"edge_roles": True}),
        )
        for student_features, teacher_features in cases:
            with self.subTest(student=student_features, teacher=teacher_features), \
                    tempfile.TemporaryDirectory() as tmp:
                cfg = config(tmp, graph_features=student_features)
                cfg.distillation["teachers"] = {
                    topology: teacher_checkpoint(
                        tmp,
                        topology,
                        nodes,
                        embedding_dim=embedding_dim,
                        graph_features=teacher_features,
                        raw_edge_roles=True,
                    )
                    for topology, nodes, embedding_dim in (
                        ("a", 3, 8),
                        ("b", 5, 12),
                    )
                }
                distill = Distillation(cfg, cfg.tasks)
                metrics = distill.offline_update(GNNSAC(cfg))
                self.assertTrue(torch.isfinite(torch.tensor(metrics["kl"])))

    def test_teacher_schemas_can_differ_by_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, graph_features={"edge_distance": True})
            cfg.distillation["teachers"] = {
                "a": teacher_checkpoint(
                    tmp, "a", 3, embedding_dim=8, graph_features={}
                ),
                "b": teacher_checkpoint(
                    tmp,
                    "b",
                    5,
                    embedding_dim=12,
                    graph_features={"node_roles": True},
                ),
            }
            distill = Distillation(cfg, cfg.tasks)
            self.assertNotEqual(
                distill.schema_pairs[cfg.tasks[0]]["teacher"],
                distill.schema_pairs[cfg.tasks[1]]["teacher"],
            )
            self.assertTrue(torch.isfinite(torch.tensor(
                distill.offline_update(GNNSAC(cfg))["kl"]
            )))

    def test_missing_schema_metadata_and_changed_schema_state_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, graph_features={"edge_roles": True})
            cfg.distillation["teachers"] = {
                topology: teacher_checkpoint(
                    tmp, topology, nodes, embedding_dim=8, graph_features={}
                )
                for topology, nodes in (("a", 3), ("b", 5))
            }
            with self.assertRaisesRegex(
                ValueError, "Cannot construct student.*edge_role"
            ):
                Distillation(cfg, cfg.tasks)

        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            changed = distill.state_dict()
            changed["schema_pairs"] = deepcopy(changed["schema_pairs"])
            changed["schema_pairs"][cfg.tasks[0]]["teacher"]["node_roles"] = True
            with self.assertRaisesRegex(ValueError, "schemas changed"):
                distill.load_state_dict(changed)

    def test_broken_node_curriculum_waits_for_zero_weight_and_propagates(self):
        class ScheduledDistillation:
            def __init__(self, initial_weight=1.0, decay_steps=10):
                self.initial_weight = initial_weight
                self.decay_steps = decay_steps
                self.step = -1

            @property
            def weight(self):
                return self.initial_weight * max(
                    0.0, 1.0 - self.step / self.decay_steps
                )

        class GateTarget:
            def __init__(self):
                self.enabled = None
                self.calls = []

            def set_broken_nodes_sampling_enabled(self, enabled):
                self.enabled = bool(enabled)
                self.calls.append(self.enabled)

        first, second, bucket = GateTarget(), GateTarget(), GateTarget()
        nested_env = SimpleNamespace(
            env=first,
            envs=[first, second],
            buckets=[bucket],
        )
        distillation = ScheduledDistillation()
        trainer = SimpleNamespace(env=nested_env, distillation=distillation, _step=9)

        self.assertFalse(OnlineTrainer._sync_broken_nodes_curriculum(trainer))
        self.assertEqual(distillation.step, 9)
        self.assertEqual([first.enabled, second.enabled, bucket.enabled], [False] * 3)
        self.assertEqual(first.calls, [False])

        # A resumed run at the boundary must enable sampling before its next reset,
        # even if the loaded distillation object's prior step was stale.
        trainer._step = 10
        distillation.step = 0
        self.assertTrue(OnlineTrainer._sync_broken_nodes_curriculum(trainer))
        self.assertEqual(distillation.step, 10)
        self.assertEqual([first.enabled, second.enabled, bucket.enabled], [True] * 3)

        zero_weight = ScheduledDistillation(initial_weight=0.0)
        immediate = SimpleNamespace(env=GateTarget(), distillation=zero_weight, _step=0)
        self.assertTrue(OnlineTrainer._sync_broken_nodes_curriculum(immediate))
        self.assertTrue(immediate.env.enabled)

        ordinary = SimpleNamespace(env=GateTarget(), distillation=None, _step=0)
        self.assertTrue(OnlineTrainer._sync_broken_nodes_curriculum(ordinary))
        self.assertTrue(ordinary.env.enabled)

    def test_cached_targets_match_live_teacher_with_all_graph_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = dict(node_roles=True, edge_roles=True, edge_distance=True)
            cfg = config(tmp, graph_features=features)
            cfg.distillation["teachers"] = {
                topology: teacher_checkpoint(
                    tmp,
                    topology,
                    nodes,
                    embedding_dim=embedding_dim,
                    graph_features=features,
                )
                for topology, nodes, embedding_dim in (("a", 3, 8), ("b", 5, 12))
            }
            distill = Distillation(cfg, cfg.tasks)
            for task, dataset in distill.datasets.items():
                batch, cached_mean, cached_log_std = dataset.resolve(
                    {"shard": 0, "indices": [0, 1]}, torch.device("cpu")
                )
                with torch.no_grad():
                    live_mean, live_log_std = distill.teachers[task].policy_distribution(batch)
                torch.testing.assert_close(cached_mean, live_mean, rtol=0, atol=0)
                torch.testing.assert_close(cached_log_std, live_log_std, rtol=0, atol=0)
                self.assertTrue((cached_log_std >= cfg.log_std_min).all())
                self.assertTrue((cached_log_std <= cfg.log_std_max).all())
                self.assertIsNotNone(batch.edge_attr)
                self.assertGreater(batch.edge_attr.size(1), 0)
            for dataset in distill.datasets.values():
                dataset.close()

    def test_target_cache_reuse_and_offline_updates_are_student_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, first = setup(tmp)
            cache_paths = {task: dataset.directory for task, dataset in first.datasets.items()}
            for dataset in first.datasets.values():
                self.assertIn("graph_feature_schema", dataset.metadata["contract"])
                self.assertNotIn(
                    "teacher_graph_feature_schema", dataset.metadata["contract"]
                )
            manifests = {task: (path / "manifest.json").stat().st_mtime_ns
                         for task, path in cache_paths.items()}
            for dataset in first.datasets.values():
                dataset.close()
            with patch.object(GNNActorCritic, "policy_distribution",
                              side_effect=AssertionError("teacher target cache recomputed")):
                reused = Distillation(cfg, cfg.tasks)
            self.assertEqual(
                {task: dataset.directory for task, dataset in reused.datasets.items()}, cache_paths
            )
            self.assertEqual(
                {task: (path / "manifest.json").stat().st_mtime_ns for task, path in cache_paths.items()},
                manifests,
            )
            agent = GNNSAC(cfg)
            for teacher in reused.teachers.values():
                teacher.policy_distribution = Mock(side_effect=AssertionError("teacher called offline"))
            reused.offline_update(agent)
            self.assertTrue(all(not teacher.policy_distribution.called
                                for teacher in reused.teachers.values()))
            datasets = list(reused.datasets.values())
            reused.pretrain_updates = reused.completed_updates
            reused.finish_pretraining(agent)
            self.assertTrue(all(dataset._executor is None for dataset in datasets))

    def test_online_only_skips_shards_and_optimizer_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = setup(tmp)
            cfg.distillation["pretrain_updates"] = 0
            with patch.object(ObservationShards, "prepare", side_effect=AssertionError("Unexpected offline data")):
                distill = Distillation(cfg, cfg.tasks)
            self.assertEqual(distill.stage, "online")
            self.assertFalse(distill.datasets)
            agent = GNNSAC(cfg)
            agent.pi_optim.state["sentinel"] = {}
            distill.finish_pretraining(agent)
            self.assertIn("sentinel", agent.pi_optim.state)

    def test_wandb_uses_monotonic_step_across_offline_and_online_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, enable_wandb=False, save_agent=False, save_csv=False)
            logger = Logger(cfg)
            logger._wandb = Mock()
            logger.log(dict(step=0, offline_updates=1, kl=1.), "distillation")
            logger.log(dict(step=0, offline_updates=2, kl=.5), "distillation")
            logger.log(dict(step=0, stage="online", offline_updates=3), "distillation")
            logger.log(dict(step=0, episode_reward=1.), "eval")
            logger.log(dict(step=7, episode_reward=2.), "eval")
            calls = logger._wandb.log.call_args_list
            self.assertEqual([call.kwargs["step"] for call in calls], [1, 2, 3, 3, 10])
            self.assertEqual([call.args[0]["global_step"] for call in calls], [1, 2, 3, 3, 10])
            self.assertEqual(calls[-1].args[0]["eval/step"], 7)

    def test_distillation_video_uses_global_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp)
            wandb = Mock()
            recorder = VideoRecorder(cfg, wandb)
            recorder.enabled = True
            recorder.frames = [torch.zeros((2, 2, 3), dtype=torch.uint8).numpy()]
            recorder.save(7)
            metrics = wandb.log.call_args.args[0]
            self.assertEqual(wandb.log.call_args.kwargs["step"], 10)
            self.assertEqual(metrics["global_step"], 10)
            self.assertEqual(metrics["eval/step"], 7)

    def test_offline_learning_only_changes_actor_and_transition_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            distill.pretrain_updates = 30
            agent = GNNSAC(cfg)
            critics = deepcopy(agent.model._Qs.state_dict())
            targets = deepcopy(agent.model._target_Qs.state_dict())
            alpha = agent.log_alpha.detach().clone()
            first = distill.offline_update(agent)["kl"]
            for _ in range(29):
                last = distill.offline_update(agent)["kl"]
            self.assertLess(last, first)
            for key, value in critics.items():
                torch.testing.assert_close(agent.model._Qs.state_dict()[key], value, rtol=0, atol=0)
            for key, value in targets.items():
                torch.testing.assert_close(agent.model._target_Qs.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(agent.log_alpha, alpha, rtol=0, atol=0)
            self.assertFalse(agent.q_optim.state)
            self.assertFalse(agent.alpha_optim.state)
            self.assertTrue(agent.pi_optim.state)
            distill.finish_pretraining(agent)
            self.assertFalse(agent.pi_optim.state)
            agent.pi_optim.state["sentinel"] = {}
            distill.finish_pretraining(agent)
            self.assertIn("sentinel", agent.pi_optim.state)

    def test_resume_matches_uninterrupted_offline_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            agent = GNNSAC(cfg)
            trainer = Trainer(cfg=cfg, env=None, agent=agent, buffer=GNNBuffer(cfg), logger=DummyLogger())
            trainer.distillation = distill
            distill.offline_update(agent)
            snapshot = trainer._checkpoint_state_snapshot()
            expected = distill.offline_update(agent)
            restored = GNNSAC(cfg)
            resumed = Trainer(cfg=cfg, env=None, agent=restored, buffer=GNNBuffer(cfg), logger=DummyLogger())
            resumed.distillation = Distillation(cfg, cfg.tasks, defer_target_cache=True)
            self.assertFalse(resumed.distillation.datasets)
            resumed.load_checkpoint_state_dict(snapshot)
            actual = resumed.distillation.offline_update(restored)
            self.assertEqual(actual, expected)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            changed = deepcopy(snapshot["distillation"])
            changed["sources"][cfg.tasks[0]]["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                resumed.distillation.load_state_dict(changed)

    def test_resume_allows_cache_performance_setting_changes_and_legacy_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            state = distill.state_dict()
            state["settings"].update(
                target_batch_size=123,
                pin_memory=False,
                prefetch=False,
                target_cache_format="newer-checkpoint-metadata",
            )
            cfg.distillation.update(target_batch_size=1, pin_memory=False, prefetch=False)
            restored = Distillation(cfg, cfg.tasks)
            restored.load_state_dict(state)
            self.assertEqual(restored.completed_updates, distill.completed_updates)

            legacy = deepcopy(state)
            for key in ("target_batch_size", "pin_memory", "prefetch", "target_cache_format"):
                legacy["settings"].pop(key, None)
            legacy.pop("pending_samples", None)
            restored.load_state_dict(legacy)

    def test_online_resume_skips_missing_target_cache_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            agent = GNNSAC(cfg)
            distill.completed_updates = distill.pretrain_updates
            distill.finish_pretraining(agent)
            online_state = distill.state_dict()

            cfg.distillation["cache_dir"] = str(Path(tmp) / "missing-target-cache")
            deferred = Distillation(cfg, cfg.tasks, defer_target_cache=True)
            self.assertFalse(deferred.datasets)
            deferred.load_state_dict(online_state)
            with patch.object(
                ObservationShards,
                "prepare",
                side_effect=AssertionError("online resume rebuilt target cache"),
            ):
                deferred.prepare_offline_datasets()
            self.assertEqual(deferred.stage, "online")
            self.assertFalse(deferred.datasets)

    def test_online_ordinary_and_pcgrad_and_zero_weight_equivalence(self):
        for pcgrad in (False, True):
            with self.subTest(pcgrad=pcgrad), tempfile.TemporaryDirectory() as tmp:
                cfg, distill = setup(tmp, pcgrad=pcgrad)
                agent = GNNSAC(cfg)
                agent.distillation = distill
                buffer = replay_buffer(cfg)
                distill.bind_replay(buffer)
                info = agent.update(buffer, compute_diagnostics=True)
                self.assertGreater(float(info["distillation/kl"]), 0)
                torch.testing.assert_close(info["pi_loss"].double(),
                    info["distillation/sac_actor_loss"].double() + info["distillation/weighted_kl"].double())
                distill.step = 50
                baseline = GNNSAC(cfg)
                baseline.load_training_state_dict(deepcopy(agent.training_state_dict()))
                rng = torch.random.get_rng_state()
                with patch.object(distill, "loss", side_effect=AssertionError("Teacher called at zero weight")):
                    agent.update(buffer)
                torch.random.set_rng_state(rng)
                baseline.update(buffer)
                for key, value in baseline.model.state_dict().items():
                    torch.testing.assert_close(value, agent.model.state_dict()[key], rtol=0, atol=0)

    def test_stage_orchestration_and_decay_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            agent = GNNSAC(cfg)
            trainer = Trainer(cfg=cfg, env=None, agent=agent, buffer=GNNBuffer(cfg), logger=DummyLogger())
            trainer.distillation = distill
            trainer._step = 0
            trainer.logger.log = Mock()
            trainer._evaluate_and_log = Mock()
            trainer.save_checkpoint = Mock()
            OnlineTrainer._run_distillation_pretraining(trainer)
            trainer._evaluate_and_log.assert_called_once()
            self.assertEqual(trainer.buffer.size, 0)
            self.assertEqual(trainer._step, 0)
            OnlineTrainer._run_distillation_pretraining(trainer)
            trainer._evaluate_and_log.assert_called_once()
            trainer._step = 25
            snapshot = trainer._checkpoint_state_snapshot()
            trainer.load_checkpoint_state_dict(snapshot)
            self.assertEqual(distill.weight, .5)
            self.assertEqual(distill.stage, "online")
            self.assertFalse(distill.datasets)

    def test_two_topology_cpu_environment_smoke(self):
        from common.logger import Logger
        from env import make_env
        from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg

        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(work_dir=tmp, steps=8, seed_steps=1, pretrain_steps=1,
                            batch_size=2, buffer_size=16, eval_freq=100, eval_episodes=1,
                            max_steps=2, nsubsteps=1, normalize_rewards=False,
                            save_video=False, enable_wandb=False, save_agent=False,
                            save_csv=False, checkpoint_freq=0, replay_ratio=1, episodic=True,
                            mpl_dims=[8], message_hidden_dims=[8], head_hidden_dims=[8],
                            log_std_min=-2., log_std_max=0., pcgrad=True)
            mapping = {}
            for topology in ("tetrahedron", "octahedron"):
                cfg = graph_test_cfg(**settings, truss_topology=topology)
                env = make_env(cfg)
                try:
                    agent, buffer = GNNSAC(cfg), GNNBuffer(cfg)
                    obs = env.reset()
                    episode = [dict(obs=obs, action=torch.zeros(obs.num_nodes, 1),
                                    reward=torch.tensor(0.), terminated=torch.tensor(False))]
                    for _ in range(2):
                        action = agent.act(obs)
                        obs, reward, _, info = env.step(action)
                        episode.append(dict(obs=obs, action=action, reward=reward,
                                            terminated=info["terminated"]))
                    buffer.add(episode)
                    trainer = Trainer(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=DummyLogger())
                    path = Path(tmp) / f"{topology}.pt"
                    torch.save(trainer.checkpoint_state_dict(), path)
                    mapping[topology] = str(path)
                finally:
                    env.close()
            cfg = graph_test_cfg(**settings, truss_topologies=["tetrahedron", "octahedron"],
                                 graph_features=dict(node_roles=True, edge_roles=False,
                                                     edge_distance=False),
                                 distillation=dict(enabled=True, teachers=mapping, pretrain_updates=2,
                                                   batch_size=2, shard_size=2, checkpoint_freq=0, log_freq=1))
            env = make_env(cfg)
            try:
                agent = GNNSAC(cfg)
                trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent,
                                        buffer=GNNBuffer(cfg), logger=Logger(cfg))
                trainer.train()
                self.assertEqual(trainer.distillation.completed_updates, 2)
                self.assertEqual(trainer.distillation.stage, "online")
                self.assertEqual(trainer._step, cfg.steps)
                self.assertGreater(trainer._optimizer_updates, 0)
                self.assertEqual(trainer.distillation.weight, 0)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
