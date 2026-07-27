from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer


def graph(marker):
    x = torch.full((3, 3), float(marker))
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


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
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def transition(marker):
    action = torch.full((1, 3, 1), float(marker))
    return [
        {
            "obs": graph(marker),
            "action": action,
            "reward": torch.tensor([float(marker)]),
            "terminated": torch.tensor([0.0]),
        },
        {
            "obs": graph(marker + 0.5),
            "action": action,
            "reward": torch.tensor([float(marker)]),
            "terminated": torch.tensor([0.0]),
        },
    ]


def agent_cfg(**overrides):
    values = vars(cfg()).copy()
    values.update(
        obs_dim=3,
        embedding_dim=8,
        mlp_dim=8,
        dropout=0.0,
        action_dim=1,
        Q_output_dim=8,
        head_hidden_dims=[8],
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy="auto",
        num_policy_actions=3,
        episode_length=100,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        tau=0.005,
        grad_clip_norm=10.0,
        pcgrad=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TaskBalancedReplayTest(unittest.TestCase):
    def test_routes_and_samples_equally(self):
        buffer = GNNBuffer(cfg())
        for marker in (1, 2, 3):
            buffer.add(transition(marker), task="truss-graph:a")
        self.assertFalse(buffer.ready)
        for marker in (11, 12):
            buffer.add(transition(marker), task="truss-graph:b")

        self.assertTrue(buffer.ready)
        self.assertEqual(buffer.sizes_by_task, {"truss-graph:a": 3, "truss-graph:b": 2})
        replay_batch = buffer.sample_with_tasks()
        self.assertEqual(list(replay_batch.by_task), ["truss-graph:a", "truss-graph:b"])
        self.assertEqual([batch[2].shape[0] for batch in replay_batch.by_task.values()], [2, 2])
        self.assertEqual(replay_batch.combined[2].shape[0], 4)

    def test_reports_multitask_replay_subphases(self):
        class PhaseRecorder:
            def __init__(self):
                self.names = []

            @contextmanager
            def subphase(self, name):
                self.names.append(name)
                yield

        buffer = GNNBuffer(cfg())
        for task, markers in (
            ("truss-graph:a", (1, 2)),
            ("truss-graph:b", (11, 12)),
        ):
            for marker in markers:
                buffer.add(transition(marker), task=task)
        profiler = PhaseRecorder()

        buffer.sample_with_tasks(performance_profiler=profiler)

        self.assertEqual(
            profiler.names,
            [
                "balanced_gather",
                "graph_preparation",
                "task_collation_transfer",
                "balanced_gather",
                "graph_preparation",
                "task_collation_transfer",
                "combined_reconstruction",
            ],
        )

    def test_multi_task_insert_requires_task(self):
        with self.assertRaisesRegex(ValueError, "explicit task"):
            GNNBuffer(cfg()).add(transition(1))

    def test_single_task_is_compatible(self):
        buffer = GNNBuffer(cfg(multitask=False, tasks=["one"], task="one", batch_size=2))
        buffer.add(transition(1))
        buffer.add(transition(2))
        self.assertTrue(buffer.ready)
        self.assertEqual(buffer.sample()[2].shape[0], 2)

    def test_divisibility_is_validated(self):
        with self.assertRaisesRegex(ValueError, "batch_size=3"):
            GNNBuffer(cfg(batch_size=3))
        with self.assertRaisesRegex(ValueError, "buffer_size=7"):
            GNNBuffer(cfg(buffer_size=7))

    def test_checkpoint_round_trip_and_layout_validation(self):
        original = GNNBuffer(cfg())
        original.add(transition(1), task="truss-graph:a")
        original.add(transition(2), task="truss-graph:b")
        state = original.state_dict()

        restored = GNNBuffer(cfg())
        restored.load_state_dict(state)
        self.assertEqual(restored.sizes_by_task, original.sizes_by_task)

        incompatible = GNNBuffer(cfg(tasks=["truss-graph:b", "truss-graph:a"]))
        with self.assertRaisesRegex(ValueError, "do not match"):
            incompatible.load_state_dict(state)


class GradientMetricTest(unittest.TestCase):
    def test_pair_metrics_cover_alignment_conflict_mismatch_and_zero(self):
        aligned = GNNSAC.gradient_pair_metrics((torch.tensor([1.0, 0.0]),), (torch.tensor([2.0, 0.0]),))
        self.assertAlmostEqual(float(aligned["cosine"]), 1.0)
        self.assertAlmostEqual(float(aligned["norm_agreement"]), 0.8)

        opposed = GNNSAC.gradient_pair_metrics((torch.tensor([1.0]),), (torch.tensor([-1.0]),))
        self.assertAlmostEqual(float(opposed["cosine"]), -1.0)
        self.assertAlmostEqual(float(opposed["norm_agreement"]), 1.0)

        zero = GNNSAC.gradient_pair_metrics((torch.tensor([0.0]),), (torch.tensor([0.0]),))
        self.assertEqual(float(zero["cosine"]), 0.0)
        self.assertEqual(float(zero["norm_agreement"]), 0.0)

    def test_diagnostics_report_actor_and_critic_without_changing_update(self):
        agent_config = agent_cfg()
        plain_agent = GNNSAC(agent_config)
        diagnostic_agent = GNNSAC(agent_config)
        diagnostic_agent.load_training_state_dict(plain_agent.training_state_dict())
        plain_buffer = GNNBuffer(agent_config)
        diagnostic_buffer = GNNBuffer(agent_config)
        for task, markers in (("truss-graph:a", (1, 2)), ("truss-graph:b", (11, 12))):
            for marker in markers:
                plain_buffer.add(transition(marker), task=task)
                diagnostic_buffer.add(transition(marker), task=task)

        torch.manual_seed(123)
        plain_agent.update(plain_buffer, compute_diagnostics=False)
        torch.manual_seed(123)
        info = diagnostic_agent.update(diagnostic_buffer, compute_diagnostics=True)

        metrics = info["gradient_diagnostics"]
        self.assertIn("actor/cosine/truss-graph_a__truss-graph_b", metrics)
        self.assertIn("critic/norm_agreement/truss-graph_a__truss-graph_b", metrics)
        self.assertIn("actor/norm/truss-graph_a", metrics)
        for plain_parameter, diagnostic_parameter in zip(
            plain_agent.parameters(), diagnostic_agent.parameters()
        ):
            torch.testing.assert_close(plain_parameter, diagnostic_parameter)


class PCGradTest(unittest.TestCase):
    def test_projects_conflicting_gradients(self):
        merged = GNNSAC.pcgrad_project(
            (
                (torch.tensor([1.0, 0.0]),),
                (torch.tensor([-1.0, 1.0]),),
            )
        )
        torch.testing.assert_close(merged[0], torch.tensor([0.25, 0.75]))

    def test_preserves_aligned_orthogonal_and_zero_gradients(self):
        aligned = GNNSAC.pcgrad_project(
            ((torch.tensor([1.0, 0.0]),), (torch.tensor([2.0, 0.0]),))
        )
        orthogonal = GNNSAC.pcgrad_project(
            ((torch.tensor([1.0, 0.0]),), (torch.tensor([0.0, 1.0]),))
        )
        with_zero = GNNSAC.pcgrad_project(
            ((torch.tensor([0.0, 0.0]),), (torch.tensor([1.0, 0.0]),))
        )
        single = GNNSAC.pcgrad_project(((torch.tensor([3.0, -2.0]),),))

        torch.testing.assert_close(aligned[0], torch.tensor([1.5, 0.0]))
        torch.testing.assert_close(orthogonal[0], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(with_zero[0], torch.tensor([0.5, 0.0]))
        torch.testing.assert_close(single[0], torch.tensor([3.0, -2.0]))

    def test_random_projection_order_is_seeded(self):
        gradients = (
            (torch.tensor([1.0, 0.0]),),
            (torch.tensor([-1.0, 1.0]),),
            (torch.tensor([-1.0, -1.0]),),
        )
        torch.manual_seed(91)
        first = GNNSAC.pcgrad_project(gradients)
        torch.manual_seed(91)
        second = GNNSAC.pcgrad_project(gradients)
        torch.testing.assert_close(first[0], second[0])

    def test_disabled_pcgrad_matches_config_without_the_flag(self):
        disabled_config = agent_cfg(pcgrad=False)
        legacy_config = agent_cfg()
        delattr(legacy_config, "pcgrad")
        disabled_agent = GNNSAC(disabled_config)
        legacy_agent = GNNSAC(legacy_config)
        legacy_agent.load_training_state_dict(disabled_agent.training_state_dict())
        disabled_buffer = GNNBuffer(disabled_config)
        legacy_buffer = GNNBuffer(legacy_config)
        for task, markers in (("truss-graph:a", (1, 2)), ("truss-graph:b", (11, 12))):
            for marker in markers:
                disabled_buffer.add(transition(marker), task=task)
                legacy_buffer.add(transition(marker), task=task)

        torch.manual_seed(123)
        disabled_agent.update(disabled_buffer)
        torch.manual_seed(123)
        legacy_agent.update(legacy_buffer)
        for disabled_parameter, legacy_parameter in zip(
            disabled_agent.parameters(), legacy_agent.parameters()
        ):
            torch.testing.assert_close(disabled_parameter, legacy_parameter)

    def test_pcgrad_updates_actor_and_critic_and_keeps_diagnostics(self):
        config = agent_cfg(pcgrad=True)
        agent = GNNSAC(config)
        buffer = GNNBuffer(config)
        for task, markers in (("truss-graph:a", (1, 2)), ("truss-graph:b", (11, 12))):
            for marker in markers:
                buffer.add(transition(marker), task=task)

        q_before = [
            parameter.detach().clone() for parameter in agent.model._Qs.parameters()
        ]
        pi_before = [
            parameter.detach().clone() for parameter in agent.model._pi.parameters()
        ]
        torch.manual_seed(123)
        info = agent.update(buffer, compute_diagnostics=True)

        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(q_before, agent.model._Qs.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(pi_before, agent.model._pi.parameters())
            )
        )
        self.assertIn(
            "actor/cosine/truss-graph_a__truss-graph_b",
            info["gradient_diagnostics"],
        )
        self.assertIn(
            "critic/cosine/truss-graph_a__truss-graph_b",
            info["gradient_diagnostics"],
        )

    def test_pcgrad_requires_task_aware_batches(self):
        class BufferWithoutTasks:
            def sample(self):
                raise AssertionError("sample should not run")

        with self.assertRaisesRegex(ValueError, "grouped by task"):
            GNNSAC(agent_cfg(pcgrad=True)).update(BufferWithoutTasks())


class DiagnosticCadenceTest(unittest.TestCase):
    def make_trainer(self, enabled):
        class Agent:
            def __init__(self):
                self.flags = []

            def update(self, buffer, compute_diagnostics=False):
                self.flags.append(compute_diagnostics)
                result = {"loss": torch.tensor(1.0)}
                if compute_diagnostics:
                    result["gradient_diagnostics"] = {"actor/norm/a": torch.tensor(2.0)}
                return result

        class Logger:
            def __init__(self):
                self.rows = []

            def log(self, metrics, category):
                self.rows.append((category, metrics))

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(
            gradient_diagnostics=enabled,
            gradient_diagnostics_freq=2,
        )
        trainer.agent = Agent()
        trainer.buffer = object()
        trainer.logger = Logger()
        trainer._optimizer_updates = 0
        trainer._step = 50
        return trainer

    def test_uses_optimizer_update_cadence(self):
        trainer = self.make_trainer(True)
        trainer._run_agent_updates(5)
        self.assertEqual(trainer.agent.flags, [False, True, False, True, False])
        self.assertEqual(
            [category for category, _ in trainer.logger.rows],
            ["gradient_diagnostics", "gradient_diagnostics"],
        )
        self.assertTrue(all(row["step"] == 50 for _, row in trainer.logger.rows))

    def test_disabled_never_requests_or_logs_diagnostics(self):
        trainer = self.make_trainer(False)
        trainer._run_agent_updates(3)
        self.assertEqual(trainer.agent.flags, [False, False, False])
        self.assertEqual(trainer.logger.rows, [])


if __name__ == "__main__":
    unittest.main()
