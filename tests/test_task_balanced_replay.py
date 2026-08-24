from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import warnings

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer, ReplayBatch
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer


def graph(marker, node_count=3):
    x = torch.full((node_count, 3), float(marker))
    nodes = torch.arange(node_count, dtype=torch.long)
    edge_index = torch.stack(
        [
            torch.cat([nodes, nodes.roll(-1)]),
            torch.cat([nodes.roll(-1), nodes]),
        ],
        dim=0,
    )
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


def transition(marker, node_count=3):
    action = torch.full((1, node_count, 1), float(marker))
    return [
        {
            "obs": graph(marker, node_count),
            "action": action,
            "reward": torch.tensor([float(marker)]),
            "terminated": torch.tensor([0.0]),
        },
        {
            "obs": graph(marker + 0.5, node_count),
            "action": action,
            "reward": torch.tensor([float(marker)]),
            "terminated": torch.tensor([0.0]),
        },
    ]


def assert_batches_equal(test_case, first, second):
    for first_value, second_value in zip(first, second):
        if hasattr(first_value, "to_dict"):
            first_values = first_value.to_dict()
            second_values = second_value.to_dict()
            test_case.assertEqual(first_values.keys(), second_values.keys())
            for key in first_values:
                torch.testing.assert_close(
                    first_values[key],
                    second_values[key],
                    rtol=0.0,
                    atol=0.0,
                )
        else:
            torch.testing.assert_close(
                first_value,
                second_value,
                rtol=0.0,
                atol=0.0,
            )


def assert_nested_equal(test_case, first, second):
    test_case.assertEqual(type(first), type(second))
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    elif isinstance(first, dict):
        test_case.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test_case, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test_case.assertEqual(len(first), len(second))
        for first_value, second_value in zip(first, second):
            assert_nested_equal(test_case, first_value, second_value)
    else:
        test_case.assertEqual(first, second)


def populate_buffer(config):
    buffer = GNNBuffer(config)
    for task, markers in (
        (config.tasks[0], (1, 2, 3, 4)),
        (config.tasks[1], (11, 12, 13, 14)),
    ):
        for marker in markers:
            buffer.add(transition(marker), task=task)
    return buffer


class LegacyReplayAdapter:
    """Exercise the pre-optimization sampling composition in equivalence tests."""

    def __init__(self, buffer):
        self.buffer = buffer

    def sample_with_tasks(self):
        by_task = self.buffer.sample_task_batches()
        return ReplayBatch(
            combined=GNNBuffer.combine_task_batches(by_task),
            by_task=by_task,
        )


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
                "balanced_gather",
                "graph_preparation",
                "combined_collation_transfer",
                "task_collation_transfer",
                "task_collation_transfer",
            ],
        )

    def test_direct_combined_sample_matches_legacy_rebatching_exactly(self):
        config = cfg(buffer_size=16)
        optimized = populate_buffer(config)
        legacy = populate_buffer(config)

        torch.manual_seed(91)
        optimized_batch = optimized.sample()
        torch.manual_seed(91)
        task_batches = legacy.sample_task_batches()
        legacy_batch = GNNBuffer.combine_task_batches(task_batches)

        assert_batches_equal(self, optimized_batch, legacy_batch)

    def test_direct_sample_preserves_virtual_node_batching(self):
        config = cfg(buffer_size=16, use_virtual_node=True)
        optimized = populate_buffer(config)
        legacy = populate_buffer(config)

        torch.manual_seed(17)
        optimized_batch = optimized.sample()
        torch.manual_seed(17)
        legacy_batch = GNNBuffer.combine_task_batches(
            legacy.sample_task_batches()
        )

        assert_batches_equal(self, optimized_batch, legacy_batch)
        self.assertEqual(int(optimized_batch[0].physical_node_mask.sum()), 12)
        self.assertEqual(optimized_batch[0].num_nodes, 16)

    def test_direct_sample_matches_legacy_for_mixed_graph_sizes(self):
        config = cfg(buffer_size=16)
        optimized = GNNBuffer(config)
        legacy = GNNBuffer(config)
        for buffer in (optimized, legacy):
            for marker in (1, 2, 3, 4):
                buffer.add(
                    transition(marker, node_count=3),
                    task="truss-graph:a",
                )
            for marker in (11, 12, 13, 14):
                buffer.add(
                    transition(marker, node_count=5),
                    task="truss-graph:b",
                )

        torch.manual_seed(31)
        optimized_batch = optimized.sample_with_tasks()
        torch.manual_seed(31)
        legacy_by_task = legacy.sample_task_batches()
        legacy_batch = ReplayBatch(
            combined=GNNBuffer.combine_task_batches(legacy_by_task),
            by_task=legacy_by_task,
        )

        assert_batches_equal(
            self,
            optimized_batch.combined,
            legacy_batch.combined,
        )
        for task in config.tasks:
            assert_batches_equal(
                self,
                optimized_batch.by_task[task],
                legacy_batch.by_task[task],
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

    def test_divisibility_values_are_rounded_and_persisted(self):
        config = cfg(buffer_size=7, batch_size=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            buffer = GNNBuffer(config)

        messages = [str(warning.message) for warning in caught]
        self.assertTrue(any("buffer_size=7" in message and "using 8" in message for message in messages))
        self.assertTrue(any("batch_size=3" in message and "using 4" in message for message in messages))
        self.assertEqual(config.buffer_size, 8)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(buffer._capacity_per_task, 4)
        self.assertEqual(buffer._batch_size_per_task, 2)

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

    def test_repeated_normal_updates_match_legacy_rebatching_exactly(self):
        config = agent_cfg(buffer_size=16)
        optimized_agent = GNNSAC(config)
        legacy_agent = GNNSAC(config)
        legacy_agent.load_training_state_dict(
            optimized_agent.training_state_dict()
        )
        optimized_buffer = populate_buffer(config)
        legacy_buffer = LegacyReplayAdapter(populate_buffer(config))

        torch.manual_seed(123)
        for _ in range(3):
            rng_state = torch.random.get_rng_state()
            legacy_metrics = legacy_agent.update(legacy_buffer)
            next_rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(rng_state)
            optimized_metrics = optimized_agent.update(optimized_buffer)
            for key in legacy_metrics:
                torch.testing.assert_close(
                    legacy_metrics[key],
                    optimized_metrics[key],
                    rtol=0.0,
                    atol=0.0,
                )
            assert_nested_equal(
                self,
                legacy_agent.training_state_dict(),
                optimized_agent.training_state_dict(),
            )
            torch.random.set_rng_state(next_rng_state)


class ReplayRoutingTest(unittest.TestCase):
    class RecordingBuffer:
        supports_replay_profiling = True

        def __init__(self):
            self.calls = []
            self.batch = (None, None, None, None, None)

        def sample(self, performance_profiler=None):
            self.calls.append("combined")
            return self.batch

        def sample_with_tasks(self, performance_profiler=None):
            self.calls.append("combined_and_tasks")
            return ReplayBatch(
                combined=self.batch,
                by_task={"task": self.batch},
            )

        def sample_task_batches(self, performance_profiler=None):
            self.calls.append("tasks")
            return {"task": self.batch}

    class Model:
        def train(self):
            return

        def soft_update_target_Q(self):
            return

        def eval(self):
            return

    def test_normal_diagnostics_and_pcgrad_request_only_needed_batches(self):
        buffer = self.RecordingBuffer()
        agent = SimpleNamespace(
            cfg=SimpleNamespace(pcgrad=False),
            model=self.Model(),
            update_q=lambda *args: (torch.tensor(1.0), torch.tensor(2.0)),
            update_pi_and_alpha=lambda obs: {"pi_loss": torch.tensor(3.0)},
            _gradient_diagnostics=lambda batches: {"metric": torch.tensor(4.0)},
        )

        GNNSAC.update(agent, buffer, compute_diagnostics=False)
        GNNSAC.update(agent, buffer, compute_diagnostics=True)

        pcgrad_agent = SimpleNamespace(
            cfg=SimpleNamespace(pcgrad=True),
            model=self.Model(),
            _pcgrad_q_update=lambda batches: (
                torch.tensor(1.0),
                torch.tensor(2.0),
                {"task": (torch.tensor(1.0),)},
            ),
            _pcgrad_pi_and_alpha_update=lambda batches: (
                {"pi_loss": torch.tensor(3.0)},
                {"task": (torch.tensor(1.0),)},
            ),
            _gradient_metrics=lambda gradients: {"metric": torch.tensor(4.0)},
        )
        GNNSAC.update(pcgrad_agent, buffer, compute_diagnostics=False)

        self.assertEqual(
            buffer.calls,
            ["combined", "combined_and_tasks", "tasks"],
        )


class PCGradTest(unittest.TestCase):
    def test_actor_optimizer_covers_gnn_and_action_projection(self):
        agent = GNNSAC(agent_cfg())
        optimized_parameter_ids = {
            id(parameter)
            for group in agent.pi_optim.param_groups
            for parameter in group["params"]
        }
        actor_parameter_ids = {
            id(parameter) for parameter in agent.model.actor_parameters()
        }

        self.assertEqual(optimized_parameter_ids, actor_parameter_ids)
        self.assertTrue(
            {id(parameter) for parameter in agent.model._action_head.parameters()}
            <= optimized_parameter_ids
        )

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
        actor_parameters = tuple(agent.model.actor_parameters())
        pi_before = [parameter.detach().clone() for parameter in actor_parameters]
        action_head_before = [
            parameter.detach().clone()
            for parameter in agent.model._action_head.parameters()
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
                for before, after in zip(pi_before, actor_parameters)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    action_head_before, agent.model._action_head.parameters()
                )
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

    def test_pcgrad_update_matches_legacy_rebatching_exactly(self):
        config = agent_cfg(pcgrad=True, buffer_size=16)
        optimized_agent = GNNSAC(config)
        legacy_agent = GNNSAC(config)
        legacy_agent.load_training_state_dict(
            optimized_agent.training_state_dict()
        )
        optimized_buffer = populate_buffer(config)
        legacy_buffer = LegacyReplayAdapter(populate_buffer(config))

        torch.manual_seed(43)
        rng_state = torch.random.get_rng_state()
        legacy_metrics = legacy_agent.update(legacy_buffer)
        torch.random.set_rng_state(rng_state)
        optimized_metrics = optimized_agent.update(optimized_buffer)

        for key in legacy_metrics:
            torch.testing.assert_close(
                legacy_metrics[key],
                optimized_metrics[key],
                rtol=0.0,
                atol=0.0,
            )
        assert_nested_equal(
            self,
            legacy_agent.training_state_dict(),
            optimized_agent.training_state_dict(),
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
