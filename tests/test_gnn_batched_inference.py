import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock
import sys
import unittest

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC
from gnn_infer import _run_vectorized_inference
from trainer.online_trainer import OnlineTrainer


def agent_cfg():
    return SimpleNamespace(
        device="cpu",
        obs_dim=3,
        embedding_dim=16,
        mlp_dim=16,
        dropout=0.0,
        action_dim=1,
        Q_output_dim=16,
        head_hidden_dims=[16],
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        lr=3e-4,
        entropy_coef=0.2,
        target_entropy="auto",
        num_policy_actions=6,
        episode_length=100,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
        tau=0.005,
        grad_clip_norm=10.0,
    )


def graph(num_nodes: int, offset: float = 0.0) -> Data:
    source = torch.arange(num_nodes, dtype=torch.long)
    target = source.roll(-1)
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])],
        dim=0,
    )
    x = torch.arange(num_nodes * 3, dtype=torch.float32).view(num_nodes, 3)
    return Data(x=x / max(num_nodes, 1) + offset, edge_index=edge_index)


class GNNBatchedInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = GNNSAC(agent_cfg())

    def test_empty_batch_does_not_invoke_actor(self):
        with mock.patch.object(self.agent.model, "pi") as pi, mock.patch.object(
            self.agent.model, "pi_mean"
        ) as pi_mean:
            self.assertEqual(self.agent.act_batch([]), [])
        pi.assert_not_called()
        pi_mean.assert_not_called()

    def test_mixed_size_deterministic_batch_matches_single_actions(self):
        observations = [graph(3), graph(5, offset=0.5), graph(4, offset=-0.25)]

        with mock.patch.object(
            self.agent.model,
            "pi_mean",
            wraps=self.agent.model.pi_mean,
        ) as pi_mean:
            batched = self.agent.act_batch(observations, eval_mode=True)
        pi_mean.assert_called_once()
        serialized = [self.agent.act(obs, eval_mode=True) for obs in observations]

        self.assertEqual(len(batched), len(observations))
        for action, expected, obs in zip(batched, serialized, observations):
            self.assertEqual(action.shape, (obs.num_nodes, 1))
            self.assertEqual(action.device.type, "cpu")
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue((action >= -1.0).all())
            self.assertTrue((action <= 1.0).all())
            torch.testing.assert_close(action, expected)

    def test_deterministic_batch_skips_stochastic_policy_path(self):
        observations = [graph(3), graph(4)]
        with mock.patch.object(
            self.agent.model,
            "pi",
            side_effect=AssertionError("stochastic policy path should not run"),
        ), mock.patch("torch.randn_like", side_effect=AssertionError("noise should not be sampled")):
            actions = self.agent.act_batch(observations, eval_mode=True)

        self.assertEqual([action.shape for action in actions], [(3, 1), (4, 1)])

    def test_stochastic_batch_returns_independent_valid_actions(self):
        observation = graph(5)
        first, second = self.agent.act_batch([observation, observation])

        self.assertEqual(first.shape, (5, 1))
        self.assertEqual(second.shape, (5, 1))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        self.assertFalse(torch.equal(first, second))


class MultiEnvActionSelectionTest(unittest.TestCase):
    def make_trainer(self, agent, step=3, seed_steps=1):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.agent = agent
        trainer.env = mock.Mock()
        trainer.cfg = SimpleNamespace(seed_steps=seed_steps, domain_randomization=False)
        trainer._step = step
        return trainer

    def test_partial_vector_step_uses_one_batched_call(self):
        class BatchedAgent:
            def __init__(self):
                self.calls = []

            def act_batch(self, observations):
                self.calls.append(observations)
                return [torch.full((obs.num_nodes, 1), float(i)) for i, obs in enumerate(observations)]

        agent = BatchedAgent()
        trainer = self.make_trainer(agent)
        observations = [graph(3), graph(4), graph(5)]
        episode_tds = [[object()], [object(), object()], [object()]]

        actions = trainer._select_multi_env_actions(observations, episode_tds, [0, 2])

        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(agent.calls[0], [observations[0], observations[2]])
        self.assertEqual([action.shape for action in actions], [(3, 1), (5, 1)])

    def test_agent_without_batch_api_uses_serialized_fallback(self):
        class SerializedAgent:
            def __init__(self):
                self.calls = []

            def act(self, observation, t0=False):
                self.calls.append((observation, t0))
                return torch.zeros(observation.num_nodes, 1)

        agent = SerializedAgent()
        trainer = self.make_trainer(agent)
        observations = [graph(3), graph(4)]
        episode_tds = [[object()], [object(), object()]]

        actions = trainer._select_multi_env_actions(observations, episode_tds, [0, 1])

        self.assertEqual(agent.calls, [(observations[0], True), (observations[1], False)])
        self.assertEqual([action.shape for action in actions], [(3, 1), (4, 1)])

    def test_seed_collection_keeps_per_environment_random_actions(self):
        agent = mock.Mock()
        trainer = self.make_trainer(agent, step=1, seed_steps=1)
        trainer.env.rand_act.side_effect = lambda env_idx: torch.tensor([float(env_idx)])

        actions = trainer._select_multi_env_actions(
            [graph(3), graph(4), graph(5)],
            [[object()], [object()], [object()]],
            [2, 0],
        )

        agent.act_batch.assert_not_called()
        self.assertEqual(trainer.env.rand_act.call_args_list, [mock.call(env_idx=2), mock.call(env_idx=0)])
        torch.testing.assert_close(actions[0], torch.tensor([2.0]))
        torch.testing.assert_close(actions[1], torch.tensor([0.0]))


class UpdateScheduleTest(unittest.TestCase):
    def make_trainer(self, *, replay_ratio=1.0, batch_size=256, iterations=None):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            iterations=iterations,
        )
        trainer._update_budget = 0.0
        trainer._pretrain_complete = True
        return trainer

    def test_vector_batch_uses_replay_sample_ratio(self):
        trainer = self.make_trainer()

        self.assertEqual(trainer._scheduled_updates(2048), 8)
        self.assertEqual(trainer._update_budget, 0.0)

    def test_fractional_update_budget_carries_between_steps(self):
        trainer = self.make_trainer()

        updates = [trainer._scheduled_updates(64) for _ in range(4)]

        self.assertEqual(updates, [0, 0, 0, 1])
        self.assertEqual(trainer._update_budget, 0.0)

    def test_pretraining_runs_once_when_replay_first_becomes_ready(self):
        trainer = self.make_trainer()
        trainer._pretrain_complete = False

        self.assertEqual(trainer._updates_after_collection(2048, pretrain_steps=1000), 1000)
        self.assertTrue(trainer._pretrain_complete)
        self.assertEqual(trainer._updates_after_collection(2048, pretrain_steps=1000), 8)

    def test_legacy_iterations_preserve_old_schedule(self):
        trainer = self.make_trainer(iterations=1)

        self.assertEqual(trainer._scheduled_updates(2048), 2048)

    def test_negative_replay_ratio_is_rejected(self):
        trainer = self.make_trainer(replay_ratio=-1.0)

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            trainer._scheduled_updates(1)

    def test_update_cadence_batches_eight_vector_steps(self):
        trainer = self.make_trainer()
        trainer.cfg.update_every_vector_steps = 8
        trainer.cfg.seed_steps = 0
        trainer.buffer = SimpleNamespace(size=2048)
        trainer._step = 0
        trainer._pending_update_transitions = 0
        trainer._vector_steps_since_update = 0

        for _ in range(7):
            trainer._queue_collected_transitions(256)
            trainer._step += 256
            self.assertEqual(trainer._updates_due(pretrain_steps=1000), 0)

        trainer._queue_collected_transitions(256)
        trainer._step += 256

        self.assertEqual(trainer._updates_due(pretrain_steps=1000), 8)
        self.assertEqual(trainer._pending_update_transitions, 0)
        self.assertEqual(trainer._vector_steps_since_update, 0)

    def test_update_cadence_one_spends_budget_each_vector_step(self):
        trainer = self.make_trainer()
        trainer.cfg.update_every_vector_steps = 1
        trainer.cfg.seed_steps = 0
        trainer.buffer = SimpleNamespace(size=256)
        trainer._step = 256
        trainer._pending_update_transitions = 0
        trainer._vector_steps_since_update = 0

        trainer._queue_collected_transitions(256)

        self.assertEqual(trainer._updates_due(pretrain_steps=1000), 1)


class VectorTrainingAccountingTest(unittest.TestCase):
    def test_exact_step_budget_flushes_partial_trajectories_and_logs_final_state(self):
        profiling_dir = TemporaryDirectory()
        self.addCleanup(profiling_dir.cleanup)

        class DummyVectorEnv:
            num_envs = 2

            def __init__(self):
                self.lengths = [0, 0]

            def reset_many(self, env_indices):
                for env_idx in env_indices:
                    self.lengths[env_idx] = 0
                return [graph(3, offset=float(env_idx)) for env_idx in env_indices]

            def rand_act(self, env_idx=None):
                return torch.zeros(3, 1)

            def step_many(self, actions, env_indices):
                results = []
                for env_idx in env_indices:
                    self.lengths[env_idx] += 1
                    done = self.lengths[env_idx] == 2
                    results.append(
                        (
                            graph(3, offset=float(env_idx + self.lengths[env_idx])),
                            torch.tensor(1.0),
                            done,
                            {
                                "success": 0.0,
                                "terminated": torch.tensor(0.0),
                                "truncated": torch.tensor(float(done)),
                            },
                        )
                    )
                return results

            def close(self):
                return

        class RecordingBuffer:
            def __init__(self):
                self.size = 0
                self.num_eps = 0
                self.insert_sizes = []

            def add(self, trajectory, count_episode=True):
                insert_size = len(trajectory) - 1
                self.insert_sizes.append(insert_size)
                self.size += insert_size
                self.num_eps += int(bool(count_episode))
                return self.num_eps

        class RecordingLogger:
            def __init__(self):
                self.rows = []

            def log(self, metrics, category):
                self.rows.append((category, dict(metrics)))

            def finish(self, agent):
                return

        class RecordingAgent:
            def __init__(self):
                self.update_buffer_sizes = []

            def act_batch(self, observations):
                return [torch.zeros(obs.num_nodes, 1) for obs in observations]

            def update(self, buffer):
                self.update_buffer_sizes.append(buffer.size)
                return {"value_loss": torch.tensor(0.0)}

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(
            steps=5,
            eval_freq=1000,
            seed_steps=0,
            pretrain_steps=0,
            batch_size=2,
            replay_ratio=2.0,
            update_every_vector_steps=2,
            iterations=None,
            episodic=True,
            checkpoint_freq=0,
            progress_freq=1000,
            domain_randomization=False,
            work_dir=profiling_dir.name,
            device="cpu",
            profiling=SimpleNamespace(
                enabled=True,
                warmup_vector_steps=1,
                active_vector_steps=2,
                trace_enabled=False,
                output_dir="profiling",
            ),
        )
        trainer.env = DummyVectorEnv()
        trainer.eval_env = trainer.env
        trainer.agent = RecordingAgent()
        trainer.buffer = RecordingBuffer()
        trainer.logger = RecordingLogger()
        trainer._step = 0
        trainer._ep_idx = 0
        trainer._start_time = 0.0
        trainer._episode_reward_components = {}
        trainer._eval_topology_indices = None
        trainer._update_budget = 0.0
        trainer._pending_update_transitions = 0
        trainer._vector_steps_since_update = 0
        trainer._pretrain_complete = False
        trainer._optimizer_updates = 0
        trainer._last_eval_step = None
        trainer._best_eval_metrics = None
        trainer.trial = None
        trainer.eval = lambda: {
            "episode_reward": 0.0,
            "episode_success": 0.0,
            "episode_length": 0.0,
        }

        trainer._train_multi_env(num_envs=2)

        self.assertEqual(trainer._step, 5)
        self.assertEqual(trainer.buffer.size, 5)
        self.assertEqual(trainer.buffer.num_eps, 2)
        self.assertEqual(trainer.buffer.insert_sizes, [1, 1, 1, 1, 1])
        self.assertEqual(trainer.agent.update_buffer_sizes, [5])
        final_category, final_metrics = next(
            (category, metrics)
            for category, metrics in reversed(trainer.logger.rows)
            if category == "train"
        )
        self.assertEqual(final_category, "train")
        self.assertEqual(final_metrics["step"], 5)
        self.assertEqual(final_metrics["buffer_size"], 5)
        self.assertEqual(final_metrics["optimizer_updates"], 1)
        eval_steps = [
            metrics["step"]
            for category, metrics in trainer.logger.rows
            if category == "eval"
        ]
        self.assertEqual(eval_steps, [0, 5])
        profiling_summary = json.loads(
            (Path(profiling_dir.name) / "profiling" / "profiling_summary.json").read_text()
        )
        self.assertEqual(profiling_summary["window"]["measured_vector_steps"], 2)
        self.assertEqual(
            profiling_summary["window"]["measured_environment_transitions"],
            3,
        )
        self.assertIn("action_selection", profiling_summary["phases"])
        self.assertIn("environment_step", profiling_summary["phases"])
        self.assertIn("replay_insertion", profiling_summary["phases"])
        self.assertFalse(
            (Path(profiling_dir.name) / "profiling" / "training_trace.json").exists()
        )


class VectorizedInferenceTest(unittest.TestCase):
    def test_episodes_are_run_in_batched_waves(self):
        class DummyVectorEnv:
            num_envs = 2

            def __init__(self):
                self.step_batches = []

            def reset_many(self, env_indices):
                return [graph(3, offset=float(index)) for index in env_indices]

            def step_many(self, actions, env_indices):
                self.step_batches.append(list(env_indices))
                return [
                    (
                        graph(3, offset=float(index)),
                        torch.tensor(float(index + 1)),
                        True,
                        {
                            "success": float(index == 0),
                            "terminated": torch.tensor(0.0),
                            "truncated": torch.tensor(1.0),
                        },
                    )
                    for index in env_indices
                ]

        class DummyAgent:
            def __init__(self):
                self.batch_sizes = []

            def act_batch(self, observations, eval_mode=False):
                self.batch_sizes.append(len(observations))
                return [torch.zeros(obs.num_nodes, 1) for obs in observations]

        env = DummyVectorEnv()
        agent = DummyAgent()
        cfg = SimpleNamespace(
            episodes=3,
            inference_max_steps=None,
            deterministic=True,
            print_position_command=False,
        )

        results = _run_vectorized_inference(cfg, env, agent)

        self.assertEqual(agent.batch_sizes, [2, 1])
        self.assertEqual(env.step_batches, [[0, 1], [0]])
        self.assertEqual([row["episode"] for row in results], [0, 1, 2])
        self.assertEqual([row["episode_length"] for row in results], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
