from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from common.logger import Logger
from gnn_sac import GNNSAC
from trainer.base import Trainer
from trainer.online_trainer import OnlineTrainer


TASKS = [
    "truss-graph:octahedron",
    "truss-graph:tetrahedron",
    "truss-graph:henneberg_n6_1tube_2",
]


def config(**overrides):
    values = dict(
        device="cpu",
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
        buffer_size=24,
        batch_size=6,
        steps=100,
        multitask=True,
        tasks=list(TASKS),
        task="truss-graph",
        mujoco_backend="mujoco",
        truss_topologies=None,
        use_virtual_node=False,
        safety_constraint={
            "enabled": True,
            "horizon": 250,
            "default_budget": 0.10,
            "budgets_by_topology": {},
            "cost_critic_lr": 3e-4,
            "lambda_lr": 0.1,
            "lambda_init": 0.1,
            "lambda_max": 100.0,
            "lambda_batch_size": 1,
        },
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def curriculum_config(**overrides):
    safety = dict(config().safety_constraint)
    safety["horizon"] = 1000
    curriculum = {
        "enabled": True,
        "initial_horizon": 50,
        "promotion_factor": 1.5,
        "consecutive_success_windows": 3,
        "boundary_sample_probability": 0.5,
        "upper_half_sample_probability": 0.25,
    }
    curriculum.update(overrides.pop("curriculum", {}))
    safety["curriculum"] = curriculum
    safety.update(overrides)
    return config(safety_constraint=safety)


def graph(marker, node_count=3):
    nodes = torch.arange(node_count, dtype=torch.long)
    return Data(
        x=torch.full((node_count, 3), float(marker)),
        edge_index=torch.stack((nodes, torch.roll(nodes, shifts=-1))),
        action_mask=torch.ones(node_count, dtype=torch.bool),
    )


def transition(marker, *, collapse=0.0, episode_end=0.0):
    return [
        {
            "obs": graph(marker),
            "action": torch.full((1, 3, 1), float("nan")),
            "reward": torch.tensor([float("nan")]),
            "collapse_cost": torch.tensor([float("nan")]),
            "terminated": torch.tensor([float("nan")]),
            "episode_end": torch.tensor([float("nan")]),
        },
        {
            "obs": graph(marker + 0.5),
            "action": torch.zeros(1, 3, 1),
            "reward": torch.tensor([1.0]),
            "collapse_cost": torch.tensor([collapse]),
            "terminated": torch.tensor([collapse]),
            "episode_end": torch.tensor([episode_end]),
        },
    ]


def populated_buffer(cfg):
    replay = GNNBuffer(cfg)
    for task_index, task in enumerate(cfg.tasks):
        for sample_index in range(3):
            replay.add(
                transition(
                    10 * task_index + sample_index,
                    collapse=float(sample_index == 0),
                    episode_end=float(sample_index == 0),
                ),
                task=task,
            )
    return replay


class FiniteHorizonCostTest(unittest.TestCase):
    def test_cost_critics_are_twin_bounded_and_horizon_conditioned(self):
        agent = GNNSAC(config())
        obs = Batch.from_data_list([graph(1), graph(2)])
        action = torch.zeros(6, 1)

        values = agent.model.cost_Q(
            obs, action, torch.tensor([1, 250]), return_type="all"
        )

        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(torch.all(values >= 0.0))
        self.assertTrue(torch.all(values <= 1.0))

    def test_finite_horizon_targets_handle_collapse_h1_end_and_continuation(self):
        agent = GNNSAC(config())
        collapse = torch.tensor([1.0, 0.0, 0.0, 0.0])
        episode_end = torch.tensor([1.0, 0.0, 1.0, 0.0])
        horizon = torch.tensor([250, 1, 250, 2])
        with patch.object(agent.model, "pi", return_value=(torch.zeros(1), {})), patch.object(
            agent.model,
            "cost_Q",
            return_value=torch.full((4,), 0.6),
        ):
            target = agent._cost_td_target(
                object(), collapse, episode_end, horizon
            )

        torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.0, 0.6]))

    def test_curriculum_sampling_is_bounded_and_boundary_weighted(self):
        agent = GNNSAC(curriculum_config())
        torch.manual_seed(7)
        horizons = agent._sample_cost_horizons(TASKS[0], 20000)

        self.assertEqual(int(horizons.min()), 1)
        self.assertEqual(int(horizons.max()), 50)
        boundary_fraction = float((horizons == 50).float().mean())
        self.assertGreater(boundary_fraction, 0.49)
        self.assertLess(boundary_fraction, 0.55)

    def test_fixed_horizon_sampling_remains_uniform_and_seed_compatible(self):
        agent = GNNSAC(config())
        torch.manual_seed(19)
        expected = torch.randint(1, 251, (64,))
        torch.manual_seed(19)
        actual = agent._sample_cost_horizons(None, 64)
        torch.testing.assert_close(actual, expected)

    def test_curriculum_critic_normalizes_by_immutable_maximum(self):
        agent = GNNSAC(curriculum_config())
        obs = Batch.from_data_list([graph(1)])
        action = torch.zeros(3, 1)
        with patch.object(
            agent.model._CostQs,
            "forward",
            return_value=torch.zeros(2, 1),
        ) as cost_critics:
            agent.model.cost_Q(obs, action, torch.tensor([50]), return_type="all")

        critic_input = cost_critics.call_args.args[0]
        torch.testing.assert_close(
            critic_input[:, -1], torch.full((3,), 0.05)
        )

    def test_curriculum_configuration_is_validated(self):
        invalid = (
            {"initial_horizon": 0},
            {"initial_horizon": 1001},
            {"promotion_factor": 1.0},
            {"consecutive_success_windows": 0},
            {"boundary_sample_probability": -0.1},
            {
                "boundary_sample_probability": 0.8,
                "upper_half_sample_probability": 0.3,
            },
        )
        for curriculum in invalid:
            with self.subTest(curriculum=curriculum), self.assertRaises(ValueError):
                GNNSAC(curriculum_config(curriculum=curriculum))


class MultiplierTest(unittest.TestCase):
    def test_updates_only_matching_topology_and_moves_with_violation(self):
        agent = GNNSAC(config())
        before = agent.safety_lambdas().detach().clone()

        metrics = agent.observe_safety_outcome(TASKS[0], 1.0)
        after_failure = agent.safety_lambdas().detach().clone()
        self.assertGreater(float(after_failure[0]), float(before[0]))
        torch.testing.assert_close(after_failure[1:], before[1:])
        self.assertIn("safety/violation/truss-graph_octahedron", metrics)

        agent.observe_safety_outcome(TASKS[1], 0.0)
        after_safe = agent.safety_lambdas().detach().clone()
        self.assertLess(float(after_safe[1]), float(after_failure[1]))
        self.assertGreaterEqual(float(after_safe.min()), 0.0)

    def test_training_state_round_trip_restores_safety_state(self):
        first = GNNSAC(config())
        first.observe_safety_outcome(TASKS[0], 1.0)
        state = first.training_state_dict()
        second = GNNSAC(config())

        second.load_training_state_dict(state)

        torch.testing.assert_close(second.raw_lambdas, first.raw_lambdas)
        self.assertEqual(second._pending_safety_outcomes, first._pending_safety_outcomes)
        self.assertEqual(second._resolved_safety_counts, first._resolved_safety_counts)
        agent_only = Trainer._agent_checkpoint_state_dict({"agent": state})
        for key in ("model", "raw_lambdas", "safety_tasks", "safety_horizon"):
            self.assertIn(key, agent_only)

    def test_actor_loss_scales_with_multiplier_and_predicted_risk(self):
        agent = GNNSAC(config())
        info = {
            "log_prob": torch.zeros(1),
            "entropy": torch.zeros(1),
        }
        with patch.object(agent.model, "pi", return_value=(torch.zeros(3, 1), info)), patch.object(
            agent.model, "Q", return_value=torch.zeros(1)
        ), patch.object(
            agent.model, "cost_Q", return_value=torch.full((2, 1), 0.4)
        ):
            low_loss, _ = agent._pi_loss(graph(1), task=TASKS[0])
            with torch.no_grad():
                agent.raw_lambdas[0] = agent._inverse_softplus(1.0)
            high_loss, _ = agent._pi_loss(graph(1), task=TASKS[0])

        self.assertAlmostEqual(float(high_loss - low_loss), 0.36, places=5)

    def test_curriculum_promotes_monotonically_and_independently(self):
        cfg = curriculum_config(
            lambda_batch_size=1,
            curriculum={"consecutive_success_windows": 1},
        )
        agent = GNNSAC(cfg)
        expected = [75, 113, 170, 255, 383, 575, 863, 1000]
        for horizon in expected:
            old_horizon = agent.active_safety_horizon(TASKS[0])
            metrics = agent.observe_safety_outcome(TASKS[0], 0.0, horizon=old_horizon)
            self.assertEqual(agent.active_safety_horizon(TASKS[0]), horizon)
            self.assertEqual(agent.active_safety_horizon(TASKS[1]), 50)
            self.assertEqual(
                metrics["safety/curriculum_promoted/truss-graph_octahedron"],
                float(horizon > old_horizon),
            )
        agent.observe_safety_outcome(TASKS[0], 0.0, horizon=1000)
        self.assertEqual(agent.active_safety_horizon(TASKS[0]), 1000)

    def test_curriculum_streak_resets_and_stale_outcomes_are_discarded(self):
        cfg = curriculum_config(lambda_batch_size=1)
        agent = GNNSAC(cfg)
        for outcome in (0.0, 0.0, 1.0, 0.0, 0.0, 0.0):
            agent.observe_safety_outcome(TASKS[0], outcome, horizon=50)
        self.assertEqual(agent.active_safety_horizon(TASKS[0]), 75)
        before_lambda = agent.safety_lambda(TASKS[0]).detach().clone()
        metrics = agent.observe_safety_outcome(TASKS[0], 1.0, horizon=50)
        torch.testing.assert_close(agent.safety_lambda(TASKS[0]), before_lambda)
        self.assertEqual(agent._pending_safety_outcomes[TASKS[0]], [])
        self.assertEqual(
            metrics["safety/curriculum_stale_outcomes/truss-graph_octahedron"], 1
        )

    def test_curriculum_training_and_agent_state_round_trip(self):
        cfg = curriculum_config(
            lambda_batch_size=1,
            curriculum={"consecutive_success_windows": 1},
        )
        first = GNNSAC(cfg)
        first.observe_safety_outcome(TASKS[0], 0.0, horizon=50)
        first.observe_safety_outcome(TASKS[1], 1.0, horizon=50)
        state = first.training_state_dict()
        second = GNNSAC(cfg)
        second.load_training_state_dict(state)

        self.assertEqual(second.active_horizon_by_task, first.active_horizon_by_task)
        self.assertEqual(second._curriculum_pass_streaks, first._curriculum_pass_streaks)
        self.assertEqual(
            second._curriculum_promotion_counts, first._curriculum_promotion_counts
        )
        agent_only = Trainer._agent_checkpoint_state_dict({"agent": state})
        self.assertIn("active_horizon_by_task", agent_only)
        third = GNNSAC(cfg)
        third.load(agent_only)
        self.assertEqual(third.active_horizon_by_task, first.active_horizon_by_task)

    def test_checkpoint_compatibility_distinguishes_fixed_and_curriculum_runs(self):
        fixed = GNNSAC(config())
        old_fixed_state = fixed.training_state_dict()
        fixed.load_training_state_dict(old_fixed_state)

        curriculum = GNNSAC(curriculum_config())
        incompatible = dict(curriculum.training_state_dict())
        incompatible.pop("active_horizon_by_task")
        with self.assertRaisesRegex(ValueError, "missing safety curriculum state"):
            curriculum.load_training_state_dict(incompatible)

    def test_actor_and_risk_prediction_use_task_active_horizon(self):
        agent = GNNSAC(curriculum_config())
        agent.active_horizon_by_task[TASKS[0]] = 113
        info = {"log_prob": torch.zeros(1), "entropy": torch.zeros(1)}
        with patch.object(agent.model, "pi", return_value=(torch.zeros(3, 1), info)), patch.object(
            agent.model, "Q", return_value=torch.zeros(1)
        ), patch.object(
            agent.model, "cost_Q", return_value=torch.full((2, 1), 0.4)
        ) as cost_q:
            agent._pi_loss(graph(1), task=TASKS[0])
            torch.testing.assert_close(
                cost_q.call_args.args[2], torch.tensor([113], dtype=torch.long)
            )

        with patch.object(agent.model, "pi_mean", return_value=torch.zeros(3, 1)), patch.object(
            agent.model, "cost_Q", return_value=torch.tensor([0.25])
        ) as cost_q:
            self.assertEqual(agent.predict_safety_risk(graph(1), task=TASKS[0]), 0.25)
            self.assertEqual(int(cost_q.call_args.args[2].item()), 113)


class TrainerContractTest(unittest.TestCase):
    def test_constrained_learning_uses_locomotion_reward_only(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = config()
        self.assertEqual(
            trainer._learning_reward(
                torch.tensor(-99.0),
                {"locomotion_reward": torch.tensor(3.0), "collapse_cost": torch.tensor(1.0)},
            ).item(),
            3.0,
        )
        with self.assertRaisesRegex(KeyError, "locomotion_reward"):
            trainer._learning_reward(torch.tensor(1.0), {"collapse_cost": 0.0})

    def test_reset_windows_resolve_once_at_horizon_or_collapse(self):
        outcomes = []
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        safety_cfg = dict(config().safety_constraint)
        safety_cfg["horizon"] = 3
        trainer.cfg = config(safety_constraint=safety_cfg)
        trainer.buffer = SimpleNamespace(task_names=[TASKS[0]])
        trainer.agent = SimpleNamespace(
            observe_safety_outcome=lambda task, outcome, horizon: outcomes.append(
                (task, outcome, horizon)
            ) or {}
        )
        trainer._safety_window_steps = {}
        trainer._safety_window_active = {}

        trainer._start_safety_window(0)
        for _ in range(3):
            trainer._record_safety_transition(0, {"collapse_cost": 0.0}, False)
        trainer._record_safety_transition(0, {"collapse_cost": 1.0}, True)
        trainer._start_safety_window(0)
        trainer._record_safety_transition(0, {"collapse_cost": 1.0}, True)

        self.assertEqual(outcomes, [(TASKS[0], 0.0, 3), (TASKS[0], 1.0, 3)])

    def test_in_flight_window_keeps_horizon_captured_at_start(self):
        outcomes = []
        active = {TASKS[0]: 3}
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = config()
        trainer.buffer = SimpleNamespace(task_names=[TASKS[0]])
        trainer.agent = SimpleNamespace(
            active_safety_horizon=lambda task: active[task],
            observe_safety_outcome=lambda task, outcome, horizon: outcomes.append(
                (task, outcome, horizon)
            ) or {},
        )
        trainer._start_safety_window(0)
        trainer._record_safety_transition(0, {"collapse_cost": 0.0}, False)
        active[TASKS[0]] = 5
        trainer._record_safety_transition(0, {"collapse_cost": 0.0}, False)
        trainer._record_safety_transition(0, {"collapse_cost": 0.0}, False)

        self.assertEqual(outcomes, [(TASKS[0], 0.0, 3)])

    def test_evaluation_uses_task_active_horizon_without_promoting(self):
        class EvalEnv:
            def reset(self, task_idx=None):
                self.step_count = 0
                return graph(1)

            def step(self, action):
                self.step_count += 1
                done = self.step_count == 2
                return graph(1), 1.0, done, {
                    "collapse_cost": float(done),
                    "success": 0.0,
                }

        predicted_tasks = []
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = config(eval_episodes=1, save_video=False)
        trainer.eval_env = EvalEnv()
        trainer.agent = SimpleNamespace(
            active_safety_horizon=lambda task: 1,
            predict_safety_risk=lambda obs, task: predicted_tasks.append(task) or 0.25,
            act=lambda obs, t0, eval_mode: torch.zeros(3, 1),
        )

        metrics = trainer._eval_one(task_name=TASKS[0])

        self.assertEqual(predicted_tasks, [TASKS[0]])
        self.assertEqual(metrics["safety_active_horizon"], 1)
        self.assertEqual(metrics["safety_horizon_collapse_rate"], 0.0)


class ReplayAndUpdateTest(unittest.TestCase):
    def test_replay_round_trip_preserves_cost_and_episode_end(self):
        cfg = config()
        replay = populated_buffer(cfg)
        state = replay.state_dict()
        restored = GNNBuffer(cfg)
        restored.load_state_dict(state)

        torch.manual_seed(7)
        first = replay.sample()
        torch.manual_seed(7)
        second = restored.sample()
        torch.testing.assert_close(first[3], second[3])
        torch.testing.assert_close(first[5], second[5])

    def test_old_replay_is_rejected_only_for_constrained_runs(self):
        cfg = config(multitask=False, tasks=[TASKS[0]], task=TASKS[0], batch_size=2)
        replay = GNNBuffer(cfg)
        replay.add(transition(1), task=TASKS[0])
        state = replay.state_dict()
        state["format_version"] = 2
        for task_state in state["buffers"].values():
            task_state.pop("collapse_cost")
            task_state.pop("episode_end")
            task_state.pop("format_version")

        with self.assertRaisesRegex(ValueError, "fresh constrained run"):
            GNNBuffer(cfg).load_state_dict(state)

        unconstrained_cfg = config(
            multitask=False,
            tasks=[TASKS[0]],
            task=TASKS[0],
            batch_size=2,
            safety_constraint={"enabled": False},
        )
        restored = GNNBuffer(unconstrained_cfg)
        restored.load_state_dict(state)
        self.assertEqual(restored.size, replay.size)

    def test_constrained_replay_requires_cost_and_end_fields(self):
        cfg = config(multitask=False, tasks=[TASKS[0]], task=TASKS[0], batch_size=2)
        payload = transition(1)
        del payload[1]["collapse_cost"]
        with self.assertRaisesRegex(ValueError, "collapse_cost"):
            GNNBuffer(cfg).add(payload, task=TASKS[0])

    def test_constrained_update_runs_with_and_without_pcgrad(self):
        for pcgrad in (False, True):
            cfg = config(pcgrad=pcgrad)
            agent = GNNSAC(cfg)
            metrics = agent.update(
                populated_buffer(cfg), compute_safety_diagnostics=True
            )
            self.assertIn("safety/cost_value_loss", metrics)
            self.assertIn("safety/predicted_risk_mean/truss-graph_octahedron", metrics)
            self.assertIn("safety/cost_target_mean/truss-graph_octahedron", metrics)
            self.assertIn("safety/cost_target_std/truss-graph_octahedron", metrics)
            self.assertIn(
                "safety/cost_target_high_fraction/truss-graph_octahedron", metrics
            )
            self.assertIn(
                "safety/risk_action_grad_norm_mean/truss-graph_octahedron", metrics
            )
            self.assertIn(
                "safety/actor_reward_grad_norm/truss-graph_octahedron", metrics
            )
            self.assertIn(
                "safety/actor_weighted_constraint_grad_norm/truss-graph_octahedron",
                metrics,
            )
            self.assertIn(
                "safety/constraint_to_reward_grad_ratio/truss-graph_octahedron",
                metrics,
            )
            self.assertTrue(torch.isfinite(metrics["safety/cost_value_loss"]))
            self.assertTrue(
                torch.isfinite(
                    metrics[
                        "safety/constraint_to_reward_grad_ratio/truss-graph_octahedron"
                    ]
                )
            )


class SafetyLoggingTest(unittest.TestCase):
    def test_safety_metrics_bypass_train_namespace(self):
        logger = Logger.__new__(Logger)
        logger._wandb = Mock()
        logger._print = Mock()

        logger.log(
            {
                "step": 12,
                "value_loss": 0.5,
                "safety/cost_value_loss": 0.25,
                "safety/lambda/truss-graph_octahedron": 2.0,
            },
            "train",
        )

        logged = logger._wandb.log.call_args.args[0]
        self.assertEqual(logged["train/value_loss"], 0.5)
        self.assertEqual(logged["safety/cost_value_loss"], 0.25)
        self.assertEqual(logged["safety/lambda/truss-graph_octahedron"], 2.0)
        self.assertNotIn("train/safety/cost_value_loss", logged)


if __name__ == "__main__":
    unittest.main()
