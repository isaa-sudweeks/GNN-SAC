"""Offline-only distillation lifecycle and fresh-online initialization checks."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sac"))

from common.distillation import Distillation
from common.gnn_buffer import GNNBuffer
from common.tensor_gnn_buffer import TensorGNNBuffer
from gnn_sac import GNNSAC
from trainer.base import Trainer
from trainer.online_trainer import OnlineTrainer
from train import initialize_agent_from_export, validate_training_mode
from tests.test_checkpointing import DummyLogger
from tests.test_distillation import config, teacher_checkpoint


class OfflineHarness(OnlineTrainer):
    """Real distillation/checkpoint methods with evaluation isolated from physics."""

    def __init__(self, cfg):
        agent = GNNSAC(cfg)
        logger = DummyLogger()
        logger.log = Mock()
        logger.finish = Mock()
        Trainer.__init__(self, cfg, None, agent, TensorGNNBuffer(cfg), logger)
        self._step = self._ep_idx = self._optimizer_updates = 0
        self._eval_count = 0
        self._last_eval_step = None
        self._pretrain_complete = False
        self.eval_env = self.env
        self.performance_profiler = SimpleNamespace(finalize=Mock())
        self._ensure_performance_profiler = Mock()
        self._vector_collection_buckets = Mock(side_effect=AssertionError("Entered collection"))
        self.eval = Mock(return_value={"episode_reward": 1.0})
        self.common_metrics = Mock(return_value={"step": 0})
        self.distillation = Distillation(cfg, cfg.tasks, defer_target_cache=True)
        self.agent.distillation = self.distillation
        self.distillation.bind_replay(self.buffer)


def offline_config(directory, **overrides):
    cfg = config(directory, steps=0, **overrides)
    cfg.distillation.update(offline_only=True, pretrain_updates=2)
    cfg.distillation["teachers"] = {
        "a": teacher_checkpoint(directory, "a", 3),
        "b": teacher_checkpoint(directory, "b", 5),
    }
    return cfg


class TrainingModeTest(unittest.TestCase):
    def test_zero_steps_requires_positive_enabled_offline_distillation(self):
        cfg = SimpleNamespace(steps=0, distillation={})
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_training_mode(cfg)
        cfg.distillation.update(offline_only=True)
        with self.assertRaisesRegex(ValueError, "enabled distillation"):
            validate_training_mode(cfg)
        cfg.distillation.update(enabled=True, pretrain_updates=1)
        validate_training_mode(cfg)
        cfg.steps = -1
        with self.assertRaisesRegex(ValueError, "steps >= 0"):
            validate_training_mode(cfg)

    def test_initialization_cannot_be_mixed_with_resume(self):
        cfg = SimpleNamespace(steps=10, distillation={}, init_from_agent="student.pt",
                              resume_from_checkpoint="latest")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_training_mode(cfg)

    def test_agent_export_initialization_does_not_restore_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            teacher = GNNSAC(cfg)
            # Populate the actor optimizer so accidental training-state restore
            # would be observable rather than comparing two empty Adam states.
            teacher.pi_optim.zero_grad()
            sum(p.sum() for p in teacher.model.actor_parameters()).backward()
            teacher.pi_optim.step()
            path = Path(directory) / "student.pt"
            torch.save({"agent": teacher.training_state_dict()}, path)
            student = GNNSAC(cfg)
            initialize_agent_from_export(student, path)
            for key, value in teacher.model.state_dict().items():
                torch.testing.assert_close(value, student.model.state_dict()[key])
            self.assertTrue(teacher.pi_optim.state)
            self.assertFalse(student.pi_optim.state)
            self.assertFalse(student.q_optim.state)
            self.assertFalse(student.alpha_optim.state)


class OfflineOnlyLifecycleTest(unittest.TestCase):
    def test_native_two_topology_padded_teachers_offline_and_online(self):
        from hydra import compose, initialize_config_dir
        from common.parser import parse_cfg
        from env import make_env
        from padded_mlp_sac import PaddedMLPSAC
        from train import run_training

        with tempfile.TemporaryDirectory() as directory:
            overrides = [
                "sac_backend=padded_mlp", "padded_mlp_hidden_dims=[16,16]",
                "device=cpu", "enable_wandb=false", "save_video=false",
                "save_agent=false", "save_csv=false", "checkpoint_freq=0",
                "domain_randomization=false", "max_steps=2", "nsubsteps=1",
                "batch_size=2", "buffer_size=8", "eval_episodes=1",
                "seed_steps=2", "pretrain_steps=1", "normalize_rewards=false",
                "replay_ratio=1", "eval_freq=100", "log_std_min=-2", "log_std_max=0",
            ]
            with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
                teacher_configs = {
                    topology: compose(config_name="config", overrides=[
                        *overrides, "steps=4", f"truss_topology={topology}",
                        f"work_dir={directory}/teacher-{topology}",
                    ]) for topology in ("tetrahedron", "octahedron")
                }
                student_cfg = compose(config_name="config", overrides=[
                    *overrides, "steps=0", "distillation=kl",
                    "truss_topologies=[tetrahedron,octahedron]",
                    "distillation.offline_only=true", "distillation.pretrain_updates=2",
                    "distillation.batch_size=1", "distillation.shard_size=2",
                    "distillation.checkpoint_freq=0", "distillation.prefetch=false",
                    "distillation.pin_memory=false", f"work_dir={directory}/offline",
                ])
            mapping = {}
            for topology, raw_cfg in teacher_configs.items():
                cfg = parse_cfg(raw_cfg)
                env = make_env(cfg)
                try:
                    teacher = PaddedMLPSAC(cfg)
                    buffer = TensorGNNBuffer(cfg)
                    obs = env.reset()
                    episode = [dict(obs=obs, action=torch.zeros(obs.num_nodes, 1),
                                    reward=torch.tensor(0.), terminated=torch.tensor(False))]
                    for _ in range(2):
                        action = teacher.act(obs)
                        obs, reward, _, info = env.step(action)
                        episode.append(dict(obs=obs, action=action, reward=reward,
                                            terminated=info["terminated"]))
                    buffer.add(episode)
                    path = Path(directory) / f"teacher-{topology}.pt"
                    torch.save(Trainer(cfg, env, teacher, buffer, DummyLogger()).checkpoint_state_dict(), path)
                    mapping[topology] = str(path)
                finally:
                    env.close()
            student_cfg.distillation.teachers = mapping
            offline_cfg = deepcopy(student_cfg)
            run_training(offline_cfg)
            state = torch.load(Path(directory) / "offline/checkpoints/latest.pt", weights_only=False)
            self.assertEqual(state["trainer"]["step"], 0)
            self.assertEqual(state["distillation"]["completed_updates"], 2)
            self.assertEqual(len(state["buffer"]["buffers"]), 2)

            captured = []

            def capture_trainer(**kwargs):
                trainer = OnlineTrainer(**kwargs)
                captured.append(trainer)
                return trainer

            student_cfg.steps = 8
            student_cfg.distillation.offline_only = False
            student_cfg.work_dir = f"{directory}/online"
            with patch("train.OnlineTrainer", side_effect=capture_trainer):
                run_training(student_cfg)
            trainer = captured[0]
            self.assertEqual(trainer._step, 16)  # 8 interactions per topology.
            self.assertGreater(trainer._optimizer_updates, 0)
            self.assertGreater(trainer.buffer.size, 0)
            self.assertEqual(trainer.distillation.stage, "online")
            self.assertEqual(trainer.distillation.weight, 0)

            # Fine-tuning an offline export is a new run with no teacher access.
            fresh_cfg = deepcopy(student_cfg)
            fresh_cfg.distillation.enabled = False
            fresh_cfg.distillation.teachers = {"tetrahedron": "/must/not/be/opened"}
            fresh_cfg.init_from_agent = f"{directory}/offline/checkpoints/latest.agent.pt"
            fresh_cfg.work_dir = f"{directory}/fresh-online"
            with patch("train.OnlineTrainer", side_effect=capture_trainer):
                run_training(fresh_cfg)
            self.assertIsNone(captured[-1].distillation)
            self.assertGreater(captured[-1]._optimizer_updates, 0)
            self.assertEqual(captured[-1]._step, 16)

    def test_native_padded_student_offline_cli_entry_and_completed_resume(self):
        from hydra import compose, initialize_config_dir
        from common.parser import parse_cfg
        from env import make_env
        from train import run_training

        with tempfile.TemporaryDirectory() as directory:
            shared = [
                "device=cpu", "enable_wandb=false", "save_video=false",
                "save_agent=false", "save_csv=false", "checkpoint_freq=0",
                "domain_randomization=false", "max_steps=2", "nsubsteps=1",
                "batch_size=2", "buffer_size=8", "truss_topology=tetrahedron",
                "eval_episodes=1", "seed_steps=2", "pretrain_steps=1",
                "normalize_rewards=false",
            ]
            with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
                raw_teacher = compose(config_name="config", overrides=[
                    *shared, "sac_backend=gnn", "steps=4", "mpl_dims=[8]",
                    "message_hidden_dims=[8]", "head_hidden_dims=[8]",
                    f"work_dir={directory}/teacher",
                ])
                student_cfg = compose(config_name="config", overrides=[
                    *shared, "sac_backend=padded_mlp", "steps=0", "distillation=kl",
                    "padded_mlp_hidden_dims=[16,16]", "distillation.offline_only=true",
                    "distillation.pretrain_updates=2", "distillation.batch_size=1",
                    "distillation.shard_size=2", "distillation.checkpoint_freq=0",
                    "distillation.prefetch=false", "distillation.pin_memory=false",
                    f"work_dir={directory}/student",
                ])
            cfg = parse_cfg(raw_teacher)
            env = make_env(cfg)
            try:
                teacher = GNNSAC(cfg)
                buffer = TensorGNNBuffer(cfg)
                obs = env.reset()
                episode = [dict(obs=obs, action=torch.zeros(obs.num_nodes, 1),
                                reward=torch.tensor(0.), terminated=torch.tensor(False))]
                for _ in range(2):
                    action = teacher.act(obs)
                    obs, reward, _, info = env.step(action)
                    episode.append(dict(obs=obs, action=action, reward=reward,
                                        terminated=info["terminated"]))
                buffer.add(episode)
                checkpoint = Path(directory) / "teacher.pt"
                trainer = Trainer(cfg, env, teacher, buffer, DummyLogger())
                torch.save(trainer.checkpoint_state_dict(), checkpoint)
            finally:
                env.close()
            student_cfg.distillation.teachers = {"tetrahedron": str(checkpoint)}
            run_training(student_cfg)
            path = Path(directory) / "student/checkpoints/latest.pt"
            state = torch.load(path, weights_only=False)
            self.assertEqual(state["trainer"]["step"], 0)
            self.assertEqual(state["trainer"]["optimizer_updates"], 0)
            self.assertEqual(state["distillation"]["stage"], "complete")
            self.assertEqual(state["distillation"]["completed_updates"], 2)
            self.assertTrue((path.parent / "latest.agent.pt").exists())
            self.assertTrue(all(item["size"] == 0 for item in state["buffer"]["buffers"].values()))
            original_model = deepcopy(state["agent"]["model"])
            student_cfg.resume_from_checkpoint = str(path)
            run_training(student_cfg)
            resumed = torch.load(path, weights_only=False)
            self.assertEqual(resumed["trainer"]["eval_count"], state["trainer"]["eval_count"])
            for key, value in original_model.items():
                torch.testing.assert_close(value, resumed["agent"]["model"][key], rtol=0, atol=0)

    def test_completed_offline_run_saves_and_resumes_without_sac_or_repeated_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = offline_config(directory)
            trainer = OfflineHarness(cfg)
            critics_before = deepcopy(trainer.agent.model._Qs.state_dict())
            alpha_before = trainer.agent.log_alpha.detach().clone()
            trainer.train()
            self.assertEqual(trainer.distillation.stage, "complete")
            self.assertEqual(trainer.distillation.completed_updates, 2)
            self.assertEqual(trainer.distillation.weight, 0)
            self.assertEqual(trainer._step, 0)
            self.assertEqual(trainer._optimizer_updates, 0)
            self.assertEqual(trainer.buffer.size, 0)
            trainer._vector_collection_buckets.assert_not_called()
            trainer.eval.assert_called_once()
            torch.testing.assert_close(alpha_before, trainer.agent.log_alpha)
            for key, value in critics_before.items():
                torch.testing.assert_close(value, trainer.agent.model._Qs.state_dict()[key])
            checkpoint_dir = Path(directory) / "checkpoints"
            self.assertTrue((checkpoint_dir / "distillation_complete.agent.pt").exists())
            state = torch.load(checkpoint_dir / "latest.pt", weights_only=False)
            metadata = json.loads((checkpoint_dir / "latest.metadata.json").read_text())
            self.assertEqual(metadata["distillation"], {
                "offline_only": True, "stage": "complete", "completed_updates": 2,
                "pretrain_updates": 2,
            })
            resumed = OfflineHarness(cfg)
            resumed.load_checkpoint_state_dict(state)
            with patch.object(resumed.distillation, "offline_update", side_effect=AssertionError("Repeated update")):
                resumed.train()
            resumed.eval.assert_not_called()
            self.assertEqual(resumed._step, 0)
            self.assertEqual(resumed._optimizer_updates, 0)
            self.assertEqual(resumed.buffer.size, 0)
            self.assertFalse(resumed.distillation.datasets)

    def test_completed_offline_checkpoint_rejects_implicit_online_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = offline_config(directory)
            trainer = OfflineHarness(cfg)
            trainer.train()
            state = trainer.checkpoint_state_dict()
            cfg.distillation["offline_only"] = False
            cfg.steps = 100
            online = Distillation(cfg, cfg.tasks, defer_target_cache=True)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                online.load_state_dict(state["distillation"])

    def test_interrupted_offline_resume_matches_uninterrupted_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = offline_config(directory)
            source = OfflineHarness(cfg)
            source.distillation.offline_update(source.agent)
            state = deepcopy(source.checkpoint_state_dict())
            expected_metrics = source.distillation.offline_update(source.agent)
            resumed = OfflineHarness(cfg)
            resumed.load_checkpoint_state_dict(state)
            actual_metrics = resumed.distillation.offline_update(resumed.agent)
            self.assertEqual(expected_metrics, actual_metrics)
            for key, value in source.agent.model.state_dict().items():
                torch.testing.assert_close(value, resumed.agent.model.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
