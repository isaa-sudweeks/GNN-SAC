"""Evaluated policy exports stay recoverable independently of replay checkpoints."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC
from padded_mlp_sac import PaddedMLPSAC
from trainer.base import Trainer
from trainer.online_trainer import OnlineTrainer
from tests.test_checkpointing import DummyAgent, DummyBuffer, DummyLogger
from tests.test_mlp_distillation import student_config


def harness(directory, *, enabled=True, agent=None):
    cfg = SimpleNamespace(
        work_dir=str(directory), checkpoint_dir="checkpoints",
        checkpoint_keep_last=1, checkpoint_async=False, checkpoint_freq=10,
        steps=100, save_eval_agent=enabled, save_video=False,
    )
    trainer = OnlineTrainer.__new__(OnlineTrainer)
    logger = DummyLogger()
    logger.log = Mock()
    Trainer.__init__(trainer, cfg, None, agent or DummyAgent(), DummyBuffer(), logger)
    trainer._step = 0
    trainer._ep_idx = 0
    trainer._eval_count = 0
    trainer.distillation = None
    trainer.eval = Mock(return_value={"episode_reward": 3.0, "episode_distance": 5.0})
    trainer.common_metrics = lambda: {"step": trainer._step}
    return trainer


class EvaluationCheckpointTest(unittest.TestCase):
    def test_disabled_by_default_leaves_evaluation_without_files(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = harness(directory, enabled=False)
            del trainer.cfg.save_eval_agent
            trainer._evaluate_and_log()
            trainer.logger.log.assert_called_once()
            self.assertFalse((Path(directory) / "checkpoints").exists())

    def test_evaluated_weights_and_metrics_survive_replay_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = harness(directory)
            trainer._step = 11  # Delayed evaluation need not align with checkpoint_freq.
            trainer.agent.model.weight.data.fill_(7.0)
            trainer._evaluate_and_log()
            path = Path(directory) / "checkpoints/evaluations/eval_step_11.agent.pt"
            for step in (20, 30):
                trainer._step = step
                trainer.agent.model.weight.data.fill_(float(step))
                trainer.save_checkpoint()
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(state["model"]["weight"].item(), 7.0)
            self.assertEqual(state["evaluation"]["step"], 11)
            self.assertEqual(state["evaluation"]["metrics"]["episode_distance"], 5.0)
            self.assertNotIn("buffer", state)
            self.assertNotIn("optim", state)
            self.assertEqual(state["config"]["steps"], 100)
            self.assertFalse((Path(directory) / "checkpoints/step_20.pt").exists())
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_gnn_and_padded_exports_keep_their_loadable_policy_schema(self):
        for backend, agent_type, schema_key in (
            ("gnn", GNNSAC, "graph_feature_schema"),
            ("padded_mlp", PaddedMLPSAC, "padded_mlp_schema"),
        ):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                cfg = student_config(directory, sac_backend=backend)
                agent = agent_type(cfg)
                trainer = harness(directory, agent=agent)
                trainer.cfg = cfg
                cfg.save_eval_agent = True
                trainer._step = 123
                trainer._evaluate_and_log()
                path = Path(directory) / "checkpoints/evaluations/eval_step_123.agent.pt"
                state = torch.load(path, map_location="cpu", weights_only=False)
                self.assertIn(schema_key, state)
                restored = agent_type(cfg)
                restored.load(path)
                for name, value in agent.model.state_dict().items():
                    torch.testing.assert_close(value, restored.model.state_dict()[name])

    def test_offline_updates_and_online_step_zero_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = harness(directory)
            trainer.cfg.distillation = {"eval_freq": 1}
            trainer.save_checkpoint = Mock()
            distillation = SimpleNamespace(
                stage="offline", completed_updates=0, pretrain_updates=2,
                log_freq=1, checkpoint_freq=0,
            )

            def offline_update(agent):
                distillation.completed_updates += 1
                agent.model.weight.data.fill_(float(distillation.completed_updates))
                return {"offline_updates": distillation.completed_updates}

            distillation.offline_update = offline_update
            distillation.finish_pretraining = lambda agent: setattr(distillation, "stage", "online")
            trainer.distillation = distillation
            trainer._run_distillation_pretraining()
            snapshots = Path(directory) / "checkpoints/evaluations"
            self.assertEqual(
                {path.name for path in snapshots.iterdir()},
                {"offline_update_1.agent.pt", "offline_update_2.agent.pt", "eval_step_0.agent.pt"},
            )
            for update in (1, 2):
                state = torch.load(snapshots / f"offline_update_{update}.agent.pt", weights_only=False)
                self.assertEqual(state["model"]["weight"].item(), float(update))
                self.assertEqual(state["evaluation"]["stage"], "offline")
                self.assertEqual(state["evaluation"]["offline_updates"], update)
            online = torch.load(snapshots / "eval_step_0.agent.pt", weights_only=False)
            self.assertEqual(online["evaluation"]["stage"], "online")
            self.assertEqual(online["evaluation"]["offline_updates"], 2)

    def test_failed_agent_save_removes_temporary_export(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = harness(directory)

            def fail(path):
                Path(path).write_bytes(b"incomplete")
                raise RuntimeError("simulated write failure")

            trainer.agent.save = fail
            with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                trainer._evaluate_and_log()
            self.assertFalse(list((Path(directory) / "checkpoints/evaluations").iterdir()))


if __name__ == "__main__":
    unittest.main()
