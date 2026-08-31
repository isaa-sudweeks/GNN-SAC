from pathlib import Path
from types import SimpleNamespace
from threading import Event
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trainer.base import Trainer
from common.logger import Logger, wandb_resume_info
from common.reward_normalizer import TaskRewardNormalizer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class DummyAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def load_training_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])

    def save(self, path):
        torch.save({"model": self.model.state_dict()}, path)


class DummyBuffer:
    def __init__(self):
        self.value = torch.tensor([0.0])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]


class DummyLogger:
    def __init__(self):
        self.rows = []

    def state_dict(self):
        return {"rows": self.rows}

    def load_state_dict(self, state_dict):
        self.rows = state_dict["rows"]


def make_trainer(work_dir, **overrides):
    cfg = SimpleNamespace(
        **{
            "work_dir": str(work_dir),
            "checkpoint_dir": "checkpoints",
            "checkpoint_async": False,
            "checkpoint_freq": 10,
            "checkpoint_keep_last": 2,
            "resume_from_checkpoint": None,
            "steps": 100,
            **overrides,
        }
    )
    trainer = Trainer(
        cfg=cfg,
        env=None,
        agent=DummyAgent(),
        buffer=DummyBuffer(),
        logger=DummyLogger(),
    )
    trainer._step = 0
    trainer._ep_idx = 0
    return trainer


class CheckpointingTest(unittest.TestCase):
    def test_checkpoint_round_trip_restores_training_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            trainer._step = 20
            trainer._ep_idx = 3
            trainer._update_budget = 0.75
            trainer._pending_update_transitions = 1536
            trainer._vector_steps_since_update = 6
            trainer._pretrain_complete = True
            trainer._optimizer_updates = 17
            trainer._last_eval_step = 20
            trainer._eval_count = 4
            trainer.reward_normalizer = TaskRewardNormalizer(
                gamma=0.9,
                allowed_tasks=["task"],
            )
            trainer.reward_normalizer.normalize(3.0, task="task", stream=0)
            trainer.agent.model.weight.data.fill_(5.0)
            trainer.buffer.value = torch.tensor([7.0])
            trainer.logger.rows = [{"step": 10, "episode_reward": 1.5}]
            checkpoint = trainer.save_checkpoint()

            resumed = make_trainer(Path(tmp_dir))
            resumed.reward_normalizer = TaskRewardNormalizer(
                gamma=0.9,
                allowed_tasks=["task"],
            )
            resumed.load_checkpoint_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))

            self.assertEqual(resumed._step, 20)
            self.assertEqual(resumed._ep_idx, 3)
            self.assertEqual(resumed._update_budget, 0.75)
            self.assertEqual(resumed._pending_update_transitions, 1536)
            self.assertEqual(resumed._vector_steps_since_update, 6)
            self.assertTrue(resumed._pretrain_complete)
            self.assertEqual(resumed._optimizer_updates, 17)
            self.assertEqual(resumed._last_eval_step, 20)
            self.assertEqual(resumed._eval_count, 4)
            self.assertEqual(float(resumed.agent.model.weight.item()), 5.0)
            self.assertEqual(float(resumed.buffer.value.item()), 7.0)
            self.assertEqual(resumed.logger.rows, [{"step": 10, "episode_reward": 1.5}])
            self.assertEqual(
                resumed.reward_normalizer.state_dict(),
                trainer.reward_normalizer.state_dict(),
            )

    def test_latest_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            for step in (10, 20, 30):
                trainer._step = step
                trainer.save_checkpoint()

            checkpoint_dir = Path(tmp_dir) / "checkpoints"
            self.assertTrue((checkpoint_dir / "latest.pt").exists())
            self.assertTrue((checkpoint_dir / "latest.agent.pt").exists())
            self.assertFalse((checkpoint_dir / "step_10.pt").exists())
            self.assertFalse((checkpoint_dir / "step_10.agent.pt").exists())
            self.assertTrue((checkpoint_dir / "step_20.pt").exists())
            self.assertTrue((checkpoint_dir / "step_20.agent.pt").exists())
            self.assertTrue((checkpoint_dir / "step_30.pt").exists())
            self.assertTrue((checkpoint_dir / "step_30.agent.pt").exists())
            metadata = json.loads((checkpoint_dir / "latest.metadata.json").read_text())
            self.assertFalse((checkpoint_dir / ".latest.metadata.tmp").exists())
            self.assertEqual(
                metadata,
                {
                    "format_version": 1,
                    "step": 30,
                    "target_steps": 100,
                    "checkpoint": "step_30.pt",
                },
            )

    def test_older_checkpoint_starts_enabled_normalizer_with_empty_statistics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            state = trainer.checkpoint_state_dict()
            self.assertNotIn("reward_normalizer", state)

            resumed = make_trainer(Path(tmp_dir))
            resumed.reward_normalizer = TaskRewardNormalizer(
                gamma=0.9,
                allowed_tasks=["task"],
            )
            resumed.load_checkpoint_state_dict(state)

            self.assertEqual(resumed.reward_normalizer.metrics(), {})

    def test_async_checkpoint_writes_snapshot_after_training_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir), checkpoint_async=True)
            trainer._step = 10
            trainer.agent.model.weight.data.fill_(3.0)
            trainer.buffer.value = torch.tensor([4.0])

            entered_writer = Event()
            release_writer = Event()
            original_writer = Trainer._write_checkpoint_files.__func__

            def delayed_writer(cls, *args, **kwargs):
                entered_writer.set()
                release_writer.wait(timeout=5)
                return original_writer(cls, *args, **kwargs)

            with patch.object(Trainer, "_write_checkpoint_files", classmethod(delayed_writer)):
                checkpoint = trainer.maybe_save_checkpoint(previous_step=9)
                self.assertTrue(entered_writer.wait(timeout=5))
                trainer._step = 20
                trainer.agent.model.weight.data.fill_(8.0)
                trainer.buffer.value = torch.tensor([9.0])
                release_writer.set()
                trainer.wait_for_async_checkpoint()

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["trainer"]["step"], 10)
            self.assertEqual(float(saved["agent"]["model"]["weight"].item()), 3.0)
            self.assertEqual(float(saved["buffer"]["value"].item()), 4.0)
            agent_sidecar = torch.load(
                Path(tmp_dir) / "checkpoints" / "latest.agent.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(float(agent_sidecar["model"]["weight"].item()), 3.0)

    def test_forced_checkpoint_waits_for_pending_async_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir), checkpoint_async=True)
            trainer._step = 10

            entered_writer = Event()
            release_writer = Event()
            original_writer = Trainer._write_checkpoint_files.__func__
            delayed_identifiers = []

            def delayed_first_writer(cls, checkpoint_dir, identifier, *args):
                if identifier == "step_10":
                    delayed_identifiers.append(identifier)
                    entered_writer.set()
                    release_writer.wait(timeout=5)
                return original_writer(cls, checkpoint_dir, identifier, *args)

            with patch.object(Trainer, "_write_checkpoint_files", classmethod(delayed_first_writer)):
                trainer.maybe_save_checkpoint(previous_step=9)
                self.assertTrue(entered_writer.wait(timeout=5))
                trainer._step = 15
                release_writer.set()
                forced_checkpoint = trainer.maybe_save_checkpoint(force=True)

            self.assertEqual(delayed_identifiers, ["step_10"])
            self.assertTrue((Path(tmp_dir) / "checkpoints" / "step_10.pt").exists())
            self.assertTrue(forced_checkpoint.exists())
            self.assertIsNone(trainer._checkpoint_future)

    def test_resume_latest_resolves_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            trainer._step = 10
            trainer.agent.model.weight.data.fill_(9.0)
            trainer.save_checkpoint()

            resumed = make_trainer(Path(tmp_dir), resume_from_checkpoint="latest")
            resumed.maybe_load_checkpoint()
            self.assertEqual(resumed._step, 10)
            self.assertEqual(float(resumed.agent.model.weight.item()), 9.0)

    def test_missing_latest_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir), resume_from_checkpoint="latest")
            self.assertIsNone(trainer.maybe_load_checkpoint())
            self.assertEqual(trainer._step, 0)

    def test_rng_restore_accepts_non_byte_tensor_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            state = trainer._rng_state_dict()
            state["torch"] = state["torch"].to(torch.int16)

            trainer._load_rng_state_dict(state)

    def test_wandb_resume_info_uses_local_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            checkpoint_dir = work_dir / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"logger": {"wandb": {"id": "checkpoint-run"}}}, checkpoint_dir / "latest.pt")
            (work_dir / "wandb_run.json").write_text(json.dumps({"id": "local-run"}))

            cfg = SimpleNamespace(
                work_dir=str(work_dir),
                checkpoint_dir="checkpoints",
                resume_from_checkpoint="latest",
            )

            self.assertEqual(wandb_resume_info(cfg, work_dir)["id"], "local-run")

    def test_wandb_resume_info_falls_back_to_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            checkpoint_dir = work_dir / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"logger": {"wandb": {"id": "checkpoint-run"}}}, checkpoint_dir / "latest.pt")

            cfg = SimpleNamespace(
                work_dir=str(work_dir),
                checkpoint_dir="checkpoints",
                resume_from_checkpoint="latest",
            )

            self.assertEqual(wandb_resume_info(cfg, work_dir)["id"], "checkpoint-run")


class WandbInitTest(unittest.TestCase):
    def test_logger_resumes_wandb_run_and_can_set_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            checkpoint_dir = work_dir / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"logger": {"wandb": {"id": "resume-run"}}}, checkpoint_dir / "latest.pt")

            init_calls = []
            artifact_names = []

            class FakeRun:
                id = "resume-run"

            class FakeArtifact:
                def __init__(self, name, type):
                    artifact_names.append(name)

                def add_file(self, path):
                    return

            class FakeWandb:
                run = FakeRun()
                Artifact = FakeArtifact

                @staticmethod
                def init(**kwargs):
                    init_calls.append(kwargs)
                    return FakeRun()

                @staticmethod
                def log_artifact(artifact):
                    return

            cfg = SimpleNamespace(
                work_dir=str(work_dir),
                save_csv=False,
                save_agent=True,
                env_name="env",
                exp_name="exp",
                seed=1,
                steps=10,
                wandb_project="project",
                wandb_dir=str(work_dir / "shared-wandb-parent"),
                wandb_entity=None,
                wandb_name="run-name",
                launch_command="python sac/gnn_train.py steps=10",
                multirun_id="job_0002_deadbeefcafe",
                wandb_silent=True,
                enable_wandb=True,
                save_video=False,
                checkpoint_dir="checkpoints",
                resume_from_checkpoint="latest",
                set_wandb_offline=True,
            )

            with patch.dict(sys.modules, {"wandb": FakeWandb}), patch.dict("os.environ", {}, clear=True):
                logger = Logger(cfg)
                logger.save_agent(DummyAgent())

            self.assertEqual(init_calls[0]["id"], "resume-run")
            self.assertEqual(init_calls[0]["resume"], "allow")
            self.assertEqual(init_calls[0]["mode"], "offline")
            self.assertEqual(init_calls[0]["name"], "run-name-job_0002_deadbeefcafe")
            self.assertEqual(init_calls[0]["dir"], str(work_dir / "shared-wandb-parent"))
            self.assertEqual(
                init_calls[0]["config"]["launch_command"],
                "python sac/gnn_train.py steps=10",
            )
            self.assertEqual(
                artifact_names,
                ["env-exp-1-job_0002_deadbeefcafe-final"],
            )
            self.assertEqual(json.loads((work_dir / "wandb_run.json").read_text())["id"], "resume-run")
            self.assertEqual(logger.state_dict()["wandb"]["id"], "resume-run")


if __name__ == "__main__":
    unittest.main()
