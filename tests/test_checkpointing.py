from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trainer.base import Trainer


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
            "checkpoint_freq": 10,
            "checkpoint_keep_last": 2,
            "resume_from_checkpoint": None,
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
            trainer.agent.model.weight.data.fill_(5.0)
            trainer.buffer.value = torch.tensor([7.0])
            trainer.logger.rows = [{"step": 10, "episode_reward": 1.5}]
            checkpoint = trainer.save_checkpoint()

            resumed = make_trainer(Path(tmp_dir))
            resumed.load_checkpoint_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))

            self.assertEqual(resumed._step, 20)
            self.assertEqual(resumed._ep_idx, 3)
            self.assertEqual(float(resumed.agent.model.weight.item()), 5.0)
            self.assertEqual(float(resumed.buffer.value.item()), 7.0)
            self.assertEqual(resumed.logger.rows, [{"step": 10, "episode_reward": 1.5}])

    def test_latest_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = make_trainer(Path(tmp_dir))
            for step in (10, 20, 30):
                trainer._step = step
                trainer.save_checkpoint()

            checkpoint_dir = Path(tmp_dir) / "checkpoints"
            self.assertTrue((checkpoint_dir / "latest.pt").exists())
            self.assertFalse((checkpoint_dir / "step_10.pt").exists())
            self.assertTrue((checkpoint_dir / "step_20.pt").exists())
            self.assertTrue((checkpoint_dir / "step_30.pt").exists())

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


if __name__ == "__main__":
    unittest.main()
