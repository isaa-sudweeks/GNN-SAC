from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_infer import load_agent_checkpoint


def fail_if_unpickled():
    raise AssertionError("replay buffer should not be unpickled for inference")


class UnloadableReplay:
    def __reduce__(self):
        return fail_if_unpickled, ()


class DummyAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.loaded = None

    def load(self, state_dict):
        self.loaded = state_dict

    def save(self, path):
        torch.save(self.loaded, path)


class InferenceCheckpointTest(unittest.TestCase):
    def test_full_checkpoint_creates_and_reuses_agent_only_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "latest.pt"
            torch.save(
                {
                    "agent": {"model": {"weight": torch.tensor([3.0])}},
                    "buffer": UnloadableReplay(),
                },
                checkpoint,
            )

            first_agent = DummyAgent()
            load_agent_checkpoint(first_agent, checkpoint)
            sidecar = Path(tmp_dir) / "latest.agent.pt"
            self.assertTrue(sidecar.exists())
            self.assertEqual(float(first_agent.loaded["model"]["weight"].item()), 3.0)

            torch.save({"model": {"weight": torch.tensor([7.0])}}, sidecar)
            sidecar.touch()
            second_agent = DummyAgent()
            load_agent_checkpoint(second_agent, checkpoint)
            self.assertEqual(float(second_agent.loaded["model"]["weight"].item()), 7.0)


if __name__ == "__main__":
    unittest.main()
