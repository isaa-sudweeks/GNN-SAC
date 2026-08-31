from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_infer import _enable_smooth_human_rendering


class SmoothHumanRenderingTest(unittest.TestCase):
    def test_smooth_advance_returns_early_collapse_telemetry(self):
        class DummyMjModel:
            def __init__(self):
                self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
                self.data = SimpleNamespace(ctrl=np.zeros(1, dtype=np.float32))
                self.uses_mjx = False
                self._critical_eigs = iter([0.5, 0.05])

            def set_external_ctrl(self, ctrl):
                self.data.ctrl[:] = ctrl

            def apply_angle_bisector_control(self):
                pass

            def collapse_check(self):
                return next(self._critical_eigs)

        class DummyEnv:
            def __init__(self):
                self.mj_model = DummyMjModel()
                self.config = SimpleNamespace(critical_eig_threshold=0.1)
                self.nsubsteps = 5
                self.steps = 0
                self.viewer = None

            def _apply_control_noise(self, ctrl):
                return ctrl

            def _advance(self, ctrl):
                self.steps += 1

        env = DummyEnv()
        cfg = SimpleNamespace(
            visualize=True,
            visualize_smooth=True,
            visualize_realtime=True,
            visualize_speed=1.0,
            visualize_fps=60,
        )
        mujoco = SimpleNamespace(mj_step=lambda model, data: None)

        with patch.dict(sys.modules, {"mujoco": mujoco}):
            self.assertTrue(_enable_smooth_human_rendering(env, cfg))

        advance_info = env._advance(np.zeros(1, dtype=np.float32))

        self.assertEqual(env.steps, 1)
        self.assertEqual(advance_info["substeps_executed"], 2)
        self.assertAlmostEqual(advance_info["minimum_substep_critical_eig_raw"], 0.05)
        self.assertTrue(advance_info["terminated_during_substeps"])


if __name__ == "__main__":
    unittest.main()
