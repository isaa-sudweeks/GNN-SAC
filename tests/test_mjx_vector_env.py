from pathlib import Path
import sys
import unittest

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import parse_cfg
from env import make_env


def mjx_cfg(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "gnn_config.yaml"),
        OmegaConf.create(
            {
                "mujoco_backend": "mjx",
                "use_control_graph": True,
                "num_envs": 2,
                "nsubsteps": 1,
                "max_steps": 2,
                "domain_randomization": False,
                "save_video": False,
                "enable_wandb": False,
                "device": "cpu",
                "steps": 4,
                "batch_size": 2,
                "work_dir": str(ROOT / "logs" / "test-mjx"),
            }
        ),
        OmegaConf.create(overrides),
    )
    return parse_cfg(cfg)


class MjxVectorEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from mujoco_truss_gen import MjxNodeVelocityEnv  # noqa: F401
        except (ImportError, AttributeError) as exc:
            raise unittest.SkipTest(f"updated mujoco-truss-gen is unavailable: {exc}")

    def test_batched_reset_step_and_selective_state_update(self):
        cfg = mjx_cfg()
        env = make_env(cfg)
        try:
            observations = env.reset_many()
            self.assertEqual(len(observations), 2)
            self.assertTrue(all(isinstance(obs, Data) for obs in observations))
            self.assertTrue(all(obs.x.shape[1] == 6 for obs in observations))
            self.assertEqual(cfg.num_policy_actions, observations[0].num_nodes)

            action = env.rand_act(env_idx=1)
            results = env.step_many([action], env_indices=[1])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][3]["env_idx"], 1)
            self.assertTrue(torch.isfinite(results[0][1]))

            step_count = env.env._jax.device_get(env.env._state.step_count)
            self.assertEqual(step_count.tolist(), [0, 1])
        finally:
            env.close()

    def test_rejects_model_domain_randomization(self):
        with self.assertRaisesRegex(ValueError, "domain_randomization=false"):
            make_env(mjx_cfg(domain_randomization=True))


if __name__ == "__main__":
    unittest.main()
