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
from common.gnn_buffer import GNNBuffer
from env import make_env
from gnn_sac import GNNSAC


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

    def test_topology_buckets_allocate_num_envs_to_each_topology_and_step_mixed_graphs(self):
        cfg = mjx_cfg(
            num_envs=2,
            truss_topologies=["octahedron", "tetrahedron"],
        )
        env = make_env(cfg)
        try:
            bucket_env = env.env
            self.assertEqual(
                bucket_env.topology_allocations,
                {"octahedron": 2, "tetrahedron": 2},
            )
            self.assertEqual(cfg.envs_per_topology, 2)
            self.assertEqual(cfg.num_envs, 4)
            self.assertEqual(env.num_envs, 4)
            self.assertEqual(
                [bucket_env.topology_for_env(index) for index in range(4)],
                ["octahedron", "tetrahedron", "octahedron", "tetrahedron"],
            )

            observations = env.reset_many()
            self.assertEqual(len(observations), 4)
            self.assertNotEqual(observations[0].num_nodes, observations[1].num_nodes)
            self.assertEqual(observations[0].num_nodes, observations[2].num_nodes)
            self.assertEqual(observations[1].num_nodes, observations[3].num_nodes)

            agent = GNNSAC(cfg)
            actions = agent.act_batch(observations, eval_mode=True)

            env.step_many(actions[:2], env_indices=[0, 1])
            for bucket in bucket_env.buckets:
                step_count = bucket._jax.device_get(bucket._state.step_count)
                self.assertEqual(step_count.tolist(), [1, 0])

            observations = env.reset_many()
            actions = agent.act_batch(observations, eval_mode=True)
            results = env.step_many(actions)
            self.assertEqual([result[3]["env_idx"] for result in results], list(range(4)))
            self.assertEqual(
                [result[3]["topology"] for result in results],
                ["octahedron", "tetrahedron", "octahedron", "tetrahedron"],
            )
            for observation, action, result in zip(observations, actions, results):
                self.assertEqual(action.shape, (observation.num_nodes, 1))
                self.assertEqual(result[0].num_nodes, observation.num_nodes)
                self.assertTrue(torch.isfinite(result[1]))

            next_observations = [result[0] for result in results]
            next_actions = agent.act_batch(next_observations, eval_mode=True)
            next_results = env.step_many(next_actions)
            buffer = GNNBuffer(cfg)
            for env_idx in (0, 1):
                first_info = results[env_idx][3]
                second_info = next_results[env_idx][3]
                buffer.add(
                    [
                        {
                            "obs": observations[env_idx],
                            "action": torch.zeros_like(actions[env_idx]).unsqueeze(0),
                            "reward": torch.tensor(0.0),
                            "terminated": torch.tensor(0.0),
                        },
                        {
                            "obs": next_observations[env_idx],
                            "action": actions[env_idx].unsqueeze(0),
                            "reward": results[env_idx][1],
                            "terminated": first_info["terminated"],
                        },
                        {
                            "obs": next_results[env_idx][0],
                            "action": next_actions[env_idx].unsqueeze(0),
                            "reward": next_results[env_idx][1],
                            "terminated": second_info["terminated"],
                        },
                    ]
                )
            update_info = agent.update(buffer)
            self.assertIn("value_loss", update_info)
            self.assertIn("pi_loss", update_info)
        finally:
            env.close()

    def test_topology_buckets_require_at_least_one_environment_per_topology(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            make_env(
                mjx_cfg(
                    num_envs=0,
                    truss_topologies=["octahedron", "tetrahedron"],
                )
            )


if __name__ == "__main__":
    unittest.main()
