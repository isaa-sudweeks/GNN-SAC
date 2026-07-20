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
        OmegaConf.load(ROOT / "config" / "sac_backend" / "gnn.yaml"),
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
        with self.assertRaisesRegex(ValueError, "fixed-shape domain randomization"):
            make_env(mjx_cfg(domain_randomization=True))

    def test_accepts_fixed_shape_runtime_domain_randomization(self):
        fixed_ranges = {
            "body_mass_multiplier": 0.5,
            "body_inertia_multiplier": 0.6,
            "dof_damping_multiplier": 0.7,
            "dof_armature": 0.01,
            "dof_frictionloss": 0.02,
            "actuator_gain_multiplier": 0.8,
            "actuator_bias_multiplier": 0.9,
            "actuator_dynprm_multiplier": 1.1,
            "geom_friction_slide": 0.75,
            "geom_friction_torsional": 0.005,
            "geom_friction_rolling": 0.0005,
            "tendon_stiffness": 10.0,
            "tendon_damping": 0.2,
            "tendon_armature": 0.01,
            "tendon_frictionloss": 0.02,
            "gravity_z": -9.5,
            "initial_translation_x": 0.5,
            "initial_translation_y": -0.25,
            "initial_yaw": 1.0,
        }
        cfg = mjx_cfg(
            domain_randomization=True,
            domain_randomization_params={
                "length_scale": {"enabled": False},
                **{
                    name: {"enabled": True, "min": value, "max": value}
                    for name, value in fixed_ranges.items()
                },
            },
        )
        env = make_env(cfg)
        try:
            observations = env.reset_many()
            self.assertEqual(len(observations), 2)
            self.assertTrue(all(obs.x.shape[1] == 6 for obs in observations))
            state = env.env._state.domain_randomization
            for name, expected in fixed_ranges.items():
                sampled = env.env._jax.device_get(getattr(state, name))
                self.assertTrue(
                    torch.allclose(
                        torch.tensor(sampled.tolist()),
                        torch.full((2,), expected, dtype=torch.float32),
                    ),
                    msg=f"unexpected samples for {name}: {sampled}",
                )
        finally:
            env.close()

    def test_realistic_mjx_reset_and_step(self):
        cfg = mjx_cfg(
            num_envs=1,
            truss_realistic=True,
            max_steps=1,
            nsubsteps=1,
        )
        env = make_env(cfg)
        try:
            observations = env.reset_many()
            self.assertEqual(len(observations), 1)
            self.assertEqual(cfg.num_policy_actions, observations[0].num_nodes)
            result = env.step_many([env.rand_act(env_idx=0)], env_indices=[0])[0]
            self.assertEqual(result[0].num_nodes, observations[0].num_nodes)
            self.assertTrue(torch.isfinite(result[1]))
        finally:
            env.close()

    def test_video_is_allowed_with_native_mujoco_evaluation(self):
        env = make_env(mjx_cfg(save_video=True, eval_backend="mujoco"))
        try:
            self.assertEqual(env.env.cfg.eval_backend, "mujoco")
        finally:
            env.close()

    def test_video_is_rejected_with_mjx_evaluation(self):
        with self.assertRaisesRegex(ValueError, "eval_backend=mujoco"):
            make_env(mjx_cfg(save_video=True, eval_backend="mjx"))

    def test_topology_buckets_split_total_num_envs_across_topologies_and_step_mixed_graphs(self):
        cfg = mjx_cfg(
            num_envs=4,
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
                    ],
                    task=first_info["task"],
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

    def test_topology_buckets_require_even_total_environment_split(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_env(
                mjx_cfg(
                    num_envs=3,
                    truss_topologies=["octahedron", "tetrahedron"],
                )
            )


if __name__ == "__main__":
    unittest.main()
