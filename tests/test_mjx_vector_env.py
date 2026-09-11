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
            self.assertEqual(env.env._core.mjx_impl, "jax")
            observations = env.reset_many()
            self.assertEqual(len(observations), 2)
            self.assertTrue(all(isinstance(obs, Data) for obs in observations))
            self.assertTrue(all(obs.x.shape[1] == 6 for obs in observations))
            self.assertTrue(all(obs.rigidity.shape == (1,) for obs in observations))
            self.assertTrue(all(torch.isfinite(obs.rigidity).all() for obs in observations))
            self.assertEqual(cfg.num_policy_actions, int(observations[0].action_mask.sum()))

            action = env.rand_act(env_idx=1)
            results = env.step_many([action], env_indices=[1])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][3]["env_idx"], 1)
            self.assertTrue(
                torch.allclose(
                    results[0][0].rigidity,
                    results[0][3]["critical_eig"].reshape(1),
                )
            )
            self.assertTrue(torch.isfinite(results[0][1]))

            step_count = env.env._jax.device_get(env.env._state.step_count)
            self.assertEqual(step_count.tolist(), [0, 1])
        finally:
            env.close()

    def test_energy_penalizes_routed_actuator_commands(self):
        cfg = mjx_cfg(
            num_envs=1,
            speed=0.05,
            forward_weight=0.0,
            energy_weight=0.1,
            alive_bonus=0.0,
            rigidity_weight=0.0,
            slip_weight=0.0,
            collapse_penalty=0.0,
            critical_eig_threshold=0.0,
        )
        env = make_env(cfg)
        try:
            env.reset_many()
            action = torch.linspace(
                -1.0,
                1.0,
                env.action_space.shape[0],
                dtype=torch.float32,
            ).reshape(env.action_space.shape)

            _, reward, _, info = env.step_many([action], env_indices=[0])[0]

            core = env.env._core
            ctrl = env.env._state.data.ctrl[0, core._actuator_ids]
            expected_penalty = float(
                env.env._jax.device_get(env.env._jnp.sum(env.env._jnp.square(ctrl)))
            )
            self.assertAlmostEqual(float(info["energy_penalty_raw"]), expected_penalty)
            self.assertAlmostEqual(float(info["energy"]), -cfg.energy_weight * expected_penalty)
            self.assertAlmostEqual(float(reward), float(info["energy"]), places=6)
            self.assertNotAlmostEqual(
                expected_penalty,
                float(torch.sum(torch.square(action * cfg.speed))),
            )
        finally:
            env.close()

    def test_configurable_graph_features_flow_through_mjx_policy(self):
        cfg = mjx_cfg(
            num_envs=1,
            graph_features={
                "node_roles": True,
                "edge_roles": True,
                "edge_distance": True,
            },
        )
        env = make_env(cfg)
        try:
            observation = env.reset_many()[0]
            self.assertEqual(observation.x.shape[1], 6)
            self.assertEqual(
                observation.edge_role.shape,
                (observation.edge_index.shape[1],),
            )
            self.assertEqual(cfg.effective_node_feature_dim, 10)
            self.assertEqual(cfg.edge_feature_dim, 4)

            agent = GNNSAC(cfg)
            action = agent.act(observation, eval_mode=True)
            self.assertEqual(action.shape, (observation.num_nodes, 1))
            self.assertTrue(torch.isfinite(action).all())
        finally:
            env.close()

    def test_broken_nodes_are_per_environment_and_partial_reset_isolated(self):
        cfg = mjx_cfg(
            num_envs=8,
            seed=17,
            domain_randomization=True,
            domain_randomization_params={
                "length_scale": {"enabled": False},
                "broken_nodes": {"enabled": True, "probability": 0.5}
            },
            graph_features={"node_roles": True},
        )
        env = make_env(cfg)
        try:
            observations = env.reset_many()
            core = env.env
            masks_before = core._broken_node_masks.clone()
            base_active = torch.as_tensor(~core._base_passive_node_mask)

            self.assertGreater(len({tuple(row.tolist()) for row in masks_before}), 1)
            self.assertGreater(len({int(obs.action_mask.sum()) for obs in observations}), 1)
            for env_idx, observation in enumerate(observations):
                self.assertTrue(torch.equal(
                    observation.action_mask.cpu(),
                    base_active & ~masks_before[env_idx].cpu(),
                ))
                self.assertGreaterEqual(int(observation.action_mask.sum()), 1)

            agent = GNNSAC(cfg)
            selected_observations = observations[:2]
            actions = agent.act_batch(selected_observations)
            for observation, action in zip(selected_observations, actions):
                self.assertEqual(action.shape, (observation.num_nodes, 1))
                self.assertTrue(torch.equal(
                    action[~observation.action_mask],
                    torch.zeros_like(action[~observation.action_mask]),
                ))
            results = env.step_many(actions, env_indices=[0, 1])
            next_observations = [result[0] for result in results]
            next_actions = agent.act_batch(next_observations)
            next_results = env.step_many(next_actions, env_indices=[0, 1])

            buffer = GNNBuffer(cfg)
            for env_idx in range(2):
                first_info = results[env_idx][3]
                second_info = next_results[env_idx][3]
                buffer.add(
                    [
                        {
                            "obs": selected_observations[env_idx],
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
            self.assertTrue(torch.isfinite(update_info["value_loss"]))
            self.assertTrue(torch.isfinite(update_info["pi_loss"]))

            reset_observation = env.reset_many(env_indices=[0])[0]
            self.assertTrue(torch.equal(
                core._broken_node_masks[1:], masks_before[1:]
            ))
            self.assertTrue(torch.equal(
                reset_observation.action_mask.cpu(),
                base_active & ~core._broken_node_masks[0].cpu(),
            ))
        finally:
            env.close()

    def test_mjx_broken_nodes_zero_commands_and_report_diagnostics(self):
        cfg = mjx_cfg(
            num_envs=1,
            domain_randomization=True,
            domain_randomization_params={
                "length_scale": {"enabled": False},
                "broken_nodes": {"enabled": True, "probability": 1.0}
            },
            graph_features={"node_roles": True},
        )
        env = make_env(cfg)
        try:
            observation = env.reset_many()[0]
            action = torch.ones(env.action_space.shape, dtype=torch.float32)
            next_observation, reward, _, info = env.step_many([action], [0])[0]

            core = env.env
            broken = core._broken_node_masks[0].cpu()
            self.assertEqual(int(observation.action_mask.sum()), 1)
            self.assertEqual(int(broken.sum()), int((~core._base_passive_node_mask).sum()) - 1)
            self.assertTrue(torch.equal(next_observation.action_mask, observation.action_mask))
            self.assertTrue(torch.equal(info["broken_node_mask"].cpu(), broken))
            self.assertEqual(info["broken_node_count"], int(broken.sum()))

            expected_nodes = observation.action_mask.to(torch.float32) * cfg.speed
            expected_ctrl = core._jnp.clip(
                core._core._incidence_matrix @ core._jnp.asarray(expected_nodes.numpy()),
                core._core._ctrl_low,
                core._core._ctrl_high,
            )
            actual_ctrl = core._state.data.ctrl[0, core._core._actuator_ids]
            self.assertTrue(torch.allclose(
                torch.tensor(core._jax.device_get(actual_ctrl).tolist()),
                torch.tensor(core._jax.device_get(expected_ctrl).tolist()),
            ))
            self.assertTrue(torch.isfinite(reward))
        finally:
            env.close()

    def test_rejects_nonpositive_warp_capacities_before_upstream_construction(self):
        with self.assertRaisesRegex(ValueError, "warp_naconmax must be a positive integer"):
            make_env(mjx_cfg(mjx_impl="warp", warp_naconmax=0))

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
                "abstract_node_mass_multiplier": {
                    "enabled": True,
                    "min": 1.25,
                    "max": 1.25,
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
            node_mass_multipliers = env.env._jax.device_get(
                state.abstract_node_mass_multipliers
            )
            self.assertEqual(
                node_mass_multipliers.shape,
                (2, len(env.env._core.mujoco_model.node_names)),
            )
            self.assertTrue(
                torch.allclose(
                    torch.tensor(node_mass_multipliers.tolist()),
                    torch.full(node_mass_multipliers.shape, 1.25),
                )
            )
        finally:
            env.close()

    def test_randomizes_realistic_hinge_position_kp_in_mjx(self):
        cfg = mjx_cfg(
            num_envs=1,
            truss_realistic=True,
            domain_randomization=True,
            domain_randomization_params={
                "length_scale": {"enabled": False},
                "hinge_position_kp": {
                    "enabled": True,
                    "min": 9.0,
                    "max": 9.0,
                },
            },
        )
        env = make_env(cfg)
        try:
            env.reset_many()
            sampled = env.env._jax.device_get(
                env.env._state.domain_randomization.hinge_position_kp
            )
            self.assertEqual(sampled.tolist(), [9.0])
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
            self.assertEqual(cfg.num_policy_actions, int(observations[0].action_mask.sum()))
            result = env.step_many([env.rand_act(env_idx=0)], env_indices=[0])[0]
            self.assertEqual(result[0].num_nodes, observations[0].num_nodes)
            self.assertTrue(torch.isfinite(result[1]))
        finally:
            env.close()

    def test_realistic_mjx_uses_connector_ball_control_observations(self):
        cfg = mjx_cfg(
            num_envs=1,
            truss_realistic=True,
            control_node_observation_source="connector_ball",
        )
        env = make_env(cfg)
        try:
            observations = env.reset_many()
            core = env.env._core
            expected_body_ids = core.mujoco_model.get_control_node_body_ids(
                "connector_ball"
            )
            actual_body_ids = env.env._jax.device_get(core._control_body_ids)
            self.assertEqual(actual_body_ids.tolist(), expected_body_ids.tolist())
            self.assertTrue(torch.isfinite(observations[0].x).all())
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
            use_virtual_node=True,
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

    def test_topology_buckets_require_positive_num_envs(self):
        with self.assertRaisesRegex(ValueError, "num_envs must be positive"):
            make_env(
                mjx_cfg(
                    num_envs=0,
                    truss_topologies=["octahedron", "tetrahedron"],
                )
            )

    def test_topology_buckets_round_and_persist_total_environment_count(self):
        cfg = mjx_cfg(
            num_envs=3,
            truss_topologies=["octahedron", "tetrahedron"],
        )
        with self.assertWarnsRegex(RuntimeWarning, "num_envs=3.*using 4"):
            env = make_env(cfg)
        try:
            self.assertEqual(cfg.num_envs, 4)
            self.assertEqual(env.num_envs, 4)
            self.assertEqual(
                env.env.topology_allocations,
                {"octahedron": 2, "tetrahedron": 2},
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
