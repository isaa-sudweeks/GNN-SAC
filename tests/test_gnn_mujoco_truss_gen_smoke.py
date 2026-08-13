from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from omegaconf import OmegaConf
from torch_geometric.data import Data
from mujoco_truss_gen import DomainRandomizationConfig, PRESETS, TrussPhysicalParameters

from common.gnn_buffer import GNNBuffer
from common.parser import parse_cfg
from env import make_env
from env.mujoco_gen.topology_envs import (
    _RUNTIME_DOMAIN_RANDOMIZATION_FIELDS,
    _domain_randomization,
    _physical_parameters_from_config,
    _randomized_physical_parameter_overrides,
)
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer


def flat_test_cfg(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.create(
            {
                "save_video": False,
                "multitask": False,
                "device": "cpu",
                "steps": 20,
                "seed_steps": 1,
                "batch_size": 2,
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "target_entropy": "auto",
                "work_dir": str(ROOT / "logs" / "test-smoke"),
            }
        ),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return parse_cfg(cfg)


def graph_test_cfg(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "sac_backend" / "gnn.yaml"),
        OmegaConf.create(
            {
                "save_video": False,
                "multitask": False,
                "device": "cpu",
                "steps": 20,
                "seed_steps": 1,
                "batch_size": 2,
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "target_entropy": "auto",
                "work_dir": str(ROOT / "logs" / "test-smoke"),
            }
        ),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return parse_cfg(cfg)


class GNNMujocoTrussGenSmokeTest(unittest.TestCase):
    def test_fixed_topology_mjx_training_uses_distinct_native_eval_env(self):
        cfg = SimpleNamespace(
            task="truss-graph",
            env_name="truss-graph",
            mujoco_backend="mjx",
            eval_backend="mujoco",
            num_envs=1024,
            domain_randomization=False,
            eval_task=None,
            resume_from_checkpoint=None,
            work_dir=str(ROOT / "logs" / "test-smoke"),
        )
        training_env = SimpleNamespace()
        native_eval_env = SimpleNamespace()
        agent = SimpleNamespace(model="dummy")

        with patch("env.make_env", return_value=native_eval_env) as make_eval_env:
            trainer = OnlineTrainer(
                cfg=cfg,
                env=training_env,
                agent=agent,
                buffer=None,
                logger=SimpleNamespace(),
            )

        eval_cfg = make_eval_env.call_args.args[0]
        self.assertIs(trainer.eval_env, native_eval_env)
        self.assertEqual(eval_cfg.mujoco_backend, "mujoco")
        self.assertEqual(eval_cfg.num_envs, 1)
        self.assertEqual(cfg.mujoco_backend, "mjx")

    def test_mjx_training_uses_native_mujoco_for_topology_evaluation(self):
        class DummyTrainingBuckets:
            is_topology_bucket = True
            topologies = ["octahedron", "tetrahedron"]
            topology_representative_indices = {"octahedron": 0, "tetrahedron": 1}

        class DummyNativeEvalEnv:
            num_envs = 2

            def __init__(self):
                self.active_env_idx = 0
                self.reset_task_indices = []
                self.selected_env_indices = []

            def set_active_env(self, env_idx):
                self.selected_env_indices.append(env_idx)

            def reset(self, task_idx=None):
                self.active_env_idx = int(task_idx or 0)
                self.reset_task_indices.append(task_idx)
                return torch.tensor([float(self.active_env_idx)])

            def step(self, action):
                reward = torch.tensor(float(self.active_env_idx + 1))
                info = {
                    "success": torch.tensor(float(self.active_env_idx)),
                    "com_delta_x": torch.tensor(0.25 * float(self.active_env_idx + 1)),
                    "terminated": torch.tensor(0.0),
                    "truncated": torch.tensor(1.0),
                }
                return torch.tensor([float(self.active_env_idx)]), reward, True, info

            def close(self):
                return

        class DummyAgent:
            model = "dummy"

            def act(self, obs, t0=False, eval_mode=False):
                return torch.tensor([0.0])

        class DummyLogger:
            video = None

        cfg = SimpleNamespace(
            task="truss-graph",
            env_name="truss-graph",
            mujoco_backend="mjx",
            eval_backend="mujoco",
            domain_randomization=False,
            eval_task=None,
            eval_episodes=1,
            save_video=False,
            multitask=False,
            truss_topologies=["octahedron", "tetrahedron"],
            tasks=["truss-graph:octahedron", "truss-graph:tetrahedron"],
            resume_from_checkpoint=None,
            work_dir=str(ROOT / "logs" / "test-smoke"),
        )
        eval_env = DummyNativeEvalEnv()
        captured_cfg = None

        def make_eval_env(eval_cfg):
            nonlocal captured_cfg
            captured_cfg = eval_cfg
            return eval_env

        with patch("env.make_env", side_effect=make_eval_env):
            trainer = OnlineTrainer(
                cfg=cfg,
                env=DummyTrainingBuckets(),
                agent=DummyAgent(),
                buffer=None,
                logger=DummyLogger(),
            )

        self.assertEqual(captured_cfg.mujoco_backend, "mujoco")
        self.assertTrue(captured_cfg.multitask)
        self.assertEqual(captured_cfg.num_envs, 1)
        self.assertEqual(
            list(captured_cfg.tasks),
            ["truss-graph:octahedron", "truss-graph:tetrahedron"],
        )

        metrics = trainer.eval()

        self.assertEqual(eval_env.reset_task_indices, [0, 1])
        self.assertEqual(metrics["octahedron_episode_reward"], 1.0)
        self.assertEqual(metrics["tetrahedron_episode_reward"], 2.0)
        self.assertEqual(metrics["episode_reward"], 1.5)
        self.assertEqual(metrics["octahedron_episode_distance"], 0.25)
        self.assertEqual(metrics["tetrahedron_episode_distance"], 0.5)
        self.assertEqual(metrics["episode_distance"], 0.375)

        trainer._activate_shared_eval_env(1724)
        self.assertEqual(eval_env.selected_env_indices, [])

    def test_eval_scalar_value_moves_tensor_to_host_scalar(self):
        self.assertEqual(OnlineTrainer._scalar_value(torch.tensor(2.5)), 2.5)
        self.assertEqual(OnlineTrainer._scalar_value(3), 3.0)
        with self.assertRaisesRegex(ValueError, "Expected a scalar tensor"):
            OnlineTrainer._scalar_value(torch.tensor([1.0, 2.0]))

    def test_shared_single_environment_evaluation_does_not_select_a_slot(self):
        class SingleEnvironmentWrapper:
            num_envs = 1

            def set_active_env(self, env_idx):
                raise AttributeError("Wrapped environment does not support multiple active envs")

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.env = SingleEnvironmentWrapper()
        trainer.eval_env = trainer.env

        trainer._activate_shared_eval_env(0)

    def test_shared_vector_environment_evaluation_selects_requested_slot(self):
        class VectorEnvironmentWrapper:
            num_envs = 2

            def __init__(self):
                self.selected_env_indices = []

            def set_active_env(self, env_idx):
                self.selected_env_indices.append(env_idx)

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.env = VectorEnvironmentWrapper()
        trainer.eval_env = trainer.env

        trainer._activate_shared_eval_env(1)

        self.assertEqual(trainer.env.selected_env_indices, [1])

    def test_action_noise_is_domain_randomization_gated(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        action = torch.zeros(8)

        trainer.cfg = SimpleNamespace(
            domain_randomization=False,
            domain_randomization_params={
                "action_noise": {
                    "enabled": True,
                    "std": 10.0,
                    "clip_low": -0.1,
                    "clip_high": 0.1,
                    "apply_to_seed_actions": True,
                }
            },
        )
        self.assertTrue(torch.equal(trainer._apply_action_noise(action), action))

        trainer.cfg.domain_randomization = True
        torch.manual_seed(1)
        noisy_action = trainer._apply_action_noise(action)
        self.assertFalse(torch.equal(noisy_action, action))
        self.assertTrue(torch.all(noisy_action <= 0.1))
        self.assertTrue(torch.all(noisy_action >= -0.1))

        trainer.cfg.domain_randomization_params["action_noise"]["apply_to_seed_actions"] = False
        self.assertTrue(torch.equal(trainer._apply_action_noise(action, seed_action=True), action))

    def test_observation_noise_is_domain_randomization_gated(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        obs = torch.zeros(8)

        trainer.cfg = SimpleNamespace(
            domain_randomization=False,
            domain_randomization_params={
                "observation_noise": {
                    "enabled": True,
                    "std": 10.0,
                    "clip_low": -0.1,
                    "clip_high": 0.1,
                }
            },
        )
        self.assertTrue(torch.equal(trainer._apply_observation_noise(obs), obs))

        trainer.cfg.domain_randomization = True
        torch.manual_seed(1)
        noisy_obs = trainer._apply_observation_noise(obs)
        self.assertFalse(torch.equal(noisy_obs, obs))
        self.assertTrue(torch.all(noisy_obs <= 0.1))
        self.assertTrue(torch.all(noisy_obs >= -0.1))

    def test_observation_noise_only_changes_graph_node_features(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(
            domain_randomization=True,
            domain_randomization_params={
                "observation_noise": {
                    "enabled": True,
                    "std": 10.0,
                    "clip_low": -0.1,
                    "clip_high": 0.1,
                }
            },
        )
        obs = Data(
            x=torch.zeros(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        )

        torch.manual_seed(1)
        noisy_obs = trainer._apply_observation_noise(obs)

        self.assertIsNot(noisy_obs, obs)
        self.assertFalse(torch.equal(noisy_obs.x, obs.x))
        self.assertTrue(torch.equal(noisy_obs.edge_index, obs.edge_index))
        self.assertTrue(torch.equal(obs.x, torch.zeros(3, 2)))
        self.assertTrue(torch.all(noisy_obs.x <= 0.1))
        self.assertTrue(torch.all(noisy_obs.x >= -0.1))

    def test_physical_parameters_are_loaded_from_config(self):
        cfg = graph_test_cfg(
            physical_parameters={
                "node_radius": 0.2,
                "box_size": [0.1, 0.2, 0.3],
                "connector_rod_length": None,
            }
        )

        params = _physical_parameters_from_config(cfg)

        self.assertIsInstance(params, TrussPhysicalParameters)
        self.assertEqual(params.node_radius, 0.2)
        self.assertEqual(params.box_size, [0.1, 0.2, 0.3])
        self.assertIsNone(params.connector_rod_length)

    def test_physical_parameters_can_be_disabled_for_mujoco_truss_gen_defaults(self):
        cfg = graph_test_cfg(
            physical_parameters_enabled=False,
            physical_parameters={
                "node_radius": 0.2,
            },
            domain_randomization=True,
            domain_randomization_params={
                "physical_parameters": {
                    "node_radius": {
                        "enabled": True,
                        "min": 0.3,
                        "max": 0.3,
                    },
                }
            },
        )

        self.assertIsNone(_physical_parameters_from_config(cfg))
        self.assertEqual(_randomized_physical_parameter_overrides(cfg, np.random.default_rng(1)), {})

    def test_physical_parameter_domain_randomization_can_be_toggled_per_field(self):
        cfg = graph_test_cfg(
            domain_randomization=True,
            physical_parameters={
                "node_radius": 0.1,
                "box_size": [0.05, 0.05, 0.1],
            },
            domain_randomization_params={
                "physical_parameters": {
                    "node_radius": {
                        "enabled": True,
                        "min": 0.2,
                        "max": 0.2,
                    },
                    "box_size": {
                        "enabled": False,
                        "min": [0.2, 0.2, 0.2],
                        "max": [0.2, 0.2, 0.2],
                    },
                }
            },
        )

        overrides = _randomized_physical_parameter_overrides(cfg, np.random.default_rng(1))
        params = _physical_parameters_from_config(cfg, overrides=overrides)

        self.assertEqual(overrides, {"node_radius": 0.2})
        self.assertEqual(params.node_radius, 0.2)
        self.assertEqual(params.box_size, [0.05, 0.05, 0.1])

        cfg.domain_randomization = False
        self.assertEqual(_randomized_physical_parameter_overrides(cfg, np.random.default_rng(1)), {})

    def test_all_upstream_runtime_randomization_ranges_are_configurable(self):
        upstream_range_fields = {
            name
            for name in DomainRandomizationConfig.__dataclass_fields__
            if name.endswith("_range")
        }
        self.assertEqual(
            set(_RUNTIME_DOMAIN_RANDOMIZATION_FIELDS.values()),
            upstream_range_fields,
        )

        yaml_cfg = OmegaConf.load(ROOT / "config" / "physics" / "domain_randomization.yaml")
        params = yaml_cfg.domain_randomization_params
        self.assertTrue(set(_RUNTIME_DOMAIN_RANDOMIZATION_FIELDS).issubset(params.keys()))

        configured_params = {
            "length_scale": {"enabled": False},
            **{
                name: {"enabled": True, "min": index + 0.25, "max": index + 0.75}
                for index, name in enumerate(_RUNTIME_DOMAIN_RANDOMIZATION_FIELDS)
            },
        }
        cfg = graph_test_cfg(
            domain_randomization=True,
            domain_randomization_params=configured_params,
        )
        randomization = _domain_randomization(cfg, "octahedron", False)

        for index, field_name in enumerate(_RUNTIME_DOMAIN_RANDOMIZATION_FIELDS.values()):
            self.assertEqual(getattr(randomization, field_name), (index + 0.25, index + 0.75))

    def test_eval_interval_crossing_with_batched_steps(self):
        self.assertFalse(OnlineTrainer._crossed_eval_interval(0, 3, 5))
        self.assertTrue(OnlineTrainer._crossed_eval_interval(3, 6, 5))
        self.assertFalse(OnlineTrainer._crossed_eval_interval(6, 9, 5))
        self.assertTrue(OnlineTrainer._crossed_eval_interval(9, 12, 5))

    def test_video_every_n_evals_records_first_then_every_tenth(self):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.cfg = SimpleNamespace(save_video=True, video_every_n_evals=10)
        trainer._step = 0
        trainer._eval_count = 0
        recorded = []
        trainer.eval = lambda: recorded.append(trainer._record_video_this_eval) or {
            "episode_reward": 0.0
        }
        trainer.common_metrics = lambda: {}
        trainer.logger = SimpleNamespace(log=lambda *args: None)
        trainer.report_eval_metrics = lambda *args: None

        for _ in range(21):
            trainer._evaluate_and_log()

        self.assertEqual(
            [index for index, should_record in enumerate(recorded) if should_record],
            [0, 10, 20],
        )
        self.assertEqual(trainer._eval_count, 21)

    def test_repeated_truss_envs_reset_and_step(self):
        cfg = flat_test_cfg(
            num_envs=4,
            max_steps=2,
            episode_length=2,
        )
        env = make_env(cfg)
        try:
            self.assertEqual(len(env.env.envs), 4)
            self.assertEqual(cfg.action_dim, 8)
            observations = env.reset_many(env_indices=range(cfg.num_envs))
            actions = [env.rand_act() for _ in range(cfg.num_envs)]
            results = env.step_many(actions, env_indices=range(cfg.num_envs))
            self.assertEqual(len(observations), cfg.num_envs)
            self.assertEqual(len(results), cfg.num_envs)
            for env_idx, (next_obs, reward, done, info) in enumerate(results):
                self.assertEqual(observations[env_idx].shape, next_obs.shape)
                self.assertEqual(info["task"], cfg.task)
                self.assertEqual(info["env_idx"], env_idx)
                self.assertTrue(float(reward) == float(reward))
            for env_idx in range(cfg.num_envs):
                obs = env.reset(task_idx=env_idx)
                action = env.rand_act()
                next_obs, reward, done, info = env.step(action)
                self.assertEqual(obs.shape, next_obs.shape)
                self.assertEqual(action.shape, (cfg.action_dim,))
                self.assertEqual(info["task"], cfg.task)
                self.assertEqual(info["env_idx"], env_idx)
                self.assertNotIn("task_idx", info)
                self.assertTrue(float(reward) == float(reward))
        finally:
            env.close()

    def test_graph_env_reset_and_node_action_step(self):
        cfg = graph_test_cfg()
        env = make_env(cfg)
        try:
            obs = env.reset()
            self.assertIsInstance(obs, Data)
            self.assertEqual(obs.x.shape[0], cfg.num_nodes)
            self.assertEqual(cfg.node_counts, [len(env.unwrapped.mj_model.control_node_names)])
            self.assertEqual(obs.edge_index.shape[0], 2)
            self.assertEqual(cfg.node_action_dim, 1)
            self.assertEqual(cfg.action_dim, cfg.node_action_dim)
            self.assertEqual(
                cfg.num_policy_actions,
                int(obs.action_mask.sum()) * cfg.node_action_dim,
            )
            self.assertEqual(cfg.num_actuators, env.unwrapped.mj_model.model.nu)

            action = env.rand_act()
            next_obs, reward, done, info = env.step(action)
            self.assertIsInstance(next_obs, Data)
            self.assertEqual(action.shape, (cfg.num_nodes, 1))
            self.assertEqual(env.unwrapped._node_action_to_actuator_action(action.numpy()).shape[0], env.unwrapped.mj_model.model.nu)
            self.assertTrue(float(reward) == float(reward))
            self.assertIn("terminated", info)
            self.assertIn("truncated", info)
        finally:
            env.close()

    def test_native_graph_features_flow_from_environment_to_policy(self):
        cfg = graph_test_cfg(
            graph_features={
                "node_roles": True,
                "edge_roles": True,
                "edge_distance": True,
            },
            max_steps=1,
            episode_length=1,
            nsubsteps=1,
        )
        env = make_env(cfg)
        try:
            obs = env.reset()
            self.assertEqual(obs.x.shape[1], 6)
            self.assertEqual(obs.edge_role.shape, (obs.edge_index.shape[1],))
            self.assertEqual(cfg.effective_node_feature_dim, 10)
            self.assertEqual(cfg.edge_feature_dim, 4)

            agent = GNNSAC(cfg)
            action = agent.act(obs, eval_mode=True)
            self.assertEqual(action.shape, (obs.num_nodes, 1))
            self.assertTrue(torch.isfinite(action).all())
        finally:
            env.close()

    def test_graph_env_scale_changes_generated_robot_size(self):
        base_env = make_env(
            graph_test_cfg(
                task="truss-graph",
                scale=1.0,
                max_steps=2,
                nsubsteps=1,
                domain_randomization=False,
            )
        )
        scaled_env = make_env(
            graph_test_cfg(
                task="truss-graph",
                scale=2.0,
                max_steps=2,
                nsubsteps=1,
                domain_randomization=False,
            )
        )
        try:
            base_dims = np.asarray(base_env.unwrapped.mj_model.initial_bounding_box_dimensions)
            scaled_dims = np.asarray(scaled_env.unwrapped.mj_model.initial_bounding_box_dimensions)
            np.testing.assert_allclose(scaled_dims, base_dims * 2.0, rtol=1e-5, atol=1e-6)
        finally:
            base_env.close()
            scaled_env.close()

    def test_control_graph_mode_uses_same_graph_for_simple_and_realistic(self):
        env_stats = []
        for topology in ["octahedron", "octahedron:realistic"]:
            cfg = graph_test_cfg(
                task="truss-graph",
                truss_topology=topology,
                use_control_graph=True,
                max_steps=2,
                nsubsteps=1,
                domain_randomization=False,
            )
            env = make_env(cfg)
            try:
                obs = env.reset()
                self.assertIsInstance(obs, Data)
                self.assertEqual(obs.edge_index.shape[0], 2)
                self.assertEqual(obs.x.shape[0], env.action_space.shape[0])
                self.assertTrue(torch.equal(
                    obs.action_mask,
                    torch.as_tensor(
                        ~np.asarray(
                            env.unwrapped.node_velocity_controller.passive_node_mask,
                            dtype=bool,
                        )
                    ),
                ))
                self.assertEqual(env.action_space.shape[1], cfg.node_action_dim)
                self.assertEqual(
                    cfg.num_policy_actions,
                    int(obs.action_mask.sum()) * cfg.node_action_dim,
                )

                action = env.rand_act()
                next_obs, reward, done, info = env.step(action)
                self.assertIsInstance(next_obs, Data)
                self.assertEqual(action.shape, env.action_space.shape)
                self.assertTrue(float(reward) == float(reward))
                self.assertIn("terminated", info)
                self.assertIn("truncated", info)
                env_stats.append((obs.x.shape[0], tuple(env.action_space.shape), tuple(obs.edge_index.shape)))
            finally:
                env.close()

        self.assertEqual(env_stats[0], env_stats[1])

    def test_control_graph_mode_uses_node_velocity_controller(self):
        cfg = graph_test_cfg(
            task="truss-graph",
            truss_topology="octahedron",
            use_control_graph=True,
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
        )
        env = make_env(cfg)
        try:
            env.reset()
            action = env.rand_act()
            normalized_node_action, ctrl = env.unwrapped._control_graph_node_action_to_actuator_ctrl(action.numpy())
            self.assertEqual(normalized_node_action.shape[0], env.action_space.shape[0])
            self.assertEqual(ctrl.shape[0], len(env.unwrapped.mj_model.external_actuator_ids))

            def fail_legacy_node_sum(_action):
                raise AssertionError("legacy node-action summation should not be used")

            env.unwrapped._node_action_to_actuator_action = fail_legacy_node_sum
            next_obs, reward, done, info = env.step(action)
            self.assertIsInstance(next_obs, Data)
            self.assertEqual(next_obs.rigidity.shape, (1,))
            self.assertTrue(torch.isfinite(next_obs.rigidity).all())
            self.assertGreaterEqual(float(next_obs.rigidity.item()), 0.0)
            self.assertAlmostEqual(
                float(next_obs.rigidity.item()),
                float(info["critical_eig"]),
                places=5,
            )
            self.assertTrue(float(reward) == float(reward))
            self.assertIn("terminated", info)
            self.assertIn("truncated", info)
        finally:
            env.close()

    def test_graph_rigidity_reward_uses_first_non_rigid_eigenvalue(self):
        cfg = graph_test_cfg(
            rigidity_weight=2.5,
            forward_weight=0.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            slip_weight=0.0,
            critical_eig_threshold=0.0,
        )
        env = make_env(cfg)
        try:
            unwrapped = env.unwrapped
            self.assertFalse(getattr(unwrapped.mj_model, "wcrm", True))

            unwrapped._initial_critical_eig = 0.5
            unwrapped.mj_model._critical_eig = lambda: 0.25
            unwrapped.mj_model.collapse_check = lambda: 99.0
            unwrapped.mj_model.get_forward_velocity = lambda: 0.0
            unwrapped.mj_model.get_slip_penalty = lambda height: 0.0

            action = np.zeros(unwrapped.mj_model.model.nu, dtype=np.float32)
            reward, info, terminated = unwrapped._compute_reward(action)

            self.assertFalse(terminated)
            self.assertEqual(info["critical_eig"], 0.5)
            self.assertEqual(info["critical_eig_raw"], 0.25)
            self.assertAlmostEqual(info["rigidity"], 2.5 * 0.5)
            self.assertAlmostEqual(reward, 2.5 * 0.5)
        finally:
            env.close()

    def test_graph_collapse_threshold_uses_normalized_rigidity(self):
        cfg = graph_test_cfg(
            rigidity_weight=0.0,
            forward_weight=0.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            slip_weight=0.0,
            critical_eig_threshold=0.6,
        )
        env = make_env(cfg)
        try:
            unwrapped = env.unwrapped
            unwrapped._initial_critical_eig = 0.5
            unwrapped.mj_model._critical_eig = lambda: 0.25
            unwrapped.mj_model.get_forward_velocity = lambda: 0.0
            unwrapped.mj_model.get_slip_penalty = lambda height: 0.0

            action = np.zeros(unwrapped.mj_model.model.nu, dtype=np.float32)
            _, info, terminated = unwrapped._compute_reward(action)

            self.assertTrue(terminated)
            self.assertEqual(info["critical_eig"], 0.5)
            self.assertEqual(info["critical_eig_raw"], 0.25)
            self.assertTrue(info["terminated_by_collapse"])
        finally:
            env.close()

    def test_unified_graph_env_supports_representative_mujoco_truss_gen_presets(self):
        # PRESETS now contains hundreds of enumerated Henneberg variants. Their
        # exhaustive generation belongs to mujoco-truss-gen's own test suite;
        # cover every preset family at this integration boundary.
        representative_presets = [
            "octahedron",
            "tetrahedron",
            "icosahedron",
            "solar_array",
            "henneberg_n5_1tube",
            "usevitch_1514879",
        ]
        self.assertTrue(set(representative_presets).issubset(PRESETS))
        for topology in representative_presets:
            with self.subTest(topology=topology):
                cfg = graph_test_cfg(
                    task="truss-graph",
                    truss_topology=topology,
                    max_steps=2,
                    nsubsteps=1,
                    domain_randomization=False,
                )
                env = make_env(cfg)
                try:
                    obs = env.reset()
                    self.assertIsInstance(obs, Data)
                    self.assertGreater(obs.num_nodes, 0)
                    self.assertEqual(obs.x.shape[1], 6)
                    self.assertEqual(obs.edge_index.shape[0], 2)

                    action = env.rand_act()
                    next_obs, reward, done, info = env.step(action)
                    self.assertIsInstance(next_obs, Data)
                    self.assertEqual(action.shape, (obs.num_nodes, 1))
                    self.assertTrue(float(reward) == float(reward))
                    self.assertIn("terminated", info)
                    self.assertIn("truncated", info)
                finally:
                    env.close()

    def test_graph_multitask_octahedron_and_tetrahedron_step(self):
        cfg = graph_test_cfg(
            multitask=True,
            tasks=["octahedron-graph-right", "tetrehedron-graph-right"],
            num_envs=1,
        )
        env = make_env(cfg)
        try:
            self.assertEqual(env.num_envs, 2)
            observations = env.reset_many(env_indices=[0, 1])
            control_node_counts = [
                len(component.unwrapped.mj_model.control_node_names)
                for component in env.env.envs
            ]
            self.assertEqual(control_node_counts, [12, 8])
            self.assertEqual(cfg.node_counts, control_node_counts)
            self.assertEqual([obs.num_nodes for obs in observations], control_node_counts)

            actions = [env.rand_act(env_idx=0), env.rand_act(env_idx=1)]
            self.assertEqual(
                [tuple(action.shape) for action in actions],
                [(node_count, 1) for node_count in control_node_counts],
            )

            results = env.step_many(actions, env_indices=[0, 1])
            self.assertEqual([result[0].num_nodes for result in results], control_node_counts)
            self.assertEqual([result[3]["task"] for result in results], cfg.tasks)
            self.assertEqual([result[3]["env_idx"] for result in results], [0, 1])
            self.assertEqual([result[3]["task_idx"] for result in results], [0, 1])
        finally:
            env.close()

    def test_graph_topology_list_builds_multitask_envs(self):
        cfg = graph_test_cfg(
            task="truss-graph",
            truss_topologies=["octahedron", "tetrahedron"],
            multitask=False,
            num_envs=1,
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
        )
        env = make_env(cfg)
        try:
            self.assertEqual(env.num_envs, 2)
            self.assertEqual(cfg.tasks, ["truss-graph:octahedron", "truss-graph:tetrahedron"])

            observations = env.reset_many(env_indices=[0, 1])
            control_node_counts = [
                len(component.unwrapped.mj_model.control_node_names)
                for component in env.env.envs
            ]
            self.assertEqual(control_node_counts, [12, 8])
            self.assertEqual(cfg.node_counts, control_node_counts)
            self.assertEqual([obs.num_nodes for obs in observations], control_node_counts)

            actions = [env.rand_act(env_idx=0), env.rand_act(env_idx=1)]
            self.assertEqual(
                [tuple(action.shape) for action in actions],
                [(node_count, 1) for node_count in control_node_counts],
            )

            results = env.step_many(actions, env_indices=[0, 1])
            self.assertEqual([result[0].num_nodes for result in results], control_node_counts)
            self.assertEqual([result[3]["task"] for result in results], cfg.tasks)
        finally:
            env.close()

    def test_graph_topology_list_allows_variable_edge_role_spaces(self):
        cfg = graph_test_cfg(
            task="truss-graph",
            truss_topologies=["tetrahedron", "octahedron", "henneberg_n6_1tube_2"],
            multitask=False,
            num_envs=1,
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
            graph_features={"edge_roles": True},
        )
        env = make_env(cfg)
        try:
            observations = env.reset_many(env_indices=[0, 1, 2])
            self.assertEqual([obs.num_nodes for obs in observations], [8, 12, 13])
            for obs in observations:
                self.assertEqual(obs.edge_role.shape, (obs.edge_index.shape[1],))
        finally:
            env.close()

    def test_graph_topology_list_accepts_realistic_variant_suffix(self):
        cfg = graph_test_cfg(
            task="truss-graph",
            truss_topologies=["octahedron", "octahedron:realistic", "solar_array"],
            multitask=False,
            num_envs=1,
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
        )
        env = make_env(cfg)
        try:
            self.assertEqual(
                cfg.tasks,
                [
                    "truss-graph:octahedron",
                    "truss-graph:octahedron:realistic",
                    "truss-graph:solar_array",
                ],
            )
            observations = env.reset_many(env_indices=[0, 1, 2])
            self.assertEqual([obs.x.shape[1] for obs in observations], [6, 6, 6])

            actions = [env.rand_act(env_idx=idx) for idx in range(3)]
            results = env.step_many(actions, env_indices=[0, 1, 2])
            self.assertEqual([result[3]["task"] for result in results], cfg.tasks)
            self.assertTrue(all(float(result[1]) == float(result[1]) for result in results))
        finally:
            env.close()

    def test_unified_mlp_env_single_topology_reset_and_step(self):
        cfg = flat_test_cfg(
            task="truss-mlp",
            truss_topology="octahedron",
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
        )
        env = make_env(cfg)
        try:
            obs = env.reset()
            action = env.rand_act()
            next_obs, reward, done, info = env.step(action)

            self.assertEqual(obs.shape, next_obs.shape)
            self.assertEqual(action.shape, (cfg.action_dim,))
            self.assertTrue(float(reward) == float(reward))
            self.assertIn("terminated", info)
            self.assertIn("truncated", info)
        finally:
            env.close()

    def test_unified_mlp_env_rejects_mismatched_topology_list(self):
        cfg = flat_test_cfg(
            task="truss-mlp",
            truss_topologies=["octahedron", "tetrahedron"],
            multitask=False,
            num_envs=1,
            max_steps=2,
            nsubsteps=1,
            domain_randomization=False,
        )

        with self.assertRaisesRegex(ValueError, "observation space|action space"):
            make_env(cfg)

    def test_multitask_eval_records_each_task_video_key(self):
        class DummyEnv:
            num_envs = 2

            def __init__(self):
                self.active_env_idx = None
                self.reset_task_indices = []

            def reset(self, task_idx=None):
                self.active_env_idx = task_idx
                self.reset_task_indices.append(task_idx)
                return torch.tensor([float(task_idx)])

            def step(self, action):
                info = {
                    "success": float(self.active_env_idx),
                    "terminated": torch.tensor(0.0),
                    "truncated": torch.tensor(1.0),
                }
                reward = torch.tensor(float(self.active_env_idx + 1))
                return torch.tensor([float(self.active_env_idx)]), reward, True, info

            def render(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        class DummyAgent:
            model = "dummy"

            def act(self, obs, t0=False, eval_mode=False):
                return torch.tensor([0.0])

        class DummyVideo:
            def __init__(self):
                self.saved_keys = []

            def init(self, env, enabled=True):
                return

            def record(self, env):
                return

            def save(self, step, key="videos/eval_video"):
                self.saved_keys.append(key)

        class DummyLogger:
            def __init__(self):
                self.video = DummyVideo()

        cfg = SimpleNamespace(
            eval_episodes=1,
            save_video=True,
            multitask=True,
            tasks=["octahedron-graph-right", "tetrehedron-graph-right"],
            resume_from_checkpoint=None,
            work_dir=str(ROOT / "logs" / "test-smoke"),
        )
        env = DummyEnv()
        logger = DummyLogger()
        trainer = OnlineTrainer(cfg=cfg, env=env, agent=DummyAgent(), buffer=None, logger=logger)

        metrics = trainer.eval()

        self.assertEqual(env.reset_task_indices, [0, 1])
        self.assertEqual(
            logger.video.saved_keys,
            [
                "videos/eval_video/octahedron-graph-right",
                "videos/eval_video/tetrehedron-graph-right",
            ],
        )
        self.assertEqual(metrics["octahedron-graph-right_episode_reward"], 1.0)
        self.assertEqual(metrics["tetrehedron-graph-right_episode_reward"], 2.0)
        self.assertEqual(metrics["episode_reward"], 1.5)

    def test_gnn_sac_update_smoke(self):
        cfg = graph_test_cfg(message_attention=True)
        env = make_env(cfg)
        try:
            agent = GNNSAC(cfg)
            buffer = GNNBuffer(cfg)
            self.assertEqual(agent.target_entropy, -float(cfg.num_policy_actions))

            obs0 = env.reset()
            action0 = agent.act(obs0)
            obs1, reward1, _, info1 = env.step(action0)
            action1 = env.rand_act()
            obs2, reward2, _, info2 = env.step(action1)

            episode = [
                {"obs": obs0, "action": action0, "reward": reward1, "terminated": info1["terminated"]},
                {"obs": obs1, "action": action1, "reward": reward2, "terminated": info2["terminated"]},
                {"obs": obs2, "action": action1, "reward": reward2, "terminated": info2["terminated"]},
            ]
            buffer.add(episode)
            obs, action, reward, terminated, next_obs = buffer.sample()

            self.assertEqual(action.shape[1], 1)
            self.assertEqual(reward.shape[0], cfg.batch_size)
            self.assertEqual(terminated.shape[0], cfg.batch_size)
            self.assertEqual(obs.x.shape[0], next_obs.x.shape[0])

            update_info = agent.update(buffer)
            self.assertIn("value_loss", update_info)
            self.assertIn("pi_loss", update_info)
            self.assertIn("alpha", update_info)
            self.assertIsNotNone(agent.model._pi.attention_score.weight.grad)
            for critic in agent.model._Qs.modules_list:
                self.assertIsNotNone(critic.attention_score.weight.grad)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
