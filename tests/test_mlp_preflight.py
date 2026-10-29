from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import gymnasium as gym
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "sac"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.parser import parse_cfg
from common.training_validation import (
    configured_topologies,
    padded_environment_contract,
    validate_padded_mlp_environment,
    validate_training_config,
)


def config(**changes):
    values = dict(sac_backend="padded_mlp", task="truss-graph", tasks=["truss-graph"],
                  truss_topology="tetrahedron", truss_topologies=None,
                  use_graph_observations=True, use_control_graph=True,
                  use_virtual_node=False, graph_features={}, padded_mlp_max_nodes=5,
                  obs_dim=6, node_action_dim=1, domain_randomization=False,
                  mujoco_backend="mujoco", eval_extra_topologies=None)
    values.update(changes)
    return SimpleNamespace(**values)


def environment(nodes=4, features=6, mask=None, names=None):
    env = SimpleNamespace(
        observation_space=gym.spaces.Dict({
            "x": gym.spaces.Box(-np.inf, np.inf, shape=(nodes, features)),
            "action_mask": gym.spaces.MultiBinary(nodes),
            "rigidity": gym.spaces.Box(0, np.inf, shape=(1,)),
        }),
        action_space=gym.spaces.Box(-1, 1, shape=(nodes, 1)),
        graph_node_names=names or [f"node-{index}" for index in range(nodes)],
        topology=f"topology-{nodes}",
        _base_passive_node_mask=~np.asarray(mask if mask is not None else [True] * nodes),
        reset=mock.Mock(side_effect=AssertionError("preflight must not collect observations")),
        step=mock.Mock(side_effect=AssertionError("preflight must not collect transitions")),
        close=mock.Mock(),
    )
    return env


class ConfigPreflightTest(unittest.TestCase):
    def test_flat_graph_and_heterogeneous_configs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires flat"):
            validate_training_config(config(sac_backend="mlp"))
        flat = config(sac_backend="mlp", task="truss-mlp", tasks=["truss-mlp"],
                      use_graph_observations=False, truss_topologies=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Heterogeneous flat"):
            validate_training_config(flat)
        flat.truss_topologies = ["a"]
        validate_training_config(flat)

    def test_mjx_rejects_only_model_rebuilding_randomization(self):
        cfg = config(mujoco_backend="mjx", domain_randomization=True,
                     domain_randomization_params={"length_scale": {"enabled": False},
                         "body_mass_multiplier": {"enabled": True},
                         "physical_parameters": {"hinge_position_kp": {"enabled": True}}})
        # Legacy hinge kp is runtime-capable, not a model-rebuilding parameter.
        validate_training_config(cfg)
        cfg.domain_randomization_params["physical_parameters"]["node_mass"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "fixed-shape"):
            validate_training_config(cfg)
        cfg.physical_parameters_enabled = False
        validate_training_config(cfg)
        cfg.domain_randomization_params["length_scale"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "fixed-shape"):
            validate_training_config(cfg)
        cfg.mujoco_backend = "mujoco"
        validate_training_config(cfg)
        cfg.mujoco_backend = "mjx"
        cfg.domain_randomization = False
        validate_training_config(cfg)

    def test_default_profile_is_fixed_shape(self):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            cfg = compose(config_name="config", overrides=["sac_backend=padded_mlp", "sim_backend=mjx"])
        cfg.work_dir = "/tmp/preflight-tests"
        validate_training_config(parse_cfg(cfg))

    def test_padded_augmentation_and_control_errors_are_early(self):
        for changes in ({"use_control_graph": False}, {"use_virtual_node": True},
                        {"graph_features": {"node_roles": True}}, {"padded_mlp_max_nodes": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_training_config(config(**changes))

    def test_resolved_topologies_include_heldout_without_changing_training(self):
        cfg = config(truss_topologies=["a", "b", "a"], eval_extra_topologies=["b", "c"])
        self.assertEqual(configured_topologies(cfg), ["a", "b"])
        self.assertEqual(configured_topologies(cfg, include_evaluation=True), ["a", "b", "c"])
        self.assertEqual(cfg.truss_topologies, ["a", "b", "a"])

    def test_named_presets_require_user_morphologies(self):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            specialist = compose(config_name="config", overrides=[
                "+experiment=padded_mlp_specialist", "truss_topology=tetrahedron"])
            generalist = compose(config_name="config", overrides=[
                "+experiment=padded_mlp_generalist", "truss_topologies=[tetrahedron,octahedron]"])
            missing_generalist = compose(config_name="config", overrides=["+experiment=padded_mlp_generalist"])
        for item in (specialist, generalist, missing_generalist):
            item.work_dir = "/tmp/preflight-tests"
        self.assertEqual(specialist.sac_backend, "padded_mlp")
        validate_training_config(parse_cfg(specialist))
        validate_training_config(parse_cfg(generalist))
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_training_config(parse_cfg(missing_generalist))

    def test_invalid_simulator_fails_before_environment_and_teacher_access(self):
        from train import run_training

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            cfg = compose(config_name="config", overrides=["sac_backend=padded_mlp", "sim_backend=mjx",
                "distillation=kl", "domain_randomization_params.length_scale.enabled=true",
                "+distillation.teachers={tetrahedron:/must-not-open.pt}"])
        cfg.work_dir = "/tmp/preflight-tests"
        with mock.patch("train.make_env") as make, mock.patch("pathlib.Path.open") as path_open:
            with self.assertRaisesRegex(ValueError, "fixed-shape"):
                run_training(cfg)
        make.assert_not_called()
        path_open.assert_not_called()

    def test_eval_task_capacity_fails_before_opening_teacher_checkpoints(self):
        from train import run_training

        for eval_task in ("truss-graph:octahedron", "octahedron-graph-right"):
            with self.subTest(eval_task=eval_task):
                with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
                    cfg = compose(config_name="config", overrides=[
                        "sac_backend=padded_mlp", "truss_topology=tetrahedron",
                        "padded_mlp_max_nodes=8", f"eval_task={eval_task}",
                        "distillation=kl", "domain_randomization=false",
                        "enable_wandb=false", "save_video=false",
                        "work_dir=/tmp/preflight-eval-task-tests",
                        "+distillation.teachers={tetrahedron:/must-not-open.pt}",
                    ])
                with mock.patch("common.distillation.Path.open") as teacher_open:
                    with self.assertRaisesRegex(ValueError, "evaluation task.*control nodes"):
                        run_training(cfg)
                teacher_open.assert_not_called()

    def test_valid_eval_task_is_inspected_without_collecting_data(self):
        from env import make_env

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            cfg = parse_cfg(compose(config_name="config", overrides=[
                "sac_backend=padded_mlp", "truss_topology=tetrahedron",
                "padded_mlp_max_nodes=12", "eval_task=truss-graph:octahedron",
                "domain_randomization=false", "work_dir=/tmp/preflight-eval-task-tests",
            ]))
        train_env = make_env(cfg)
        try:
            with mock.patch.object(train_env, "reset", side_effect=AssertionError("reset training")):
                records = validate_padded_mlp_environment(cfg, train_env)
            self.assertEqual([record["control_nodes"] for record in records], [8, 12])
            self.assertEqual(cfg.truss_topology, "tetrahedron")
        finally:
            train_env.close()


class EnvironmentContractTest(unittest.TestCase):
    def test_every_heterogeneous_training_slot_is_checked(self):
        first, too_large = environment(), environment(nodes=6)
        wrapped = SimpleNamespace(env=SimpleNamespace(envs=[first, too_large]))
        with self.assertRaisesRegex(ValueError, "6 control nodes"):
            validate_padded_mlp_environment(config(), wrapped)
        first.reset.assert_not_called()
        first.step.assert_not_called()

    def test_contract_requires_mask_names_features_and_rigidity(self):
        for env in (environment(features=7), environment(mask=[False] * 4),
                    environment(names=["same"] * 4)):
            with self.subTest(env=env), self.assertRaises(ValueError):
                padded_environment_contract(config(), env)
        valid = padded_environment_contract(config(), environment(mask=[True, False, True, False]))
        self.assertEqual(valid["active_actions"], 2)
        self.assertEqual(valid["action_node_names"], ["node-0", "node-2"])

    def test_heldout_capacity_is_checked_without_reset_or_mutating_training(self):
        cfg = config(eval_extra_topologies=["heldout"])
        saved = deepcopy(cfg)
        first, heldout = environment(), environment(nodes=6)
        with mock.patch("env.make_env", return_value=heldout) as factory:
            with self.assertRaisesRegex(ValueError, "evaluation heldout"):
                validate_padded_mlp_environment(cfg, first)
        self.assertEqual(vars(cfg), vars(saved))
        heldout.close.assert_called_once()
        first.reset.assert_not_called()
        heldout.reset.assert_not_called()
        self.assertFalse(factory.call_args.args[0].domain_randomization)
        self.assertEqual(factory.call_args.args[0].mujoco_backend, "mujoco")


if __name__ == "__main__":
    unittest.main()
