import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.cross_validation import resolve_cross_validation, validate_cross_validation_spec
from scripts.launch_cross_validation import build_launch, load_definition, main, ordered_folds


def spec(**overrides):
    value = {
        "enabled": True,
        "name": "node_groups",
        "groups": {
            "small": ["tetrahedron"],
            "medium": ["octahedron", "solar_array"],
            "large": ["henneberg_n6_1tube_2"],
        },
        "final_test": ["henneberg_n7_1tube_1"],
        "held_out_group": None,
    }
    value.update(overrides)
    return value


def training_cfg(cross_validation):
    return OmegaConf.create(
        {
            "work_dir": "/tmp/cross-validation-test",
            "task": "truss-graph",
            "exp_name": "test",
            "seed": 1,
            "eval_backend": "mujoco",
            "topologies": None,
            "truss_topologies": None,
            "eval_extra_topologies": None,
            "cross_validation": cross_validation,
        }
    )


class CrossValidationResolutionTest(unittest.TestCase):
    def test_resolves_training_and_heldout_topologies_without_touching_final_test(self):
        cfg = training_cfg(spec(held_out_group="medium"))

        resolve_cross_validation(cfg)

        self.assertEqual(
            list(cfg.truss_topologies),
            ["tetrahedron", "henneberg_n6_1tube_2"],
        )
        self.assertEqual(list(cfg.eval_extra_topologies), ["octahedron", "solar_array"])
        self.assertEqual(list(cfg.cross_validation.final_test), ["henneberg_n7_1tube_1"])
        self.assertEqual(list(cfg.cross_validation.training_groups), ["small", "large"])
        self.assertEqual(cfg.cross_validation.fold_index, 1)
        self.assertEqual(cfg.cross_validation.fold_name, "holdout_medium")

    def test_rejects_duplicate_development_topologies(self):
        value = spec(groups={"first": ["octahedron"], "second": ["octahedron"]})
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_cross_validation_spec(value)

    def test_rejects_final_test_overlap(self):
        value = spec(final_test=["tetrahedron"])
        with self.assertRaisesRegex(ValueError, "also appears"):
            validate_cross_validation_spec(value)

    def test_rejects_missing_or_unknown_fold(self):
        with self.assertRaisesRegex(ValueError, "must be selected"):
            resolve_cross_validation(training_cfg(spec()))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_cross_validation(training_cfg(spec(held_out_group="missing")))

    def test_rejects_non_native_evaluation_backend(self):
        cfg = training_cfg(spec(held_out_group="small"))
        cfg.eval_backend = "mjx"
        with self.assertRaisesRegex(ValueError, "eval_backend=mujoco"):
            resolve_cross_validation(cfg)

    def test_rejects_manual_topology_overrides(self):
        cfg = training_cfg(spec(held_out_group="small"))
        cfg.truss_topologies = ["octahedron"]
        with self.assertRaisesRegex(ValueError, "owns truss_topologies"):
            resolve_cross_validation(cfg)


class CrossValidationLauncherTest(unittest.TestCase):
    def setUp(self):
        self.spec = validate_cross_validation_spec(spec())

    def test_builds_complete_fold_seed_matrix_and_forwards_overrides(self):
        command, jobs = build_launch(
            config_name="node_groups",
            spec=self.spec,
            seeds=[3, 7],
            shuffle_seed=11,
            overrides=["platform=supercomputer", "steps=1000"],
            python_executable="python-test",
        )

        self.assertEqual(len(jobs), 6)
        self.assertEqual(
            {(job["held_out_group"], job["seed"]) for job in jobs},
            {(group, seed) for group in self.spec["groups"] for seed in (3, 7)},
        )
        self.assertIn("platform=supercomputer", command)
        self.assertIn("steps=1000", command)
        self.assertIn("seed=3,7", command)
        self.assertTrue(any(arg.startswith("cross_validation.held_out_group=") for arg in command))
        self.assertNotIn("henneberg_n7_1tube_1", json.dumps(jobs))

    def test_shuffle_is_reproducible_and_changes_only_order(self):
        first = ordered_folds(self.spec, 42)
        second = ordered_folds(self.spec, 42)
        other = ordered_folds(self.spec, 7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(other))
        self.assertNotEqual(first, other)

    def test_rejects_launcher_owned_overrides(self):
        with self.assertRaisesRegex(ValueError, "owned"):
            build_launch(
                config_name="node_groups",
                spec=self.spec,
                seeds=[1],
                shuffle_seed=0,
                overrides=["seed=9"],
            )

    def test_dry_run_writes_manifest_without_launching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.json"
            with patch("scripts.launch_cross_validation.subprocess.run") as run:
                result = main(
                    [
                        "cross_validation=smoke",
                        "--seeds",
                        "1,2",
                        "--shuffle-seed",
                        "5",
                        "--manifest",
                        str(manifest),
                        "--dry-run",
                        "platform=local",
                    ]
                )

            self.assertEqual(result, 0)
            run.assert_not_called()
            payload = json.loads(manifest.read_text())
            self.assertEqual(len(payload["jobs"]), 4)
            self.assertEqual(payload["seeds"], [1, 2])
            self.assertEqual(payload["final_test"], [])
            self.assertIn("platform=local", payload["command"])

    def test_loads_smoke_definition(self):
        loaded = load_definition("smoke", ROOT / "config")
        self.assertEqual(list(loaded["groups"]), ["octahedron_group", "tetrahedron_group"])


if __name__ == "__main__":
    unittest.main()
