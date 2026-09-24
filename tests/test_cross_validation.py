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

from common.cross_validation import (
    random_partition,
    resolve_cross_validation,
    validate_cross_validation_spec,
)
from scripts.launch_cross_validation import build_launch, load_definition, main, ordered_folds
from scripts.make_random_cross_validation import build_definition, render_definition


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
            python_command=["python-test"],
        )

        self.assertEqual(len(jobs), 6)
        self.assertEqual(command[0], "python-test")
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

    def test_defaults_to_uv_managed_python(self):
        command, _ = build_launch(
            config_name="node_groups",
            spec=self.spec,
            seeds=[1],
            shuffle_seed=0,
            overrides=[],
        )
        self.assertEqual(command[:3], ["uv", "run", "python"])
        self.assertTrue(command[3].endswith("sac/gnn_train.py"))

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


class RandomPartitionTest(unittest.TestCase):
    topologies = [f"topology_{index}" for index in range(10)]

    def test_same_seed_is_reproducible_and_order_independent(self):
        first = random_partition(self.topologies, 5, seed=3)
        second = random_partition(list(reversed(self.topologies)), 5, seed=3)
        other = random_partition(self.topologies, 5, seed=4)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(list(first), [f"fold_{index}" for index in range(5)])

    def test_folds_are_disjoint_complete_and_balanced(self):
        for num_folds in (2, 3, 4, 7, 10):
            folds = random_partition(self.topologies, num_folds, seed=0)
            members = [topology for fold in folds.values() for topology in fold]
            sizes = [len(fold) for fold in folds.values()]

            self.assertEqual(sorted(members), sorted(self.topologies))
            self.assertEqual(len(members), len(set(members)))
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_rejects_invalid_fold_counts_and_duplicates(self):
        for num_folds in (1, 11):
            with self.assertRaisesRegex(ValueError, "num_folds"):
                random_partition(self.topologies, num_folds, seed=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            random_partition(["octahedron", "octahedron", "tetrahedron"], 2, seed=0)

    def test_committed_random_5fold_matches_generator(self):
        path = ROOT / "config" / "cross_validation" / "random_5fold.yaml"
        split = OmegaConf.load(path).cross_validation.split
        regenerated = build_definition(
            source=split.source,
            num_folds=split.num_folds,
            split_seed=split.seed,
            name="random_5fold",
        )
        loaded = load_definition("random_5fold", ROOT / "config")
        source = load_definition(split.source, ROOT / "config")
        source_pool = {
            topology for topologies in source["groups"].values() for topology in topologies
        }
        members = [topology for fold in loaded["groups"].values() for topology in fold]

        self.assertEqual(path.read_text(), render_definition(regenerated))
        self.assertEqual(loaded["groups"], regenerated["groups"])
        self.assertEqual(set(members), source_pool)
        self.assertEqual(loaded["final_test"], source["final_test"])
        self.assertFalse(set(members) & set(loaded["final_test"]))

    def test_random_5fold_launch_matrix(self):
        loaded = load_definition("random_5fold", ROOT / "config")
        _, jobs = build_launch(
            config_name="random_5fold",
            spec=loaded,
            seeds=[1, 2],
            shuffle_seed=17,
            overrides=[],
            python_command=["python-test"],
        )

        self.assertEqual(len(jobs), 10)
        for job in jobs:
            self.assertEqual(len(job["heldout_topologies"]), 2)
            self.assertEqual(len(job["training_topologies"]), 8)
            self.assertFalse(set(job["heldout_topologies"]) & set(job["training_topologies"]))
            self.assertFalse(set(job["training_topologies"]) & set(loaded["final_test"]))


if __name__ == "__main__":
    unittest.main()
