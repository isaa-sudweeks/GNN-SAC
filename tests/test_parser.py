import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.parser import (
    LAUNCH_COMMAND_ENV,
    capture_launch_command,
    multirun_id,
    multirun_work_dir,
    parse_cfg,
)


def hydra_job(job_num: int, override_dirname: str):
    return OmegaConf.create(
        {
            "job": {
                "num": job_num,
                "override_dirname": override_dirname,
            }
        }
    )


def training_cfg(work_dir: Path):
    return OmegaConf.create(
        {
            "work_dir": str(work_dir),
            "task": "truss-graph",
            "exp_name": "test",
            "seed": 1,
            "isolate_multirun_runs": True,
        }
    )


def topology_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "work_dir": "/tmp/test",
            "task": "truss-graph",
            "exp_name": "test",
            "seed": 1,
            "topologies": None,
            "truss_topologies": None,
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


class TopologyAliasTest(unittest.TestCase):
    def test_topologies_alias_populates_truss_topologies(self):
        cfg = parse_cfg(topology_cfg(topologies=["octahedron", "tetrahedron"]))

        self.assertEqual(cfg.truss_topologies, ["octahedron", "tetrahedron"])

    def test_matching_topology_aliases_are_allowed(self):
        cfg = parse_cfg(
            topology_cfg(
                topologies=["octahedron", "tetrahedron"],
                truss_topologies=["octahedron", "tetrahedron"],
            )
        )

        self.assertEqual(cfg.truss_topologies, ["octahedron", "tetrahedron"])

    def test_conflicting_topology_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Use either topologies or truss_topologies"):
            parse_cfg(
                topology_cfg(
                    topologies=["octahedron"],
                    truss_topologies=["tetrahedron"],
                )
            )


class RunMetadataTest(unittest.TestCase):
    def test_defaults_wandb_storage_to_project_root(self):
        cfg = parse_cfg(topology_cfg())

        self.assertEqual(cfg.wandb_dir, ROOT)

    def test_records_shell_safe_original_launch_command(self):
        argv = [".venv/bin/python", "sac/gnn_train.py", "exp_name=my run", "steps=1000"]
        with patch.dict(os.environ, {}, clear=True), patch("common.parser.sys.orig_argv", argv):
            cfg = parse_cfg(topology_cfg())

        self.assertEqual(
            cfg.launch_command,
            ".venv/bin/python sac/gnn_train.py 'exp_name=my run' steps=1000",
        )

    def test_captured_sweep_command_survives_submitit_worker_argv(self):
        sweep_argv = [
            ".venv/bin/python",
            "sac/gnn_train.py",
            "-m",
            "platform=supercomputer",
            "seed=1,2,3",
            "exp_name=everything",
        ]
        worker_argv = [
            "/home/isuds/robotics/GNN-SAC/.venv/bin/python",
            "-u",
            "-m",
            "submitit.core._submit",
            "/runs/hydra/.submitit/%j",
        ]
        with patch.dict(os.environ, {}, clear=True):
            expected = capture_launch_command(sweep_argv)
            with patch("common.parser.sys.orig_argv", worker_argv):
                cfg = parse_cfg(topology_cfg())

        self.assertEqual(
            expected,
            ".venv/bin/python sac/gnn_train.py -m platform=supercomputer seed=1,2,3 "
            "exp_name=everything",
        )
        self.assertEqual(cfg.launch_command, expected)

    def test_submitit_import_does_not_overwrite_captured_sweep_command(self):
        sweep_command = ".venv/bin/python sac/gnn_train.py -m seed=1,2,3"
        worker_argv = [
            ".venv/bin/python",
            "-m",
            "submitit.core._submit",
            "/runs/.submitit/%j",
        ]
        with patch.dict(os.environ, {LAUNCH_COMMAND_ENV: sweep_command}, clear=True):
            captured = capture_launch_command(worker_argv)

        self.assertEqual(captured, sweep_command)


class MultirunWorkDirTest(unittest.TestCase):
    def test_pure_identity_helpers_match_parser_directory_shape(self):
        identity = multirun_id(7, "exp_name=test,seed=2")

        self.assertRegex(identity, r"^job_0007_[0-9a-f]{12}$")
        self.assertEqual(
            multirun_work_dir(
                "/runs/test/seed_2",
                isolate_multirun_runs=True,
                job_num=7,
                override_dirname="exp_name=test,seed=2",
            ),
            Path("/runs/test/seed_2") / identity,
        )

    def test_different_overrides_use_different_work_dirs(self):
        with patch(
            "hydra.core.hydra_config.HydraConfig.get",
            return_value=hydra_job(0, "exp_name=test,lr=0.0003,seed=1"),
        ):
            first = parse_cfg(training_cfg(Path("/runs/test/seed_1")))

        with patch(
            "hydra.core.hydra_config.HydraConfig.get",
            return_value=hydra_job(1, "exp_name=test,lr=0.0001,seed=1"),
        ):
            second = parse_cfg(training_cfg(Path("/runs/test/seed_1")))

        self.assertNotEqual(first.work_dir, second.work_dir)
        self.assertRegex(first.work_dir.name, r"^job_0000_[0-9a-f]{12}$")
        self.assertRegex(second.work_dir.name, r"^job_0001_[0-9a-f]{12}$")

    def test_same_hydra_job_keeps_stable_work_dir_for_requeue(self):
        job = hydra_job(3, "embedding_dim=256,exp_name=test,seed=1")
        with patch("hydra.core.hydra_config.HydraConfig.get", return_value=job):
            first = parse_cfg(training_cfg(Path("/runs/test/seed_1")))
            second = parse_cfg(training_cfg(Path("/runs/test/seed_1")))

        self.assertEqual(first.work_dir, second.work_dir)

    def test_duplicate_configs_in_one_sweep_use_job_number_for_isolation(self):
        override_dirname = "exp_name=test,lr=0.0003,seed=1"
        with patch(
            "hydra.core.hydra_config.HydraConfig.get",
            return_value=hydra_job(0, override_dirname),
        ):
            first = parse_cfg(training_cfg(Path("/runs/test/seed_1")))

        with patch(
            "hydra.core.hydra_config.HydraConfig.get",
            return_value=hydra_job(1, override_dirname),
        ):
            duplicate = parse_cfg(training_cfg(Path("/runs/test/seed_1")))

        self.assertNotEqual(first.work_dir, duplicate.work_dir)


if __name__ == "__main__":
    unittest.main()
