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

from common.parser import parse_cfg


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
        with patch("common.parser.sys.orig_argv", argv):
            cfg = parse_cfg(topology_cfg())

        self.assertEqual(
            cfg.launch_command,
            ".venv/bin/python sac/gnn_train.py 'exp_name=my run' steps=1000",
        )


class MultirunWorkDirTest(unittest.TestCase):
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
