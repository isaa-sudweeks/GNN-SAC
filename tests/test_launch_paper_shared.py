import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.launch_paper_shared import (
    ARMS,
    PROJECT_ROOT,
    TOTAL_TRANSITIONS,
    arm_overrides,
    build_manifest,
    check_teachers,
    main,
    resolved_arm,
    submit_arms,
)


class PaperSharedLaunchTest(unittest.TestCase):
    def test_preset_split_matches_paper_baseline_and_total_budget(self):
        with (PROJECT_ROOT / "figures/greedy_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        expected_train = {row["topology"] for row in rows if row["held_out"] == "No"}
        expected_heldout = {row["topology"] for row in rows if row["held_out"] == "Yes"}
        for arm in ("gnn_scratch", "padded_mlp_scratch"):
            with self.subTest(arm=arm):
                cfg = resolved_arm(arm)
                self.assertEqual(set(cfg.truss_topologies), expected_train)
                self.assertEqual(set(cfg.eval_extra_topologies), expected_heldout)
                self.assertEqual(int(cfg.steps) * len(expected_train), TOTAL_TRANSITIONS)
                self.assertFalse(cfg.pcgrad)
                self.assertEqual(cfg.critical_eig_threshold, .01)
                self.assertEqual(cfg.discount, .95)
                self.assertTrue(cfg.save_eval_agent)
                self.assertEqual(cfg.eval_backend, "mujoco")
                self.assertEqual(cfg.num_envs % len(expected_train), 0)

    def test_only_resource_storage_overrides_are_allowed(self):
        for override in ("steps=10000000", "seed=99", "+truss_topologies=[octahedron]",
                         "distillation.pretrain_updates=0", "save_eval_agent=false"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                arm_overrides("gnn_scratch", [override])
        cfg = resolved_arm("gnn_scratch", ["num_envs=160", "hydra.launcher.mem_gb=96"])
        self.assertEqual(cfg.num_envs, 160)
        with self.assertRaisesRegex(ValueError, "divisible"):
            resolved_arm("gnn_scratch", ["num_envs=321"])

    def test_four_arms_have_three_seeds_and_ten_million_total_each(self):
        manifest = build_manifest(list(ARMS))
        self.assertEqual(manifest["job_count"], 12)
        for entry in manifest["arms"]:
            self.assertEqual([job["seed"] for job in entry["jobs"]], [1, 2, 3])
            self.assertEqual(entry["total_online_transitions"], 10_000_000)
            self.assertEqual(entry["per_topology_steps"], 1_000_000)
            self.assertEqual(set(entry["teachers"]) & set(entry["heldout_topologies"]), set())
            if entry["arm"].endswith("distill"):
                self.assertEqual(set(entry["teachers"]), set(entry["training_topologies"]))
                self.assertEqual(entry["offline_pretrain_updates"], 10000)
            else:
                self.assertEqual(entry["teachers"], {})
        distilled = [entry for entry in manifest["arms"] if entry["arm"].endswith("distill")]
        self.assertEqual(distilled[0]["teachers"], distilled[1]["teachers"])

    def test_all_conditions_start_before_any_controller_is_waited_on(self):
        events = []
        entries = [{"arm": arm, "command": ["fake", arm]} for arm in ARMS]

        def start(command, **kwargs):
            arm = command[1]
            events.append(("start", arm))

            class Controller:
                pid = 123

                def wait(self):
                    self_case.assertEqual(sum(e[0] == "start" for e in events), 4)
                    events.append(("wait", arm))
                    return 0

            return Controller()

        self_case = self
        with tempfile.TemporaryDirectory() as temp, \
                patch("scripts.launch_paper_shared.subprocess.Popen", side_effect=start):
            self.assertEqual(submit_arms(entries, {}, Path(temp)), 0)
            self.assertEqual([event for event, _ in events[:4]], ["start"] * 4)
            self.assertEqual({p.name for p in Path(temp).glob("*.log")},
                             {arm + ".log" for arm in ARMS})

    def test_failed_condition_does_not_prevent_other_conditions_starting(self):
        from unittest.mock import Mock

        controllers = [Mock(pid=i) for i in range(4)]
        for index, controller in enumerate(controllers):
            controller.wait.return_value = 2 if index == 0 else 0
        entries = [{"arm": arm, "command": ["fake", arm]} for arm in ARMS]
        with tempfile.TemporaryDirectory() as temp, \
                patch("scripts.launch_paper_shared.subprocess.Popen", side_effect=controllers) as start:
            self.assertEqual(submit_arms(entries, {}, Path(temp)), 1)
            self.assertEqual(start.call_count, 4)
            for controller in controllers:
                controller.wait.assert_called_once()

    def test_teacher_check_rejects_agent_only_or_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "step_100.agent.pt"
            path.write_bytes(b"weights")
            cfg = OmegaConf.create({"distillation": {"enabled": True, "teachers": {"test": str(path)}}})
            with self.assertRaisesRegex(FileNotFoundError, "non-full"):
                check_teachers(cfg)
            cfg.distillation.teachers.test = str(Path(temp) / "missing.pt")
            with self.assertRaises(FileNotFoundError):
                check_teachers(cfg)

    def test_default_dry_run_never_submits(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("scripts.launch_paper_shared.subprocess.Popen") as submit, \
                patch("scripts.launch_paper_shared.subprocess.check_output", return_value="test-sha\n"):
            self.assertEqual(main(["--arms", "gnn_scratch", "--manifest", str(Path(temp) / "manifest.json")]), 0)
            submit.assert_not_called()

    def test_submit_checks_all_teachers_before_submitting_any_arm(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("scripts.launch_paper_shared.subprocess.Popen") as submit, \
                patch("scripts.launch_paper_shared.subprocess.check_output", return_value="test-sha\n"), \
                patch("scripts.launch_paper_shared.check_teachers", side_effect=FileNotFoundError("missing teacher")):
            with self.assertRaises(SystemExit):
                main(["--submit", "--manifest", str(Path(temp) / "manifest.json")])
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
