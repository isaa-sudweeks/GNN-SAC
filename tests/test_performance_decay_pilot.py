"""Audit provenance and paired launch configuration regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    sys.path.insert(0, str(path))

from common.distillation import Distillation, model_state_sha256
from common.distillation_schedule import PerformanceDecay
from scripts.launch_performance_decay_pilot import main, pilot_config, validate_audit
from tests.test_distillation import setup


class PerformanceDecayPilotTest(unittest.TestCase):
    def test_audit_covers_selected_checkpoints_and_isolates_final_test(self):
        cfg = pilot_config([])
        with patch("scripts.launch_performance_decay_pilot.torch.load", side_effect=AssertionError("Teacher opened during dry run")):
            rows = validate_audit(cfg, check_teachers=False)
        self.assertEqual(len(rows), 15)
        self.assertEqual(sum(row["adaptive_eligible"] for row in rows), 10)
        self.assertEqual(cfg.steps, 30000000)
        self.assertEqual(len(cfg.distillation.teachers), 19)
        self.assertEqual(set(cfg.eval_extra_topologies), {
            "henneberg_n6_1tube_1", "henneberg_n6_3tube_1",
            "henneberg_n8_2tube_187", "henneberg_n8_3tube_11"})
        for row in rows:
            self.assertIn(row["audited_resets"], (15, 30))
        for heldout in list(cfg.eval_extra_topologies) + list(cfg.cross_validation.final_test):
            self.assertNotIn(heldout, [row["topology"] for row in rows])

    def test_rejects_changed_audit_metrics_or_evaluation_settings(self):
        for change in ("distance", "randomization", "termination"):
            cfg = pilot_config([])
            if change == "distance":
                cfg.distillation.performance_decay.references.tetrahedron.distance += 1
            elif change == "randomization":
                cfg.eval_domain_randomization = False
            else:
                cfg.critical_eig_threshold = 0.0001
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_audit(cfg, check_teachers=False)

    def test_dry_run_records_six_jobs_without_submitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.launch_performance_decay_pilot.subprocess.run") as run:
            path = Path(tmp) / "manifest.json"
            self.assertEqual(main(["--dry-run", "--manifest", str(path), "--teacher-root", "/relocated"]), 0)
            run.assert_not_called()
            result = json.loads(path.read_text())
            self.assertEqual(result["job_count"], 6)
            self.assertFalse(result["teachers_verified"])
            self.assertIn("distillation.performance_decay.enabled=false,true", result["command"])
            self.assertTrue(all(row["checkpoint"].startswith("/relocated/") for row in result["teachers"]))

    def test_distillation_checks_model_hash_and_policy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, original = setup(tmp)
            references = {}
            for task, source in original.sources.items():
                state = torch.load(source["path"], map_location="cpu", weights_only=False)
                references[task.split(":")[1]] = dict(
                    distance=10, survival=1, horizon=100,
                    teacher_model_sha256=model_state_sha256(state["agent"]["model"]),
                    policy_config={"log_std_min": cfg.log_std_min})
            cfg.distillation["performance_decay"] = dict(enabled=True, references=references)
            distill = Distillation(cfg, cfg.tasks)
            self.assertEqual(distill.sources[cfg.tasks[0]]["model_sha256"],
                             references["a"]["teacher_model_sha256"])
            references["a"]["policy_config"]["log_std_min"] += 1
            with self.assertRaisesRegex(ValueError, "policy config differs"):
                Distillation(cfg, cfg.tasks)
            references["a"]["policy_config"]["log_std_min"] -= 1
            references["a"]["teacher_model_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "mismatched performance reference"):
                Distillation(cfg, cfg.tasks)

    def test_model_hash_binding_and_ineligible_references(self):
        state = {"b": torch.tensor([1.0]), "a": torch.tensor([2.0])}
        digest = model_state_sha256(state)
        self.assertEqual(digest, model_state_sha256(dict(reversed(list(state.items())))))
        changed = deepcopy(state)
        changed["b"][0] += 1
        self.assertNotEqual(digest, model_state_sha256(changed))
        options = dict(patience=1, release_steps=10, references={"a": dict(
            distance=-1.0, survival=0.0, horizon=1000,
            eligible=False, teacher_model_sha256=digest)})
        schedule = PerformanceDecay(options, {"task:a": "a"}, {"task:a": {"model_sha256": digest}})
        schedule.observe("task:a", 1, 100, 1, 1000)
        self.assertEqual(schedule.multiplier("task:a", 1000), 1.0)
        self.assertIsNone(schedule.progress["task:a"]["release_start"])
        with self.assertRaises(ValueError):
            PerformanceDecay(options, {"task:a": "a"}, {"task:a": {"model_sha256": "wrong"}})


if __name__ == "__main__":
    unittest.main()
