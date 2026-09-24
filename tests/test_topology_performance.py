import unittest
from types import SimpleNamespace

from figures import _topology_catalog
from figures.topology_performance import collect_seed_results

FOLD = "node_4"
MEMBERSHIP = _topology_catalog.fold_membership()[FOLD]
HELD_OUT_TOPOLOGY = MEMBERSHIP["held_out"][0]
TRAIN_TOPOLOGY = MEMBERSHIP["train"][0]


def _run(run_id, seed, history, *, fold=FOLD, cv_name="node_count_loso"):
    run = SimpleNamespace(
        id=run_id,
        name=f"cv-{fold}-{seed}",
        state="finished",
        config={
            "seed": seed,
            "cross_validation": {"enabled": True, "name": cv_name, "held_out_group": fold},
        },
        summary={"_step": max(step for step, *_ in history)},
    )
    run.history = lambda keys, samples, pandas: [
        {
            "_step": step,
            f"{HELD_OUT_TOPOLOGY}_episode_distance": heldout_distance,
            f"{TRAIN_TOPOLOGY}_episode_distance": train_distance,
        }
        for step, heldout_distance, train_distance in history
    ]
    return run


class TopologyPerformanceTest(unittest.TestCase):
    def test_tags_train_and_heldout_roles_from_fold_membership(self):
        rows = collect_seed_results(
            [_run("one", 1, [(100, 1.0, 5.0)]), _run("two", 2, [(100, 3.0, 7.0)])],
            "node_count_loso",
        )
        roles = dict(zip(rows["topology"], rows["role"]))
        self.assertEqual(roles[HELD_OUT_TOPOLOGY], "heldout")
        self.assertEqual(roles[TRAIN_TOPOLOGY], "train")

    def test_selects_peak_across_seed_mean_step(self):
        rows = collect_seed_results(
            [
                _run("one", 1, [(100, 9.0, 1.0), (200, 6.0, 1.0)]),
                _run("two", 2, [(100, 1.0, 1.0), (200, 6.0, 1.0)]),
            ],
            "node_count_loso",
        )
        heldout_rows = rows[rows["topology"] == HELD_OUT_TOPOLOGY]
        self.assertEqual(set(heldout_rows["evaluation_step"]), {200})

    def test_ignores_runs_from_a_different_cross_validation_name(self):
        with self.assertRaises(RuntimeError):
            collect_seed_results(
                [_run("one", 1, [(100, 1.0, 5.0)], cv_name="other_split")],
                "node_count_loso",
            )

    def test_ignores_runs_missing_cross_validation_config(self):
        run = SimpleNamespace(
            id="plain", name="plain-run", state="finished",
            config={"seed": 1}, summary={"_step": 0},
        )
        run.history = lambda keys, samples, pandas: []
        with self.assertRaises(RuntimeError):
            collect_seed_results([run], "node_count_loso")


if __name__ == "__main__":
    unittest.main()
