import math
import unittest
from types import SimpleNamespace

from figures import _topology_catalog
from figures.topology_performance import (
    collect_seed_results,
    generalization_summary,
    noise_floor,
)

CV = "node_count_loso"
EXPERIMENT = "CV-BASE"
FOLD = "node_4"
MEMBERSHIP = _topology_catalog.fold_membership()[FOLD]
HELD_OUT = MEMBERSHIP["held_out"][0]
TRAIN = MEMBERSHIP["train"]


def _key(topology):
    return f"eval/{topology}_episode_distance"


def _run(run_id, seed, evals, *, fold=FOLD, cv_name=CV, group=f"truss-graph-{EXPERIMENT}",
         tags=None, summary_step=None, heldout=None, api="scan"):
    """``evals`` is a list of (step, train_value); held-out value = train_value unless given."""
    heldout = heldout or {}
    membership = _topology_catalog.fold_membership()[fold]
    rows = []
    for step, value in evals:
        row = {"_step": step}
        for name in membership["train"]:
            row[_key(name)] = value
        for name in membership["held_out"]:
            row[_key(name)] = heldout.get(step, value)
        rows.append(row)
        # Non-eval rows (train logging) interleave with eval rows in real histories.
        rows.append({"_step": step + 1})

    run = SimpleNamespace(
        id=run_id,
        name=f"{EXPERIMENT}-{seed}-holdout_{fold}",
        group=group,
        tags=tags if tags is not None else [],
        state="failed",
        config={
            "seed": seed,
            "cross_validation": {"enabled": True, "name": cv_name, "held_out_group": fold},
        },
        summary={"_step": summary_step if summary_step is not None else max(s for s, _ in evals)},
    )
    if api == "scan":
        run.scan_history = lambda keys: [
            {"_step": row["_step"], **{key: row.get(key) for key in keys}} for row in rows
        ]
    else:
        # Mimic W&B sampled history: only rows containing every requested key.
        run.history = lambda keys, samples, pandas: [
            {"_step": row["_step"], **{key: row[key] for key in keys}}
            for row in rows
            if all(key in row for key in keys)
        ]
    return run


def _rows(runs, **kwargs):
    return collect_seed_results(runs, CV, experiment=EXPERIMENT, **kwargs)


class TopologyPerformanceTest(unittest.TestCase):
    def test_tags_train_and_heldout_roles_from_fold_membership(self):
        rows = _rows([_run("one", 1, [(0, 0.0), (100, 1.0)])], window=1)
        roles = dict(zip(rows["topology"], rows["role"]))
        self.assertEqual(roles[HELD_OUT], "heldout")
        self.assertTrue(all(roles[name] == "train" for name in TRAIN))

    def test_reads_eval_prefixed_keys(self):
        run = _run("one", 1, [(0, 0.0), (100, 2.0)])
        requested = []
        scan = run.scan_history
        run.scan_history = lambda keys: (requested.extend(keys), scan(keys))[1]
        rows = _rows([run], window=1)
        self.assertTrue(requested and all(k.startswith("eval/") for k in requested))
        self.assertEqual(set(rows["distance_m"]), {2.0})

    def test_history_fallback_queries_keys_individually(self):
        rows = _rows([_run("one", 1, [(0, 0.0), (100, 2.0), (200, 4.0)], api="history")], window=1)
        self.assertEqual(set(rows["selected_step"]), {200})
        self.assertEqual(set(rows["distance_m"]), {4.0})

    def test_misaligned_seed_steps_are_not_truncated(self):
        rows = _rows(
            [
                _run("one", 1, [(0, 0.0), (100, 1.0), (200, 2.0), (300, 3.0)]),
                _run("two", 2, [(0, 0.0), (150, 1.0), (250, 2.0), (350, 3.0), (5000, 9.0)]),
            ],
            window=1,
        )
        by_seed = rows.drop_duplicates("seed").set_index("seed")
        self.assertEqual(by_seed.loc[1, "max_logged_eval_step"], 300)
        self.assertEqual(by_seed.loc[2, "max_logged_eval_step"], 5000)
        self.assertEqual(by_seed.loc[2, "selected_step"], 5000)
        self.assertEqual(by_seed.loc[1, "selected_step"], 300)

    def test_heldout_spike_does_not_move_selected_step(self):
        evals = [(0, 0.0), (100, 1.0), (200, 5.0), (300, 1.0)]
        rows = _rows([_run("one", 1, evals, heldout={100: 1000.0})], window=1)
        self.assertEqual(set(rows["selected_step"]), {200})
        heldout = rows[rows["topology"] == HELD_OUT].iloc[0]
        self.assertEqual(heldout["distance_m"], 5.0)

    def test_window_average_at_selected_step(self):
        # Train mean: a single spike at 400 is beaten by a sustained plateau.
        evals = [(0, 0.0), (100, 1.0), (200, 1.0), (300, 1.0), (400, 10.0),
                 (500, 1.0), (600, 1.0), (700, 6.0), (800, 6.0), (900, 6.0)]
        rows = _rows([_run("one", 1, evals, heldout={700: 3.0, 800: 6.0, 900: 9.0})], window=3)
        self.assertEqual(set(rows["selected_step"]), {900})
        self.assertEqual(set(rows["n_window_evals"]), {3})
        heldout = rows[rows["topology"] == HELD_OUT].iloc[0]
        self.assertAlmostEqual(heldout["distance_m"], 6.0)
        self.assertAlmostEqual(heldout["distance_final_m"], 6.0)

    def test_final_variant_uses_last_window_evals(self):
        evals = [(0, 0.0), (100, 9.0), (200, 9.0), (300, 1.0), (400, 3.0)]
        rows = _rows([_run("one", 1, evals)], window=2)
        row = rows.iloc[0]
        self.assertEqual(row["selected_step"], 200)
        self.assertAlmostEqual(row["distance_m"], 9.0)
        self.assertAlmostEqual(row["distance_final_m"], 2.0)

    def test_baseline_is_first_eval_and_excluded_from_windows(self):
        rows = _rows([_run("one", 1, [(0, 50.0), (100, 1.0), (200, 2.0)])], window=5)
        row = rows.iloc[0]
        self.assertEqual(row["baseline_step"], 0)
        self.assertEqual(row["baseline_distance_m"], 50.0)
        self.assertEqual(row["n_window_evals"], 2)
        self.assertAlmostEqual(row["distance_m"], 1.5)

    def test_filters_to_requested_experiment(self):
        rows = _rows(
            [
                _run("base", 1, [(0, 0.0), (100, 1.0)]),
                _run("tagged", 2, [(0, 0.0), (100, 2.0)], group="other", tags=[EXPERIMENT]),
                _run("other", 3, [(0, 0.0), (100, 3.0)], group="truss-graph-CV-3WAY-BASE",
                     tags=["CV-3WAY-BASE"]),
            ],
            window=1,
        )
        self.assertEqual(set(rows["run_id"]), {"base", "tagged"})

    def test_keeps_longest_run_per_fold_and_seed(self):
        rows = _rows(
            [
                _run("aborted", 1, [(0, 0.0), (100, 99.0)], summary_step=60_000),
                _run("full", 1, [(0, 0.0), (100, 1.0), (200, 2.0)], summary_step=8_000_000),
            ],
            window=1,
        )
        self.assertEqual(set(rows["run_id"]), {"full"})

    def test_ignores_runs_from_a_different_cross_validation_name(self):
        with self.assertRaises(RuntimeError):
            _rows([_run("one", 1, [(0, 0.0), (100, 1.0)], cv_name="other_split")])

    def test_ignores_runs_missing_cross_validation_config(self):
        run = SimpleNamespace(
            id="plain", name="plain-run", state="finished", group=f"truss-graph-{EXPERIMENT}",
            tags=[], config={"seed": 1}, summary={"_step": 0},
        )
        run.scan_history = lambda keys: []
        with self.assertRaises(RuntimeError):
            _rows([run])

    def test_noise_floor_and_summary(self):
        folds = _topology_catalog.fold_membership()
        other_fold = next(name for name in folds if name != FOLD)
        runs = []
        for seed, base in ((1, 0.0), (2, 0.2), (3, 0.4)):
            runs.append(_run(f"h{seed}", seed, [(0, base), (100, 4.0)], heldout={100: 2.0 + seed}))
            runs.append(_run(f"t{seed}", seed, [(0, base), (100, 4.0)], fold=other_fold))
        rows = _rows(runs, window=1)
        floor = noise_floor(rows).set_index("topology")
        self.assertEqual(floor.loc[HELD_OUT, "n_runs"], 6)
        self.assertAlmostEqual(floor.loc[HELD_OUT, "floor_mean_m"], 0.2)

        summary = generalization_summary(rows, noise_floor(rows)).set_index("topology")
        row = summary.loc[HELD_OUT]
        self.assertAlmostEqual(row["heldout_distance_m"], 4.0)  # mean of 3, 4, 5
        self.assertAlmostEqual(row["in_training_distance_m"], 4.0)
        self.assertAlmostEqual(row["transfer_ratio"], 1.0)
        floor_std = floor.loc[HELD_OUT, "floor_std_m"]
        self.assertEqual(floor.loc[HELD_OUT, "n_seeds"], 3)
        expected_z = (4.0 - 0.2) / math.sqrt(floor_std**2 / 3 + 1.0**2 / 3)
        self.assertAlmostEqual(row["heldout_above_floor_z"], expected_z)
        self.assertAlmostEqual(row["heldout_gain_m"], 3.8)


if __name__ == "__main__":
    unittest.main()
