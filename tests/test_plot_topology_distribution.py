import unittest

import numpy as np
import pandas as pd

from figures import plot_topology_distribution as ptd


def _seed_results(heldout_gain_by_topology, *, folds=None, n_seeds=3, seed_noise=0.0):
    """Fake per-(fold, seed, topology) rows: each held-out topology is alone in its fold."""
    rng = np.random.default_rng(0)
    rows = []
    topologies = list(heldout_gain_by_topology)
    folds = folds or {name: f"fold_{i}" for i, name in enumerate(topologies)}
    for heldout_name in topologies:
        fold = folds[heldout_name]
        for seed in range(1, n_seeds + 1):
            for name in topologies:
                role = "heldout" if name == heldout_name else "train"
                baseline = 0.1 * seed
                gain = heldout_gain_by_topology[name] if role == "heldout" else 3.0
                rows.append({
                    "fold": fold, "seed": seed, "topology": name, "role": role,
                    "distance_m": baseline + gain + seed_noise * rng.standard_normal(),
                    "baseline_distance_m": baseline,
                })
    return pd.DataFrame(rows)


class SeedGainTest(unittest.TestCase):
    def test_gain_is_paired_against_the_runs_own_step_zero(self):
        results = _seed_results({"a": 0.5, "b": -0.2, "c": 0.0})
        performance = ptd.topology_performance(results)
        heldout = performance[performance["role"] == "heldout"].set_index("topology")
        np.testing.assert_allclose(heldout.loc[["a", "b", "c"], "gain_m"], [0.5, -0.2, 0.0])

    def test_noise_floor_counts_seeds_not_runs(self):
        results = _seed_results({"a": 0.5, "b": -0.2, "c": 0.0})
        floor = ptd.noise_floor(results).set_index("topology")
        # 3 folds x 3 seeds = 9 runs per topology, but only 3 independent seeds.
        self.assertTrue((floor["n_floor_seeds"] == 3).all())
        np.testing.assert_allclose(floor["floor_std_m"], np.std([0.1, 0.2, 0.3], ddof=1))


class DistanceNormalizationTest(unittest.TestCase):
    def test_relative_distance_divides_by_median_train_spacing(self):
        distances = pd.DataFrame({
            "topology": ["t1", "t2", "t3", "h"],
            "role": ["train", "train", "train", "heldout"],
            "knn_distance": [1.0, 2.0, 3.0, 4.0],
        })
        normalized = ptd._normalize_by_train_spacing(distances)
        self.assertEqual(normalized["train_spacing"].iloc[0], 2.0)
        self.assertEqual(normalized.set_index("topology").loc["h", "knn_distance_rel"], 2.0)

    def test_gnn_distance_excludes_self_for_train_and_ignores_heldout_neighbors(self):
        topologies = ["t1", "t2", "t3", "h"]
        embeddings = np.array([[0.0], [1.0], [3.0], [10.0]])
        membership = {"train": ["t1", "t2", "t3"], "held_out": ["h"]}
        distances = ptd._gnn_knn_distances(topologies, embeddings, membership, k=1).set_index("topology")
        self.assertEqual(distances.loc["t1", "knn_distance"], 1.0)
        self.assertEqual(distances.loc["t3", "knn_distance"], 2.0)
        self.assertEqual(distances.loc["h", "knn_distance"], 7.0)


class DistanceStatisticsTest(unittest.TestCase):
    def _heldout(self, gains, distances):
        results = _seed_results(dict(zip([f"topo{i}" for i in range(len(gains))], gains)), seed_noise=0.01)
        performance = ptd.topology_performance(results)
        floor = ptd.noise_floor(results)
        distance_rows = pd.DataFrame({
            "fold": [f"fold_{i}" for i in range(len(gains))],
            "space": "hand_deduplicated",
            "topology": [f"topo{i}" for i in range(len(gains))],
            "role": "heldout",
            "knn_distance": distances,
            "knn_distance_rel": distances,
            "extrapolated": False,
        })
        return ptd.heldout_table(distance_rows, performance, floor), ptd.seed_gains(results)

    def test_one_row_per_heldout_topology_and_train_rows_excluded(self):
        heldout, _ = self._heldout([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(heldout), 6)
        self.assertTrue((heldout["role"] == "heldout").all())

    def test_perfect_monotone_relation_is_detected(self):
        distances = np.arange(8, dtype=float)
        heldout, gains = self._heldout(list(-distances), distances)
        stats = ptd.distance_statistics(heldout, gains, n_permutations=2000, n_bootstrap=200).iloc[0]
        self.assertAlmostEqual(stats["spearman_rho"], -1.0)
        self.assertLess(stats["permutation_p"], 0.01)
        self.assertLess(stats["rho_ci_high"], -0.9)

    def test_permutation_p_value_is_never_zero(self):
        rng = np.random.default_rng(0)
        x = np.arange(6, dtype=float)
        p_value = ptd.permutation_p_value(x, x, n_permutations=50, rng=rng)
        self.assertGreater(p_value, 0.0)

    def test_too_few_topologies_reports_no_statistics(self):
        heldout, gains = self._heldout([0.1, 0.2, 0.3], [1, 2, 3])
        stats = ptd.distance_statistics(heldout, gains, n_permutations=100, n_bootstrap=50).iloc[0]
        self.assertEqual(stats["n_heldout_topologies"], 3)
        self.assertTrue(np.isnan(stats["spearman_rho"]))

    def test_statistics_are_deterministic_for_a_fixed_seed(self):
        heldout, gains = self._heldout([0.3, -0.1, 0.2, 0.0, 0.5, -0.4], [1, 5, 2, 4, 3, 6])
        first = ptd.distance_statistics(heldout, gains, n_permutations=500, n_bootstrap=100, seed=3)
        second = ptd.distance_statistics(heldout, gains, n_permutations=500, n_bootstrap=100, seed=3)
        pd.testing.assert_frame_equal(first, second)

    def test_signal_warning_when_nothing_beats_the_floor(self):
        heldout, _ = self._heldout([0.0] * 6, [1, 2, 3, 4, 5, 6])
        self.assertIn("fitting noise", ptd._signal_warning(heldout))


if __name__ == "__main__":
    unittest.main()
