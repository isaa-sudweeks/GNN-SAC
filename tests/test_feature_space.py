import unittest

import numpy as np
import pandas as pd

from figures._feature_space import (
    FEATURE_SPACES,
    build_feature_space,
    fit_residualization,
    knn_distances,
    residualize_on,
    select_distance_features,
)


def _synthetic() -> pd.DataFrame:
    n = np.array([4, 5, 6, 6, 7, 7, 8, 8, 9], dtype=float)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "topology": [f"t{i}" for i in range(len(n))],
            "n_nodes": n,
            "n_edges": 3 * n - 6,                      # exact copy of size
            "constant": np.zeros(len(n)),               # zero variance
            "one_off": [1.0] + [2.0] * (len(n) - 1),    # near-constant
            "shape": rng.normal(size=len(n)),           # independent
            "tubes": np.array([2, 1, 1, 2, 3, 1, 2, 3, 1], dtype=float),
        }
    )


class SelectDistanceFeaturesTest(unittest.TestCase):
    def test_drops_zero_variance_and_near_constant(self):
        df = _synthetic()
        cols, report = select_distance_features(df, ["n_nodes", "constant", "one_off", "shape"])
        self.assertEqual(report.zero_variance, ("constant",))
        self.assertEqual(report.near_constant, ("one_off",))
        self.assertEqual(cols, ["n_nodes", "shape"])

    def test_near_constant_can_be_disabled(self):
        cols, report = select_distance_features(
            _synthetic(), ["one_off", "shape"], min_minority_count=1
        )
        self.assertEqual(report.near_constant, ())
        self.assertIn("one_off", cols)

    def test_correlated_group_collapses_to_preferred_representative(self):
        df = _synthetic()
        # n_edges listed first, but n_nodes is preferred for the size group.
        cols, report = select_distance_features(df, ["n_edges", "n_nodes", "shape"])
        self.assertEqual(cols, ["n_nodes", "shape"])
        size_group = next(g for g in report.groups if g.representative == "n_nodes")
        self.assertEqual(size_group.dropped, ("n_edges",))
        self.assertAlmostEqual(size_group.abs_correlations[0], 1.0)

    def test_collapse_is_deterministic_and_order_independent(self):
        df = _synthetic()
        df["a"] = df["shape"] * 2.0 + 1.0
        df["b"] = -df["shape"]
        first = select_distance_features(df, ["a", "b", "shape"], preference=())
        second = select_distance_features(df, ["a", "b", "shape"], preference=())
        self.assertEqual(first, second)
        # Without a preference, the first listed column represents the group.
        self.assertEqual(first[0], ["a"])
        cols, _ = select_distance_features(df, ["b", "shape", "a"], preference=("shape",))
        self.assertEqual(cols, ["shape"])

    def test_no_single_linkage_chaining(self):
        # x~y and y~z above threshold, x~z below: greedy from x keeps z.
        t = np.linspace(0.0, 1.0, 50)
        noise = np.sin(37 * t)
        df = pd.DataFrame({"x": t, "y": t + 0.08 * noise, "z": t + 0.16 * noise})
        corr = df.corr().abs()
        threshold = (corr.loc["x", "z"] + min(corr.loc["x", "y"], corr.loc["y", "z"])) / 2
        self.assertLess(corr.loc["x", "z"], threshold)
        cols, _ = select_distance_features(
            df, ["x", "y", "z"], corr_threshold=threshold, preference=(),
            linear_dependence_r2=None,
        )
        self.assertEqual(cols, ["x", "z"])

    def test_linear_dependence_on_two_kept_features(self):
        df = _synthetic()
        # Exact identity, but pairwise |r| with either predictor stays < 0.95.
        df["active"] = df["n_nodes"] - 2.0 * df["tubes"]
        self.assertLess(df[["active", "n_nodes", "tubes"]].corr().abs()["active"].iloc[1:].max(), 0.95)
        cols, report = select_distance_features(
            df, ["n_nodes", "tubes", "active", "shape"], preference=("n_nodes", "tubes")
        )
        self.assertNotIn("active", cols)
        self.assertEqual(report.linear_dependent[0].feature, "active")
        self.assertEqual(set(report.linear_dependent[0].predictors), {"n_nodes", "tubes"})
        self.assertIn("linear_dependent", set(report.to_frame()["status"]))

    def test_nan_is_rejected(self):
        df = _synthetic()
        df.loc[0, "shape"] = np.nan
        with self.assertRaises(ValueError):
            select_distance_features(df, ["shape"])


class ResidualizationTest(unittest.TestCase):
    def test_fit_uses_only_fit_rows(self):
        df = pd.DataFrame(
            {"n_nodes": [1.0, 2.0, 3.0, 10.0], "y": [2.0, 4.0, 6.0, 1000.0]},
            index=["a", "b", "c", "held"],
        )
        residuals = residualize_on(df, ["y"], fit_rows=["a", "b", "c"])
        np.testing.assert_allclose(residuals.loc[["a", "b", "c"], "y"], 0.0, atol=1e-12)
        # Held-out row is transformed with the train fit y = 2 x.
        self.assertAlmostEqual(residuals.loc["held", "y"], 1000.0 - 20.0)
        self.assertNotIn("n_nodes", residuals.columns)
        # Including the outlier in the fit changes the result.
        self.assertNotAlmostEqual(residualize_on(df, ["y"]).loc["a", "y"], 0.0)

    def test_boolean_mask_fit_rows(self):
        df = pd.DataFrame({"n_nodes": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 30.0]})
        by_mask = residualize_on(df, ["y"], fit_rows=[True, True, False])
        by_label = residualize_on(df, ["y"], fit_rows=[0, 1])
        pd.testing.assert_frame_equal(by_mask, by_label)

    def test_degenerate_single_covariate_value_is_intercept_only(self):
        df = pd.DataFrame(
            {"n_nodes": [6.0, 6.0, 6.0, 8.0], "y": [1.0, 2.0, 3.0, 10.0]},
            index=["a", "b", "c", "held"],
        )
        fit = fit_residualization(df, ["y"], fit_rows=["a", "b", "c"])
        self.assertTrue(fit.degenerate)
        self.assertEqual(fit.slopes["y"], 0.0)
        residuals = fit.transform(df)
        np.testing.assert_allclose(residuals["y"].to_numpy(), [-1.0, 0.0, 1.0, 8.0])
        self.assertTrue(fit.extrapolates(df).loc["held"])
        self.assertFalse(fit.extrapolates(df).loc["a"])


class BuildFeatureSpaceTest(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic()
        self.columns = ["n_nodes", "n_edges", "constant", "one_off", "shape", "tubes"]
        self.train = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]

    def test_variants_select_expected_columns(self):
        spaces = {
            variant: build_feature_space(
                self.df, variant, self.train, ["t8"], feature_columns=self.columns
            )
            for variant in FEATURE_SPACES
        }
        self.assertEqual(spaces["full"].columns, ("n_nodes", "n_edges", "shape", "tubes"))
        self.assertEqual(spaces["deduplicated"].columns, ("n_nodes", "shape", "tubes"))
        self.assertEqual(spaces["size_controlled"].columns, ("shape", "tubes"))
        self.assertIsNotNone(spaces["size_controlled"].residualization)

    def test_standardization_fit_on_train_rows_only(self):
        space = build_feature_space(
            self.df, "deduplicated", self.train, ["t8"], feature_columns=self.columns
        )
        np.testing.assert_allclose(space.train.mean().to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(space.train.std(ddof=0).to_numpy(), 1.0)
        train_n = self.df.set_index("topology").loc[self.train, "n_nodes"]
        expected = (9.0 - train_n.mean()) / train_n.std(ddof=0)
        self.assertAlmostEqual(space.held_out.loc["t8", "n_nodes"], expected)

    def test_size_controlled_flags_extrapolated_held_out(self):
        inside = build_feature_space(
            self.df, "size_controlled", ["t0", "t1", "t2", "t5", "t7", "t8"], ["t4"],
            feature_columns=self.columns,
        )
        self.assertFalse(inside.extrapolated)
        outside = build_feature_space(
            self.df, "size_controlled", self.train, ["t0", "t8"], feature_columns=self.columns
        )
        self.assertTrue(outside.extrapolated)
        self.assertEqual(outside.extrapolated_topologies, ("t0", "t8"))

    def test_size_controlled_residuals_uncorrelated_with_size_on_train(self):
        space = build_feature_space(
            self.df, "size_controlled", self.train, ["t8"], feature_columns=self.columns
        )
        n = self.df.set_index("topology").loc[self.train, "n_nodes"].to_numpy()
        for column in space.columns:
            self.assertAlmostEqual(float(np.dot(n - n.mean(), space.train[column])), 0.0)

    def test_train_constant_column_uses_all_row_scale(self):
        df = self.df.copy()
        # Constant on train rows (t1..t7); differs on t0 and t8 only.
        df["rare"] = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        space = build_feature_space(
            df, "full", self.train, ["t0"], feature_columns=["shape", "rare"]
        )
        self.assertEqual(space.fallback_scale_columns, ("rare",))
        self.assertTrue(np.isfinite(space.held_out.to_numpy()).all())

    def test_overlapping_splits_rejected(self):
        with self.assertRaises(ValueError):
            build_feature_space(self.df, "full", ["t0", "t1"], ["t1"], feature_columns=["shape"])
        with self.assertRaises(ValueError):
            build_feature_space(self.df, "bogus", ["t0"], ["t1"], feature_columns=["shape"])

    def test_knn_distances_excludes_self_for_train_rows(self):
        space = build_feature_space(
            self.df, "deduplicated", self.train, ["t8"], feature_columns=self.columns
        )
        distances = knn_distances(space, k=2)
        self.assertEqual(list(distances["role"]), ["train"] * 7 + ["heldout"])
        self.assertTrue((distances["knn_distance"] > 0).all())
        point = space.held_out.loc["t8"].to_numpy()
        manual = np.sort(np.linalg.norm(space.train.to_numpy() - point, axis=1))[:2].mean()
        self.assertAlmostEqual(distances.iloc[-1]["knn_distance"], manual)


if __name__ == "__main__":
    unittest.main()
