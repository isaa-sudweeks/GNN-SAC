"""Feature-space construction for fold-conditional topology distances.

The hand-crafted topology features (``topology_hand_features.py``) are used to
ask "how far is a held-out topology from its fold's training topologies?".
Two problems make raw standardized Euclidean distance misleading:

* Several features are constant on the catalog, and many are near-exact
  functions of ``n_nodes`` (``n_edges = 3n - 6``, ``degree_mean``,
  ``spectral_radius``, ...). Every such copy re-counts size in the distance.
* Folds are split by node count (leave-one-node-count-out), so any size-driven
  distance tracks the fold split by construction.

This module therefore builds three variants (``FEATURE_SPACES``):

``"full"``
    Every non-constant (and not near-constant) feature, standardized on the
    fold's training rows.
``"deduplicated"``
    After collapsing groups of highly correlated features (``|Pearson| >=
    corr_threshold``) to one interpretable representative each, and dropping
    exact linear identities (e.g. ``n_active_nodes = n_edges - n_tubes``).
``"size_controlled"``
    The deduplicated features, minus the covariate (``n_nodes``), each
    replaced by its residual from a linear fit on the covariate over the
    fold's *training* rows only. Distance then asks "is this topology unusual
    beyond what its size predicts?".

Feature *selection* (zero-variance and correlation collapse) is computed on
all rows passed in. That uses no performance labels, so it is not label
leakage; standardization and residualization are fit on training rows only.

Extrapolation caveat: under leave-one-node-count-out, a held-out node count
can fall outside the training range (the smallest and largest groups, e.g.
``node_4`` and ``node_9``). The residual fit is then evaluated outside the
data it was fit on, so ``size_controlled`` distances for those folds rest on
linear extrapolation and should be read with care; ``FeatureSpace`` exposes
``extrapolated`` / ``extrapolated_topologies`` so callers can flag them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FEATURE_SPACES: tuple[str, ...] = ("full", "deduplicated", "size_controlled")

DEFAULT_COVARIATE = "n_nodes"
DEFAULT_CORR_THRESHOLD = 0.95

# Representative preference when collapsing a correlated group: earlier names
# win. Chosen for interpretability (plain counts and named structural
# quantities before derived statistics). Columns not listed follow, in the
# order they were passed.
DEFAULT_PREFERENCE: tuple[str, ...] = (
    "n_nodes",
    "n_tubes",
    "tube_length_mean",
    "closed_tube_fraction",
    "static_rigidity",
    "algebraic_connectivity",
    "degree_std",
    "degree_max",
    "degree_min",
    "n_passive_nodes",
    "n_active_nodes",
    "active_node_fraction",
    "node_multiplicity_mean",
    "node_multiplicity_max",
    "n_connectors",
    "n_edges",
    "degree_mean",
    "spectral_radius",
)

# Relative tolerance under which a column is treated as constant.
_ZERO_VARIANCE_RTOL = 1e-12

# A column is "near-constant" when fewer than this many rows differ from its
# modal value (e.g. ``diameter``: 2 everywhere except the tetrahedron). Such a
# column is a one-topology indicator, not a dimension of the space.
DEFAULT_MIN_MINORITY_COUNT = 2

# A column is "linearly dependent" when ordinary least squares on at most
# ``DEFAULT_MAX_PREDICTORS`` already-kept columns explains it with R^2 at or
# above this (e.g. ``n_active_nodes = n_edges - n_tubes`` exactly). Kept to
# very small predictor sets: with ~13 rows, larger sets fit anything.
DEFAULT_LINEAR_DEPENDENCE_R2 = 0.999
DEFAULT_MAX_PREDICTORS = 2


@dataclass(frozen=True)
class CorrelationGroup:
    """One collapsed group: ``representative`` is kept, ``dropped`` are removed."""

    representative: str
    dropped: tuple[str, ...]
    # |Pearson r| between the representative and each dropped member, same order.
    abs_correlations: tuple[float, ...]


@dataclass(frozen=True)
class LinearDependence:
    """A column dropped because ``predictors`` (kept columns) reproduce it."""

    feature: str
    predictors: tuple[str, ...]
    r2: float


@dataclass(frozen=True)
class SelectionReport:
    """What ``select_distance_features`` removed and why.

    Steps run in order: ``zero_variance`` -> ``near_constant`` -> pairwise
    correlation ``groups`` -> ``linear_dependent``.
    """

    corr_threshold: float
    zero_variance: tuple[str, ...]
    near_constant: tuple[str, ...]
    groups: tuple[CorrelationGroup, ...]
    linear_dependent: tuple[LinearDependence, ...]
    selected: tuple[str, ...]

    @property
    def correlated_dropped(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group.dropped)

    def to_frame(self) -> pd.DataFrame:
        """One row per input feature: ``feature, status, representative, abs_corr``."""
        rows = [
            {"feature": name, "status": "zero_variance", "representative": None, "abs_corr": np.nan}
            for name in self.zero_variance
        ]
        rows += [
            {"feature": name, "status": "near_constant", "representative": None, "abs_corr": np.nan}
            for name in self.near_constant
        ]
        dependent = {entry.feature: entry for entry in self.linear_dependent}
        for group in self.groups:
            if group.representative in dependent:
                entry = dependent[group.representative]
                rows.append(
                    {"feature": entry.feature, "status": "linear_dependent",
                     "representative": " + ".join(entry.predictors), "abs_corr": entry.r2}
                )
            else:
                rows.append(
                    {"feature": group.representative, "status": "kept",
                     "representative": group.representative, "abs_corr": 1.0}
                )
            for name, corr in zip(group.dropped, group.abs_correlations):
                rows.append(
                    {"feature": name, "status": "collapsed",
                     "representative": group.representative, "abs_corr": corr}
                )
        # For linear_dependent rows ``abs_corr`` holds R^2.
        return pd.DataFrame(rows, columns=["feature", "status", "representative", "abs_corr"])

    def format(self) -> str:
        lines = [f"Zero-variance (dropped): {', '.join(self.zero_variance) or 'none'}"]
        lines.append(f"Near-constant (dropped): {', '.join(self.near_constant) or 'none'}")
        lines.append(f"Correlation groups (|r| >= {self.corr_threshold:g}):")
        for group in self.groups:
            if group.dropped:
                members = ", ".join(
                    f"{name} ({corr:.3f})"
                    for name, corr in zip(group.dropped, group.abs_correlations)
                )
                lines.append(f"  keep {group.representative}; drop {members}")
        if self.linear_dependent:
            lines.append("Linearly dependent on kept features (dropped):")
            for entry in self.linear_dependent:
                lines.append(
                    f"  {entry.feature} ~ {' + '.join(entry.predictors)} (R^2 = {entry.r2:.4f})"
                )
        lines.append(f"Selected ({len(self.selected)}): {', '.join(self.selected)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ResidualizationFit:
    """Per-column linear fit ``feature ~ intercept + slope * covariate``."""

    covariate: str
    intercepts: dict[str, float]
    slopes: dict[str, float]
    # True when the fit rows had < 2 distinct covariate values; the fit then
    # falls back to intercept-only (slope 0), i.e. residual = x - mean(x_fit).
    degenerate: bool
    covariate_min: float
    covariate_max: float

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.covariate].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                column: df[column].to_numpy(dtype=float)
                - (self.intercepts[column] + self.slopes[column] * x)
                for column in self.intercepts
            },
            index=df.index,
        )

    def extrapolates(self, df: pd.DataFrame) -> pd.Series:
        """Boolean per row: covariate outside the fit rows' [min, max]."""
        x = df[self.covariate].to_numpy(dtype=float)
        return pd.Series((x < self.covariate_min) | (x > self.covariate_max), index=df.index)


@dataclass(frozen=True)
class FeatureSpace:
    """Standardized fold-conditional feature matrices for one variant.

    ``train`` / ``held_out`` are indexed by topology name with ``columns`` as
    columns, standardized with the training rows' mean and population std.
    """

    variant: str
    columns: tuple[str, ...]
    train: pd.DataFrame
    held_out: pd.DataFrame
    selection: SelectionReport
    residualization: ResidualizationFit | None
    center: dict[str, float]
    scale: dict[str, float]
    # Columns with zero std on the training rows; scaled by their std over all
    # rows instead (label-free), or dropped if that is zero too.
    fallback_scale_columns: tuple[str, ...] = ()
    dropped_constant_columns: tuple[str, ...] = ()
    extrapolated_topologies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def extrapolated(self) -> bool:
        """True when some held-out covariate lies outside the training range."""
        return bool(self.extrapolated_topologies)


def _ordered_by_preference(columns: Sequence[str], preference: Sequence[str]) -> list[str]:
    rank = {name: index for index, name in enumerate(preference)}
    position = {name: index for index, name in enumerate(columns)}
    return sorted(columns, key=lambda name: (rank.get(name, len(rank)), position[name]))


def _is_zero_variance(values: np.ndarray) -> bool:
    if len(values) == 0:
        return True
    spread = float(np.max(values) - np.min(values))
    return spread <= _ZERO_VARIANCE_RTOL * max(1.0, float(np.max(np.abs(values))))


def _minority_count(values: np.ndarray) -> int:
    """Rows whose value differs from the modal value (ties: smallest value)."""
    rounded = np.round(values.astype(float), 9)
    unique, counts = np.unique(rounded, return_counts=True)
    return int(len(values) - counts.max())


def _r2(target: np.ndarray, predictors: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(target)), predictors])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0


def _find_linear_dependence(
    values: pd.DataFrame, column: str, kept: Sequence[str], max_predictors: int, r2_threshold: float
) -> LinearDependence | None:
    from itertools import combinations

    target = values[column].to_numpy(dtype=float)
    for size in range(1, max_predictors + 1):
        best: LinearDependence | None = None
        for subset in combinations(kept, size):
            r2 = _r2(target, values[list(subset)].to_numpy(dtype=float))
            if r2 >= r2_threshold and (best is None or r2 > best.r2 + 1e-12):
                best = LinearDependence(column, tuple(subset), round(r2, 6))
        if best is not None:
            return best
    return None


def select_distance_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    preference: Sequence[str] = DEFAULT_PREFERENCE,
    min_minority_count: int = DEFAULT_MIN_MINORITY_COUNT,
    linear_dependence_r2: float | None = DEFAULT_LINEAR_DEPENDENCE_R2,
    max_predictors: int = DEFAULT_MAX_PREDICTORS,
) -> tuple[list[str], SelectionReport]:
    """Choose non-redundant distance features. Deterministic; label-free.

    1. Drop zero-variance columns.
    2. Drop near-constant columns: fewer than ``min_minority_count`` rows
       differ from the modal value (``min_minority_count=1`` disables).
    3. Collapse correlated groups, greedily: columns are visited in
       ``preference`` order (unlisted ones after, in ``feature_columns``
       order); each not-yet-assigned column becomes a representative and
       absorbs every unassigned column with ``|Pearson r| >= corr_threshold``
       to it. Every dropped column is thus highly correlated with the kept
       one that replaced it (no single-linkage chaining). Pass
       ``corr_threshold=np.inf`` to skip this step and step 4.
    4. Drop representatives that are (near-)exact linear combinations of up
       to ``max_predictors`` earlier-kept representatives (R^2 >=
       ``linear_dependence_r2``; ``None`` disables). Catches identities that
       pairwise correlation misses, e.g. ``n_active_nodes = n_edges - n_tubes``.

    Returns ``(selected_columns, report)``; ``selected_columns`` keeps the
    original ``feature_columns`` order.
    """
    feature_columns = list(feature_columns)
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns contains duplicates")
    missing = [name for name in feature_columns if name not in df.columns]
    if missing:
        raise KeyError(f"Columns not in DataFrame: {missing}")
    values = df[feature_columns].astype(float)
    if values.isna().any().any():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Feature columns contain NaN: {bad}")

    zero_variance = [
        name for name in feature_columns if _is_zero_variance(values[name].to_numpy())
    ]
    remaining = [name for name in feature_columns if name not in zero_variance]
    near_constant = [
        name for name in remaining
        if _minority_count(values[name].to_numpy()) < min_minority_count
    ]
    candidates = [name for name in remaining if name not in near_constant]
    collapse = np.isfinite(corr_threshold)

    groups: list[CorrelationGroup] = []
    if candidates and not collapse:
        groups = [CorrelationGroup(name, (), ()) for name in candidates]
    elif candidates:
        corr = np.abs(np.corrcoef(values[candidates].to_numpy().T))
        corr = np.atleast_2d(corr)
        index = {name: i for i, name in enumerate(candidates)}
        assigned: set[str] = set()
        ordered = _ordered_by_preference(candidates, preference)
        for representative in ordered:
            if representative in assigned:
                continue
            assigned.add(representative)
            dropped, dropped_corr = [], []
            for other in ordered:
                if other in assigned:
                    continue
                value = float(corr[index[representative], index[other]])
                if value >= corr_threshold:
                    assigned.add(other)
                    dropped.append(other)
                    dropped_corr.append(round(value, 6))
            groups.append(CorrelationGroup(representative, tuple(dropped), tuple(dropped_corr)))

    dependent: list[LinearDependence] = []
    if collapse and linear_dependence_r2 is not None and max_predictors > 0:
        kept_so_far: list[str] = []
        for group in groups:  # already in preference order
            found = _find_linear_dependence(
                values, group.representative, kept_so_far, max_predictors, linear_dependence_r2
            )
            if found is None:
                kept_so_far.append(group.representative)
            else:
                dependent.append(found)

    dropped_dependent = {entry.feature for entry in dependent}
    kept = {group.representative for group in groups} - dropped_dependent
    selected = [name for name in feature_columns if name in kept]
    report = SelectionReport(
        corr_threshold=corr_threshold,
        zero_variance=tuple(zero_variance),
        near_constant=tuple(near_constant),
        groups=tuple(groups),
        linear_dependent=tuple(dependent),
        selected=tuple(selected),
    )
    return selected, report


def _row_selector(df: pd.DataFrame, fit_rows: Iterable | None) -> pd.DataFrame:
    if fit_rows is None:
        return df
    fit_rows = list(fit_rows)
    if fit_rows and all(isinstance(value, (bool, np.bool_)) for value in fit_rows):
        if len(fit_rows) != len(df):
            raise ValueError("Boolean fit_rows mask must match the DataFrame length")
        return df[np.asarray(fit_rows, dtype=bool)]
    missing = [row for row in fit_rows if row not in df.index]
    if missing:
        raise KeyError(f"fit_rows not in DataFrame index: {missing}")
    return df.loc[fit_rows]


def fit_residualization(
    df: pd.DataFrame,
    columns: Sequence[str],
    covariate: str = DEFAULT_COVARIATE,
    fit_rows: Iterable | None = None,
) -> ResidualizationFit:
    """Least-squares fit of each column on ``covariate`` using ``fit_rows`` only.

    ``fit_rows`` is a list of index labels or a boolean mask (``None`` = all
    rows). With fewer than two distinct covariate values among the fit rows
    the slope is unidentifiable, so the fit is intercept-only (``degenerate``).
    """
    columns = [name for name in columns if name != covariate]
    fit_df = _row_selector(df, fit_rows)
    if fit_df.empty:
        raise ValueError("fit_rows selects no rows")
    x = fit_df[covariate].to_numpy(dtype=float)
    x_mean = float(x.mean())
    x_centered = x - x_mean
    sxx = float(np.dot(x_centered, x_centered))
    degenerate = len(np.unique(x)) < 2 or sxx <= _ZERO_VARIANCE_RTOL * max(1.0, x_mean**2)

    intercepts: dict[str, float] = {}
    slopes: dict[str, float] = {}
    for column in columns:
        y = fit_df[column].to_numpy(dtype=float)
        y_mean = float(y.mean())
        slope = 0.0 if degenerate else float(np.dot(x_centered, y - y_mean) / sxx)
        slopes[column] = slope
        intercepts[column] = y_mean - slope * x_mean
    return ResidualizationFit(
        covariate=covariate,
        intercepts=intercepts,
        slopes=slopes,
        degenerate=degenerate,
        covariate_min=float(x.min()),
        covariate_max=float(x.max()),
    )


def residualize_on(
    df: pd.DataFrame,
    columns: Sequence[str],
    covariate: str = DEFAULT_COVARIATE,
    fit_rows: Iterable | None = None,
) -> pd.DataFrame:
    """Residuals of ``columns`` after a linear fit on ``covariate``.

    The fit uses only ``fit_rows`` (e.g. a fold's training topologies); the
    residuals are returned for *every* row of ``df`` (same index), so held-out
    rows are transformed with the training fit. The covariate itself is
    excluded from the output. See ``fit_residualization`` for the degenerate
    (single covariate value) fallback.
    """
    return fit_residualization(df, columns, covariate, fit_rows).transform(df)


def build_feature_space(
    df: pd.DataFrame,
    variant: str,
    train_topologies: Sequence[str],
    held_out_topologies: Sequence[str],
    *,
    feature_columns: Sequence[str] | None = None,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    covariate: str = DEFAULT_COVARIATE,
    topology_column: str = "topology",
    preference: Sequence[str] = DEFAULT_PREFERENCE,
    min_minority_count: int = DEFAULT_MIN_MINORITY_COUNT,
    linear_dependence_r2: float | None = DEFAULT_LINEAR_DEPENDENCE_R2,
) -> FeatureSpace:
    """Build one fold's standardized train / held-out matrices for ``variant``.

    ``df`` has one row per topology (a ``topology_column`` column, or already
    indexed by topology). ``feature_columns`` defaults to every numeric column.
    Selection uses all rows of ``df``; standardization (and residualization for
    ``"size_controlled"``) is fit on ``train_topologies`` only.
    """
    if variant not in FEATURE_SPACES:
        raise ValueError(f"Unknown feature space {variant!r}; expected one of {FEATURE_SPACES}")
    table = df.set_index(topology_column) if topology_column in df.columns else df
    if not table.index.is_unique:
        raise ValueError("Topology index is not unique")
    if feature_columns is None:
        feature_columns = [
            name for name in table.columns if pd.api.types.is_numeric_dtype(table[name])
        ]
    train_topologies = list(train_topologies)
    held_out_topologies = list(held_out_topologies)
    overlap = set(train_topologies) & set(held_out_topologies)
    if overlap:
        raise ValueError(f"Topologies in both train and held-out: {sorted(overlap)}")
    if not train_topologies:
        raise ValueError("train_topologies is empty")

    threshold = corr_threshold if variant != "full" else np.inf
    columns, report = select_distance_features(
        table, feature_columns, corr_threshold=threshold, preference=preference,
        min_minority_count=min_minority_count, linear_dependence_r2=linear_dependence_r2,
    )

    residualization: ResidualizationFit | None = None
    extrapolated: tuple[str, ...] = ()
    if variant == "size_controlled":
        if covariate not in table.columns:
            raise KeyError(f"Covariate {covariate!r} not in DataFrame")
        columns = [name for name in columns if name != covariate]
        residualization = fit_residualization(table, columns, covariate, train_topologies)
        values = residualization.transform(table)
        flags = residualization.extrapolates(table.loc[held_out_topologies])
        extrapolated = tuple(name for name in held_out_topologies if flags[name])
    else:
        values = table[columns].astype(float)

    train_values = values.loc[train_topologies]
    center, scale = {}, {}
    fallback, dropped = [], []
    for column in columns:
        std = float(train_values[column].std(ddof=0))
        if _is_zero_variance(train_values[column].to_numpy()):
            std = float(values[column].std(ddof=0))
            if _is_zero_variance(values[column].to_numpy()):
                dropped.append(column)
                continue
            fallback.append(column)
        center[column] = float(train_values[column].mean())
        scale[column] = std
    kept = [column for column in columns if column not in dropped]

    def _standardize(rows: Sequence[str]) -> pd.DataFrame:
        block = values.loc[list(rows), kept]
        return (block - pd.Series(center)[kept]) / pd.Series(scale)[kept]

    return FeatureSpace(
        variant=variant,
        columns=tuple(kept),
        train=_standardize(train_topologies),
        held_out=_standardize(held_out_topologies),
        selection=report,
        residualization=residualization,
        center=center,
        scale=scale,
        fallback_scale_columns=tuple(fallback),
        dropped_constant_columns=tuple(dropped),
        extrapolated_topologies=extrapolated,
    )


def knn_distances(space: FeatureSpace, k: int) -> pd.DataFrame:
    """Mean Euclidean distance to the ``k`` nearest training topologies.

    Returns rows ``topology, role ("train" | "heldout"), knn_distance``;
    training rows exclude themselves from their own neighbor set.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    train = space.train.to_numpy(dtype=float)
    if len(train) <= k:
        raise ValueError(f"Need more than k={k} training topologies, got {len(train)}")
    rows = []
    for position, name in enumerate(space.train.index):
        distances = np.delete(np.linalg.norm(train - train[position], axis=1), position)
        rows.append({"topology": name, "role": "train",
                     "knn_distance": float(np.mean(np.sort(distances)[:k]))})
    for name, point in zip(space.held_out.index, space.held_out.to_numpy(dtype=float)):
        distances = np.linalg.norm(train - point, axis=1)
        rows.append({"topology": name, "role": "heldout",
                     "knn_distance": float(np.mean(np.sort(distances)[:k]))})
    return pd.DataFrame(rows, columns=["topology", "role", "knn_distance"])
