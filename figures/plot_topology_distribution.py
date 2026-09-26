"""Test whether held-out generalization tracks distance from the training distribution.

Inputs (each produced by its own script in this directory):

- ``topology_hand_features.csv`` (``topology_hand_features.py``)
- ``topology_gnn_embeddings/{fold}.npz`` (``topology_gnn_embeddings.py``), one
  file per fold, each from that fold's own checkpoint; folds without a file
  are skipped for the GNN space, never filled from another fold
- ``topology_performance_by_fold.csv`` (``topology_performance.py``), one row
  per (fold, seed, topology), with checkpoint selection on training
  topologies only and an untrained (step-0) baseline

Analysis unit. Every CV topology is held out in exactly one fold, so the
held-out analysis has exactly one row per topology (n = 10 for the hand
spaces, fewer for the GNN space). Training rows are never pooled into the
held-out correlation, and no topology is counted more than once.

Distance. For each fold and each feature space, k-NN distance is computed in
a space fit on that fold's training topologies only (standardization and, for
``size_controlled``, the n_nodes regression; see ``_feature_space.py``).
Because each fold has its own scale, distances are divided by the median
leave-one-out k-NN distance among that fold's training topologies
(``knn_distance_rel``): 1.0 means "as far from the training set as a typical
training topology is from the others." Only the relative distance is
compared across folds. GNN embeddings are used as-is (Euclidean in the raw
embedding space of the fold's own model), then normalized the same way.

Outcome. ``heldout_gain_m``: per seed, the held-out topology's eval distance
at the train-selected checkpoint minus that run's own step-0 distance for
the same topology, averaged over seeds. 0 = no better than untrained.

Statistics (per feature space):

- Spearman rho between ``knn_distance_rel`` and ``heldout_gain_m``.
- Permutation p-value (two-sided, outcomes permuted across topologies).
- 95% seed-bootstrap CI for rho: resample each fold's seeds with replacement,
  recompute every topology's gain, recompute rho.
- A signal check: how many held-out topologies beat their untrained noise
  floor at all. If none do, a distance/performance correlation is fitting
  noise, and the figures say so in their titles.

Topologies held out together in one fold share a training set, so the n
rows are not fully independent; with n = 10 treat every p-value as a screen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _feature_space
import _topology_catalog

HAND_FEATURES_CSV = FIGURES_ROOT / "topology_hand_features.csv"
GNN_EMBEDDINGS_DIR = FIGURES_ROOT / "topology_gnn_embeddings"
PERFORMANCE_CSV = FIGURES_ROOT / "topology_performance_by_fold.csv"

ID_COLUMNS = ("topology", "family")
HAND_SPACES = tuple(f"hand_{variant}" for variant in _feature_space.FEATURE_SPACES)
GNN_SPACE = "gnn"
PRIMARY_HAND_SPACE = "hand_deduplicated"
DEFAULT_K = 3
DEFAULT_N_PERMUTATIONS = 10_000
DEFAULT_N_BOOTSTRAP = 2_000
MIN_ROWS_FOR_STATS = 5
SIGNAL_Z = 2.0
TRAIN_COLOR = "#636EFA"
HELDOUT_COLOR = "#EF553B"


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def _feature_columns(hand_features: pd.DataFrame) -> list[str]:
    return [column for column in hand_features.columns if column not in ID_COLUMNS]


def _gnn_knn_distances(
    topologies: list[str], embeddings: np.ndarray, membership: dict, k: int
) -> pd.DataFrame:
    """k-NN distance in one fold's raw embedding space (train rows leave-one-out)."""
    index = {name: i for i, name in enumerate(topologies)}
    train_names = [name for name in membership["train"] if name in index]
    if len(train_names) <= k:
        raise ValueError(f"Need more than k={k} training topologies, got {len(train_names)}")
    train = embeddings[[index[name] for name in train_names]]
    rows = []
    for position, name in enumerate(train_names):
        distances = np.delete(np.linalg.norm(train - train[position], axis=1), position)
        rows.append({"topology": name, "role": "train", "knn_distance": float(np.sort(distances)[:k].mean())})
    for name in membership["held_out"]:
        distances = np.linalg.norm(train - embeddings[index[name]], axis=1)
        rows.append({"topology": name, "role": "heldout", "knn_distance": float(np.sort(distances)[:k].mean())})
    return pd.DataFrame(rows)


def _normalize_by_train_spacing(distances: pd.DataFrame) -> pd.DataFrame:
    """Divide by the median leave-one-out train k-NN distance of the same fold/space."""
    scale = distances[distances["role"] == "train"]["knn_distance"].median()
    return distances.assign(train_spacing=scale, knn_distance_rel=distances["knn_distance"] / scale)


def compute_distances(
    hand_features: pd.DataFrame,
    gnn_embeddings: dict[str, np.ndarray],
    fold_membership: dict,
    *,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """One row per (fold, space, topology in that fold): train LOO or held-out distance."""
    topologies = hand_features["topology"].tolist()
    feature_columns = _feature_columns(hand_features)
    frames = []
    for fold, membership in fold_membership.items():
        for variant in _feature_space.FEATURE_SPACES:
            space = _feature_space.build_feature_space(
                hand_features, variant, membership["train"], membership["held_out"],
                feature_columns=feature_columns,
            )
            distances = _feature_space.knn_distances(space, k)
            extrapolated = set(space.extrapolated_topologies)
            distances["extrapolated"] = distances["topology"].isin(extrapolated)
            frames.append(_normalize_by_train_spacing(distances).assign(fold=fold, space=f"hand_{variant}"))
        if fold in gnn_embeddings:
            distances = _gnn_knn_distances(topologies, gnn_embeddings[fold], membership, k)
            distances["extrapolated"] = False
            frames.append(_normalize_by_train_spacing(distances).assign(fold=fold, space=GNN_SPACE))
    columns = ["fold", "space", "topology", "role", "knn_distance", "train_spacing", "knn_distance_rel", "extrapolated"]
    return pd.concat(frames, ignore_index=True)[columns]


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def seed_gains(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Per (fold, seed, topology): distance at the train-selected checkpoint minus step 0."""
    return seed_results.assign(gain_m=seed_results["distance_m"] - seed_results["baseline_distance_m"])


def topology_performance(seed_results: pd.DataFrame) -> pd.DataFrame:
    """One row per (fold, topology, role) with seed-averaged performance and gain."""
    gains = seed_gains(seed_results)
    return (
        gains.groupby(["fold", "topology", "role"], as_index=False)
        .agg(
            distance_m=("distance_m", "mean"),
            distance_std_m=("distance_m", "std"),
            gain_m=("gain_m", "mean"),
            gain_std_m=("gain_m", "std"),
            n_seeds=("seed", "nunique"),
        )
        .fillna({"distance_std_m": 0.0, "gain_std_m": 0.0})
    )


def noise_floor(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Untrained (step-0) spread per topology, with seeds as the independent unit.

    Runs sharing a seed start from the same initialization, so the step-0
    evals are first averaged per seed, then summarized across seeds.
    """
    per_seed = seed_results.groupby(["topology", "seed"], as_index=False)["baseline_distance_m"].mean()
    return per_seed.groupby("topology", as_index=False).agg(
        floor_mean_m=("baseline_distance_m", "mean"),
        floor_std_m=("baseline_distance_m", "std"),
        n_floor_seeds=("seed", "nunique"),
    )


def heldout_table(distances: pd.DataFrame, performance: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """One row per (space, held-out topology): distance, gain, and noise-floor check."""
    heldout = distances[distances["role"] == "heldout"]
    table = heldout.merge(performance, on=["fold", "topology", "role"], how="left")
    table = table.merge(floor, on="topology", how="left")
    # Welch-style: held-out gain vs. the untrained seed-to-seed spread.
    standard_error = np.sqrt(
        table["floor_std_m"] ** 2 / table["n_floor_seeds"] + table["gain_std_m"] ** 2 / table["n_seeds"]
    )
    table["gain_z"] = table["gain_m"] / standard_error.replace(0.0, np.nan)
    table["above_floor"] = table["gain_z"] > SIGNAL_Z
    return table


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def permutation_p_value(x: np.ndarray, y: np.ndarray, *, n_permutations: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for Spearman rho (with the +1 correction)."""
    observed = _spearman(x, y)
    if np.isnan(observed):
        return float("nan")
    null = np.array([_spearman(x, rng.permutation(y)) for _ in range(n_permutations)])
    return float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_permutations + 1))


def seed_bootstrap_ci(
    table: pd.DataFrame,
    gains: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI for rho, resampling each fold's seeds with replacement.

    Distances are held fixed; only the held-out gains are re-estimated.
    """
    heldout_gains = gains[gains["role"] == "heldout"]
    by_fold = {
        fold: group.pivot(index="seed", columns="topology", values="gain_m")
        for fold, group in heldout_gains.groupby("fold")
    }
    topologies = table["topology"].tolist()
    x = table["knn_distance_rel"].to_numpy()
    rhos = []
    for _ in range(n_bootstrap):
        resampled = {}
        for fold, seed_by_topology in by_fold.items():
            rows = rng.integers(0, len(seed_by_topology), size=len(seed_by_topology))
            resampled.update(seed_by_topology.iloc[rows].mean(axis=0).to_dict())
        y = np.array([resampled[name] for name in topologies])
        rhos.append(_spearman(x, y))
    rhos = np.array(rhos)
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(rhos, [2.5, 97.5])
    return float(low), float(high)


def distance_statistics(
    heldout: pd.DataFrame,
    gains: pd.DataFrame,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-space Spearman rho, permutation p, and seed-bootstrap CI (held-out rows only)."""
    rows = []
    for space, table in heldout.groupby("space", sort=False):
        table = table.dropna(subset=["knn_distance_rel", "gain_m"])
        n = len(table)
        row = {
            "space": space,
            "n_heldout_topologies": n,
            "folds": ",".join(sorted(table["fold"].unique())),
            "n_above_floor": int(table["above_floor"].sum()),
            "spearman_rho": float("nan"),
            "permutation_p": float("nan"),
            "rho_ci_low": float("nan"),
            "rho_ci_high": float("nan"),
        }
        if n >= MIN_ROWS_FOR_STATS:
            rng = np.random.default_rng(seed)
            x = table["knn_distance_rel"].to_numpy()
            y = table["gain_m"].to_numpy()
            row["spearman_rho"] = _spearman(x, y)
            row["permutation_p"] = permutation_p_value(x, y, n_permutations=n_permutations, rng=rng)
            row["rho_ci_low"], row["rho_ci_high"] = seed_bootstrap_ci(
                table, gains, n_bootstrap=n_bootstrap, rng=rng
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _stats_label(stats_row: pd.Series) -> str:
    if np.isnan(stats_row["spearman_rho"]):
        return f"n={stats_row['n_heldout_topologies']} (< {MIN_ROWS_FOR_STATS}: no statistics)"
    return (
        f"ρ={stats_row['spearman_rho']:.2f} [{stats_row['rho_ci_low']:.2f}, {stats_row['rho_ci_high']:.2f}], "
        f"perm p={stats_row['permutation_p']:.3f}, n={stats_row['n_heldout_topologies']}"
    )


def _signal_warning(heldout: pd.DataFrame) -> str:
    primary = heldout[heldout["space"] == PRIMARY_HAND_SPACE]
    n_above = int(primary["above_floor"].sum())
    if n_above == 0:
        return "⚠ No held-out topology beats its untrained floor: any correlation here is fitting noise."
    return f"{n_above}/{len(primary)} held-out topologies beat their untrained floor (z > {SIGNAL_Z:g})."


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _layout(fig, **kwargs):
    fig.update_layout(font_family="Arial", plot_bgcolor="white", paper_bgcolor="white", **kwargs)
    return fig


def _projection(matrix: np.ndarray, method: str) -> np.ndarray:
    """2D layout for visualization only (fit on all topologies; no labels used)."""
    standardized = StandardScaler().fit_transform(matrix)
    if method == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(standardized)
    if method == "tsne":
        # n=13 is far below what t-SNE needs for a stable global layout;
        # perplexity is clamped down, and PCA remains the default.
        perplexity = max(2, min(5, (len(matrix) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0,
                    learning_rate="auto").fit_transform(standardized)
    raise ValueError(f"Unknown projection method: {method!r}")


def fold_projection_table(
    hand_features: pd.DataFrame,
    gnn_embeddings: dict[str, np.ndarray],
    performance: pd.DataFrame,
    fold: str,
    membership: dict,
    *,
    projection: str,
) -> pd.DataFrame:
    """One row per topology in ``fold`` (plus final-test ones), with 2D coordinates."""
    columns, _ = _feature_space.select_distance_features(hand_features, _feature_columns(hand_features))
    table = hand_features[list(ID_COLUMNS)].copy()
    hand_xy = _projection(hand_features[columns].to_numpy(dtype=float), projection)
    table["hand_x"], table["hand_y"] = hand_xy[:, 0], hand_xy[:, 1]
    if fold in gnn_embeddings:
        gnn_xy = _projection(gnn_embeddings[fold], projection)
        table["gnn_x"], table["gnn_y"] = gnn_xy[:, 0], gnn_xy[:, 1]

    roles = {name: "train" for name in membership["train"]}
    roles.update({name: "heldout" for name in membership["held_out"]})
    table["role"] = table["topology"].map(roles).fillna("final_test")
    fold_performance = performance[performance["fold"] == fold][["topology", "distance_m", "gain_m"]]
    return table.merge(fold_performance, on="topology", how="left").assign(fold=fold)


def make_projection_figure(table: pd.DataFrame, *, x_col: str, y_col: str, title: str, subtitle: str):
    """Train points colored by in-fold performance; held-out and final-test marked by symbol."""
    fig = go.Figure()
    train = table[table["role"] == "train"]
    fig.add_trace(go.Scatter(
        x=train[x_col], y=train[y_col], mode="markers+text", name="train",
        text=train["topology"], textposition="bottom center", textfont=dict(size=10),
        marker=dict(size=13, color=train["distance_m"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Train eval<br>distance (m)"), line=dict(width=1, color="DarkSlateGrey")),
        customdata=np.stack([train["distance_m"], train["gain_m"]], axis=1),
        hovertemplate="%{text}<br>distance %{customdata[0]:.2f} m<br>gain %{customdata[1]:.2f} m<extra>train</extra>",
    ))
    heldout = table[table["role"] == "heldout"]
    fig.add_trace(go.Scatter(
        x=heldout[x_col], y=heldout[y_col], mode="markers+text", name="held out",
        text=[f"{name}<br>gain {gain:+.2f} m" for name, gain in zip(heldout["topology"], heldout["gain_m"])],
        textposition="top center", textfont=dict(size=12, color="red"),
        marker=dict(symbol="star", size=24, color="white", line=dict(width=3, color="red")),
        hovertemplate="%{text}<extra>held out</extra>",
    ))
    final_test = table[table["role"] == "final_test"]
    fig.add_trace(go.Scatter(
        x=final_test[x_col], y=final_test[y_col], mode="markers+text", name="final test (not in CV)",
        text=final_test["topology"], textposition="bottom center", textfont=dict(size=10, color="grey"),
        marker=dict(symbol="x", size=11, color="grey"),
        hovertemplate="%{text}<extra>final test</extra>",
    ))
    for trace in fig.data:
        trace.cliponaxis = False
    return _layout(
        fig, title=f"{title}<br><sup>{subtitle}</sup>", width=950, height=780,
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        margin=dict(t=110, r=140),
    )


def make_distance_gain_figure(heldout: pd.DataFrame, stats: pd.DataFrame, *, warning: str):
    """One panel per feature space: held-out gain vs. relative k-NN distance."""
    spaces = [space for space in (*HAND_SPACES, GNN_SPACE) if space in set(heldout["space"])]
    stats_by_space = stats.set_index("space")
    fig = make_subplots(
        rows=1, cols=len(spaces), shared_yaxes=True, horizontal_spacing=0.04,
        subplot_titles=[f"{space}<br><sup>{_stats_label(stats_by_space.loc[space])}</sup>" for space in spaces],
    )
    for col, space in enumerate(spaces, start=1):
        table = heldout[heldout["space"] == space]
        fig.add_trace(go.Scatter(
            x=table["knn_distance_rel"], y=table["gain_m"], mode="markers",
            error_y=dict(type="data", array=table["gain_std_m"], visible=True),
            marker=dict(size=11, color=np.where(table["extrapolated"], "orange", HELDOUT_COLOR),
                        line=dict(width=1, color="black")),
            text=table["topology"] + " (" + table["fold"] + ")",
            hovertemplate="%{text}<br>rel. distance %{x:.2f}<br>gain %{y:.2f} m<extra></extra>",
            showlegend=False,
        ), row=1, col=col)
        # Shapes after the trace: plotly skips shapes on still-empty subplots.
        floor_band = table["floor_std_m"].median()
        fig.add_hrect(y0=-floor_band, y1=floor_band, fillcolor="lightgrey", opacity=0.4, line_width=0,
                      layer="below", row=1, col=col)
        fig.add_hline(y=0, line=dict(color="black", dash="dot"), row=1, col=col)
        fig.update_xaxes(title_text="k-NN distance / train spacing", row=1, col=col)
    fig.update_yaxes(title_text="Held-out gain over step 0 (m)", row=1, col=1)
    return _layout(
        fig, width=420 * len(spaces), height=560, margin=dict(t=140),
        title=("Held-out gain vs. distance from the fold's training set (one point per topology)"
               f"<br><sup>{warning} Grey band: median untrained seed spread. Orange: size-extrapolated.</sup>"),
    )


def make_per_feature_figure(heldout: pd.DataFrame, hand_features: pd.DataFrame, columns: list[str], *, warning: str):
    """Held-out gain vs. each selected hand feature (one point per held-out topology)."""
    primary = heldout[heldout["space"] == PRIMARY_HAND_SPACE][["topology", "gain_m", "gain_std_m"]]
    table = primary.merge(hand_features, on="topology")
    n_cols = 3
    n_rows = -(-len(columns) // n_cols)
    titles = []
    for feature in columns:
        rho = _spearman(table[feature].to_numpy(), table["gain_m"].to_numpy())
        titles.append(f"{feature} (ρ={rho:.2f})" if not np.isnan(rho) else feature)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, vertical_spacing=0.06)
    for i, feature in enumerate(columns):
        row, col = divmod(i, n_cols)
        fig.add_trace(go.Scatter(
            x=table[feature], y=table["gain_m"], mode="markers",
            error_y=dict(type="data", array=table["gain_std_m"], visible=True),
            marker=dict(color=HELDOUT_COLOR), text=table["topology"], showlegend=False,
            hovertemplate="%{text}<br>%{x}<br>gain %{y:.2f} m<extra></extra>",
        ), row=row + 1, col=col + 1)
    return _layout(
        fig, width=1100, height=300 * n_rows + 120,
        title=("Held-out gain vs. individual (deduplicated) hand features, n=10, uncorrected"
               f"<br><sup>{warning}</sup>"),
    )


def make_confound_figure(distances: pd.DataFrame, performance: pd.DataFrame, *, n_permutations: int, seed: int):
    """Training topologies: in-training performance vs. leave-one-out distance.

    Each topology is a training topology in 4 folds; its distance and
    performance are averaged over those folds so it appears once.
    """
    train = distances[(distances["role"] == "train") & (distances["space"] == PRIMARY_HAND_SPACE)]
    train = train.merge(performance, on=["fold", "topology", "role"], how="left")
    per_topology = train.groupby("topology", as_index=False).agg(
        knn_distance_rel=("knn_distance_rel", "mean"), distance_m=("distance_m", "mean"),
    )
    x = per_topology["knn_distance_rel"].to_numpy()
    y = per_topology["distance_m"].to_numpy()
    rho = _spearman(x, y)
    p_value = permutation_p_value(x, y, n_permutations=n_permutations, rng=np.random.default_rng(seed))
    fig = px.scatter(
        per_topology, x="knn_distance_rel", y="distance_m", text="topology",
        labels={"knn_distance_rel": "Leave-one-out k-NN distance / train spacing (mean over folds)",
                "distance_m": "In-training eval distance (m, mean over folds and seeds)"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=11, color=TRAIN_COLOR))
    return _layout(
        fig, width=850, height=650,
        title=("Confound check: do atypical training topologies also train worse?"
               f"<br><sup>{PRIMARY_HAND_SPACE}; ρ={rho:.2f}, perm p={p_value:.3f}, n={len(per_topology)}</sup>"),
    )


def _write_figure(fig, output_dir: Path, name: str) -> None:
    fig.write_image(output_dir / f"{name}.png", scale=3)
    fig.write_html(output_dir / f"{name}.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand-features-csv", type=Path, default=HAND_FEATURES_CSV)
    parser.add_argument("--gnn-embeddings-dir", type=Path, default=GNN_EMBEDDINGS_DIR)
    parser.add_argument("--performance-csv", type=Path, default=PERFORMANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--projection", choices=["pca", "tsne"], default="pca",
        help="2D layout for the per-fold distribution scatter plots (PCA is the safer choice at n=13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_membership = _topology_catalog.fold_membership()

    hand_features = pd.read_csv(args.hand_features_csv)
    topologies = hand_features["topology"].tolist()
    seed_results = pd.read_csv(args.performance_csv)
    gnn_embeddings = (
        _load_fold_embeddings(args.gnn_embeddings_dir, topologies) if args.gnn_embeddings_dir.exists() else {}
    )
    missing_gnn = sorted(set(fold_membership) - set(gnn_embeddings))
    if missing_gnn:
        print(f"No GNN embeddings for folds {missing_gnn}; the GNN space covers {sorted(gnn_embeddings)} only.")

    distances = compute_distances(hand_features, gnn_embeddings, fold_membership, k=args.k)
    performance = topology_performance(seed_results)
    gains = seed_gains(seed_results)
    floor = noise_floor(seed_results)
    heldout = heldout_table(distances, performance, floor)
    stats = distance_statistics(
        heldout, gains, n_permutations=args.n_permutations, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    warning = _signal_warning(heldout)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    distances.merge(performance, on=["fold", "topology", "role"], how="left").to_csv(
        output_dir / "topology_distribution_joined.csv", index=False
    )
    heldout.to_csv(output_dir / "topology_heldout_distance_analysis.csv", index=False)
    stats.to_csv(output_dir / "topology_distance_stats.csv", index=False)
    selected, report = _feature_space.select_distance_features(hand_features, _feature_columns(hand_features))
    report.to_frame().to_csv(output_dir / "topology_feature_selection.csv", index=False)

    _write_figure(make_distance_gain_figure(heldout, stats, warning=warning), output_dir,
                  "topology_heldout_gain_vs_distance")
    _write_figure(make_per_feature_figure(heldout, hand_features, selected, warning=warning), output_dir,
                  "topology_heldout_gain_vs_individual_features")
    _write_figure(make_confound_figure(distances, performance, n_permutations=args.n_permutations, seed=args.seed),
                  output_dir, "topology_confound_training_distance")

    method_label = "PCA" if args.projection == "pca" else "t-SNE"
    for fold, membership in fold_membership.items():
        fold_dir = output_dir / fold
        fold_dir.mkdir(exist_ok=True)
        table = fold_projection_table(
            hand_features, gnn_embeddings, performance, fold, membership, projection=args.projection,
        )
        table.to_csv(fold_dir / "topology_projection_table.csv", index=False)
        subtitle = f"fold {fold}: held out {', '.join(membership['held_out'])}<br>{warning}"
        _write_figure(
            make_projection_figure(table, x_col="hand_x", y_col="hand_y",
                                   title=f"Hand-feature {method_label} (deduplicated features)", subtitle=subtitle),
            fold_dir, "topology_projection_hand_features",
        )
        if fold in gnn_embeddings:
            _write_figure(
                make_projection_figure(table, x_col="gnn_x", y_col="gnn_y",
                                       title=f"GNN-embedding {method_label} ({fold}'s own checkpoint)",
                                       subtitle=subtitle),
                fold_dir, "topology_projection_gnn_embeddings",
            )

    print(warning)
    print(stats.to_string(index=False))
    print(f"Wrote plots and CSVs to {output_dir}")


def _load_fold_embeddings(directory: Path, topologies: list[str]) -> dict[str, np.ndarray]:
    # Imported lazily: topology_gnn_embeddings pulls in torch/Hydra/the env stack.
    from topology_gnn_embeddings import load_fold_embeddings

    return load_fold_embeddings(directory, topologies)


if __name__ == "__main__":
    main()
