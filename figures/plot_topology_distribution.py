"""Plot the training-topology distribution and held-out performance vs. distance.

Combines the outputs of ``topology_hand_features.py``,
``topology_gnn_embeddings.py`` (optional -- skipped if not yet generated),
and ``topology_performance.py`` into:

- Plot A: hand-feature PCA scatter (train vs. held-out, colored by performance)
- Plot B: GNN-embedding PCA scatter (same), if embeddings are available
- Plot C: performance vs. hand-feature k-NN distance
- Plot D: performance vs. GNN-embedding k-NN distance, if embeddings are available
- Bonus: per-hand-feature small multiples, and a distance-vs-training-performance
  confound-check panel

k-NN distance is computed in the standardized full-dimensional feature space
(never in the 2D PCA projection), per fold: a held-out topology's distance is
to its own fold's training set only, and a training topology gets a
leave-one-out distance against the rest of that fold's training set, since
different folds train on different topology subsets.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _topology_catalog

HAND_FEATURES_CSV = Path(__file__).with_name("topology_hand_features.csv")
GNN_EMBEDDINGS_NPZ = Path(__file__).with_name("topology_gnn_embeddings.npz")
PERFORMANCE_CSV = Path(__file__).with_name("topology_performance_by_fold.csv")

NON_FEATURE_COLUMNS = {"topology", "family"}
DEFAULT_K = 3


def _pca_projection(matrix: np.ndarray) -> np.ndarray:
    standardized = StandardScaler().fit_transform(matrix)
    return PCA(n_components=2, random_state=0).fit_transform(standardized)


def _tsne_projection(matrix: np.ndarray) -> np.ndarray:
    """t-SNE projection, with perplexity clamped down for tiny sample counts.

    At n=13 topologies this is on shaky ground: t-SNE is built to preserve
    local neighborhoods within a perplexity-sized crowd, and needs many more
    points than that to produce a *stable* global layout -- distances between
    far-apart clusters aren't meaningful, and results can shift noticeably
    with the random seed. Included because it was asked for, not because
    it's the recommended default here; PCA remains that (see the module
    docstring / --projection help).
    """
    standardized = StandardScaler().fit_transform(matrix)
    perplexity = max(2, min(5, (len(matrix) - 1) // 3))
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=0, learning_rate="auto",
    ).fit_transform(standardized)


def _projection(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        return _pca_projection(matrix)
    if method == "tsne":
        return _tsne_projection(matrix)
    raise ValueError(f"Unknown projection method: {method!r}")


def _knn_distance_per_fold(
    topologies: list[str], feature_matrix: np.ndarray, fold_membership: dict, k: int = DEFAULT_K
) -> pd.DataFrame:
    """Fold-conditional k-NN distance: standardize and fit neighbors on each
    fold's training topologies only, so held-out topologies never leak into
    the scale/neighbor set used to judge how far they are."""
    index_by_topology = {name: i for i, name in enumerate(topologies)}
    rows = []
    for fold, membership in fold_membership.items():
        train_names = [name for name in membership["train"] if name in index_by_topology]
        if len(train_names) <= k:
            continue
        train_idx = [index_by_topology[name] for name in train_names]
        scaler = StandardScaler().fit(feature_matrix[train_idx])
        train_scaled = scaler.transform(feature_matrix[train_idx])

        def _knn_distance(point: np.ndarray, exclude_self_idx: int | None) -> float:
            distances = np.linalg.norm(train_scaled - point, axis=1)
            if exclude_self_idx is not None:
                distances = np.delete(distances, exclude_self_idx)
            return float(np.mean(np.sort(distances)[:k]))

        for local_idx, name in enumerate(train_names):
            point = train_scaled[local_idx]
            rows.append(
                {
                    "fold": fold,
                    "topology": name,
                    "role": "train",
                    "knn_distance": _knn_distance(point, local_idx),
                }
            )
        for name in membership["held_out"]:
            if name not in index_by_topology:
                continue
            point = scaler.transform(feature_matrix[[index_by_topology[name]]])[0]
            rows.append(
                {
                    "fold": fold,
                    "topology": name,
                    "role": "heldout",
                    "knn_distance": _knn_distance(point, None),
                }
            )
    return pd.DataFrame(rows)


def _load_gnn_embeddings(path: Path, topologies: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    archive = np.load(path)
    missing = [name for name in topologies if name not in archive]
    if missing:
        raise ValueError(f"{path} is missing topologies: {missing}")
    return np.stack([archive[name] for name in topologies])


def _load_performance(path: Path) -> pd.DataFrame:
    seed_results = pd.read_csv(path)
    return (
        seed_results.groupby(["fold", "topology", "role"], as_index=False)
        .agg(distance_m=("distance_m", "mean"), distance_std_m=("distance_m", "std"), n_seeds=("seed", "nunique"))
        .fillna({"distance_std_m": 0.0})
    )


def build_joined_table(
    hand_features: pd.DataFrame,
    gnn_embeddings: np.ndarray | None,
    performance: pd.DataFrame,
    fold_membership: dict,
    *,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    topologies = hand_features["topology"].tolist()
    feature_matrix = hand_features.drop(columns=list(NON_FEATURE_COLUMNS)).to_numpy(dtype=float)

    hand_knn = _knn_distance_per_fold(topologies, feature_matrix, fold_membership, k=k)
    hand_knn = hand_knn.rename(columns={"knn_distance": "hand_feature_knn_distance"})

    if gnn_embeddings is not None:
        gnn_knn = _knn_distance_per_fold(topologies, gnn_embeddings, fold_membership, k=k)
        gnn_knn = gnn_knn.rename(columns={"knn_distance": "gnn_embedding_knn_distance"})
    else:
        gnn_knn = pd.DataFrame(columns=["fold", "topology", "role", "gnn_embedding_knn_distance"])

    joined = performance.merge(hand_knn, on=["fold", "topology", "role"], how="left")
    joined = joined.merge(gnn_knn, on=["fold", "topology", "role"], how="left")
    joined = joined.merge(hand_features, on="topology", how="left")
    return joined


def _reference_fold_performance(
    joined: pd.DataFrame, fold_membership: dict, reference_fold: str
) -> pd.DataFrame:
    """One performance value per topology, for the PCA plots.

    Pooling "held out in *any* fold" doesn't work here: node_count_loso runs
    5 folds, each holding out a different node-count group, so across all 5
    folds combined every topology is held out in exactly one of them -- that
    made the old version of this plot mark literally everything "heldout,"
    which is technically true per-fold but not what a single checkpoint's
    train/held-out split actually looks like. A topology's PCA position
    doesn't change across folds anyway, so plotting every (fold, topology)
    row would draw overlapping markers at the same point regardless. Instead
    this picks ONE reference fold -- the one whose held-out group matches
    the checkpoint that produced the GNN embeddings -- and reports each
    topology's role and performance within just that fold.
    """
    membership = fold_membership[reference_fold]
    role_by_topology = {name: "heldout" for name in membership["held_out"]}
    role_by_topology.update({name: "train" for name in membership["train"]})

    fold_rows = joined[joined["fold"] == reference_fold]
    performance_by_topology = dict(zip(fold_rows["topology"], fold_rows["distance_m"]))

    rows = [
        {
            "topology": topology,
            "role": role,
            "fold": reference_fold,
            "distance_m": performance_by_topology.get(topology, float("nan")),
        }
        for topology, role in role_by_topology.items()
    ]
    return pd.DataFrame(rows)


def build_pca_table(
    hand_features: pd.DataFrame,
    gnn_embeddings: np.ndarray | None,
    joined: pd.DataFrame,
    fold_membership: dict,
    *,
    reference_fold: str,
    projection: str = "pca",
) -> pd.DataFrame:
    """One row per topology (2D projection coordinates + performance within
    ``reference_fold``, the single fold whose held-out group the plot's
    train/held-out split reflects)."""
    topologies = hand_features["topology"].tolist()
    feature_matrix = hand_features.drop(columns=list(NON_FEATURE_COLUMNS)).to_numpy(dtype=float)
    hand_xy = _projection(feature_matrix, projection)

    pca_df = hand_features[["topology", "family"]].copy()
    pca_df["hand_pca_x"] = hand_xy[:, 0]
    pca_df["hand_pca_y"] = hand_xy[:, 1]
    if gnn_embeddings is not None:
        gnn_xy = _projection(gnn_embeddings, projection)
        pca_df["gnn_pca_x"] = gnn_xy[:, 0]
        pca_df["gnn_pca_y"] = gnn_xy[:, 1]

    reference_performance = _reference_fold_performance(joined, fold_membership, reference_fold)
    return reference_performance.merge(pca_df, on="topology", how="left")


def make_pca_figure(pca_table: pd.DataFrame, *, x_col: str, y_col: str, title: str, subtitle: str = ""):
    # Held-out points need to read as held-out at a glance, not just on hover:
    # a big red-ringed star with its name labeled beats a same-size diamond
    # in the same color scale as everything else. Every point gets a label
    # now (not just held-out), so train labels sit *below* their marker and
    # held-out labels sit *above*, to cut down on the two sets colliding.
    plot_table = pca_table.copy()
    plot_table["label"] = plot_table["topology"]

    fig = px.scatter(
        plot_table,
        x=x_col,
        y=y_col,
        color="distance_m",
        symbol="role",
        symbol_map={"train": "circle", "heldout": "star"},
        text="label",
        hover_data=["topology", "fold", "family", "distance_m"],
        color_continuous_scale="Viridis",
        title=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
        labels={"distance_m": "Eval distance (m)"},
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
    fig.for_each_trace(
        lambda trace: trace.update(
            marker=dict(size=26, line=dict(width=3, color="red")),
            textposition="top center",
            textfont=dict(size=13, color="red", family="Arial Black"),
            cliponaxis=False,
        )
        if trace.name == "heldout"
        else trace.update(
            textposition="bottom center",
            textfont=dict(size=11, color="black"),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        font_family="Arial", plot_bgcolor="white", paper_bgcolor="white",
        width=900, height=750,
        # The discrete "role" legend and the continuous distance colorbar
        # both default to the right edge and overlap; push the legend below
        # the plot instead of stacking them.
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        coloraxis_colorbar=dict(x=1.02),
        margin=dict(r=120, t=110),
    )
    return fig


def _add_trend_and_spearman(fig, x: np.ndarray, y: np.ndarray, *, title: str):
    """Annotate a scatter with a linear fit line and Spearman r (no statsmodels dependency)."""
    from scipy.stats import spearmanr

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 2)
        fig.add_trace(
            go.Scatter(
                x=x_line, y=slope * x_line + intercept, mode="lines",
                line=dict(color="black", dash="dash"), name="linear fit",
            )
        )
        correlation, p_value = spearmanr(x, y)
        title = f"{title}<br><sup>Spearman r={correlation:.2f}, p={p_value:.3f}, n={len(x)}</sup>"
    fig.update_layout(title=title)
    return fig


def make_distance_performance_figure(joined: pd.DataFrame, *, distance_col: str, title: str):
    plot_data = joined.dropna(subset=[distance_col, "distance_m"])
    fig = px.scatter(
        plot_data,
        x=distance_col,
        y="distance_m",
        color="role",
        color_discrete_map={"train": "#636EFA", "heldout": "#EF553B"},
        error_y="distance_std_m",
        hover_data=["topology", "fold"],
        labels={distance_col: "k-NN distance (standardized feature space)", "distance_m": "Eval distance (m)"},
    )
    fig.for_each_trace(
        lambda trace: trace.update(marker=dict(size=13, line=dict(width=2, color="black")))
        if trace.name == "heldout"
        else trace.update(marker=dict(size=8))
    )
    _add_trend_and_spearman(
        fig, plot_data[distance_col].to_numpy(), plot_data["distance_m"].to_numpy(), title=title
    )
    fig.update_layout(font_family="Arial", plot_bgcolor="white", paper_bgcolor="white", width=800, height=650)
    return fig


def make_per_feature_small_multiples(joined: pd.DataFrame, feature_columns: list[str]):
    n_cols = 3
    n_rows = -(-len(feature_columns) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feature_columns)
    for i, feature in enumerate(feature_columns):
        row, col = divmod(i, n_cols)
        for role, color in (("train", "#636EFA"), ("heldout", "#EF553B")):
            subset = joined[joined["role"] == role]
            fig.add_trace(
                go.Scatter(
                    x=subset[feature], y=subset["distance_m"], mode="markers",
                    name=role, marker=dict(color=color), showlegend=(i == 0),
                ),
                row=row + 1, col=col + 1,
            )
    fig.update_layout(
        font_family="Arial", plot_bgcolor="white", paper_bgcolor="white",
        width=1100, height=350 * n_rows, title="Performance vs. individual hand-crafted features",
    )
    return fig


def make_confound_figure(joined: pd.DataFrame, *, distance_col: str):
    train_only = joined[joined["role"] == "train"].dropna(subset=[distance_col, "distance_m"])
    fig = px.scatter(
        train_only,
        x=distance_col,
        y="distance_m",
        hover_data=["topology", "fold"],
        labels={distance_col: "Leave-one-out k-NN distance", "distance_m": "Eval distance (m)"},
    )
    _add_trend_and_spearman(
        fig, train_only[distance_col].to_numpy(), train_only["distance_m"].to_numpy(),
        title="Confound check: training-topology performance vs. leave-one-out distance",
    )
    fig.update_layout(font_family="Arial", plot_bgcolor="white", paper_bgcolor="white", width=800, height=650)
    return fig


def _write_figure(fig, output_dir: Path, name: str) -> None:
    fig.write_image(output_dir / f"{name}.png", scale=3)
    fig.write_html(output_dir / f"{name}.html", include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-features-csv", type=Path, default=HAND_FEATURES_CSV)
    parser.add_argument("--gnn-embeddings-npz", type=Path, default=GNN_EMBEDDINGS_NPZ)
    parser.add_argument("--performance-csv", type=Path, default=PERFORMANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--reference-fold", default="node_8",
        help=(
            "Which node_count_loso fold's held-out group the PCA plots' train/held-out "
            "split reflects -- pick the fold matching whatever checkpoint produced "
            "--gnn-embeddings-npz (default node_8, matching this repo's current checkpoint). "
            "Every topology is held out in exactly one of the 5 folds, so without picking "
            "one fold as the reference, pooling across all of them marks everything "
            "'heldout'. Run with --list-folds to see each fold's held-out group."
        ),
    )
    parser.add_argument(
        "--list-folds", action="store_true",
        help="Print each node_count_loso fold's held-out group and exit.",
    )
    parser.add_argument(
        "--projection", choices=["pca", "tsne"], default="pca",
        help=(
            "2D layout for the distribution scatter plots. PCA (default) is the safer "
            "choice at n=13 topologies; t-SNE's perplexity is auto-clamped down for this "
            "sample size but its global layout is still less trustworthy here than PCA's."
        ),
    )
    parser.add_argument(
        "--gnn-run-url", default="https://wandb.ai/i-suds/CV/runs/w5sigp4m",
        help="W&B run URL for the checkpoint used to produce --gnn-embeddings-npz, printed "
        "as a subtitle on the GNN-embedding plot so it's traceable back to its source run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_membership = _topology_catalog.fold_membership()

    if args.list_folds:
        for fold, membership in fold_membership.items():
            print(f"{fold}: held_out={membership['held_out']}")
        return

    if args.reference_fold not in fold_membership:
        raise ValueError(
            f"--reference-fold {args.reference_fold!r} is not a node_count_loso fold; "
            f"choices are {sorted(fold_membership)} (see --list-folds)."
        )
    print(
        f"Using fold {args.reference_fold!r} as the PCA plots' train/held-out reference "
        f"-- held out: {fold_membership[args.reference_fold]['held_out']}"
    )

    hand_features = pd.read_csv(args.hand_features_csv)
    performance = _load_performance(args.performance_csv)
    gnn_embeddings = _load_gnn_embeddings(args.gnn_embeddings_npz, hand_features["topology"].tolist())

    joined = build_joined_table(hand_features, gnn_embeddings, performance, fold_membership, k=args.k)
    pca_table = build_pca_table(
        hand_features, gnn_embeddings, joined, fold_membership,
        reference_fold=args.reference_fold, projection=args.projection,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "topology_distribution_joined.csv", index=False)
    pca_table.to_csv(args.output_dir / "topology_pca_table.csv", index=False)

    method_label = args.projection.upper() if args.projection == "pca" else "t-SNE"
    _write_figure(
        make_pca_figure(
            pca_table, x_col="hand_pca_x", y_col="hand_pca_y", title=f"Hand-feature {method_label}",
        ),
        args.output_dir, "topology_pca_hand_features",
    )
    _write_figure(
        make_distance_performance_figure(
            joined, distance_col="hand_feature_knn_distance",
            title="Performance vs. hand-feature k-NN distance",
        ),
        args.output_dir, "topology_performance_vs_hand_feature_distance",
    )
    feature_columns = [c for c in hand_features.columns if c not in NON_FEATURE_COLUMNS]
    _write_figure(
        make_per_feature_small_multiples(joined, feature_columns),
        args.output_dir, "topology_performance_vs_individual_features",
    )
    _write_figure(
        make_confound_figure(joined, distance_col="hand_feature_knn_distance"),
        args.output_dir, "topology_confound_training_distance",
    )

    if gnn_embeddings is not None:
        _write_figure(
            make_pca_figure(
                pca_table, x_col="gnn_pca_x", y_col="gnn_pca_y", title=f"GNN-embedding {method_label}",
                subtitle=f"checkpoint run: {args.gnn_run_url}",
            ),
            args.output_dir, "topology_pca_gnn_embeddings",
        )
        _write_figure(
            make_distance_performance_figure(
                joined, distance_col="gnn_embedding_knn_distance",
                title="Performance vs. GNN-embedding k-NN distance",
            ),
            args.output_dir, "topology_performance_vs_gnn_embedding_distance",
        )
    else:
        print(
            f"{args.gnn_embeddings_npz} not found -- skipping GNN-embedding plots. "
            "Run topology_gnn_embeddings.py with a checkpoint reference to generate them."
        )

    print(f"Wrote plots and joined CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
