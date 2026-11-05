"""Download paper evaluation results from W&B and plot distance by topology."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd
import wandb


TOPOLOGIES = [
    "henneberg_n5_1tube_1",
    "henneberg_n6_1tube_2",
    "henneberg_n7_1tube_3",
    "henneberg_n8_1tube_57",
    "henneberg_n6_2tube_1",
    "henneberg_n8_2tube_187",
    "tetrahedron",
    "henneberg_n7_3tube_1",
    "henneberg_n8_3tube_64",
    "usevitch_60243677150_p1",
    "usevitch_210272254_p1",
    "usevitch_1514879",
    "octahedron",
]
METRIC = "eval/episode_distance"
HISTORY_SAMPLES = 10_000
DEFAULT_GREEDY_METRICS = Path(__file__).with_name("greedy_metrics.csv")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def collect_seed_results(runs: Iterable[Any], run_prefix: str) -> pd.DataFrame:
    """Return seed values at each topology's peak across-seed mean eval step."""
    candidates: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        topology = config.get("truss_topology")
        seed = config.get("seed")
        if topology not in TOPOLOGIES or seed is None:
            continue
        if run_prefix and not str(run.name).startswith(run_prefix):
            continue
        candidates.append(
            {
                "topology": topology,
                "seed": int(seed),
                "run": run,
                "run_id": run.id,
                "run_name": run.name,
                "run_state": run.state,
                "logged_step": int(run.summary.get("eval/step", run.summary.get("_step", -1))),
                "run_url": run.url,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No runs matched prefix {run_prefix!r} for a configured topology and seed."
        )

    selected_runs = (
        pd.DataFrame(candidates)
        .sort_values(["topology", "seed", "logged_step", "run_id"])
        .drop_duplicates(["topology", "seed"], keep="last")
        .reset_index(drop=True)
    )

    history_rows: list[dict[str, Any]] = []
    for candidate in selected_runs.to_dict("records"):
        run = candidate.pop("run")
        # Use environment steps to align seeds even after logging resumes.
        # At 20k evaluation intervals these 10M-step runs are below the sample cap.
        evaluation_history = run.history(
            keys=[METRIC, "eval/step"], samples=HISTORY_SAMPLES, pandas=False
        )
        if len(evaluation_history) >= HISTORY_SAMPLES:
            raise RuntimeError(f"Run {run.id} reached the history sample cap; retrieve exact history.")
        for history_row in evaluation_history:
            step = _finite_float(history_row.get("eval/step", history_row.get("_step")))
            distance = _finite_float(history_row.get(METRIC))
            if step is None or distance is None:
                continue
            history_rows.append(
                {
                    **candidate,
                    "evaluation_step": int(step),
                    "distance_m": distance,
                }
            )

    if not history_rows:
        raise RuntimeError(f"Matched runs have no finite {METRIC!r} history rows.")

    history = (
        pd.DataFrame(history_rows)
        .sort_values(["topology", "seed", "evaluation_step"])
        .drop_duplicates(["topology", "seed", "evaluation_step"], keep="last")
    )
    selected_rows = []
    for topology, topology_history in history.groupby("topology", sort=False):
        seed_curves = topology_history.pivot(
            index="evaluation_step", columns="seed", values="distance_m"
        ).dropna()
        if seed_curves.empty:
            raise RuntimeError(
                f"Topology {topology!r} has no evaluation step shared by every seed."
            )
        step_means = seed_curves.mean(axis=1)
        peak_mean = step_means.max()
        selected_step = int(step_means[step_means == peak_mean].index.max())
        rows = topology_history[
            topology_history["evaluation_step"] == selected_step
        ].copy()
        rows["selection_mean_distance_m"] = peak_mean
        selected_rows.append(rows)

    return pd.concat(selected_rows, ignore_index=True).drop(columns="logged_step")


def collect_expert_results(runs: Iterable[Any]) -> pd.DataFrame:
    """Select matched-seed peaks independently for each expert architecture.

    Retain interrupted runs' recorded evaluations rather than silently dropping
    seeds. Exclude the superseded paper-v2 run, which has no evaluation metric.
    """
    families = {"GNN specialist": [], "MLP specialist": []}
    for run in runs:
        if str(run.name).startswith(("paper-v4-", "paper-expert-gnn-")):
            controller = "GNN specialist"
        elif str(run.name).startswith("paper-expert-mlp-"):
            controller = "MLP specialist"
        else:
            continue
        families[controller].append(run)
    results = []
    for controller, family in families.items():
        rows = collect_seed_results(family, "")
        for topology in TOPOLOGIES:
            seeds = set(rows.loc[rows.topology == topology, "seed"])
            if seeds != {1, 2, 3}:
                raise RuntimeError(f"{controller}, {topology}: expected seeds 1,2,3; got {seeds}")
        results.append(rows.assign(controller=controller))
    return pd.concat(results, ignore_index=True)


def aggregate_experts(seed_results: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        aggregate_results(rows).assign(controller=controller)
        for controller, rows in seed_results.groupby("controller", sort=False)
    ], ignore_index=True)


def aggregate_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute across-seed mean and sample standard deviation per topology."""
    aggregate = (
        seed_results.groupby("topology", as_index=False)
        .agg(
            distance_m=("distance_m", "mean"),
            distance_std_m=("distance_m", "std"),
            n_seeds=("seed", "nunique"),
            selected_step=("evaluation_step", "first"),
        )
        .fillna({"distance_std_m": 0.0})
    )
    aggregate["topology"] = pd.Categorical(
        aggregate["topology"], categories=TOPOLOGIES, ordered=True
    )
    return aggregate.sort_values("topology").reset_index(drop=True)


def load_greedy_results(path: Path = DEFAULT_GREEDY_METRICS) -> pd.DataFrame:
    """Load and validate measured greedy-controller distance summaries."""
    greedy = pd.read_csv(path)
    required_columns = {"topology", "distance_m", "distance_std_m"}
    missing_columns = required_columns - set(greedy.columns)
    if missing_columns:
        raise ValueError(
            f"Greedy metrics are missing columns: {', '.join(sorted(missing_columns))}"
        )

    duplicate_topologies = sorted(
        greedy.loc[greedy["topology"].duplicated(), "topology"].astype(str).unique()
    )
    if duplicate_topologies:
        raise ValueError(
            f"Greedy metrics contain duplicate topologies: "
            f"{', '.join(duplicate_topologies)}"
        )

    available_topologies = set(greedy["topology"])
    missing_topologies = [name for name in TOPOLOGIES if name not in available_topologies]
    extra_topologies = sorted(available_topologies - set(TOPOLOGIES))
    if missing_topologies or extra_topologies:
        details = []
        if missing_topologies:
            details.append(f"missing: {', '.join(missing_topologies)}")
        if extra_topologies:
            details.append(f"unexpected: {', '.join(extra_topologies)}")
        raise ValueError(f"Greedy metrics topology mismatch ({'; '.join(details)})")

    for column in ("distance_m", "distance_std_m"):
        greedy[column] = pd.to_numeric(greedy[column], errors="coerce")
        invalid = ~greedy[column].map(lambda value: math.isfinite(float(value)))
        if invalid.any():
            names = ", ".join(greedy.loc[invalid, "topology"].astype(str))
            raise ValueError(f"Greedy metrics contain invalid {column} for: {names}")
    if (greedy["distance_std_m"] < 0).any():
        names = ", ".join(
            greedy.loc[greedy["distance_std_m"] < 0, "topology"].astype(str)
        )
        raise ValueError(f"Greedy metrics contain negative distance_std_m for: {names}")

    greedy["topology"] = pd.Categorical(
        greedy["topology"], categories=TOPOLOGIES, ordered=True
    )
    return greedy.sort_values("topology").reset_index(drop=True)


LABELS = {
    "henneberg_n5_1tube_1": "5 nodes / 1 tube",
    "henneberg_n6_1tube_2": "6 nodes / 1 tube",
    "henneberg_n7_1tube_3": "7 nodes / 1 tube",
    "henneberg_n8_1tube_57": "8 nodes / 1 tube",
    "henneberg_n6_2tube_1": "6 nodes / 2 tubes",
    "henneberg_n8_2tube_187": "8 nodes / 2 tubes",
    "tetrahedron": "Tetrahedron",
    "henneberg_n7_3tube_1": "7 nodes / 3 tubes",
    "henneberg_n8_3tube_64": "8 nodes / 3 tubes",
    "usevitch_60243677150_p1": "9 nodes / multi-triangle",
    "usevitch_210272254_p1": "8 nodes / multi-triangle",
    "usevitch_1514879": "7 nodes / multi-triangle",
    "octahedron": "Octahedron",
}
STYLES = {
    "GNN specialist": ("#0072B2", "o", "circle", -0.24),
    "MLP specialist": ("#D55E00", "s", "square", 0.0),
    "Greedy": ("#444444", "D", "diamond", 0.24),
}


def plot_rows(aggregate: pd.DataFrame, greedy_metrics: pd.DataFrame | None = None):
    learned = aggregate.copy()
    if "controller" not in learned:
        learned["controller"] = "GNN specialist"
    greedy = load_greedy_results() if greedy_metrics is None else greedy_metrics.copy()
    return pd.concat([learned, greedy.assign(controller="Greedy")], ignore_index=True)


def make_figure(aggregate: pd.DataFrame, greedy_metrics: pd.DataFrame | None = None):
    """Interactive companion, using the same offsets and encodings as the PDF."""
    import plotly.graph_objects as go

    fig = go.Figure()
    data = plot_rows(aggregate, greedy_metrics)
    for controller, (color, _, symbol, offset) in STYLES.items():
        rows = data[data.controller == controller]
        fig.add_trace(go.Scatter(
            x=rows.distance_m,
            y=[TOPOLOGIES.index(str(t)) + offset for t in rows.topology],
            name=controller, mode="markers",
            marker=dict(color=color, symbol=symbol, size=8),
            error_x=dict(type="data", array=rows.distance_std_m, thickness=1, width=3),
            text=[LABELS[str(t)] for t in rows.topology],
            hovertemplate="%{text}<br>%{x:.2f} m<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        template="simple_white", font=dict(family="Arial", size=13),
        width=1000, height=640, margin=dict(l=210, r=25, t=60, b=60),
        xaxis=dict(title="Positive-x displacement (m)", showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(tickvals=list(range(len(TOPOLOGIES))),
                   ticktext=[LABELS[t] for t in TOPOLOGIES], autorange="reversed"),
        legend=dict(orientation="h", x=0, y=1.08),
    )
    return fig


def save_publication_figure(aggregate: pd.DataFrame, seed_results: pd.DataFrame,
                            output_dir: Path, greedy_metrics: pd.DataFrame) -> None:
    """Export vector PDF with embedded TrueType fonts at IEEE double-column width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = plot_rows(aggregate, greedy_metrics)
    incomplete = set(seed_results.loc[seed_results.run_state != "finished", "topology"])
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": 0.6,
    }):
        fig, ax = plt.subplots(figsize=(7.16, 4.65))
        fig.subplots_adjust(left=0.285, right=0.985, bottom=0.13, top=0.90)
        for i in range(len(TOPOLOGIES)):
            if i % 2 == 0:
                ax.axhspan(i - .48, i + .48, color="0.97", lw=0, zorder=0)
        for controller, (color, marker, _, offset) in STYLES.items():
            rows = data[data.controller == controller]
            ypos = [TOPOLOGIES.index(str(t)) + offset for t in rows.topology]
            ax.errorbar(rows.distance_m, ypos, xerr=rows.distance_std_m,
                        fmt=marker, color=color, markersize=4.0, markeredgewidth=.7,
                        elinewidth=.85, capsize=2.2, capthick=.7,
                        label=controller, zorder=3)
        # Small unfilled marks expose the three individual seed evaluations.
        for controller, (color, marker, _, offset) in STYLES.items():
            if controller == "Greedy":
                continue
            rows = seed_results[seed_results.controller == controller]
            ypos = [TOPOLOGIES.index(str(t)) + offset for t in rows.topology]
            ax.scatter(rows.distance_m, ypos, marker=marker, s=8,
                       facecolors="white", edgecolors=color, linewidths=.5, zorder=2)
        ax.set_yticks(range(len(TOPOLOGIES)), [
            LABELS[t] + (" †" if t in incomplete else "") for t in TOPOLOGIES
        ])
        ax.set_ylim(len(TOPOLOGIES) - .5, -.6)
        low = min(0., float((data.distance_m - data.distance_std_m).min()))
        high = float((data.distance_m + data.distance_std_m).max())
        ax.set_xlim(np.floor(low / 10) * 10 - 1, np.ceil(high / 10) * 10 + 1)
        ax.set_xlabel("Positive-x displacement (m)", labelpad=7)
        ax.xaxis.grid(True, color="0.88", linewidth=.5)
        ax.set_axisbelow(True)
        ax.axvline(0, color="0.55", linewidth=.6)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", direction="out", width=.6, length=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("0.45")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(.59, .995), ncol=3, frameon=False,
                   handletextpad=.5, columnspacing=1.1)
        for suffix in ("pdf", "png"):
            fig.savefig(output_dir / f"distance_achieved_by_topology.{suffix}", dpi=600)
        plt.close(fig)


def load_history_snapshot(path: Path) -> list[Any]:
    """Adapt a source-preserving history export to the W&B run interface."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        snapshot = json.load(stream)
    runs = []
    for item in snapshot["runs"]:
        h = snapshot["histories"][item["id"]]
        if h.get("error") or not h.get("exact") or h.get("source_truncated"):
            raise RuntimeError(f"Incomplete history export for {item['id']}")
        run = SimpleNamespace(
            id=item["id"], name=item["display_name"], state=item["state"],
            config=item["config"], summary=item.get("summary", {}), url=item["url"],
        )
        run.history = lambda keys, samples, pandas, rows=h["rows"]: rows
        runs.append(run)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="i-suds/Paper-Experts")
    parser.add_argument(
        "--run-prefix", default=None,
        help="Legacy single-family mode; default selects both Paper-Experts architectures.",
    )
    parser.add_argument(
        "--greedy-metrics",
        type=Path,
        default=DEFAULT_GREEDY_METRICS,
        help="CSV containing measured greedy distance_m and distance_std_m by topology.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--history-json", type=Path, help="Offline W&B history snapshot.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_history_snapshot(args.history_json) if args.history_json else wandb.Api().runs(args.project)
    if args.run_prefix is None:
        seed_results = collect_expert_results(runs)
        aggregate = aggregate_experts(seed_results)
    else:
        seed_results = collect_seed_results(runs, args.run_prefix).assign(controller="GNN specialist")
        aggregate = aggregate_experts(seed_results)

    missing = [name for name in TOPOLOGIES if name not in set(aggregate["topology"])]
    if missing:
        raise RuntimeError(f"Missing usable results for topologies: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(args.output_dir / "distance_by_topology_seed_results.csv", index=False)
    aggregate.to_csv(args.output_dir / "distance_by_topology_summary.csv", index=False)

    greedy = load_greedy_results(args.greedy_metrics)
    fig = make_figure(aggregate, greedy)
    save_publication_figure(aggregate, seed_results, args.output_dir, greedy)
    fig.write_html(
        args.output_dir / "distance_achieved_by_topology.html", include_plotlyjs="cdn"
    )
    if args.show:
        fig.show()

    print(f"Saved figure and data to {args.output_dir}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
