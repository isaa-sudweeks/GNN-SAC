"""Download paper evaluation results from W&B and plot distance by topology."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
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


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def collect_seed_results(runs: Iterable[Any], run_prefix: str) -> pd.DataFrame:
    """Return one row per topology/seed, preferring the most advanced retry."""
    candidates: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        topology = config.get("truss_topology")
        seed = config.get("seed")
        distance = _finite_float(run.summary.get(METRIC))
        if topology not in TOPOLOGIES or seed is None or distance is None:
            continue
        if run_prefix and not str(run.name).startswith(run_prefix):
            continue
        candidates.append(
            {
                "topology": topology,
                "seed": int(seed),
                "distance_m": distance,
                "run_id": run.id,
                "run_name": run.name,
                "run_state": run.state,
                "logged_step": int(run.summary.get("_step", -1)),
                "run_url": run.url,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No runs matched prefix {run_prefix!r} with a finite {METRIC!r} summary."
        )

    results = pd.DataFrame(candidates)
    return (
        results.sort_values(["topology", "seed", "logged_step", "run_id"])
        .drop_duplicates(["topology", "seed"], keep="last")
        .reset_index(drop=True)
    )


def aggregate_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute across-seed mean and sample standard deviation per topology."""
    aggregate = (
        seed_results.groupby("topology", as_index=False)
        .agg(
            distance_m=("distance_m", "mean"),
            distance_std_m=("distance_m", "std"),
            n_seeds=("seed", "nunique"),
        )
        .fillna({"distance_std_m": 0.0})
    )
    aggregate["topology"] = pd.Categorical(
        aggregate["topology"], categories=TOPOLOGIES, ordered=True
    )
    return aggregate.sort_values("topology").reset_index(drop=True)


def make_figure(aggregate: pd.DataFrame):
    """Create the RA-L-style horizontal distance plot."""
    gnn_sac = aggregate.assign(controller="GNN-SAC")
    greedy = pd.DataFrame(
        {
            "topology": TOPOLOGIES,
            "distance_m": 5.0,
            "distance_std_m": float("nan"),
            "n_seeds": 0,
            "controller": "Greedy",
        }
    )
    plot_data = pd.concat([gnn_sac, greedy], ignore_index=True)
    fig = px.scatter(
        plot_data,
        x="distance_m",
        y="topology",
        color="controller",
        error_x="distance_std_m",
        category_orders={"topology": list(reversed(TOPOLOGIES))},
        color_discrete_map={"GNN-SAC": "#636EFA", "Greedy": "#EF553B"},
        title="Distance Achieved by Topology",
        labels={
            "distance_m": "Distance Traveled (m)",
            "topology": "Robot Configuration",
            "controller": "Controller",
        },
        hover_data={"n_seeds": True, "distance_std_m": ":.3f"},
    )
    fig.update_layout(
        font_family="Arial",
        font_size=12,
        title_font_size=16,
        xaxis=dict(title_font_size=14, tickfont_size=12, showgrid=False),
        yaxis=dict(title_font_size=14, tickfont_size=12, showgrid=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1,
            font_size=12,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=50, t=80, b=80),
        width=900,
        height=650,
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")),
        error_x=dict(thickness=1.5, width=3, color="dimgray"),
        selector=dict(mode="markers"),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="i-suds/paper_results")
    parser.add_argument(
        "--run-prefix", default="paper-v2-",
        help="Only include run names beginning with this value; pass '' for all runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_results = collect_seed_results(wandb.Api().runs(args.project), args.run_prefix)
    aggregate = aggregate_results(seed_results)

    missing = [name for name in TOPOLOGIES if name not in set(aggregate["topology"])]
    if missing:
        raise RuntimeError(f"Missing usable results for topologies: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(args.output_dir / "distance_by_topology_seed_results.csv", index=False)
    aggregate.to_csv(args.output_dir / "distance_by_topology_summary.csv", index=False)

    fig = make_figure(aggregate)
    fig.write_image(args.output_dir / "distance_achieved_by_topology.png", scale=3)
    fig.write_html(
        args.output_dir / "distance_achieved_by_topology.html", include_plotlyjs="cdn"
    )
    if args.show:
        fig.show()

    print(f"Saved figure and data to {args.output_dir}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
