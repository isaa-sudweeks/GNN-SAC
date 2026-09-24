"""Pull per-topology, per-fold eval performance from cross-validation W&B runs.

Mirrors ``figures/plot_distance_by_topology.py``'s peak-across-seed-mean
selection, but grouped by ``(fold, topology)`` instead of just ``topology``,
since ``sac/trainer/online_trainer.py``'s ``_eval_topologies`` logs a
per-topology metric (``{topology}_episode_distance``) for every topology in
a cross-validation fold -- both the ones that fold trained on and the ones
it held out.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import wandb

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _topology_catalog

HISTORY_SAMPLES = 10_000
DEFAULT_OUTPUT_CSV = Path(__file__).with_name("topology_performance_by_fold.csv")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _cross_validation_config(run) -> dict:
    config = dict(run.config)
    cv_config = config.get("cross_validation")
    if isinstance(cv_config, dict):
        return cv_config
    # Fall back to dot-flattened keys, in case W&B flattened the nested group.
    prefix = "cross_validation."
    flattened = {
        key[len(prefix):]: value for key, value in config.items() if key.startswith(prefix)
    }
    return flattened


def collect_seed_results(runs: Iterable[Any], cv_name: str) -> pd.DataFrame:
    """Return seed values at each (fold, topology)'s peak across-seed mean eval step."""
    fold_membership = _topology_catalog.fold_membership()

    candidates: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        cv_config = _cross_validation_config(run)
        if not cv_config or str(cv_config.get("name")) != cv_name:
            continue
        fold = cv_config.get("held_out_group")
        seed = config.get("seed")
        if fold not in fold_membership or seed is None:
            continue
        candidates.append(
            {
                "fold": fold,
                "seed": int(seed),
                "run": run,
                "run_id": run.id,
                "run_name": run.name,
                "run_state": run.state,
                "logged_step": int(run.summary.get("_step", -1)),
            }
        )

    if not candidates:
        raise RuntimeError(f"No runs matched cross_validation.name={cv_name!r}.")

    selected_runs = (
        pd.DataFrame(candidates)
        .sort_values(["fold", "seed", "logged_step", "run_id"])
        .drop_duplicates(["fold", "seed"], keep="last")
        .reset_index(drop=True)
    )

    history_rows: list[dict[str, Any]] = []
    for candidate in selected_runs.to_dict("records"):
        run = candidate.pop("run")
        fold = candidate["fold"]
        membership = fold_membership[fold]
        topology_roles = {name: "train" for name in membership["train"]}
        topology_roles.update({name: "heldout" for name in membership["held_out"]})
        metric_keys = [f"{topology}_episode_distance" for topology in topology_roles]

        for history_row in run.history(keys=metric_keys, samples=HISTORY_SAMPLES, pandas=False):
            step = _finite_float(history_row.get("_step"))
            if step is None:
                continue
            for topology, role in topology_roles.items():
                distance = _finite_float(history_row.get(f"{topology}_episode_distance"))
                if distance is None:
                    continue
                history_rows.append(
                    {
                        **candidate,
                        "topology": topology,
                        "role": role,
                        "evaluation_step": int(step),
                        "distance_m": distance,
                    }
                )

    if not history_rows:
        raise RuntimeError(f"Matched runs have no finite per-topology distance history.")

    history = (
        pd.DataFrame(history_rows)
        .sort_values(["fold", "topology", "seed", "evaluation_step"])
        .drop_duplicates(["fold", "topology", "seed", "evaluation_step"], keep="last")
    )

    selected_rows = []
    for (fold, topology), group_history in history.groupby(["fold", "topology"], sort=False):
        seed_curves = group_history.pivot(
            index="evaluation_step", columns="seed", values="distance_m"
        ).dropna()
        if seed_curves.empty:
            raise RuntimeError(
                f"Fold {fold!r} topology {topology!r} has no evaluation step shared by every seed."
            )
        step_means = seed_curves.mean(axis=1)
        peak_mean = step_means.max()
        selected_step = int(step_means[step_means == peak_mean].index.max())
        rows = group_history[group_history["evaluation_step"] == selected_step].copy()
        rows["selection_mean_distance_m"] = peak_mean
        selected_rows.append(rows)

    return pd.concat(selected_rows, ignore_index=True).drop(columns="logged_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="i-suds/paper_results")
    parser.add_argument("--cv-name", default="node_count_loso")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_results = collect_seed_results(wandb.Api().runs(args.project), args.cv_name)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(seed_results)} rows to {args.output_csv}")
    print(
        seed_results.groupby(["fold", "topology", "role"], as_index=False)["distance_m"]
        .mean()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
