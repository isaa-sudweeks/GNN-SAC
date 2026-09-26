"""Pull per-(fold, seed, topology) eval performance from cross-validation W&B runs.

``sac/trainer/online_trainer.py``'s ``_eval_topologies`` evaluates every
topology in a cross-validation fold -- both the ones the fold trains on and
the ones it holds out -- and ``sac/common/logger.py`` logs each as
``eval/{topology}_episode_distance`` with ``_step`` equal to the env step.

Only runs of a single experiment are used (``--experiment``, matched on
``run.group == "truss-graph-{experiment}"`` or an exact tag). When a
``(fold, seed)`` has several runs (e.g. short aborted duplicates), the one
with the largest ``summary._step`` is kept.

Definitions (all per seed run, never pooled across seeds before selection):

* Eval history: every row of the run where the topology's metric is finite,
  fetched unsampled with ``scan_history``. Seeds are never aligned on shared
  steps; each run's curve is used as logged.
* Baseline: the run's first eval (minimum step, normally step 0) is an
  untrained policy. ``baseline_distance_m`` is each topology's value there.
  The baseline eval is excluded from every window below.
* Checkpoint selection (training data only): at each post-baseline eval
  step, average eval distance over the fold's TRAINING topologies; smooth
  with a trailing rolling mean over ``window`` evals (full windows only,
  unless the run has fewer than ``window`` post-baseline evals, in which case
  all of them form one window); ``selected_step`` is the step maximizing that
  smoothed train mean (latest on ties). Held-out metrics never enter this.
* ``distance_m``: for every topology (train and held-out) the mean of its eval
  distance over the same window of evals ending at ``selected_step``.
* ``distance_final_m``: mean over the run's last ``window`` post-baseline evals.
* Noise floor (``topology_noise_floor.csv``): per topology, mean and std
  (ddof=1) of the baseline eval across all selected runs of the experiment
  (every topology is evaluated in every run at step 0). ``n_runs`` counts
  runs; ``n_seeds`` counts distinct seeds. Runs sharing a seed start from the
  same initialization and their step-0 evals are near-identical across folds,
  so the floor has only ``n_seeds`` effectively independent samples.

Generalization summary (``topology_generalization_summary.csv``), one row per
CV topology:

* ``heldout_distance_m``: mean of ``distance_m`` over the seeds of the fold
  holding the topology out; ``heldout_seed_std_m`` their std (ddof=1).
* ``in_training_distance_m``: mean of ``distance_m`` over every (fold, seed)
  in which the topology is a training topology.
* ``transfer_ratio = (heldout - floor_mean) / (in_training - floor_mean)``,
  NaN when the denominator is <= 0. 0 means no better than untrained, 1 means
  as good as when trained on.
* ``heldout_above_floor_z = (heldout - floor_mean) /
  sqrt(floor_std**2 / n_floor_seeds + heldout_seed_std**2 / n_heldout_seeds)``:
  a Welch-style standardized difference between the held-out mean and the
  untrained floor mean, using the number of distinct seeds (not runs) as the
  floor sample size for the reason above. With 3 seeds per side it is only a
  rough screen (|z| > ~2 is suggestive, not a calibrated p-value), and a
  "significant" z can still be a practically negligible distance; read it
  together with ``transfer_ratio``.
* ``heldout_gain_m`` / ``heldout_gain_std_m``: the paired alternative -- mean
  and std over held-out seeds of ``distance_m - baseline_distance_m``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

FIGURES_ROOT = Path(__file__).resolve().parent
if str(FIGURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURES_ROOT))

import _topology_catalog

DEFAULT_PROJECT = "i-suds/CV"
DEFAULT_EXPERIMENT = "CV-BASE"
DEFAULT_WINDOW = 5
HISTORY_SAMPLES = 100_000
DEFAULT_OUTPUT_CSV = FIGURES_ROOT / "topology_performance_by_fold.csv"
DEFAULT_FLOOR_CSV = FIGURES_ROOT / "topology_noise_floor.csv"
DEFAULT_SUMMARY_CSV = FIGURES_ROOT / "topology_generalization_summary.csv"

OUTPUT_COLUMNS = [
    "fold",
    "seed",
    "run_id",
    "run_name",
    "topology",
    "role",
    "selected_step",
    "n_window_evals",
    "distance_m",
    "distance_final_m",
    "baseline_distance_m",
    "baseline_step",
    "max_logged_eval_step",
    "n_evals",
    "summary_step",
]


def metric_key(topology: str) -> str:
    return f"eval/{topology}_episode_distance"


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
    return {key[len(prefix):]: value for key, value in config.items() if key.startswith(prefix)}


def run_matches_experiment(run, experiment: str) -> bool:
    """True if the run's group is ``truss-graph-{experiment}`` or it has the exact tag."""
    if getattr(run, "group", None) == f"truss-graph-{experiment}":
        return True
    return experiment in (getattr(run, "tags", None) or [])


def _eval_history(run, keys: list[str]) -> pd.DataFrame:
    """Unsampled eval history: one row per ``_step``, NaN where a key is missing.

    ``scan_history(keys=...)`` returns every row with ``None`` for absent keys
    and never subsamples. The ``history`` fallback is queried one key at a time,
    because W&B's sampled ``history(keys=...)`` drops rows lacking any key.
    """
    records: list[dict[str, Any]] = []
    if hasattr(run, "scan_history"):
        records = list(run.scan_history(keys=keys))
    else:
        for key in keys:
            records.extend(run.history(keys=[key], samples=HISTORY_SAMPLES, pandas=False))

    rows: dict[int, dict[str, float]] = {}
    for record in records:
        step = _finite_float(record.get("_step"))
        if step is None:
            continue
        row = rows.setdefault(int(step), {})
        for key in keys:
            value = _finite_float(record.get(key))
            if value is not None:
                row[key] = value  # later rows at the same step win
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=keys).sort_index()
    frame.index.name = "_step"
    return frame.dropna(how="all")


def select_runs(runs: Iterable[Any], cv_name: str, experiment: str | None) -> pd.DataFrame:
    """One run per (fold, seed): the longest by ``summary._step``."""
    fold_membership = _topology_catalog.fold_membership()
    candidates: list[dict[str, Any]] = []
    for run in runs:
        if experiment is not None and not run_matches_experiment(run, experiment):
            continue
        cv_config = _cross_validation_config(run)
        if not cv_config or str(cv_config.get("name")) != cv_name:
            continue
        fold = cv_config.get("held_out_group")
        seed = dict(run.config).get("seed")
        if fold not in fold_membership or seed is None:
            continue
        summary_step = _finite_float(run.summary.get("_step"))
        candidates.append(
            {
                "fold": fold,
                "seed": int(seed),
                "run": run,
                "run_id": run.id,
                "run_name": run.name,
                "summary_step": int(summary_step) if summary_step is not None else -1,
            }
        )
    if not candidates:
        raise RuntimeError(
            f"No runs matched cross_validation.name={cv_name!r}, experiment={experiment!r}."
        )
    return (
        pd.DataFrame(candidates)
        .sort_values(["fold", "seed", "summary_step", "run_id"])
        .drop_duplicates(["fold", "seed"], keep="last")
        .reset_index(drop=True)
    )


def _run_rows(candidate: dict[str, Any], history: pd.DataFrame, window: int) -> list[dict]:
    membership = _topology_catalog.fold_membership()[candidate["fold"]]
    roles = {name: "train" for name in membership["train"]}
    roles.update({name: "heldout" for name in membership["held_out"]})
    train_keys = [metric_key(name) for name in membership["train"]]
    label = f"{candidate['run_id']} ({candidate['fold']}, seed {candidate['seed']})"

    history = history.dropna(subset=train_keys)
    if len(history) < 2:
        raise RuntimeError(f"Run {label} has fewer than 2 complete evals (baseline + 1).")

    baseline_step = int(history.index[0])
    baseline = history.iloc[0]
    trained = history.iloc[1:]

    effective_window = min(window, len(trained))
    smoothed = (
        trained[train_keys]
        .mean(axis=1)
        .rolling(effective_window, min_periods=effective_window)
        .mean()
        .dropna()
    )
    best = smoothed.max()
    selected_step = int(smoothed.index[smoothed == best].max())
    selected_pos = trained.index.get_loc(selected_step)
    selected_window = trained.iloc[selected_pos - effective_window + 1 : selected_pos + 1]
    final_window = trained.iloc[-effective_window:]

    rows = []
    for topology, role in roles.items():
        key = metric_key(topology)
        present = history[key].dropna() if key in history else pd.Series(dtype=float)
        rows.append(
            {
                "fold": candidate["fold"],
                "seed": candidate["seed"],
                "run_id": candidate["run_id"],
                "run_name": candidate["run_name"],
                "topology": topology,
                "role": role,
                "selected_step": selected_step,
                "n_window_evals": int(selected_window[key].notna().sum()),
                "distance_m": float(selected_window[key].mean()),
                "distance_final_m": float(final_window[key].mean()),
                "baseline_distance_m": float(baseline.get(key, np.nan)),
                "baseline_step": baseline_step,
                "max_logged_eval_step": int(present.index.max()) if len(present) else -1,
                "n_evals": int(len(present)),
                "summary_step": candidate["summary_step"],
            }
        )
    return rows


def collect_seed_results(
    runs: Iterable[Any],
    cv_name: str,
    *,
    experiment: str | None = DEFAULT_EXPERIMENT,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Return one row per (fold, seed, topology); see the module docstring."""
    if window < 1:
        raise ValueError("window must be >= 1")
    fold_membership = _topology_catalog.fold_membership()
    rows: list[dict[str, Any]] = []
    for candidate in select_runs(runs, cv_name, experiment).to_dict("records"):
        run = candidate.pop("run")
        membership = fold_membership[candidate["fold"]]
        keys = [metric_key(name) for name in membership["train"] + membership["held_out"]]
        rows.extend(_run_rows(candidate, _eval_history(run, keys), window))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def noise_floor(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Per-topology mean/std of the untrained (first-eval) distance across runs."""
    baseline = seed_results.dropna(subset=["baseline_distance_m"])
    return (
        baseline.groupby("topology")
        .agg(
            floor_mean_m=("baseline_distance_m", "mean"),
            floor_std_m=("baseline_distance_m", "std"),
            n_runs=("baseline_distance_m", "count"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )


def generalization_summary(seed_results: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """One row per topology comparing held-out, in-training, and untrained distance."""
    floor_by_topology = floor.set_index("topology")
    frame = seed_results.assign(gain_m=seed_results["distance_m"] - seed_results["baseline_distance_m"])
    rows = []
    for topology, group in frame.groupby("topology", sort=True):
        heldout = group[group["role"] == "heldout"]
        trained = group[group["role"] == "train"]
        floor_mean, floor_std, n_floor, n_floor_seeds = (
            floor_by_topology.loc[
                topology, ["floor_mean_m", "floor_std_m", "n_runs", "n_seeds"]
            ]
            if topology in floor_by_topology.index
            else (np.nan, np.nan, 0, 0)
        )
        heldout_mean = heldout["distance_m"].mean()
        heldout_std = heldout["distance_m"].std()
        n_heldout = int(heldout["distance_m"].notna().sum())
        in_training = trained["distance_m"].mean()

        denominator = in_training - floor_mean
        transfer_ratio = (heldout_mean - floor_mean) / denominator if denominator > 0 else np.nan
        se = math.sqrt(
            (floor_std**2 / n_floor_seeds if n_floor_seeds > 1 else np.nan)
            + (heldout_std**2 / n_heldout if n_heldout > 1 else np.nan)
        )
        z = (heldout_mean - floor_mean) / se if se > 0 else np.nan
        rows.append(
            {
                "topology": topology,
                "heldout_fold": heldout["fold"].iloc[0] if len(heldout) else None,
                "n_heldout_seeds": n_heldout,
                "heldout_distance_m": heldout_mean,
                "heldout_seed_std_m": heldout_std,
                "n_in_training_runs": int(trained["distance_m"].notna().sum()),
                "in_training_distance_m": in_training,
                "in_training_std_m": trained["distance_m"].std(),
                "floor_mean_m": floor_mean,
                "floor_std_m": floor_std,
                "n_floor_runs": int(n_floor),
                "n_floor_seeds": int(n_floor_seeds),
                "transfer_ratio": transfer_ratio,
                "heldout_above_floor_z": z,
                "heldout_gain_m": heldout["gain_m"].mean(),
                "heldout_gain_std_m": heldout["gain_m"].std(),
                "heldout_final_distance_m": heldout["distance_final_m"].mean(),
                "in_training_final_distance_m": trained["distance_final_m"].mean(),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--cv-name", default="node_count_loso")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--floor-csv", type=Path, default=DEFAULT_FLOOR_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def main() -> None:
    import wandb

    args = parse_args()
    runs = wandb.Api(timeout=120).runs(
        args.project,
        filters={"$or": [{"group": f"truss-graph-{args.experiment}"}, {"tags": args.experiment}]},
    )
    seed_results = collect_seed_results(
        runs, args.cv_name, experiment=args.experiment, window=args.window
    )
    floor = noise_floor(seed_results)
    summary = generalization_summary(seed_results, floor)
    for path, frame in (
        (args.output_csv, seed_results),
        (args.floor_csv, floor),
        (args.summary_csv, summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame)} rows to {path}")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nPer-(fold, seed) runs:")
        print(
            seed_results.drop_duplicates(["fold", "seed"])[
                ["fold", "seed", "run_id", "selected_step", "n_window_evals",
                 "baseline_step", "max_logged_eval_step", "n_evals", "summary_step"]
            ].to_string(index=False)
        )
        print("\nGeneralization summary:")
        print(
            summary[
                ["topology", "heldout_fold", "heldout_distance_m", "heldout_seed_std_m",
                 "in_training_distance_m", "floor_mean_m", "floor_std_m", "transfer_ratio",
                 "heldout_above_floor_z", "heldout_gain_m"]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )


if __name__ == "__main__":
    main()
