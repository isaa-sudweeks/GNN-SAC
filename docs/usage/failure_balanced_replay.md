# Failure-adjacent replay and frozen control experiments

Follow the mandatory Obsidian notebook workflow in `AGENTS.md`. Experiments18/19
are predeclared in `My vault/Research/06 Experiment Notebook/`, with the reviewed
[joint plan](../plans/failure_balanced_replay.md). Experiment3 is excluded.

This is fixed-replay continuation of the original three-source shared gamma=.995
models, comparing uniform sampling to50/50 failure-adjacent/complement sampling.
It preserves the native online update mathematics and changes only the two replay
sampling calls. The recursive call samples positions in episode-safe starts, not
raw transition indices. The complement is not a safe-state label. Explicit fallback
to uniform is recorded when a stratum is absent; no importance weighting is used.

Use the trusted original campaign and its compatible runtime checkout explicitly:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_PROJECT_ENVIRONMENT=/absolute/runtime/GNN-SAC/.venv \
UV_CACHE_DIR=/tmp/gnn-candidate-uv-cache \
uv run --no-sync python scripts/run_failure_balanced_replay.py \
  --runtime-root /absolute/runtime/GNN-SAC \
  --campaign-root /absolute/runtime/GNN-SAC/outputs/dynamics-pilot/online-multi-source-2026-10-24 \
  --output outputs/dynamics-pilot/failure-balanced-replay-NEW
```

`config/failure_balanced_replay.yaml` contains the declared counts, reset namespaces,
sampling fraction, workers and elapsed cap. Original training/model/physics/planner
settings come from the campaign checkpoint. Full checkpoint normalization, shared
model/Adam/target identity, actor states, per-topology RNG and update counts are
restored. Each arm/seed adds2,000 batches per source and6,000 shared Adam updates.

`--benchmark` uses a separate output directory and100 fitting rounds without
validation or control outcome inspection. `--resume` requires identical inputs,
sources/config/dependencies and prepared-data manifest. Offline fitting resumes at
complete topology-round checkpoints with exact optimizer/sampler/RNG state; stale
progress after the checkpoint is discarded. Completed reports are skipped.
Incomplete episode-generation roles are retained and explicitly refused rather
than overwritten. Operational recovery of such a role requires a separately
documented fresh artifact path; do not bypass source hashes or duplicate live jobs.

Before fitting, three fresh original-policy validation episodes per source/seed
are generated using literal21M reset IDs. One state per episode freezes41 candidate
sequences and simulator outcomes. The validation/reference manifest is hashed before
any fitting. All final models predict the same validation actions and score the
same candidates; these data never enter replay, normalization or checkpoint choice.

After fixed-budget fitting, original, uniform and balanced shared models are
evaluated on five paired22M resets per source/seed with no exploration or adaptation.
Both continuation arms are evaluated regardless of validation results. Episode cap
remains80 controls/16 seconds, training horizon5 and planning horizon10/2 seconds,
CEM16x2, nominal fixed physics and gamma=.995. Actual state hashes verify pairing.
The global three-hour deadline covers preparation, fitting, validation and control.

The runner saves protocol/source hashes, frozen validation/reference evidence,
restart/final models, actual exposure counts, interval losses, validation cohorts,
candidate metrics, episode reward/state/dt traces and completed group reports.
All large generated files remain ignored. Source/seed results and paired differences
are summarized after independent audit with:

```bash
uv run python scripts/report_failure_balanced_replay.py \
  --run outputs/dynamics-pilot/failure-balanced-replay-2026-10-25 \
  --destination docs/usage/dynamics_results/failure_balanced_replay_2026-10-25.md
```

The reporting script also saves a standalone interactive learning-curve HTML.
Training losses use different sampling distributions; validation is the common
comparison. Primary prediction sensitivity uses any imagined H10 collapse on
terminal-ending windows; endpoint sensitivity, ordinary false positives and state
errors are reported too. Empty cohorts are inconclusive. Read survival alongside
movement and task return; stationary behavior is not a locomotion improvement.

Run deterministic sampler and artifact-backed native integration tests with:

```bash
GNN_DYNAMICS_RUNTIME=/absolute/runtime/GNN-SAC \
GNN_DYNAMICS_CAMPAIGN=/absolute/runtime/GNN-SAC/outputs/dynamics-pilot/online-multi-source-2026-10-24 \
uv run python -m unittest tests.test_failure_balanced_replay tests.test_failure_balanced_integration -v
```

The integration suite skips explicitly when trusted runtime/checkpoint paths are
not supplied. It tests exact native-uniform update parity, uninterrupted-versus-
resumed shared state and a real simulator restoration/reset/boundary smoke path.

The fixed campaign completed; the executed plan/config/runner remain unchanged for
provenance. See [measured results](dynamics_results/failure_balanced_replay_2026-10-25.md).
Both independent artifact/RNG verification and no_grad frozen-forward reproduction
are required before the reporting script publishes a final result.
