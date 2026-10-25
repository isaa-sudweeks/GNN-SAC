# Frozen online candidate-ranking diagnostic

Read `AGENTS.md` and the Obsidian experiment notebook before working on this
experiment. The predeclared protocol is
[`online_candidate_ranking.md`](../plans/online_candidate_ranking.md), notebook
Experiment17 in `My vault/Research/06 Experiment Notebook/`.

This checkout's diagnostic uses an explicitly supplied compatible runtime checkout
and the trusted local gamma=.995 campaign checkpoints. It does not import an
arbitrary installed dynamics package or copy training code between worktrees.
Both runtime and saved campaign sources are hashed; their Python ASTs must agree,
and installed dependency versions must match the original campaign. Checkpoint
normalization remains fixed. Paths must refer to the trusted experiment artifacts.

```bash
UV_PROJECT_ENVIRONMENT=/absolute/runtime/GNN-SAC/.venv \
UV_CACHE_DIR=/tmp/gnn-candidate-uv-cache \
uv run --no-sync python scripts/run_online_candidate_ranking.py \
  --runtime-root /absolute/runtime/GNN-SAC \
  --campaign-root /absolute/runtime/GNN-SAC/outputs/dynamics-pilot/online-multi-source-2026-10-24 \
  --output outputs/dynamics-pilot/online-candidate-ranking-NEW
```

Add `--benchmark` and use a distinct output directory for the first single-state
cost/integrity benchmark. Existing output directories are refused. Defaults for
the diagnostic are in `config/online_candidate_ranking.yaml`; learning, simulator
and planner settings are loaded from the original campaign protocol. The supplied
runtime must contain its `sac/dynamics` implementation and collection helpers.
There is no automatic resume, methodological repair, extra training or unseen
topology access. A failure preserves partial results and a traceback.

Each fixed model generates fresh real source episodes with no updates. Selected
snapshots pool both models' actual CEM proposals, zero and smooth random commands.
Exact CEM-selected indices are retained even under ties. Each sequence is evaluated
by both models and from the same full simulator snapshot in an auxiliary branch.
Scoring uses the online per-step discounted task reward, not the older offline
planner's remaining-horizon collapse penalty. Branches never change timeout limits
or counters to extend a horizon. Actual shortened terminal intervals are retained.

Outputs include `protocol.json`, `results.json` with per-state/seed/generator/stratum
metrics, `report.md`, `verification.json`, episode traces/plans and per-state candidate
arrays plus trusted local pickle snapshots/branch traces. All-candidate metrics
retain duplicate proposals; unique-candidate and provenance-specific sensitivity
metrics are also saved. Large generated files stay ignored.

The table label `native` means each model's exact CEM choice from its own proposal
set under a common diagnostic seed and the generating episode's warm start. It is
not necessarily the original trajectory's chosen plan. The original generating
plan is separately retained. Report both these choices and pooled-set argmax.

An H10 branch executes open-loop for up to two simulated seconds. Real MPC replans
after each action. A missed collapsing branch with a surviving alternative shows
finite-set misranking; it does not establish that a first action caused actual
closed-loop failure. Source results do not establish dynamics transfer or superiority
to model-free control. That requires a separately declared matched source-trained
GNN-SAC generalist versus frozen dynamics-MPC comparison on unseen robots.

Validate pure decision and branch contracts with:

```bash
uv run python -m unittest tests.test_online_candidate_ranking -v
```

The measured run uses the existing `c400` uv environment explicitly as above;
ordinary `uv run` also works after syncing this checkout's declared dependencies.

Completed Experiment17 results are in
[the measured report](dynamics_results/online_candidate_ranking_2026-10-25.md).
The frozen plan is retained byte-for-byte for executed-source provenance. After
independent artifact verification, regenerate its measured interpretation with:

```bash
uv run python scripts/report_online_candidate_ranking.py \
  --run outputs/dynamics-pilot/online-candidate-ranking-2026-10-25 \
  --destination docs/usage/dynamics_results/online_candidate_ranking_2026-10-25.md
```

The reporter requires the matching independent audit and the exact fixed
Experiment17 counts; it cannot publish a different or incomplete run as this result.
