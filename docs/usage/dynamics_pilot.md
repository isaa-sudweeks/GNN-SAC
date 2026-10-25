# Dynamics and MPC prerequisite pilot

## Current measured status — October25,2026

The offline pilot and subsequent online source experiments are complete. The
latest [candidate-ranking diagnostic](dynamics_results/online_candidate_ranking_2026-10-25.md)
identified missed collapse risks and finite-set scoring errors. The
[failure-balanced replay/control test](dynamics_results/failure_balanced_replay_2026-10-25.md)
did not provide a consistent repair. Those source results do not establish
unseen-topology transfer or comparative superiority to model-free control.

Current next-stage design:
[matched dynamics-MPC versus source-trained GNN-SAC generalist](../plans/dynamics_vs_modelfree_generalization.md).
The user will initiate that experiment separately. Read the Obsidian notebook and
the [measured results index](dynamics_results/README.md) before treating historical
gate/next-step statements below as the current recommendation.

The pilot provides source replay auditing, physical-unit coordinate helpers,
native MuJoCo branches, smooth CEM proposals, indexed source collection, and an
observable-input GNN with independent-episode fitting and recursive evaluation.
Held-out transfer has not been evaluated by these source experiments; the next
comparison requires its own predeclared matched protocol.
Simulator diagnostic settings are in `config/dynamics_pilot.yaml`.

Use the project's uv environment. A reusable external uv environment can be
selected with `UV_PROJECT_ENVIRONMENT`; `--no-sync` uses that existing environment.

```bash
uv run python -m unittest tests.test_dynamics_planner tests.test_dynamics_simulator tests.test_dynamics_state tests.test_dynamics_replay_audit -v
uv run python scripts/audit_dynamics_replay.py /trusted/source/checkpoint.pt --output outputs/dynamics-pilot/replay.json
uv run python scripts/probe_dynamics_state.py
uv run python scripts/benchmark_dynamics_simulator.py
uv run python scripts/run_simulator_mpc_pilot.py
uv run python scripts/collect_dynamics_source_probe.py
uv run python scripts/validate_dynamics_source_probe.py
uv run python scripts/save_dynamics_reset_manifest.py
uv run python scripts/plot_dynamics_pilot.py
```

The checkpoint auditor opens only an explicitly selected trusted checkpoint.
It filters replay tasks against the eight `random_5fold/fold_4` source topologies
before inspecting rows. Legacy generic specialist task names are resolved only
from an unambiguous saved topology config. It samples 128 rows with seed zero,
records valid circular-buffer indices and a checkpoint hash, and never assumes
that adjacent replay rows form trajectories. Its one-step candidate status is
not state-sufficiency approval. Only sampled rows receive finite-value checks.

The original prerequisite simulator diagnostics are restricted to source tetrahedron,
nominal abstract physics, control-graph actions, and no rendering. The indexed
training collector explicitly allows all eight source topologies, using the same
nominal abstract control-graph contract and separate restoration checks. The branch evaluator
uses a separate environment, copies complete `MjData`, and restores explicitly
audited environment/wrapper counters, caches and RNG. New wrappers, realistic
controllers, or randomized physics require another restoration audit.

The planner proposes bounded node commands with three linearly interpolated
knots. It scores actual simulator rollouts, executes one action, and replans.
The previous plan is shifted by one step for the next proposal mean. It retains
the best actually evaluated candidate across iterations. The diagnostic config
uses 32 candidates, two iterations and ten control steps; this is a measured-cost
source development setting, not the original 256-by-3 proposal or frozen holdout
configuration. The original matrix is expensive on native CPU simulation.

Its objective is forward **physical-node mean** displacement divided by initial
bounding-box diagonal, minus 0.1 times the time integral of mean squared routed,
clipped actuator commands. A collapsed/nonfinite candidate pays 1 plus its
remaining horizon in seconds. No additional terminal reward is used. These
weights are initial source settings, not established optimal values. Full raw
environment return is reported separately. Neither command effort nor planning
score is SAC return. The horizon spans 2 seconds at a 0.2-second control interval;
the three knots occur at 0, 0.9 and 1.8 seconds (sampled on the control grid).
The diagnostic episode spans 8 simulated seconds.

`run_simulator_mpc_pilot.py` pairs zero, smooth random and simulator MPC using exact
copied initial states for five reset seeds. It retains individual actions,
position traces, collapse status, raw returns and per-decision latency. Reset
variation is the nominal generator's seeded noise; it is not a broad distribution
of orientations or near-failure conditions. The random control uses clipped
normal knots every approximately five steps, while MPC uses its configured
planning-horizon knots. Both use the same bounds, initial scale and action mask.
This is an initial smooth-random control, not a matched search-compute comparator.

The source evaluation collector saves four independent 40-step episodes with
episode/environment/timestep IDs, separate termination/truncation, true next
observations, raw reward components, elapsed time, physical positions, actuator
activation, external controls and MuJoCo integration snapshots. Each episode is
a separate compressed file with a SHA-256 manifest. Schema v2 additionally saves
full `MjData` and wrapper snapshots because integration state alone did not
reproduce the raw reward exactly. These trusted local pickle files preserve
derived caches; the validator checks saved labels against restored transitions.
Existing datasets are never
overwritten. These 160 probe transitions are evaluation-only and excluded from
fitting and checkpoint selection; they demonstrate logging, not action coverage.

## Replay eligibility and observable-input baseline

The audited teacher checkpoint has one million transition pairs but no episode,
environment or timestep IDs, truncation field, or actuator activation state.
A hidden-activation intervention showed that visible node features and commands
do not uniquely determine exact simulator transitions. It did not establish that
approximate prediction on normal trajectories is inadequate for MPC.

Begin with physical-node positions and velocities, available structural/routing
metadata and known commands; initially use three message-passing layers of width
128. Hidden actuator targets and full simulator snapshots are diagnostic ground
truth, not required deployment-facing model features. Check units, centered-xy
translation bookkeeping, physical-node masks, balanced graph losses and recursive
geometry refresh before fitting. Never refresh imagined state from future truth.

Before reusing each source replay, trace clean/noisy observation and executed-action
labels and verify backend/settings compatibility. The saved configuration enables
observation/action noise and uses MJX. Missing actuator targets alone do not reject
one-step fitting. Missing episode provenance prevents sequence validation. Reuse
compatible transitions for one-step fitting; otherwise collect nominal source
rollouts with independent fitting and validation episodes. Keep the evaluation-only
probe excluded from fitting, normalization and checkpoint selection. Diagnose source
errors and action coverage before adding history or changing model capacity.

See `docs/plans/gnn_dynamics_mpc_pilot.md` for the active execution plan and gates.

Results under `outputs/dynamics-pilot/` are generated artifacts and must not be
committed. Experiment notes and figures live in the user's Obsidian notebook.

## Observable dynamics training and source evaluation

The nominal source collector uses the eight training topologies from
`random_5fold/fold_4` and refuses non-source topology names. Its declared fitting,
validation and evaluation roles have disjoint episode/reset identities. The
80-step episodes use smooth random actions at several amplitudes; these data do
not establish teacher-action or planner-proposal coverage. Terminal outcomes are
retained, with actual elapsed time and separate truncation flags. Graph aliases
are centered using the unique physical-node mean. Static metadata includes
routing incidence, actuator bounds, physical-node representatives and structural
bars; snapshots retain simulator internals for ground truth only.

```bash
uv run python scripts/collect_dynamics_training_data.py
uv run python scripts/validate_dynamics_training_data.py
uv run python scripts/train_dynamics_model.py
uv run python scripts/evaluate_dynamics_model.py --data outputs/dynamics-pilot/source-training-2026-10-23 --checkpoint outputs/dynamics-pilot/model-source-2026-10-23/best-model.pt --output outputs/dynamics-pilot/frozen-source-evaluation.json
uv run python scripts/evaluate_dynamics_ranking.py --data outputs/dynamics-pilot/source-training-2026-10-23 --checkpoint outputs/dynamics-pilot/model-source-2026-10-23/best-model.pt --output outputs/dynamics-pilot/source-ranking-2026-10-23
```

Settings live in `config/dynamics_training_data.yaml`, `config/dynamics_model.yaml`
and `config/dynamics_ranking.yaml`. Collectors/training refuse existing output
paths unless the collector's explicit resume option is used. Resume is for an
interrupted identical protocol, not changing settings in an existing dataset.
The loader checks episode/metadata hashes, roles, seeds, finite labels, aliases,
progress and temporal adjacency. It never interprets legacy replay rows as episodes.

The model predicts physical position and velocity changes from observable state,
known node commands, current clipped routed commands and static graph metadata.
Geometry is recomputed from each imagined state. Three width-128 message-passing
layers are the initial baseline. Loss excludes duplicate physical nodes; sampling
balances source topologies. Normalization uses fitting data only. Checkpoints are
selected using independent validation episodes; evaluation episodes are loaded
only after selection. The original evaluation-only probe remains excluded.

The recursive evaluator reports forward physical-node-mean progress error,
world-aligned physical-node position error, velocity error, geometry-only rigidity
and endpoint collapse errors at 1/5/10/20 steps. Overlapping windows are not
independent samples. Terminal/nonfinite outcomes remain in the metrics, with the
finite-error convention recorded before fitting. The model predicts a nominal
control-step transition, including terminal stopping; it does not consume realized
future terminal duration. Constant velocity aligned with true elapsed durations is
a conservative diagnostic near shortened terminal steps. Report those cases
separately from full-duration windows before interpreting timing or collapse.

`evaluate_dynamics_ranking.py` compares the same actual model-CEM candidates in
frozen model and restored simulator branches on source evaluation snapshots. It
reports Spearman correlation (constant-score states are uninformative), finite-set
selection regret, collapse false negatives and selected collapses. Its default
four source states and 64 candidates are a development diagnostic, not the full
20-state-per-topology held-out gate. The objective matches the simulator diagnostic:
physical-node-mean forward progress/initial diagonal, minus integrated mean squared
clipped routed commands and fixed failure/remaining-horizon costs. No rollout uses
future true geometry, contact, rigidity or actuator activation as a model input.

## Geometry-only rigidity

`dynamics.rigidity` reproduces the installed nominal abstract-truss rigidity
calculation from physical-node positions and one structural bar per edge. It
uses the fixed initial reference, axis selection and weighted/unweighted setting.
Control-graph alias edges cannot substitute for structural bars. Nonfinite
geometry is flagged as a failure. Realistic connector graphs require another audit.

```bash
uv run python scripts/audit_dynamics_rigidity.py
uv run python -m unittest tests.test_dynamics_rigidity tests.test_dynamics_ranking tests.test_dynamics_model tests.test_dynamics_dataset -v
```

## One allowed source repair

After the first source prediction and ranking failure, the recorded repair keeps
architecture and data fixed and adds ten-step recursive state/cumulative-progress
loss to the one-step loss. It starts from the validation-selected baseline and
selects the repaired checkpoint by ten-step source-validation forward-progress MAE.
`config/dynamics_model_multistep.yaml` records the extra update budget and weights.
Original and repaired runs use separate output directories. This consumes the
pilot's one targeted repair; unsuccessful source gates stop the iteration.

```bash
uv run python scripts/train_dynamics_model.py --config config/dynamics_model_multistep.yaml --output outputs/dynamics-pilot/model-source-multistep-2026-10-23
```

## Fixed-budget longer-training test

`early_stopping: false` runs the full configured update budget while retaining
validation-only best-checkpoint selection. `last-model.pt` preserves final weights;
it is separate from `best-model.pt`. Histories now retain `train_loss_mean`,
one-step/rollout component means and per-source batch means/counts over each logging
interval. The legacy `train_loss` field remains a single-batch snapshot.

```bash
uv run python scripts/train_dynamics_model.py --config config/dynamics_model_multistep_4k.yaml --output outputs/dynamics-pilot/model-source-multistep-4k-2026-10-23
```

The 4,000-update comparison starts from the same initialization as the earlier
2,000-update multi-step run. It uses the same model/data/loss, with patience stopping
disabled. It is not an optimizer-state resume from the repaired best checkpoint.

## Online learned-MPC source pilot

`config/dynamics_online.yaml` defines a separate user-authorized source experiment.
Actual simulator transitions train the observable model; learned-model CEM selects
training actions after a common random warm-up. The reward/value variant adds a
learned immediate reward, twin TD Q values, delayed actor and target networks to
estimate a discounted terminal tail. It is TD-MPC2-inspired, not full TD-MPC2.
No simulator-MPC or teacher data is used for fitting. Episode boundaries are explicit;
true terminal transitions disable value bootstrapping, while time-limit truncations
bootstrap. Fixed-budget checkpoints are evaluated without exploration or updates.

```bash
uv run python -m unittest tests.test_dynamics_online -v
uv run python scripts/run_online_dynamics_pilot.py --output outputs/dynamics-pilot/online-source-2026-10-24 --diagnostic-data /trusted/source-training-2026-10-23
```

See `docs/plans/gnn_dynamics_online_pilot.md` for fixed budgets, common reward,
comparison limits and stopping rules. Generated replay, traces and models stay ignored.

Plot the completed experiment with `scripts/report_online_dynamics_pilot.py OUTPUT_DIRECTORY` in a Python environment with Matplotlib (for example, `uv run --with matplotlib python ...`). Inspect each frozen controller’s own evaluation errors with `uv run python scripts/diagnose_online_dynamics_pilot.py OUTPUT_DIRECTORY`. These are descriptive checks and perform no additional fitting.

## Fixed-method longer online run

`config/dynamics_online_long.yaml` extends both learned-MPC variants to 20,000 real
transitions per seed. `scripts/run_online_dynamics_long.py` reruns identical seeds,
checks the exact 1,600-transition prefix against the short runs, retains monitoring
checkpoints and evaluates fresh final source resets. Learning settings stay fixed.
The runner adds full native/RNG/replay restart state; `--resume` requires identical
source/configuration and verifies checkpoint hashes. Independent workers have one
Torch thread each. See `docs/plans/gnn_dynamics_online_long.md` for the full protocol,
monitoring/fresh seed separation, budget and stopping rules.

## Shared versus separate online source dynamics

The controlled three-source experiment uses `config/dynamics_online_multi.yaml` and
`scripts/run_online_dynamics_multi.py`. See
[`gnn_dynamics_online_multi.md`](../plans/gnn_dynamics_online_multi.md) for the frozen
protocol, topology choice, exact exposure accounting, common fitting-only
normalization and stopping rules. It compares dynamics-only learned MPC under equal
20,000-transition and 20,000-update budgets per topology, with three training seeds.
Round-robin topology-homogeneous batches support variable graph sizes without
padding. Frozen checkpoint evaluation and fresh final evaluation stay out of replay.

Run `scripts/benchmark_online_dynamics_multi.py` first. Latest and milestone
checkpoints contain all three real replays, models/optimizers, graph metadata, RNGs,
CEM plans and live native simulator states. Resume with the identical command plus
`--resume`; source/config/dependency mismatches are rejected. Failure evidence is
retained. `scripts/report_online_dynamics_multi.py OUTPUT` verifies budgets, paired
starts, warmups, normalization and artifact integrity and builds per-topology,
per-seed learning curves and paired comparisons. Outputs are source-training
comparisons only and cannot establish unseen-topology transfer.

The October 24 three-source campaign is complete: all 360,000 real fitting
transitions and declared evaluations passed integrity checks (4.08 elapsed hours).
Sharing reduced tetrahedron fresh return in all three seeds and increased collapses
from 0/15 to 4/15, reduced six-node collapses from 13/15 to 9/15, and left both
conditions at 15/15 eight-node collapses. Shared learning without a control cost was
not demonstrated. The per-seed report, paired data, learning curves and common-trace
prediction diagnostics are in `outputs/dynamics-pilot/online-multi-source-2026-10-24/`;
see the protocol's completed-outcome section for the source-only next recommendation.

## Discount-only rerun

The separately authorized gamma=.95 replication uses
`config/dynamics_online_multi_gamma095.yaml` with the unchanged shared/separate
runner. Only gamma differs from the completed .995 campaign; see
[`gnn_dynamics_online_multi_gamma095.md`](../plans/gnn_dynamics_online_multi_gamma095.md).
It repeats the same 360,000-transition protocol from matched initial weights.
`scripts/report_online_dynamics_discount_comparison.py` compares both campaigns'
saved traces at common discounts .95/.995 and undiscounted return, alongside
unchanged displacement/collapse metrics. The separate saved-data diagnostic may
not change this active experiment.

## Frozen failure and gradient diagnostic

The original gamma=.995 campaign's saved-data diagnostic is complete. See
[`gnn_dynamics_multi_diagnostic.md`](../plans/gnn_dynamics_multi_diagnostic.md),
`scripts/run_dynamics_multi_diagnostic.py`, and its ignored output report under
`outputs/dynamics-pilot/online-multi-diagnostic-2026-10-24/`. It measures both frozen
models on identical episode-safe executed-action windows and balanced fitting
examples. No optimizer steps or simulator evaluations occur. Final-weight gradient
cosines are local evidence, not a causal training-interference test. Three
cohort/gradient/nonfinite regression tests pass. Gamma=.95 training remains unchanged.

The gamma=.95 rerun is complete (360,000 transitions, 4.01 elapsed hours). Common-
discount comparisons show no consistent improvement: tetrahedron specialist
collapses rise 0/15 to 5/15, shared six-node rises 9/15 to 15/15, separate six-node
falls 13/15 to 11/15, and both eight-node methods remain at 15/15. The full native
report and `discount-comparison/report.md` retain every seed, both common scoring
discounts, undiscounted return, movement, collapse, prediction and latency evidence.
No follow-on training or unseen-topology evaluation was performed.

Versioned measured reports and static figures are available in
[dynamics_results/README.md](dynamics_results/README.md). Raw experiment outputs
remain local and ignored.
