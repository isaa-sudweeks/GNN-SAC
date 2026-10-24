# GNN dynamics and MPC generalization pilot


## User-authorized training-budget test — October 23, 2026

The user requested a run twice the previous length after reviewing learning curves.
The earlier source-gate failure is a result for the tested budgets, not proof of
optimization convergence. This authorization supersedes the previous stop decision
for one fixed-budget test, without changing architecture, data, loss or holdout access.

Run `config/dynamics_model_multistep_4k.yaml`: 4,000 multi-step updates starting from
the same original first-model checkpoint as the 2,000-update repair, same seed 9321,
batch 32, optimizer/learning rate and source split. This is a controlled longer rerun,
not 4,000 extra updates from the repaired best checkpoint. Disable patience stopping
so the requested budget is actually reached; select best checkpoint on source
validation only and save final weights separately. Log interval means over every
training batch and by source topology, plus one-step/rollout components, every 200
updates. Compare the first 2,000 points with the earlier run to verify that logging
changes preserved the training trajectory. Budget estimate: 7–9 CPU minutes fitting,
plus source evaluation/ranking; within the original planning allocation.

Evaluate the selected checkpoint recursively at 1/5/10/20 steps on the unchanged
source validation/evaluation roles and repeat the same four-state tetrahedron ranking
diagnostic. Preserve all original data, curves and model checkpoints. Source evaluation
is development evidence; no held-out/final-test fitting or evaluation is included.
Record protocol, metrics and updated curves in Obsidian Experiment 11, under
`Research/06 Experiment Notebook/`. Generated files: `outputs/dynamics-pilot/model-source-multistep-4k-2026-10-23/`.

## Prior outcome — source gate unsuccessful at initial budgets; superseded by authorized test

Source model fitting and source diagnostics are complete. The single allowed
multi-step repair improved progress prediction but passed zero of eight source
screens; candidate ranking and collapse detection remain inadequate. Held-out
control/transfer remains gated because source criteria failed. Preserve the original
and repaired configurations; a new source-method experiment needs a separate
protocol. The minimum first milestone (source model plus simulator demonstration)
is complete, but learned-MPC generalization has not been established.

## Active contract — October 23, 2026

Complete source prediction before frozen-model holdout gates. Baseline: observable
physical-node positions/velocities, static graph/routing metadata and known commands;
three message-passing layers, width 128. Simulator actuator targets are diagnostic
fields, not mandatory model inputs. Preserve physical units, centered-xy progress,
physical-node masks, graph-balanced losses and recursive geometry refresh.

Source split: `random_5fold/fold_4`, eight source topologies; `octahedron` and
`henneberg_n8_1tube_57` held out. Keep final seven-node robots untouched. No held-out
fitting, normalization, checkpoint selection or tuning. No online adaptation.

## Execution order

1. Audit saved replay labels, noise, actions, backend/settings compatibility and each
   selected source buffer. Missing provenance prevents sequence use, not automatically
   one-step fitting. Missing actuator targets alone is not rejection grounds.
2. Predeclare fitting/validation collection seeds, batch size/capacity, source balance,
   update-to-data ratio, stopping rules and cost. Use nominal source rollout fallback
   when replay compatibility or coverage prevents reuse. Independent episodes are
   required for validation; existing evaluation probe seeds 8100–8103 remain excluded
   from fitting, normalization and checkpoint selection.
3. Implement and fit the observable GNN baseline. Compare persistence and constant
   velocity recursively at 1/5/10/20 steps and the selected planning horizon.
4. Diagnose source errors and action coverage. Measure progress/position/velocity,
   rigidity and collapse failures; compare shared simulator/model candidate sequences
   for ranking and regret before learned MPC. Do not manufacture missing state strata.
5. Freeze model and settings only after source evidence supports prediction/planning.
   Then evaluate held-out dynamics, action ranking and control with paired starts.
   Prediction screen: planning-horizon progress MAE at least 25% below the stronger
   baseline. Ranking: median per-topology Spearman at least 0.5 plus regret and collapse
   selection reporting. Control: positive median paired improvement over both zero and
   smooth random controls, at least 7/10 wins, at most one additional collapse.

One targeted methodological repair is allowed, justified by source diagnostics.
Holdout-driven repairs turn those holdouts into development evidence. Retain failures.
One implementation day/four GPU-hours/four CPU-hours are revisable planning targets;
record measured cost and revise allocation before projected overruns. Larger-design
holdout and multi-fold confirmation remain conditional.

## Prerequisite evidence at handoff

- 21 prerequisite tests previously passed; rerun on the consolidated checkout.
- Five paired source tetrahedron simulator-MPC trials improved over zero and smooth
  random in all pairs; median normalized movement 0.337102, median planning latency
  3.032 seconds/decision. Diagnostic timing; no learned-transfer conclusion.
- 160 evaluation-only source transitions (seeds 8100–8103). Twelve full-snapshot restored
  transitions matched positions, actuator targets and raw rewards exactly. Limited
  smooth-random motion coverage; no learned model yet at handoff.
- Original generated evidence remains in checkout `6781/outputs/dynamics-pilot`.

## Records

Experiments and audits belong in Obsidian `Research/06 Experiment Notebook/`.
Parent: `GNN Dynamics and MPC Generalization Pilot.md`. Code usage and reproduction:
`docs/usage/dynamics_pilot.md`. Record actual outcomes and uncompleted gates here after
execution; generated datasets/checkpoints stay under ignored `outputs/`.

## Source-driven repair declared before execution

The first model improves one-step validation progress over persistence, but its
10-step validation progress MAE is 0.010777 versus persistence 0.009989 (topology
balanced), and error worsens recursively. Source candidate ordering is also poor.
Use the one permitted targeted repair for recursive 10-step observable-state and
cumulative-progress supervision, with the same width-128/three-layer architecture
and fixed data. Start from the validation-selected baseline checkpoint. Keep its
one-step loss; add equal-weight mean recursive normalized state/progress loss.
Batch 32, up to 2,000 extra updates, same optimizer/seed, validation every 200;
select by topology-balanced 10-step forward-progress MAE on validation only.
Keep original weights/metrics. Estimated additional CPU cost: five minutes fitting
plus evaluation; within the four-hour planning target. No hidden targets, history,
new data, holdout tuning or additional repair is included. Stop this iteration if
source gates remain unsuccessful. These source evaluation results are development
evidence; any future confirmation requires new untouched episodes/splits.

## Completed source implementation and audits — October 23, 2026

- Handoff consolidated into checkout `958c`, branch `codex/gnn-dynamics-mpc-pilot`;
  prerequisite code committed/pushed as `d74e7bf`. Original checkout/artifacts retained.
- All eight legacy source buffers audited: eight million stored pairs, 128 sampled
  rows per source (1,024 total), no sampled finite/shape failures. Six v4 plus two
  large v3 checkpoints. Saved settings enable noisy observations without clean
  counterparts and lack episode provenance; backend equivalence remains unverified.
  The fresh clean nominal fallback is justified independently of missing targets.
- Fresh data: 160 independent source episodes, 12,403 transitions. Fitting 96/7,455;
  validation 32/2,441; evaluation 32/2,507. All 17 early-collapse episodes retained.
  All hashes/finite/alias/adjacency/role-seed checks pass; executed routed-command
  reconstruction max error zero. Sixteen selected full-snapshot transitions across
  eight sources reproduce state/progress/raw reward exactly. No blanket snapshot proof.
- Geometry-only rigidity matches 120 ordinary tetrahedron states exactly. Near-collapse
  timing, realistic connector graphs and unseen geometries remain unvalidated.
- First model: three width-128 layers, seed 9321, 2,000 CPU updates, batch 64;
  validation selected update 1,600; fitting 38.699 wall seconds. No hidden actuator
  inputs. Zero of eight sources pass the 25% horizon-10 progress improvement screen.
- Source ranking: four tetrahedron evaluation states, 64 actual CEM candidates each,
  horizon 10, restored simulator vs frozen model. Median Spearman -0.416117; all
  states informative, selected finite-set regret positive in every state. No candidate
  collapse occurred, so failure detection is not established. Cost 13.851 seconds.
- Seventeen terminal labels have shortened elapsed time (.002–.2 seconds). The model
  receives the commanded fixed interval, never realized future terminal duration.
  Constant velocity uses actual elapsed duration; this oracle timing advantage near
  failure is explicit. Reports retain full-duration and shortened-window metrics.
- Broad regression run completed 384 tests with 3 skipped in 190.738 seconds before
  final targeted repair/guard additions. Rerun all affected pilot tests after those.

### Evidence locations

In active checkout `outputs/dynamics-pilot/`: `replay-eligibility-2026-10-23/`,
`source-training-2026-10-23/`, `rigidity-2026-10-23/`, `model-source-2026-10-23/`,
`source-ranking-2026-10-23/`. Generated datasets/weights are intentionally ignored.
Audit/experiment notes are saved in Obsidian `Research/06 Experiment Notebook/`:
05 Handoff and Geometry Rigidity Audit; 06 Replay Eligibility Audit;
07 Nominal Source Data and Snapshot Audit; 08 Observable Model and Source Prediction;
09 Source Candidate Ranking; 10 Multi-step Repair and Stop Decision (final result below).

Measured replay audit wall-time sum: 1,062.907 seconds, largely the two legacy
loads. Source per-episode collection/save sum 122.532 seconds excludes setup/reset;
a no-op resume overwrote the first topology wall total, so exact full collection
wall time is unavailable (cost accounting fixed). Source validator 36.383 seconds.
No GPU used. These stages remain within the four-CPU-hour planning target; CPU
wall time, overlapping subagent work and test time are reported separately rather
than pretending an exact aggregate core-hour measurement.

## Final repair result and remaining gates

- Completed 2,000 extra updates, selected update 1,600 by source-validation horizon-10
  progress MAE (0.009325). CPU fitting 205.888 seconds; architecture/data unchanged.
- Repaired model source progress MAE improved 4.91–17.84% vs persistence. Zero of
  eight meet the required 25% screen. The globally selected baseline is persistence.

| Source | Persistence MAE | First GNN MAE | Repaired GNN MAE | Repair improvement |
|---|---:|---:|---:|---:|
| tetrahedron | 0.007652 | 0.008400 | 0.006615 | 13.55% |
| henneberg_n8_2tube_187 | 0.016914 | 0.016356 | 0.014658 | 13.34% |
| henneberg_n6_2tube_1 | 0.007715 | 0.008246 | 0.007039 | 8.76% |
| usevitch_210272254_p1 | 0.015409 | 0.014136 | 0.013794 | 10.49% |
| henneberg_n5_1tube_1 | 0.008502 | 0.007972 | 0.006985 | 17.84% |
| henneberg_n6_1tube_2 | 0.013799 | 0.013608 | 0.012146 | 11.98% |
| henneberg_n8_3tube_64 | 0.009556 | 0.009189 | 0.008342 | 12.70% |
| usevitch_60243677150_p1 | 0.015154 | 0.015538 | 0.014410 | 4.91% |

- Topology-balanced source-evaluation MAE: persistence 0.011838, repaired
  GNN 0.010499 (11.31% reduction). No nonfinite GNN horizon-10 windows.
- Repaired-model source ranking median Spearman -0.115522; all four
  states informative, positive selected-sequence regret in every state, no candidate
  collapse. Refined CEM candidate sets differ by model; this is not a controlled
  comparison of the two models on an identical full candidate set.
- Both terminal-ending source evaluation windows are collapse false negatives at
  horizon 10. No learned control or near-failure reliability claim is supported.
- Final targeted verification: 43 pilot tests pass after repair and guards. Broad
  run: 384 tests ran successfully with 3 skipped before those final additions. `git diff --check`
  passes. No generated datasets/checkpoints/W&B artifacts are committed.

Tests 4–8 (held-out dynamics, full ranking, learned MPC, larger-design holdout and
confirmation) remain unrun because source prerequisites failed. The one repair is
consumed. Stop this iteration. Next, separately predeclare a source experiment to
distinguish teacher/planner/locomotion coverage from partial observability and
rollout objective error; no evidence yet establishes a required history length or
that privileged actuator targets are necessary. Keep all held-out and final-test
robots untouched by fitting/tuning.
