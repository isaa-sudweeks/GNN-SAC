# GNN dynamics and MPC generalization pilot

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

## Existing evidence

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
