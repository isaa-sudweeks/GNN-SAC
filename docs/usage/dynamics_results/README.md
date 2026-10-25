# Measured source dynamics-MPC experiments

These reports preserve the completed three-topology experiments and the saved-data
diagnostic. They test learning across trained source robots; they do not establish
unseen-topology transfer. The original single-source duration results remain in
[the longer-run protocol](../../plans/gnn_dynamics_online_long.md).

- [Shared versus separate, gamma .995](shared_source_2026-10-24.md): three seeds,
  20,000 real transitions per topology/condition/seed, 360,000 total; tetrahedron
  sharing incurs a fresh-reset reliability cost.
- [Gamma .95 rerun](gamma095_source_2026-10-25.md): identical fixed budgets with only
  gamma changed; per-seed control, prediction and planning-time evidence.
- [Common-discount comparison](discount_comparison_2026-10-25.md): both campaigns'
  saved trajectories scored at .95, .995 and 1.0; the lower gamma is not uniformly
  better across source topologies and conditions.
- [Frozen failure and gradient diagnostic](frozen_diagnostic_2026-10-24.md): identical
  saved action windows and balanced examples; poor near-collapse predictions,
  mixed local gradients, no fitting or simulator execution.
- [Experiment17 online candidate ranking](online_candidate_ranking_2026-10-25.md):
  108 fresh source snapshots and7,884 simulator branches; finite-set scoring errors
  remain even when each model's own proposals include better surviving sequences.
- [Experiments18–19 failure-balanced replay and frozen control](failure_balanced_replay_2026-10-25.md):
  uniform versus50/50 failure-adjacent continuation, common validation and135 paired
  control episodes; no consistent source-control improvement from the repair.

The current recommended next step is the
[matched dynamics-versus-model-free comparison](../../plans/dynamics_vs_modelfree_generalization.md).
Use the original shared gamma=.995 checkpoint family as the reference; keep the
replay continuations as ablations. This next experiment has not run. A suggested
[kickoff request](../../plans/generalization_kickoff_prompt.md) is included.

Static SVG figures are included. `provenance.json` records the original report and
figure hashes; copied SVG bytes are unchanged. Full raw outputs, interactive plots,
replay, checkpoints and verification manifests remain ignored under
`outputs/dynamics-pilot/`. Do not interpret overlapping windows or repeated resets
as independent experimental replications. No evaluation outcome selected weights.

`source_publication_changes.json` records removal of one trailing space in
`sac/dynamics/training.py` for publication; its Python AST is identical. The original
executed bytes remain in the local experiment source snapshots.

`experiments_17_19_provenance.json` records the new reports/config/code hashes and
measured verification counts. Large raw traces/checkpoints and independent audit
outputs remain local and ignored; publication does not rerun or modify experiments.
