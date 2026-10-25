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

Static SVG figures are included. `provenance.json` records the original report and
figure hashes; copied SVG bytes are unchanged. Full raw outputs, interactive plots,
replay, checkpoints and verification manifests remain ignored under
`outputs/dynamics-pilot/`. Do not interpret overlapping windows or repeated resets
as independent experimental replications. No evaluation outcome selected weights.

`source_publication_changes.json` records removal of one trailing space in
`sac/dynamics/training.py` for publication; its Python AST is identical. The original
executed bytes remain in the local experiment source snapshots.
