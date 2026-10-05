# Signed-routing hypothesis experiment

The hypothesis is that a bidirectional control graph does not expose the signed
node-command-to-actuator mapping needed for transfer. This experiment tests that
representation change; it does not assume the hypothesis is correct.

`graph_features.edge_direction` appends +1 along the actual actuator metadata's
source-to-destination edge, -1 for the reverse message, and 0 for connector,
unactuated tube, and architectural virtual edges. It never derives signs from
node numbering or position. Some structural tube edges intentionally have no
direct actuator. Controller commands, physical routing, and action order are
unchanged. The flag requires `use_control_graph=true`.

The three arms all retain node roles, observed edge distance, attention, virtual
node, two 128-wide message-passing layers, PCGrad, normalization, and the same
paper-v4 teachers. Baseline has no edge types/signs; `edge_types` adds tube,
connector, and virtual types; `signed` additionally adds incidence signs.

Use the existing random development `fold_0`, three seeds, and 15 training
topologies. The four development holdouts are tetrahedron, usevitch_212365307,
henneberg_n8_2tube_127, and henneberg_n8_2tube_134. The N7 final test stays isolated.
All arms use 10,000 offline updates and 2,000,010 total online transitions. The
slightly non-round total is `steps=133334` times 15 under the parser's per-topology
budget convention. Batch size, replay capacity, and parallel environments retain
the supercomputer profile (1530 effective environments for 15 equal buckets).
KL decay is explicitly 75M transitions, matching the original random-CV schedule;
shortening the experiment therefore does not also shorten distillation guidance.
Evaluation occurs every 200k transitions, plus post-offline and final evaluation.
Training checkpoints are retained around 800k, 1.6M, and the final step.

```bash
uv run python scripts/launch_routing_ablation.py \
  --run-root /home/isuds/nobackup/autodelete/GNN-SAC/routing-v1 \
  --cache-dir /home/isuds/nobackup/autodelete/GNN-SAC/distillation_cache/routing-v1
```

This validates nine resolved configurations and writes `experiment_manifest.json`.
Add `--execute` to launch with Hydra/Submitit; at most three GPU jobs run at once.
Use a persistent login-node session for the launcher. After training returns, it
submits read-only diagnostics as separate CPU Slurm jobs for available run
checkpoints, including partial failed runs. Diagnostic logs live beside each run.
Do not run teacher checkpoint loading on a login node with a small memory budget.
W&B offline fragments need the usual login-node sync watcher.

The diagnostic script can also be run independently on a compute node:

```bash
uv run python scripts/evaluate_routing_ablation.py \
  --checkpoint-dir /path/to/run/checkpoints \
  --output /path/to/run/routing_diagnostics.json
```

It reads full checkpoints with memory mapping and evaluates post-offline,
first-at-or-after-800k, and final checkpoints. It never selects checkpoints using
held-out performance. On each permitted holdout it measures Gaussian teacher KL
and tanh-mean action MSE on 256 uniformly sampled replay states with sampling seed
314159. It then measures native-MuJoCo deterministic distance/survival with reset
seeds 1000–1004, using configured reset randomization and clean observations/actions
(no wrapper observation/action noise). These additional rollouts are distinct
from ordinary training evaluation. Teacher replay and policies are read-only;
no holdout observations enter the student optimizer. The script rejects N7 final
test overlap.

Compare each seed's signed-vs-types and types-vs-baseline changes at matched
milestones, per holdout and in the equal-topology mean. Better held-out action
prediction together with better survival/distance supports the routing hypothesis.
Low held-out action error but failed rollouts instead prioritizes distribution
shift/dynamics; no reproducible signed-vs-types gain weakens the hypothesis. Keep
seen-task distance as a guardrail and report all seeds. This small development
experiment is a causal probe, not final-test evidence.

Validation includes controller-reversal semantics, metadata reconstruction with
exact preservation of legacy teacher predictions, missing/corrupt metadata errors,
tensor/object feature parity, replay resume, and a two-topology CPU distillation
smoke test. An MJX test exercises signed observations and policy inference.
