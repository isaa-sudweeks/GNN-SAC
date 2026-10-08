# Tube virtual-node ablation

`graph_features.tube_nodes=true` adds one aggregation hub per structural connected
component in the raw control graph. Connector edges are excluded from membership;
passive structural edges are included. Raw observations stay physical and retain
the legacy six-channel schema, so native and MJX wrappers and old teacher replays
use the same transform. Requires `use_control_graph=true`, `use_virtual_node=true`,
and `graph_features.edge_roles=true`.

Each hub has a dedicated node-type channel, false physical/action masks, and
bidirectional membership edges only to its control instances. Membership edges
use the fourth role (`membership`), zero incidence sign, and zero distance.
The existing global node only connects physical instances and retains rigidity;
critic readout selects it using `global_node_mask`. Physical positions and
velocities are normalized before architectural nodes are added.

Object replay, dense replay, teacher-target shards, checkpoints, and inference
share this feature contract. Legacy teachers receive their original feature view
and produce targets in the original action order; students receive augmented
graphs. Reconstructed roles/signs are checked against replay ordering. Tube labels
are never supplied as numeric input features. Checkpoints with tube features
require a matching schema when resumed.

## Run the development screen

From the repository checkout on ORC (with the implementation available there):

```bash
uv run python scripts/run_routing_canary.py \
  --tube-nodes --run-root "$HOME/nobackup/autodelete/GNN-SAC/runs/tube-canary"

uv run python scripts/launch_tube_ablation.py \
  --stage offline \
  --run-root "$HOME/nobackup/autodelete/GNN-SAC/runs/tube-offline-v1" \
  --cache-dir "$HOME/nobackup/autodelete/gnn-sac-tube-cache" \
  --execute
```

Omit `--execute` to write the manifest and print the Hydra command without
submitting. The default run root is
`~/nobackup/autodelete/GNN-SAC/runs/tube-<stage>-v1`; the default distillation cache
is `~/nobackup/autodelete/gnn-sac-tube-cache`. On a local CPU machine, add `--cpu`
to the canary.

Experiment jobs explicitly enable offline W&B logging, with records under
`<run_root>/wandb/offline-run-*`. The launcher also places W&B artifact staging
and cache under `<run_root>/cache`. Upload those offline records later using
`wandb sync`. The canary disables W&B entirely. Separate development diagnostics
are saved in `tube_diagnostics.json`; they are not automatically added to W&B.

The launcher compares signed versus signed plus tube membership across seeds
1–3: six jobs total, on **one fold only**. It fixes random_5fold/fold_0 (15 training presets, four development holdouts),
10,000 offline updates, attention/global-node settings, teachers, and a
75M-transition KL decay. N7 final tests are excluded. The manifest reports
trainable parameter counts; width/depth stay fixed, so parameter counts differ
slightly because input feature widths increase. No holdout-based checkpoint
selection is performed.

`distillation.offline_only=true` completes offline updates, evaluates the initial
policy, writes `checkpoints/distillation.pt`, and finishes without SAC transitions.
The configured online budget stays 2,000,010 so the KL schedule and run configuration
remain matched to the online protocol. The default teacher paths are defined in
`config/distillation/kl_paper_v4.yaml`.

After training workers return, `--execute` automatically submits one CPU Slurm
job per available checkpoint directory for development diagnostics, and records
submission IDs/results in the manifest. Results appear as `tube_diagnostics.json`
in each job directory. Diagnostic submission is asynchronous; monitor the listed
Slurm jobs for completion. These diagnostics open development teachers, including
holdouts. To rerun one checkpoint directory manually on a compute node:

```bash
uv run python scripts/evaluate_routing_ablation.py \
  --checkpoint-dir /absolute/path/to/job/checkpoints \
  --output /absolute/path/to/job/tube_diagnostics.json
```

Diagnostics report held-out teacher Gaussian KL/action MSE and deterministic
rollout distance/survival over reset seeds 1000–1004, per topology. Compare all
three training seeds. To check whether gains survive online learning, run a fresh
matched offline-plus-online experiment:

```bash
uv run python scripts/launch_tube_ablation.py \
  --stage online \
  --run-root "$HOME/nobackup/autodelete/GNN-SAC/runs/tube-online-v1" \
  --cache-dir "$HOME/nobackup/autodelete/gnn-sac-tube-cache" \
  --execute
```

This stage repeats the same offline training, then collects 2,000,010 transitions.
Use separate run roots for the two stages. It preserves the offline checkpoint for
comparison with online checkpoints.

## Scope and validation

This implements the plan's first, grouping-only experiment. Physical conserved
length/reference/residual channels and the optional shuffled-group control remain
separate follow-up arms. Legacy replays do not store tendon lengths; deriving
lengths directly from anisotropically normalized xyz would violate the plan's
physical-feature contract. Membership alone provides a tube summary, and does
not enforce conservation or supply an exact sum.

Regression tests cover source tube multisets for all 19 development definitions,
representative compiled native graphs, relabeling equivariance, mixed-batch critic
readouts, masks/signs/distances, object/dense feature parity, replay resume,
legacy-teacher distillation, diagnostics, and offline-only completion. Native/MJX structure and actor inference are compared using the same physical
input state (reset noise differs between simulators). GPU MJX/Warp
validation is provided by the canary; run it on ORC before submitting experiments.

Measure actor overhead and prepared tensor storage on the target GPU separately
from simulator timing:

```bash
uv run python scripts/benchmark_tube_virtual_nodes.py \
  --topology henneberg_n8_2tube_127 --device cuda \
  --batch-size 256 --iterations 100 --output /tmp/tube_overhead.json
```

This reports actor batch latency, nodes/edges, prepared tensor bytes, and parameter
counts. It excludes simulator, graph preprocessing, and optimizer memory; use the
training profiler (`profiling.enabled=true`) for online phase timings.
