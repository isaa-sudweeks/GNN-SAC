# Paper shared-controller campaign: 10M total transitions

Status: fresh-run protocol v2 prepared on 7 November 2026; no v2 jobs submitted.
Base branch: `paper/general`; v2 changes: `codex/generalist-features-pcgrad`.
W&B project: `i-suds/Paper-Generalists-v2`.
The launcher now uses experiment names ending in `-10m-v2` to keep the revised
settings separate from the original campaign and its resume checkpoints.

## Run matrix and budgets

| Student | Scratch SAC | Expert distillation, then SAC |
|---|---|---|
| GNN | seeds 1,2,3 | seeds 1,2,3 |
| Padded MLP | seeds 1,2,3 | seeds 1,2,3 |

There are 12 runs. **Every run collects at most 10,000,000 total online
training transitions**, including random seed collection. `steps=1_000_000`
is a per-topology request; the parser multiplies it by the ten training
topologies. The final vector collection is clipped to the remaining budget.
Do not override this to `steps=10_000_000`, which would request 100M total.

Both distilled conditions use the **same frozen teacher bank**, with 10,000
offline KL optimizer updates before SAC and online KL decaying linearly from
weight 1 to **0.75 at 10M collected transitions**. The existing schedule is
`beta(s) = max(0, 1 - s / 40_000_000)`, configured with
`distillation.initial_weight=1` and `distillation.decay_steps=40000000`.
Thus beta is 0.875 at 5M; its zero point lies beyond this run's budget. This
changes the online KL coefficient, not the offline update count or a fraction
of environment actions. Offline minibatches have 256 samples
per topology: 25.6M replay-sample presentations total, with no new training
environment transitions. Evaluation rollouts, offline work, and teacher
training are additional costs; they do not advance the 10M online counter.

The explicit split in `config/experiment/paper_shared_10m.yaml` matches the
paper baseline CSV: ten training topologies, three seven-node held-out
topologies (`henneberg_n7_1tube_3`, `henneberg_n7_3tube_1`,
`usevitch_1514879`). Held-out topologies enter evaluation only. No held-out
teacher appears in the teacher mapping. The older 19-topology CV/teacher
preset is not used.

Both student architectures have **PCGrad enabled** for online per-topology
actor and critic gradients; offline distillation is unchanged. Discount is .95 and the
collapse threshold is .01, matching the experts. Abstract models, action
routing, timing, normalization, domain randomization and its current parameter
settings match the ten saved teacher configurations. The GNN enables all of
the following input features in both scratch and distilled arms:

| Switch | Information supplied |
|---|---|
| `graph_features.node_roles=true` | Actuated/passive node roles |
| `graph_features.edge_roles=true` | Tube, connector, virtual and membership edge roles |
| `graph_features.edge_distance=true` | Endpoint distances from observed coordinates |
| `graph_features.edge_direction=true` | Signed actuator routing |
| `graph_features.tube_nodes=true` | One aggregation hub per structural tube |

Node and edge identities here mean semantic roles, rather than arbitrary
numeric IDs. The GNN retains its control graph, global virtual node, attention,
skip connections and combined physical/global critic readout. Tube physics
(`graph_features.tube_physics`) stays disabled: the saved teacher replay has
observation noise and randomized geometry, while that feature requires fresh
clean, fixed-scale teacher replay. The existing metadata reconstruction opt-in
validates and restores legacy teacher edge roles/routing for the student;
teacher inputs and targets retain their saved schemas.

The padded MLP keeps all graph feature switches disabled, its 21-node capacity
and 736/736 hidden widths. No claim of exact parameter matching is made.
Training uses 320 MJX/Warp environments (32 per topology); evaluation uses
native MuJoCo. The existing ORC resource settings are preserved: one GPU per
job through Slurm GRES (`gpu:1`), allowing A100, H100, H200, B200 and L40S
nodes, with P100, V100 and ARM GH200 hosts excluded. Requests retain 128 GB
host RAM and five CPUs per job.
The launcher starts all four Hydra controllers before waiting for any one
condition to finish. Each immediately submits its own three-seed array, so all
twelve jobs can be queued together and may run concurrently if resources permit.
The Slurm scheduler still controls actual start times. Each controller writes
to a separate timestamped log under `outputs/paper_shared_10m_v2/launch_logs/`.
The parent launcher remains running until all controllers finish; keep it in
a persistent terminal or tmux session.

## Selected teachers

The highest logged mean evaluation displacement among the **retained full
trainer checkpoints** with an evaluation at the exact saved environment step
wins for each training topology. Both expert architectures and all three seeds
compete. This selected two GNNs and eight padded MLPs. Earlier historical peaks
whose checkpoint files were pruned cannot be used. Scores average five
rollouts and selection does not impose an additional survival criterion.

| Topology | Teacher | Seed | Saved step | Evaluation displacement (m) |
|---|---|---:|---:|---:|
| `henneberg_n5_1tube_1` | gnn | 2 | 9,901,056 | 59.60 |
| `henneberg_n6_1tube_2` | padded_mlp | 3 | 9,801,216 | 51.83 |
| `henneberg_n8_1tube_57` | padded_mlp | 2 | 9,701,376 | 69.26 |
| `henneberg_n6_2tube_1` | padded_mlp | 3 | 10,000,000 | 81.61 |
| `henneberg_n8_2tube_187` | padded_mlp | 1 | 9,600,000 | 138.37 |
| `tetrahedron` | padded_mlp | 2 | 10,000,000 | 27.91 |
| `henneberg_n8_3tube_64` | padded_mlp | 3 | 9,801,216 | 82.55 |
| `usevitch_60243677150_p1` | padded_mlp | 2 | 9,701,376 | 87.29 |
| `usevitch_210272254_p1` | padded_mlp | 1 | 9,600,000 | 63.10 |
| `octahedron` | gnn | 1 | 9,600,000 | 71.90 |

The paths are pinned in `config/distillation/kl_paper_experts_best.yaml`.
`config/distillation/paper_experts_best_selection.json` records source run URLs,
scores, exact steps, file sizes, checkpoint configurations and selection evidence.
The audit covered 78 runs and 390 retained numbered full checkpoints. ORC
CSV scores and W&B histories agreed at all jointly present candidate steps.
All ten selected archives were readable and loaded with `torch.load` in
memory-mapped CPU mode, confirming full config/agent/replay content. Each
teacher policy was instantiated, its weights loaded and checked finite, and
the existing distillation validator accepted it for both student backends. The
W&B run for the tetrahedron seed-2 teacher is marked failed, but its retained
10M full checkpoint and exact 10M evaluation exist and were verified.

## Launch on ORC

Use a clean checkout of the new branch; leave running experiments' source
checkouts alone. From that checkout:

```bash
uv sync --frozen
export GNN_SAC_RUN_ROOT=/home/isuds/nobackup/autodelete/GNN-SAC/paper-shared-10m
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python scripts/launch_paper_shared.py --dry-run --check-teachers
uv run python scripts/launch_paper_shared.py --submit
```

The script is a dry run by default. Before submitting anything it composes all
selected arms, verifies split/budget/teacher invariants, and checks all teacher
paths. The manifest at `outputs/paper_shared_10m_v2/launch_manifest.json` records
protocol version, source commit, lock hash, commands, seeds, budgets, PCGrad,
graph feature flags, distillation schedule/endpoint weight, W&B destination and
teacher paths. Only
resource/storage overrides are accepted. For example, `num_envs=160` reduces
the environment count without changing the budget; the count must divide ten.
Use `--manifest PATH` to preserve separate submission records.

To submit the conditions separately instead of all twelve jobs at once:

```bash
uv run python scripts/launch_paper_shared.py --submit --arms gnn_scratch,padded_mlp_scratch
uv run python scripts/launch_paper_shared.py --submit --arms gnn_distill,padded_mlp_distill
```

Run either the all-arm submission or the subset submissions. V2 must start
fresh: the GNN feature schema and distillation schedule differ from v1, so old
checkpoints cannot resume under these settings. Do not remove the `-v2` suffix
or rename old job directories to reuse them. For a later resubmission of v2,
preserve override ordering and run-root settings so its own checkpoint directories
are reused. This campaign retains supercomputer
standby/requeue/resume behavior. Offline W&B files are written under
`/home/isuds/robotics/paper-generalists-v2/wandb`, independent of the checkout.
The separate ORC sync monitor `general-v2-wandb-monitor` on `login03` watches
that directory for `Paper-Generalists-v2` runs and starts one W&B beta-sync
worker per fragment. It preserves the run ID and destination recorded by the
trainer; existing `Paper-Generalists` runs stay in their original project.
Monitor logs/state are under
`/home/isuds/robotics/wandb-sync-paper-generalists-v2-20261107/`.
If overriding `wandb_dir`, update the monitor's roots accordingly. The launcher
does not itself start a monitor. Inspect its status with:

```bash
tail -n 20 /home/isuds/robotics/wandb-sync-paper-generalists-v2-20261107/monitor.log
```

## Evaluation and later figures

All 13 topologies are evaluated throughout training at nominal 20k-transition
intervals, five episodes per topology. Actual evaluation can be deferred until
an episode completes; use the logged actual environment step. Held-out means
have their own `eval/heldout_episode_distance` and per-topology metrics.

`save_eval_agent=true` preserves the **exact evaluated policy** with its resolved
configuration, metrics and actual step under
`checkpoints/evaluations/eval_step_N.agent.pt`. These lightweight exports
survive pruning of the five most recent resumable replay checkpoints. The
distilled policy is evaluated and exported at step zero before online SAC.
Optional intermediate offline evaluations use distinct `offline_update_N`
filenames. Full checkpoints every 100k transitions remain available for resume;
evaluation exports are inference-only and cannot resume SAC.

For the requested retrospective held-out selection, choose the best saved
checkpoint per seed using the logged mean displacement over the three held-out
topologies, then average the selected per-topology scores over the three seeds
for the dotplot. Use the same selected shared checkpoint for all topologies
within that seed. Label these as evaluation-selected results; the held-out
scores are not independent test estimates after selection. Selecting a separate
checkpoint for every topology instead would describe a per-topology peak
envelope, not the performance of one shared policy. Record the chosen rule
with the final figure/table and keep final-step values as a separate reference.

## Concurrent submission correction

The original launcher used a blocking subprocess for each arm and therefore
submitted only three seeds at a time. It now starts every selected arm's
controller before waiting on any controller. A controller failure is reported
without cancelling other conditions. Stop an older launcher and cancel its old
array before resubmitting the campaign, to avoid duplicate submissions.

## Validation

V2 regression checks cover PCGrad on all four arms, GNN-only feature switches,
distinct experiment names, the 0.75 endpoint, unchanged teachers/split/budgets,
and manifest provenance. Compare v2 runs with one another; changes from v1
combine features, PCGrad and the distillation schedule and cannot isolate any
one setting's effect. Improved GNN performance remains an experimental question.

On 7 November, all 86 focused tests passed (10 launcher tests plus 76 tests in
`test_distillation`, `test_mlp_distillation`, `test_padded_mlp` and
`test_tube_virtual_nodes`). This includes the two-topology CPU smoke with the
enabled GNN features, legacy-teacher distillation and PCGrad, plus padded MLP
PCGrad/distillation regression checks. A launcher dry run produced twelve
10M-transition jobs; a direct runtime schedule check confirmed weights
1.0, 0.975, 0.875 and 0.75 at 0, 1M, 5M and 10M total transitions.
`git diff --check` passed. This preparation did not submit v2 jobs or qualify
full-campaign GPU throughput.

Original campaign validation:

- The original 39 focused tests passed: launch matrix/budget/held-out exclusion, per-evaluation
  export round trips for both backends, checkpoint retention, atomic failure
  handling, offline distillation and existing plotting/checkpoint behavior.
- Additional regression tests verify that all four controllers start before
  any wait, and that one failed condition does not suppress other submissions.
- The full dry run composes four conditions with three seeds, exactly 10M total
  online transitions each, and identical training-only teacher maps.
- Teacher files and saved conventions were checked directly on ORC; all ten
  policy weight loads/finite checks and twenty student compatibility checks passed.
- Full ten-topology GPU throughput and a production training run have not been
  tested in this task. No experiment has been submitted.

The plotting labels were also corrected to the generated presets' actual
multi-triangle physical node counts: 1514879=7, 210272254_p1=8, 60243677150_p1=9.

## Shared scratch GNN architecture ablations

Use `scripts/launch_generalist_ablation.py` with the launch manifest from the
reference scratch campaign. It composes each of seven arms for all three
recorded seeds and checks that each changes only the targeted setting and
the run identifiers/output paths. The full scratch GNN serves as the
reference. The masks retain the input channels, graph layout, and action
mask. Uniform attention uses per-destination mean aggregation in the actor
and critics. The legacy attention-off setting uses sum aggregation.

```bash
uv run python -m scripts.launch_generalist_ablation \
  --baseline-manifest /path/to/baseline/launch_manifest.json \
  --baseline-id 15267215 \
  --run-root /path/to/new/ablation-run-root \
  --manifest /path/to/preflight.json
```

On ORC, use a fresh output root and a new manifest path with `--submit`
from a persistent login session. The launcher starts all seven three-seed
sweeps together, preserving the baseline's scheduler resources and W&B
project. Keep the source checkout and logs for reproducibility. Compare
against the matched baseline seed at the same total transition budget.
