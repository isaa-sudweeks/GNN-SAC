# Paper shared-controller campaign: 10M total transitions

Status: prepared and tested on 4 November 2026; no production jobs submitted.
Branch: `paper/general`. W&B project: `i-suds/Paper-Generalists`.

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
offline KL optimizer updates before SAC and online KL decaying from weight 1
to zero over 5M collected transitions. Offline minibatches have 256 samples
per topology: 25.6M replay-sample presentations total, with no new training
environment transitions. Evaluation rollouts, offline work, and teacher
training are additional costs; they do not advance the 10M online counter.

The explicit split in `config/experiment/paper_shared_10m.yaml` matches the
paper baseline CSV: ten training topologies, three seven-node held-out
topologies (`henneberg_n7_1tube_3`, `henneberg_n7_3tube_1`,
`usevitch_1514879`). Held-out topologies enter evaluation only. No held-out
teacher appears in the teacher mapping. The older 19-topology CV/teacher
preset is not used.

Both student architectures have PCGrad disabled. Discount is .95 and the
collapse threshold is .01, matching the experts. Abstract models, action
routing, timing, normalization, domain randomization and its current parameter
settings match the ten saved teacher configurations. The GNN retains its
virtual-node graph architecture; the padded MLP retains its 21-node capacity
and 736/736 hidden widths. No claim of exact parameter matching is made.
Training uses 320 MJX/Warp environments (32 per topology); evaluation uses
native MuJoCo. One A100 per job is requested explicitly through Slurm GRES.
Each arm submits a three-seed array; across four arms up to twelve jobs may
run concurrently if resources permit.

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
paths. The manifest at `outputs/paper_shared_10m/launch_manifest.json` records
source commit, lock hash, commands, seeds, budgets and teacher paths. Only
resource/storage overrides are accepted. For example, `num_envs=160` reduces
the environment count without changing the budget; the count must divide ten.
Use `--manifest PATH` to preserve separate submission records.

To submit the conditions separately instead of all twelve jobs at once:

```bash
uv run python scripts/launch_paper_shared.py --submit --arms gnn_scratch,padded_mlp_scratch
uv run python scripts/launch_paper_shared.py --submit --arms gnn_distill,padded_mlp_distill
```

Run either the all-arm submission or the subset submissions. Preserve override
ordering and run-root settings on resubmission so the existing job directories
and resume checkpoints are reused. This campaign retains supercomputer
standby/requeue/resume behavior. Offline W&B files are written by default;
ORC's existing sync service must upload the new project for live web metrics.
No new sync monitor is started by this launcher.

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

## Validation

- 39 focused tests passed: launch matrix/budget/held-out exclusion, per-evaluation
  export round trips for both backends, checkpoint retention, atomic failure
  handling, offline distillation and existing plotting/checkpoint behavior.
- The full dry run composes four conditions with three seeds, exactly 10M total
  online transitions each, and identical training-only teacher maps.
- Teacher files and saved conventions were checked directly on ORC; all ten
  policy weight loads/finite checks and twenty student compatibility checks passed.
- Full ten-topology GPU throughput and a production training run have not been
  tested in this task. No experiment has been submitted.

The plotting labels were also corrected to the generated presets' actual
multi-triangle physical node counts: 1514879=7, 210272254_p1=8, 60243677150_p1=9.
