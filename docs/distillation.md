# Multi-teacher Gaussian KL distillation

Distillation is optional and disabled by default. It supports GNN teachers and
a GNN student, with different network widths/depths but matching observation,
graph augmentation, and action conventions.

## Start a run

Supply **full trainer checkpoints** containing `config`, `agent`, and `buffer`.
Agent-only exports such as `final.pt` or `latest.agent.pt` are insufficient.
Map topology names to checkpoint paths; the runner selects only the topologies
in the resolved training split. Missing training teachers are errors. Extra
mapping entries, including held-out teachers, are never opened.

```bash
python sac/gnn_train.py distillation=kl \
  'truss_topologies=[tetrahedron,octahedron]' \
  '+distillation.teachers={tetrahedron:/path/to/tetrahedron/checkpoints/latest.pt,octahedron:/path/to/octahedron/checkpoints/latest.pt}' \
  distillation.cache_dir=/path/to/observation-cache
```

For cross-validation, use the existing `cross_validation` overrides **instead
of** setting `truss_topologies`. Supply a mapping covering every possible
training topology; the resolved fold controls which entries are used.

The `distillation=kl` preset performs 10,000 offline updates, using 256
observations per topology per update. Then ordinary online SAC begins with
an actor KL weight of 1, decaying linearly to zero over half of `cfg.steps`.
These are initial experiment settings, not tuned or empirically validated
hyperparameters. The online schedule uses the trainer's total collected
transition counter, including seed collection; offline updates do not advance it.

Useful overrides:

```bash
# Skip offline pretraining, retaining the decaying online KL objective.
distillation.pretrain_updates=0

# Set an explicit transition count instead of a fraction of cfg.steps.
distillation.decay_steps=2000000

# Tune offline work, sampling, and initial online guidance.
distillation.pretrain_updates=20000 distillation.batch_size=128 distillation.initial_weight=0.1

# Return to the existing SAC behavior, without opening teacher files.
distillation=disabled
```

Offline `distillation.pretrain_updates` is unrelated to the existing
`pretrain_steps`: the latter performs SAC updates on newly collected seed data.
Student replay starts empty, critics are initialized normally, and the seed-data
SAC phase retains its existing behavior.

## Objective and data

`GNNActorCritic.policy_distribution(obs)` exposes the pre-squash diagonal
Gaussian mean and log standard deviation, in active-node order. Distillation
minimizes analytic **KL(teacher || student)**, matching both parameters. Because
both policies use the same invertible `tanh` squash, this is also their squashed
distribution KL. It is computed before any safety projection or actuator routing.

Each loss sums action coordinates per active node, averages nodes within each
graph, and then averages graphs and topology groups equally. Passive and virtual
nodes contribute nothing. No KL clipping or teacher variance floor is applied;
nonfinite parameters/losses fail explicitly. A very narrow teacher/student
distribution can produce a large KL, so inspect the logged KL and gradient norms
when choosing the weight.

Offline training uses the **final teacher checkpoint's distribution** evaluated
on stored observations, not historical actions. All valid observations in each
replay remain eligible, including wrapped ring buffers. Only actor weights and
the actor optimizer change. Adam moments are cleared once when offline training
finishes; actor weights are retained. Evaluation runs immediately afterward.

Online actor optimization adds `lambda * KL` on student replay observations,
with the appropriate frozen teacher for each topology. PCGrad combines SAC and
KL within each topology before projection. Critic, temperature, and target
updates retain their SAC objectives. At zero weight, teacher forward passes stop.

## Memory, caching, and resume

Teacher checkpoints are read sequentially on CPU. The first extraction still
requires enough RAM to deserialize **one full checkpoint**, including its replay.
Observation-only shards avoid retaining every teacher's full transition buffer.
The default cache is `work_dir/distillation_cache`; set `cache_dir` explicitly to
reuse it across runs. Cache directories include checkpoint SHA-256 and shard size.
Keep this generated cache outside version control.

Offline sampling selects a shard proportional to its observation count, then
samples the minibatch uniformly inside that shard. Every observation has equal
marginal probability; observations within a batch share a shard. Only one shard
per topology is resident. Topology losses are accumulated sequentially into one
actor update. Frozen policies live on CPU and are transferred one at a time for
inference, trading transfer overhead for bounded accelerator memory.

Offline checkpoints use `distillation.pt` and also update `latest.pt` and the
agent-only sidecars. `distillation.checkpoint_freq` is measured in offline updates
(default 1,000; zero disables these writes). Normal online checkpoint cadence
remains controlled by `checkpoint_freq`.

```bash
# Repeat the original options and teacher mapping, adding:
resume_from_checkpoint=/path/to/student/checkpoints/latest.pt
```

Checkpoints preserve offline progress, stage, Adam state, sampler RNG, teacher
paths/hashes, and schedule settings. Resume rejects changed teacher sources or
distillation settings and does not repeat finished pretraining or clear online
Adam moments again. Exact offline continuation is covered by regression tests;
online simulator-state resume retains the repository's existing limitations.

Inference uses the ordinary student export and requires no teacher artifacts.

## Metrics and validation

Offline logs use `distillation/kl`, per-task KL, `offline_updates`, and `stage`.
Online training logs include `train/distillation/kl`, per-task KL,
`sac_actor_loss`, `weighted_kl`, `weight`, and `offline_updates`.
Distillation-enabled W&B runs use an automatic event counter with explicit
environment-step/offline-update axes, so repeated updates at environment step
zero are retained. Existing disabled runs keep their logging behavior.

```bash
python -m unittest tests.test_distillation -v
python -m unittest discover -s tests -v
```

Coverage includes analytic KL, gradients and masks, teacher compatibility,
replay/shard handling, held-out exclusion, both actor optimizers, zero-weight
baseline equivalence, offline resume, and a short two-topology native MuJoCo
training smoke test using generated fixture teachers. This validates mechanics;
it does not establish transfer performance for trained specialist checkpoints.
