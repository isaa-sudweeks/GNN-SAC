# Multi-teacher Gaussian KL distillation

Distillation is optional and disabled by default. It supports GNN teachers and
a GNN student with different network widths, depths, and configurable graph
feature schemas. Each teacher uses the schema saved in its checkpoint, while
the student uses the resolved run configuration. Topology, base observation
width, virtual-node use, action order, normalization, and control conventions
must still match.

## Start a run

Supply **full trainer checkpoints** containing `config`, `agent`, and `buffer`.
Agent-only exports such as `final.pt` or `latest.agent.pt` are insufficient.
Map topology names to checkpoint paths; the runner selects only the topologies
in the resolved training split. Missing training teachers are errors. Extra
mapping entries, including held-out teachers, are never opened.

```bash
uv run python sac/train.py sac_backend=gnn distillation=kl \
  'truss_topologies=[tetrahedron,octahedron]' \
  '+distillation.teachers={tetrahedron:/path/to/tetrahedron/checkpoints/latest.pt,octahedron:/path/to/octahedron/checkpoints/latest.pt}' \
  distillation.cache_dir=/path/to/target-cache
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
Teacher Gaussian means and bounded log standard deviations are computed once
while creating the cache. Offline optimizer updates then run only the student.

Online actor optimization adds `lambda * KL` on student replay observations,
with the appropriate frozen teacher for each topology. PCGrad combines SAC and
KL within each topology before projection. Critic, temperature, and target
updates retain their SAC objectives. At zero weight, teacher forward passes stop.

## Memory, caching, and resume

Teacher checkpoints are read sequentially on CPU. The first extraction still
requires enough RAM to deserialize **one full checkpoint**, including its replay.
Versioned tensor shards store prepared observations plus teacher distribution
targets without retaining every teacher's full transition buffer.
The default cache is `work_dir/distillation_cache`; set `cache_dir` explicitly to
reuse it across student runs. Cache directories and manifests include the
checkpoint SHA-256, cache schema, graph-feature/topology contract, and shard size;
old observation-only caches cannot be loaded as target caches. Keep this generated
cache outside version control.

Offline sampling selects a shard proportional to its observation count, then
samples the minibatch uniformly inside that shard. Every observation has equal
marginal probability; observations within a batch share a shard. The current and
at most one prefetched shard per topology are resident. The prefetched sample
indices are checkpointed with the sampler RNG, preserving exact offline resume.
`target_batch_size` bounds accelerator memory while creating a missing cache;
`shard_size` bounds each resident CPU tensor shard. With CUDA, `pin_memory=true`
uses pinned sampled tensors and non-blocking transfers. Set `prefetch=false` or
`pin_memory=false` for synchronous/diagnostic operation. Topology losses are
accumulated sequentially into one actor update. Frozen policies remain on the
training device for online KL; offline updates do not invoke them after caching.
The cache-build batch size, pinning, and prefetch options may change when resuming;
they do not alter the saved optimizer/sampler contract.
For focused online profiling, enable `profiling.optimization_subphases`; teacher
and student KL work is reported as the `distillation_kl` optimization subphase.
Target caches are opened or built only after checkpoint loading establishes that
the run is still in the offline stage. Resuming an online-stage checkpoint does
not require the offline target cache, even if its configured directory is absent.

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

## Teacher and student feature schemas

`graph_features.node_roles`, `graph_features.edge_roles`, and
`graph_features.edge_distance` may differ independently between teachers and
the student. Distillation prepares two views of the same raw graph: the teacher
view produces the Gaussian target, and the student view receives gradients.
Online KL therefore retains the raw student replay observations alongside the
ordinary student-prepared learner batch. Equal-schema runs reuse the learner
batch directly and do not perform the extra teacher preparation.

For example, specialists trained without node identification can supervise a
student that appends actuated/passive node roles:

```bash
uv run python sac/train.py sac_backend=gnn distillation=kl \
  graph_features.node_roles=true \
  '+distillation.teachers={tetrahedron:/path/to/teacher.pt}'
```

Optional features must be constructible from the raw replay. Node roles use the
action mask and edge distance uses observed xyz coordinates. Edge roles require
`edge_role` metadata; a checkpoint collected without that metadata cannot be
used for an offline student that requires edge roles. Likewise, online teacher
edge roles require the student's raw replay to retain them. Missing metadata
fails during compatibility validation with the affected policy and schema.
Distillation never guesses semantic roles.

Target-cache contracts include both feature schemas and both prepared graph
signatures, so incompatible pairs cannot share tensors. Existing v2 caches
remain reusable for equal-schema teacher/student pairs; mismatched pairs receive
their own schema-pair-keyed v2 cache. New checkpoints persist the resolved
schema pairs. Legacy distillation checkpoints remain loadable only for the
equal-schema case that older code supported.

## Metrics and validation

Offline logs use `distillation/kl`, per-task KL, `offline_updates`, and `stage`.
Online training logs include `train/distillation/kl`, per-task KL,
`sac_actor_loss`, `weighted_kl`, `weight`, and `offline_updates`.
Distillation-enabled W&B runs use one monotonic global step. Offline updates
occupy steps `1..pretrain_updates`; online environment step `k` is logged at
`pretrain_updates + k`. The raw `offline_updates` and per-category `step`
metrics remain available, and runs with distillation disabled keep their
existing environment-step behavior.

```bash
uv run python -m unittest tests.test_distillation -v
uv run python -m unittest discover -s tests -v
```

Coverage includes analytic KL, gradients and masks, teacher compatibility,
replay/shard handling, held-out exclusion, both actor optimizers, zero-weight
baseline equivalence, offline resume, and a short two-topology native MuJoCo
training smoke test using generated fixture teachers. This validates mechanics;
it does not establish transfer performance for trained specialist checkpoints.

## Performance-triggered per-topology release

Set `distillation.performance_decay.enabled=true` to accelerate online KL decay
independently for each training topology. The default is disabled, retaining the
global schedule. Collection, replay allocation, offline pretraining, critic and
temperature objectives are unchanged.

Supply audited teacher references in a Hydra configuration (illustrative values
below; replace them with matched evaluations of your actual checkpoints):

```yaml
distillation:
  performance_decay:
    enabled: true
    distance_ratio: 0.9
    survival_tolerance: 0.05
    patience: 3
    release_steps: 100000
    references:
      tetrahedron:
        distance: 10.0
        survival: 1.0
        horizon: 1000
        teacher_sha256: REPLACE_WITH_CHECKPOINT_SHA256
```

Provide one reference for every **training** topology. `distance` is mean episode
forward displacement; `survival` is the fraction of episodes reaching the full
`horizon` in control steps. Evaluate the frozen teacher under the same simulator,
physics, reset distribution, horizon, and deterministic policy settings as the
student's nominal evaluation, using multiple resets. The loader checks the
checkpoint SHA-256 and positive finite reference performance; it cannot verify
that externally supplied audit metrics were measured under matching conditions.
Zero-distance or zero-survival teachers are unsuitable references and are rejected.
Extra reference entries are ignored and never become scheduling inputs.

At each online evaluation, a topology qualifies when student distance is at least
`distance_ratio * teacher_distance` and student survival is at least
`teacher_survival - survival_tolerance`, with an identical horizon. Invalid or
nonqualifying observations reset its consecutive-evaluation streak. Duplicate
transition counts do not add evidence. After `patience` qualifying evaluations,
its additional multiplier decreases linearly from one to zero over
`release_steps` collected transitions. The effective weight is:

```text
weight(topology, step) = global_linear_weight(step) * release_multiplier(topology, step)
```

Release never reverses if performance later falls, and the global schedule still
forces all guidance to zero at its original deadline. Released topology terms
remain zero in the original loss average; remaining teachers are not upweighted.
Teacher forward passes for a released topology stop. Broken-node regime tasks
remain exempt from KL. The staged broken-node curriculum still waits for the
**global** KL deadline, preserving the existing curriculum timing.

Only training-topology evaluations update the schedule. Held-out distance or
survival cannot trigger release. Offline-stage evaluations do not count.
Evaluation now logs `episode_survival` for a single task and
`<topology>_episode_survival` for topology evaluations. Training logs include
`distillation/weight/<task>` and `distillation/weighted_kl`, with the latter
reflecting the actual topology weights. The existing `distillation/weight` remains
the global upper bound. Checkpoints preserve qualification streaks, last observed
transition counts, and release start times; resume rejects changed references or
adaptive settings. Legacy global-only checkpoints remain compatible when the
feature is disabled.

For an initial ablation, compare enabled versus disabled with identical teacher
checkpoints, offline work, seeds, collection/replay settings, and transition
budgets. Compare development-held-out distance and survival plus mean/worst
training-topology performance. Keep final-test topologies outside tuning and
schedule inputs. These thresholds are experimental defaults, not validated
performance improvements.
