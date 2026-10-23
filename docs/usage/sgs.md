# Success Guided Sampling across training topologies

SGS is opt-in. It uses rolling **completed training episode** outcomes to
allocate fresh collection, replay samples, or both. Evaluation outcomes never
enter the sampler. This is an adaptation from reset configurations with PPO to
fixed topology batches with SAC, not a reproduction of the paper's benchmarks.

## Modes

| Override | New experience | Replay minibatch |
|---|---|---|
| `sgs.mode=disabled` (default) | Existing uniform collector | Existing balanced replay |
| `sgs.mode=collection` | Adaptive topology scheduling | Balanced replay |
| `sgs.mode=replay` | Existing uniform collector | Adaptive topology sample counts |
| `sgs.mode=both` | Adaptive topology scheduling | Adaptive topology sample counts |

Supported: `sac_backend=gnn` or `padded_mlp`, tensor or legacy graph replay,
native MuJoCo multitopology and MJX topology buckets. SGS currently rejects
broken-regime replay splitting and fewer than two training tasks. Keep the
training topology set fixed; held-out evaluation topologies are not eligible.

## Exact beta sampler

For topology i, estimate p_i using the latest `sgs.history_size` Boolean
completed-episode outcomes. The kernel is Beta(1 + k*t, 1 + k*(1-t)):

```text
w_i = (p_i + epsilon)^(k*t) * (1 - p_i + epsilon)^(k*(1-t))
logit_i = log(max(w_i, epsilon) + epsilon)
q = softmax(logit / max(temperature, 1))
```

This implements §3.2 and the weight floor/temperature clamp in Appendix B.2 of
[A Balanced Data Diet](https://arxiv.org/html/2610.12465v1#S3.SS2).
`config/sgs.yaml` uses the main-text locomotion defaults: target 0.66,
concentration 5, temperature 2, epsilon 1e-8, history 100. The appendix's
sensitivity table lists a different nominal epsilon; the config follows §3.2.
The calculation uses float64 log space, including the additive epsilon.

Optional `sgs.uniform_mix` mixes q with a uniform distribution. It defaults to
zero, retaining the paper kernel. For example, 0.3 reserves 30% for uniform
coverage. SGS already gives nonzero probability to zero/perfect success.

## Warm-up and outcome definition

Both allocation mechanisms retain their existing uniform/balanced behavior
until **both** `sgs.warmup_steps` total stored transitions have been collected
and every training task has `sgs.min_episodes` completed outcomes. The defaults
are 5000 transitions and 10 episodes per topology. These two parameters are
extensions for startup stability, not paper hyperparameters. Collection also
remains uniform until the initial seed-data pretraining finishes, ensuring that
all arms start their replay-ratio update budget at the same collection boundary.
Warm-up transitions
are not scaled by topology count.

`sgs.success_metric=distance` (default) requires at least **15 meters** of net
forward displacement (accumulated `com_delta_x`) and **400 survived environment
steps** in the same completed episode. Configure these with
`sgs.distance_threshold=15.0 sgs.min_survival_steps=400`. Steps are policy/environment
steps, not internal MuJoCo substeps. A true termination transition does not count
as a survived step: collapse on step 400 fails; collapse on step 401 can qualify
if displacement also reaches 15 meters. Later failure does not erase the minimum
survival duration. Outcomes are recorded at episode completion.

`sgs.require_survival=false` by default. Set it to true to additionally require
survival through the entire episode, rather than just 400 steps. Timeout
truncation is allowed. This leaves rewards, termination, and existing evaluation
success metrics unchanged; use `sgs/success/...` for the sampler's outcomes.
Ensure `max_steps >= 400` for the success target to be reachable.

For environments that provide real Boolean task outcomes, select
`sgs.success_metric=info sgs.success_key=success`. Current truss environments do
not define this success field; some wrappers synthesize zero. Use distance for
these environments. Missing/nonfinite distance or missing/non-Boolean configured
info outcomes fail rather than silently assigning failure.
If all topologies have similar near-zero or near-one success, q stays close to
uniform; the sampler cannot invent a learning frontier.

## Collection and replay semantics

Collection schedules an existing homogeneous topology group for
`sgs.collection_block_steps` consecutive vector steps, then selects again.
Inactive groups keep simulator state, observations, and unfinished episodes.
Native multitask training (including the disabled uniform control) now uses
separate evaluation environments so evaluation cannot overwrite ongoing or
paused training episodes. This also fixes the previous shared-evaluation rollout
corruption in the native uniform baseline. No models are rebuilt or compiled when q changes. Group selection probability
is proportional to q_i/group_size_i, making q describe the expected share of
new **transitions**, rather than episodes or batch invocations. Native collection
uses the corresponding environment indices; MJX's tensor path steps the whole
selected bucket. During warm-up all groups are stepped as before.

As with ordinary tensor collection, a final partial step can simulate a few
extra slots; only budgeted transitions enter replay, normalization statistics,
and SGS outcomes/counters. Prefer budgets divisible by the full vector size to
avoid this tail. A completed episode is counted once, including one ending on
the final budgeted transition.

Replay draws multinomial topology counts summing to the unchanged total batch
size, then samples uniformly **with replacement** inside each selected topology.
Tasks with zero samples are omitted from that update. Buffer capacities and
readiness requirements are unchanged. Collection-only keeps equal replay
sub-batches; it may reuse low-collection topologies' older data more often.

With adaptive replay and `pcgrad=true`, each task gradient is scaled by its
transition fraction before PCGrad averaging. Positive scaling preserves conflict
projection directions and prevents equal task averaging from undoing the
adaptive proportions. Disabled/collection-only PCGrad is unchanged. With
PCGrad disabled, SAC uses the combined adaptive batch as usual; the existing
separate teacher-KL aggregation remains task averaged. Keep teacher settings
fixed across experiment arms.

The global `steps` budget, total batch size, replay-ratio update budget, reward
normalization, and evaluation split retain their existing meanings. `steps` is
still entered per topology and expanded to a total budget by the parser; SGS
changes how that total is distributed, so adaptive collection does **not**
guarantee equal final transitions per topology. Wall time can differ because
only selected buckets are stepped/inferred and graph sizes differ.

## Run the comparison

A native CPU sweep (three seeds, three SGS arms plus uniform control):

```bash
uv run python sac/train.py -m sac_backend=gnn sim_backend=mujoco \
  'truss_topologies=[octahedron,tetrahedron]' \
  sgs.mode=disabled,collection,replay,both seed=1,2,3 \
  steps=100000 enable_wandb=false save_video=false \
  'hydra.sweep.dir=outputs/sgs-comparison/${now:%Y-%m-%d_%H-%M-%S}' \
  'hydra.sweep.subdir=${sgs.mode}/seed_${seed}' \
  'exp_name=sgs-${sgs.mode}'
```

For MJX, select `sim_backend=mjx num_envs=256`; on ORC add the usual
`platform=supercomputer` and launcher settings. Each arm must start fresh, with
identical topology sets, budgets, seeds, rewards, teacher settings, and reset
distributions. Use `eval_extra_topologies=[...]` for a predeclared validation
set, preserving final-test isolation. Sweep runs are independent trainings,
not mode switches within one training trajectory.

## Diagnostics and resume

At each `progress_freq` interval and at the end, `sgs/...` W&B metrics include
per-task success rates, probabilities, completed episode counts, collected
transitions, and replayed transitions. `sgs/adaptive` indicates warm-up is over.
Counters distinguish requested q from realized finite-budget allocation.
Console output prints an SGS progress row; full per-task metrics are in W&B and
sampler checkpoints. The existing `save_csv` option continues to save evaluation
metrics only, not SGS counters.

Checkpoints include histories, counters, settings, both independent RNG states,
and the remaining collection block. Resume requires the same SGS mode, tasks,
and sampler settings. Disabled runs still load older disabled checkpoints.
As with existing trainer resumes, simulator/unfinished episode state is reset;
resumption preserves the curriculum and sampling RNG, not the physical rollout.
Use fresh runs when comparing modes rather than resuming across them.

## Recommended first generalization experiment: 15 m / 400 steps

Use one existing `random_5fold` development fold first, with four modes and
three paired seeds (12 independent training runs). Hold out `fold_0`
(`tetrahedron`, `henneberg_n8_2tube_187`); train on the other eight development
topologies. The three seven-node final-test morphologies stay excluded from
training and evaluation. This tests one development split, not generalization
across all possible morphology splits.

On ORC, from this branch's checkout:

```bash
uv sync --frozen
uv run python sac/train.py -m \
  platform=supercomputer sac_backend=gnn sim_backend=mjx \
  cross_validation=random_5fold cross_validation.held_out_group=fold_0 \
  sgs.mode=disabled,collection,replay,both seed=1,2,3 \
  steps=1000000 max_steps=1000 num_envs=1536 \
  sgs.distance_threshold=15.0 sgs.min_survival_steps=400 \
  sgs.require_survival=false sgs.uniform_mix=0.0 \
  distillation=disabled domain_randomization=false \
  eval_freq=100000 eval_episodes=10 checkpoint_freq=1000000 \
  save_video=false exp_name=sgs-15m400-v1 \
  'wandb_name=${exp_name}-${sgs.mode}-seed${seed}'
```

The `steps` override becomes **8 million total stored transitions per run**
(1 million times eight training topologies), shared adaptively where enabled.
The 1536 environments divide into eight fixed batches of 192 slots. Keep the
same reward, architecture, PCGrad, replay ratio, and reset distribution in every
arm. This is a from-scratch SAC comparison without teacher-guidance confounds.
The platform isolates mode/seed jobs by job number and override hash. Its
checkpoint resume default is `latest`, so rerunning the exact command resumes
these jobs; use a new experiment name for a fresh repeat. ORC's preset uses Warp
and offline W&B logging. CPU MuJoCo/JAX-MJX paths were tested locally; verify the
locked GPU/Warp runtime on ORC before committing to the full sweep.

First inspect whether the 15 m / 400-step success signal develops a nontrivial
range across training topologies. If all outcomes remain failures, SGS remains
uniform and this sweep cannot meaningfully distinguish frontier allocation.
Use the uniform run as the reference for this check; do not tune the success
threshold separately by mode or seed.

Primary comparison: **final-checkpoint held-out distance**, averaged equally
across the two held-out morphologies and reported for each of the three seeds.
Report held-out episode length alongside it to expose early collapse; the
current evaluation success metric is not the SGS success criterion. Also
inspect each topology separately, training distance, realized collection/replay
counts, and the evolution of `sgs/success/...` and `sgs/probability/...`. Choose
using the predeclared final checkpoint rather than each arm's best observed
validation point. A training-only improvement establishes acquisition benefit,
not held-out morphology generalization. If one adaptive mode wins consistently,
confirm it versus uniform on the remaining development folds before opening the
final-test set.

## Distilled SGS comparison (recommended when retaining teacher guidance)

The `kl_paper_v4` teacher mapping and `sgs_teacher_5fold` split preserve the
existing audited 19-topology development workflow. They are separate from this
branch's older ten-topology `random_5fold`: two identifiers in that older split
have no teacher in the audited mapping. Do not mix those definitions.

Use `fold_1`: 15 training topologies and four held-out configurations (two
six-node and two eight-node), retaining those node counts in training. N7 stays
reserved. This evaluates configuration transfer, rather than holding out an
entire node count. Run the following on ORC:

```bash
uv run python sac/train.py -m \
  platform=supercomputer sac_backend=gnn sim_backend=mjx \
  cross_validation=sgs_teacher_5fold cross_validation.held_out_group=fold_1 \
  distillation=kl_paper_v4 \
  sgs.mode=disabled,collection,replay,both seed=1,2,3 \
  steps=1000000 max_steps=1000 num_envs=1530 \
  sgs.distance_threshold=15.0 sgs.min_survival_steps=400 \
  sgs.require_survival=false sgs.uniform_mix=0.0 \
  distillation.pretrain_updates=10000 distillation.initial_weight=1.0 \
  distillation.decay_fraction=0.5 \
  domain_randomization=false eval_freq=100000 eval_episodes=10 \
  checkpoint_freq=1000000 save_video=false \
  exp_name=sgs-distilled-15m400-v1 \
  'wandb_name=${exp_name}-${sgs.mode}-seed${seed}'
```

This submits 12 jobs, each with 10,000 offline distillation updates and **15
million total online transitions** (1 million per training topology). The 1530
slots divide into 15 fixed batches of 102. All arms use the same teachers,
offline budget, and global online KL schedule; at this budget guidance decays
to zero over 7.5 million total online transitions. SGS affects online collection
and/or replay, not offline distillation sampling. The disabled arm still uses
full distillation and is the matched control.

The preset records absolute ORC teacher paths under
`/home/isuds/nobackup/autodelete/GNN-SAC/truss-graph`; override
`distillation.teacher_root` if those checkpoints have moved. Configuration
composition and teacher coverage can be validated locally, but current file
availability and runtime checkpoint compatibility require an ORC preflight.
Missing teachers fail during initialization. Target caches remain run-local by
default, avoiding reliance on any preexisting shared cache.

Compare final-checkpoint held-out distance and episode lengths across paired
seeds as above, and confirm the 15 m/400-step success signal produces nonuniform
probabilities before interpreting any difference as an SGS effect. Existing
evaluation success metrics remain unchanged; the SGS outcome is logged under
`sgs/success/...`. This is a new comparison, not directly comparable to the
from-scratch eight-topology pilot because the training split and teacher
initialization differ.

## ORC GPU placement fix

The October 22 sweep failed on `msg-1-1` (V100, SM 7.0), which the CUDA 13
JAX wheels do not support. `platform=supercomputer` now uses a positive
compatible-partition allowlist; pull the branch and rerun the same command.
See [cluster GPU compatibility](cluster_and_resume.md#supercomputer-platform-submititslurm).
No experiment hyperparameters or dependency versions were changed by this fix.
