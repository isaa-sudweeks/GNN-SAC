# Audit-backed performance-decay pilot

The paired experiment uses the audit-selected `distillation=kl_paper_v4` mapping
from the October 14 refresh, the 19-development-topology `random_5fold` definition,
and `fold_1`. The four development holdouts are `henneberg_n6_1tube_1`,
`henneberg_n6_3tube_1`, `henneberg_n8_2tube_187`, and `henneberg_n8_3tube_11`.
All N7 final-test topologies remain excluded. Collection/replay allocation,
teachers, initial offline pretraining, and online budgets match between arms.

From an ORC checkout of `codex/per-topology-distillation-decay`:

```bash
uv sync --frozen
uv run python scripts/launch_performance_decay_pilot.py
```

This checks the audit evidence and all 15 training checkpoint files before
submitting six Slurm jobs: global/adaptive decay × seeds 1, 2, 3. It saves a
preflight manifest under `<run_root>/launch/`. It opens no
held-out or final-test teacher files. Missing files, changed model weights,
changed policy configuration, environment settings, or MuJoCo package versions
abort before submission. The script memory-maps checkpoint tensors during
preflight; normal distillation initialization retains its existing full-checkpoint
loader. Preflight does not replace a full GPU execution canary.

To inspect the configuration without accessing cluster checkpoint files:

```bash
uv run python scripts/launch_performance_decay_pilot.py --dry-run \
  --manifest /tmp/performance-decay-pilot.json
```

On ORC, `--check-only` verifies checkpoint files without submitting.
`--teacher-root /absolute/path` relocates the existing checkpoint tree;
`--run-root /absolute/path` changes the isolated experiment output root.
The default root is
`/home/isuds/nobackup/autodelete/GNN-SAC/runs/performance-decay-v1`.
Hydra separates arms/seeds by job identity and override hash; matching reruns
resume their own checkpoints. W&B names include the arm and seed. The inherited
supercomputer preset uses offline W&B logging, standby QoS, account `nusey`, one
GPU, and 128 GB RAM per job. This launcher does not start a W&B uploader.

The pilot restricts Slurm to `cs,cssp1,cs2,cs3,dw,dwmatrix,eng,m13l,m13h`.
The October 22 inventory lists A100, H100/H200, B200, or L40S GPUs in these
x86 partitions. CUDA 13 requires compute capability 7.5 or newer, so the
V100 (`msg`) and P100 (`m9g`) partitions are unsuitable. The untyped `m8g`
and ARM `mgh` partitions are also omitted. The training entry point checks
the allocated GPU before constructing environments and reports unsupported
hardware directly. The earlier `ptxas too old` message on `msg-1-1` was
preceded by `ptxas does not support CC 7.0`: it was a hardware mismatch,
not an outdated bundled assembler.

These resource settings live in the preset, preserving the original CLI
overrides, run directories, and resume identities. Relaunch with the same
command above; existing checkpoints are reused where available. Changing
resource settings does not guarantee immediate standby GPU availability.

The exact submitted command is:

```bash
uv run python sac/train.py --config-name performance_decay_pilot -m \
  distillation.performance_decay.enabled=false,true seed=1,2,3
```

## Budget and schedule

The repository scales `steps`, `batch_size`, and `buffer_size` by the number of
training topologies. Therefore `steps=2000000` means 30 million total transitions
for this 15-topology fold, not two million total. The global KL duration uses
`decay_fraction=1.0` with `decay_steps=null`, reaching zero at that 30M deadline.
Both arms perform the same 10,000 offline distillation updates first.
There are 1,530 parallel environments (102 per training topology).

Evaluation runs every 1M total transitions with 15 episodes per topology. The
adaptive arm requires three consecutive qualifying evaluations, at least 90% of
audited teacher mean distance, and survival at least as high as the teacher's
observed rate (`survival_tolerance=0`). Its extra multiplier then reaches zero
over 100,000 transitions, capped by the global schedule. Ten training teachers
have positive audited distance and survival and can trigger release. The other
five retain global decay in both arms. This preserves the experiment's isolation:
poor-teacher imitation removal is not introduced at the same time.

## Audit provenance and comparability

`config/performance_decay/paper_v4_audit.yaml` contains per-teacher references and
the common audited environment settings. The compact evidence record
[`paper_v4_performance_audit.json`](../experiments/paper_v4_performance_audit.json)
contains all 19 selected models' episode records, exact policy settings, model
hashes, source-result hashes, and historical runtime provenance. Unchanged
teachers have 30 reset episodes; replacement candidates have 15 confirmation
resets. Repeated seed observations are deduplicated, not pooled twice.

The original audit lives at
`/Users/isaacsudweeks/.codex/worktrees/89fa/GNN-SAC/outputs/teacher-audit-20261014`.
It used native MuJoCo, deterministic mean actions, 1,000 control steps, noise and
length scaling disabled, and randomized initial x/y translation and yaw. The
pilot restores those exact physics/termination/reset settings; notably the
termination threshold is `critical_eig_threshold=0.01`. It opts into evaluation
randomization so teacher and student references use the same reset distribution.
Training uses MJX/Warp; periodic evaluation uses native MuJoCo. Online resets
advance normally, rather than reusing the historical audit seeds as tuning data.

Converted tensor-replay checkpoints have different whole-file hashes while
preserving the audited model weights. References therefore bind to the original
audit's sorted model tensor hash plus policy configuration. Normal distillation
checkpoints additionally retain the actual loaded file SHA-256 for resume.
No audit scores are inferred from W&B endpoint summaries or assigned from a
different checkpoint. Negative-distance or zero-survival references are marked
ineligible for ratio-based release instead of being fabricated or divided by.

The audit used the same locked dependency file, but its source runtime included
tube-feature additions absent from this branch. The evidence preserves those
historical source hashes. Preflight checks the currently loaded model tensors,
policy settings, and MuJoCo package versions; it does not claim to reproduce the
historical rollout scores or prove improved morphology generalization.

Compare held-out distance/survival and learning-curve AUC between matched seeds,
then mean/worst training-topology performance and per-topology release weights.
Use the N7 set only after the method and settings are frozen.
