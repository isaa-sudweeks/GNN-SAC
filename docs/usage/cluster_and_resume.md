# Checkpoints, resuming, and cluster runs

## Checkpoints

Training writes resumable checkpoints to `${work_dir}/checkpoints` every
`checkpoint_freq` environment steps. Each checkpoint contains:

- agent weights and optimizer states
- the replay buffer
- trainer counters, logger state, and RNG state
- resolved config metadata

`latest.pt` is updated every time a numbered `step_<N>.pt` is written.
Lightweight `*.agent.pt` companion files contain only the inference weights.
`checkpoint_keep_last` prunes both kinds of file together. Periodic checkpoints
are written in a background thread (`checkpoint_async=true`); the final
checkpoint is written synchronously.

A small sidecar, `checkpoints/latest.metadata.json`, records checkpoint progress
so tools can read the step count without loading the replay buffer.

With `finite_checks=true` (the default), the agent refuses to save or load
non-finite model, optimizer, or temperature state. Training also stops as soon
as a loss, gradient, action, or replay sample becomes non-finite.

## Resuming

Resume by rerunning the same training config with the same `work_dir`, or by
giving an explicit checkpoint path:

```bash
uv run python sac/train.py sac_backend=gnn work_dir=/path/to/run resume_from_checkpoint=latest
uv run python sac/train.py sac_backend=gnn resume_from_checkpoint=/path/to/run/checkpoints/step_50000.pt
```

- The replay buffer, optimizer state, and pending update budget are restored.
- MuJoCo environment state is not serialized, so a resumed run starts a fresh
  episode at the restored global step.
- With W&B enabled, a resumed run reuses the previous W&B run ID from
  `${work_dir}/wandb_run.json` (or the checkpoint metadata), so logs append to
  the original run.
- `set_wandb_offline=true` forces offline logging. The supercomputer platform
  turns it on by default.

Padded MLP checkpoints also validate the saved input capacity and node feature/
action dimensions. Repeat the resolved architecture and environment settings
when resuming or loading an export. `init_from_agent=/path/to/latest.agent.pt`
loads model/alpha only for a fresh training run and cannot be combined with
`resume_from_checkpoint`.

Offline-only distillation writes `distillation_complete.pt` plus `latest.pt`
and agent companions after its final evaluation, even with periodic saving
disabled. Resume with `distillation.offline_only=true` to finish interrupted
imitation or reopen a completed run without entering SAC. Completion metadata
records the distillation stage and offline-update budget; the Slurm launcher
uses that state to skip completed offline-only jobs resumed from `latest`.

## Supercomputer platform (Submitit/Slurm)

`platform=supercomputer` launches Hydra multiruns through Submitit with
`--requeue`. It also uses a stable `work_dir`, `resume_from_checkpoint=latest`,
`device=cuda`, and `mjx_impl=warp`, so preempted jobs resume automatically.

```bash
uv run python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx --multirun
```

**Run directories.** Each multirun job is isolated under

```text
${run_root}/${task}/${exp_name}/seed_${seed}/job_<number>_<override-hash>
```

The hash comes from the job's Hydra overrides, and the job number separates
duplicate configurations within one sweep. A requeued job keeps its directory,
so it resumes only its own checkpoint and W&B run.

**Storage and account.** Set `GNN_SAC_RUN_ROOT` to put runs on shared persistent
storage. The default Slurm account is `nusey`. Override cluster-specific values
on the command line:

```bash
GNN_SAC_RUN_ROOT=/scratch/$USER/gnn-sac-runs \
uv run python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx --multirun \
  hydra.launcher.partition=gpu hydra.launcher.account=my_account
```

**Skipping completed jobs.** Before submitting, the launcher checks each job
directory. Jobs whose saved checkpoint step has already reached `steps` are
skipped.

- The check reads `latest.metadata.json`. Older runs without that file fall back
  to the highest complete `step_<N>.pt` filename.
- Missing or malformed metadata counts as incomplete, so the job is submitted.
- Job numbers always refer to the full sweep. To reuse existing directories,
  rerun the same sweep with the overrides in the same order.

To submit every job regardless of progress:

```bash
uv run python sac/train.py platform=supercomputer --multirun \
  hydra.launcher.skip_completed_jobs=false
```

### Realistic octahedron sim-to-real sweep

```bash
uv run --frozen python sac/train.py --multirun +experiment=sim_to_real_octahedron
```

This submits six jobs, one per seed (1, 2, and 3) in each arm, with 10 million
training transitions per job. Slurm can assign any compatible GPU. The
supercomputer launcher excludes the P100 and V100 nodes for CUDA 13
compatibility. The preset uses the realistic octahedron in MJX Warp,
enables the expert/generalist graph features (tube nodes, node and edge roles,
edge distance, and signed routing), and samples the internal angle-bisector
hinge position gain uniformly from 6 to 8 at reset. It uses the v3 collapse
threshold of 0.01, records native MuJoCo evaluation video, and writes offline
W&B runs to the `sim_to_real` project.

Jobs 0–2 are the baseline. Jobs 3–5 subtract
`edge_extension_weight * sum(max(L - L_nominal, 0)^2)` from the existing
reward each step, with the sum taken over the undirected physical control-graph
edges. Lengths are in meters and their nominal values come from the compiled
model's initial configuration. Both native MuJoCo evaluation and MJX training
report the weighted reward component as `edge_extension_penalty` when enabled.
The initial weight of 1.0 is a trial value. The extension term is a geometric
proxy for collapse risk.
