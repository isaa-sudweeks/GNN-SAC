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
