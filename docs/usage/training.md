# Training

All training runs through `sac/train.py`. Choose the policy with `sac_backend`
and the simulator with `sim_backend`:

```bash
uv run python sac/train.py sac_backend=mlp steps=10000
uv run python sac/train.py sac_backend=gnn steps=10000
uv run python sac/train.py sac_backend=padded_mlp 'truss_topologies=[tetrahedron,octahedron]' steps=10000
uv run python sac/train.py sac_backend=gnn sim_backend=mjx 'truss_topologies=[octahedron,tetrahedron]'
```

`device` defaults to `cpu` (the supercomputer platform sets `cuda`). For quick
local checks, add `enable_wandb=false`.

## Update scheduling

`replay_ratio` sets how many replay samples are consumed per newly collected
transition. The number of optimizer updates is

```text
updates = replay_ratio * collected_transitions / batch_size
```

Any fractional remainder carries over to the next vector step and is saved in
checkpoints. Transitions enter replay after every vector step.
`update_every_vector_steps` controls how often the accumulated update budget is
spent. The defaults (`config/algorithm.yaml`) are `replay_ratio=10` and
`update_every_vector_steps=1`, meaning updates happen after every vector step.

`iterations` is deprecated. Set it only to reproduce the legacy schedule of one
optimizer update per collected transition.

## Discount factor

`discount` defaults to `0.995` (`config/algorithm.yaml`). Set `discount=null` to
derive it from `episode_length` instead: `(frac - 1) / frac` with
`frac = episode_length / discount_denom`, clipped to
`[discount_min, discount_max]`. With the default `max_steps=1000` that gives
`0.95`, and the MuJoCo wrapper lowers `discount_max` to `0.99`. Reward
normalization uses the same discount unless `reward_norm_gamma` is set.

## Multiple topologies and `num_envs`

`num_envs` is the **total** number of vector environments, and it must divide
evenly by the number of topologies. For example, this runs 500 octahedron and
500 tetrahedron environments behind a single mixed-graph policy inference call:

```bash
uv run python sac/train.py sac_backend=gnn sim_backend=mjx \
  num_envs=1000 'topologies=[octahedron,tetrahedron]'
```

`steps`, `batch_size`, and `buffer_size` are per-topology values. See
[configuration.md](configuration.md#selecting-topologies).

With `pcgrad=true` (the GNN default), critic and actor gradients are computed
separately for each topology, and conflicting gradients are projected before
the optimizer step.

## MJX

`sim_backend=mjx` runs training environments batch-natively on the accelerator
through `mujoco-truss-gen`. Each topology gets its own compiled MJX step
function and a fixed-size state batch. Observations and actions pass between
JAX and PyTorch through DLPack (`mjx_zero_copy=true`).

Constraints:

- MJX requires `mujoco-truss-gen[warp]==0.12.5` (pinned in `pyproject.toml`)
  and training-time rendering must be disabled.
- MJX always uses the control graph (`graph_view="control"`), so it applies only
  to the graph backends.
- Realistic models and fixed-shape runtime domain randomization are supported.
  Randomization that rebuilds the model is not supported (`length_scale` and
  `domain_randomization_params.physical_parameters`).
- Use `sim_backend=mujoco` if you need training-time rendering or
  model-rebuilding randomization.

**Evaluation.** Evaluation uses a separate native MuJoCo environment by default
(`eval_backend=mujoco`). This avoids slow single-environment MJX evaluation and
means `save_video=true` still works while training stays on the accelerator.

**Warp.** JAX is the default MJX physics implementation. On NVIDIA CUDA hosts
you can select Warp instead (`platform=supercomputer` already does this). The
locked environment includes Warp through the `mujoco-truss-gen[warp]` extra
(`warp-lang` in `uv.lock`), so `uv sync --frozen` is enough:

```bash
uv run python sac/train.py sac_backend=gnn sim_backend=mjx mjx_impl=warp device=cuda
```

`warp_graph_mode` defaults to `warp_staged`, the fastest mode in the A100
benchmark. It also accepts `warp` and `warp_staged_ex`. `warp_naconmax` and
`warp_njmax` optionally fix contact and constraint capacities. Warp replaces
only the physics implementation inside the MJX environment; the rest of the
pipeline is unchanged.

## Evaluating checkpoints

```bash
# Native MuJoCo, optional viewer
uv run python sac/gnn_infer.py model=/path/to/final.pt visualize=true

# Vectorized MJX evaluation in waves
uv run python sac/gnn_infer.py --config-name inference/gnn_mjx \
  model=/path/to/final.pt episodes=256 num_envs=256
```

`model` accepts a local path, a W&B artifact reference or URL, or a W&B run-file
URL. See `config/inference/gnn.yaml`.

## Profiling

Set `profiling.enabled=true` to time training phases over a bounded window of
vector steps. The window sizes and optional Chrome-trace export are under
`profiling` in `config/training.yaml`.
