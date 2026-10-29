# Padded MLP baseline

`sac_backend=padded_mlp` trains a fixed-width MLP on one or many morphologies.
It uses the GNN's `truss-graph` environment, six state values per control node,
node-command routing, topology-balanced replay, rewards and evaluation path.
Tensor replay is the default; `replay_backend=legacy` selects `GNNBuffer`.
PyG objects carry variable-size observations; the MLP ignores `edge_index`.
Node slot order remains meaningful, so this baseline is not permutation invariant.

The input is `M * (F + 2) + 1`: flattened node state, existence mask, action
mask and normalized rigidity. With the default `padded_mlp_max_nodes=21` and
six features this is 169 values. Output slots for passive/padded nodes do not
contribute to simulation, critic actions, entropy or distillation. Oversized
morphologies fail before training. The historical 13-topology validator suite
requires at most 21 control nodes and 17 active node actions; these are different
from abstract robot-node counts. No paper morphology split is selected here.

## Specialist and generalist recipes

The named presets require caller-selected morphology inputs. These are small
native CPU examples, not paper hyperparameters:

```bash
uv run python sac/train.py +experiment=padded_mlp_specialist \
  truss_topology=tetrahedron device=cpu enable_wandb=false domain_randomization=false \
  'padded_mlp_hidden_dims=[16,16]' steps=32 seed_steps=8 pretrain_steps=1 \
  max_steps=4 nsubsteps=1 batch_size=4 buffer_size=64 replay_ratio=1 \
  save_csv=false save_agent=false save_video=false checkpoint_freq=0 eval_at_end=false

uv run python sac/train.py +experiment=padded_mlp_generalist \
  'truss_topologies=[tetrahedron,octahedron]' device=cpu enable_wandb=false \
  domain_randomization=false 'padded_mlp_hidden_dims=[16,16]' \
  steps=32 seed_steps=8 pretrain_steps=1 max_steps=4 nsubsteps=1 \
  batch_size=4 buffer_size=64 replay_ratio=1 \
  save_csv=false save_agent=false save_video=false checkpoint_freq=0 eval_at_end=false
```

Alternatively, use `sac_backend=padded_mlp` directly. The generalist preset accepts
existing `cross_validation` overrides instead of `truss_topologies` and requires
at least two resolved training morphologies. `steps`, `batch_size` and
`buffer_size` are per-topology inputs and are scaled once by the parser.
Use the same declared capacity across paired seeds/folds.

## Capacity and launch validation

Startup checks backend/task and simulator/randomization compatibility before
teacher loading. After environment construction it checks every training slot's
spaces, feature width, node identities, action masks and capacity. Additional
`eval_extra_topologies` are inspected with isolated native models without
resetting or stepping them, creating teacher targets or updating normalization.

Validate an explicit set, a cross-validation split, or full/partial run YAML:

```bash
uv run python scripts/validate_padded_mlp_topologies.py --topologies tetrahedron octahedron
uv run python scripts/validate_padded_mlp_topologies.py \
  cross_validation=smoke cross_validation.held_out_group=tetrahedron_group
uv run python scripts/validate_padded_mlp_topologies.py --config /path/to/run.yaml
uv run python scripts/validate_padded_mlp_topologies.py \
  truss_topology=tetrahedron eval_task=octahedron-graph-right padded_mlp_max_nodes=12
```

The report includes ordered morphology contracts, action-node names, maximum
control-node count, capacity, per-topology/effective budgets and trainable
actor/critic parameter counts. Explicit `eval_task` models, including registered
aliases, are checked and included in the contracts. No arguments retains the historical 13-topology
upstream-contract regression suite. This default suite is not a paper split.
The validator inspects nominal native models without training rollouts or MJX
compilation; it does not qualify GPU performance or randomized rollout behavior.

For MJX, select `sim_backend=mjx` and an explicitly compatible fixed-shape DR
profile. Current defaults disable length-scale and model-rebuilding physical
parameter randomization; enabling either requires native `sim_backend=mujoco`.
The supported legacy `physical_parameters.hinge_position_kp` alias is runtime
randomization and does not count as model rebuilding. Validation never silently
disables configured training randomization.

## Paired experiments and distillation

Set the same `pcgrad` choice explicitly in paired GNN and MLP launches. Padded
MLP supports PCGrad but defaults to false, while the GNN preset defaults to true.
The default `[736,736]` padded model has 2,074,828 trainable actor/critic parameters;
recount the actual comparison GNN configuration before claiming parameter matching.
Frozen target critics are excluded from this count.

See [distillation.md](distillation.md) for GNN or padded-MLP expert distillation
into a padded student, offline-only export and online SAC continuation. Use
padded specialists for an all-MLP experiment with the same control interface.
The separate `sac_backend=mlp task=truss-mlp` flat specialist emits actuator
commands; it is a distinct experimental condition and cannot provide a
node-command Gaussian teacher through the same distillation interface.
