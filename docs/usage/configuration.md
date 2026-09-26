# Configuration and topologies

All entry points compose Hydra configuration from `config/config.yaml`:

```yaml
defaults:
  - algorithm                          # optimizer, replay, SAC objective
  - environment                        # task, topology, reward, observation settings
  - physics/physical_parameters        # nominal generated-model physics
  - physics/domain_randomization       # see domain_randomization.md
  - cross_validation: disabled         # see cross_validation.md
  - training                           # schedule, evaluation, checkpoints, W&B
  - distillation: disabled             # see distillation.md
  - sac_backend: mlp
  - sim_backend: mujoco
  - platform: local
```

Every key has an inline comment in its YAML file, so check the YAML for the
full list. This page covers the settings that are not obvious from those comments.

## Config groups

| Group | File | Notes |
|---|---|---|
| `sac_backend=mlp` | `config/sac_backend/mlp.yaml` | Flat observations and actuator actions (`truss-mlp` or the hand-authored truss tasks). |
| `sac_backend=gnn` | `config/sac_backend/gnn.yaml` | Graph policy on `truss-graph` with the control graph, virtual node, and PCGrad enabled. |
| `sac_backend=padded_mlp` | `config/sac_backend/padded_mlp.yaml` | Fixed-width MLP on the same graph environment. See [padded_mlp_baseline.md](padded_mlp_baseline.md). |
| `sim_backend=mujoco` | `config/sim_backend/mujoco.yaml` | Native Gymnasium environments. |
| `sim_backend=mjx` | `config/sim_backend/mjx.yaml` | Batch-native MJX training. See [training.md](training.md#mjx). |
| `distillation=disabled`, `distillation=kl` | `config/distillation/` | Multi-teacher policy distillation. See [distillation.md](distillation.md). |
| `platform=local` | `config/platform/local.yaml` | Default. |
| `platform=supercomputer` | `config/platform/supercomputer.yaml` | Submitit/Slurm. See [cluster_and_resume.md](cluster_and_resume.md). |
| `inference/gnn`, `inference/gnn_mjx` | `config/inference/` | Configs selected with `--config-name` for `sac/gnn_infer.py`. |

The `sim_backend` group sets the lower-level key `mujoco_backend`, which is what
the code reads. Always select the backend with `sim_backend=...` on the command
line.

`config/archieved/` holds legacy wrapper configs that are kept only so older
commands still work. Do not use them for new runs.

## Environments

- **`truss-graph`** is a graph-observation environment for any
  `mujoco-truss-gen` preset. Its observations are PyTorch Geometric graphs, and
  it takes one scalar action per graph node. See
  [architecture.md](../design/architecture.md) for how node actions become
  tendon commands.
- **`truss-mlp`** is the flat observation/action version of the same generated
  robots. It is used only for single-topology MLP baselines.
- `sac_backend=padded_mlp` uses `truss-graph`, not `truss-mlp`. Use it for any
  multi-topology comparison between an MLP and the GNN.

## Selecting topologies

Valid names come from `mujoco_truss_gen.PRESETS`. They include `octahedron`,
`tetrahedron`, `icosahedron`, and `solar_array`, plus the enumerated Henneberg
and Usevitch families.

**One topology:**

```yaml
task: truss-graph
truss_topology: octahedron
```

**Several topologies (graph backends only).** Use `truss_topologies`, or its
shorter command-line alias `topologies`. The environment factory expands the
list into tasks such as `truss-graph:octahedron`.

```yaml
task: truss-graph
truss_topologies:
  - octahedron
  - octahedron:realistic   # :realistic applies to this entry only
  - solar_array
```

In zsh, quote list overrides and leave out spaces:

```bash
uv run python sac/train.py sac_backend=gnn 'truss_topologies=[octahedron,octahedron:realistic,solar_array]'
```

**Per-topology budgets.** `steps`, `batch_size`, and `buffer_size` are given
*per topology*. For multi-topology runs the parser multiplies them by the number
of topologies and records the original values as `per_topology_<name>`. Replay
is task-balanced: each topology gets an equal share of both capacity and every
batch. See [replay.md](replay.md) for replay backends and storage placement.

**Evaluation topologies.**

- `eval_task: truss-graph:icosahedron` evaluates on a different task and
  topology. The `task:topology` form sets both in one string.
- `eval_extra_topologies: [...]` adds held-out topologies. These are evaluated
  periodically but never used for collection, replay, or normalization. Their
  metrics are logged as `heldout_episode_*`, alongside `episode_*` (training
  topologies) and `all_episode_*` (both combined).

**Flat MLP limits.** `truss-mlp` accepts several topologies only when their flat
observation and action spaces match; mismatched lists fail early. For mixed
sizes, use `padded_mlp`.

## Realistic models and graph views

- `truss_realistic: true` requests realistic generated models for every
  topology. `:realistic` on a single list entry does the same for that entry only.
- `truss_graph_view: auto` uses physical graph nodes for abstract models and
  logical nodes for realistic models. You can also set `physical` or `logical`
  explicitly.
- `use_control_graph: true` is the default for the GNN and padded-MLP backends.
  It overrides `truss_graph_view` and uses the control graph from
  `mujoco-truss-gen`.
- With the control graph enabled, `control_node_observation_source:
  connector_ball` (the default) observes each realistic logical node through its
  connector ball. Abstract models have no connector balls, so they always use
  physical-node kinematics. Set `physical_node` to use physical-node kinematics
  for realistic models too.

## Optional graph features

`graph_features` in `config/environment.yaml` appends features to the default
six per-node values (relative position and velocity):

| Flag | Adds |
|---|---|
| `node_roles` | `[actuated, passive]` one-hot per node |
| `edge_roles` | `[tube, connector, virtual]` one-hot per directed edge |
| `edge_distance` | Endpoint distance per edge |

Checkpoints record the feature schema, so a checkpoint can only be loaded with
the same flags it was trained with.
