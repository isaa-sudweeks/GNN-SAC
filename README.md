# GNN-SAC

Graph neural network Soft Actor-Critic for variable-topology truss robots. One
graph policy controls many `mujoco-truss-gen` robot topologies. The goal is to
show that it generalizes across topologies better than fixed-size MLP policies.

## Install

The only supported install path is [uv](https://docs.astral.sh/uv/). It uses the
checked-in `uv.lock` file and requires Python 3.12 or newer. On Linux, uv
installs the CUDA 13.0 (`cu130`) PyTorch build and the JAX CUDA 13 plugin; on
macOS it uses the native PyTorch and JAX wheels.

```bash
uv sync --frozen
```

`uv run` runs a command inside the locked environment.

## Quickstart

A short CPU training run with the graph policy:

```bash
uv run python sac/train.py sac_backend=gnn device=cpu steps=1000 enable_wandb=false
```

Evaluate a checkpoint:

```bash
uv run python sac/gnn_infer.py --config-name inference/gnn model=/path/to/final.pt
```

Run the tests:

```bash
uv run python -m unittest discover -s tests -v
```

## Entry points

| Script | Purpose |
|---|---|
| `sac/train.py` | Training for every backend. `sac_backend` defaults to `mlp`, so pass `sac_backend=gnn` for the graph policy. |
| `sac/gnn_infer.py` | Checkpoint evaluation and visualization (`--config-name inference/gnn` or `inference/gnn_mjx`). |
| `scripts/launch_cross_validation.py` | Leave-one-group-out topology cross-validation. See [docs/usage/cross_validation.md](docs/usage/cross_validation.md). |
| `sac/gnn_train.py` | **Legacy.** Equivalent to `sac/train.py sac_backend=gnn`. Do not use it for new commands. |

## Configuration at a glance

Configuration is composed with Hydra from `config/config.yaml`. The main switches
are three config groups:

| Group | Options | Selects |
|---|---|---|
| `sac_backend` | `mlp` (default), `gnn`, `padded_mlp` | Policy, replay buffer, and matching environment |
| `sim_backend` | `mujoco` (default), `mjx` | Native Gymnasium MuJoCo or batch-native MJX simulation |
| `platform` | `local` (default), `supercomputer` | Local run or Submitit/Slurm launch with requeue-safe run directories |

```bash
uv run python sac/train.py sac_backend=gnn sim_backend=mjx 'truss_topologies=[octahedron,tetrahedron]'
```

In zsh, quote list overrides as shown above.

## Documentation

See [docs/README.md](docs/README.md) for the full index. Common starting points:

- [Configuration and topologies](docs/usage/configuration.md)
- [Training: backends, MJX, and update scheduling](docs/usage/training.md)
- [Checkpoints, resuming, and cluster runs](docs/usage/cluster_and_resume.md)
- [Distillation](docs/usage/distillation.md)
- [Architecture](docs/design/architecture.md)
