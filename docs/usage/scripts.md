# Scripts

Most scripts accept `uv run python <path> --help`. The exceptions are marked
**no CLI** below: they take no arguments and start running as soon as they are
invoked, even if you pass `--help`.

## Experiment launch and validation

| Script | Purpose |
|---|---|
| `scripts/launch_cross_validation.py` | Launch every leave-one-group-out fold and seed as a Hydra multirun. See [cross_validation.md](cross_validation.md). |
| `scripts/make_random_cross_validation.py` | Generate a random-fold cross-validation definition (for example `random_5fold`) from an existing one. |
| `scripts/validate_padded_mlp_topologies.py` | **No CLI.** Check that the fixed padded-MLP capacity covers the thesis topology set. Rerun after upgrading `mujoco-truss-gen`. |
| `scripts/validate_domain_randomization.py` | Run the domain-randomization validation plan and write machine-readable results. |
| `scripts/run_domain_randomization_training_smoke.py` | **No CLI.** Starts the full training matrix immediately and writes its outputs. Runs the three-seed training smoke matrix for each randomization family. |
| `scripts/convert_gnn_replay_checkpoint.py` | Convert an object-based (legacy) replay checkpoint to tensor replay without modifying the original. See [replay.md](replay.md). |
| `scripts/validate_production_replay.py` | Check replay performance and learner equivalence using real checkpoints. |

## Benchmarks

| Script | Measures |
|---|---|
| `scripts/benchmark_mujoco_backends.py` | Native MuJoCo vs MJX step throughput for local truss environments |
| `scripts/benchmark_multi_env_runs.py` | Native repeated environments vs batch-native MJX, across `--num-envs` |
| `scripts/benchmark_mjx_implementations.py` | JAX vs Warp MJX physics through the GNN-SAC MJX adapter |
| `scripts/benchmark_actor_inference.py` | Serialized vs batched GNN actor inference |
| `scripts/benchmark_gnn_replay.py` | Legacy vs tensor GNN replay: sampling, insertion, checkpoint cost (`--prototype`) |
| `scripts/benchmark_pcgrad.py` | Current PCGrad projection vs the frozen legacy implementation |
| `scripts/probe_pcgrad_cuda_graph.py` | Whether one production-sized PCGrad task can be captured in a CUDA Graph |

## Cluster batch files

`scripts/orc_benchmark_torchrl_replay.sbatch` and
`scripts/orc_validate_production_replay.sbatch` are the Slurm jobs used to
produce [replay_benchmark_results.md](../design/replay_benchmark_results.md).
They read their inputs from environment variables (`REPLAY_*`).

## Figures

`figures/plot_distance_by_topology.py` downloads paper evaluation results from
W&B (`--project`, default `i-suds/paper_results`). It writes the
distance-by-topology plot and summary CSVs to `figures/`. The generated files
are checked in there next to the script.
