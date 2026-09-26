# Scripts

Run each script with `uv run python <path> --help` to see its full option list.

## Experiment launch and validation

| Script | Purpose |
|---|---|
| `scripts/launch_cross_validation.py` | Launch every leave-one-group-out fold and seed as a Hydra multirun. See [cross_validation.md](cross_validation.md). |
| `scripts/validate_padded_mlp_topologies.py` | Check that the fixed padded-MLP capacity covers the thesis topology set. Rerun after upgrading `mujoco-truss-gen`. |
| `scripts/validate_domain_randomization.py` | Run the domain-randomization validation plan and write machine-readable results. |
| `scripts/run_domain_randomization_training_smoke.py` | Run the three-seed training smoke matrix for each randomization family. |

## Benchmarks

| Script | Measures |
|---|---|
| `scripts/benchmark_mujoco_backends.py` | Native MuJoCo vs MJX step throughput for local truss environments |
| `scripts/benchmark_multi_env_runs.py` | Native repeated environments vs batch-native MJX, across `--num-envs` |
| `scripts/benchmark_mjx_implementations.py` | JAX vs Warp MJX physics through the GNN-SAC MJX adapter |
| `scripts/benchmark_actor_inference.py` | Serialized vs batched GNN actor inference |
| `scripts/benchmark_gnn_replay.py` | Legacy vs direct-collation GNN replay sampling |
| `scripts/benchmark_pcgrad.py` | Current PCGrad projection vs the frozen legacy implementation |
| `scripts/probe_pcgrad_cuda_graph.py` | Whether one production-sized PCGrad task can be captured in a CUDA Graph |

## Figures

`figures/plot_distance_by_topology.py` downloads paper evaluation results from
W&B (`--project`, default `i-suds/paper_results`). It writes the
distance-by-topology plot and summary CSVs to `figures/`. The generated files
are checked in there next to the script.
