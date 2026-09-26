# Documentation

| Folder | Contents | Must match the code? |
|---|---|---|
| `usage/` | How to run things | Yes. Update it in the same PR as the behaviour change. |
| `design/` | How the current system works | Yes |
| `plans/` | Test and experiment plans not yet fully carried out | No, but mark items done as they land |
| `archive/` | Superseded proposals and completed plans, each with a banner | No. Frozen. |

## Usage

- [configuration.md](usage/configuration.md): config groups, environments,
  topology selection, realistic models, graph features
- [training.md](usage/training.md): backends, update scheduling, multi-topology
  `num_envs`, MJX/Warp, checkpoint evaluation, profiling
- [cluster_and_resume.md](usage/cluster_and_resume.md): checkpoints, resuming,
  Submitit/Slurm runs
- [domain_randomization.md](usage/domain_randomization.md): physical parameters
  and randomization families
- [cross_validation.md](usage/cross_validation.md): leave-one-group-out and
  random-fold topology cross-validation
- [distillation.md](usage/distillation.md): multi-teacher Gaussian KL
  distillation into one GNN student
- [replay.md](usage/replay.md): tensor vs legacy replay, storage placement,
  checkpoint conversion
- [padded_mlp_baseline.md](usage/padded_mlp_baseline.md): fixed-width MLP
  comparison baseline
- [scripts.md](usage/scripts.md): launch, validation, benchmark, and figure scripts

## Design

- [architecture.md](design/architecture.md): graph observation, networks, action
  routing, reward, replay
- [performance.md](design/performance.md): what has been optimized and what is
  still open
- [replay_benchmark_results.md](design/replay_benchmark_results.md): September
  2026 A100 evidence for the tensor replay default

## Plans

- [domain_randomization_test_plan.md](plans/domain_randomization_test_plan.md)

## Archive

- [problem_definition.md](archive/problem_definition.md),
  [graph_representation.md](archive/graph_representation.md),
  [experiment_plan.md](archive/experiment_plan.md): early (April 2026) design and
  experiment framing
- [gnn_sac_mujoco_truss_gen_plan.md](archive/gnn_sac_mujoco_truss_gen_plan.md):
  first `mujoco-truss-gen` integration plan
- [gpu_optimization_audit.md](archive/gpu_optimization_audit.md): July 2026
  performance audit with local measurements
