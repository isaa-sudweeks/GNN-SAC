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
- [online_candidate_ranking.md](usage/online_candidate_ranking.md): frozen online
  shared/separate model candidate ranking and collapse diagnostics
- [failure_balanced_replay.md](usage/failure_balanced_replay.md): sampling-only
  shared-model continuation and paired frozen source control
- [Experiment17 measured results](usage/dynamics_results/online_candidate_ranking_2026-10-25.md):
  completed candidate-ranking experiment and independent audit
- [Experiments18–19 measured results](usage/dynamics_results/failure_balanced_replay_2026-10-25.md):
  completed failure-balanced replay ablation and paired frozen control outcomes

- [Dynamics and MPC pilot](usage/dynamics_pilot.md): source collection, learned MPC,
  restartable experiments and diagnostic commands
- [Measured dynamics-MPC results](usage/dynamics_results/README.md): shared/separate
  source learning, gamma comparison, frozen diagnostics and figures

## Design

- [architecture.md](design/architecture.md): graph observation, networks, action
  routing, reward, replay
- [performance.md](design/performance.md): what has been optimized and what is
  still open
- [replay_benchmark_results.md](design/replay_benchmark_results.md): September
  2026 A100 evidence for the tensor replay default

## Plans

- [domain_randomization_test_plan.md](plans/domain_randomization_test_plan.md)
- [online_candidate_ranking.md](plans/online_candidate_ranking.md): reviewed source
  diagnostic supporting the dynamics-versus-model-free generalization question
- [dynamics_vs_modelfree_generalization.md](plans/dynamics_vs_modelfree_generalization.md):
  proposed matched comparison; not executed
- [failure_balanced_replay.md](plans/failure_balanced_replay.md): reviewed
  Experiments18–19 sampling ablation and fixed source control protocol
- [generalization_kickoff_prompt.md](plans/generalization_kickoff_prompt.md):
  suggested request for the next matched comparison; not executed

## Dynamics experiment protocols

Completed outcomes are recorded in each protocol; all retain source-only limits.

- [Observable dynamics pilot](plans/gnn_dynamics_mpc_pilot.md)
- [Online learned-MPC source experiment](plans/gnn_dynamics_online_pilot.md)
- [Longer online learned-MPC experiment](plans/gnn_dynamics_online_long.md)
- [Shared versus separate source dynamics](plans/gnn_dynamics_online_multi.md)
- [Discount-only gamma .95 rerun](plans/gnn_dynamics_online_multi_gamma095.md)
- [Frozen failure and gradient diagnostic](plans/gnn_dynamics_multi_diagnostic.md)

## Archive

- [problem_definition.md](archive/problem_definition.md),
  [graph_representation.md](archive/graph_representation.md),
  [experiment_plan.md](archive/experiment_plan.md): early (April 2026) design and
  experiment framing
- [gnn_sac_mujoco_truss_gen_plan.md](archive/gnn_sac_mujoco_truss_gen_plan.md):
  first `mujoco-truss-gen` integration plan
- [gpu_optimization_audit.md](archive/gpu_optimization_audit.md): July 2026
  performance audit with local measurements
