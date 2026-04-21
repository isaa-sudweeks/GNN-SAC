# Experiment Plan

This document outlines the staged experiment plan for the project. Each stage should produce a clear yes/no signal before moving to the next one.

## Stage 0: Problem Lockdown

Goal:

Define the first task, graph schema, action meaning, and reward before writing the graph SAC implementation.

Deliverables:

- finalized node definition
- finalized edge definition
- list of actuated tendon edges
- node feature table
- edge feature table
- action scaling rule
- reward equation
- termination conditions

Exit criteria:

- A single environment wrapper can return one complete graph observation.
- Every actuator command can be mapped from one graph edge action.

## Stage 1: Trusted Fixed-Topology Baseline

Goal:

Establish a known-good SAC result on one fixed topology using SB3 or SBL.

Runs:

- fixed topology
- vector observations
- vector actions
- MLP actor and critic
- at least 3 random seeds

Metrics:

- episode return
- episode length
- control effort
- training curves
- final evaluation performance

Exit criteria:

- The baseline learns a non-trivial behavior.
- The run configuration and metrics are saved reproducibly.

## Stage 2: Minimal Custom SAC Validation

Goal:

Validate the custom SAC implementation before introducing graph-specific complexity.

Runs:

- same environment as Stage 1
- same observation and action spaces
- MLP actor and critic
- matched or comparable hyperparameters
- at least 3 random seeds

Required components:

- replay buffer
- actor loss
- critic loss
- target critic update
- entropy temperature tuning
- evaluation loop
- checkpointing

Exit criteria:

- Custom SAC reaches comparable performance to the trusted baseline.
- Loss curves and entropy behavior look reasonable.
- Implementation passes unit tests for core loss and replay-buffer behavior.

## Stage 3: GNN SAC On One Fixed Topology

Goal:

Replace the MLP models with graph actor and critic models while keeping the topology fixed.

Runs:

- one fixed topology
- graph observations
- edge-level actuator actions
- GNN actor
- GNN critic
- at least 3 random seeds

Comparisons:

- trusted MLP SAC baseline
- custom MLP SAC baseline
- GNN SAC fixed-topology result

Exit criteria:

- GNN SAC learns the same task on one topology.
- Per-actuator actions are valid and correctly mapped to the simulator.
- Permutation tests pass for actor and critic.

## Stage 4: Multi-Topology Training

Goal:

Train one graph policy across multiple robot topologies.

Runs:

- train on a set of related topologies
- evaluate on training topologies
- evaluate on held-out topologies
- at least 5 random seeds if compute allows

Topology split:

- train topologies: small but diverse set
- validation topologies: used for model selection
- test topologies: untouched until final evaluation

Metrics:

- return on seen topologies
- return on held-out topologies
- sample efficiency
- robustness to node and edge order permutations
- action effort

Exit criteria:

- One GNN policy performs meaningfully on held-out topologies.
- Results are better than or competitive with reasonable non-graph baselines.

## Stage 5: Thesis Claim Evaluation

Goal:

Measure the actual claim: topology generalization from graph-structured SAC.

Core questions:

- Does the GNN policy generalize to unseen topologies better than fixed-size MLP alternatives?
- Does the graph representation preserve performance when node and edge orderings are permuted?
- Is performance explained by topology generalization rather than training-set memorization?

Required plots:

- learning curves by method
- final return by topology
- held-out topology performance
- seed variance
- permutation robustness checks
- sample efficiency comparison

Required tables:

- environment/task definition
- model architecture summary
- hyperparameters
- topology split
- final evaluation metrics

Exit criteria:

- Results are strong enough to support, weaken, or reject the thesis claim.
- Experimental limitations are clear and documented.

## Later Extensions

Only consider these after Stage 5 has a clear result:

- multiple behaviors with one policy
- command-conditioned locomotion
- rolling and walking within one network
- dynamic contact edges
- larger topology distributions
- sim-to-real considerations

## Run Naming

Use consistent run names:

```text
{stage}_{method}_{env}_{topology_set}_seed{seed}
```

Examples:

```text
stage1_sb3_sac_mujoco_fixed_seed0
stage2_custom_sac_mujoco_fixed_seed0
stage3_gnn_sac_tensegrity_fixed_seed0
stage4_gnn_sac_tensegrity_multitopo_seed0
```

## Minimum Reproducibility Rules

Every run should save:

- config file
- git commit hash
- random seed
- environment name and topology id
- model checkpoint
- evaluation metrics
- training curves

Generated outputs should go under `runs/` or `checkpoints/`, which are intentionally ignored by git.
