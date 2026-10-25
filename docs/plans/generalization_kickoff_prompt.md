# Kickoff prompt: matched dynamics versus model-free generalization

This is a suggested request for a separate task. It has not been executed.

```text
Implement and run Experiment 3: test whether GNN dynamics + MPC generalizes across robot morphologies more effectively than model-free GNN-SAC at matched training-data budgets.

First follow AGENTS.md, read the Obsidian experiment index and Experiments 14–19, and read docs/plans/dynamics_vs_modelfree_generalization.md. Write a fixed protocol, have a subagent review it, incorporate corrections, benchmark the cost, then execute and report it.

Use the original shared gamma=.995 dynamics checkpoint family as the reference. Compare it with a source-trained GNN-SAC generalist on the same three source robots: tetrahedron, henneberg_n6_2tube_1 and henneberg_n8_3tube_64. Match task reward, physics, observable information, action space and real fitting interactions. Audit existing checkpoints before reuse; do not substitute specialists or differently trained generalists.

Use three seeds and, where feasible, the existing 1,600/4,000/8,000/12,000/20,000-transitions-per-source checkpoints. Evaluate familiar sources and predeclared unseen octahedron and henneberg_n8_1tube_57 on paired starts, with no adaptation or holdout-driven selection/tuning. Keep final seven-node robots untouched. Start from the recorded 80-step episodes and 10-step MPC horizon unless the reviewed source-only protocol explicitly changes them.

Report survival, displacement, task return and source-to-held-out changes per robot/seed/budget, plus training compute and planning latency. Distinguish weak source control from transfer failure. No automatic repairs or budget extensions. Record the protocol and verified results in the same Obsidian experiment folder and update repo documentation. Resolve routine implementation choices autonomously.
```
