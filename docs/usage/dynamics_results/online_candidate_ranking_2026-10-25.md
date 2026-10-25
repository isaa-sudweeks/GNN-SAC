# Online dynamics candidate ranking — measured results, October 25, 2026

**Both shared and separate models make finite-horizon scoring errors that favor collapsing sequences.**

The reviewed fixed experiment completed in 13.86 minutes: 36 source episodes, 18 paired fresh resets, three training seeds, three source robots, 108 snapshots and 7,884 simulator candidate branches. No fitting or unseen robots.

This extends Experiment09 (older offline model, no collapsing candidates) and Experiment16 (newer models, executed-action prediction only). Protocol and review: [frozen plan](../../plans/online_candidate_ranking.md). Notebook: `My vault/Research/06 Experiment Notebook/GNN Dynamics Pilot - 17 Online Candidate Ranking - 2026-10-25.md`.

## Direct decision evidence

| Robot | Model | Snapshot tests | Selected collapsing sequence | Missed selected collapse | Own pool had surviving alternative | Own survivor had better true score |
|---|---|---:|---:|---:|---:|---:|
| tetrahedron | shared | 36 | 4 | 4 | 4 | 4 |
| tetrahedron | separate | 36 | 3 | 3 | 3 | 3 |
| henneberg_n6_2tube_1 | shared | 36 | 15 | 14 | 9 | 9 |
| henneberg_n6_2tube_1 | separate | 36 | 11 | 11 | 6 | 6 |
| henneberg_n8_3tube_64 | shared | 36 | 15 | 14 | 13 | 13 |
| henneberg_n8_3tube_64 | separate | 36 | 13 | 13 | 11 | 11 |

Across61 selected collapsing sequences,59 were predicted to survive. In46 cases, the selecting model already proposed a surviving alternative with a better true task score. The smallest true-score advantage among those46 was 0.941020. This subset demonstrates a score-ranking error within the model’s own candidate pool, rather than simply an absence of surviving proposals or a survival/progress preference.

In the other15 selected-collapse cases, the model’s own32 proposals all collapsed, while the73-sequence union contained a survivor. Candidate coverage and scoring can both matter; the union contained a surviving sequence at every snapshot. Neither observation establishes a globally safe policy or long-term recovery.

Mean regret was higher for shared models on all three topology means, but shared regret was higher in only two of three seeds for each topology. Do not infer uniform sharing degradation or a statistically established difference from these small samples.

## What was held fixed

Final20,000-transition gamma=.995 checkpoints; original normalization and physics; 80-control-step episodes (16 seconds); H10 planning (2 seconds);16 CEM candidates in two iterations. Both models score identical73-sequence unions:32 proposals per model, one zero sequence and eight smooth-random sequences. New reset identities are excluded from previous fitting/evaluation roles. No exploration or online updates.

CEM selections here are recreated with a common proposal seed and the generating controller’s predecision warm start. They are not necessarily the actual episode’s original plan; original plans are saved. All73 include duplicates; unique-sequence and candidate-provenance sensitivity metrics are retained. Full raw and grouped seed/generator/stratum rows are in results.json.

## Verification and limits

Independent audit:77,157 checks pass, including reward/metric recomputation, paired resets, selected indices, duplicate sensitivity and89 preserved source/checkpoint hashes. The runner also verifies exact recorded continuation, candidate-order independence, fixed models and unchanged live state. All8 new diagnostic unit tests and57 existing dynamics tests pass; git diff --check passes.

Actual terminal durations were retained:2,328 candidate branches include a shortened terminal interval. Learned rollouts use the nominal .2-second interval. This timing approximation remains part of the tested method, not a newly corrected model.

Branches execute whole sequences open-loop; deployed MPC executes the first action and replans. These are108 correlated states drawn from36 episodes, not independent candidate replications. Selection includes hindsight-labeled precollapse states; the rates are diagnostic, not an estimate of ordinary deployment failure probability. No unique causal explanation or effective repair has been established.

## Relation to the research goal

This test identifies a source-control limitation. It does **not** establish whether dynamics generalizes more easily than model-free control. The next-stage [comparative design](../../plans/dynamics_vs_modelfree_generalization.md) requires a source-trained GNN-SAC generalist with the same source topology access, real interaction budget, task reward, physics and paired unseen evaluation states. Existing specialists or differently trained generalists cannot substitute for that comparator. Keep fitting and tuning out of held-out/final-test robots.

A targeted source repair would need to improve collapse-sensitive scoring on independent validation states, followed by fixed-budget control checks. This experiment does not authorize or execute such a repair; avoid replacing the central matched transfer comparison with an open-ended series of diagnostics.

## Artifact locations

Full local run: `/Users/isaacsudweeks/.codex/worktrees/96de/GNN-SAC/outputs/dynamics-pilot/online-candidate-ranking-2026-10-25`. `report.md` contains every topology/seed summary; `results.json` includes per-state native/pooled/unique/provenance metrics; `verification.json`, `independent-verification.json` and `independent-own-proposal-coverage.json` preserve audits and own-pool comparisons. Per-state arrays contain all actions/scores/flags; trusted local pickle files retain full snapshots and simulator reward/state traces. Raw artifacts remain ignored.
