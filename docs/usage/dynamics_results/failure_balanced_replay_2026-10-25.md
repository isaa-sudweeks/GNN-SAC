# Failure-balanced replay and frozen source control — October25,2026

**Failure-balanced replay did not provide a consistent source-control improvement at this budget.**

Experiments18–19 completed in 18.47 elapsed minutes. Three original shared-model training seeds, three source topologies, six continuation jobs; 36,000 added shared Adam updates. No new fitting interactions or unseen robots. The user reserved Experiment3; no model-free/transfer comparison was executed.

Summed worker wall time including preparation, fitting/checkpoints and evaluation: 54.36 minutes with three concurrent workers. Post-run audit and reporting are additional. There were no campaign failures or operational restarts.

Plan: [reviewed protocol](../../plans/failure_balanced_replay.md). Notebook: `My vault/Research/06 Experiment Notebook/`, Experiments18 and19.

## Measured interpretation

The six-node source improves in two seeds and regresses in the third. Tetrahedron mean movement is lower in all three seeds under balanced replay; eight-node discounted task return is lower in all three. Across the three source means, balanced sampling is not a general repair. These are descriptive three-seed results, not a significance claim or a verdict on other sampling fractions, budgets or losses.

Both continuation arms have28/45 total control collapses. Pooling would conceal the six-node improvement and regressions on the other sources. The original reference has27/45, so extra fixed-replay fitting itself is not a uniform gain.

The eight-node fixed-pool selected-collapse metric improves from8/9 (uniform) to4/9 (balanced), but closed-loop collapses rise from14/15 to15/15. That improvement does not establish useful control: the pool comes from the original policy and open-loop branch outcomes differ from the repaired controller’s replanning/state distribution. Six-node terminal sensitivity ties at6/8 missed H10 events, while eight-node sensitivity is worse for balanced replay (5/8 missed versus2/8). Tetrahedron validation has no terminal-ending H10 cohort, so its failure sensitivity is inconclusive. Ordinary-state progress error is higher for balanced than uniform on each source mean. No unique cause of the control differences is identified.

Retain these results as a test of the failure-emphasis hypothesis; do not adopt 50/50 sampling as the general default on this evidence. No further repair or Experiment3 was launched.

## Frozen closed-loop outcomes

| Source | Model | Collapses /15 | Mean movement | Mean discounted task return |
|---|---|---:|---:|---:|
| tetrahedron | original | 1 | 0.484970 | 0.362605 |
| tetrahedron | uniform | 1 | 0.500606 | 0.370271 |
| tetrahedron | balanced | 2 | 0.468956 | 0.292106 |
| henneberg_n6_2tube_1 | original | 13 | 0.341931 | -0.368532 |
| henneberg_n6_2tube_1 | uniform | 13 | 0.348956 | -0.369488 |
| henneberg_n6_2tube_1 | balanced | 11 | 0.363105 | -0.263781 |
| henneberg_n8_3tube_64 | original | 13 | 0.294892 | -0.442082 |
| henneberg_n8_3tube_64 | uniform | 14 | 0.366573 | -0.379554 |
| henneberg_n8_3tube_64 | balanced | 15 | 0.320464 | -0.503069 |

Movement is forward unique-physical-node mean displacement divided by initial diagonal. Every row has three model seeds and five independent paired resets per seed. Episodes last at most80 controls/16 simulated seconds, with H10/2-second CEM planning. Artificial limits preserve their truncation labels; all true collapses are retained.

## Paired balanced minus uniform effects, every seed

| Source | Seed | Collapse difference /5 | Movement difference | Discounted return difference |
|---|---:|---:|---:|---:|
| tetrahedron | 0 | +2 | -0.052656 | -0.338293 |
| henneberg_n6_2tube_1 | 0 | -2 | +0.094679 | +0.421665 |
| henneberg_n8_3tube_64 | 0 | +0 | -0.096769 | -0.152059 |
| tetrahedron | 1 | -1 | -0.023072 | +0.118327 |
| henneberg_n6_2tube_1 | 1 | -1 | +0.125674 | +0.274824 |
| henneberg_n8_3tube_64 | 1 | +0 | -0.056836 | -0.083389 |
| tetrahedron | 2 | +0 | -0.019223 | -0.014528 |
| henneberg_n6_2tube_1 | 2 | +1 | -0.177906 | -0.379370 |
| henneberg_n8_3tube_64 | 2 | +1 | +0.015277 | -0.135095 |

## H10 collapse-sensitive validation

| Source | Model | Terminal misses (any imagined collapse) | Terminal misses (endpoint only) | Ordinary false positives | Ordinary progress MAE |
|---|---|---:|---:|---:|---:|
| tetrahedron | original | n/a (empty cohort) | n/a (empty cohort) | 5/639 | 0.012993 |
| tetrahedron | uniform | n/a (empty cohort) | n/a (empty cohort) | 0/639 | 0.006807 |
| tetrahedron | balanced | n/a (empty cohort) | n/a (empty cohort) | 4/639 | 0.007393 |
| henneberg_n6_2tube_1 | original | 7/8 | 7/8 | 3/291 | 0.016674 |
| henneberg_n6_2tube_1 | uniform | 6/8 | 6/8 | 8/291 | 0.014375 |
| henneberg_n6_2tube_1 | balanced | 6/8 | 7/8 | 8/291 | 0.015191 |
| henneberg_n8_3tube_64 | original | 6/8 | 6/8 | 4/296 | 0.006789 |
| henneberg_n8_3tube_64 | uniform | 2/8 | 2/8 | 33/296 | 0.005323 |
| henneberg_n8_3tube_64 | balanced | 5/8 | 5/8 | 14/296 | 0.005517 |

Terminal, precollapse and ordinary cohorts are defined separately at each horizon. Primary H10 sensitivity uses any imagined collapse, matching planner stopping; endpoint sensitivity is also retained. All models see the same27 new original-policy validation episodes, excluded from fitting and normalization. Empty cohorts are inconclusive. Overlapping windows are not independent examples. Per-cell H1/5/10 state errors and true-geometry/terminal-label agreement are retained in each group report.

Across the recorded validation cells, nonfinite windows=0 and true-geometry/terminal-label disagreements=0; repeated cells across models/horizons are not independent event counts.

## Identical fixed validation candidate pools

| Source | Model | Selected collapse /9 | Mean regret | Median Spearman | Missed collapsing candidates | False alarms |
|---|---|---:|---:|---:|---:|---:|
| tetrahedron | original | 0 | 0.022759 | 0.7666 | 11/24 | 2/345 |
| tetrahedron | uniform | 0 | 0.020410 | 0.7333 | 16/24 | 0/345 |
| tetrahedron | balanced | 0 | 0.018090 | 0.7822 | 13/24 | 0/345 |
| henneberg_n6_2tube_1 | original | 7 | 0.772105 | 0.1949 | 157/234 | 5/135 |
| henneberg_n6_2tube_1 | uniform | 7 | 0.770109 | 0.2948 | 130/234 | 6/135 |
| henneberg_n6_2tube_1 | balanced | 7 | 0.769312 | 0.5354 | 92/234 | 14/135 |
| henneberg_n8_3tube_64 | original | 5 | 0.548891 | 0.5737 | 112/220 | 19/149 |
| henneberg_n8_3tube_64 | uniform | 8 | 0.880166 | 0.6510 | 87/220 | 22/149 |
| henneberg_n8_3tube_64 | balanced | 4 | 0.445244 | 0.6599 | 82/220 | 16/149 |

Each validation state has41 candidates:32 original-model CEM proposals plus zero and eight smooth-random sequences. Twenty-seven pools/1,107 branches were fixed before fitting. These are a common original-policy distribution, not the repaired models’ refined proposal distributions. Unique-sequence sensitivity is retained. Branches execute open-loop while deployed MPC replans after each first action.

## Failure-adjacent fitting availability

| Source | Seed | Failure-adjacent transitions / all | Failure-adjacent valid windows / all |
|---|---:|---:|---:|
| tetrahedron | 0 | 627/20000 | 627/18956 |
| henneberg_n6_2tube_1 | 0 | 3894/20000 | 3894/18392 |
| henneberg_n8_3tube_64 | 0 | 4444/20000 | 4444/18324 |
| tetrahedron | 1 | 506/20000 | 506/18964 |
| henneberg_n6_2tube_1 | 1 | 3641/20000 | 3641/18433 |
| henneberg_n8_3tube_64 | 1 | 4191/20000 | 4191/18388 |
| tetrahedron | 2 | 638/20000 | 638/18952 |
| henneberg_n6_2tube_1 | 2 | 3630/20000 | 3630/18464 |
| henneberg_n8_3tube_64 | 2 | 4356/20000 | 4356/18368 |

These pools are identical across continuation arms within seed. Full exposure counts and any empty-stratum fallbacks are retained by arm/source/seed.

## Fitting intervention and interpretation

Both arms start from exactly the same20,000-transition/source shared model, frozen normalization and Adam state within seed. Each receives2,000 extra batches/source (6,000 shared updates), ending at66,000 Adam steps and22,000 updates per view. Each source/arm/seed gets64,000 one-step draws and64,000 five-step windows. Balanced sampling draws16 failure-adjacent and16 complement examples per batch when both strata exist; uniform retains the original distribution. Loss, model, learning rate, clipping and target update are unchanged. No importance correction is applied: this intentionally changes the effective objective distribution.

Failure-adjacent means endpoint distance0..10 to an observed true terminal, with episode-safe recursive windows. The complement is not proven safe and includes budget-cut episodes. Availability, actual exposure counts and fallback calls are reported by source/seed in measured-summary.json and fitting reports. Training losses come from different sampling distributions and are not a common validation metric.

This is a fixed-replay continuation test of one sampling intervention, not a from-scratch online replication. Source validation is assessed at the fixed final checkpoint; no best-validation selection or conditional suppression of control results. Survival gains must be read with movement/return and false alarms. No binary repair-pass threshold or significance claim is inferred from three seeds.

## Artifacts and verification

Raw run: `/Users/isaacsudweeks/.codex/worktrees/96de/GNN-SAC/outputs/dynamics-pilot/failure-balanced-replay-2026-10-25`. `protocol.json` hashes source/config/checkpoints; `prepared-data-manifest.json` freezes validation and original-reference evidence before fitting. Each fitting job retains full shared optimizer/model/target and sampler restart state, averaged losses, final diagnostics and control traces. `completion.json` records elapsed time and no Experiment3 execution. Large generated artifacts remain ignored.

Twenty-one affected tests pass, including exact uniform/native fitting parity, full-round model/Adam/target/sampler restart, stale-progress recovery and actual simulator reset/boundary/restoration smoke. The independent artifact audit passes 14,386 checks, including67 source/input and147 frozen-data hashes, exact sampling RNG/exposure accounting, paired resets and raw task-reward recomputation. A separate one-thread no_grad frozen-forward audit passes15,488 checks, reproducing729 prediction cells and3,321 candidate score predictions without simulator execution or optimizer steps. Both audit scripts/results are preserved with the outputs.
