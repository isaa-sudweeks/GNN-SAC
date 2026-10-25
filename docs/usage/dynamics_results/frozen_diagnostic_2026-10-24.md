> Published measured report. Static figures are included here. Raw CSV/JSON, interactive HTML, replay and checkpoint filenames mentioned below refer to ignored local artifacts under `outputs/dynamics-pilot/online-multi-diagnostic-2026-10-24/`; large generated artifacts are not part of this Git snapshot.

# Frozen source dynamics failure and gradient diagnostic

Completed the predeclared saved-data diagnostic in 42.47 seconds using one Torch/inter-op thread. No optimizer steps, simulator execution, new collection, normalization changes, checkpoint selection or unseen topology access occurred.

Input: final fixed 20,000-transition checkpoints, real fitting replay and already-recorded independent fresh-final source trajectories from the gamma=.995 campaign. All input hashes are verified again after diagnosis; every gradient call verifies parameters, buffers, Torch RNG and `.grad` fields remain unchanged.

## Fresh terminal-ending H10 windows

Both frozen models receive exactly the same executed-action windows from the union of both controllers. Each generator cell contributes at most 32 selected windows; raw per-generator rows remain in `prediction-cells.csv`. Counts below sum selected windows; errors are weighted by windows, not independent episode claims. Empty cells are retained. A false negative means the endpoint model geometry does not predict an actual recorded collapse.

| Topology | Seed | Windows | Shared progress MAE | Separate progress MAE | Difference | Shared missed collapse | Separate missed collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetrahedron | 0 | 2 | 0.007542 | 0.002878 | +0.004664 | 2/2 | 1/2 |
| tetrahedron | 1 | 2 | 0.008590 | 0.013732 | -0.005142 | 2/2 | 1/2 |
| tetrahedron | 2 | 0 | n/a | n/a | n/a | n/a | n/a |
| henneberg_n6_2tube_1 | 0 | 8 | 0.028529 | 0.009412 | +0.019116 | 8/8 | 8/8 |
| henneberg_n6_2tube_1 | 1 | 8 | 0.043212 | 0.006695 | +0.036517 | 6/8 | 8/8 |
| henneberg_n6_2tube_1 | 2 | 6 | 0.023049 | 0.012095 | +0.010953 | 5/6 | 3/6 |
| henneberg_n8_3tube_64 | 0 | 10 | 0.019758 | 0.017719 | +0.002039 | 8/10 | 8/10 |
| henneberg_n8_3tube_64 | 1 | 10 | 0.011670 | 0.014529 | -0.002859 | 8/10 | 6/10 |
| henneberg_n8_3tube_64 | 2 | 10 | 0.012063 | 0.023331 | -0.011268 | 7/10 | 8/10 |

## Local fitting gradients at final shared weights

Four predetermined batches per topology and seed each contain 16 one-step examples and 16 five-step windows from each fitting generator. Cosines compare topology gradients at the same shared parameter vector. Raw gradients are measured before clipping; specialist norms/losses use the same own-topology examples. Negative cosine indicates local first-order opposition on the selected examples, not proven sustained interference or a demonstrated cause of control collapse.

| Seed | Topology pair | Mean total cosine | Negative batches | Range |
|---:|---|---:|---:|---|
| 0 | tetrahedron / henneberg_n6_2tube_1 | +0.1985 | 0/4 | +0.0396 to +0.4864 |
| 0 | tetrahedron / henneberg_n8_3tube_64 | +0.0483 | 1/4 | -0.1618 to +0.2009 |
| 0 | henneberg_n6_2tube_1 / henneberg_n8_3tube_64 | +0.0508 | 1/4 | -0.1124 to +0.1426 |
| 1 | tetrahedron / henneberg_n6_2tube_1 | +0.1897 | 0/4 | +0.0617 to +0.2524 |
| 1 | tetrahedron / henneberg_n8_3tube_64 | +0.0048 | 3/4 | -0.0894 to +0.1854 |
| 1 | henneberg_n6_2tube_1 / henneberg_n8_3tube_64 | -0.0031 | 2/4 | -0.0478 to +0.0770 |
| 2 | tetrahedron / henneberg_n6_2tube_1 | -0.0733 | 2/4 | -0.2996 to +0.0344 |
| 2 | tetrahedron / henneberg_n8_3tube_64 | +0.0049 | 1/4 | -0.0763 to +0.0844 |
| 2 | henneberg_n6_2tube_1 / henneberg_n8_3tube_64 | +0.0700 | 1/4 | -0.0265 to +0.1675 |

## Matched fitting objective and gradient norms

| Topology | Seed | Shared loss | Separate loss | Shared total-gradient norm | Separate norm |
|---|---:|---:|---:|---:|---:|
| tetrahedron | 0 | 1.4723 | 1.2624 | 122.8007 | 68.7958 |
| tetrahedron | 1 | 1.3892 | 1.3919 | 69.7134 | 13.8622 |
| tetrahedron | 2 | 1.9336 | 5.2495 | 27.2026 | 53.1119 |
| henneberg_n6_2tube_1 | 0 | 7.3866 | 6.5031 | 36.6144 | 46.8863 |
| henneberg_n6_2tube_1 | 1 | 5.9123 | 5.9789 | 41.3332 | 52.6398 |
| henneberg_n6_2tube_1 | 2 | 8.8561 | 6.7231 | 58.7714 | 38.7428 |
| henneberg_n8_3tube_64 | 0 | 3.3791 | 3.2807 | 13.5854 | 16.5174 |
| henneberg_n8_3tube_64 | 1 | 5.1230 | 4.1582 | 19.6345 | 16.2346 |
| henneberg_n8_3tube_64 | 2 | 6.0918 | 5.6834 | 31.6129 | 27.2513 |

## Failure interpretation by topology

- tetrahedron: shared misses 4/4 selected fresh H10 collapse endpoints; separate misses 2/4. Shared H10 progress error is greater in 1/2 nonempty seed cohorts.
- henneberg_n6_2tube_1: shared misses 19/22 selected fresh H10 collapse endpoints; separate misses 19/22. Shared H10 progress error is greater in 3/3 nonempty seed cohorts.
- henneberg_n8_3tube_64: shared misses 23/30 selected fresh H10 collapse endpoints; separate misses 22/30. Shared H10 progress error is greater in 1/3 nonempty seed cohorts.

Recorded-vs-geometric endpoint label disagreements across selected fresh H10 cohorts: 0. Nonfinite predictions across both frozen models in those cohorts: 0.

The control outcomes themselves remain those of the completed campaign: tetrahedron had four additional shared fresh collapses, six-node sharing had four fewer collapses but less movement in all seeds, and both methods collapsed on every eight-node fresh episode. This diagnostic measures executed-action prediction, not scores/ranking of unexecuted CEM candidates or counterfactual action outcomes. It cannot uniquely attribute the failures to model capacity, local gradient conflict, planner ranking, state aliasing, exploration or the discount factor.

## Limits and next recommendation

Use these results to target a separately predeclared source-only diagnostic or ablation; do not infer unseen transfer. If both models miss near-collapse geometry, repair claims should first be checked on a new source validation manifest with collapse sensitivity and candidate ranking measured separately. A shared-specific fitting or near-collapse error motivates a controlled interference/capacity/update-order comparison, but final-point gradient cosines are insufficient to choose one repair. The concurrently authorized gamma=.95 rerun is independent and must retain its fixed protocol; this diagnostic neither selects nor modifies it.

Fitting errors are in-sample and optimistic. Fresh samples were previously evaluated by the campaign, so these are post hoc explanations, not untouched confirmation. Sampled windows overlap; endpoints and repeated reset episodes are not independent replicates. Three training seeds and four gradient batches do not establish a general optimization law. Generator/cohort counts are explicit in `samples.json` and `prediction-cells.csv`; no numerical or difficult cases were silently dropped.

[Gradient cosine plot](figures/diagnostic-gradient-cosines.svg) · [Fresh H10 prediction plot](figures/diagnostic-near-collapse-errors.svg)

Implementation: `scripts/run_dynamics_multi_diagnostic.py`; tests: `tests/test_dynamics_multi_diagnostic.py`; predeclared protocol: `docs/plans/gnn_dynamics_multi_diagnostic.md`.

## Collapse misses across recursive horizons

Counts use all available selected fresh terminal-ending windows, evaluated by both frozen models on identical executed actions. Longer rollouts can predict a collapse spuriously; greater endpoint sensitivity alone is not a reliability claim. Full precollapse/ordinary false-positive counts remain in the cell data.

| Topology | Horizon | Shared missed endpoints | Separate missed endpoints |
|---|---:|---:|---:|
| tetrahedron | 1 | 4/4 | 3/4 |
| tetrahedron | 5 | 4/4 | 4/4 |
| tetrahedron | 10 | 4/4 | 2/4 |
| tetrahedron | 20 | 4/4 | 1/4 |
| henneberg_n6_2tube_1 | 1 | 12/22 | 13/22 |
| henneberg_n6_2tube_1 | 5 | 20/22 | 21/22 |
| henneberg_n6_2tube_1 | 10 | 19/22 | 19/22 |
| henneberg_n6_2tube_1 | 20 | 19/22 | 15/22 |
| henneberg_n8_3tube_64 | 1 | 20/30 | 21/30 |
| henneberg_n8_3tube_64 | 5 | 23/30 | 24/30 |
| henneberg_n8_3tube_64 | 10 | 23/30 | 22/30 |
| henneberg_n8_3tube_64 | 20 | 17/30 | 17/30 |

The immediate source-diagnostic priority is near-collapse model sensitivity and whether that error affects candidate ranking, because both conditions miss most larger-topology H10 failures. The present gradients do not justify a broad gradient-surgery repair: 25 of 36 total-gradient cosines are nonnegative and pair means are small or positive in most seed/pair combinations. These are final-point local measurements, so earlier training interference remains possible.
