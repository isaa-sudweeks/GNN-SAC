# Frozen shared-dynamics diagnostic — October 24, 2026

The user authorized a diagnostic of the completed gamma=.995 three-source campaign,
concurrently with a separate gamma=.95 rerun. This plan is recorded before computing
new diagnostic outcomes. It does not alter either campaign or infer unseen transfer.

## Fixed scope and hypotheses

Use the fixed final 20,000-transition checkpoints for shared and specialist dynamics,
all three training seeds and the same tetrahedron/six-node/eight-node sources. Read
only saved fitting replay and saved independent fresh-final real trajectories from
`outputs/dynamics-pilot/online-multi-source-2026-10-24`. Verify original source-snapshot,
checkpoint and trace hashes. No optimizer steps, new normalization, CEM selection,
simulator execution, data collection, checkpoint selection, tuning or unseen access.

Distinguish (1) poor near-collapse prediction common to shared and separate models,
(2) shared-specific prediction errors on identical examples, and (3) local conflicting
fitting gradients at final shared weights. None alone proves the cause of closed-loop
failure; observed replay was collected by different learned controllers.

## Common-distribution prediction cohorts

For each seed, topology, data role (fitting or fresh-final), generator (shared or
separate), and horizon 1/5/10/20, construct episode-safe executed-action windows.
A valid window has one episode identity throughout and no interior true terminal or
time-limit marker. A terminal at the final element is retained. Do not concatenate
fresh episodes or infer trajectories from interleaved data.

Partition valid windows into terminal-ending, precollapse (endpoint lies 1–10 actual
control steps before that episode's observed terminal), and ordinary (endpoint is
more than 10 steps before an observed terminal, or episode has no true terminal).
Draw at most 32 windows without replacement per cell using fixed diagnostic seed
271828 plus deterministic cell offsets. Preserve zero/undersized cohorts. Use the
identical selected windows for both frozen models. Report each generator and role,
then within each topology/seed/stratum; never pool unequal episode/stratum counts as
independent evidence. The source fitting cohort is descriptive in-sample evidence;
fresh-final trajectories were excluded from original fitting and remain frozen.

Report normalized accumulated forward-progress MAE, world-aligned unique physical
node position RMSE, velocity RMSE, endpoint normalized rigidity MAE, and endpoint/any
imagined collapse confusion counts. Compare replay terminals with geometric labels
on true endpoint positions to expose a potential label/classifier mismatch. A
nonfinite model prediction is retained as failure with normalized progress penalty
100, position penalty 100 initial diagonals, and velocity penalty 100 m/s. Any
nonfinite imagined state counts as predicted collapse; never drop such windows.

## Balanced fitting-gradient measurements

At final shared weights, use four predetermined replicate batches per topology.
Each one-step batch and each five-step recursive-window batch contains 16 examples
from each condition's fitting replay, sampled with replacement. All topologies have
batch size 32 and equal replicate counts. Use the original one-step plus five-step
recursive physical-node/progress loss, frozen original normalization, and no clipping
when measuring raw gradients. Record one-step/rollout/total losses and gradient norms.
Compute pairwise gradient cosine similarities only between topology gradients at the
same shared parameter vector. Report all seeds and replicates plus total-gradient
cosine and per-component gradients. Specialist own-topology gradient norms/losses use
exactly the same examples, but do not compare specialist gradient vectors across
different parameter states as an interference estimate. Use `autograd.grad` without
changing parameters, buffers, RNG state or `.grad` fields; assert this contract.
Negative local cosine indicates local first-order opposition on these sampled fitting
examples, not proof of sustained training interference or a beneficial repair.

## Compute, stopping and outputs

One process, one Torch/BLAS thread, no parallel agents/workers. Process one seed and
checkpoint at a time; retain only needed dynamics tensors after loading. Budget cap:
20 minutes elapsed for the full diagnostic, 32-window prediction cells and four
balanced gradient replicates fixed above. Stop and preserve partial results if the
cap or a numerical/integrity failure occurs; do not increase samples or repair a
method automatically. Save protocol, exact sampled indices/cohort counts, hashes,
per-cell metrics, per-seed gradient comparisons, no-mutation audit, plots and a plain
report under ignored `outputs/dynamics-pilot/online-multi-diagnostic-2026-10-24`.

Implementation and deterministic cohort/gradient no-mutation tests live only in new
`*_multi_diagnostic*` files. Existing campaign source files remain byte-identical for
the concurrent gamma=.95 experiment. Parent integrates conclusions into shared docs
and Obsidian after verifying measured results. No diagnostic result selects or
changes the concurrent gamma=.95 run.

## Completed measured diagnostic

Completed in 42.47 seconds, one Torch/inter-op thread, with 864 prediction cells,
72 frozen gradient batch evaluations and 36 shared cross-topology gradient cosines.
All 114 read tensor-artifact hashes remain unchanged; every gradient evaluation
passes parameter/buffer/RNG/.grad no-mutation assertions. Three deterministic tests
pass, including exact autograd equivalence and episode/cohort/nonfinite handling.

On saved fresh H10 terminal-ending windows, shared dynamics misses all four observed
tetrahedron collapse endpoints; specialists miss two on the identical windows.
Both conditions miss 19/22 six-node endpoints; shared and separate miss 23/30 and
22/30 eight-node endpoints respectively. True endpoint geometry agrees with logged
terminal labels on these cohorts, and no nonfinite model predictions occur. The
shared six-node terminal-window progress MAE is larger in all three seeds despite
that condition's lower closed-loop collapse count.

Only 11/36 measured total-gradient cosines are negative, with mixed near-zero or
positive pair means in most seed/pair combinations. Local gradient opposition
exists in some samples but is not a general causal explanation of these failures.
Prioritize near-collapse prediction/sensitivity and a separately authorized source
candidate-ranking diagnostic before choosing a training repair. No ranking of
unexecuted candidates, training change, additional controller evaluation or unseen
robot access occurred here. The concurrent gamma=.95 rerun is independent.

Full measured report and generator/seed/horizon/cohort rows are under
`outputs/dynamics-pilot/online-multi-diagnostic-2026-10-24/`. Parent integrates this
completed diagnostic into shared project notes after central verification.
