# Shared versus separate online source dynamics — October 24, 2026

Question: can a single dynamics GNN learn three trained source topologies without
sacrificing control relative to topology specialists? This is not unseen transfer.
User authorized implementation and the complete fixed-budget experiment.

## Pre-execution protocol

Use tetrahedron (four physical nodes, compact minimally rigid baseline),
`henneberg_n6_2tube_1` (six physical nodes, two-tube routing), and
`henneberg_n8_3tube_64` (eight physical nodes, three-tube routing). All three are
members of the existing eight source-training topologies in
`config/dynamics_training_data.yaml`; the latter two increase structural size and
routing complexity. Measure actual graph/actuator/alias counts in metadata and the
benchmark. Verified graph nodes / unique physical nodes / actuators are 8/4/6,
14/6/12 and 21/8/18 respectively (structural bars 6, 12, 18). Neither held-out robot nor any seven-node final-test robot is accessed.

Conditions are one shared GNN versus three independent GNNs, seeds 0, 1, 2.
Each seed/condition is a group with three independently stepped native MuJoCo envs.
Every topology receives exactly 20,000 actual fitting transitions per condition and
seed, including 160 smooth-random warm-up transitions. Total fitting interactions:
360,000. After warm-up each robot executes its own learned dynamics-only CEM-MPC
with the pilot's Gaussian exploration. No simulator planner, teacher replay, reward
head or value head trains or controls the experiment. Existing unused task/policy
modules are retained for compatibility but do not enter dynamics-only planning/loss.

Architecture, dynamics loss, five-step recursion, batch size 32, AdamW learning rate
0.001 and decay 1e-5, clipping 10, nominal physics, 80-step time limits, effort and
collapse charges, exploration 0.15, gamma .995, and H10/16-candidate/two-iteration
CEM are unchanged from `config/dynamics_online_long.yaml`.

## Exact exposure accounting

Complete one real interaction per topology per round, in declared topology order.
After round 160, perform 160 update rounds. Thereafter perform one update round per
interaction round. An update round consists of one batch and optimizer step per
topology, always in declared order. Thus each topology contributes exactly 20,000
updates, 640,000 one-step draws and 640,000 five-step windows (3,200,000 recursive
target exposures), sampled with replacement. This counts draws, not distinct examples.
Shared Adam receives 60,000 steps; each specialist receives 20,000. The separate
condition also receives 60,000 aggregate optimizer steps. Per-topology contributions
are identical; shared parameters necessarily also receive the other two topologies'
updates. Fixed ordering is a limitation. Checkpoints are after complete rounds.

Homogeneous batches avoid padding variable-size graphs. Replays remain separate and
recursive starts must remain in the same episode with no intermediate terminal or
time-limit boundary. Terminal-ending windows are retained. Absence of a valid window
is an explicit failure rather than an unrecorded change in exposure accounting.
Graph-specific routing, masks, aliases, initial rigidity and loss representatives
remain attached to each view. Loss averages unique physical nodes within each graph;
larger robots do not receive extra batch weight.

## Matched initialization and common fitting-only normalization

Seeded dynamics parameter initialization is independent of graph size and identical
across all four models within a seed. Random warm-ups have matched RNG/reset streams
across conditions for each topology. At the end of all three warm-ups compute the
existing `fitting_statistics` protocol: average physical-node first/second moments
and delta second moments equally across topologies; compute std and delta RMS with
.001 floors; use arithmetic mean initial diagonal for geometry scaling. Apply the
same resulting statistics to all four models, freeze thereafter. This common
normalization is the documented compatibility change from topology-specific pilot
normalization. Specialists therefore access other source warm-up moments but no
other source labels/gradients. No evaluation or held-out state enters normalization.
There is no requirement to reproduce the earlier tetrahedron trajectory under this
new common normalization.

## Frozen evaluation and interpretation

Evaluate at 1,600, 4,000, 8,000, 12,000 and 20,000 transitions per topology. Use three
paired reset/planner seeds per topology/training seed, reused across checkpoints;
verify exact initial-state hashes across conditions. At the fixed final checkpoint
use five additional paired resets, disjoint from training and monitoring. Use the
existing reset-seed formula with topology offset 100 in monitoring and offset 30
for fresh evaluation. Training uses separate million-spaced topology streams.
Evaluation never changes fitting data, scales, schedule, budget or checkpoint choice.

Report every topology/seed: normalized forward displacement, collapse-penalized
discounted return, collapse counts, prediction errors at 1/5/10/20 steps, and planning
median/p95 latency. Preserve raw returns, actual initial states and all trajectories.
Pair shared-minus-separate episode outcomes, summarize within training seed first.
Prediction errors on own-controller states are distribution dependent; additionally
score both final frozen models on the union of both fresh trajectory sets for a
common-distribution comparison. Windows overlap and are not independent samples.

Primary descriptive endpoint is fresh-final return difference, with displacement
and failures considered jointly on each topology. Three seeds/five fresh resets do
not establish noninferiority or safety; mixed results do not justify a claim that
sharing has no cost. No significance threshold or best-checkpoint selection.

## Resources, stopping and persistence

Benchmark both full-architecture conditions before the campaign; record observed
throughput and extrapolate collection/update costs plus fixed evaluation overhead.
Run three CPU workers with one Torch thread each, six groups total. Initial estimate
from the tetrahedron pilot is 9.4 aggregate CPU hours before larger-graph overhead;
benchmark will replace this estimate. Record elapsed and summed group wall time.
Compute estimates are planning estimates, not permission to extend interactions.
Stop exactly at 20,000 per topology, or fail the affected group on numerical or
integrity failure; preserve failure reports, logs and last valid checkpoint. Do not
automatically tune, restart failed learning with new settings, add seeds, increase
budgets, or access unseen topologies. Operational restart must match source/config/
dependency hashes and reproduce exact saved simulator, RNG, replay and optimizer state.

Atomic latest checkpoints every 1,000 rounds, plus retained evaluation checkpoints,
contain full real replay, all model/optimizer states, Python/NumPy/Torch/local RNGs,
MuJoCo data/wrapper state, episode counters and CEM plans. Exact mid-episode resume
regressions cover both conditions. Generated artifacts stay ignored under outputs.

## Commands

```bash
uv run python -m unittest tests.test_dynamics_multitopology tests.test_dynamics_online_long tests.test_dynamics_online -v
uv run python scripts/run_online_dynamics_multi.py --output outputs/dynamics-pilot/online-multi-source-2026-10-24
# Operational resume only, identical sources/config/dependencies:
uv run python scripts/run_online_dynamics_multi.py --output outputs/dynamics-pilot/online-multi-source-2026-10-24 --resume
uv run python scripts/report_online_dynamics_multi.py outputs/dynamics-pilot/online-multi-source-2026-10-24
```

## Measured benchmark and declared allocation (before campaign launch)

Full pilot architecture, three concurrent workers, three disposable benchmark groups
(shared seeds 90/91, separate seed 90), 320 transitions per topology per group:
141.68 seconds elapsed. Slowest steady 160-to-240 interval cost 0.336978 seconds per
three-topology round. Extrapolation plus 25% evaluation/persistence allowance gives
4.68 hours elapsed / 14.04 aggregate group-hours. This replaces the rough
single-topology estimate; larger graphs and contention explain the revised estimate.
The allocation is six complete groups at the fixed 360,000-transition total. No
interaction budget, model setting or evaluation schedule is changed. Disposable
benchmark replay and learned parameters are never reused by the campaign.
Evidence: `outputs/dynamics-pilot/online-multi-benchmark-2026-10-24/benchmark.json`.

## Campaign launch

Launched October 24, 2026 at 20:50:14 UTC with three CPU workers and one Torch
thread per worker. Six fixed groups are queued; no method changes followed the
benchmark. Completion follow-up `finish-shared-dynamics-mpc-experiment` checks every
15 minutes and stays quiet during normal progress, then finishes integrity checks,
visual inspection, measured interpretation, repository and Obsidian reporting.

Prelaunch validation: two new comprehensive multi-topology tests pass (variable
sizes/masks/routing/aliases, balanced Adam counts, frozen normalization, exact
mid-episode resume for both conditions); 21 existing online/long/model/planner
regression tests pass. A disposable six-group reduced end-to-end campaign also
passes reporting and artifact checks. These are software checks, not campaign
control results. Verification logs and exact source snapshots are retained under
the campaign output. `git diff --check` passes. Protocol and index notes were saved
through the official Obsidian CLI with exact readback and prior-content checks.

## Completed outcome — October 24, 2026

All six groups completed exactly 20,000 real transitions and updates per topology,
360,000 fitting transitions total, without run failures or operational restarts.
Elapsed campaign time was 4.08 hours and summed group wall time 12.05 hours, versus
the prelaunch 4.68 / 14.04 hour estimate. No learning settings, interaction budgets,
checkpoint schedule or topology scope changed after launch.

This protocol did not demonstrate shared learning without a control cost.
Fresh-final mean shared-minus-separate task return differences are -0.210452 for
tetrahedron, +0.176721 for henneberg_n6_2tube_1 and +0.045430 for
henneberg_n8_3tube_64. Corresponding collapses are 4/15 versus 0/15, 9/15 versus
13/15, and 15/15 versus 15/15. Tetrahedron return is lower with sharing in all three
training seeds. Six-node movement is lower with sharing in all three seeds; its
return advantage in two seeds comes with fewer collapses. Eight-node survival
fails under both methods. Pooled collapses (28/45 each) hide this topology tradeoff.

All 732 automatic and 208 independent completion checks pass, including matched
initialization, warmups and frozen normalization, exact interaction/update counts,
all milestone checkpoints, replay alias/episode consistency, paired actual starts,
role separation and source/artifact hashes. Both plots were visually inspected.
Initial sandboxed SVG export failed; successful PNG/SVG renderings preserve the
plotted data and the original errors remain recorded. Per-topology/per-seed tables,
paired differences, curves and common-distribution prediction errors are in
`outputs/dynamics-pilot/online-multi-source-2026-10-24/report.md`.

Recommendation: a separately predeclared source-only failure diagnostic on saved
real replay/frozen checkpoints, plus independent source-seed/reset replication,
before unseen-topology work. Examine near-collapse recursive errors and balanced
per-topology gradients; gradient interference is a hypothesis, not a proven cause.
No follow-on training, tuning, diagnostic experiment or unseen evaluation ran here.
The Obsidian experiment note/index receive the same measured result and figures.

Completion reporting is verified: both Obsidian notes preserve their prior content
and all appended chunks read back exactly; copied figure hashes match. Final artifact
hashes pass. The completed campaign process has exited and its completion monitor is
removed after reporting. No follow-on experiment is running.
