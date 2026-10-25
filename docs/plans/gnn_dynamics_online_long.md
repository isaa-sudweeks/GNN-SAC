# Longer online learned-MPC source experiment — October 24, 2026

The user approved the proposed longer-run comparison before any method change or
multi-topology expansion. This separately recorded duration experiment preserves
the original short runs and their results.

## Fixed protocol

- Source tetrahedron only; conditions `dynamics_mpc`, `reward_value_mpc`; seeds 0,1,2.
- Exactly 20,000 actual environment training transitions per run (120,000 total).
- The same 160-transition random warm-up, 160 initial updates, then one model update
  per real transition. Same frozen warm-up normalization, architecture, losses,
  reward scaling, learning rates, exploration, actor update schedule and discount.
- Same 80-step training/evaluation episode limit and nominal physics. True terminals
  disable bootstrap; artificial time limits retain bootstrap.
- Evaluate frozen weights at 1,600 / 4,000 / 8,000 / 12,000 / 20,000 transitions using
  the original three paired reset seeds per model seed. Record all outcomes rather
  than selecting a checkpoint from evaluation performance.
- At 20,000 only, evaluate five fresh paired resets per seed (15 per condition),
  with matched smooth-random controls on those same starts. These source seeds are
  disjoint from training and monitoring resets; no held-out topology is accessed.
- Track displacement, raw and collapse-penalized task returns, collapses, latency,
  averaged fitting losses, and terminal-Q prediction error on real fitting terminals.
- Use the existing common random-action source evaluation dataset only for frozen
  final prediction diagnostics, never for fitting or normalization.

## Initialization, reproducibility and restarting

The short checkpoints omitted RNG and live simulator state, so this is an extended
rerun from identical original seeds rather than an inexact optimizer-only resume.
At 1,600 steps, require bitwise equality with the corresponding short run for model,
actor, targets, both optimizer states, real replay arrays, and fixed evaluation
trace arrays. If that fails, record the mismatch rather than attributing results
solely to longer duration.

The new runner leaves the original short runner and learning implementation intact.
It saves full real replay, model/actor/target/optimizer states, Torch/Python/NumPy and
local-generator RNG states, live MuJoCo/wrapper snapshots, current episode state and
CEM warm-start plan. Save an atomic latest checkpoint every 1,000 transitions and
retain each declared evaluation checkpoint. Resume requires matching configuration,
source hashes and checkpoint hash. A mid-episode deterministic regression compares
uninterrupted versus resumed training and evaluation arrays.

## Resources and stopping

Use three independent CPU workers, one Torch thread each, with six runs total.
The measured short-run extrapolation is approximately 2–3 aggregate CPU experiment
hours before worker contention and extra evaluation/checkpoint overhead. Concurrent
workers shorten elapsed time; do not count their summed wall times as elapsed time.
Record both elapsed time and per-run time. Revise the estimate if measured contention
or overhead projects an overrun; this is a planning target rather than a hard cap.

Stop each run at 20,000 transitions. No automatic loss/scaling repair, learning-rate
change, capacity change, history input, topology expansion, or extra training budget.
Retain numerical failures and all collapse outcomes. Future multi-topology work is
conditional on reviewing these source results, not part of this run.

## Reproduce

```bash
uv run python -m unittest tests.test_dynamics_online_long tests.test_dynamics_online -v
uv run python scripts/run_online_dynamics_long.py --output outputs/dynamics-pilot/online-long-source-2026-10-24 --prefix-root outputs/dynamics-pilot/online-source-2026-10-24 --diagnostic-data /trusted/source-training-2026-10-23
# After an interruption, repeat the identical command with --resume.
```

Settings: `config/dynamics_online_long.yaml`. Logs, progress, checkpoints, real replay,
traces, prefix comparisons and final reports remain ignored in the output directory.
The reward/value method is TD-MPC2-inspired explicit-state graph learning, not a
reproduction of the full published TD-MPC2 algorithm. No transfer claim is supported
by this single-source experiment.

## Completed outcome

The approved duration experiment supports giving this online loop more experience before redesigning it. Dynamics-only MPC improved its monitoring mean discounted task return from -0.273019 at 1,600 transitions to 0.449122 at 20,000, with improvements in all three training-seed means. Monitoring collapses fell from 7/9 to 0/9. All six 1,600-step prefixes matched exactly, so differences after that prefix reflect the longer fixed-method collection/update trajectory rather than a changed initial implementation.

Reward/value MPC showed a weaker, nonmonotonic duration result: monitoring mean return was 0.070505 at 1,600, 0.310368 at 8,000, and 0.097975 at the fixed 20,000 endpoint. Monitoring collapse counts were 3/9, 0/9 and 3/9 at those budgets. We did not select the 8,000 checkpoint after observing that peak. Of its three training-seed mean monitoring returns, one decreased and two increased between 1,600 and 20,000; this is not a consistent superiority result.

On fresh final source resets, both methods moved forward with positive average collapse-penalized return. Dynamics-only had mean movement 0.524257 and return 0.248958 with 4/15 collapses; reward/value had movement 0.434774 and return 0.262905 with 2/15 collapses. Random control had movement -0.011749 and return -0.112175 with 2/15 collapses. The small difference in mean penalized return between the learned methods does not establish a winner. Fresh outcomes also qualify the dynamics-only 0/9 monitoring-collapse result: it is not evidence of zero failure risk. Short-run checkpoints were not evaluated on these fresh resets, so fresh data do not provide a direct short-versus-long comparison.

Terminal-Q calibration is still unresolved. At 20,000 transitions the reward/value model's MAE on accumulated fitting terminal transitions ranges from 0.514327 to 1.081427 task units across seeds. Corresponding mean predicted min-Q values are -0.500243, -0.310987 and +0.082661 while realized terminal rewards average near -1. These terminal-example cohorts grow with collection and are not a fixed calibration test set. The observed errors nonetheless do not support treating the learned tail as accurately calibrated, or attributing the result solely to a lack of short-run data.

**Recommendation:** a controlled shared-model experiment on two or three source topologies is now justified as the next feasibility step, with balanced replay, equal per-topology interaction budgets, per-topology learning curves, and separate-model references under matched budgets. Retain the dynamics-only comparator and the reward/value variant; this source experiment does not justify dropping either or claiming the value extension is consistently better. Record terminal calibration alongside control. Keep the original held-out and final-test topologies untouched. No topology expansion, additional tuning or training was started here.

The runs completed without recorded failures, with 120,000 fitting transitions and 120,000 updates total. Actual elapsed time was 63.70 minutes; summed per-run wall time was 3.14 hours with three concurrent workers. The latter slightly exceeds the initial 2–3 hour extrapolation because of added evaluation, persistence and worker overhead; the fixed interaction budget was not extended. Both plots were visually inspected and all artifact, source, reset-role, pairing, budget and replay checks passed.
