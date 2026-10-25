# Online learned-MPC source pilot — October 24, 2026

User authorized online learned-planner data collection and a reward/value extension.
This is a separately scoped experiment after the unsuccessful offline source gate.
It supersedes the proposed simulator-MPC collection comparison: no simulator MPC
or teacher generates fitting data here. Preserve the previous configuration/results.

## Frozen experiment

`config/dynamics_online.yaml` declares three conditions, each with seeds 0,1,2:

- Random smooth collection, with a trained dynamics model as a data-collection control.
- Learned observable dynamics + known task score + CEM MPC, collecting its own data.
- Learned observable dynamics + learned immediate task reward + terminal twin-Q
  value + CEM MPC, collecting its own data.

All start from scratch, share the same per-seed 160-transition random warm-up,
freeze normalization from that warm-up, perform 160 initial fitting updates, then
one update per executed real transition until exactly 1,600 fitting transitions.
Replay retains every real transition with episode ID and separate termination and
truncation. No imagined transition is treated as real experience. Each update uses
32 replay transitions plus 32 within-episode five-step windows. All models use
three width-128 message-passing layers. Ten-step planning uses 16 candidates in two
CEM iterations, three smooth knots, and 0.15 execution exploration standard deviation.
This is a small source feasibility experiment, not a hyperparameter search.

The reward is forward physical-node-mean displacement / initial bounding-box
diagonal minus 0.1 times actual elapsed-time integral of mean squared routed actuator
commands, minus 1 on true collapse. Raw environment reward is reported separately.
Both planners discount with gamma=0.995. The value condition predicts immediate
progress/effort reward, retains the same geometry-based collapse gate/charge, and
adds gamma^H min(Q1,Q2)(s_H,pi(s_H)) for surviving rollouts. A terminal value estimates
return beyond the finite horizon; it does not guarantee exact infinite-horizon control.

The value arm uses scalar Huber reward/TD losses, a delayed deterministic graph actor,
Polyak target networks, clipped target-policy smoothing and true-terminal masking.
Time-limit truncations bootstrap. Dynamics, reward and Q share graph features. This
is **TD-MPC2-inspired**, not a reproduction of its latent consistency, categorical
reward/Q, ensemble, entropy or planning-policy machinery. The combined comparison
cannot isolate reward supervision from terminal value or shared-feature effects.

Evaluate frozen models at 160,800,1600 interactions on three paired reset seeds per
training seed; no execution exploration, model updates or replay writes in evaluation.
Use the fixed final checkpoints rather than selecting from evaluation performance.
Report every paired outcome, task/raw returns, displacement, collapses and planning
latency. Common existing tetrahedron evaluation-role random episodes additionally
measure recursive prediction at horizons 1,5,10,20; these are source development
checks, excluded from all fitting, normalization and checkpoint selection.

## Budget and stopping

Smoke benchmark all conditions before full runs. Initial target <=30 CPU wall minutes
for nine fixed runs; revise the measured projection before an overrun. Stop after the
declared budget or a recorded implementation/numerical failure, retain evidence, and
report. No automatic repair/search, transfer evaluation or training extension.

## Interpretation

A better result for online MPC supports usefulness of this collection/training setup
on tetrahedron; it does not establish topology transfer or isolate a causal data effect,
since the conditions visit different states. Failure may reflect data, model, value
extrapolation or optimization; 1,600 transitions is a screening budget. Held-out and
final-test robots stay untouched. Three seeds and nine paired final evaluation episodes
are descriptive evidence, not confirmation-level reliability.

## Implementation and artifacts

- `sac/dynamics/online.py`, `scripts/run_online_dynamics_pilot.py`.
- `tests/test_dynamics_online.py`: real-replay updates, reset-safe windows,
  true-terminal bootstrap masks, geometry parity and model-only planning.
- Exact config/source hashes, output traces/replay/checkpoints and measured costs are
  saved under ignored `outputs/dynamics-pilot/online-source-2026-10-24/`.
- Previous pilot dependencies were copied unchanged from committed branch revision
  `c1d9f14` into this checkout; `ObservableDynamicsGNN.node_features` is now exposed
  for the shared task heads, preserving its original dynamics forward calculation.

## Completed results

The user-authorized online collection experiment is complete: three conditions × three training seeds, 1,600 real fitting transitions and 1,600 updates per run, 14,400 fitting transitions total. Nine full runs took 16.52 CPU wall minutes; no GPU was used. All 48 affected tests pass. Warm-up arrays are identical within seed, replay adjacency and budgets are exact, evaluation starts are paired, and artifact hashes pass.

Both learned-model MPC loops exceeded random control's forward displacement in all nine paired final resets. Mean normalized displacement: random 0.001051, dynamics-only 0.354245, learned reward/value 0.361744. Final evaluation collapses: 0/9, 7/9, 3/9 respectively. Mean discounted task returns: 0.000316, -0.273019, 0.070505. Reward/value improved mean task return over dynamics-only for all three training seeds, but one seed remains negative. This is promising source behavior, not reliable control or transfer evidence.

Post-run diagnostics show a concrete value-learning problem: on 59 fitting terminal transitions, per-seed mean min-Q is +0.022 to +0.047 while realized terminal reward is about -1. The terminal TD target has no bootstrap. This does not establish a unique cause or mandatory hidden input, but the terminal tail is not calibrated. Next recommended work is a predeclared source value-target/scaling/loss audit and repair on saved real replay before broader topology experiments. No extra training or repair was run.

The variant is TD-MPC2-inspired explicit-state GNN learning, not the full TD-MPC2 architecture. No simulator-MPC or teacher data generated fitting experience. Held-out and final-test topologies remain untouched. Full protocol, curves and evidence: [[GNN Dynamics Pilot - 12 Online Learned MPC and Reward Value - 2026-10-24]].
