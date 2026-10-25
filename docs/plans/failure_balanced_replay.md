# Failure-adjacent replay ablation and closed-loop source test

Status: independently reviewed, October25,2026; prelaunch parity/benchmark checks pending.

## Question and scope

Notebook Experiment17 found61 selected collapsing sequences,59 predicted to survive;
46 had a better surviving alternative inside the same model's proposals. This does
not establish the reason for prediction error. Test whether emphasizing already
available failure-adjacent fitting examples improves a shared dynamics model's
collapse-sensitive prediction and then closed-loop survival without losing motion.

User authorizes these two source experiments only. Model-free/generalization
Experiment3 remains for the user to plan; no unseen or final-test robots are used.
Use the existing shared model because shared dynamics is the intended transferable
method. Separate specialists are not retrained in this ablation.

## Experiment18: fixed-replay sampling intervention

- Sources tetrahedron, henneberg_n6_2tube_1, henneberg_n8_3tube_64; original training
  seeds0/1/2 and final gamma=.995 shared checkpoints from the c400 campaign.
- Two continuation arms, uniform and failure-balanced. Each arm starts from exactly
  the same final model, frozen normalization and Adam state within seed. Fixed saved
  shared-controller fitting replay only:20,000 real transitions/source/seed. No new
  fitting data, teacher data or Experiment17 evaluation data enters fitting.
- Exactly2,000 additional updates per topology, strict original topology order,
  6,000 shared Adam steps per arm/seed; six fitting jobs,36,000 updates total. This
  is a controlled fixed-replay continuation, not a from-scratch replication of the
  original online training or evidence that2,000 updates is universally sufficient.
- Preserve OnlineLearner.update unchanged: batch32 one-step draws plus32 valid
  five-step recursive windows, equal-topology exposure, architecture3x128, loss,
  learning rate/weight decay/clipping/target update, graph masks and routing.
  Intercept only the two replay sampling calls, with parity tests for the uniform arm.
- A failure-adjacent transition is in the last11 transitions of an episode ending
  in true termination (distance0..10 from its terminal). Time-limit episodes are
  not failures. A recursive window is failure-adjacent if its endpoint is in that
  band; no interior termination/truncation or episode crossing is allowed.
- Uniform draws match the original distribution. Balanced draws use16 examples
  from the failure-adjacent pool and16 from its complement, with replacement, for
  both one-step examples and recursive windows, then shuffle. No importance weights:
  this intentionally changes the effective fitting distribution. If a stratum is
  absent, explicitly record fallback to uniform; never manufacture failures.
- Checkpoint every500 topology rounds, preserving exact optimizer and sampler RNG
  state. Fixed endpoint2,000; no validation-selected checkpoint, patience or tuning.
  Log sampling counts/stratum availability and averaged losses per source.

## Independent source validation

Before fitting, use original frozen shared models on three new resets per source
and training seed,80-step episodes, no exploration. Reset namespace21,000,000 with
topology_index*10,000+seed*100+episode. These27 episodes are validation only; retain
all failures and ordinary trajectories. No validation writes to fitting replay or
normalization. Confirm reset exclusion against previous fitting/evaluation records.

All three models (original, uniform continuation, balanced continuation) predict
the same executed-action validation windows at H1/5/10. Report position/velocity/
progress errors, collapse false negatives AND false positives, with denominators
and terminal-ending/precollapse/ordinary cohorts by source and seed. Use all
episode-safe H10 windows for the primary diagnostic; overlapping windows are not
independent. Missing terminal cohorts are inconclusive, never a pass. Freeze
selection and window definitions before fitting. Evaluate only fixed final weights.

Also freeze one candidate-ranking state per validation episode: five steps before
true collapse, or timestep20 for a survivor (last available if shorter). Before
fitting, pool32 original-model CEM proposals plus zero and eight smooth-random
sequences (41 total), using the original generator's predecision warm start. Save
all arrays and restored simulator branches. This is27 states/1,107 branches.
After fitting, all three frozen models score those identical pools. Report H10
collapse confusion, false alarms, Spearman and finite-set regret (plus duplicate
sensitivity) by seed/source. These original-policy candidate pools are a common
validation distribution, not each repaired model's refined planner proposals.

## Experiment19: closed-loop source confirmation

Always evaluate both continuation arms, regardless of validation outcome, plus
the original frozen shared reference. Use five new paired resets per source and
training seed:45 episodes per model,135 total. Reset namespace22,000,000 with the
same topology/seed/episode offsets, distinct from validation and all fitting.

Freeze models; no exploration, fitting or replay writes. Preserve80-step/16-second
episode cap, H10/2-second planning, CEM16x2 and3 knots, gamma=.995, same nominal
physics and original task reward. Pair actual initial state hashes and planner RNG
seeds across conditions; histories diverge naturally after different actions.

Report displacement, task return, collapse count/time, raw environment return,
planner latency and balanced-minus-uniform paired differences for every source and
training seed. Original reference shows whether either continuation helps or harms
the starting method. Do not pool away topology-specific survival/motion tradeoffs.
No longer episodes or changed planning horizon are included.

## Interpretation and stopping

Failure-balanced sampling is supported as a useful repair only if independent
validation improves failure-sensitive prediction without a large ordinary-state
regression AND closed-loop survival improves without merely stopping locomotion.
Report effect sizes, counts and every seed; this small source experiment establishes
neither significance nor morphology transfer. Better open-loop prediction without
control benefit does not demonstrate a useful controller repair. No improvement
does not exclude other loss/capacity/state/coverage explanations.
Primary endpoints are H10 terminal-ending collapse false-negative count/rate and
paired balanced-minus-uniform closed-loop collapse difference. Report descriptive
effect sizes rather than an uncalibrated binary repair-pass rule. Ordinary false
positives, motion, return and ordinary-state prediction errors qualify any gain.
Validation cohorts are horizon-specific: endpoint distance0 terminal-ending,1..10
precollapse,>10/no observed terminal ordinary. A fitting complement is merely not
observed failure-adjacent, including budget-cut episodes; it is not a safe label.
For primary collapse sensitivity use any-imagined collapse within H10, matching
the planner's stopping rule; report endpoint-only sensitivity alongside it.

Benchmark100 fitting rounds and one short simulator validation/evaluation path
before projecting full cost. Initial target <=60 minutes elapsed with3 independent
CPU workers, one Torch/BLAS thread each; revise estimate before a projected overrun.
Hard cap3 hours elapsed; retain partial failures and checkpoints. No automatic
repair, extra updates, seed expansion or Experiment3 execution. Operational fixes
must preserve protocol and retain failed attempts. Resume only matching hashes and
saved optimizer/RNG/round accounting; avoid duplicate running work.

Record reviewed protocols in notebook Experiments18/19 before execution, and append
results/failures/tests/compute/artifact links plus parent-index updates with verified
CLI readback afterward. Runtime/checkpoint sources must match the original campaign
math/dependencies, with pre/post hashes. Generated artifacts stay ignored. Add tests
for episode-safe strata, replacement/fallback/counts, exact uniform sampling,
unchanged update math and exact continuation/resume.
Operational recovery filters progress history to the restored checkpoint round.
The three-hour campaign deadline is enforced during preparation, fitting,
validation and control, including queued workers; it is not a per-job allowance.

Sampler contract: the recursive sampling call returns positions in replay.starts(5),
not raw starts. Assert alternating one-step/window call phase and argument bounds;
persist phase, RNG and exposure counters in every restart checkpoint. Uniform
delegation makes no extra RNG draws. Verify bitwise model/target/Adam parity over
complete topology rounds against unwrapped native updates. Original shared Adam
starts at60,000 steps and must end at66,000; each learner view ends at22,000 updates.
Each topology/arm/seed receives64,000 one-step examples,64,000 five-step windows
and320,000 recursive targets. Restore actor/target-actor states too. Use literal
reset formulas rather than the old evaluate helper's transformed seed API.

## Independent review

Reviewer `/root/review_candidate_plan` checked the notebook and owning native
update/checkpoint code. All required sampler, state-sharing, update accounting,
round-boundary resume, reset-identity, validation-cohort, descriptive-endpoint and
complement-label clarifications above were incorporated before outcomes. Uniform
native-update parity and exact checkpoint-resume checks are mandatory prelaunch.

## Prelaunch clearance and measured compute

Independent reviewer performed a final read-only check after all implementation
fixes and reported no new blockers, ready to launch Experiments18–19 only. Eight
sampler tests and three native integration tests pass, including bitwise uniform
update parity, exact shared model/Adam/target/sampler restart and real simulator
reset/boundary/restoration smoke. Prior candidate metric tests also remain relevant.
The100-round/300-update single-thread fitting benchmark took14.7589 seconds,
projecting1,771 seconds (29.5 minutes) aggregate fitting across six jobs. With three
workers and preparation/validation/control, allow25–45 minutes elapsed within the
60-minute target and fixed three-hour campaign cap. No budget/method expansion.
Full execution uses one Torch/BLAS thread per CPU worker and preserves this plan,
config and runner for executed-source provenance. Completion is recorded separately.
