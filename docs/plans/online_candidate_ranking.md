# Frozen online dynamics candidate-ranking diagnostic — October 25, 2026

Status: independently reviewed; reviewer revisions incorporated before execution.

## Research question and completed evidence

Ultimate question: does learning transferable dynamics make cross-morphology control
easier than learning a model-free policy? Notebook Experiments 09 and 12–16 were
checked. Experiment 09 ranked candidates for an older offline tetrahedron model,
with no candidate collapses. Experiment 16 evaluated newer frozen shared/separate
models on executed actions, not counterfactual candidates. Experiments 14/15 already
completed shared/separate source control and the gamma-only comparison.

This experiment asks whether the final gamma=.995 online models select collapsing
sequences when the same finite candidate set contains surviving alternatives. It
tests a prerequisite for dynamics transfer, not the ultimate comparative hypothesis.
No further training, repair, gamma search or unseen-topology evaluation is included.

## Fixed protocol

- Use the final 20,000-transition gamma=.995 shared and separate checkpoints for
  seeds 0/1/2, and only tetrahedron, henneberg_n6_2tube_1, henneberg_n8_3tube_64.
- Read campaign artifacts in checkout `c400`, using its existing compatible uv
  runtime and source modules. The new diagnostic lives in this checkout; pass the
  runtime/campaign paths explicitly, hash both source trees and input artifacts,
  record dependency versions, and verify inputs remain unchanged. This avoids
  copying the earlier experiment implementation into this older checkout.
- Two new paired reset episodes per topology/training seed, each generated once
  by each frozen controller: 36 episodes total. Reset seeds are
  12300000 + topology_index*10000 + training_seed*100 + episode_index. Verify these
  identities do not occur in saved fitting/monitoring/fresh-final records.
- No exploration or updates. Episode cap 80 controls (16 simulated seconds),
  planning horizon 10 controls (2 seconds), nsubsteps=100, gamma=.995. Preserve
  original normalization, action routing/masks, physics, effort weight .1, unit
  true-collapse penalty, CEM16 candidates x2 iterations and three knots. No terminal Q.
- Capture full simulator/wrapper snapshots before every real action. Per episode
  select the initial state, timestep20 if available, and five steps before its true
  terminal (or ten steps before the time cap for a surviving episode). Deduplicate
  identical timesteps; retain short episodes and missing strata. Label all strata
  by realized trajectory, and keep both generators separate. Selection uses no
  counterfactual outcomes. At most108 states; all source-development evidence.
- From each snapshot run both models' original CEM with common proposal seed and
  the generator's predecision warm-start plan. Pool every actual proposal from both
  searches (32 each), one zero sequence and eight seeded smooth-random sequences.
  Both models score the exact same73 sequences, with duplicates retained/labeled.
  Compare exact CEM-selected sequences and pooled-set argmax selections separately.
  Preserve PlanResult.actions and its exact evaluated index, including ties. These
  are recreated CEM selections with a common seed/generator warm start, not both
  controllers' original episode decisions. Save the original generating plans too.
  Retain all73 candidates but report unique-sequence and provenance-specific
  sensitivity metrics so duplicate first-iteration proposals do not dominate.
- Execute every sequence from the identical full snapshot in a separate simulator
  branch. Score actual discounted task reward from the original actual_step helper,
  stopping on true termination; artificial episode time caps must not truncate the
  diagnostic horizon. The underlying timeout is1000; do not alter counters or limits.
  Assert H10 branches from t<=79 never truncate. Save actual terminal elapsed
  intervals separately from the model's nominal .2-second assumption.
  Verify snapshot restoration, exact continuation and candidate-order
  independence before the full run. Save actions, model scores/collapse flags,
  simulator rewards/collapse flags and relevant state traces.

## Metrics and interpretation

Report by topology, training seed, generating controller and state stratum:
Spearman score correlation (constant scores explicitly uninformative), finite-set
regret, candidate collapse confusion including false positives, selected collapse
rate, and selected collapse when at least one candidate survives. Report whether
any surviving and any positive-progress surviving candidates exist. Candidate
counts and overlapping states are not independent samples; summarize paired
shared-minus-separate effects within states and at seed level, with raw rows.

If safe candidates exist and are repeatedly rejected for collapsing candidates,
this supports a model-scoring problem. If none exist, distinguish constrained
candidate coverage from physical unavoidability: this finite set proves neither
global impossibility nor adequate search. A 2-second surviving branch need not
support long-term survival. This test does not prove that collapse prediction alone
causes failure or that a particular repair will work.
Counterfactual branches execute open-loop sequences; deployed MPC executes one
action then replans. A selected collapsing branch is not causal proof that its
first action caused the observed closed-loop failure.

## Verification, budget and reporting

Review this plan with a subagent and record resolved findings before execution.
Add deterministic tests for episode/state selection, common-candidate accounting,
score/collapse metrics, ties and no-safe-candidate cases. Check native-model score
agreement, frozen parameters/buffers and hash preservation. Benchmark one complete
state before the fixed run; target <=30 minutes elapsed, hard cap60 minutes with
partial artifacts/failures retained. No automatic sample expansion or tuning.

Save incremental per-state outputs under ignored
`outputs/dynamics-pilot/online-candidate-ranking-2026-10-25/`. Write a versioned
measured report and update usage docs. Before launch, create notebook Experiment17
in `Research/06 Experiment Notebook/` and link from the parent. Append completed
results, test/reviewer evidence, compute, limitations and next decision; verify CLI
readback and preserve previous content.

## Later comparative experiment (not launched here)

Once source control supports meaningful transfer evaluation, predeclare a matched
source-trained GNN-SAC generalist versus frozen dynamics-MPC experiment on the same
unseen topologies, with no adaptation. Match real interaction budgets, physics,
objective, topology access and paired resets; disclose optimizer/planning compute
and latency. Report source-to-held-out performance drops, survival and displacement,
not just absolute source return. Easier generalization needs an operational metric
(e.g. held-out control at matched data budgets), and cannot be inferred from this
source diagnostic. Keep final seven-node robots untouched.

## Independent review and revisions

Reviewer `/root/review_candidate_plan` read the notebook index/Experiment16 and
owning runtime code. Revisions above preserve exact CEM tie selection, the online
per-step discounted objective (not the offline remaining-horizon penalty), common
warm-start interpretation, duplicate/provenance sensitivity, open-loop claim limits,
unchanged timeout counters, full checkpoint normalization, source/dependency hashes,
exact continuation and unmodified live state. Reset-dependent initial edge lengths
are logged separately; static graph/routing/physics invariants are checked, while
the model retains its original checkpoint reference lengths. No outcomes were
inspected to choose these revisions.

## Prelaunch benchmark and final integrity revisions

The one-state benchmark completed in7.38 seconds total,4.02 seconds for the
73-candidate diagnostic state; exact continuation/order, pairing, source/checkpoint
hashes and frozen model checks passed. At most108 states plus36 trajectory episodes
projects roughly8–12 minutes on this CPU, within the30-minute target and60-minute
hard cap. The full run keeps the fixed protocol unchanged. Before full execution,
reporting was extended to explicit seed/generator/stratum summaries; all eight
unit tests pass again. Fitting reset ranges are checked directly from checkpoint
runtime records. Nonfinite imagined states/progress fail with preserved evidence;
actual terminal elapsed times and unchanged live qpos/qvel/time/state are audited.
Notebook CLI append saves asynchronously; subsequent readback verified every
protocol paragraph and parent-index link after an initial stale read.
