# Comparative morphology generalization: next-stage design

Status: proposed follow-on, not executed or a frozen experiment. Experiments17–19
do not answer this question. Review notebook history and completed results before
freezing any comparative protocol. The user will start this comparison separately.

## Decision after Experiments18–19

The one targeted failure-balanced replay repair did not consistently improve
source control. Uniform/balanced collapses per15 fresh control episodes were
tetrahedron1/2, six-node13/11 and eight-node14/15. Both continuations had28/45 total
collapses versus27/45 for the original model. Six-node gains varied by seed;
tetrahedron movement was lower in all three seeds and eight-node return was lower
in all three. Improved eight-node fixed-pool ranking did not improve survival.
See [measured results](../usage/dynamics_results/failure_balanced_replay_2026-10-25.md).

Proceed to the matched comparison rather than automatically testing another
sampling repair. Use the original shared gamma=.995 model/checkpoint family as
the predeclared dynamics reference. Keep uniform and balanced continuation results
as source-development ablations; do not choose a different model per robot from
these evaluation outcomes. Weak source control is an interpretation limit, not a
requirement to pursue an unlimited series of prerequisite repairs.

## Question to measure

At equal real simulator interaction budgets and identical source-topology access,
does a shared learned dynamics model with frozen MPC retain more control performance
on unseen morphologies than a shared model-free GNN-SAC policy?

Operationalize "easier to generalize" as held-out locomotion and survival at a
matched data budget, with source-to-held-out degradation reported separately.
Prediction transfer is supporting evidence, not a substitute for control transfer.
One fixed budget tests one operating point; sample-efficiency claims need a
predeclared budget curve. Record fitting updates and planner compute/latency rather
than equating interaction counts with total compute.

## Matched comparison requirements

- Train one source-generalist dynamics model and one source-generalist GNN-SAC
  policy on exactly the same source morphologies, with matched physics and
  observation/action availability. A topology-specific specialist or a policy
  trained on a different source set is not the main zero-shot comparator.
- Existing gamma=.995 models were trained on three sources,20,000 transitions per
  topology and three seeds. Reusing those as one arm requires matching that source
  access and budget in the other arm, or rerunning both under a new frozen protocol.
  Do not silently compare against eight-source or differently trained policies.
- Both methods should optimize the same task reward: normalized physical-node
  progress, time-integrated routed-command effort and collapse charge. Current SAC
  environment return and MPC task score are not automatically equivalent. Audit or
  implement the matching reward before training. Document normalization/discount,
  time-limit bootstrap, update budgets, architecture and exploration differences.
- Pair evaluation reset states and report each topology and training seed. Freeze
  checkpoint selection using source data only; no adaptation, reward tuning,
  normalization fitting or checkpoint choice from held-out outcomes.
- Evaluate familiar-source control alongside held-out control. Poor source control
  limits attribution of poor held-out results to transfer; source perfection is
  not required to measure performance. Avoid unstable retention ratios if source
  return is near zero or negative; use absolute differences with survival/movement.
- Select held-out morphologies before evaluation, retaining the existing exclusions
  (`octahedron`, `henneberg_n8_1tube_57`) and final seven-node confirmation robots.
  Compatibility inspection is allowed; repeated tuning on held-outs consumes their
  role and requires reserving untouched confirmation morphologies.
- Use common episode limits and source-frozen planning settings. Report collapse
  timing, displacement, task return and latency separately. Longer episode and
  longer planning-horizon changes are separate factors, not interchangeable tests.

## Decision after Experiment17

Use the finite-candidate diagnostic to identify whether model risk prediction or
candidate coverage warrants one source-only change. It cannot identify a uniquely
correct repair. If a repair is tested, predeclare it and confirm source outcomes
before incorporating it in the comparative arm. Keep a fixed unmodified reference.
Do not let an open-ended sequence of diagnostics replace the central comparative
experiment. Freeze the matched comparator protocol, compute estimate and stopping
rules next; record it in the same Obsidian experiment folder before execution.
This historical decision has now been followed by Experiments18–19; the current
recommendation is recorded at the top of this document.

## Existing artifacts and reproducible handoff

- Original online dynamics implementation and recorded campaign source baseline:
  commit `3bea10554df7a25f4dbce503b77ae63dfa209840`, branch
  `codex/shared-dynamics-mpc-experiments`; included in the publication history of
  this follow-up. Raw generated checkpoints are not in Git.
- Original shared/separate campaign:
  `/Users/isaacsudweeks/.codex/worktrees/c400/GNN-SAC/outputs/dynamics-pilot/online-multi-source-2026-10-24`.
  Use `shared-seed{0,1,2}/checkpoint-{001600,004000,008000,012000,020000}.pt` only
  after verifying settings, hashes, source access and optimizer/interaction budgets.
- Compatible existing uv runtime:
  `/Users/isaacsudweeks/.codex/worktrees/c400/GNN-SAC/.venv`. Use uv with an explicit
  `UV_PROJECT_ENVIRONMENT` when reusing it; do not silently sync/change the recorded
  environment. The project uses `pyproject.toml` and `uv.lock`.
- Experiments17–19 artifacts are under this checkout's ignored
  `outputs/dynamics-pilot/online-candidate-ranking-2026-10-25/` and
  `outputs/dynamics-pilot/failure-balanced-replay-2026-10-25/`.
- Proposed source set: `tetrahedron`, `henneberg_n6_2tube_1`,
  `henneberg_n8_3tube_64`. Proposed untouched pilot holdouts: `octahedron`,
  `henneberg_n8_1tube_57`. Preserve final seven-node confirmation robots.
- Existing settings: 80-control-step episodes, .2-second controls, dynamics
  training horizon5, MPC horizon10, CEM16 candidates x2 iterations, gamma=.995.
  Keep these source-frozen or explicitly predeclare a different matched protocol
  before accessing holdout outcomes. Account for MPC planning latency separately.
- Recommended budget curve: 1,600/4,000/8,000/12,000/20,000 fitting transitions per
  source, three training seeds. Evaluate all predeclared checkpoints; do not select
  a preferred budget/checkpoint using held-out results. One budget alone is not
  evidence for a general sample-efficiency claim.

The native SAC environment reward differs from the MPC task objective. Matching
the reward and auditing graph/state/action conventions are concrete implementation
prerequisites for the next experiment, not completed work. Write and independently
review a fixed protocol, benchmark its cost, then execute it only in the new task.
Use `sac/train.py sac_backend=gnn` for any SAC commands, never legacy `gnn_train.py`.

Suggested starting request: [kickoff prompt](generalization_kickoff_prompt.md).
