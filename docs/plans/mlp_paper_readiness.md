# MLP audit and paper-readiness plan

Audit date: 2026-10-27. Baseline commit: `ed57cb4`.
Status: implementation and CPU integration checks completed; final review and
validation record below distinguish delivered capabilities from paper-run qualification.

## Recommended scope

Use `sac_backend=padded_mlp` for both single-morphology specialists and the
multi-morphology generalist in the primary comparison. Both then consume the
same control-node state and produce the same node commands as the GNN. Keep
`sac_backend=mlp` as the existing flat-policy option, with an explicit generated
`task=truss-mlp` specialist recipe. Its actuator-action interface is a different
experimental condition and should be labeled separately.

Deliver three supported paper workflows:

1. Train and evaluate a single-morphology MLP specialist from scratch.
2. Train and evaluate one padded MLP across a frozen morphology split.
3. Distill specialist experts into a padded MLP, then optionally continue online
   SAC with the existing decaying KL guidance. Support both one and many teachers.
   Implement an explicit offline-only mode that distills, evaluates, exports and
   stops; this mode does not exist in the current trainer.

Start with GNN experts (to reuse existing specialist artifacts) and padded-MLP
experts (to allow an all-MLP pipeline). Arbitrary flat actuator-action experts
need a separate, validated interface adapter; they are not interchangeable with
control-node Gaussian experts.

## Baseline audit findings (before implementation)

| Capability | Audit finding | Owning source |
|---|---|---|
| Flat specialist SAC | Implemented; short generated tetrahedron CPU training passed. Default MLP config still selects the legacy XML velocity task, so `truss_topology` alone does not select a generated specialist. | `sac/sac.py`, `sac/common/actor_critic.py`, `config/sac_backend/mlp.yaml`, `env/mujoco.py` |
| Padded specialist SAC | Implemented; short tetrahedron CPU training passed. | `sac/padded_mlp_sac.py` |
| Padded generalist SAC | Implemented; short heterogeneous tetrahedron/octahedron CPU training passed with default tensor replay. | `sac/train.py`, `sac/gnn_sac.py`, `sac/common/tensor_gnn_buffer.py` |
| Padding and masking | Fixed capacity; state, existence mask, action mask, rigidity; passive actions removed from policy entropy and zeroed for criticism/routing; oversized graphs rejected on observation encoding. | `sac/common/padded_mlp_actor_critic.py` |
| Connectivity | MLP ignores edges; node order remains meaningful. It is not permutation invariant. | `sac/common/padded_mlp_actor_critic.py` |
| PCGrad | Inherited by padded MLP and unit-tested. Defaults differ: GNN on, padded MLP off. | `sac/gnn_sac.py`, backend YAMLs |
| Checkpoint/inference | Padded schema includes capacity and dimensions; incompatible capacity rejected. Graph inference dispatch supports padded MLP. Full padded trainer resume and production inference still need dedicated integration checks. | `sac/padded_mlp_sac.py`, `sac/gnn_infer.py`, `tests/test_padded_mlp.py` |
| Expert distillation | Implemented only for GNN students and GNN teachers. Both backend gates explicitly reject MLPs. | `sac/common/distillation.py:396`, `sac/common/distillation.py:570` |
| Student Gaussian interface | Padded actor has internal distribution calculation but no public `policy_distribution`; flat actor also lacks the distillation interface. | `sac/common/padded_mlp_actor_critic.py`, `sac/common/actor_critic.py` |
| GPU collection | Padded MLP inherits `act_dense`; MJX vector collection accepts it. GPU correctness, memory and throughput were not established by this CPU audit. | `sac/gnn_sac.py`, `sac/trainer/online_trainer.py` |
| Capacity coverage | Config and validator describe a fixed 13-topology distribution with 21 maximum control nodes. The notebook's newer 15-training/4-validation study is not this checkout's `random_5fold` split. Do not assume coverage of the intended paper set. | `scripts/validate_padded_mlp_topologies.py`, `config/cross_validation/random_5fold.yaml` |

The existing padded input is `M * (F + 2) + 1`, currently 169 at `M=21`,
`F=6`; outputs allocate 21 node-action slots. `[736,736]` is documented as
roughly parameter-matched to an older GNN. The implementation audit recounts
the current defaults as 2,074,828 padded parameters (701,450 actor; 1,373,378
critics) versus 1,977,912 GNN parameters (570,386 actor; 1,407,526 critics),
about 4.9% larger. Recount the actual paper models before claiming matching.

## Priority 1 — Freeze the experiment contract and make validation explicit

- [x] Add named specialist/generalist paper presets, with explicit generated
  task, control graph, observation source, configurable morphology inputs, capacity, hidden
  sizes, entropy convention and PCGrad setting. Preserve generic MLP defaults.
  Leave paper morphology lists unspecified for now, per the user's direction;
  do not choose a split from the existing repository or notebook campaigns.
- [x] Generalize the capacity validator to accept the resolved split/config;
  check control-node count, active action count, feature width and action order.
  Fail before training if any requested morphology exceeds capacity. Validate
  evaluation capacities without collecting training data or teacher targets.
- [x] Keep one fixed capacity across paired folds/seeds and checkpoint it. Size
  it from a declared morphology envelope, never held-out rollout outcomes.
- [x] Reject incompatible backend/task combinations at startup, including flat
  MLP with graph observations and heterogeneous flat multi-morphology runs.
- [x] Validate simulator/domain-randomization compatibility during startup,
  before teacher checkpoint loading or target-cache preparation. MJX rejects
  model-rebuilding randomization, including explicitly enabled length-scale
  and physical-parameter rebuilding. The current default profile disables both.
  Require an explicit compatible
  fixed-shape DR profile for MJX, or select native MuJoCo when rebuilding is part
  of the experiment. Never silently disable randomization to make a run launch.
- [x] Record actual trainable actor/critic counts separately and together for
  GNN, padded specialist and padded generalist; distinguish parameter matching
  from equal architecture width or equal compute.

Acceptance: config resolution and dry-run validation identify all morphology
IDs, physical/control dimensions, effective budgets and architecture counts;
bad masks, oversized graphs and incompatible task/backend or simulator/DR
choices fail early. Regression-test invalid MJX/DR configs with teacher paths
that must never be opened, and test valid fixed-shape MJX and native rebuilding
profiles. Freeze the selected simulator and DR profile in each paper preset.

## Priority 2 — Extend Gaussian distillation to padded MLPs

- [x] Add `PaddedMLPActorCritic.policy_distribution(obs, deterministic=True)`
  returning bounded pre-tanh mean/log-std in active-node order, with gradients.
  Reuse `_distribution`; do not include padding/passive rows in the loss.
- [x] Replace GNN-only construction/gates with a small explicit dispatch for
  supported graph-transport policy backends (`gnn`, `padded_mlp`). Reuse the
  current distillation trainer, optimizers, sharded caches and schedules.
- [x] Validate padded teacher checkpoint schemas before loading weights.
  Preserve frozen teachers, training-topology-only selection and full trainer
  checkpoint requirements (`config`, `agent`, `buffer`).
- [x] Separate physical/control semantics from model-specific preparation.
  The current validator requires equal `use_virtual_node` and identical prepared
  graph structure. The default GNN uses a virtual node; padded MLP forbids it.
  Prepare both from the same raw graph and explicitly prove active physical
  action identity/order even when the teacher adds a virtual node. Do not just
  remove the compatibility check.
- [x] Fix the online teacher-view reuse condition in `Distillation.loss`:
  `graph_feature_schema` currently describes feature flags but excludes
  `use_virtual_node`, so equal schemas do not prove compatible prepared inputs.
  Reuse the student view only when the full preparation contracts agree;
  otherwise build the teacher view from raw replay. Include virtual-node
  preparation and backend-specific input semantics in that decision.
- [x] Preserve support for teacher graph features when the padded student uses
  only raw node state, masks and rigidity. A teacher's richer graph view must
  never be appended to the padded student's fixed input by accident.
- [x] Include teacher/student backend, padded schema and preparation contract
  in cache and resume compatibility. Version cache contracts as needed; reject
  incompatible reuse while retaining existing supported GNN resumes.
- [x] Retain analytic `KL(teacher || student)`, averaging active nodes within a
  graph and topology groups equally. Preserve actor-only offline updates,
  independent critic initialization, one-time Adam reset, online KL decay,
  zero-weight teacher bypass and PCGrad combination semantics.
- [x] Add an explicit offline-only distillation option (proposed
  `distillation.offline_only=true`; disabled by default). Require distillation
  enabled and a positive offline-update budget. After offline updates, evaluate,
  save the student export and a resumable completion checkpoint, then stop before
  seed collection or SAC updates. Update `run_training` validation so this mode
  can use `steps=0`; ordinary online training must still require positive steps.
  Offline-only runs must not derive a required online KL decay duration from
  the zero interaction budget. Persist the execution mode and completion state;
  resume must preserve the mode and must not repeat completed updates or start
  SAC implicitly. Online fine-tuning can start as a separately configured run
  initialized from the exported student, with fresh online optimizer/replay state.

Acceptance tests: GNN→padded and padded→padded, one and two morphologies;
different graph sizes/passive masks; default virtual-node GNN teacher; exact
Gaussian-target alignment; offline actor gradients with unchanged critics;
teacher parameters unchanged; finite KL falling on a fixed fixture dataset;
ordinary and PCGrad online updates; zero-weight baseline equivalence; missing
teachers/schema errors; held-out teacher paths never opened; deterministic
offline save/resume, cache invalidation and online resume without target cache.
In particular, test identical graph-feature flags with a virtual-node GNN
teacher and a non-virtual padded student through the online teacher-view branch;
compare live teacher parameters and KL against cached targets on the same raw
observations. Test both compatible-view reuse and required reconstruction.
For offline-only runs, assert zero collected training transitions, zero SAC
updates, unchanged critics/alpha, successful evaluation/export, and completed
resume without repeated optimization. Evaluation rollouts are separate from
training interaction counts. Check that ordinary distill-then-SAC still works.

Flat `SAC` expert support is a separate extension if required. Its flat
observation/action ordering and replay lack the graph contract, and actuator
Gaussians cannot simply be reinterpreted as node-command Gaussians. Use padded
specialists for the same-interface MLP expert pipeline.

## Priority 3 — Close training, resume and evaluation integration gaps

- [x] Add bounded CPU integration coverage for padded specialist, heterogeneous
  generalist, offline-only distilled student and distill-then-SAC generalist
  using production tensor replay; test the
  legacy replay fallback where distillation promises compatibility.
- [x] Exercise tensor replay learner updates with PCGrad on/off, batched and
  serial actions, passive/broken masks if present in the paper protocol, and
  finite checks. Existing padded unit updates use legacy `GNNBuffer` fixtures.
- [x] Verify full trainer checkpoints after optimizer updates, restoration of
  actor/critics/alpha/optimizers/replay/counters, continued updates and schema
  rejection. Keep simulator-state resume limitations explicit.
- [x] Verify saved padded policy inference through `sac/gnn_infer.py`, including
  non-default capacity/width and deterministic action equivalence. Document
  required config reconstruction for agent-only exports. Give flat specialists
  a tested evaluation/export recipe if retained in the paper matrix.
- [x] Test held-out evaluation isolation: no held-out replay, teacher load,
  normalization updates or offline targets. Use per-morphology metrics.

Acceptance: every workflow trains, saves, reloads, evaluates and resumes under
bounded integration tests; disabling distillation preserves ordinary SAC.

## Priority 4 — Qualify on the paper GPU and freeze launch recipes

- [ ] On the intended cluster, run the same padded workflows with the selected
  simulator and the explicit DR profile validated in Priority 1. For MJX, freeze
  the implementation and fixed-shape DR settings, use tensor replay, and check
  `act_dense` against serial actions and zero passive outputs. If the paper
  requires model rebuilding, qualify native MuJoCo instead; MJX acceleration
  must not change the declared physical distribution.
- [ ] Record peak device/host memory, collection and optimizer throughput,
  compilation/startup time and checkpoint/cache sizes on the full split.
  Include teacher residency and first full-checkpoint deserialization cost.
- [ ] Run a short learning pilot beyond seed collection with actual specialist
  checkpoints. Check offline imitation, post-distillation evaluation, online
  fine-tuning and the point where guidance reaches zero. CPU mechanics alone
  do not establish learning or transfer.
- [ ] Test interruption and Slurm resume; verify unique run directories,
  checkpoint retention and monotonic W&B steps across offline/online stages.
- [x] Update `docs/usage/padded_mlp_baseline.md`, `distillation.md`, `training.md`,
  `cluster_and_resume.md` and the docs index alongside implementation changes.
  Provide exact launch commands, teacher manifests and a launch preflight.

Acceptance: full-shape GPU pilots finish with finite training state, successful
resume/evaluation, measured resource headroom and reproducible commands.

## Paper run matrix and budget rules

| Condition | Training data | Initialization/guidance |
|---|---|---|
| Padded MLP specialist | One training morphology per policy | Scratch SAC |
| Padded MLP generalist | All training morphologies | Scratch SAC |
| Offline-only distilled padded MLP | Training experts' replay only | Offline KL; evaluate/export without online SAC |
| Distilled padded MLP generalist | Training experts' replay, then training morphologies | Offline KL, then SAC + decaying KL |
| Matched GNN conditions | Identical splits and simulator/control conventions | Match the corresponding scratch/distilled protocol |

Treat a flat actuator-action specialist as an additional labeled baseline if
needed. Do not fold its scores into an architecture-only comparison.

Freeze seeds (at least the existing three-seed convention), model capacity,
PCGrad, reward/normalization, simulator timing, realistic settings, DR, replay
ratio and evaluation seeds before launch. Report teacher-training and offline
imitation costs separately from student online interactions. A scratch versus
distilled comparison is not equal total data exposure merely because the online
transition counts match.

`steps`, `batch_size` and `buffer_size` are configured **per topology** and scaled
by the parser. Thus `steps=24` with two topologies collected 48 transitions in the
audit smoke. Report both per-topology requested values and effective totals;
do not accidentally multiply generalist budgets twice. Offline distillation
updates are separate from SAC `pretrain_steps` and do not advance interactions.

Choose evaluation endpoints before inspecting results. Report distance,
survival/episode length, per-morphology values and seed variation at matched
budgets. Preserve final-test exclusion until the protocol's final evaluation.
The notebook's earlier decay/sampling pilots do not establish final transfer
performance and are not evidence that the current checkout includes those
branches' capabilities.

## Decisions to clarify before freezing paper presets

Implementation can proceed on the policy interfaces and regression fixtures
while these experiment choices remain open. Do not launch paper runs or claim
matched comparisons until they are resolved:

- Which simulator/physical distribution should the paired paper runs use:
  MJX with explicit fixed-shape randomization, or native MuJoCo with
  model-rebuilding randomization? A different DR distribution is a separate
  experimental condition, not a performance-only setting.

Working scope includes both offline-only and distill-then-SAC operation, and
GNN and padded-MLP experts. Exact budgets, seeds, teacher manifests, capacity,
PCGrad and architecture counts must be recorded during Priority 1; these do not
need to block the backend implementation.

## Audit validation record

- Installed locked dependencies with `uv sync --frozen`; no dependency edits.
- `uv run --no-sync python -m unittest tests.test_padded_mlp tests.test_distillation -v`:
  **49 passed**, including the existing GNN distillation environment smoke.
- Three bounded native CPU CLI smokes: flat `task=truss-mlp` tetrahedron,
  padded tetrahedron, padded tetrahedron/octahedron. Each used hidden width 16,
  `steps=24`, `seed_steps=8`, `pretrain_steps=1`, `batch_size=4`,
  `buffer_size=64`, `replay_ratio=1`, `max_steps=4`, `nsubsteps=1`.
  W&B, video, agent/CSV saves and periodic checkpoints were disabled.
  Each completed online optimizer updates (5, 5 and 6 respectively).
- `uv run --no-sync python scripts/validate_padded_mlp_topologies.py`: passed
  for all 13 listed topologies; confirmed maximum capacity of 21 control nodes.
  This confirms that declared set only, not the newer notebook split.
- GPU/cluster runs, long-run learning, new MLP distillation and full-suite
  qualification were not performed. No production experiments were launched.
- Notebook context read through Obsidian: `Research/06 Experiment Notebook/2026-10-22-Performance-Based-Distillation-Decay.md`
  and `2026-10-22-Success-Guided-Sampling.md`. This audit is not a new experiment.

Implement in order: contract/preflight → Gaussian distillation → lifecycle
integration → GPU pilots and frozen launch recipes. Completion means every
acceptance gate above passes, not merely that a long job starts.


## Implementation and review record

Implemented on `codex/mlp-paper-readiness`; morphology inputs remain configurable,
and no paper split or production training budget has been selected.

- Named specialist/generalist presets, early backend/simulator validation and
  static capacity/action-order checks cover training, held-out lists and explicit
  evaluation task aliases. The standalone validator accepts topology lists,
  cross-validation settings or run YAML and reports counts/budgets/contracts.
- GNN and padded experts can teach padded students, including mismatched virtual
  nodes and graph features. Cache and resume contracts distinguish input semantics;
  padded targets retain rigidity. Unsupported flat actuator experts fail explicitly.
- Offline-only mode evaluates/exports/checkpoints without SAC interactions;
  completed resumes do not optimize again. Fresh online initialization from an
  export is separate from full resume. Slurm completion detection recognizes
  offline-only completion metadata.
- Review comments addressed: (1) preflight originally missed `eval_task` and
  aliases; now tested before teacher access and included in the CLI report;
  (2) online mismatched-view KL originally repeated per-graph CPU checks; replay
  insertion/load contracts now enforce invariants while bound online batches
  avoid those synchronizations. Unbound diagnostics remain defensive.
- Final independent lifecycle/contract review reported no remaining concrete
  blockers. This is a code/mechanics review, not evidence of learned performance.
- Focused validation: **51 passed**, including native specialist/generalist
  tensor-replay updates, PCGrad on/off, dense/serial passive-mask equivalence,
  CPU MJX vectorized padded training, exports, full state restoration,
  held-out normalization/replay isolation, virtual-GNN and padded-teacher
  distillation, offline-only completion/resume and fresh online fine-tuning.
- Final full discovery: **383 tests run, 380 passed and 3 accelerator-only tests
  skipped**, with no failures/errors. Command:
  `uv run --no-sync python -m unittest discover -s tests -v`.
- Both named specialist/generalist CLI presets completed bounded native CPU
  training with optimizer updates. The oversized explicit evaluation alias was
  rejected with its required 12 nodes versus configured capacity 8.
- The historical 13-topology validator and a separate registered evaluation-task
  alias capacity validation passed; the negative capacity check failed as expected.
- GPU qualification attempted on ORC `login01`: Python 3.12.13, PyTorch
  2.13.0+cu130, driver 580.178.04, Tesla T4. Device discovery succeeds, but compute
  mode is `Prohibited`; the first CUDA allocation fails with
  `cudaErrorDevicesUnavailable`. No GPU learning/throughput claim is supported,
  and device mode, existing checkouts, environments and jobs were left intact.
- Priority 4 hardware qualification, actual-expert learning pilots, resource
  profiling, cluster interruption/requeue and frozen paper launch budgets remain
  pending. No long-running or production experiment was launched.
