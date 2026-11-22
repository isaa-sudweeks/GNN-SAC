# GNN versus padded-MLP experts: baseline review

Review date: 2026-10-28. Status: published, locally validated code baseline; no production
run launched. The user selected existing GNN versus padded-MLP specialists,
approved the paper branch/tag workflow, and approved all listed archive candidates.

## Candidate code baseline

The starting point is `dcd910c1d5793f59e2f62760986a4a28f1336624` (main after
PR #40). The user approved integrating the following branches for later shared
controller studies while keeping the first expert comparison unchanged:

| Approved integration | Source tip | Default behavior |
|---|---|---|
| `experiments/diversity` | `59e288b` | Adds selectable CV/teacher tools; `cross_validation=disabled` remains the default. |
| `codex/signed-routing-ablation` | `ec1e6fe` | `graph_features.edge_direction=false`. |
| `codex/tube-virtual-nodes` | `c734e08` | Retains the tube-node design assessment. |
| `codex/implement-tube-virtual-nodes` | `962c6cd` | `graph_features.tube_nodes=false` and `graph_features.tube_physics=false`. |

The branches form a dependency chain and have been merged in that order on
`codex/paper-expert-baseline`. Merge resolution preserves PR #40's GNN/padded-MLP
policy contracts, physical action ordering and offline-only completion flow.
Tube diagnostics now prefer `distillation_complete.pt`, falling back to the
older `distillation.pt` for existing runs. Optional metadata reconstruction is
also false by default. Named ablation presets explicitly opt into their features;
ordinary GNN and padded-MLP launches do not.

`sim-to-real` is retained unchanged at both its local and remote tips. The 72
approved local candidates and 10 approved remote candidates are archived; all
82 annotated recovery tags are published and verified on origin. All four
integrated source branches remain in place. The paper branch and
`paper-experts-v1` tag are published; the tag remains fixed at
`a1519b5f1c8439f6882ea1f18ee1e7b187db8256`. Production qualification and
experiment settings remain pending. See [paper reproducibility](../usage/paper_reproducibility.md).

The question for this stage is: **How well can each architecture learn a
separate expert policy for each robot in a declared morphology set?** Each
policy trains on its own morphology. This does not measure a single shared
controller's capability or unseen-morphology transfer.

## Features to retain from main

| Feature | Production decision |
|---|---|
| GNN with two message-passing layers, scalar message attention, virtual-node context and combined critic readout | Retain current architecture; freeze its resolved configuration. |
| Padded MLP with existence/action masks and rigidity | Use the same control-node observation/action interface and environment as the GNN. Fix capacity across all experts. |
| Tensor replay, bounded replay storage, replay-ratio scheduling | Retain and match settings across architectures. |
| Batched MJX collection and Warp support | Retain capability; qualify the selected GPU configuration separately before launch. |
| Checkpoint/resume and finite-value checks | Retain; test resume on the production backend. |
| Corrected routed-action energy penalty, terminal velocity accounting and reward normalization | Retain existing fixes; use identical rewards and timing for both architectures. |
| PCGrad | Explicitly set the same choice for both. Recommend false for this single-task experiment; backend defaults currently differ. |
| Distillation, held-out evaluation and broken-node regimes | Preserve available code, but leave disabled in scratch-trained nominal experts. |
| Optional semantic graph features | Leave disabled for the initial existing-architecture comparison; a richer GNN input would require a separately described condition. |

The flat `sac_backend=mlp task=truss-mlp` emits actuator commands and is a
different control interface. It is outside the user-selected primary comparison.

Recounted from the actual backend defaults, excluding frozen target critics:

| Model | Actor | Critics | Total |
|---|---:|---:|---:|
| GNN | 570,386 | 1,407,526 | 1,977,912 |
| Padded MLP, capacity 21, widths 736/736 | 701,450 | 1,373,378 | 2,074,828 |

The padded MLP is about 4.9% larger overall. The pre-PR-40 YAML claimed a
0.2% match to a 2,078,008-parameter GNN; PR #40 already removed that stale
claim. This review records the current counts without changing model widths.

## Branch review and cleanup decisions

These are decisions about inclusion in this experiment, not claims that an
excluded feature is incorrect or scientifically uninteresting. Proposed
archival preserves exact committed tips and was explicitly approved by the user.
The [full inventory](paper_branch_inventory.md) includes every local and
remote ref, exact SHAs, worktree exclusions and the proposed recovery process.

| Branch or family | Review finding | Recommendation |
|---|---|---|
| `codex/add-configurable-multihead-attention` | Introduces multiple attention heads; the default head count is one, but higher counts change architecture and parameter count. | Do not merge for the existing-GNN comparison; archive candidate. |
| `codex/batched-mjx-rollouts` | Older alternate batched environment implementation under `env/truss`; main already has the generated-topology tensor-batched collector from PR #39. | Use main's implementation; archive candidate. |
| `experiments/graph-identity-features` | Historical graph semantic features; main already has configurable node roles, edge roles and distances through PR #12. | Use main's implementation; archive candidate. |
| `optimization/timing` | PCGrad optimization/profiling lineage; main already contains PCGrad optimization from PR #19 and subphase profiling. | No wholesale merge of the old branch; archive candidate. |
| `reward_function` | Adds a danger-zone barrier reward and additional evaluation diagnostics. Its local descendant also contains a virtual-node action fix already present in main's critic mapping. | Keep existing objective for paired experts; archive candidate, including distinct local/remote tips. |
| `experiments/lagrangian-rigidity`, `experiments/1-sample`, `experiments/safety_logging`, `experiments/safety_surrogates` | Constrained safety objectives, curriculum and surrogate penalties; these alter the learning objective or introduce additional experimental conditions. | Separate ablations; archive candidates. |
| `experiments/diversity` | Adds farthest-point/random-teacher CV utilities and changes split configuration. | Merged by user request for future general-controller work; retain source branch per user decision. No CV enabled by default. |
| `codex/signed-routing-ablation` | Adds signed routing features and a controlled transfer study on top of diversity/CV work. | Merged by user request; retain source branch per user decision. Signed routing is off by default. |
| `codex/tube-virtual-nodes`, `codex/implement-tube-virtual-nodes` | Tube-specific graph nodes, tube-physics experiments and teacher/student fidelity diagnostics. | Both merged by user request; retain source branches per user decision. Tube nodes and tube physics are off by default. |
| `claude/robot-topology-distribution-9e3b7a` | Analysis pipeline plus committed figures/data and legacy requirements edits. Local descendants contain different analysis corrections. | Do not merge the old analysis branch wholesale; remote/archive candidates preserved separately. Retain the attached local analysis worktree. |
| `sim-to-real` | Hardware inference, tracker mapping, serial commands and session diagnostics; not the simulator training baseline. A local tracker-analysis branch contains additional replay safeguards. | Leave both local and remote `sim-to-real` branches unchanged, per user instruction. |
| `codex/per-topology-distillation-decay` | Adaptive teacher release; notebook records incomplete runs and concerning post-release declines. | Exclude from scratch experts; retain branch because it has an attached research worktree. |
| `codex/sgs-topology-sampling` | Adaptive collection/replay; notebook records incomplete runs and no established useful transfer. | Exclude from scratch experts; retain attached research branch. |
| `codex/shared-dynamics-mpc-experiments`, `codex/dynamics-generalization-experiments`, `codex/gnn-dynamics-mpc-pilot` | Learned dynamics/MPC is a different controller and research question. | Exclude from the expert architecture baseline; retain attached research branches. |

Most other local branches are implementation/review ancestors of merged PRs.
Squash merging means ancestry alone cannot establish that a feature is missing.
Do not merge those branches just because `git log main..branch` is nonempty.
The completed archive preserves both local-only work and differing remote tips;
attached branches remain in place. The inventory records all recovery tags and
completed local/remote operations. The table above retains the original review
recommendations; every archive candidate in that table has now been archived.

## Before selecting the final paper commit

1. Use the published code-baseline tag and paper branch with the run-record
   workflow. Branch cleanup and recovery-tag publication are complete.
2. Declare morphology IDs, seeds, simulator (native or MJX/Warp), abstract or
   realistic physical models, control interval and randomization distribution.
   The validator's historical 13-topology list is not an approved paper set.
3. Freeze transitions per expert, seed collection, pretraining updates,
   replay ratio/capacity, batch size, normalization, entropy and checkpoint
   selection. `steps`, `batch_size` and `buffer_size` are per-topology inputs.
4. Fix padded capacity from the declared morphology envelope, and count actual
   trainable actor/critic parameters for both configurations. Describe whether
   the comparison matches parameter count, compute or a tuning budget.
5. Declare evaluation reset seeds and episode horizon, distance and survival
   measures, and final-checkpoint versus validation selection before results.
   Training-seed equality alone does not establish paired evaluation resets.
6. Run bounded GPU pilots at the intended vector count and model widths for
   both architectures, checking finite learning, memory, checkpoint/resume and
   evaluation. CPU tests cannot establish GPU throughput or learning quality.
7. Save resolved configs, commands, dependency lock hash, environment/GPU
   details and an immutable code tag. After the protocol and pilots pass,
   freeze the exact integrated commit used by every production run.

## Validation

The locked environment was installed with `uv sync --frozen`. The following
record describes the original PR #40 baseline before the four integrations:

- `uv run --no-sync python -m unittest discover -s tests -v`: **383 tests
  run; 381 passed and 2 CUDA-only tests skipped**, in 220.593 seconds.
  This includes native and CPU MJX integration, padded-MLP training,
  distillation, checkpoint/resume and vectorized collection checks.
- `uv run --no-sync python scripts/validate_padded_mlp_topologies.py`:
  **passed** for the historical 13-topology contract, with a maximum of 21
  control nodes and 17 active actions. This does not choose the paper set.
- Constructed both actor/critic models from the actual Hydra backend defaults
  and recounted trainable parameters as recorded above.
- `git diff --check`: passed.

Post-integration validation on the completed merge resolutions:

- `uv run --no-sync python -m unittest discover -s tests -v`: **426 tests
  run; 424 passed and 2 CUDA-only tests skipped**, in 359.631 seconds.
- Includes explicit checks that optional graph features, CV and distillation
  remain disabled by default for both backends, tube-GNN teacher to padded-MLP
  target alignment, completion-checkpoint selection, and tube offline/online
  training. Tube membership and route lengths are also checked across the
  configured development morphologies.
- `uv run --no-sync python scripts/validate_padded_mlp_topologies.py`: **passed**
  for the historical 13-topology capacity contract after integration.
- All four approved source tips are ancestors of the preparation branch.
  Both local and remote `sim-to-real` tips remain
  `939bb0295c05b674efe246b3f6f2b6a53d7b52ab`.
- `git diff dcd910c..HEAD --check`: passed.

No production experiments were launched. The Obsidian notebook was read for
branch inclusion decisions; no new experiment was performed or recorded there.

Notebook sources read through the official CLI:
`Research/06 Experiment Notebook/2026-10-22-Success-Guided-Sampling.md` and
`Research/06 Experiment Notebook/2026-10-22-Performance-Based-Distillation-Decay.md`.
