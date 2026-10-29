# Revised paper baseline branch inventory

Snapshot: 2026-10-28, after fetching origin and integrating the four user-approved branches.
The user approved all 82 archive candidates. **72 local branches are now archived** under annotated local recovery tags.
The **10 remote candidates remain unchanged**, pending explicit permission to publish their recovery tags to the public origin.
All 82 recovery tags exist locally; none has been published. No worktree was removed.

## Decisions already made

| Branch | User decision | Current status |
|---|---|---|
| `experiments/diversity` | Merge for future shared control | Integrated; source branches retained. |
| `codex/signed-routing-ablation` | Merge, off by default | Integrated; `graph_features.edge_direction=false`. |
| `codex/tube-virtual-nodes` | Merge, off by default | Integrated design assessment; source branches retained. |
| `codex/implement-tube-virtual-nodes` | Merge, off by default | Integrated; `graph_features.tube_nodes=false`, `graph_features.tube_physics=false`. |
| `sim-to-real` | Leave as is | Both local and remote tips unchanged; excluded from archive candidates. |

## Approved remote archives pending publication

| Branch | Reason to exclude from the current baseline |
|---|---|
| `claude/robot-topology-distribution-9e3b7a` | Older analysis/figure pipeline; attached local analysis descendant retained. |
| `codex/add-configurable-multihead-attention` | Additional architecture variant; existing scalar attention retained. |
| `codex/batched-mjx-rollouts` | Older alternate rollout implementation; main already has tensor-batched collection. |
| `experiments/1-sample` | Safety-horizon sampling is a separate objective/curriculum experiment. |
| `experiments/graph-identity-features` | Historical implementation; semantic graph features already available in main. |
| `experiments/lagrangian-rigidity` | Separate constrained safety objective. |
| `experiments/safety_logging` | Diagnostics built on the constrained safety branch. |
| `experiments/safety_surrogates` | Separate surrogate safety penalties. |
| `optimization/timing` | Historical PCGrad optimization/profiling work; existing main implementation retained. |
| `reward_function` | Additional barrier reward and diagnostics would change the baseline objective. |

All local branches and live origin branches are listed below. `origin/HEAD` is a symbolic alias and is omitted.
Ahead/behind counts are relative to the starting main commit `dcd910c1d5793f59e2f62760986a4a28f1336624`, before integration.
Counts describe ancestry, not missing features: squash merges can leave historical commits ahead.
Local and remote tips can differ; preserve each exact SHA separately.
Branches attached to existing worktrees are retained. Related local hardware branches remain separately listed for review;
the user-protected `sim-to-real` branch itself is excluded from cleanup.
No worktree will be removed or modified by the proposed cleanup.

## Recovery and remaining archive procedure

Recovery tags for every approved exact tip have been created locally under
`archive/paper-prep-2026-10-28/local/<branch>` or `archive/paper-prep-2026-10-28/remote/<branch>`.
Local candidates were removed only after verifying exact local tag targets. For remote candidates,
publish and verify recovery tags on origin before removal. Recheck worktree ownership and ref SHAs;
skip branches that moved or became attached. Delete remote branches only with an expected-tip lease.
Tags preserve committed history only; never delete worktrees or uncommitted files.
Restore a branch using `git switch -c <branch> <archive-tag>`.

## Origin branches

21 remote branches at the review snapshot; **10 approved archives remain pending**.

| Branch | Exact tip | Behind / ahead | Action / status |
|---|---|---|---|
| `claude/robot-topology-distribution-9e3b7a` | `b5e4525a2cc36bbcbb7daf8aa38a50d614e8cc56` | 6 / 1 | approved; remote unchanged until recovery tags are published |
| `codex/add-configurable-multihead-attention` | `69f36f17387ccdaaf4567d1a5ac3eb485f345297` | 11 / 1 | approved; remote unchanged until recovery tags are published |
| `codex/batched-mjx-rollouts` | `21af5f32e125d833c76fad2029d0b1fadf902a98` | 73 / 4 | approved; remote unchanged until recovery tags are published |
| `codex/dynamics-generalization-experiments` | `6b618955f425ad234d141c1dcd63025f3126963e` | 1 / 3 | keep attached worktree branch |
| `codex/gnn-dynamics-mpc-pilot` | `c1d9f141bacaaeebebce19b10b4463b339e6f85d` | 1 / 4 | keep attached worktree branch |
| `codex/implement-tube-virtual-nodes` | `962c6cd8e48d907b382631b5a7ae17469f34d643` | 1 / 13 | integrated; retain source branch |
| `codex/per-topology-distillation-decay` | `243e8b6ea5f169d3e8da910777eab2531c76be76` | 1 / 3 | keep attached worktree branch |
| `codex/sgs-topology-sampling` | `12ef6b966d7fab040d69b9a494aa455b9d549856` | 1 / 3 | keep attached worktree branch |
| `codex/shared-dynamics-mpc-experiments` | `3bea10554df7a25f4dbce503b77ae63dfa209840` | 1 / 1 | keep attached worktree branch |
| `codex/signed-routing-ablation` | `ec1e6feea4cc8fe7c05211ca830b2ebefa2099a6` | 1 / 6 | integrated; retain source branch |
| `codex/tube-virtual-nodes` | `c734e08b736c4d00189d60d0ab487070c068f4d2` | 1 / 7 | integrated; retain source branch |
| `experiments/1-sample` | `c973596709e92b03dcba54ab582a7a4be23663af` | 38 / 4 | approved; remote unchanged until recovery tags are published |
| `experiments/diversity` | `59e288ba01bdd78cc5f61465fe5132c308cd0f40` | 1 / 2 | integrated; retain source branch |
| `experiments/graph-identity-features` | `36267f09db8118607fb90dac4b3bc93004769749` | 38 / 5 | approved; remote unchanged until recovery tags are published |
| `experiments/lagrangian-rigidity` | `fa7d1562521744d607200cfd1ad19489454503c2` | 38 / 3 | approved; remote unchanged until recovery tags are published |
| `experiments/safety_logging` | `f735d10f86d2d9137e46629791d04cbc0475a438` | 38 / 4 | approved; remote unchanged until recovery tags are published |
| `experiments/safety_surrogates` | `fe7d2c776b95c61df8b5236f673aa6360eadd505` | 38 / 4 | approved; remote unchanged until recovery tags are published |
| `main` | `dcd910c1d5793f59e2f62760986a4a28f1336624` | 0 / 0 | keep main |
| `optimization/timing` | `36fa74b124e8bde2367a7f414beb6082a4bb812b` | 14 / 8 | approved; remote unchanged until recovery tags are published |
| `reward_function` | `c043a0adf0d2a7e2fb447fda35367a5ee3620f04` | 39 / 4 | approved; remote unchanged until recovery tags are published |
| `sim-to-real` | `939bb0295c05b674efe246b3f6f2b6a53d7b52ab` | 11 / 11 | leave unchanged (user) |

## Local branches

87 local branches at the review snapshot; **72 have been archived locally**. Retained branches and the new paper branch remain.

| Branch | Exact tip | Behind / ahead | Action / status |
|---|---|---|---|
| `broken-node-interleaved-regime` | `f539129522fdffc112719ecf13f282ff08b2fabe` | 11 / 12 | archived locally; recovery tag retained |
| `claude/broken-node-rl-training-plan-4f6b6b` | `000146a60e757495e40047cbbbfa9f5766bf6464` | 11 / 9 | archived locally; recovery tag retained |
| `claude/config-distance-eval-audit-e71b37` | `096a683187557343bb6a9477afa771c890e71040` | 6 / 2 | keep attached worktree branch |
| `claude/cross-validation-distillation-analysis-92645a` | `1995813d6c1b15fa8f6a8cbef469c7f6720e2782` | 6 / 0 | archived locally; recovery tag retained |
| `claude/cv-training-generalization-analysis-deefcf` | `1995813d6c1b15fa8f6a8cbef469c7f6720e2782` | 6 / 0 | archived locally; recovery tag retained |
| `claude/discount-factor-default-8f3900` | `31cba71bcdc7ec4849d5d41f11af0efa2b4bf25f` | 3 / 1 | archived locally; recovery tag retained |
| `claude/distillation-pr-chain-018592` | `7d7cb0ec5a051a307d696d98767501003bbf9109` | 11 / 14 | archived locally; recovery tag retained |
| `claude/distribution-plots-robot-topology-d95aa0` | `47b3657b26e98ea0bfe8cb65b9343b27932488e8` | 6 / 2 | archived locally; recovery tag retained |
| `claude/documentation-cleanup-proposal-2cbeaa` | `7ddeb70926423b28e655cafee4b5363051d49395` | 5 / 4 | archived locally; recovery tag retained |
| `claude/gpu-optimization-learning-regression-8895de` | `ed57cb4f6f47535a57ec28222222514aa96352da` | 1 / 0 | keep attached worktree branch |
| `claude/gpu-utilization-training-ff5b72` | `9e20513a3adfeaef9b9497ed228af38fd6af7bf1` | 2 / 2 | archived locally; recovery tag retained |
| `claude/pensive-cartwright-51ab63` | `25ebbcd3ffc1d86a21115ea1e7b7fa878541be94` | 5 / 1 | archived locally; recovery tag retained |
| `claude/pr-35-comment-response-7a92fd` | `63f6c8b5200680a41a9aa0fa371cd6d956a06a2d` | 6 / 9 | archived locally; recovery tag retained |
| `claude/random-cross-validation-a896d0` | `4b8b8412622bcf065b753163ea5244fd257c4434` | 11 / 6 | archived locally; recovery tag retained |
| `claude/robot-topology-distribution-9e3b7a` | `b5e4525a2cc36bbcbb7daf8aa38a50d614e8cc56` | 6 / 1 | archived locally; recovery tag retained |
| `claude/robot-topology-distribution-c6d6f0` | `147eada9d15499c74078e69172b4b38f965b5a8a` | 3 / 0 | archived locally; recovery tag retained |
| `claude/robot-topology-generalization-eec5e0` | `1995813d6c1b15fa8f6a8cbef469c7f6720e2782` | 6 / 0 | archived locally; recovery tag retained |
| `claude/robot-topology-selection-4670c6` | `f4bff3baa69ff9125bd2bc10d6fdae596035834f` | 5 / 0 | archived locally; recovery tag retained |
| `claude/wandb-mcp-server-setup-c6bba4` | `1995813d6c1b15fa8f6a8cbef469c7f6720e2782` | 6 / 0 | archived locally; recovery tag retained |
| `codex/GPU_optimizations` | `13e50e2696082929738dc6595c6e6ab4aa314644` | 56 / 20 | archived locally; recovery tag retained |
| `codex/add-configurable-multihead-attention` | `69f36f17387ccdaaf4567d1a5ac3eb485f345297` | 11 / 1 | archived locally; recovery tag retained |
| `codex/add-simtoreal-tracker-analysis` | `5310280b199878890652a305b77bf6b416915112` | 24 / 11 | archived locally; recovery tag retained |
| `codex/add-topology-distance-plot` | `00423a20923f352a7d6969f686fe718cac587127` | 24 / 0 | archived locally; recovery tag retained |
| `codex/add-untrained-evaluation-configs` | `ee04d7985fd829abceb65ad275102da603da3b9f` | 35 / 1 | archived locally; recovery tag retained |
| `codex/add-warp-backend-to-gnnsac` | `970a6b5ec861819583e03e1fec5dfb632f951869` | 42 / 1 | archived locally; recovery tag retained |
| `codex/attention-default` | `dea87113958ddfac60cb82dd6293abccee202908` | 39 / 2 | archived locally; recovery tag retained |
| `codex/batched-mjx-rollouts` | `21af5f32e125d833c76fad2029d0b1fadf902a98` | 73 / 4 | archived locally; recovery tag retained |
| `codex/conflicting_gradients` | `c86e3a3572548d7e0d18bc07c9d49d29370a781c` | 52 / 3 | archived locally; recovery tag retained |
| `codex/cross-validation-launcher` | `49bb95a820ccef50fa82d5c00422d57456bcc2be` | 24 / 5 | archived locally; recovery tag retained |
| `codex/decouple-distillation-feature-schemas` | `95294bb8f3f456e5053ff3ac15e2cf79b4aebaf4` | 7 / 5 | archived locally; recovery tag retained |
| `codex/diagnose-hydra-slurm-error` | `b0765f0880c3a8b65d438feeee91ffba2caf9d99` | 14 / 1 | archived locally; recovery tag retained |
| `codex/diagnose-wb-simulation-stall` | `c8d8ae799c567b7aa17a1e69ebf8645f8b319152` | 7 / 2 | archived locally; recovery tag retained |
| `codex/distillation-throughput` | `6d23290edab11210483eb92cbb70e70a3f9cbad7` | 11 / 3 | archived locally; recovery tag retained |
| `codex/dynamics-generalization-experiments` | `6b618955f425ad234d141c1dcd63025f3126963e` | 1 / 3 | keep attached worktree branch |
| `codex/dynamics-vs-control-experiment3` | `6b618955f425ad234d141c1dcd63025f3126963e` | 1 / 3 | keep attached worktree branch |
| `codex/few-shot-adaptation` | `ed57cb4f6f47535a57ec28222222514aa96352da` | 1 / 0 | archived locally; recovery tag retained |
| `codex/fix-dynamic-broken-node-replay` | `2de0e9658c1abf0b40982c3958bea93121412860` | 11 / 9 | archived locally; recovery tag retained |
| `codex/fix-gnn-action-head-optimization` | `f444e6adb73f2e08eda0f59f83c01132154f02f1` | 23 / 2 | archived locally; recovery tag retained |
| `codex/fix-inconsistent-energy-penalty` | `224ea12922631abd6ea3c8ec08fc99df17c03814` | 18 / 1 | archived locally; recovery tag retained |
| `codex/fix-native-terminal-velocity-reward` | `f4eca4de2f51768e6739d4f3fe460d20c1b427d6` | 17 / 3 | archived locally; recovery tag retained |
| `codex/fix-reward-normalization-resume` | `fd0960f50cce768dd4cba900334fb9fee957f5ee` | 18 / 1 | archived locally; recovery tag retained |
| `codex/fix-vive-tracker-coordinate-map` | `800fa861f9c28194a967dbd10372a38fb3569904` | 24 / 4 | archived locally; recovery tag retained |
| `codex/gnn-dynamics-mpc-pilot` | `c1d9f141bacaaeebebce19b10b4463b339e6f85d` | 1 / 4 | keep attached worktree branch |
| `codex/graph-identity-features-pr` | `f0ed1c824673d696c3327deaa7c96c2c9df5875a` | 34 / 6 | archived locally; recovery tag retained |
| `codex/implement-tube-virtual-nodes` | `962c6cd8e48d907b382631b5a7ae17469f34d643` | 1 / 13 | integrated; retain source branch |
| `codex/jax` | `eb4c086e610d5fd9ce48e819574474b13161e420` | 79 / 0 | archived locally; recovery tag retained |
| `codex/mlp-paper-readiness` | `65f8750874a304803a6bb204677c30efb78e8d5c` | 1 / 1 | archived locally; recovery tag retained |
| `codex/padded_mlp` | `f11e088a318ee9c1a26f9d4ec0bb2e7a2fca12ce` | 24 / 1 | archived locally; recovery tag retained |
| `codex/paper-expert-baseline` | `3631552a6fb49f0bd48c2369ff538e26be437d89` | 0 / 18 | keep attached worktree branch |
| `codex/per-topology-distillation-decay` | `243e8b6ea5f169d3e8da910777eab2531c76be76` | 1 / 3 | keep attached worktree branch |
| `codex/pr10-conflict-resolution` | `2f98547400a5d5c8a883e925496971d8ad1df6a4` | 36 / 3 | archived locally; recovery tag retained |
| `codex/pr28-fix` | `d1575c180107c77bd1e34ba78f8177d4ca30ad94` | 11 / 14 | archived locally; recovery tag retained |
| `codex/pr9-p2` | `37b86610f637df87d4228520ad7cdbbd64a6d841` | 39 / 4 | archived locally; recovery tag retained |
| `codex/production-training-defaults` | `cc45ef82a0a663b8c4a9ffa4a2ac973e4a94aa00` | 27 / 0 | archived locally; recovery tag retained |
| `codex/propagate-distillation-wandb-broken-node` | `8ff71c91aa60a81581bc9964fb675071d3356d8f` | 11 / 13 | archived locally; recovery tag retained |
| `codex/propagate-distillation-wandb-sim2real` | `edfe00357a5cea2b42265f1081ae1716990b8a58` | 11 / 8 | archived locally; recovery tag retained |
| `codex/resolve-pr11-conflicts` | `5040562abdfd44282c0be52795000450982a4bd2` | 22 / 6 | archived locally; recovery tag retained |
| `codex/safety-diagnostics` | `f735d10f86d2d9137e46629791d04cbc0475a438` | 38 / 4 | archived locally; recovery tag retained |
| `codex/scale-multitopology-learning-budgets` | `6960e8d22102a6e0b29b3013fae7819a833b655e` | 11 / 1 | archived locally; recovery tag retained |
| `codex/sgs-topology-sampling` | `12ef6b966d7fab040d69b9a494aa455b9d549856` | 1 / 3 | keep attached worktree branch |
| `codex/shared-dynamics-mpc-experiments` | `3bea10554df7a25f4dbce503b77ae63dfa209840` | 1 / 1 | keep attached worktree branch |
| `codex/signed-routing-ablation` | `ec1e6feea4cc8fe7c05211ca830b2ebefa2099a6` | 1 / 6 | integrated; retain source branch |
| `codex/skip-completed-hydra-jobs` | `a3dfebe5480b9a90932084e8f1b0d6acc6da68b2` | 22 / 2 | archived locally; recovery tag retained |
| `codex/torchrl-replay-prototype` | `3496405227c2d535487d701874d436208cbe945e` | 11 / 9 | archived locally; recovery tag retained |
| `codex/tube-virtual-nodes` | `c734e08b736c4d00189d60d0ab487070c068f4d2` | 1 / 7 | integrated; retain source branch |
| `codex/update-mujocotrussgen-hinge-kp` | `51043af544081023f09fa444e7a499ef31142b9c` | 11 / 2 | archived locally; recovery tag retained |
| `codex/upgrade-mujocotrussgen-features` | `757562488daafd1ecd46da9001498e65d55605c5` | 12 / 1 | archived locally; recovery tag retained |
| `experiment/async_critic` | `82f6aa49c13e308d511170c5021de3d1dfd668e5` | 40 / 3 | archived locally; recovery tag retained |
| `experiments/1-sample` | `c973596709e92b03dcba54ab582a7a4be23663af` | 38 / 4 | archived locally; recovery tag retained |
| `experiments/attention` | `7a3738e044ec66c6689a824602b01fdbbb5ed9aa` | 38 / 2 | archived locally; recovery tag retained |
| `experiments/critic_vnode` | `db87abb196f7e746d7065f7c93841ebb51999d49` | 39 / 3 | archived locally; recovery tag retained |
| `experiments/distillation` | `4b8b8412622bcf065b753163ea5244fd257c4434` | 11 / 6 | archived locally; recovery tag retained |
| `experiments/diversity` | `d3d8e5668c71d686ff46a93317e4754a28b5d322` | 1 / 1 | integrated; retain source branch |
| `experiments/graph-identity-features` | `36267f09db8118607fb90dac4b3bc93004769749` | 38 / 5 | archived locally; recovery tag retained |
| `experiments/lagrangian-rigidity` | `91731e09fbcc3f26326c5ca0b6712ef5ce33508a` | 38 / 2 | archived locally; recovery tag retained |
| `experiments/multi_message_passing` | `36f630bb8ab20dec7c112a01dd9223d9bcc1fdf3` | 55 / 2 | archived locally; recovery tag retained |
| `experiments/random-cross-validation` | `7d7cb0ec5a051a307d696d98767501003bbf9109` | 11 / 14 | archived locally; recovery tag retained |
| `experiments/reward_normalization` | `3cee6e48bc0ccfab9a6c26bb404e5b9de6405893` | 50 / 6 | archived locally; recovery tag retained |
| `experiments/safety_logging` | `f735d10f86d2d9137e46629791d04cbc0475a438` | 38 / 4 | archived locally; recovery tag retained |
| `experiments/safety_surrogates` | `fe7d2c776b95c61df8b5236f673aa6360eadd505` | 38 / 4 | archived locally; recovery tag retained |
| `experiments/virtual_node` | `3bf3f105df44e26263be6d887cc7243c0e0a69ba` | 51 / 5 | archived locally; recovery tag retained |
| `main` | `ed57cb4f6f47535a57ec28222222514aa96352da` | 1 / 0 | keep main |
| `optimization/pcgrad` | `36fa74b124e8bde2367a7f414beb6082a4bb812b` | 14 / 8 | archived locally; recovery tag retained |
| `optimization/timing` | `36fa74b124e8bde2367a7f414beb6082a4bb812b` | 14 / 8 | archived locally; recovery tag retained |
| `reward_function` | `c043a0adf0d2a7e2fb447fda35367a5ee3620f04` | 39 / 4 | archived locally; recovery tag retained |
| `sim-to-real` | `939bb0295c05b674efe246b3f6f2b6a53d7b52ab` | 11 / 11 | leave unchanged (user) |
| `sim2real/hinge_randomizaton` | `3b1e8e6faf92a85e24663fb1c726b601f4e78b0e` | 11 / 7 | archived locally; recovery tag retained |

## Recovery-tag map

Each row preserves the exact tip listed above; both local and remote tips are saved separately.

| Original ref | Local annotated recovery tag | Status |
|---|---|---|
| `refs/heads/broken-node-interleaved-regime` | `archive/paper-prep-2026-10-28/local/broken-node-interleaved-regime` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/broken-node-rl-training-plan-4f6b6b` | `archive/paper-prep-2026-10-28/local/claude/broken-node-rl-training-plan-4f6b6b` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/cross-validation-distillation-analysis-92645a` | `archive/paper-prep-2026-10-28/local/claude/cross-validation-distillation-analysis-92645a` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/cv-training-generalization-analysis-deefcf` | `archive/paper-prep-2026-10-28/local/claude/cv-training-generalization-analysis-deefcf` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/discount-factor-default-8f3900` | `archive/paper-prep-2026-10-28/local/claude/discount-factor-default-8f3900` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/distillation-pr-chain-018592` | `archive/paper-prep-2026-10-28/local/claude/distillation-pr-chain-018592` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/distribution-plots-robot-topology-d95aa0` | `archive/paper-prep-2026-10-28/local/claude/distribution-plots-robot-topology-d95aa0` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/documentation-cleanup-proposal-2cbeaa` | `archive/paper-prep-2026-10-28/local/claude/documentation-cleanup-proposal-2cbeaa` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/gpu-utilization-training-ff5b72` | `archive/paper-prep-2026-10-28/local/claude/gpu-utilization-training-ff5b72` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/pensive-cartwright-51ab63` | `archive/paper-prep-2026-10-28/local/claude/pensive-cartwright-51ab63` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/pr-35-comment-response-7a92fd` | `archive/paper-prep-2026-10-28/local/claude/pr-35-comment-response-7a92fd` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/random-cross-validation-a896d0` | `archive/paper-prep-2026-10-28/local/claude/random-cross-validation-a896d0` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/robot-topology-distribution-9e3b7a` | `archive/paper-prep-2026-10-28/local/claude/robot-topology-distribution-9e3b7a` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/robot-topology-distribution-c6d6f0` | `archive/paper-prep-2026-10-28/local/claude/robot-topology-distribution-c6d6f0` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/robot-topology-generalization-eec5e0` | `archive/paper-prep-2026-10-28/local/claude/robot-topology-generalization-eec5e0` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/robot-topology-selection-4670c6` | `archive/paper-prep-2026-10-28/local/claude/robot-topology-selection-4670c6` | archived locally; public recovery-tag publication pending |
| `refs/heads/claude/wandb-mcp-server-setup-c6bba4` | `archive/paper-prep-2026-10-28/local/claude/wandb-mcp-server-setup-c6bba4` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/GPU_optimizations` | `archive/paper-prep-2026-10-28/local/codex/GPU_optimizations` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/add-configurable-multihead-attention` | `archive/paper-prep-2026-10-28/local/codex/add-configurable-multihead-attention` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/add-simtoreal-tracker-analysis` | `archive/paper-prep-2026-10-28/local/codex/add-simtoreal-tracker-analysis` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/add-topology-distance-plot` | `archive/paper-prep-2026-10-28/local/codex/add-topology-distance-plot` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/add-untrained-evaluation-configs` | `archive/paper-prep-2026-10-28/local/codex/add-untrained-evaluation-configs` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/add-warp-backend-to-gnnsac` | `archive/paper-prep-2026-10-28/local/codex/add-warp-backend-to-gnnsac` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/attention-default` | `archive/paper-prep-2026-10-28/local/codex/attention-default` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/batched-mjx-rollouts` | `archive/paper-prep-2026-10-28/local/codex/batched-mjx-rollouts` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/conflicting_gradients` | `archive/paper-prep-2026-10-28/local/codex/conflicting_gradients` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/cross-validation-launcher` | `archive/paper-prep-2026-10-28/local/codex/cross-validation-launcher` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/decouple-distillation-feature-schemas` | `archive/paper-prep-2026-10-28/local/codex/decouple-distillation-feature-schemas` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/diagnose-hydra-slurm-error` | `archive/paper-prep-2026-10-28/local/codex/diagnose-hydra-slurm-error` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/diagnose-wb-simulation-stall` | `archive/paper-prep-2026-10-28/local/codex/diagnose-wb-simulation-stall` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/distillation-throughput` | `archive/paper-prep-2026-10-28/local/codex/distillation-throughput` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/few-shot-adaptation` | `archive/paper-prep-2026-10-28/local/codex/few-shot-adaptation` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-dynamic-broken-node-replay` | `archive/paper-prep-2026-10-28/local/codex/fix-dynamic-broken-node-replay` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-gnn-action-head-optimization` | `archive/paper-prep-2026-10-28/local/codex/fix-gnn-action-head-optimization` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-inconsistent-energy-penalty` | `archive/paper-prep-2026-10-28/local/codex/fix-inconsistent-energy-penalty` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-native-terminal-velocity-reward` | `archive/paper-prep-2026-10-28/local/codex/fix-native-terminal-velocity-reward` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-reward-normalization-resume` | `archive/paper-prep-2026-10-28/local/codex/fix-reward-normalization-resume` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/fix-vive-tracker-coordinate-map` | `archive/paper-prep-2026-10-28/local/codex/fix-vive-tracker-coordinate-map` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/graph-identity-features-pr` | `archive/paper-prep-2026-10-28/local/codex/graph-identity-features-pr` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/jax` | `archive/paper-prep-2026-10-28/local/codex/jax` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/mlp-paper-readiness` | `archive/paper-prep-2026-10-28/local/codex/mlp-paper-readiness` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/padded_mlp` | `archive/paper-prep-2026-10-28/local/codex/padded_mlp` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/pr10-conflict-resolution` | `archive/paper-prep-2026-10-28/local/codex/pr10-conflict-resolution` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/pr28-fix` | `archive/paper-prep-2026-10-28/local/codex/pr28-fix` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/pr9-p2` | `archive/paper-prep-2026-10-28/local/codex/pr9-p2` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/production-training-defaults` | `archive/paper-prep-2026-10-28/local/codex/production-training-defaults` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/propagate-distillation-wandb-broken-node` | `archive/paper-prep-2026-10-28/local/codex/propagate-distillation-wandb-broken-node` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/propagate-distillation-wandb-sim2real` | `archive/paper-prep-2026-10-28/local/codex/propagate-distillation-wandb-sim2real` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/resolve-pr11-conflicts` | `archive/paper-prep-2026-10-28/local/codex/resolve-pr11-conflicts` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/safety-diagnostics` | `archive/paper-prep-2026-10-28/local/codex/safety-diagnostics` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/scale-multitopology-learning-budgets` | `archive/paper-prep-2026-10-28/local/codex/scale-multitopology-learning-budgets` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/skip-completed-hydra-jobs` | `archive/paper-prep-2026-10-28/local/codex/skip-completed-hydra-jobs` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/torchrl-replay-prototype` | `archive/paper-prep-2026-10-28/local/codex/torchrl-replay-prototype` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/update-mujocotrussgen-hinge-kp` | `archive/paper-prep-2026-10-28/local/codex/update-mujocotrussgen-hinge-kp` | archived locally; public recovery-tag publication pending |
| `refs/heads/codex/upgrade-mujocotrussgen-features` | `archive/paper-prep-2026-10-28/local/codex/upgrade-mujocotrussgen-features` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiment/async_critic` | `archive/paper-prep-2026-10-28/local/experiment/async_critic` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/1-sample` | `archive/paper-prep-2026-10-28/local/experiments/1-sample` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/attention` | `archive/paper-prep-2026-10-28/local/experiments/attention` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/critic_vnode` | `archive/paper-prep-2026-10-28/local/experiments/critic_vnode` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/distillation` | `archive/paper-prep-2026-10-28/local/experiments/distillation` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/graph-identity-features` | `archive/paper-prep-2026-10-28/local/experiments/graph-identity-features` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/lagrangian-rigidity` | `archive/paper-prep-2026-10-28/local/experiments/lagrangian-rigidity` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/multi_message_passing` | `archive/paper-prep-2026-10-28/local/experiments/multi_message_passing` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/random-cross-validation` | `archive/paper-prep-2026-10-28/local/experiments/random-cross-validation` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/reward_normalization` | `archive/paper-prep-2026-10-28/local/experiments/reward_normalization` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/safety_logging` | `archive/paper-prep-2026-10-28/local/experiments/safety_logging` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/safety_surrogates` | `archive/paper-prep-2026-10-28/local/experiments/safety_surrogates` | archived locally; public recovery-tag publication pending |
| `refs/heads/experiments/virtual_node` | `archive/paper-prep-2026-10-28/local/experiments/virtual_node` | archived locally; public recovery-tag publication pending |
| `refs/heads/optimization/pcgrad` | `archive/paper-prep-2026-10-28/local/optimization/pcgrad` | archived locally; public recovery-tag publication pending |
| `refs/heads/optimization/timing` | `archive/paper-prep-2026-10-28/local/optimization/timing` | archived locally; public recovery-tag publication pending |
| `refs/heads/reward_function` | `archive/paper-prep-2026-10-28/local/reward_function` | archived locally; public recovery-tag publication pending |
| `refs/heads/sim2real/hinge_randomizaton` | `archive/paper-prep-2026-10-28/local/sim2real/hinge_randomizaton` | archived locally; public recovery-tag publication pending |
| `refs/remotes/origin/claude/robot-topology-distribution-9e3b7a` | `archive/paper-prep-2026-10-28/remote/claude/robot-topology-distribution-9e3b7a` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/codex/add-configurable-multihead-attention` | `archive/paper-prep-2026-10-28/remote/codex/add-configurable-multihead-attention` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/codex/batched-mjx-rollouts` | `archive/paper-prep-2026-10-28/remote/codex/batched-mjx-rollouts` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/experiments/1-sample` | `archive/paper-prep-2026-10-28/remote/experiments/1-sample` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/experiments/graph-identity-features` | `archive/paper-prep-2026-10-28/remote/experiments/graph-identity-features` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/experiments/lagrangian-rigidity` | `archive/paper-prep-2026-10-28/remote/experiments/lagrangian-rigidity` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/experiments/safety_logging` | `archive/paper-prep-2026-10-28/remote/experiments/safety_logging` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/experiments/safety_surrogates` | `archive/paper-prep-2026-10-28/remote/experiments/safety_surrogates` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/optimization/timing` | `archive/paper-prep-2026-10-28/remote/optimization/timing` | not removed; awaiting public recovery-tag publication approval |
| `refs/remotes/origin/reward_function` | `archive/paper-prep-2026-10-28/remote/reward_function` | not removed; awaiting public recovery-tag publication approval |
