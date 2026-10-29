# Paper branches, versions and run records

The expert comparison uses a separate GNN or padded-MLP policy for each declared
morphology. The shared-controller tools are available for later studies;
signed routing, tube nodes and tube physics remain disabled by default.

## Repository references

| Reference | Purpose |
|---|---|
| `main` | Integrated project development. |
| `codex/paper-expert-comparison` | Ongoing paper configurations, evaluation and reproduction work. This branch can advance. |
| `paper-experts-v1` | Annotated, fixed code-baseline tag. Never move or overwrite it. |

The first tag records the integrated, locally tested code baseline: 424 tests
passed and two CUDA-only tests were skipped. It does not assert GPU qualification,
choose the final morphology set or establish any paper result. The run protocol
and GPU pilots in [the baseline plan](../plans/paper_expert_baseline.md) are still
required before production. If subsequent fixes are needed, create a new version
such as `paper-experts-v2`; keep the original tag and existing run records.

The paper branch and annotated tag are published on the project
[GitHub repository](https://github.com/isaa-sudweeks/GNN-SAC).
`paper-experts-v1` remains fixed at
`a1519b5f1c8439f6882ea1f18ee1e7b187db8256`; later documentation and experiment
work can advance the paper branch. Integration into `main` is tracked by the
preparation PR containing this documentation. All 82 cleanup recovery tags are
also published and verified; see the [branch inventory](../plans/paper_branch_inventory.md).

## Before each production run

1. Start from an approved tag or a recorded commit on the paper branch. Require
   a clean working tree, including untracked code/configuration files; do not
   report only a branch name or tag when the checkout has local changes.
2. Save the exact commit SHA, dependency lock hash and complete launch command.
3. Save the Hydra configuration and overrides, plus the resolved runtime config
   stored in the full trainer checkpoint. Keep topology IDs, training/evaluation
   seeds, requested/effective budgets, simulator/physics settings, model sizes,
   feature flags and checkpoint-selection rules with the run.
4. Record the W&B run ID, full checkpoint path, exported-policy path and the
   evaluation data used by each figure/table. Keep generated artifacts outside
   Git; preserve their locations and checksums in the campaign's records.

For example, before launch, save the source identity alongside the run:

```bash
paper_run_dir=/absolute/path/to/run
mkdir -p "$paper_run_dir"
git rev-parse HEAD > "$paper_run_dir/git_commit.txt"
git describe --tags --exact-match HEAD > "$paper_run_dir/git_tag.txt"
git status --porcelain > "$paper_run_dir/git_status.txt"
shasum -a 256 uv.lock > "$paper_run_dir/uv_lock.sha256"
```

`git describe --exact-match` succeeds only on a tagged commit. A development
commit can still be recorded by SHA; do not claim it is the tagged baseline.
Check that `git_status.txt` is empty before launch. Also save Python/package
versions, GPU/driver details and the exact launch command; a Git tag does not
capture those or replace the experiment protocol.

## Corrections and submission

Keep paper changes on the paper branch and merge generally useful changes into
`main` through reviewed PRs. Each run retains its original code/configuration
identity even as either branch advances. If a correction changes the experiment,
record the affected runs and rerun them under a new version rather than changing
their labels to point at new code.

At submission, create a GitHub release from the tag used for the reported
results, with reproduction commands and links to configuration, data and model
artifacts. Optionally archive that release with Zenodo and cite its
version-specific DOI. A submission release and DOI have not been created yet.
