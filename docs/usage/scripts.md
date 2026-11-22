# Scripts

Most scripts accept `uv run python <path> --help`. The exceptions are marked
**No CLI** below: they take no arguments and start running as soon as they are
invoked, even if you pass `--help`.

## Experiment launch and validation

| Script | Purpose |
|---|---|
| `scripts/launch_cross_validation.py` | Launch every leave-one-group-out fold and seed as a Hydra multirun. See [cross_validation.md](cross_validation.md). |
| `scripts/launch_random_teacher_cv.py` | Launch random five-fold GNN CV with the paper-v4 teachers; defaults to three seeds on the supercomputer and supports `--dry-run`. |
| `scripts/make_random_cross_validation.py` | Generate a random-fold cross-validation definition (for example `random_5fold`) from an existing one. |
| `scripts/make_farthest_point_cross_validation.py` | Generate frozen morphology-cluster folds using farthest-point prototypes and nearest-prototype assignment. |
| `scripts/validate_padded_mlp_topologies.py` | **No CLI.** Check that the fixed padded-MLP capacity covers the thesis topology set. Rerun after upgrading `mujoco-truss-gen`. |
| `scripts/validate_domain_randomization.py` | Run the domain-randomization validation plan and write machine-readable results. |
| `scripts/run_domain_randomization_training_smoke.py` | **No CLI.** Immediately runs the full three-seed training smoke matrix (every randomization family) and writes its outputs. |
| `scripts/convert_gnn_replay_checkpoint.py` | Convert an object-based (legacy) replay checkpoint to tensor replay without modifying the original. See [replay.md](replay.md). |
| `scripts/validate_production_replay.py` | Check replay performance and learner equivalence using real checkpoints. |

## Benchmarks

| Script | Measures |
|---|---|
| `scripts/benchmark_mujoco_backends.py` | Native MuJoCo vs MJX step throughput for local truss environments |
| `scripts/benchmark_multi_env_runs.py` | Native repeated environments vs batch-native MJX, across `--num-envs` |
| `scripts/benchmark_mjx_implementations.py` | JAX vs Warp MJX physics through the GNN-SAC MJX adapter |
| `scripts/benchmark_actor_inference.py` | Serialized vs batched GNN actor inference |
| `scripts/benchmark_gnn_replay.py` | Legacy vs tensor GNN replay: sampling, insertion, checkpoint cost (`--prototype`) |
| `scripts/benchmark_pcgrad.py` | Current PCGrad projection vs the frozen legacy implementation |
| `scripts/probe_pcgrad_cuda_graph.py` | Whether one production-sized PCGrad task can be captured in a CUDA Graph |

## Cluster batch files

`scripts/orc_benchmark_torchrl_replay.sbatch` and
`scripts/orc_validate_production_replay.sbatch` are the Slurm jobs used to
produce [replay_benchmark_results.md](../design/replay_benchmark_results.md).
They read their inputs from environment variables (`REPLAY_*`).

## Figures

`figures/plot_distance_by_topology.py` downloads paper evaluation results from
W&B (`--project`, default `i-suds/Paper-Experts`). By default it combines the
33 `paper-v4-` GNN runs with six `paper-expert-gnn-` runs, plus the 39
`paper-expert-mlp-` runs. The superseded `paper-v2-` run has no evaluation
metric and is excluded. All 13 topologies must have seeds 1, 2, and 3 for
both expert architectures. `--run-prefix` retains legacy single-family mode.

Within each architecture and topology, select the **same environment step**
for all seeds: the step with the largest across-seed mean evaluation
`eval/episode_distance` (ties choose the latest step). Each logged value is
an evaluation mean over five episodes. Error bars are across-seed **sample
standard deviations**, not confidence intervals. This is a retrospective
peak-of-evaluation comparison, not an independent test of the selected
checkpoint. Specialists train on their own topology, including topologies
held out from shared-policy training; these are not zero-shot results.

Interrupted runs retain their available evaluations so that a seed is not
silently dropped; selection is limited to steps shared by all three seeds.
The November 4 snapshot includes two failed MLP seed-2 runs (tetrahedron and
7-node/1-tube) with final evaluations at 9,801,216 steps. Their topology labels
carry a dagger. Other runs reach 10M evaluation steps. The measured greedy
baseline remains in `figures/greedy_metrics.csv` with its original per-topology
standard deviations.

Outputs are a 7.16-inch vector PDF with embedded fonts, a 600-dpi PNG, an
interactive HTML companion, and seed/summary CSVs. Solid markers show means;
small open markers show individual seeds. Shapes distinguish controllers in
grayscale. Friendly labels map explicitly to canonical IDs in `LABELS`.

Reproduce offline using the source-preserving compressed history snapshot:

```bash
uv run python figures/plot_distance_by_topology.py \
  --history-json figures/paper_experts_history.json.gz
```

The snapshot records source run URLs, states, selected configuration fields,
retrieval time, and all 38,860 evaluation rows (SDK history with a 10,000-row
request per run, above every returned history length). It contains no API
credentials. Omit `--history-json` to refresh directly from W&B. Generated
figure files and CSVs are kept beside the script; the RA-L manuscript uses a
copy of the PDF in its own `figures/` folder.

- `launch_tube_ablation.py`: manifest/dry run or ORC submission for matched signed,
  membership and optional physics arms, with an offline screen and online stage.
  `--arms signed membership physics` prepares shared clean frozen-teacher replay
  on CPU before submitting students. See [tube virtual nodes](tube_virtual_nodes.md).
- `prepare_tube_replay.py`: CPU Slurm worker for deterministic, noise-free replay
  collection and provenance-checked cache reuse.
- `run_routing_canary.py --tube-nodes`: legacy-teacher tube student canary; add
  `--cpu` for a native local smoke run.

- `benchmark_tube_virtual_nodes.py`: signed/membership actor inference timing and
  prepared tensor storage, with explicit batch size and device.
