# Performance

This page summarizes the current performance state and the work still open. It
replaces the July 2026 audit, which is kept for its measurements and reasoning
in [archive/gpu_optimization_audit.md](../archive/gpu_optimization_audit.md).

## Implemented

- **Batch-native MJX training** (`sim_backend=mjx`). There is one compiled
  environment batch per topology. JAX and PyTorch exchange data through DLPack
  without going through the host, and evaluation runs separately on native
  MuJoCo.
- **Warp MJX physics** (`mjx_impl=warp`). `warp_staged` was the fastest mode in
  the A100 benchmark (`scripts/benchmark_mjx_implementations.py`).
- **Batched actor inference.** Each vector step makes one actor forward pass over
  a mixed-topology PyG `Batch`, and actions move to the CPU once per batch.
- **Replay-ratio update scheduling** (`replay_ratio`, `update_every_vector_steps`).
  This replaces one optimizer update per transition.
- **Per-step replay insertion.** Transitions enter replay after every vector
  step instead of at episode end.
- **Direct-collation replay sampling** (`scripts/benchmark_gnn_replay.py`).
- **Checkpoints.** Writes are asynchronous and atomic (temporary file then
  rename), and a small `latest.metadata.json` sidecar lets the Slurm launcher
  skip completed jobs without loading replay.

## Open items

These are ordered roughly by expected impact.

1. **`nsubsteps=100`.** Every environment step runs 100 physics substeps. Lower
   values would speed up simulation roughly in proportion, but they change the
   control interval, so they must be validated for both dynamics and learning
   quality.
2. **Native realistic-model throughput.** In the July audit
   (`mujoco-truss-gen` 0.9.0), the Python angle-bisector controller took about
   95% of realistic-model step time on the native backend. This has not been
   re-profiled against 0.12.5.
3. **Native multi-environment collection.** `env/wrappers/repeated.py` uses a
   `ThreadPoolExecutor`, which scaled poorly in local measurements. Candidates
   are process-based actors with shared-memory transfer, or MJX wherever it
   supports the model.
4. **Replay storage.** Each topology's buffer holds Python lists of PyG objects
   on the CPU. Current and next graphs are stored separately, and the host
   memory is not pinned. Contiguous tensor storage with static `edge_index`
   stored once per topology would reduce memory use and sampling time.
5. **Checkpoint size.** Each checkpoint serializes the full replay buffer, and
   then writes the same data a second time to `latest.pt`. Saving weights frequently and replay
   snapshots rarely (or in segments) would reduce pauses and filesystem traffic.
6. **Evaluation cost.** Evaluation is synchronous (`eval_freq=20_000`,
   `eval_episodes=5` by default). Asynchronous or post-hoc evaluation would
   remove it from the training critical path.
7. **Learner-side GPU work.** Candidates are mixed precision, fused optimizers,
   `torch.compile`, and CUDA graphs. Only pursue these once profiling shows the
   learner is the bottleneck; dynamic PyG shapes complicate compilation.
8. **Multiple GPUs.** Until a single GPU is saturated, use extra GPUs for
   independent seeds, folds, or sweeps rather than a distributed learner.

## Measuring

Use `profiling.enabled=true` for synchronized phase timings within training
(see [training.md](../usage/training.md#profiling)). The scripts in
[scripts.md](../usage/scripts.md#benchmarks) benchmark individual components.
Base decisions on profiles from the target GPU nodes, not local CPU numbers.
