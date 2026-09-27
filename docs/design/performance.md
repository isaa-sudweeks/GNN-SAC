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
- **Tensor replay** (`replay_backend=torchrl_tensor`, the default). Static
  graph data is stored once per topology, placement across CPU and GPU is
  bounded, and sampling was 33x faster in matched MJX training. Full-checkpoint
  writes went from about 7 s to about 37 ms. See
  [replay_benchmark_results.md](replay_benchmark_results.md).
- **Tensor-batched MJX collection** (`vectorized_collection=auto`). Actions,
  environment steps, reward normalization, episode statistics, and replay
  insertion run as whole-bucket tensor operations instead of Python work per
  environment. On an L40S with the CV setup (8 topologies, `num_envs=1536`,
  PCGrad), a vector step went from 6.06 s to 3.90 s (254 to 394 env steps/s):
  replay insertion, transition processing, and action selection fell from 1.89 s
  to 0.02 s, and the environment step from 0.85 s to 0.64 s.
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
4. **Checkpoint size.** Each checkpoint still serializes the full replay
   buffer, and then writes the same data a second time to `latest.pt`. Tensor
   replay made this fast, but the files remain large (about 4 GB for a
   nine-topology run at 70% occupancy).
5. **Evaluation cost.** Evaluation is synchronous (`eval_freq=20_000`,
   `eval_episodes=5` by default). Asynchronous or post-hoc evaluation would
   remove it from the training critical path.
6. **Learner-side GPU work.** After tensor-batched collection, optimization is
   about 79% of an MJX vector step in the CV setup (0.41 s per PCGrad update,
   7.5 updates per step) while the GPU stays mostly idle, so the learner is
   launch-bound rather than compute-bound. Candidates are fewer per-task
   passes in PCGrad, fused optimizers, `torch.compile`, and CUDA graphs;
   dynamic PyG shapes complicate compilation.
7. **Multiple GPUs.** Until a single GPU is saturated, use extra GPUs for
   independent seeds, folds, or sweeps rather than a distributed learner.

## Measuring

Use `profiling.enabled=true` for synchronized phase timings within training
(see [training.md](../usage/training.md#profiling)). The scripts in
[scripts.md](../usage/scripts.md#benchmarks) benchmark individual components.
Base decisions on profiles from the target GPU nodes, not local CPU numbers.
