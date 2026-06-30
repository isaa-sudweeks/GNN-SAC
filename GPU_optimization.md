# GNN-SAC Performance Audit and GPU Optimization Plan

## Scope

This audit targets multi-million-timestep GNN-SAC training runs on research-cluster hardware with one to eight NVIDIA A100 or H200 GPUs. It separates bottlenecks in this repository from bottlenecks in `mujoco-truss-gen` so library changes can be handled independently.

The current GNN-SAC training path is primarily CPU/environment-bound rather than GPU-bound. The `mujoco-truss-gen` `codex/GPU_optimization` branch now provides a batched MJX environment for fixed abstract models, but this repository does not yet use it. Adding GPUs to the current entry points will therefore provide limited benefit until that integration, replay storage, and checkpointing are addressed.

Checklist items are marked complete only when the relevant repository branch contains an implementation. A checked item does not imply that the broader numbered bottleneck is fully resolved.

## Repository Bottlenecks

The following issues are ranked by expected wall-clock impact.

### 1. Evaluation can perform five times more simulation than training

The previous default configuration evaluated 10 episodes of up to 5,000 steps every 10,000 training steps. This could execute 50,000 evaluation steps per 10,000 training steps, in addition to an evaluation at step zero. If episodes regularly reached the time limit, total simulator work could be approximately six times the requested training workload.

Relevant code and configuration:

- `config/gnn_config.yaml`: optimized defaults are `eval_freq: 100_000` and `eval_episodes: 3`
- `config/environment.yaml`: `max_steps: 5_000`
- `sac/trainer/online_trainer.py`: synchronous evaluation loop

Recommended changes:

- [x] Use one to three evaluation episodes during training. The default is now three.
- [x] Evaluate every 100,000 to 250,000 training steps. The default is now 100,000.
- [ ] Run full evaluation asynchronously or after training.
- [ ] Avoid the initial step-zero evaluation unless it is explicitly required as a baseline.

### 2. `nsubsteps: 100` multiplies simulator and controller costs

Every environment action performs 100 MuJoCo integration steps. This multiplies native physics work and, for realistic trusses, Python controller work by 100.

Reducing `nsubsteps` from 100 to 20 would provide an approximately fivefold improvement in environment-side work. This changes the control interval and possibly the physical behavior, so the lowest acceptable value must be selected through a dynamics and learning-quality validation rather than as a purely mechanical optimization.

### 3. Policy inference is serialized and synchronized per environment

The multi-environment trainer loops over environments and calls `agent.act()` once per observation. Each call:

1. Creates a one-graph PyTorch Geometric batch.
2. Transfers the observation to the GPU.
3. Launches the actor network.
4. Copies the action to the CPU, forcing synchronization.

This prevents effective GPU utilization as environment count increases. A local CPU measurement demonstrated the batching opportunity:

| Actor batch size | Approximate graph inference throughput |
|---:|---:|
| 1 | 1,200 graphs/s |
| 8 | 4,200 graphs/s |
| 64 | 12,700 graphs/s |
| 256 | 18,200 graphs/s |

Recommended changes:

- [x] Construct one PyG `Batch` from all active environment observations.
- [x] Perform one actor forward pass per vector step.
- [x] Split the resulting node actions by graph afterward.
- [x] Avoid unnecessary stochastic sampling and log-probability calculation during deterministic evaluation.
- [x] Transfer actions to CPU once per batch rather than once per environment.

### 4. Environment vectorization uses Python threads

`RepeatedEnvWrapper` uses `ThreadPoolExecutor`. This is ineffective for Python-heavy environment work and showed poor scaling in local measurements:

| Environments | Simple truss aggregate | Realistic truss aggregate |
|---:|---:|---:|
| 1 | 910 steps/s | 19.5 steps/s |
| 2 | 1,343 steps/s | 14.3 steps/s |
| 4 | 1,231 steps/s | 10.5 steps/s |
| 8 | 854 steps/s | 10.3 steps/s |

These measurements are from local CPU hardware and should not be treated as cluster throughput predictions. They do establish that the current threading design does not scale consistently.

For fixed abstract models, the new upstream `MjxNodeVelocityEnv` provides a better alternative to CPU workers: batch hundreds or thousands of environment states directly on the accelerator. Process-based actors remain the fallback for realistic models and the legacy Gymnasium path until those models are supported by MJX. Neither path is integrated into the current trainer yet.

Recommended changes:

- [ ] Integrate the batched MJX environment for compatible abstract-model training.
- [ ] Use process-based actors instead of threads for the CPU/realistic fallback.
- [ ] Assign CPU affinity to CPU actor processes.
- [ ] Use shared-memory queues or ring buffers rather than pickling graph objects between processes.
- [ ] Separate environment collection from learner execution.

### 5. Replay storage is object-heavy, duplicated, and rebuilt every sample

The graph replay buffer maintains five Python lists with up to one million entries. It stores separate PyG objects for current and next observations, duplicating most graph state. Every update selects entries with Python list operations and constructs two new PyG batches on the CPU.

The use of `non_blocking=True` does not provide asynchronous host-to-device transfer because the stored CPU tensors are not pinned.

Local measurements with an octahedron graph and batch size 256 found:

- Approximately 4.5 ms per replay sample/batch construction.
- A checkpoint containing 2,000 populated transitions in a one-million-capacity buffer occupied 10.6 MiB.
- A full one-million-transition replay payload is projected to occupy roughly 3 GiB, depending on topology and serialization overhead.

Recommended changes:

- Store observations, actions, rewards, and termination flags in contiguous tensors.
- Store static `edge_index` data once per topology rather than once per transition.
- Use topology-specific tensor slabs for variable-size multi-topology training.
- Use pinned host memory or GPU-resident replay where capacity permits.
- Represent next state through contiguous transition indexing when possible instead of duplicating every graph.

### 6. Multi-environment collection creates one sequential update per transition

For `N` active environments, the trainer performs `iterations * N` sequential SAC updates after each vector step. Faster collection therefore causes the learner loop to become the next bottleneck without amortizing optimizer work.

Recommended changes:

- Make update-to-data ratio an explicit experiment parameter.
- Decouple actor collection and learner updates.
- Benchmark larger replay batches and fewer updates per collected transition.
- Preserve learning-quality comparisons because reducing update-to-data ratio changes the algorithm's sample efficiency.
- Use CUDA graphs, compilation, and fused optimizers only after input shapes and replay delivery have been stabilized.

### 7. Transitions enter replay only when an episode ends

The trainer retains up to 5,000 graph observations per environment in episode-local Python lists. At termination, it clones the entire episode into replay using a Python loop. This delays availability of fresh experience, increases peak memory use, and produces periodic insertion stalls.

Recommended change:

- Insert `(obs, action, reward, next_obs, terminated)` directly into replay after every environment step.

### 8. Full replay checkpoints are written twice and block training

Each checkpoint serializes the complete agent, optimizer, replay buffer, RNG state, and logger state. It writes the same state once as `step_<N>.pt` and again as `latest.pt`. The cluster configuration performs this every 10,000 steps, which will cause substantial pauses and shared-filesystem traffic as replay grows.

Checkpoint resume can also load the full checkpoint once to recover W&B metadata and again to restore the trainer.

Recommended changes:

- Save lightweight model and optimizer checkpoints frequently.
- Save replay snapshots much less frequently.
- Store replay in append-only or segmented files so unchanged data is not repeatedly serialized.
- Update `latest` through an atomic link or manifest rather than duplicating the checkpoint.
- Perform replay persistence asynchronously where cluster storage permits it.
- Keep W&B identity in a small metadata file that never requires loading replay.

### 9. Multi-GPU training is not implemented

The repository currently has no DistributedDataParallel path, learner replicas, distributed replay, mixed-precision training path, or distributed actor architecture. The GNN actor-critic has approximately one million learnable parameters, so naively splitting the current batch size of 256 over eight H200 GPUs would probably be slower than using a single GPU because communication and launch overhead would dominate.

Recommended approach:

- First saturate one GPU with batched inference, tensorized replay, and asynchronous actors.
- Use additional GPUs for independent seeds, topology splits, or hyperparameter experiments.
- Add multi-GPU learner replication only if profiling shows a single learner GPU is saturated.
- If multi-GPU learning is needed, use a substantially larger global batch and sharded replay with gradient all-reduce.

### 10. Secondary GPU-side inefficiencies

These are lower priority than the data pipeline:

- The two critic networks are executed through a Python list and stacked afterward.
- Adam is configured as capturable but is not using a CUDA graph or fused optimizer.
- Target-network updates iterate through parameters in Python.
- There is no BF16/FP16 autocast path.
- There is no active `torch.compile` path.
- Dynamic PyG batch shapes may create compilation or CUDA graph challenges across topologies.

These optimizations should be addressed after environment throughput and replay delivery no longer starve the GPU.

## `mujoco-truss-gen` Bottlenecks

The original measurements apply to installed `mujoco-truss-gen` version 0.9.0. Implementation status was updated against [`codex/GPU_optimization` at commit `552455b`](https://github.com/isaa-sudweeks/mujoco-truss-gen/tree/codex/GPU_optimization), based on version 0.10.2. The branch adds canonical abstract-model conversion tests and `MjxNodeVelocityEnv`; it does not change the older CPU profile numbers below.

### 1. The realistic angle-bisector controller dominates environment execution

For realistic trusses, `AngleBisectorController.update()` executes Python/NumPy geometry for every controller target on every physics substep.

In a local profile using 100 substeps:

- Angle-bisector controller work: approximately 1.014 seconds.
- Native `mujoco.mj_step` work: approximately 0.052 seconds.
- The controller consumed about 95% of `_advance()` time.
- Profiled realistic throughput was approximately 9 to 20 environment steps/s, compared with roughly 500 to 900 steps/s for the simple octahedron.

Recommended library changes, in preferred order:

1. Express the controller through MuJoCo-native constraints, tendons, actuators, or model structure.
2. If native modeling is insufficient, move the controller to compiled or JAX code.
3. At minimum, vectorize target calculations, cache constant arrays, preallocate outputs, and remove per-target NumPy calls and allocations.

### 2. A batched MJX path exists for fixed abstract models but is not integrated into GNN-SAC

The upstream branch adds a pure, batch-native `MjxNodeVelocityEnv`. Its explicit `MjxEnvState`, `reset()`, `step()`, and selective `reset_where()` methods operate on a leading environment batch dimension. Physics substeps, node-to-edge command conversion, observations, rewards, rigidity termination, and episode limits are implemented with JAX/MJX and can be wrapped in `jax.jit`; per-environment work is vectorized with `jax.vmap`.

The branch also tests MJX model conversion for every canonical abstract preset, representative state conversion, batched reset and step semantics, CPU/MJX observation and reward agreement, selective reset, deterministic random-key handling, and unsupported configurations.

Important limitations remain:

- Only one fixed abstract model and topology is supported per compiled environment instance.
- Realistic models with angle-bisector or other internal actuators are explicitly rejected.
- `DomainRandomizationConfig`, rendering, and mixed model shapes within a batch are not supported.
- A different batch size triggers a separate JAX compilation.
- The MJX observation is a flat batched array, while GNN-SAC expects PyG graph observations and static `edge_index` metadata.
- GNN-SAC does not construct `MjxNodeVelocityEnv`. Its `mujoco_backend: mjx` setting still does not activate this new path for generated graph environments.
- The learner is PyTorch/PyG while the simulator is JAX/MJX, so integration must avoid CPU round trips, preferably through a validated device-interchange path or a single-framework rollout pipeline.

Recommended library changes:

- [x] Provide an explicit batched environment-state API.
- [x] Implement MJX-compatible reset, node-velocity command conversion, observation, reward, termination, and stepping for fixed abstract models.
- [x] Keep state transformations pure and suitable for `jax.jit` and `jax.vmap`.
- [x] Test all canonical abstract presets for MJX model conversion.
- [x] Document and validate unsupported configurations.
- [ ] Support realistic models and the angle-bisector/internal-actuator path.
- [ ] Support device-native domain randomization without changing compiled model shapes.

Required GNN-SAC integration work:

- [ ] Add a graph-observation adapter that reshapes batched node features and supplies topology metadata once per model.
- [ ] Select the MJX environment through Hydra configuration for compatible generated tasks.
- [ ] Transfer observations and actions between JAX and PyTorch without staging through CPU memory.
- [ ] Add end-to-end training smoke tests and A100/H200 throughput benchmarks.

### 3. Domain randomization recompiles the MuJoCo model on every reset

When `DomainRandomizationConfig.model_factory` is used, reset constructs and compiles a fresh model. Local reset measurements increased from approximately 1–2 ms without model randomization to 22–27 ms with it.

This is relatively small for full 5,000-step episodes but becomes significant when collapse produces short episodes or many environments reset together.

Recommended library changes:

- Separate parameters that require recompilation from parameters mutable through `MjModel` arrays.
- Use `mj_setConst` for valid runtime changes.
- Cache compiled model variants for discretized randomization distributions.
- Support pools of precompiled randomized models.
- Avoid invoking a model factory when only action noise or runtime-mutable randomization is enabled.

### 4. Rigidity reward performs a dense eigendecomposition every environment step

`_critical_eig()` reconstructs the rigidity matrix and computes eigenvalues of a dense `R.T @ R` matrix on every environment step. In the simple-octahedron profile, rigidity reward work consumed roughly 11–14% of step time. Its cost will grow rapidly for larger topologies.

The matrix construction also performs repeated node-name lookup and new allocations.

The MJX path now caches node IDs, rigidity edge indices, and axis indices at environment construction, then builds the matrix and computes the eigendecomposition as a batched JAX device operation. This removes repeated Python name lookup for that path, but it still constructs a dense matrix and performs a full eigendecomposition on every step. The legacy CPU path is unchanged.

Recommended library changes:

- [x] Cache node, edge, and axis indices in the MJX environment.
- [x] Implement rigidity calculation as a batched device operation for the MJX path.
- [ ] Cache equivalent metadata in the legacy CPU path and precompute the static scatter pattern.
- [ ] Preallocate and update only coordinate-dependent matrix values where the execution model permits it.
- [ ] Evaluate a direct singular-value or smallest-relevant-eigenvalue method.
- [ ] Provide a configurable rigidity-evaluation interval.
- [ ] Separate termination checks from reward calculation if they can safely use different cadences.

### 5. Static graph topology is reconstructed for every observation

`get_edge_index()` repeatedly performs node-name lookup, edge-list construction, array allocation, transposition, and `np.unique`, even though topology is static between model changes.

Recommended library changes:

- Cache physical, logical, and control graph edge indices when the model is created.
- Invalidate the cache only when the model changes.
- Return an immutable cached array or a cheap view.

The repository can also cache `edge_index` in the environment as an immediate workaround.

### 6. Node-feature extraction performs repeated Python work and allocations

Physical graph features use Python list comprehensions over body IDs. Logical realistic graph features additionally build dictionaries, regroup cloned nodes, and perform repeated `mj_name2id` calls for connector balls.

The MJX environment caches control-node and physical-node body-ID arrays and uses direct batched indexing into `xpos` and `cvel`. This resolves the repeated Python lookup for its flat abstract-model observation, but not for the legacy CPU graph observations or logical realistic graph views.

Recommended library changes:

- [x] Cache body-ID arrays and use direct indexed observation extraction in the MJX abstract-model path.
- [ ] Cache body-ID arrays for each legacy CPU graph view.
- [ ] Cache physical-to-logical aggregation indices for realistic models.
- [ ] Use direct NumPy indexing into `data.xpos` and `data.cvel` in the CPU path.
- [ ] Expose a vectorized graph-observation call that returns node features with cached topology metadata.

### 7. Node command conversion performs avoidable copies and allocations

`NodeVelocityController.transform()` copies node commands, edge commands, and diagnostic arrays on every action. This is smaller than the realistic angle controller but occurs every environment step.

The MJX environment caches the incidence matrix, passive-node mask, actuator IDs, and control bounds, then performs node-to-edge command conversion with compiled JAX array operations. The existing CPU controller still has the allocation behavior described above.

Recommended library changes:

- [x] Replace Python command conversion with batched JAX matrix operations in the MJX path.
- [ ] Preallocate command buffers in the CPU controller.
- [ ] Make CPU diagnostic snapshots optional.
- [ ] Allow CPU callers to provide output buffers.

## Recommended Training Architecture

### Initial optimized architecture: one learner GPU

For fixed abstract models, the preferred first high-throughput version should use:

1. One compiled `MjxNodeVelocityEnv` batch containing many states for a fixed model/topology.
2. A graph adapter that keeps node features, topology metadata, policy inference, and simulation on the accelerator.
3. A validated JAX-to-PyTorch interchange path that does not copy through host memory, or a single-framework rollout implementation.
4. One batched actor-inference call per environment batch.
5. A tensorized replay buffer with topology metadata stored once.
6. One asynchronous learner on one A100 or H200.
7. BF16 where numerically validated, with critical loss and normalization operations retained in FP32 where needed.
8. Lightweight periodic checkpoints and infrequent replay snapshots.

For realistic models, retain the distributed CPU actor/learner plan until the angle-bisector/internal-actuator path works in MJX: multiple process-based actors, CPU affinity and controlled MuJoCo threading, and shared-memory observation and transition rings. The environment actors should not block on every optimizer update, and the learner should not wait for a single environment step.

### Use of additional GPUs

Until one GPU is saturated, the highest-value use of two to eight GPUs is independent experiments:

- Random seeds.
- Hyperparameter trials.
- Topology train/validation splits.
- Ablation studies.
- Simple versus realistic model comparisons.

This improves research iteration time and statistical coverage more reliably than premature multi-GPU training of a small model.

### Multi-GPU single-run option

If one learner GPU becomes compute-bound after the pipeline is fixed:

- Run one learner replica per GPU.
- Give each replica a local replay shard and local actor group.
- Use DistributedDataParallel gradient synchronization.
- Increase global batch size enough that each GPU receives substantial work.
- Measure policy lag, replay-distribution differences, and learning stability.

The current global batch size of 256 is too small for eight H200s.

## Recommended Implementation Order

1. Reduce evaluation overhead and benchmark valid `nsubsteps` values.
   - [x] Reduce the default evaluation episode count and frequency.
   - [ ] Benchmark and validate lower `nsubsteps` values.
2. Integrate and benchmark `MjxNodeVelocityEnv` for fixed abstract-model training.
   - [x] Implement the upstream batched MJX environment and abstract-preset conversion coverage.
   - [ ] Add the GNN graph adapter, Hydra selection, and device-native JAX/PyTorch interchange.
3. Fix or port the realistic angle-bisector controller in `mujoco-truss-gen`.
4. [x] Add batched actor inference in this repository.
5. Replace threaded environments with process-based actors for the CPU/realistic fallback.
6. Replace the graph-object replay buffer with contiguous tensor storage.
7. Insert transitions online instead of at episode termination.
8. Decouple actor collection from learner updates.
9. Redesign checkpoint persistence.
10. Profile one A100/H200 and add mixed precision, fused optimizers, compilation, or CUDA graphs where useful.
11. Add multi-GPU learner support only if a single GPU is saturated.

## Required Cluster Information

The following hardware details are needed before selecting actor counts and the final distributed layout:

- CPU cores available per GPU.
- RAM available per GPU or per node.
- Whether local NVMe scratch storage is available.
- Whether checkpoints must use a shared parallel filesystem.
- Number of GPUs per node.
- NVLink or NVSwitch availability within a node.
- Network fabric between nodes.
- Scheduler constraints on CPU-to-GPU allocation and long-running worker processes.

## Measurement Notes

All benchmark numbers in this document were collected locally on CPU and are intended to identify dominant code paths and scaling failures. They are not predictions of research-cluster throughput. Final decisions should be based on end-to-end profiles from the target A100 or H200 nodes, including:

- Environment steps per second.
- Physics steps per second.
- Actor inference batch size and latency.
- Replay sampling latency.
- Learner updates per second.
- GPU utilization and kernel occupancy.
- Host-to-device transfer time.
- Actor idle time and learner idle time.
- Checkpoint duration and generated storage traffic.
