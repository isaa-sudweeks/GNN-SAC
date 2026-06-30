# GNN-SAC Performance Audit and GPU Optimization Plan

## Scope

This audit targets multi-million-timestep GNN-SAC training runs on research-cluster hardware with one to eight NVIDIA A100 or H200 GPUs. It separates bottlenecks in this repository from bottlenecks in `mujoco-truss-gen` so library changes can be handled independently.

The current system is primarily CPU/environment-bound rather than GPU-bound. Adding GPUs will provide limited benefit until environment collection, policy batching, replay storage, and checkpointing are redesigned.

## Repository Bottlenecks

The following issues are ranked by expected wall-clock impact.

### 1. Evaluation can perform five times more simulation than training

The default configuration evaluates 10 episodes of up to 5,000 steps every 10,000 training steps. This can execute 50,000 evaluation steps per 10,000 training steps, in addition to an evaluation at step zero. If episodes regularly reach the time limit, total simulator work can be approximately six times the requested training workload.

Relevant code and configuration:

- `config/gnn_config.yaml`: `eval_freq: 10_000` and `eval_episodes: 10`
- `config/environment.yaml`: `max_steps: 5_000`
- `sac/trainer/online_trainer.py`: synchronous evaluation loop

Recommended changes:

- Use one to three evaluation episodes during training.
- Evaluate every 100,000 to 250,000 training steps.
- Run full evaluation asynchronously or after training.
- Avoid the initial step-zero evaluation unless it is explicitly required as a baseline.

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

- Construct one PyG `Batch` from all active environment observations.
- Perform one actor forward pass per vector step.
- Split the resulting node actions by graph afterward.
- Avoid unnecessary stochastic sampling and log-probability calculation during deterministic evaluation.
- Transfer actions to CPU once per batch rather than once per environment.

### 4. Environment vectorization uses Python threads

`RepeatedEnvWrapper` uses `ThreadPoolExecutor`. This is ineffective for Python-heavy environment work and showed poor scaling in local measurements:

| Environments | Simple truss aggregate | Realistic truss aggregate |
|---:|---:|---:|
| 1 | 910 steps/s | 19.5 steps/s |
| 2 | 1,343 steps/s | 14.3 steps/s |
| 4 | 1,231 steps/s | 10.5 steps/s |
| 8 | 854 steps/s | 10.3 steps/s |

These measurements are from local CPU hardware and should not be treated as cluster throughput predictions. They do establish that the current threading design does not scale consistently.

Recommended changes:

- Use process-based actors instead of threads.
- Assign CPU affinity to actor processes.
- Use shared-memory queues or ring buffers rather than pickling graph objects between processes.
- Separate environment collection from learner execution.

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

These findings apply to installed `mujoco-truss-gen` version 0.9.0 and are ranked by expected impact.

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

### 2. Generated truss environments have no MJX or batched step path

`MujocoTrussEnv._advance()` always loops over native `mujoco.mj_step`. The repository's `mujoco_backend: mjx` setting is not passed into the generated `truss-graph` environment configuration, so it has no effect on the primary graph-training path.

The repository's existing backend benchmark targets older local truss environments and does not demonstrate that generated `mujoco-truss-gen` graph environments use MJX.

Recommended library changes:

- Provide an explicit batched environment-state API.
- Implement MJX-compatible reset, controller, observation, reward, termination, and stepping functions.
- Keep the state transformations pure and suitable for `jax.jit` and `jax.vmap`.
- Document unsupported MuJoCo features for realistic models before treating MJX as the primary path.

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

Recommended library changes:

- Cache node and edge indices and the static matrix sparsity pattern.
- Preallocate and update only coordinate-dependent matrix values.
- Evaluate a direct singular-value or smallest-relevant-eigenvalue method.
- Provide a configurable rigidity-evaluation interval.
- Separate termination checks from reward calculation if they can safely use different cadences.

### 5. Static graph topology is reconstructed for every observation

`get_edge_index()` repeatedly performs node-name lookup, edge-list construction, array allocation, transposition, and `np.unique`, even though topology is static between model changes.

Recommended library changes:

- Cache physical, logical, and control graph edge indices when the model is created.
- Invalidate the cache only when the model changes.
- Return an immutable cached array or a cheap view.

The repository can also cache `edge_index` in the environment as an immediate workaround.

### 6. Node-feature extraction performs repeated Python work and allocations

Physical graph features use Python list comprehensions over body IDs. Logical realistic graph features additionally build dictionaries, regroup cloned nodes, and perform repeated `mj_name2id` calls for connector balls.

Recommended library changes:

- Cache body-ID arrays for each graph view.
- Cache physical-to-logical aggregation indices.
- Use direct NumPy indexing into `data.xpos` and `data.cvel`.
- Expose a single vectorized graph-observation call that returns features with cached topology metadata.

### 7. Node command conversion performs avoidable copies and allocations

`NodeVelocityController.transform()` copies node commands, edge commands, and diagnostic arrays on every action. This is smaller than the realistic angle controller but occurs every environment step.

Recommended library changes:

- Preallocate command buffers.
- Make diagnostic snapshots optional.
- Allow callers to provide output buffers.

## Recommended Training Architecture

### Initial optimized architecture: one learner GPU

The first high-throughput version should use:

1. Multiple process-based CPU environment actors.
2. CPU affinity and controlled MuJoCo threading per actor.
3. Shared-memory observation and transition rings.
4. One batched actor-inference call across ready environments.
5. A tensorized replay buffer with topology metadata stored once.
6. One asynchronous learner on one A100 or H200.
7. BF16 where numerically validated, with critical loss and normalization operations retained in FP32 where needed.
8. Lightweight periodic checkpoints and infrequent replay snapshots.

This resembles a distributed off-policy actor/learner design. The environment actors should not block on every optimizer update, and the learner should not wait for a single environment step.

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
2. Fix the realistic angle-bisector controller in `mujoco-truss-gen`.
3. Add batched actor inference in this repository.
4. Replace threaded environments with process-based actors.
5. Replace the graph-object replay buffer with contiguous tensor storage.
6. Insert transitions online instead of at episode termination.
7. Decouple actor collection from learner updates.
8. Redesign checkpoint persistence.
9. Profile one A100/H200 and add mixed precision, fused optimizers, compilation, or CUDA graphs where useful.
10. Add multi-GPU learner support only if a single GPU is saturated.

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
