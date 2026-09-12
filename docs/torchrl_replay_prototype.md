# TorchRL tensor replay prototype

The prototype is opt-in. Existing runs continue to use the object-based replay
buffer because `replay_backend: legacy` remains the default.

Use the tensor backend with one of three fixed-at-startup storage policies:

```text
replay_backend=torchrl_tensor replay_storage=cpu_pinned
replay_backend=torchrl_tensor replay_storage=cuda
replay_backend=torchrl_tensor replay_storage=auto
```

`auto` assigns whole topology buffers, smallest first with task-name tie
breaking, within the minimum of `replay_gpu_fraction` of total VRAM,
`replay_gpu_max_gb`, and currently free VRAM less
`replay_gpu_reserve_gb`. It never migrates a buffer during training. `cuda`
fails before allocation unless every topology's estimated tensor storage fits
the same safety budget. Samples from CPU storage are pinned by TorchRL and
copied non-blockingly to a CUDA learner.

Each topology owns a `TensorDictReplayBuffer` and stores only changing tensor
fields. Graph structure, action masks, and edge roles are stored once. The
coordinator generates indices on CPU in existing task order and constructs PyG
batches directly from dense tensor blocks. This preserves the legacy Torch RNG
sequence and topology weighting.

`ReplayBufferEnsemble(sample_from_all=True)` is available only through the
benchmark path. It is adopted only when its batch and RNG results are exact and
its median overhead is at most 5%; training otherwise uses the direct
coordinator.

## Benchmark

Run a CPU smoke benchmark with:

```bash
python scripts/benchmark_gnn_replay.py --device cpu --prototype \
  --storage cpu_pinned --node-counts 4,6,8 --batch-size 255 \
  --entries-per-task 10000 --output replay-cpu.json
```

On CUDA, repeat with `--storage cuda` and `--storage auto`. JSON output includes
insertion, direct and ensemble sampling, checkpoint snapshot/serialization/
deserialization/restoration, file size, process memory, CUDA allocator memory,
estimated and actual replay bytes, placements, and fallback reasons.

## Checkpoints and teacher conversion

Tensor replay is stored as format v3 CPU tensors. CUDA placement is metadata,
so a resumed run may select a different storage policy. Async checkpoints take
one isolated CPU snapshot before dispatching background serialization.

Convert an old replay checkpoint non-destructively:

```bash
python scripts/convert_gnn_replay_checkpoint.py old.pt converted.pt
```

The converter refuses to overwrite either input or an existing destination,
verifies the new file, and writes `converted.pt.conversion.json` with hashes,
sizes, topology counts, and phase timings. Distillation reads the tensor fields
directly when building its teacher-target shards.

Do not change the default backend until the CUDA equivalence, memory-headroom,
restart, replay, checkpoint, and bounded end-to-end throughput promotion gates
have all passed.
