# TorchRL replay prototype benchmark results

Benchmark date: 2026-09-11 (ORC)

Commit under test: `5dc66989b26911f8ffee0e17a2d3016b8a928cb2`. The
mixed-placement budget correction is commit `5e142fc` and changes only the
reproducible Slurm command from `0.06` to `0.006` GiB.

Hardware: NVIDIA A100-SXM4-80GB. The replay benchmark used three full
10,000-transition topology buffers with 4, 6, and 8 nodes, batch size 255, 10
warmups, 100 timing repetitions, three checkpoint repetitions, and 20 complete
optimizer updates. All CUDA equivalence tests passed.

## Replay microbenchmark

| Storage | Placement | Legacy direct median | Tensor direct median | Sampling speedup | Optimizer throughput gain |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU pinned | CPU, CPU, CPU | 9.880 ms | 2.919 ms | 3.38x | 28.94% |
| CUDA | CUDA, CUDA, CUDA | 9.855 ms | 3.086 ms | 3.19x | 28.83% |
| Auto | CUDA, CUDA, CUDA | 9.892 ms | 3.075 ms | 3.22x | 28.58% |
| Forced mixed | CUDA, CUDA, CPU | 8.654 ms | 2.923 ms | 2.96x | 24.87% |

CPU-pinned storage was marginally fastest in this isolated sampling workload.
CUDA storage was not faster enough to justify assuming that replay belongs on
the GPU for every workload. `auto` remains valuable for enforcing a fixed
memory budget and deterministic whole-topology fallback.

For default `auto`, full checkpoint writing fell from 7,017.4 ms to 36.9 ms
(190.2x faster), and full replay loading/restoration fell from 7,992.3 ms to
12.5 ms (641.7x faster). The forced mixed result was also exact and restored in
9.6 ms versus 6,959.0 ms for legacy.

`ReplayBufferEnsemble(sample_from_all=True)` is rejected. It was exact for
CPU-pinned storage but added 8.19% median overhead, above the 5% limit. Its CUDA
result was not exact, and heterogeneous mixed storage cannot be represented by
one ensemble. Training therefore uses the direct coordinator.

## Matched bounded MJX training

The matched three-topology Warp-MJX runs used 384 environments, 9,600
transitions, batch size 255, replay ratio 10, and identical seed/configuration.

| Backend | Environment throughput | Median replay sampling |
| --- | ---: | ---: |
| Legacy | 98.15 transitions/s | 143.88 ms |
| TorchRL tensor, auto | 207.10 transitions/s | 4.36 ms |

Tensor-auto improved bounded end-to-end throughput by 111.0% and replay
sampling by 33.0x. Peak observed GPU use across the serial protocol was 2,369
MiB, leaving at least 78,783 MiB free. The configured `auto` placement estimated
596.0 MB for the full one-million-transition training capacity, below its 8 GiB
budget.

## Decision

The prototype clears the bounded replay, checkpoint, end-to-end, equivalence,
and memory-safety gates. It remains opt-in (`replay_backend=torchrl_tensor`)
rather than replacing the legacy default yet because the original
production-scale checkpoint responsible for the roughly 30-minute restart was
not available in this run. Before changing the default, convert a copy of that
checkpoint and verify its actual under-three-minute restart plus a matched
CPU-pinned-versus-auto training comparison at production replay occupancy.
