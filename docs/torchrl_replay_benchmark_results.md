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

## Production checkpoint validation

Validation on 2026-09-12 used an immutable 4,173,889,757-byte full trainer
checkpoint from a live nine-topology distillation-pilot run at step 700,245.
Each topology contained 77,805 transitions, for 70.0% occupancy of the
999,999-transition rounded replay capacity. Tests ran serially on A100-SXM4
80GB allocations so one case could not distort another case's CPU-sensitive
replay timing.

The non-destructive conversion retained every per-topology size and reduced the
checkpoint to 897,563,009 bytes (4.65x smaller). Source deserialization took
276.1 seconds, tensorization took 32.5 seconds, and the complete conversion,
save, verification, and hashing process took 315.4 seconds. This is a one-time
conversion cost; the source SHA-256 remained
`b1bd5e04a097739739c129f12bd1eca61a0369f4560d01f4f0aabdc3fbfa6f73`.

The converted checkpoint reached its first successful optimizer update in
3.57 seconds: 0.55 seconds to deserialize, 0.33 seconds to initialize the
buffer and agent, 0.18 seconds to restore their state, and 2.24 seconds for the
first update. This clears the under-three-minute restart gate with substantial
margin. `auto` placed all nine buffers on CUDA using 867.6 MB of replay
storage, while measured peak PyTorch CUDA allocation was 1.13 GB.

### Production-occupancy performance

Each backend used the same checkpoint state and seed schedule. Results include
100 standalone replay samples plus three independent 100-update blocks.

| Backend and storage | Median replay sample | Replay speedup | Median updates/s | Update throughput gain | Replay placement |
| --- | ---: | ---: | ---: | ---: | --- |
| Legacy | 142.672 ms | 1.00x | 1.170 | baseline | CPU objects |
| Tensor CPU-pinned | 8.681 ms | 16.44x | 1.573 | 34.4% | CPU pinned |
| Tensor auto | 9.470 ms | 15.07x | 1.597 | 36.4% | 867.6 MB CUDA |

The tensor backend remains materially faster at real replay occupancy. `auto`
was only 1.5% faster than CPU-pinned storage on median optimizer throughput,
while CPU-pinned sampling was 8.3% faster and used 867.6 MB less replay VRAM.
This run therefore does not establish one tensor placement as universally
better: CPU-pinned is the resource-efficient choice, while `auto` provides a
small measured end-to-end gain within a fixed VRAM budget.

### Long matched learner fork

A production-state legacy-versus-tensor fork compared 100 sampled batches and
then ran 1,000 paired CUDA optimizer updates. Every compared replay batch and
the CPU and CUDA RNG states were exactly equal. Learner metrics were not
bit-exact: a strict `rtol=1e-5`, `atol=1e-6` probe first rejected a
`pi_grad_norm` difference of 1.29e-5 at update 24. The final predeclared
float32 gate (`rtol=1e-4`, `atol=1e-5`) was first exceeded at update 20, and
the maximum absolute difference across saved learner state was 0.216 at update
1,000.

This drift cannot be attributed to the replay backend. A 300-update
tensor-versus-tensor control, using the same checkpoint twice, also exceeded
the float32 gate and reached 0.328 maximum absolute learner-state drift at
update 300. The legacy-versus-tensor fork was comparable at 0.358 at update
300. Sequential CUDA graph reductions therefore do not provide deterministic
long-horizon parameter identity even when both sides use the same replay
backend.

The defensible correctness result is narrower: production replay values,
ordering, sampling, and RNG consumption are exact, and short CPU learner
updates remain bit-exact. Long CUDA parameter equality is not a usable
promotion criterion for this code path. Learning-outcome equivalence requires
a paired multi-seed evaluation rather than progressively widening numerical
tolerances.

### Checkpoint regressions

New deterministic tests exercise a completely full ring at a nonzero cursor,
format-v3 restoration, subsequent overwrites, and wrapped legacy-to-tensor
conversion. Physical slots, future cursor movement, sampled batches, and RNG
state remain exact. A production-path asynchronous checkpoint test also blocks
the background writer while live tensor replay is overwritten and verifies
that the checkpoint remains an isolated, exactly restorable dispatch-time
snapshot.

The indexed-CUDA regression exposed and fixed a separate placement bug:
`device=cuda:1` previously budgeted and allocated unindexed `cuda` storage.
Memory queries and replay placement now preserve the configured CUDA device.

## Decision and next steps

The production restart, occupancy, checkpoint, exact replay-data, bounded
end-to-end, and memory-safety gates support the tensor replay backend as the
better computational path. It remains opt-in in this PR so changing the
default is an explicit follow-up decision rather than an incidental behavior
change.

Before changing the default, run one sustained training job through multiple
periodic asynchronous checkpoints and a real scheduler restart. Then run at
least three paired seeds with periodic evaluation and compare learning-curve
area and final-window performance under the same checkpoint-selection rule;
the CUDA fork shows that single-seed parameter identity cannot substitute for
that experiment. Prefer CPU-pinned storage when conserving VRAM matters;
retain `auto` when its bounded placement policy and small measured throughput
gain are worth the additional allocation.
