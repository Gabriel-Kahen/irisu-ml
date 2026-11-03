# Device arena and search scheduling

`clone/cuda/batch.hpp` provides allocation-free CUDA branch, checkpoint, stable
rank, and survivor compaction operations. It works with fixed-size,
pointer-relative guest heaps and metadata. Each branch copies the parent's
complete heap, then optionally patches one aligned 32-bit input word. A
translated simulator can run many dependent updates in its own kernel between
branch creation and ranking. The API does not move heaps or read scores through
Python or the CPU at each update or search depth.

This is scheduling infrastructure, **not an Irisu GPU simulator**. Its validator
uses explicitly synthetic integer scores. Its throughput measures branch copies
and scheduling, not simulation ticks, game search candidates, or improvement in
high scores. A native CPU heap cannot be copied onto a GPU without translating
its embedded pointers; guest pointers must already be relative to each lane's
heap. The API does not modify the existing exact runtime identity/promotion
contract. A physics implementation and independent differential evidence are
required before attaching an exact-backend identity.

## Reproduce on V100

The local toolkit is CUDA 12.9, with GCC 14 as its host compiler. The inherited
desktop environment sets `CUDA_VISIBLE_DEVICES=-1`; the runner selects device
0 **only in its child process**. It does not change the parent environment or
GPU power limits.

```bash
python3 tools/cuda-exact/batch_benchmark.py --matrix --sanitize \
  --output artifacts/cuda-batch-v100.json
```

For a single fixed budget:

```bash
python3 tools/cuda-exact/batch_benchmark.py \
  --parents 256 --fanout 4 --stride 65536 --depths 16 --repeats 5 \
  --cpu-threads 1 8 16 --output artifacts/cuda-batch-fixed.json
```

Only allocation and buffer ownership are host operations. The `*_async`
functions accept a CUDA stream and caller-owned device arrays; they allocate
nothing and support CUDA graph capture. Device results are not read by their
host wrappers. Launch failures are returned as CUDA errors; device-dependent
invalid parents, mutations, or selections produce zeroed error lanes. Error
and empty lanes rank below every eligible lane and cannot become ready through
branching or compaction. Terminal lanes can survive compaction but cannot be
expanded into fresh ready worlds.

All four objective fields are signed 64-bit integers, sorted in descending
lexicographic order; ties preserve source-lane order. Caller code must populate
them from its actual objective. No floating-point sort behavior or implicit
terminal preference is imposed. Ranking uses a caller-owned padded index
buffer, whose required length is returned by `rank_capacity()`. Invalid lanes
remain flagged in the sorted suffix; a request for more survivors than eligible
lanes yields flagged empty/error survivors, not invented valid worlds.

Source and destination storage must be disjoint, including partial overlaps and
metadata. Parallel operations on separate streams require the normal CUDA
ownership/event synchronization; these wrappers cannot infer cross-stream
dependencies. The allocator initializes metadata to empty; state bytes are
uninitialized until upload, checkpoint, or branching. Freeing an arena requires
all of its users to have completed. Lanes have fixed capacity and stride; the
heap and metadata are never silently resized. Full copies trade memory bandwidth
for straightforward exact checkpoint bytes; copy-on-write remains a future
optimization requiring a translated-memory write barrier.

The standalone validator checks bytewise CPU/GPU parity for duplicated-parent
branches, divergent mutations, snapshot/restore, error-lane zeroing, unaligned
and overflowing mutations, inactive/error/terminal parents, invalid survivor
indices, capacity and overlap guards, padded ranking, ties, signed extrema, and
non-power-of-two counts across shared/global sort boundaries up to 10,007 lanes.
It checks a multi-depth resident pipeline and its CUDA graph replay against a
CPU reference, then records warmed repeated CPU/GPU wall and event times. CPU
comparisons use 1, 8, and 16 OpenMP threads for independent loops and serial stable
sorting. Allocation, upload, graph construction, and final download are outside
both fixed-budget timings; device wall time includes synchronization.

The benchmark emits source/binary hashes, commands, measured memory allocations,
the GPU identity/toolchain, and whole-device utilization/VRAM/power/temperature
samples. Concurrent workloads can contaminate those samples; they do not
establish exclusive process utilization or end-to-end exact-search performance.

The checked-in V100 matrix at
`benchmarks/results/cuda-batch-v100-2026-11-03.json` uses 64KiB heaps, fanout 4,
and 16 depths. Its graph median for 256 parents/1024 children was about 4.03ms,
including branch expansion, synthetic scoring, ranking, and copying 256
survivors at every depth. For 1024 parents/4096 children it was about 13.55ms.
These numbers only establish that GPU-resident scheduling is practical; they
give no estimate of the speed of the untranslated exact physics. Both CPU and
GPU synthetic paths pass bytewise checks before samples are emitted.
