# Second GPU efficiency experiment

This experiment targets another 2x relative to the preceding 2.171-second
hybrid campaign, not relative to the original CPU or 4.62-second GPU result.
**The additional 2x target was not reached.** Three final paired runs measured
2.166970 seconds before and 1.830900 seconds after: **1.184x**. Candidate times
ranged from 1.830748 to 1.836700 seconds. Allocation dropped from 27.85 GB to
13.40 GB (decimal units). The selected outputs remain bit exact to native.
See the [full receipt](../../benchmarks/results/cuda-native-efficiency2-2026-11-04.json).
The selected improvement is an optional 512-contact typed physics pool.
Default capacity remains 4096; SAP pair capacity and Game vectors stay 4096.
Exhaustion returns an explicit capacity failure. This restricted profile is
qualified for the recorded replay and listed search workloads, not arbitrary
512-body scenes. See the [capacity contract](contact_capacity_contract.md).

## Build and reproduce

```sh
cmake -S . -B build-cuda-native-efficiency-cap \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DIRISU_BUILD_CUDA_NATIVE_EXPERIMENTS=ON \
  -DIRISU_BUILD_CUDA_EXPERIMENTS=OFF -DIRISU_BUILD_TESTS=OFF \
  -DIRISU_CUDA_NATIVE_ARCHITECTURES=70 \
  -DIRISU_CUDA_NATIVE_BOUNDED_MATH=ON -DIRISU_CUDA_NATIVE_BOUNDED_STORE=ON \
  -DIRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC=ON -DIRISU_CUDA_NATIVE_CACHED_ROTATION=ON \
  -DIRISU_CUDA_NATIVE_INLINE_STEP=ON -DIRISU_CUDA_NATIVE_WIDE_REBIND=ON \
  -DIRISU_CUDA_NATIVE_MAX_CONTACTS=512 \
  -DIRISU_CUDA_NATIVE_GAME_FAST_COMPILE=min -DIRISU_CUDA_NATIVE_MAX_REGISTERS=0 \
  -DIRISU_CUDA_NATIVE_F32_STORE=OFF -DIRISU_CUDA_NATIVE_F32_DIV_STORE=OFF \
  -DIRISU_CUDA_NATIVE_DEVICE_TRIG=0 -DIRISU_CUDA_NATIVE_INTERLEAVED_SOLVER=OFF \
  -DIRISU_CUDA_NATIVE_ENTRY_INDEX=OFF -DIRISU_CUDA_NATIVE_SPARSE_RESET=OFF
cmake --build build-cuda-native-efficiency-cap \
  --target irisu-cuda-native-search-runner -j2
```

Generate the distinct specs and actual-native expected packet with the
[existing search commands](search_README.md). Compare against the preserved
preceding selected executable, with the same optimized runtime flags on both:

```sh
python3 tools/cuda-native/benchmark_gpu_efficiency.py \
  --baseline artifacts/gpu-efficiency2-20261104/search-baseline \
  --candidate build-cuda-native-efficiency-cap/clone/cuda_native/irisu-cuda-native-search-runner \
  --replay irisu-high-score-4393101.rpy --specs /tmp/search-distinct.bin \
  --expected /tmp/search-native.bin --output /tmp/efficiency2.json \
  --baseline-flag=--host-prefix --baseline-flag=--compact-clone \
  --baseline-flag=--merge-rank --baseline-flag=--tiled-selection \
  --baseline-flag=--parallel-prefix-guard \
  --candidate-flag=--host-prefix --candidate-flag=--compact-clone \
  --candidate-flag=--merge-rank --candidate-flag=--tiled-selection \
  --candidate-flag=--parallel-prefix-guard
```

Each measured process executes 20 fresh batches of 12,288 distinct four-tick
futures: 983,040 GPU physics ticks, plus one shared 100-tick CPU prefix. Full
process wall includes setup, allocation, output and teardown. Prefix guards
remain enabled; extra selected-state continuation checks pass separately.
Both baseline and candidate must match the complete actual-native packet.
Remove both host-prefix flags for a separate all-GPU-physics comparison.

The [qualification runbook](contact_capacity_gates.md) covers the full native
replay, dynamic Scene/SAP state, compact snapshots, independent selected-state
continuations, poisoned scratch and CUDA memory checks. Raw fixed Scene packet
layouts differ across capacities; the dynamic packet format is independent.

## Optional experiments retained alongside the scalar runner

`irisu-cuda-native-batched-search-runner` owns independent pools for one or two
concurrent batches. `--concurrent-batches 1|2` selects launch grouping;
`--order-candidates` groups lexicographically similar effective action words,
then restores original Result IDs and ranking tie-breaks. Warmups and measured
batches never share a group. Every measured future executes independently;
`total_future_ticks` counts their actual work. Phase samples cover a launch
group and must be divided by its `batches` for per-batch attribution.
These scheduling variants measured slower and are not selected.

`IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER=ON` enables a configured external hot-field
workspace in the batched runner. The [solver contract](solver_interleaved_contract.md)
describes exact serial contact order, dense body mapping and explicit GPU
layout fallbacks. Its 152-case full-byte stage differential passes, but the
fastest measured version still loses to the scalar capacity profile.

[Adjacent float division stores](div_store_contract.md) are exact on their
qualified domain, but showed no meaningful campaign gain. They remain off.
[Device trig](device_trig_contract.md) is a separate approximate profile:
library mode passed sampled 1-ULP bounds but changed body/trace hashes for
12,274 of 12,288 candidates while saving less than 1%; intrinsic mode failed
sampled error limits. Both remain off. `--candidate-comparison outcome` is an
explicit benchmark diagnostic permitting only digest changes while requiring
all discrete Result fields and ranks to match native; exact comparison is
the default and is required for the selected result.

Instrumentation runners v1/v2/v3 are diagnostics, not benchmark executables.
V2/v3 use private per-thread counters; their summed SM clock intervals include
stalls and instrumentation. They do not measure utilization or kernel wall.
