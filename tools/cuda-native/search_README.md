# Independent typed game search

`search_runner.cu` uses the separate typed physics implementation and generated
rules. It does not link the translated guest or call the native physics library.
The shared search-v2 Candidate/Result/rank wire remains unchanged.

Generate the genuinely distinct workload used in the new campaign:

```sh
python tools/cuda-native/search_specs.py irisu-high-score-4393101.rpy /tmp/search-distinct.bin --prefix 100 --horizon 4 --count 12288
```

Its SHA-256 is `29fc880c37e2cced33ca14849f1c6f85471ddd1eaed7f4b0015479b455484c3c`.
The generator reuses the read-only canonical mutation algorithm and checks unique
40-byte specifications. Native qualification also found 12,288 distinct Result
packets. For the earlier repeated-work scaling fixture, use `--count 512
--repeat 16` or `--repeat 24`; those are explicitly repeated specifications.

After building `irisu-native-search-campaign` in the native exact configuration
and `irisu-cuda-native-search-runner` in the separate CUDA-native configuration:

```sh
env LD_LIBRARY_PATH="$PWD/.local/exact-runtime-20260824/exact-runtime/guest" \
  build-project-exact/irisu-native-search-campaign irisu-high-score-4393101.rpy /tmp/search-distinct.bin 100 4 8 20 4393102 --final-only > /tmp/search-native.bin
env -u CUDA_VISIBLE_DEVICES \
  build-cuda-native-bounded/clone/cuda_native/irisu-cuda-native-search-runner irisu-high-score-4393101.rpy /tmp/search-distinct.bin 100 4 --copy-u64 --reuse-prefix --physical-snapshot --warmups 0 --repeats 20 --final-only --skip-compaction-check > /tmp/search-gpu.bin
cmp /tmp/search-native.bin /tmp/search-gpu.bin
```

The bounded build is explicitly opt-in; its numeric contract is in
[bounded_math_contract.md](bounded_math_contract.md). The checked build remains
the default. Both use strict RN operations and observed binary32 stores.

`--reuse-prefix` prepares one immutable checkpoint per process. Without
`--host-prefix`, initialization, reset, prefix and every future execute on GPU.
By default all Game and Scene bytes, including scratch, are retained in every clone.
`--copy-u64` uses aligned global 64-bit PTX transport; byte copying stays default.
`--split-rebind` is an optional scheduling control that showed no benefit.
Default prefix behavior replays the prefix for every batch.

Each result/rank packet is compared internally with the preceding batch. With
prefix reuse, a separate raw snapshot checks every cached World, Scene and
prefix-Result byte after every batch; the source is never normalized or rebound.
Selected clones additionally undergo raw state comparison after pointer
normalization, then independent continuation and another full comparison.
`--skip-compaction-check` skips only these extra selected continuations; the
cached-prefix guard and complete future/compaction copies remain enabled.

Stderr reports prefix execution count, guards, copy/profile flags and phase
samples. Later reused-prefix events are explicitly zero. The program outputs
only the final complete packet; every repeated batch still executes its futures.
Use `--warmups 0` for complete campaign wall measurements, including cold CUDA
allocation and teardown. Compare the native **cached-prefix campaign**, also one
process/one prefix, using unrestricted workers 8 and 16. Read native
`verification_ms` separately for a conservative verification-free CPU control.
GPU prefix-guard costs remain included. Keep one-shot and campaign results
separate; a warm phase divided by native process wall is not matched timing.

Qualification receipts preserve source/binary/input hashes and raw samples.
Earlier CPU media contention invalidated headline claims from older campaign
measurements; unrelated processes were not modified. The final paired campaign
measured **6.458x**: best verification-free CPU median 30.062683 s versus GPU
full-wall median 4.655091 s. It has three runs per CPU worker configuration and
six GPU runs. Both retain one prefix per process. Native v2 excludes parent-only
verification buffers from candidate forks using dedicated MADV_DONTFORK
mappings. All 12 final packets match native execution. Small cold searches
still favor CPU; this result does not establish universal speedup or admission.

Reproduce paired timing with the native output above as the expected packet:

```sh
python3 tools/cuda-native/benchmark_campaign.py \
  --native build-project-exact/irisu-native-search-campaign \
  --gpu build-cuda-native-bounded/clone/cuda_native/irisu-cuda-native-search-runner \
  --replay irisu-high-score-4393101.rpy --specs /tmp/search-distinct.bin \
  --expected /tmp/search-native.bin \
  --library .local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so \
  --output /tmp/search-paired.json
```

See [the final receipt](../../benchmarks/results/cuda-native-final-campaign-2026-11-04.json)
for all samples, input/binary pins and limits. Build with
`IRISU_CUDA_NATIVE_BOUNDED_MATH=ON` in a separate directory; see
[the build guide](../../docs/cuda-native.md).

## Optional physical snapshot

`--physical-snapshot` retains every Game byte and the complete World and
broad-phase state, copying Scene bytes `[0, offsetof(Scene, scratch))`. It omits
only transient SceneScratch plus its trailing padding. This reduces transported
checkpoint bytes from 2,265,064 to 1,415,136; full allocation strides remain
unchanged. All warm impulses, pending deletes and SAP buffers remain intact.
The default still copies all scratch.

This option applies only after complete public calls. Its source audit, native
edge fixtures, two workspace poisons, full persistent-byte continuation checks,
memcheck and virgin-workspace initcheck passed. `--scratch-poison 1` or `2` is
validation-only; the actual fast path leaves workspace untouched. Selected-state
checks in reduced mode compare every Game and persistent Scene byte, while the
cached-prefix guard continues comparing the **entire** source checkpoint.

The final pure-GPU campaign command above uses `--physical-snapshot` and
`--repeats 20` with native `batches=20`. CPU workers remain unrestricted;
physical-core pinning was slower and was rejected. Final speedup claims require
the paired native v2 cached controller with parent verification buffers excluded
from candidate forks. See `search_physical_qualification.json` for exact gates;
the earlier one-sample GPU pilot is superseded by the repeated paired campaign.

## Additional efficiency profile

Final paired medians: pure-GPU physics improves **1.956x** (4.618 → 2.361 s);
explicit CPU prefix plus GPU futures improves **2.129x** (4.622 → 2.171 s).
The additional 2x target is reached in the latter mode. These are complete
20-batch campaign wall times, with three runs per configuration in each mode.

The [build guide](../../docs/cuda-native.md#further-gpu-efficiency) provides the
selected compile flags. Generate the same distinct packet as above, then run:

```sh
env -u CUDA_VISIBLE_DEVICES \
  build-cuda-native-efficiency-opt/clone/cuda_native/irisu-cuda-native-search-runner \
  irisu-high-score-4393101.rpy /tmp/search-distinct.bin 100 4 \
  --copy-u64 --reuse-prefix --physical-snapshot --compact-clone \
  --merge-rank --tiled-selection --parallel-prefix-guard \
  --warmups 0 --repeats 20 --final-only --skip-compaction-check \
  --host-prefix > /tmp/search-efficient.bin
cmp /tmp/search-native.bin /tmp/search-efficient.bin
```

This explicitly uses CPU preparation for the one shared prefix. Remove
`--host-prefix` for pure-GPU physics. It changes the prefix location, not the
number of future rollouts. Every batch recomputes all 12,288 futures; repeated
results are compared, not reused. Omit `--skip-compaction-check` for full selected
state checks and independent continuations; prefix guards always remain on.

Merge sorting retains the original full comparator and index tie-breaks.
Tiled selected copies distribute work across more GPU blocks. Active cloning
copies occupied physical pools, all live Game vector values and scalar state,
with the complete broadphase. Inactive pointer tails are explicitly restored
to zero before live-pointer relocation. All runtime pool capacities and full
allocation strides remain unchanged. Poison tests and native continuations
qualify the representation; raw inactive-byte equality is not its contract.

The previous GPU executable is preserved locally at
`artifacts/gpu-efficiency-20261104/search-baseline`, SHA-256
`2f27d7ff719f66a53b9883ed602f654a1af5bab43dfa0d8c12fe6bc835967f72`.
Its source/build profile is the prior `d3cb2d4` campaign, with bounded math,
frontend `mid`, and compiler-default register allocation. Compare complete
processes against that baseline, with a fresh native expected packet:

```sh
python3 tools/cuda-native/benchmark_gpu_efficiency.py \
  --baseline artifacts/gpu-efficiency-20261104/search-baseline \
  --candidate build-cuda-native-efficiency-opt/clone/cuda_native/irisu-cuda-native-search-runner \
  --replay irisu-high-score-4393101.rpy --specs /tmp/search-distinct.bin \
  --expected /tmp/search-native.bin --output /tmp/efficiency-hybrid.json \
  --candidate-flag=--compact-clone --candidate-flag=--merge-rank \
  --candidate-flag=--tiled-selection --candidate-flag=--parallel-prefix-guard \
  --candidate-flag=--host-prefix
```

Remove the last flag for a separate pure-GPU comparison. The script alternates
order across three paired rounds, pins both binaries/inputs, checks native
packets and all counts, and includes setup, output and teardown. Two seconds
of idle time before each process are outside the timer to avoid overlapping
CUDA context teardown between successive GPU processes. Warmups are reported
separately. Keep the device free of competing work during comparison.

Actual-native search and continuation gates are reproducible with
`qualify_search_efficiency.py`; raw sample pins and the selected/rejected
experiments are in the [efficiency receipt](../../benchmarks/results/cuda-native-efficiency-2026-11-04.json).

## Capacity512 follow-up

Another 2x was not achieved: the final matched hybrid medians were
**2.166970 → 1.830900 seconds (1.184x)**. This is an optional smaller physical
contact pool, with explicit overflow failure and default capacity 4096 preserved.
All 983,040 GPU future ticks still execute and the complete native packet matches.
See the [current experiment guide](efficiency2_README.md) for the exact build,
matched runtime flags, capacity boundary, qualification and rejected trials.
