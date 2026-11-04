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
build-project-exact/irisu-native-search-campaign irisu-high-score-4393101.rpy /tmp/search-distinct.bin 100 4 8 12 4393102 --final-only > /tmp/search-native.bin
build-cuda-native-bounded/clone/cuda_native/irisu-cuda-native-search-runner irisu-high-score-4393101.rpy /tmp/search-distinct.bin 100 4 --copy-u64 --reuse-prefix --warmups 0 --repeats 12 --final-only > /tmp/search-gpu.bin
cmp /tmp/search-native.bin /tmp/search-gpu.bin
```

The bounded build is explicitly opt-in; its numeric contract is in
[bounded_math_contract.md](bounded_math_contract.md). The checked build remains
the default. Both use strict RN operations and observed binary32 stores.

`--reuse-prefix` prepares one immutable checkpoint per process. Without
`--host-prefix`, initialization, reset, prefix and every future execute on GPU.
All Game and Scene bytes, including scratch, are retained in every clone.
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
Later CPU media contention invalidated headline claims from the older campaign
measurements; unrelated processes were not modified. New fair CPU comparisons
require a quiet window and repeated paired campaigns. The tools remain
experimental; these fixtures do not establish universal speedup or admission.
