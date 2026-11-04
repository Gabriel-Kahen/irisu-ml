# Separate typed GPU physics

`clone/cuda_native` is an opt-in implementation alongside the existing CPU
backends and translated `clone/cuda` backend. It does not replace them. The
typed hot path contains bodies, shapes, contacts, constraints and integer SAP
state, with array indices replacing persistent pointers. It does not execute
Guest instructions or fall back to another physics engine.

The numerical helpers in `tools/cuda-exact/x87_numeric.cuh` are reused read-only.
They preserve the original PC53 operation order and observed binary32 stores.
This removes the emulated register machine without silently changing the
physical model. Unsupported numerical domains fail explicitly.

The scene pool has 512 bodies including the native shape-free ground, so its
one-shape-per-body API admits at most 511 shaped bodies. The separate SAP stage
has the native 512-proxy capacity. Joint creation is outside this implementation.

## Build

Use a separate 64-bit build tree. The original production exact worker remains
32-bit. V100 requires CUDA 12.x; these experiments use CUDA 12.9 and GCC 14.

```sh
cmake -S . -B build-cuda-native \
  -DIRISU_BUILD_CUDA_NATIVE_EXPERIMENTS=ON \
  -DIRISU_BUILD_CUDA_EXPERIMENTS=OFF \
  -DIRISU_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
cmake --build build-cuda-native --target \
  irisu-native-collision irisu-native-solver irisu-native-broadphase \
  irisu-cuda-native-factory-runner irisu-cuda-native-scene-runner \
  irisu-cuda-native-game-runner irisu-cuda-native-search-runner irisu-native-domain -j 2
```

GPU commands below remove `CUDA_VISIBLE_DEVICES` only from the child process.
The shell's environment and other running services are unaffected.

## Qualification

The independent ELF32 oracles call the actual pinned native library, not a
portable source rebuild. Its SHA256 is
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`.
See [runtime recovery](exact-runtime-recovery.md) for recovery and provenance.
Reference source is an aid; native instruction order and native comparisons
resolve discrepancies, including the wrapper's triangle vertex order.

Current component gates cover:

| Component | Native comparison |
| --- | --- |
| Collision | Circle/circle, polygon/circle and polygon/polygon; complete 60-byte manifolds, preserved impulses and unused slots |
| Solver | Constraint construction, PreSolve, velocity/position iterations and PostSolve; explicit stored fields |
| Factory | 1,536 circle/box/triangle wrapper creations; mass, inertia, COM, transforms, material, vertices, normals, core vertices and local OBB |
| Broadphase | Full 95,292-byte normalized state after operations; callback/query order, ties, timestamp wrap, quantization, all 512 proxies, removal and reuse |
| Fixed scenes | 512 native contacted-circle frames plus independent integration, sleep and shape-cache stage fixtures |
| Dynamic scenes | 3,072 native full-step frames, including mixed shapes, pair transitions, deletion/reuse, setters, partial bounds and freeze; initialized physical fields and live broadphase state |
| Game adapter | Entire 97,220-row record digest on typed host and bounded-math GPU; first 4,097 rows on checked GPU; native body/observation/event/transition digests |

Factory comparison excludes allocation/proxy bookkeeping, qualified separately.
Broadphase normalization removes only native owner/callback pointer words.
Small stage proofs do not establish arbitrary full-world or full-game behavior.
The fixed-pair scene entry rejects topology transitions explicitly. The separate
`ScenePipelineState` implements real SAP callbacks, proxy changes, contact
creation/removal, deferred cleanup, origin/velocity setters and freeze behavior.

The checks and their native ABI contracts live in `tools/cuda-native`:

- [Collision and broadphase oracle contract](../tools/cuda-native/oracle_contract.md)
- [Solver contract and commands](../tools/cuda-native/solver_README.md)
- [Scene contract](../tools/cuda-native/scene_contract.md)

`irisu-native-domain` checks invalid geometry/numerics, native pair-order
requirements, preserved no-contact data, and transactional rejection.

The game runner reuses the existing generated game rules through a new typed
adapter. GPU initialization receives empty object storage prepared on the host;
all physical reset/body creation and replay steps execute on the GPU. This
avoids device compilation of giant aggregate default constructors. Game and
physics calls have explicit function boundaries to control compilation size.
`IRISU_CUDA_NATIVE_GAME_FAST_COMPILE=mid` controls frontend optimization work;
the strict floating-point flags remain enabled. Components use ordinary O3.

```sh
python3 tools/cuda-native/qualify_game.py \
  --native-oracle build-project-exact/irisu-game-oracle \
  --runner build-cuda-native/clone/cuda_native/irisu-cuda-native-game-runner \
  --library .local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so \
  --replay irisu-high-score-4393101.rpy --steps 4096 --gpu \
  --output artifacts/gpu-native-game-qualification.json
```

The full host record also passes same-input clone/rebind continuation from tick
96,000. The bounded-math GPU run executes all 97,219 steps in one continuous
kernel, including physical reset and body creation, with zero fallback calls.
Its 9,333,120-byte digest packet matches actual native execution. Recorded score
remains 4,393,101, level 100, chain 70; live score is separately 4,393,699.
These checks compare complete digest packets, not raw hidden native allocator
memory. The original replay is unchanged. The GPU record qualifier took about
139 seconds including per-frame hashes; it is a correctness run, not a batch
throughput benchmark.

## Optional bounded arithmetic

`IRISU_CUDA_NATIVE_BOUNDED_MATH=ON` is an experiment for the current qualified
physics graphs. It removes redundant add/sub/mul binary64 range checks on the
device while preserving the same separately rounded operations. Host arithmetic
stays checked. F32 store checks, division/sqrt domains, trig qualification,
sticky failure, structural guards and strict floating-point flags remain.
The checked implementation stays the default.

This is not a generic unchecked `Context`: the finite F32 store-to-store graph
bounds are essential. See the [contract and counterexample](../tools/cuda-native/bounded_math_contract.md).
Both numerical modes pass the 1,072-case GPU boundary qualifier, including 23
rejections. The opt-in profile also passes native collision, solver, factory,
dynamic world, game and search fixtures, with actual-device memory checks.
Build it in a separate directory using the build options above plus this flag.

## Further GPU efficiency

The next profile adds exact merge ranking, active-state checkpoint transport,
parallel raw prefix guards, and aligned body-reference relocation. It retains
all 512 body/shape slots and 4,096 contact slots. The original implementations
and the checked defaults remain available.

The additional 2x target is met with checked CPU preparation of the shared
prefix and GPU execution of every future. Pure-GPU execution reaches 1.956x.
Matched full-process medians on V100 are:

| Mode | Previous GPU seconds | New seconds | Speedup |
| --- | --- | --- | --- |
| All physics on GPU | 4.618 | 2.361 | 1.956x |
| CPU prefix, GPU futures | 4.622 | 2.171 | 2.129x |

Each comparison has three runs per configuration, alternates order and includes
setup, allocation, prefix, all 20 batches, output and teardown. Both execute the
same 12,288 distinct four-tick futures per batch: 983,040 future ticks plus one
100-tick prefix. The extra speedup is a batched-search result. Full device
allocations remain about 27.85 GB; reduced transport is not reduced reservation.

The selected profile passes the complete 97,219-step native record, eight
native search/continuation modes, 12,288 independent physical snapshots and
memory/init checks. The original eight CPU suites pass. The compact plan's
unused descriptor bytes are initialized before diagnostic host copies; the
final sanitizer runs report zero errors and emit native-identical packets.

Build the selected profile in its own directory:

```sh
cmake -S . -B build-cuda-native-efficiency-opt \
  -DIRISU_BUILD_CUDA_NATIVE_EXPERIMENTS=ON \
  -DIRISU_BUILD_CUDA_EXPERIMENTS=OFF -DIRISU_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DIRISU_CUDA_NATIVE_BOUNDED_MATH=ON \
  -DIRISU_CUDA_NATIVE_BOUNDED_STORE=ON \
  -DIRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC=ON \
  -DIRISU_CUDA_NATIVE_CACHED_ROTATION=ON \
  -DIRISU_CUDA_NATIVE_INLINE_STEP=ON \
  -DIRISU_CUDA_NATIVE_WIDE_REBIND=ON \
  -DIRISU_CUDA_NATIVE_GAME_FAST_COMPILE=min \
  -DIRISU_CUDA_NATIVE_MAX_REGISTERS=0 \
  -DIRISU_CUDA_NATIVE_F32_STORE=OFF \
  -DIRISU_CUDA_NATIVE_SPARSE_RESET=OFF \
  -DIRISU_CUDA_NATIVE_ENTRY_INDEX=OFF
cmake --build build-cuda-native-efficiency-opt \
  --target irisu-cuda-native-search-runner -j 2
```

`BOUNDED_STORE` removes only the redundant binary64-normal check inside the
previously audited device graphs. Float conversion, finite-F32 rejection,
sticky failure, division/sqrt/trig checks and rounding remain. Host/free stores
stay checked. This profile excludes arbitrary raw-double store calls; see the
[bounded-store contract](../tools/cuda-native/bounded_store_contract.md).
Rotation reuse requires the factory-created canonical matrices described in
[its contract](../tools/cuda-native/cached_rotation_contract.md).

`--compact-clone` restores every live Game value, occupied physical record and
complete SAP history. Allocation overwrites omitted inactive pool payloads;
pointer tails are zeroed before rebinding. Runtime allocations retain their
full sizes. Selected leaves still receive complete physical copies and undergo
independent continuation checks outside timing. See the
[checkpoint contract](../tools/cuda-native/compact_snapshot_contract.md).

The full cached source remains immutable and every raw source byte, including
scratch, is checked after each batch. `--parallel-prefix-guard` parallelizes
that comparison; it does not replace it with a hash. Wide relocation preserves
all capacity slots and existing pointer validation, using CUDA-global aligned
u64 transport; [qualification](../tools/cuda-native/body_ref_contract.md) covers
early-return partial writes and malformed references.

The two execution modes must be distinguished. Without `--host-prefix`, all
physics executes on GPU. With it, the shared 100-tick prefix is explicitly
prepared on checked typed CPU physics; all 983,040 future physics ticks execute
on GPU. Both modes include prefix preparation and process setup in timing.
See [commands and measurements](../tools/cuda-native/search_README.md) and the
[efficiency receipt](../benchmarks/results/cuda-native-efficiency-2026-11-04.json).

## Performance interpretation

The first paired search campaign reached **6.46x** on the V100 versus the
native cached-prefix CPU controller. Both prepare one 100-tick prefix, then
execute 20 batches of 12,288 distinct four-tick futures: 983,040 future physics
ticks per process. GPU initialization, prefix and future physics all execute
on the device. External wall includes loading, allocation, output and teardown.

| Backend | Runs | Median seconds | Range seconds |
| --- | --- | --- | --- |
| Native, 8 workers | 3 | 30.101 | 29.823–30.830 |
| Native, 16 workers | 3 | 30.146 | 30.094–31.126 |
| GPU | 6 | 4.655 | 4.623–4.658 |

The conservative ratio uses the best CPU median after subtracting its internal
verification cost: 30.0627 / 4.65509 = 6.4580. GPU prefix-guard costs remain
included. The final 16-worker sample saw increased CPU pressure; retaining it
does not determine the winning CPU denominator. Even the fastest individual
CPU sample, with verification removed, exceeds the slowest GPU sample by 6.39x.
Every output matches the same complete 3,194,896-byte native packet; every batch
executes its futures and compares its results and ranks internally.

This uses opt-in bounded arithmetic, aligned 64-bit transport, prefix reuse and
physical snapshots. Snapshots preserve every Game, World and broadphase byte,
including warm impulses, while omitting transient SceneScratch at completed
call boundaries. Poison, independent continuation, memcheck and virgin-scratch
initcheck gates qualify that omission. Timing skips only the extra selected-state
continuation checks, which pass separately. Existing implementations and
defaults remain intact.

See the [search commands](../tools/cuda-native/search_README.md) and
[final receipt](../benchmarks/results/cuda-native-final-campaign-2026-11-04.json).
This is a large batched-search result, not a universal 5x claim: small cold
searches still favor CPU, and a single game has not demonstrated this speedup.
The run allocates about 27.85 GB of device storage. The backend is experimental.

A million independent circle pairs are a component benchmark, not a game or
search benchmark. The native denominator uses persistent pthread pools with
PC53 set in each worker, and identical input/output bytes. CUDA event samples
measure resident collision work separately from process setup and I/O.

The initial matrix measured 2.397216 ms on V100 versus 10.089557 ms for the
fastest native CPU pool (8 threads): **4.2089x for the collision stage**. Whole
process medians were 0.235628 s GPU versus 0.220160 s for the fastest CPU
configuration (16 threads), so GPU process wall time was about 7% higher.
All runs matched 60,000,000 native output bytes. These component figures are
separate from the complete search campaign above.

Scene qualification hashes every frame; its kernel timings include validation
instrumentation. Performance runs must keep complete worlds resident, omit
per-frame qualification hashes from the timed physics path, and compare the
same work against actual native CPU execution.

## Capacity512 follow-up

The next 2x target was not reached. A separate optional 512-contact profile
improved the preceding hybrid campaign from **2.167 to 1.831 seconds** in
three paired runs: **1.184x**, with the same 983,040 GPU future ticks and one
shared 100-tick CPU prefix. Allocation fell from 27.85 GB to 13.40 GB.
Default capacity remains 4096; the smaller profile explicitly rejects overflow.

The full 97,220-frame GPU replay, dynamic physical/SAP state, search and selected
continuation matrix, compact snapshots, memcheck and scratch initcheck all pass.
The selected profile retains exact native output. Concurrent batches, an
interleaved solver, candidate ordering and alternative math remain optional
experiments; none improved enough to select. See the
[build and experiment guide](../tools/cuda-native/efficiency2_README.md) and
[paired receipt](../benchmarks/results/cuda-native-efficiency2-2026-11-04.json).
