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

## Performance interpretation

A million independent circle pairs are a component benchmark, not a game or
search benchmark. The native denominator uses persistent pthread pools with
PC53 set in each worker, and identical input/output bytes. CUDA event samples
measure resident collision work separately from process setup and I/O.

The initial matrix measured 2.397216 ms on V100 versus 10.089557 ms for the
fastest native CPU pool (8 threads): **4.2089x for the collision stage**. Whole
process medians were 0.235628 s GPU versus 0.220160 s for the fastest CPU
configuration (16 threads), so GPU process wall time was about 7% higher.
All runs matched 60,000,000 native output bytes. These figures do not establish
a full-game or search speedup, or the requested 5x target.

Scene qualification hashes every frame; its kernel timings include validation
instrumentation. Performance runs must keep complete worlds resident, omit
per-frame qualification hashes from the timed physics path, and compare the
same work against actual native CPU execution.
