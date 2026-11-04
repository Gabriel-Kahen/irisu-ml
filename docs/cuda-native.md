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
  irisu-cuda-native-factory-runner irisu-cuda-native-scene-runner irisu-native-domain -j 2
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

Factory comparison excludes allocation/proxy bookkeeping, qualified separately.
Broadphase normalization removes only native owner/callback pointer words.
Small stage proofs do not establish arbitrary full-world or full-game behavior.
The fixed-pair scene entry rejects topology transitions explicitly.

The checks and their native ABI contracts live in `tools/cuda-native`:

- [Collision and broadphase oracle contract](../tools/cuda-native/oracle_contract.md)
- [Solver contract and commands](../tools/cuda-native/solver_README.md)
- [Scene contract](../tools/cuda-native/scene_contract.md)

`irisu-native-domain` checks invalid geometry/numerics, native pair-order
requirements, preserved no-contact data, and transactional rejection.

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
