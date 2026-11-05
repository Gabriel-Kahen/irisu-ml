# Island-lifetime GPU hot mirror

`IRISU_CUDA_NATIVE_PERSISTENT_SOLVER` defaults off. The optional implementation
uses `clone/cuda_native/solver_persistent.hpp`; existing scalar and interleaved
implementations remain available. It changes storage and transport, not contact
order, floating-point expressions, rounded stores, or iteration counts.

The runner allocates `persistent_solver::words_required(lanes)` uint32 words and
calls `configure(Descriptor{words, scene_base, scene_stride, lanes, body_capacity,
constraint_capacity, counters})` in its own CUDA translation unit. A Scene's
World must be its first member. Default capacities retain all512 body slots and
the configured contact capacity. Counters are optional and should be null for
performance runs. An empty descriptor disables the mirror.

Workspace is tiled across32 independent world lanes. Body and constraint fields
are interleaved within each tile, while original reusable body slots and native
constraint ordinals stay unchanged. A small membership bitset validates island
participants in linear time; no dense-ID searches, quadratic duplicate checks,
or contact-by-body remapping is performed. The workspace adds allocation; it
does not replace or shrink canonical World/Scene storage.

The current boundary is one complete island after original constraint building:

1. Validate and copy only participating hot body fields and live constraints.
2. Run PreSolve, every velocity iteration, position integration, every required
   position iteration, and PostSolve while the hot mirror remains authoritative.
3. Flush the mutable body and constraint fields before returning to the original
   manifold copy, shape synchronization, SAP, sleep, and island-flag operations.

Allocation persists across waves, but hot contents are refreshed per island.
This does not claim authoritative SoA state across complete game ticks or direct
constraint construction into SoA. The old per-phase pack/unpack bridge is not
used. The numerical graphs are the unchanged generated48-byte body solver
specializations from `solver_hot.hpp`, with contact-local velocity anchor reuse
mechanically retained in `solver_persistent_velocity.hpp` so the experiment
does not discard the selected scalar optimization. Position integration retains
the exact graph and store/error boundaries from `world.hpp`.

Every numeric failure flushes the same partial body/constraint writes before
returning the original status. Integration errors preserve native body IDs.
PostSolve retains its partial manifold writes. Rejected layout preparation
modifies only private workspace; unavailable/out-of-pool/insufficient layouts
run the original GPU scalar path from unchanged canonical state. This is not
CPU, native-library, or approximate-physics fallback. Empty-contact islands
also retain the original path.

Canonical Game/World/Scene layouts do not change. All success/error returns
flush the mirror, so existing snapshots remain valid at completed call
boundaries. Scratch workspace is not transported with snapshots. Before reuse,
all membership bits and consumed hot fields/constraints are overwritten; unused
workspace payloads are never read.

`solver_persistent_validate.cu` compares native-captured PreSolve/velocity/position
inputs against the unchanged scalar graphs. It also composes a whole island
from the same numeric inputs, including position integration and PostSolve,
and compares all body/constraint/manifold bytes plus status/body diagnostics.
These compositions use synthetic PostSolve destinations and do not constitute
direct actual-native complete-world admission.

Host checks currently cover96 captured stages expanded to897 cases, including
sparse original body IDs, malformed participants/point counts, late structural
rejection, unsupported rotations/anchors, overflow, zero contacts/points, and
numeric partial failures. Iterations0,1,8,1000 and both canonical-rotation modes
pass. GPU/native full-world/replay, memory checking, and matched timing must be
qualified separately before any selection or speed claim.

```sh
runner=build-cuda-native-gpu-first/clone/cuda_native/irisu-native-persistent-solver-validate
cases=artifacts/gpu-native-20261103/bounded-solver/cases.bin
"$runner" "$cases" --gpu 8
"$runner" "$cases" --gpu --canonical 8
"$runner" "$cases" --gpu 0
```
