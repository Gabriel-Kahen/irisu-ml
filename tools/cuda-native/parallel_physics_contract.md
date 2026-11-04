# Optional parallel physics and exact collision cache

Both profiles default off. Existing scalar and hybrid implementations remain
available. These prototypes change execution and scratch storage, not physics
arithmetic, solver contact order, native commit order, or result formats.

## Worker service

`IRISU_CUDA_NATIVE_PARALLEL_PHASES` enables hooks in the typed world pipeline.
`parallel_physics::Descriptor` maps a contiguous pool of complete Scene objects
to separate `Channel` allocations. `World` must be the first Scene member.
Descriptors are translation-unit local and must be configured in the same
CUDA translation unit as their consumers. Out-of-pool, inactive, or disabled
stages select the original implementation.

Each cooperative world uses a 64-thread CTA. Thread0 calls `begin_slot`, then
all64 threads execute one block barrier. Thread0 executes the existing game
future and calls `stop_slot` afterward. Threads32–63 execute `run_worker`;
threads1–31 have no work after the initial barrier. The dedicated worker warp
avoids divergent warp barriers and lets the controller retain the original
synchronous game/physics interface. This has a substantial residency cost and
is an experiment, not an assumption that more threads make the solver faster.

Requests publish their arguments through a block-scoped release epoch. Workers
acquire the epoch, write disjoint outcomes, and use an acquire/release completion
counter. The controller acquires completion before reading any outcomes. No
worker mutates canonical World/Scene/SAP state. Channels reset at every launch;
no prior stop command or transient plan is reused.

Parallel stages are:

- Narrowphase collision and polygon point-ID impulse matching, with transforms
  and sleep flags fixed. The controller commits manifolds and contact linking
  in original contact-list order. It commits the failing stage's partial output
  before returning its status and does not commit later calculated outcomes.
- Independent velocity and position integration for a unique island body list.
  Body results commit in original order through the first failure. Duplicate
  body lists retain the original sequential path.
- Shape QuickSync and swept-bound calculation after the position solver.
  Original rotation and per-island boundaries remain. The controller commits
  each shape in original order, honors current proxy removal/freeze state, and
  performs native-order SAP mutations. Unsupported old transforms and bounds
  retain the reference's destination-write behavior. Origin setters and static
  forced synchronization retain their original implementation.

Dependent warm starts and Gauss–Seidel velocity/position constraints stay
sequential. A configured persistent solver can own position integration;
parallel position integration then applies only to the original solver path.
Pure bounds still calculate after the persistent solver flushes its hot state.

`Channel` is external transient storage and does not enter snapshots or hashes.
Its leading `Stats` can be downloaded with a pitched copy, avoiding transfer of
all scratch outcomes. `bytes_required(slots)` reports its full reservation.

## Exact collision memoization

`IRISU_CUDA_NATIVE_COLLISION_CACHE` selects `collision_memo::collide_exact`.
A descriptor owns a bounded power-of-two entry array and optional sharded
counters. Clear the array only while no consumer is executing. Entries never
evict or change after publication; a full or busy table computes the original
collision stage. The cache can remain resident across distinct roots because
its inputs contain no world addresses or IDs.

The key serializes consumed value fields explicitly: shape kinds; circle
positions/radii; polygon position, rotation, count, and active vertices/normals;
all15 input-manifold words, including preserved unused point values; and the
conservative flag. It does not read ABI padding, inactive geometry arrays, or
unconsumed shape metadata. Unsupported structures bypass caching. Accepted
and rejected stage results retain the complete60-byte output manifold and
status/body fields.

A hash selects up to four probes, but acceptance requires complete key equality.
An entry changes from empty to owned through CAS, then becomes immutable through
a device-scoped release store. Readers acquire the published state before
inspecting payloads. Hash collisions never authorize a hit. Optional counters
record calls, hits, misses, insertions, busy probes, full probes, and bypasses;
disable them for uninstrumented timing.

## Qualification

`collision_memo_validate.cu` exercises actual-native circle, polygon/circle and
polygon/polygon captures, mutations of consumed fields, all input-manifold
words, conservative mode, numeric/structural rejections, and simultaneous
publication followed by warm reads. Capacity4 forces collisions/full-table
fallbacks. `--virgin-shapes` writes only consumed fields for all three collision
families before GPU execution, leaving inactive polygon vertices uninitialized
so initcheck can detect accidental reads of padding/cold fields. A deliberately
published entry has the correct hash and length but a different full key and
poisoned output; the qualifier requires rejecting it before genuine warm hits.

`parallel_physics_validate.cu` compares every dynamic native hash plus complete
persistent World/SAP and actor-map bytes, requires all four parallel stages to
execute, and compares rejection/partial-write state for synthetic failures.
Memcheck, initcheck, synccheck, active-descriptor full replay, and distinct-root
search qualification are separate gates. Component or host results alone do
not establish full-game admission or throughput improvement.
