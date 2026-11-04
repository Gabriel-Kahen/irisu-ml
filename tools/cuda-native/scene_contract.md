# Independent typed scene stage

The new `irisu::cuda_native` implementation is separate from the old Guest
backend. It reuses numeric primitives, not the register machine, translated
instructions, Guest allocation, or old PhysicsWorld adapter.

`World` contains fixed-capacity typed bodies, shapes, and contacts. Body and
contact lists use indices in native head-insertion order. Stable body IDs do
not determine iteration order. Deferred body removal unlinks immediately and
reclaims storage at the next cleanup stage. A copy needs no pointer rebinding.
`restore` checks the body/deferred lists before replacing the destination.

`step_shape_free` is an independent shape-free path. `step_fixed_circle_pairs`
is the interim circle pipeline. The latter imports native-created geometry,
mass/COM, caches, and initial contact topology before entering the hot path.
It then runs typed collision, contact-graph transitions, DFS islands, force and
damping integration, constraint construction, PreSolve, velocity iterations,
position integration, position iterations, PostSolve, shape synchronization,
force clearing, and island sleep. Warm starting and position correction use
the pinned wrapper's enabled defaults. No joints are admitted.

Pair creation/removal and out-of-range shapes return `bridge_required` and
invalidate the advanced lane. The guard checks the complete eligible circle
pair set against swept AABBs. It is intentionally conservative; it does not
stand in for integer broadphase state, proxy ordering, or PairManager callback
ordering. Native import is initialization, not a per-step fallback. Native
shape construction, general broadphase, polygon scene steps, the Simulator
adapter, and full-game parity remain separate qualification requirements.

The operation graph was checked against the pinned MSVC methods:

| Method | Native entry | Stores relevant to this stage |
| --- | ---: | --- |
| b2World::Step | 81344 | cleanup, collide, native list order, DFS, sleep, commit |
| b2Island::Solve | 42640 | float vector temporaries before velocity/position additions; angular chains retain PC53 intermediates |
| b2Island::UpdateSleep | 43520 | float tolerance squares and dot temporary; island minimum controls sleep flags |
| circle QuickSync | 72320 | cached matrix copied; rotated local vector stored before adding body position |
| polygon QuickSync | 72544 | same store graph, local centroid instead of circle local position |
| circle Synchronize | 74448 | cached geometry followed by swept AABB construction |

Native Body is136 bytes. Position is at4/8, rotation12, matrix16..28,
previous transform32..40, velocity44/48, angular velocity52, force56/60,
torque64, center68/72, inverse mass108, inverse inertia116, damping120/124,
sleep time128. Native circle shape is76 bytes, with cached matrix8..20,
position24/28, body40, material44/48, proxy/filter56..62, local position64/68,
and radius72. Contact is128 bytes including its60-byte manifold at68.
Contact-node1 has **other16, contact20, prev24, next28**; node2 starts at32.
These pointer fields are mapped to indices outside the device hot path.

`scene_wire.hpp` defines a versioned stream of semantic32-bit words. It does
not use `sizeof(ShapeState)` as a wire contract. All active body fields, shape
caches, contact-list/edge indices, and all60 manifold bytes enter the per-step
digest, including unused manifold slots. Scratch arrays and padding do not.

The GPU qualifier hashes every step and scans fixed-capacity storage for
comparison. Its measured kernel time includes that instrumentation and is not
a performance benchmark. A passing small fixed-pair corpus is not full-scene,
full-game, arbitrary-input, or production training admission.
