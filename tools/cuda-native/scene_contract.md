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
retains the first, fixed-topology circle qualifier. The concrete
`ScenePipelineState` in `world_scene.hpp` now owns the typed world, integer
SAP/PairManager, and nonsemantic solver scratch. Its normal step runs typed
collision, contact-graph transitions, DFS islands, force and
damping integration, constraint construction, PreSolve, velocity iterations,
position integration, position iterations, PostSolve, shape synchronization,
force clearing, and island sleep. Warm starting and position correction use
the pinned wrapper's enabled defaults, and native-order pair commit. No joints
are admitted. Hot steps traverse live intrusive lists; full-pool validation
and comparison hashes are outside the normal step.

Only fixed-pair mode uses the conservative swept-pair guard and rejects pair
transitions. The normal scene path creates/removes pairs through the actual
typed SAP, preserves callback order, accepts partially out-of-range proxies
under native InRange semantics, and freezes/destroys proxies when needed.
Cold body/shape construction uses the independent factory. Native import is
oracle access, not a runtime fallback.

The scene API accepts world units. `initialize(bounds,gravity)` resets storage
and SAP; the caller creates the native shape-free ground body explicitly.
`create_body_shape` supports one shape per body, matching the Irisu wrappers.
`queue_destroy` defers reclamation until step cleanup. `set_velocity` stores
raw velocity without waking; `set_origin` includes the wrapper's unconditional
zero linear velocity, even when the frozen body ignores the transform.

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
fixed-topology digest, including unused manifold slots. Scratch arrays and
padding do not. Version2 appends polygon core/OBB and world bookkeeping; the
reader retains version1 support.

Dynamic native comparisons canonicalize allocation indices by body tags and
contact-list/edge order. They compare active initialized manifold points and
geometry, live broadphase endpoints/proxies/pairs, hash tables, free lists,
timestamps and active queries. Opaque uninitialized native allocator contents
are not a cross-backend equality requirement. Typed scene snapshots preserve
their complete own state, including pending SAP buffers, without rebinding.

The GPU qualifier hashes every step and scans fixed-capacity storage for
comparison. Its measured kernel time includes that instrumentation and is not
a performance benchmark. A passing small fixed-pair corpus is not full-scene,
full-game, arbitrary-input, or production training admission.
