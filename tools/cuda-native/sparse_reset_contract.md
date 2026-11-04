# Experimental sparse World reset

`IRISU_CUDA_NATIVE_SPARSE_RESET` is opt-in; undefined or zero retains the
original complete BodyState/ShapeState/ContactState assignments. The optional
branch changes only the pool reset loops: every `occupied` flag becomes false.
All existing World metadata reset statements remain identical, including
gravity, heads, counts, trace-ID counters, tick, sleep configuration, immediate
deletion mode and status. Bounds retain their previous behavior; Scene
initialization separately assigns bounds and reinitializes BroadPhase.

The profile retains contact capacity4096. It does not change arithmetic,
collision order, body/shape geometry, solver warm-start rules or layout. A
reset discards the old scene, including its old warm caches. New active cache
values are initialized by the existing allocation paths, not inherited from
inactive storage.

The acceptance argument is the same inactive-pool contract as active compact
snapshots:

- `World.create_body` reads the occupancy flag to select a free slot, then
  assigns the supplied complete BodyState before initializing its new links.
- `ScenePipelineState.create_body_shape` selects a false shape occupancy flag
  and assigns the complete freshly constructed ShapeState before proxy use.
- `WorldPairCallbacks.PairAdded` selects a false contact occupancy flag, assigns
  ContactState defaults, and establishes all new links/materials before use.
- Body/shape/contact hot traversal starts from World metadata or initialized
  live links. Reset empties every head/count, including the pending-delete
  head; old inactive link or manifold contents are unreachable.
- Validators/scene visitors test occupancy before reading a slot's payload.
  Compact descriptors likewise include only occupied slots. SAP reset is still
  the existing complete BroadPhase.Initialize, retaining its own native4096
  pair capacity and initialization algorithm.

The actual runner prepares fully initialized persistent Game/Scene storage
once. Its initialization reset0 then reset(seed) can reuse inactive payloads
without reading them; newly created boundary/actor slots are assigned again.
SceneScratch retains its independently qualified scratch-only contract.

This is semantic reset, not equality of every inactive object byte with the
default profile. Subsequent resets can leave old values and ABI padding in
inactive slots. Reject this profile for consumers that inspect inactive
payloads, demand a zero/default-filled pool dump, or reuse an inactive payload
without the existing allocation constructor. Such consumers must keep full
reset. Invalid/partial scenes are not repaired by retaining their old payload;
only reset metadata and subsequent admitted construction establish a new scene.

Qualification must compare actual native game frames and initialized physical
state under the explicitly reported profile, including repeated reset with
different seeds and dirty/pending scenes. Poisoned inactive payloads can test
semantic irrelevance; initcheck should match the runner's once-initialized
persistent storage and virgin SceneScratch. No performance improvement is
claimed by this source change before a coordinated measurement.
