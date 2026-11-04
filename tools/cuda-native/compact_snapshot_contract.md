# Experimental active checkpoints

This is an opt-in semantic checkpoint representation alongside the original
full snapshots. Adding its headers changes no execution default. Qualification
results must identify the representation and numeric profile independently.

`compact_snapshot.cuh` retains every occupied body, shape and contact at its
original pool index, the complete World scalar tail, and every BroadPhase byte.
Occupied pending-deletion bodies/shapes and deferred-removal contacts remain
included. Manifolds, warm impulses, cached rotations/geometry, intrusive links,
IDs, destroy lists, SAP buffers, hash buckets, allocator/free-list history and
errors therefore persist. The descriptor is invalid once source occupancy
changes; immutable prefixes can reuse it. Selected leaves require fresh plans.

Inactive World payloads are omitted. Allocation reads their `occupied` flag and
then assigns the entire new BodyState, ShapeState or ContactState before using
its physical fields. `clear_missing` initializes every inactive occupied flag,
including on virgin storage. Subsequent initialized, successful destinations
can instead use `clear_reachable`: walk active plus pending body lists, their
shapes, and the global contact list before replacing metadata. This alternative
requires every occupied slot to be reachable, as valid complete scenes require.
It is not a recovery operation for corrupted/partially failed scenes. Source
and destination must differ.

`compact_game_snapshot.cuh` preserves every Game byte except unused storage
above `count` in nine vectors: adapter entries/contacts/ordered and Simulator
bodies/groups/active-contact-pairs/scripted-origins/current-contacts/actor-refs.
It retains all live values, headers, counts, overflow flags, RNG, configuration,
terminal metadata, actor colors/cursors and counters. Game scratch live values
are retained here; their independent omission is not assumed.

The vector implementation traverses `[begin,end)` where `end=begin+count`.
Sort/find/erase operate on that live range. `push_back`/`resize` placement-create
new elements before exposing them. `assign_live` copies only live elements.
The generated group's `back()` use follows `push_back`; overflow preserves an
existing live last element and the overflow flag, so existing failure behavior
is retained. The sole capacity-wide consumers are body-pointer relocation in
the adapter's `ordered_` and Simulator's `actor_refs_scratch_`. Their omitted
tails are explicitly zero-written before calling unchanged `rebind`, including
for virgin destinations. Live pointers retain the original relocation checks.

Game equality compares preserved spans after equal scene bindings and pointer
normalization. Scene equality compares occupied flags, occupied records and
the entire metadata/SAP suffix. Neither claims equality for inactive payloads
or SceneScratch. Full-size runtime allocations remain: reduced transport alone
does not reduce VRAM reservation. Batching/reusing runtime lane storage is a
separate runner change, not implemented by these headers.

`compact_snapshot_validate.cu` reuses the independent actual-native 16-case,
192-step scene packet. It compares full and compact continuation before/after
each call with pre/post-operation checkpoint variants. Normal eight-iteration
steps check native hashes; an eight-then-zero-iteration variant checks typed
continuation with observed warm impulses. Witnesses require pending deletion,
origin changes and sleeping/frozen bodies. Both pool-scan and prior-list flag
clearing are exercised. Poison and virgin-scratch modes are separate runs.

`compact_game_snapshot_validate.cu` accepts a replay, actual-native 96-byte
digest rows, prefix and suffix length. A checked typed host prepares the
explicit prefix. GPU execution repeatedly makes a full reference checkpoint
and an active compact checkpoint, compares preserved Game/Scene state before
and after, and checks every suffix Frame against the actual-native capture.
Default mode poisons omitted Game payloads and SceneScratch differently each
checkpoint; `--virgin` leaves unused CUDA allocations uninitialized for
initcheck, while pointer tails/occupied flags are initialized as required.
Use prefix 100 plus later dense/warm/deletion points and inspect the emitted
witness counts; a successful early prefix does not prove those later cases.

These representations exclude mid-island checkpoints, direct solver-phase
resumption, external consumers of inactive bytes, and malformed object graphs.
All physical warm caches remain persistent. No timing claim follows from a
qualifier's runtime, which includes hashing, comparisons and synchronization.
