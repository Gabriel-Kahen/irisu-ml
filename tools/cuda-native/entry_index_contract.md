# Optional adapter entry lookup index

`IRISU_CUDA_NATIVE_ENTRY_INDEX` defaults off when undefined or zero. Existing
linear lookup, adapter layout and execution remain unchanged in that profile.
The opt-in profile adds a1024-slot open-addressed index keyed by BodyId and
returning an index in the unchanged `entries_` vector. It changes no iteration,
construction, synchronization, contact or solver ordering.

Reset initializes stamps once. Successful create inserts its appended index.
Pending-delete marking changes no vector indices. After actual erase shifts
entries, a generation increment invalidates the old map and rebuilds it in
vector order; no erase means no rebuild. Generation wrap clears all stamps.
Duplicate insertion preserves the first entry, matching old linear find.
Full-table failure reports capacity; at most512 vector entries use1024 slots.
Const and mutable lookup use the same table and validate the returned live
index and BodyId before accessing its entry.

The index is persistent adapter state, included automatically among compact
Game scalar spans, and retained by full snapshots. Profiles have different
Game sizes; raw Game snapshots are profile-local. Reset must precede lookup,
as with the adapter's existing initialized scene requirement. No external
entry-vector mutation is admitted: the vector is private, and its three
mutation paths (clear, append, stable erase) maintain the index explicitly.

`entry_index_validate.cpp` independently compares index results to first-match
linear lookup through dense and colliding IDs, duplicate insertion,1000 delete
and rebuild cycles, epoch wrap, reset and explicit full-table rejection.
Full Game/native Frame and compact checkpoint qualification remains necessary;
this helper test alone makes no GPU or performance claim.
