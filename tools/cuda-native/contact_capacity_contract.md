# Optional typed contact capacity

`IRISU_CUDA_NATIVE_MAX_CONTACTS` defaults to4096. Explicit512 and1024 profiles
reduce only the independent typed World's ContactState pool and its contact
order/solver-contact/manifold/constraint scratch arrays and compact descriptors.
Body/shape capacity512, native SAP4096 pair/hash/buffer capacity and layout, and
all generated Game vector capacities remain unchanged. Existing CPU/translated
implementations are unaffected. Arithmetic, order, IDs and warm-start data in
admitted scenes retain the same algorithms and element layouts.

Capacity means occupied ContactState records, including broadphase contacts
with zero manifolds and deferred-removal records. It does not mean touching
points. WorldPairCallbacks.PairAdded scans this pool and reports sticky
Status::capacity when no free slot exists. Scene check after SAP commit rejects
the step; no accepted run drops a contact, approximates it or falls back. A
failed step can have partial state changes and must not be accepted. Pair
tokens and callbacks continue using stable original slot+1 IDs.

Raw World/Scene/plan snapshots are profile-local, and every host/device TU
sharing those objects must use the same capacity. The original scene_wire
v1/v2 also serializes an occupancy word for every pool slot; its encoded length
and scene_hash therefore depend on capacity. Do not decode default4096 fixed
scene packets in a512 profile. Dynamic SceneCase packets and their canonical
active-state/SAP digest,96-byte Game Frames, and search result/candidate wires
are independent of this capacity and can reuse the actual-native captures.

`contact_capacity_validate.cpp` runs a checked typed host replay, compares every
Frame against the actual-native capture, and reports contact occupancy at
public call boundaries. That observed typed peak is not a direct measurement
of hidden native occupancy or a transient allocator high-water mark. Successful
complete replay under the smaller cap proves admission for that replay because
any allocation overflow fails the run. It does not prove arbitrary512-proxy
worlds fit. Full-record GPU replay,3072 dynamic native scene frames, and compact
snapshot/rebind qualification are required before choosing the profile.

The capacity helper also exercises exhaustion explicitly and checks that
PairAdded returns no token and sticky Status::capacity. Performance and memory
claims require separate actual allocation and timing observations; predicted
sizeof changes alone are not a speed result.
