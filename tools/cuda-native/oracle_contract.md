# Independent native stage oracle

This is a new experimental test path. The admitted CPU backend, translated CUDA
backend, their pins, and historical results remain separate. The target is the
actual MSVC9 r58 behavior in the pinned ELF32 library, not a recompiled C++ model.

## Authority and callable boundaries

`oracle_native.h` hashes the held native file with SHA-256, requires
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`, loads that
file descriptor, reads its ELF32 local symbol table, checks each requested symbol
and offset, and combines it with the actual loader base. No library bytes are
patched. Hidden methods use the i386 MSVC `thiscall` ABI; collision functions are
cdecl. Calls run in a fresh, single-threaded oracle process with x87 PC53 and
round-nearest. Keep all native pointer/layout handling inside the ELF32 oracle.

| Native stage | ELF offset | Calling convention / state |
|---|---:|---|
| Circle/circle narrow phase | `0x45f0` | cdecl, manifold + two circle shapes + conservative flag |
| Polygon/circle narrow phase | `0x4730` | cdecl, manifold + polygon + circle + conservative flag |
| Polygon/polygon narrow phase | `0x56e0` | cdecl, manifold + two polygons + conservative flag |
| Contact solver constructor | `0x6e20` | thiscall, contact pointer array/count + stack allocator |
| Solver destructor | `0x72e0` | thiscall |
| PreSolve / warm start | `0x72f0` | thiscall, solver constraints and bodies |
| Velocity constraints | `0x7610` | thiscall, sequential constraint order |
| Position constraints | `0x7b60` | thiscall, float beta, bool result |
| PostSolve / impulse writeback | `0x7510` | thiscall, constraint manifold references |
| Island Solve | `0xa690` | thiscall, time-step pointer + gravity reference |
| Island UpdateSleep | `0xaa00` | thiscall, float dt |
| World Step | `0x13dc0` | thiscall, float dt + iteration count |
| Stack allocator ctor / dtor | `0x13460` / `0x13480` | thiscall |

The source aid is `third_party/box2d_legacy/Source`, especially
`Dynamics/Contacts/b2ContactSolver.*`, `Dynamics/b2Island.cpp`, and
`Collision/b2CollideCircle.cpp`. Native disassembly and independently captured
output bits remain authoritative for operation ordering and rounding stores.
The bridge source is `reference/native-box2d/multiworld/box2d-wrapper-msvc.cpp`;
its handle is `{world pointer, contact cursor, float magnification}`.

## Circle wire v1

All integers and float words are little-endian. The 16-byte input header is
`{0x31434e49, 1, case_count, 22}` as four uint32 values. Each 88-byte case is:

- Six binary32 words: `x1,y1,radius1,x2,y2,radius2`.
- One uint32 conservative flag, exactly zero or one.
- Fifteen raw words containing the initial complete 60-byte manifold.

Both native and typed runners emit exactly 60 raw manifold bytes per case.
Metadata/statuses go to stderr. Native circle shape storage is 76 bytes:
position at byte 24/28 and radius at 72. All other shape bytes use sentinels and
must remain unchanged. A manifold has two 24-byte points, normal at 48/52, and
point count at 56. Within each point, position is at 0/4, separation 8,
normal/tangent impulses 12/16, and contact ID 20.

For a contact, only first point position/separation/ID, normal, and point count
may change. For a disjoint pair, only point count may change. The corpus retains
warm-impulse and second-point sentinels and compares every output byte. Cases
cover touching and neighboring float values, epsilon-distance branching,
coincident centers, signed zero, subnormal scales, conservative separation, and
seeded random valid geometry. Arithmetic-overflow and nonfinite domains require
an explicit separate admission contract before adding them to typed parity.

Run `oracle_collision.py` with `--oracle`, `--library`, and `--output`; optionally
add `--runner` and `--gpu`. A runner consumes `cases.bin [--gpu]`. Generation
uses seed 661003 and 4096 random cases by default; 84 boundary cases are added.

## Solver and world snapshot contracts

A native solver is 12 bytes: allocator pointer at 0, constraints pointer at 4,
count at 8. Constraint size is 120 bytes; its two points are 44 bytes each.
Native bodies are 136 bytes. Physical offsets include position 4/8, rotation
12, rotation matrix 16..28, prior position 32/36 and rotation 40, velocity 44/48,
angular velocity 52, force 56/60, torque 64, center 68/72, inverse mass 108,
inverse inertia 116, damping 120/124 and sleep timer 128. Stable array indices
replace pointers in host64/GPU payloads; source ordering must be recorded.

Real contacted scene data can feed the original solver constructor and each
stage without patching the DSO. A contact array collected in world-list order
is a valid independent stage fixture; it is **not** proof of the island order
used by a complete native World Step. Game-engine work owns no-contact island
integration/sleep probes; solver work owns contacted-stage fixtures.

Admission is layered: exact stage snapshots first, complete ordered world
steps next, then continuous full-game/replay and search parity. Kernel-only
throughput cannot establish world or search speed. Setup, transfers, branching,
active/terminal work, ordering, and retained snapshots belong in later complete
CPU/GPU comparisons. No stage receipt alone authorizes production promotion.

## Geometry wire v2

`collision_wire.hpp` freezes geometry independently of mutable `ShapeState`
metadata. Header words are `{0x32534e49,2,count,98}`. Each 392-byte case contains
two 164-byte `ShapePacket` values, the fifteen initial manifold words, and the
conservative uint32 flag. Shape words are kind, vertex count, position[2],
column-major R[4], radius, vertices[16], normals[16]. Circle kind is zero and
polygon kind one. Polygon/circle puts polygon first; reversed order is not
silently swapped. Output remains raw 60-byte manifolds per case.

The native-only adapter maps polygon vertices to byte 104, vertex count to 232,
and normals to 236 in a 300-byte native shape. It checks all native shape bytes
remain unchanged. Accumulated impulse words remain sentinels; unused output
point fields can have path-specific native behavior, so complete equality is
required rather than imposing an invented zero-fill rule.

Native contact nodes have **other body first, contact pointer second**. The
first node fields are other16/contact20/prev24/next28; the second node fields
are other32/contact36/prev40/next44. This follows `b2Contact.h` and native World
Step. An early coordination message reversed the first two pointers; no oracle
code used that ordering, and scene importers must use the corrected mapping.

## Circle stage benchmark

`benchmark_collision.c` calls the actual native cdecl stage from persistent
pthread workers. Each worker installs PC53 and round-nearest. Immutable native
shape inputs are prepared once, then each measured pass resets all 60 manifold
bytes and invokes the original function. There is no per-case fork/RPC and no
extra validation pass inside the timed process; the independent oracle checks
shape immutability before timing. CPU workers 1/8/16 and GPU receive identical
geometry/manifold packets. `benchmark_collision.py` checks every output byte
on each warmup/repetition and reports process wall separately from stage time.
Native stage timers include barrier dispatch; GPU kernel events exclude driver
startup and transfers. These differences are explicit and cannot support a
complete world/search speed claim.

## Sweep-and-prune operation oracle

`oracle_broadphase.cpp` calls the pinned native constructor and public native
methods directly. The operation packet is defined by `broadphase_wire.hpp`:
32-byte header `{0x3153424e,1,operation_count,8,worldAABB[4]}`, then 32-byte
operations `{kind,logicalSlot,userIdOrQueryLimit,aabb[4],reserved=0}`. Kinds are
create1, move2, destroy3, commit4, query5. Kind6 sets timestamp65535 as an explicit
private-state precondition for testing the next wrap transition.

Each result is five uint32 values `{ordinal,result,queryCount,eventCount,23823}`,
then ordered query user IDs, ordered four-word callbacks `{add1|remove2,user1,
user2,pairToken}`, and all 95,292 native state bytes. Only the initial two
PairManager owner/callback pointers are zeroed. Other user-data pointers are
fixture scalar IDs/tokens, so there are no address-dependent values to mask.
Callback vtable slots are the MSVC deleting destructor, PairAdded, PairRemoved;
GCC's ordinary virtual-class ABI is not substituted.

Actual native snapshot evidence establishes proxyPool0x12014 (16-byte stride),
freeProxy0x14014, **bounds0x14016**, queryResults0x17016, queryCount0x17418,
worldAABB0x1741c, quantization0x1742c, proxyCount0x17434, timestamp0x17438.
Bound elements have alignment2; two padding bytes occur *after* queryResults,
not before bounds. An early alignment inference placed bounds at0x14018; the
raw native comparison caught it before admission. Pair entries are 12 bytes,
4096 in number; proxy capacity is512. These are native ABI offsets, not a
suggestion to encode geometry using C++ `sizeof`.

Oracle backing storage is explicitly zeroed before construction. The original
PairManager does not initialize its buffered-count field; the later portable
source contains a safety patch. This initial condition must be stated rather
than treating uninitialized bytes as universal native semantics. Default
histories stay below64 simultaneously live proxies (at most2016pairs), keeping
the original native fixed-capacity implementation safe even for all-overlap
fixtures. Full512-proxy capacity needs separate qualification. Optional
`--null-callback` histories exercise zero callback tokens and need their own
typed comparison before claiming that edge case. Histories cover endpoint ties, pending
move/commit order, remove/reinsert, proxy free-list reuse, query limits,
clamping and timestamp wrap. Every raw state byte and event/query word is
compared, not only final pair sets or state hashes.

Callback fixture policy: PairAdded returns/logs token0 if either proxy user ID
is0; otherwise it returns the next positive token. PairRemoved must preserve
the recorded token, including0. A null returned token still represents a final
pair and must not suppress the later native removal callback. Normal corpus
proxy IDs are nonzero, preserving earlier captures byte for byte.

The opt-in `--capacity-512` fixture permits at most512 live proxies while
conservatively checking `pairCount+active<=4096` and
`bufferedCount+2*active<=4096` before native create/move/destroy calls. It
qualifies a disjoint512-proxy history with query limits, destruction and reuse;
it does not admit arbitrary dense512-proxy histories. Default64-proxy scope
remains unchanged. Invalid513th allocation is tested on the typed implementation
only, never by overflowing the original native fixed-size arrays.
