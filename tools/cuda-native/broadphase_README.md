# Independent native-order sweep-and-prune

`clone/cuda_native/broadphase.hpp` and `broadphase_algorithm.inc` adapt the
vendored r58 SAP and PairManager into pointer-free fixed arrays. The original
Erin Catto notice is retained; this is marked as altered source. The algorithm
uses no Guest, emulated registers or instruction dispatcher. The pinned MSVC
implementation has the same capacities: 512 proxies and 4,096 pairs.

`BroadPhase::Initialize(world)` establishes an empty state. `CreateProxy(aabb,
user,callback)` returns a proxy ID and commits immediately. `MoveProxy` buffers
changes; `Commit(callback)` finalizes them. `DestroyProxy(id,callback)` also
commits immediately. `Query` returns proxy user values in native query order.
Callbacks provide `uint32_t PairAdded(uint32_t,uint32_t)` and
`void PairRemoved(uint32_t,uint32_t,uint32_t token)`. A returned zero token is
valid and is retained for the eventual removal. User values and tokens are
opaque pool handles. The scene integration owns their lifetimes and filtering.
Check sticky `error` after each operation; a failed state, including any partial
writes, must not be accepted. There is no approximate fallback.

State is trivially pointer-free and copyable; no rebind is needed. Its first
95,292 bytes mirror the actual native layout, with owner/callback words zero.
The trailing `Status error` is outside that prefix. Native bounds begin at
`0x14016`, query results at `0x17016`, with alignment padding before query count
`0x17418`. Imported native continuations must preserve endpoints, freelists,
pair/hash/buffer state and timestamps, and normalize active shape/contact
pointers into handles. Recreating proxies in array-slot order is not an exact
substitute. New scenes must create proxies in original shape-creation order.

Quantization follows the native instructions: world extent subtraction is
stored to binary32 before division and binary32 quantization-factor storage.
`ComputeBounds` clamps binary32 coordinates, subtracts and multiplies at PC53
without an intermediate binary32 store, then truncates to int32 and applies the
native uint16 lower/upper masks. The boundary fixture includes 32 cases that
would differ if that subtraction were incorrectly rounded to binary32.

`broadphase_generate.py` emits the order-sensitive integer method bodies from
the vendored source; numeric adapters, state and pair management are explicit
in the header. `broadphase_edges.py` creates boundary/cancellation fixtures.
The shared native oracle and wire are owned by the oracle workstream.
`broadphase_runner.cu operations.bin [--gpu]` emits ordered events, queries and a
full normalized state after every operation. Its GPU history kernel uses one
thread for dependent SAP operations and its block for snapshot transport; this
is a correctness harness, not a throughput benchmark. `broadphase_domain.cpp`
checks capacity rejection, full pair-buffer reuse/removal and numeric errors.

The native oracle starts from calloc-backed memory because the original native
PairManager constructor leaves its buffer count uninitialized. Exact raw-state
claims use this explicit defined starting condition. The typed initializer
sets that count to zero. `broadphase_qualification.json` records the bounded
native/CPU/GPU and sanitizer gates; it makes no whole-game or speedup claim.
