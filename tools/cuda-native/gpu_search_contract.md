# GPU-first resident search

`gpu_search_runner.cu` is a separate consumer. Existing scalar and hybrid
executables, candidate/Result wires, native oracle, and published replay remain
available. GPU-first execution performs initialization, physical reset, original
prefixes, shared partial trajectories, future physics, selection and ranking on
the device. Host work is input/planning/dispatch/output and validation; cold host
objects provide empty storage without evaluating physics.

## Roots, jobs and execution slots

The new reader admits 1–131,072 candidates and at most32 roots. Logical
candidate `i` belongs to root `i % roots`, so `N` means `N` total jobs, not
`N` jobs per root. Root seeds and prefixes are the replay seed/prefix plus
`root * --root-seed-step` and `root * --root-prefix-step`. Overflow and prefixes
outside the replay are rejected. Root parameters are immutable per process.

Logical jobs and Results retain original candidate IDs. `--slots N` controls
the number of mutable World/Scene execution slots independently of frontier
size. Wave dispatch restores the appropriate root/checkpoint into each slot,
executes the remaining future, and preserves global topK states before slots
are reused. Global ranking uses the original stable candidate-ID tie break.

`--order-by-depth` changes dispatch order, grouping by remaining horizon, root
and checkpoint. A permutation and its inverse preserve original Result IDs and
translate global ranks back to physical slots for state retention. It changes
neither inputs nor ranking semantics.

`--adaptive-shared-prefixes` uses the existing exact sparse-input planner for
each root. Partial checkpoint physics always executes on GPU. With
`--reuse-checkpoints`, partial states are prepared once per immutable root-bank
load and retained across batches. Every leaf still executes its remaining
future on each batch; Results are not cached. Without that flag, partial
checkpoints are freshly prepared each batch. Counters distinguish logical
ticks, actual GPU future ticks, and checkpoint preparation ticks/executions.
Original prefix preparation is separately timed and occurs once per root.

Every root and partial checkpoint has a saved raw World/Scene/carry-state copy.
Every batch checks every source byte, including scratch and inactive storage,
after futures and validation. Source initialization makes these raw checks
compatible with initcheck; mutable execution scratch follows the existing
qualified physical-snapshot contract.

## State retention and validation

Single-wave execution ranks all Results once, then copies the selected physical
states in parallel. Multi-wave execution maintains double-buffered topK state
banks before reusing any source slot. `keep` can exceed the active slot count;
empty selections remain sentinels until enough jobs complete.

The optional `--persistent-queue` kernel claims work in warp-sized groups from
a device atomic cursor. A separate initialization flag for every physical slot
handles reuse across batches even when different worker warps receive work.
Each warp preserves its competitive topK states before claiming another group.
Only lane0 takes the bounded device-scope acquire/release retention mutex;
other contenders occupy different warps. Queue mode retains actual states,
rather than reconstructing states after evicting all original results.

Execution errors and completed-job counts are checked before global ranking.
Retained IDs are checked against GPU global ranks before any reference-state
indexing. With selected-state checks enabled, every retained candidate is then
independently replayed from its original GPU root. Complete Results, normalized
live Game values, occupied physical state and full SAP history are compared,
followed by independently poisoned-scratch continuations. Existing compact
snapshot allowances for inactive payload and listed live ABI padding remain.

## Optional device execution experiments

- `--solver-mirror` binds the persistent solver workspace to the execution slot
  pool when `IRISU_CUDA_NATIVE_PERSISTENT_SOLVER` is compiled. Root, cached and
  reference scenes outside that pool use the existing device solver.
- `--cooperative-physics` requires `IRISU_CUDA_NATIVE_PARALLEL_PHASES`. A64-thread
  CTA gives one controller the existing game future and one worker warp the
  independent collision/integration/bounds work. Canonical mutations commit in
  native order. It is separate from the persistent queue mode.
- `--collision-cache-mib N` configures the immutable exact collision table when
  `IRISU_CUDA_NATIVE_COLLISION_CACHE` is compiled. Capacity rounds down to a
  power of two entries; actual entries/bytes are reported. Hits require full
  key equality. Busy/full entries use original device calculations.
- `--diagnostic-counters` enables activation/hit/fallback/service counters.
  Diagnostic atomics and downloads perturb timing; benchmark modes leave them
  off. Counter names and scope are explicit in metadata.

## Memory admission and qualification

Memory admission queries actual free VRAM after fixed allocations and admits
execution slots only after charging World/Scene storage, optional mirror and
worker channels,512MiB default reserve, and a conservative whole-device stack
upper bound. That bound can double-count already reserved driver stacks; it
deliberately rejects borderline requests. Explicit oversized requests fail,
while automatic slot count decreases within the memory budget. Stack settings
require their own real-device parity and memory gates.

`gpu_search_specs.py` generates a wide distinct menu without changing the old
16,384-candidate reader. `gpu_search_native_reference.py` partitions candidates
by root and native chunks of at most16,384, checks the actual mapped native DSO
against its SHA pin, keeps all256-byte Results, and composes canonical global
ranks. Its chunk partition validator compares7-versus64 candidate partitions,
multiple roots and trace/final-only packets. The host planner validator covers
2,000 shared/unshared plans, job mapping, effective-prefix equivalence, dispatch
bijection, work accounting and overflow rejection. Full device/native packet,
retained-state, continuation, memory and timing gates are separate requirements
for selecting any execution mode.
