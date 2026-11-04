# Interleaved solver workspace experiment

This optional layout keeps independent worlds in separate SIMD lanes and
preserves each world's serial contact and point order. It does not parallelize
dependent Gauss-Seidel impulses. The original `solver.hpp` routines remain the
arithmetic implementation; no numerical expression or binary32 store boundary
is rewritten.

The 48-byte prefix of `BodyState` contains precisely the fields consumed or
modified by these solver stages: position, velocity, rotation, angular
velocity, inverse mass/inertia, and rotation matrix. Workspace body words are
indexed `[field][island-local body][world lane]`. All 120 constraint bytes use
the analogous layout. Contiguous world lanes can therefore coalesce the same
field accesses even when complete Scene objects are megabytes apart.

Packing maps native body slots to dense island-local indices. A contact loads
its complete constraint and its two hot bodies, calls the original single
constraint stage, then commits every partial write before propagating an error.
Unpacking restores original body indices and every constraint byte. Other
BodyState fields remain untouched in their original allocation. Position-stage
convergence is the conjunction of each contact's unchanged threshold result,
which equals the original minimum-separation threshold.

`IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER` defaults off. Its optional bridge in
`world_pipeline.hpp` stages only the repeated velocity and position loops;
construction, warm starting, integration, collision, SAP, and game rules keep
their existing path. Host execution remains unchanged. A null descriptor,
world outside the configured pool, or island exceeding the configured 32-body
or 64-contact capacities selects the unchanged exact GPU solver. These are
explicit counted layout fallbacks, not approximate physics or CPU execution.
Contact-free islands and zero-iteration calls require no workspace.

The runner owns a separate global allocation of
`words_required(lanes,32,64)` uint32 words, initializes five zero uint64
counters, and uploads `Descriptor` to the consumer's translation-unit-local
`device_workspace` symbol. The descriptor records the base Scene address,
`sizeof(Scene)` stride and pool lane count. `World` must be the first Scene
member. Every concurrently executing world needs a distinct lane; workspace
and source arrays cannot alias. Prefix/selected allocations outside this pool
continue through the original solver. Counters report velocity islands,
position islands, capacity fallback, outside-pool fallback, and errors.

Structural pack validation occurs before source mutation. Invalid structural
inputs are rejected; their partial writes need not equal the reference's
partially executed invalid traversal. For valid structures, numeric-failure
partial writes are committed and compared by the qualifier. Workspace contents
are transient and do not enter snapshots. A failed stage must still reject
the world, as in the original solver.

`solver_interleaved_validate.cu` loads the existing native-captured stage
packets, compares repeated stages against the original solver, and checks every
BodyState and constraint byte plus complete SolverResult fields. Reversed
island body order exercises index remapping; untouched body bytes and inactive
slots carry poison. Additional finite-overflow and unsupported-angle fixtures
exercise partial failure writes. `solver_interleaved_domain.cpp` checks pool,
capacity, structural rejection and unchanged host bridge behavior.

These checks establish temporary-stage equivalence only. Full native game,
dynamic scene, snapshot/lifecycle, memory and real campaign gates are required
before this optional bridge is qualified. Coalescing is a hypothesis, not a
speedup claim; packing, register pressure and divergence can erase its benefit.
