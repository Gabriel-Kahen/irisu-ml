# Cached and local solver experiments

The four opt-in profiles default off:

- `IRISU_CUDA_NATIVE_CACHED_VELOCITY` computes world-space velocity anchors once
  after PreSolve and reuses them throughout an island's velocity iterations.
- `IRISU_CUDA_NATIVE_LOCAL_VELOCITY` holds the two participants' linear and angular
  velocities in local variables for one contact, committing at contact exit.
- `IRISU_CUDA_NATIVE_LOCAL_POSITION` similarly holds position, angle, and rotation
  matrix locally for one position-solver contact.
- `IRISU_CUDA_NATIVE_CONTACT_ANCHORS` retains each point's world anchors between
  a contact's normal and tangent passes, using small local storage and no new
  SceneScratch allocation. It can combine with local velocity. When enabled,
  this path takes precedence over the whole-island velocity cache.

All profiles preserve each world's serial contact order, each contact's normal
and tangent passes, and every PC53 operation and explicit binary32 rounding
point. They do not parallelize dependent Gauss-Seidel updates. Generated headers
retain the original license and record the source `solver.hpp` hash; their
generators mechanically specialize that graph without altering expressions.
Strict CUDA floating-point build flags remain required.

Velocity anchors depend only on local anchors and body rotation matrices.
PreSolve and velocity iterations update velocities and impulses, leaving both
inputs unchanged. The cache is rebuilt for every island, after PreSolve and
before the first velocity iteration. Its `2*max_contacts` entries are transient
`SceneScratch` storage; only active point entries are read. Poisoned or unused
entries are not reused. Structural or numeric cache rejection occurs before
the original solver mutates any current-iteration state, then selects the
unchanged solver for the entire velocity phase. The cache helper must not be
reused after a position/rotation/local-anchor change.

The contact-local alternative creates each cached anchor during its original
normal pass, then reads it in the tangent pass. Velocity impulses cannot change
rotation matrices or local anchors. Its two-point loops use an explicit bound
of two after the original point-count validation; CUDA unrolls this fixed bound
to expose small register values. Inactive points break before any read. No
extra anchor arithmetic runs ahead of the original normal pass, preserving
numeric rejection and partial writes without a phase-level fallback.

Local stages validate distinct, in-range body indices before loading their
participants. Their small binding objects commit exactly the mutable body
fields on every exit, including numeric failure, preserving the reference's
partial-write state. Explicit `Context` stores still round to binary32 before
the next arithmetic operation even while values reside in registers. Each
contact commits before the next contact executes. Body and constraint storage
must be distinct, as in the original scene/solver APIs. These helpers do not
support observers or concurrent mutation within a solver call.

The host scene path remains unchanged. Direct host helper qualifiers compare
every body/constraint byte, status, failed constraint/point, and convergence
against the original stages. Native-captured cases cover the actual stage
inputs. Additional cases exercise zero points, malformed point counts, invalid
and identical body indices, late structural failure, unsupported anchors and
rotations, and overflow partial writes. Both cached and uncached local velocity
paths are checked; local position is checked with canonical rotation both off
and on. Stage equivalence alone does not establish whole-game/native admission
or campaign performance. Full native replay, physical-state/snapshot,
continuation, and GPU memory gates remain necessary for a selected profile.

Qualifier sources:

- `solver_velocity_cached_validate.cu cases.bin [--gpu] [--local] [--no-cache] [--contact] [iterations]`
- `solver_position_local_validate.cu cases.bin [--gpu] [--canonical] [iterations]`

`cases.bin` uses the existing native solver capture format. The default is ten
iterations. Every optimized mode must preserve accepted output and rejected
partial state; approximate comparison is not an admission option.

For a configured CUDA-native build with strict arithmetic and cached rotation,
the stage matrix is:

```sh
build_dir=build-cuda-native-efficiency-cap
cases=artifacts/gpu-native-20261103/bounded-solver/cases.bin
cmake --build "$build_dir" --target irisu-native-cached-velocity-validate irisu-native-local-position-validate -j2
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-cached-velocity-validate" "$cases" --gpu
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-cached-velocity-validate" "$cases" --gpu --local
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-cached-velocity-validate" "$cases" --gpu --local --no-cache
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-cached-velocity-validate" "$cases" --gpu --contact
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-cached-velocity-validate" "$cases" --gpu --contact --local
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-local-position-validate" "$cases" --gpu
env -u CUDA_VISIBLE_DEVICES "$build_dir/clone/cuda_native/irisu-native-local-position-validate" "$cases" --gpu --canonical
```

Removing `--gpu` exercises the same direct helpers on the host. This does not
activate them in the normal host scene pipeline. Ten-iteration host checks
cover 256 cases per mode, including 32 native-captured stages. Velocity modes
have 192 rejected cases; position modes have 193. Additional one- and
1,000-iteration checks exercise repeated updates and early exits. GPU and
whole-search receipts must be assessed separately before selection. Performance
measurements and trial rejection decisions belong in the campaign receipt,
not these stage-equivalence results.
