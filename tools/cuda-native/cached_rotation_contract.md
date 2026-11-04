# Optional cached rotation matrices

`IRISU_CUDA_NATIVE_CACHED_ROTATION` defaults to zero. It requires an
engine-generated scene in which each accepted body's matrix is exactly
`{cos(angle), sin(angle), -sin(angle), cos(angle)}`, including signed zeros.
Factories and successful origin setters establish this invariant. Integration
and position correction maintain it. No external inconsistent finite matrix
may be injected into an admitted cached scene.

The shortcuts retain every position/angle calculation and binary32 store:

* Integration reuses R only when the stored new angle has the same **bits** as
  the old one and the angle lies in the original supported sincos domain.
* Position correction applies the same rule separately to each body's update.
* Shape synchronization reuses current R for the previous pose only when
  rotation0 and rotation have identical bits and the same domain check passes.
* The final island recomputation is redundant: integration and each successful
  position update already established the current matrix, and PostSolve does
  not modify it. The supported-angle check remains on this path.

The support predicate is exactly the existing binary32 trig range,
`abs_bits(angle) < 0x49000000`; NaN, infinity and out-of-range values cannot hit
an unchanged-angle cache. A changed zero sign always recomputes the matrix.
No new stored state, body/scene layout, arithmetic reassociation or approximate
trig is introduced. FP exception flags are not part of this numeric-state API.

Direct `integrate_position`, `solver_position` and `sync_body_shapes` callers
retain `canonical_rotation=false` by default, even in an enabled translation
unit. Shape-free/fixed-pair standalone paths therefore preserve arbitrary-R
behavior. The full Scene/BroadPhase engine path explicitly certifies the
invariant. A successful origin setter certifies it immediately after computing
its new matrix.

`cached_rotation_validate.cpp` compares complete body/constraint bytes against
the unchanged false path for integration, position correction, signed-zero and
numeric rejection boundaries, and verifies the standalone default still
recomputes an arbitrary matrix. Whole native scene/game continuation and the
published-record gates remain required for the engine-level admission claim.
