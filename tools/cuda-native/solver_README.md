# Independent typed contact solver experiment

`clone/cuda_native/solver.hpp` implements contact-constraint construction,
PreSolve, velocity iterations, position iterations and PostSolve on ordinary
`BodyState`, `ContactConstraint` and `Manifold` arrays. The hot functions contain
no Guest, emulated registers, FP stack or instruction dispatcher. The runner
links no translated-region library or native oracle library.

The algorithm derives from the vendored 2006–2007 Box2D source; its original
license is retained and the implementation is marked as altered. Operation and
binary32 store order follow the pinned MSVC instructions, rather than assuming
that compiling the original C++ expressions reproduces their evaluation.
`solver_math.hpp` shares the existing read-only numeric PC53/CORE-MATH helpers;
sharing these numeric functions does not invoke the translated physics engine.
Host callers require round-to-nearest. CUDA builds must retain strict division,
square root, subnormal and contraction flags.

The caller supplies contact order, body/manifold indices, material values and
array bounds. Constraint construction preserves point fields the native
constructor leaves untouched; PreSolve initializes the position impulse before
it is consumed. Constraints and their points execute serially within a world;
independent worlds can execute in parallel. A failed `SolverResult` invalidates
the stage, including any partial writes. There is no approximate fallback.
Position convergence is a separate `SolverResult::converged` value.

`solver_native_oracle.c` executes hidden methods of the actual pinned ELF32
library, resolved by symbol name and offset after SHA-256 verification. It
constructs real native circle/box/triangle contact scenes, invokes the native
virtual GetManifolds method and uses a fresh native stack allocator. It exports
pointer-free before/after packets for all five stages. Constructor scratch is
initialized to a known pattern, allowing untouched fields to be compared too.
These fixtures use contact-list order; they do not attest original island DFS
order. The separate whole-scene integration is responsible for that ordering.

`solver_runner.cu` evaluates the packets using the typed solver on CPU or CUDA.
`solver_validate.py` compares every output byte against native execution and
optionally runs Compute Sanitizer. `solver_domain.cpp` checks focused error and
signed-zero contracts. `solver_stage_bridge.hpp` is a host-only test adapter
that compares typed physical fields after each solver stage in existing
600-tick scenes; conversion/reference execution are outside the new hot path.
It is not linked by the standalone runner or the new scene implementation.

The tracked `solver_qualification.json` records the measured gates. Its timing
samples are small component diagnostics, not a whole-backend speedup claim.
