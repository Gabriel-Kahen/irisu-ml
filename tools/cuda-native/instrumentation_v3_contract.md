# V3 original-physics stage diagnostic

`instrumented_runner_v3.cu` preserves V1/V2 and all existing sources. It locally
forces `IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER=0`, preloads original physics
headers, and redirects later pipeline/adapter free-function calls to thin
perfect-forwarding wrappers. Preloaded originals' internal calls stay inside
their enclosing stage; they are not recursively redirected or double-timed.

The V2 thread-owned counter scheme remains: packed CUDA threads, default-stream
serial execution, no counter atomics, no selected continuation checks. Sixteen
phases use 4 MiB of counters; a separate 64 KiB thread-owned flag array scopes
stage accumulation to `Base::physics.step`. External setters therefore do not
pollute the physics denominator. Both arrays are cleared before runner entry.

Solver build/pre/velocity/position/post calls with zero input count execute
their original functions without timing or records. This preserves outputs
such as build's reset of its output count. Static-body integration and empty
sleep calls similarly skip timing while retaining original execution. Stage
call counts consequently describe nonempty/dynamic work, not all source calls.

The profile covers solver stages, narrow-phase collision, shape quick-sync and
circle/polygon swept bounds, velocity/position integration and sleep. Remaining
physics time includes island traversal, SAP, adapter work, untimed no-op loops
and observer overhead; it is not labelled pure broadphase time. Ratios are
per-thread elapsed SM-clock diagnostics, not wall-time or utilization fractions.
Nested private-counter updates still perturb outer intervals. Compare normal
and diagnostic rollout time before interpreting stage dominance.

Use checked host prefix, zero warmups, one repeat and skipped selected checks.
Normal native-comparable stdout remains unchanged; the final stderr JSON adds
cycle/call totals and fractions. Exact packet/call checks precede interpretation.
No build, GPU proof or performance admission is implied by this source contract.
