# Search cycle attribution diagnostics

These executables preserve normal Result/rank output, but are not benchmark
binaries. CUDA is preinitialized for counters, wrappers change compilation, and
clock reads/counter updates perturb execution. Native packet equality and exact
call counts must pass before interpreting any cycle ratios.

V1 (`instrumented_runner.cu`, commit `3b10a60`, SHA-256
`b08031e336c2e50ce0a94917cba1bd9225c287b62a24b0c99af5bc5fbe681976`)
used same-address global atomics. Its observed extra runtime can distort even
outer residual fractions. The unchanged source and an additional local copy at
`artifacts/gpu-efficiency2-20261104/instrumented-v1-preserved.cu` retain that
negative evidence. Do not promote V1 fractions to uncontended phase costs.

V2 (`instrumented_runner_v2.cu`) uses `counters[16384][8]`, one 64-byte row per
linear CUDA thread. Counter updates use ordinary `+=`, with no counter atomics.
The full 1,048,576-byte array is cleared once before the original runner entry,
copied back afterward, and summed on the CPU. Generated future kernels must use
packed one-world-per-thread execution and default-stream serial ordering.
`--isolated-warps` and `--cpu` are rejected, and `--skip-compaction-check` is
required. An unexpected out-of-range counter index traps rather than silently
dropping samples. The existing candidate limit is 16,384. Selected continuation
checks remain an independent correctness gate rather than part of this profile.

Use `--host-prefix --warmups 0 --repeats 1 --skip-compaction-check` for future-only
attribution. GPU prefix/reset calls would otherwise contribute to counters.
Normal stdout remains the complete native-comparable packet; stderr has the
original runner JSON followed by a separate diagnostic JSON.

The four intervals are Base physics step (including its internal synchronize),
external Game synchronize, whole Base Game step, and digest `make_frame`.
The digest interval excludes observation-summary construction and `wire.fold`.
Counter updates occur after their own intervals; outer Game timing still
includes nested private-counter update costs. Its residual is rules plus
instrumentation, not an exact isolated rule measurement. `clock64` measures
elapsed SM clocks for each thread, including stalls/divergence; sums are neither
kernel wall time nor device utilization. Compare diagnostic and normal rollout
time to assess perturbation before choosing an optimization direction.
