# CUDA numeric boundary

`x87_numeric.cuh` supplies host/device numeric helpers for a translated exact
physics experiment. It is not a floating-point-unit emulator or a claim of
full physics/replay parity.

The namespace is `irisu::cuda_exact`:

- `add`, `sub`, `mul`, `div`, and `square_root` return `Number { value,
  supported }`. They use explicit CUDA round-to-nearest operations and reject
  inputs/results requiring x87's wider exponent range. NaN/Infinity, binary64
  subnormal operands/results, and unsupported exceptional operations are also
  rejected conservatively. A caller must check `supported` before consuming
  the value. These helpers do not emulate exception flags or an x87 stack.
- `sincos_float` returns `TrigPair { sine, cosine, supported }`. It adapts the
  existing MIT-licensed CORE-MATH binary32 implementation, with separate
  rounding points and a constant-memory table. It accepts finite binary32
  operands with magnitude strictly below `2^19`, including signed zero and
  binary32 subnormals. Every other operand is rejected.
- `sincos_reloaded_float(double)` additionally requires that conversion to
  binary32 and back preserves the operand exactly. The returned sine/cosine
  values are **final binary32 outputs**. They must not replace unrounded x87
  transcendental intermediates where the original instruction stream consumes
  those intermediates directly.

The host implementations require round-to-nearest, as does the simulator. The
validator explicitly installs and checks x87 control word `0x027f`. Low-level
`*_rn` helpers expose separate rounding operations without the support checks;
callers using those directly own the required range checks.

Build on the V100 with CUDA 12.9 and GCC 14:

```sh
mkdir -p build-cuda-numeric
/opt/cuda/bin/nvcc -std=c++20 -O3 -arch=sm_70 -ccbin g++-14 \
  --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
  tools/cuda-exact/arithmetic_validate.cu \
  -o build-cuda-numeric/arithmetic_validate
env -u CUDA_VISIBLE_DEVICES build-cuda-numeric/arithmetic_validate 10000000 7
```

The inherited shell on the development host set `CUDA_VISIBLE_DEVICES=-1`.
Unsetting that variable for a command exposes its single V100 without changing
the user's global environment, power limit, or running services.

The mixed random/curated gate compares supported output bits against actual
hardware x87 operations, not host libm. It includes signed zero, binary32 and
binary64 subnormal cases, cancellation, divide-by-zero, overflow, hard-to-round
trig inputs, bound endpoints, Infinity, and NaN. The reported arithmetic
rejections are deliberate, and are not counted as successful comparisons.

Run the complete bounded binary32 trig gate in bounded-memory chunks:

```sh
python3 tools/cuda-exact/arithmetic_sweep.py \
  build-cuda-numeric/arithmetic_validate \
  --output build-cuda-numeric/exhaustive-trig
```

This enumerates both signs of every encoding in exponent bins 0 through 145:
2,449,473,536 operands including both zeros. Each chunk compares CUDA and host
helper outputs with independent `FSIN`/`FCOS` hardware results. Per-chunk JSON
records the raw contiguous range, comparison counts, mismatches, and validator
binary identity. Completed chunks can be resumed with the same binary.

The 2026-11-03 complete V100 gate compared all 2,449,473,536 operands with
zero GPU or host-helper mismatches against hardware x87. CUDA `sincosf`
disagreed on 125,647,062 operands in that same domain. This is exhaustive
evidence for these final binary32 result values on the tested hardware,
not a result for x87 exception/status bits or unrestricted operands.
The compact checked-in evidence is
[`cuda-x87-numeric-2026-11-03.json`](../../benchmarks/results/cuda-x87-numeric-2026-11-03.json).

The measured 10,000,000-sample gate on the Tesla V100 PCIe 32GB accepted
7,853,579 trig inputs and 39,923,352 arithmetic operation results, with no
raw-bit mismatches. Ordinary CUDA `sincosf` disagreed with x87 on 292,428 of
the accepted trig inputs. This establishes why ordinary CUDA trigonometry
cannot silently substitute for the exact backend.

The kernel in this gate computes **both** the candidate and CUDA-libm baseline.
Moving the candidate's table from per-thread local storage to constant memory
reduced its warmed median for the same ten million inputs from approximately
6.79 ms to 0.834 ms on this host. These are component timings; they exclude
transfers and CPU comparison and establish no end-to-end physics or search
speedup. The JSON evidence states this scope explicitly.
