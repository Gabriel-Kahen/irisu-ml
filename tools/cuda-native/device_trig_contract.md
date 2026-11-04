# Optional approximate device trigonometry

`IRISU_CUDA_NATIVE_DEVICE_TRIG` defaults to 0. Its values are:

* 0: existing exact bounded CORE-MATH implementation on host and device.
* 1: CUDA device library `sincosf` on device; original exact helper on host.
* 2: CUDA intrinsic `__sincosf` on device; original exact helper on host.

Modes 1 and 2 are explicitly approximate physical profiles. They do not claim
native MSVC bit parity. Root runner metadata must identify the selected mode,
including hybrid host-prefix/device-future runs where only future trig changes.
The original numeric implementation and old backend are not edited.

The hook retains the exact `abs(angle) < 2^19` binary32 admission predicate,
which rejects NaN, infinity, and boundary/outside angles. Both signed zeros
explicitly return their original sine and cosine 1. Nonfinite library results
fail closed through the existing `supported` result. All other math operations,
store boundaries, structural checks, contact order, and cached-rotation
conditions remain unchanged. Host code always uses the original helper even
when this macro is enabled. No global fast-math, FTZ, or reassociation flag is
introduced.

The expensive nontrivial exact path performs double argument reduction,
table accesses, two double polynomials, and separate double rounding points
before its final float conversion. A device library/intrinsic may replace that
work with substantially fewer device instructions. This is a hypothesis;
actual solver/body/game costs and campaign speed must be measured.

`device_trig_validate.cu` reports both candidate profiles in one run. Its
default corpus contains 1,056 bit-pattern boundaries plus three independent
million-input families: arbitrary float words, dense angles across the whole
admitted interval, and float neighbors of multiples of pi/2 across that range.
It compares each candidate with the unchanged exact helper on the same device.
It reports admission and selected-hook mismatches, nonfinite results, signed
zero errors, differing result bits, maximum sine/cosine absolute and ULP error,
and maximum `abs(sine*sine + cosine*cosine - 1)`.

The explicit sampled local-error policy is at most 8 ULP and `2^-20` absolute
error per component, and `2^-18` unit-norm error. Those are qualification
thresholds, not a proven global error bound. An intrinsic profile may fail
these thresholds, especially for large angles; its failures must be preserved
and must not be relabeled as a pass. The exit code checks the selected compile
profile (library mode when compiled exact), while JSON retains both results.
`--cpu` requires complete bit equality for both profiles, verifying that host
execution remains unchanged.

Small local trig differences can change contacts, game trajectories, scores,
and rankings over time. This sampled qualifier establishes no accumulated
trajectory error bound, gameplay equivalence, native packet parity, or speed.
Actual game/search runs, lifecycle/state consistency and device memory gates
must be evaluated and reported separately before using an approximate profile.

The first V100 sample passed mode 1: maximum component error was 1 ULP
(`5.9604644775390625e-8` absolute). Mode 2 failed the stated sample policy on
2,551,540 inputs; maximum component absolute error was about 0.07008.
Those results are retained in `device_trig_qualification.json`. The short
library-profile search pilot preserved discrete results/ranks but changed
12,274 of 12,288 body/trace hashes. Root rejected the accuracy tradeoff for the
small observed incremental timing difference; the selected profile remains 0.
