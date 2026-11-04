# Optional exact adjacent binary32 division stores

`IRISU_CUDA_NATIVE_F32_DIV_STORE` defaults to zero. With the option enabled,
`Context.div_store(float,float)` uses explicitly round-to-nearest binary32
division for finite operands and a nonzero denominator, then retains the
finite-output check and sticky failure. Zero denominators and nonfinite inputs
use the original checked binary64 division/store. Other operand types also
retain that graph, except an integral numerator with a float denominator may
use the float overload after checking `double(float(a)) == double(a)`.
The equality is constant-foldable for the current integer literals 1 and 65535.
An integer such as 16777217 must keep the checked original representation.

Only an adjacent `m.store(m.div(a,b))` is rewritten as `m.div_store(a,b)`.
No raw double quotient, nested unstored division, or earlier rounding point is
changed. For example, a double subtraction in a clip denominator and the
unstored `width/2` wrapper quotient continue through the generic checked
fallback. `Context.div`, the free numeric helpers, the old backend, and all
default behavior remain unchanged. The separate F32 add/multiply experiment
need not be enabled.

## Rounding argument

Write a positive nonzero finite binary32 ratio as `x = n * 2^k / d`, where
positive integers n and d have at most 24 significant bits. Its exact magnitude
is strictly between `2^-277` and `2^277`, so binary64 division cannot encounter
its exponent limits or zero-underflow. A binary32 rounding midpoint is a
dyadic `y = m * 2^e` with m having at most 25 bits. This includes subnormal
midpoints, the half-minimum-subnormal boundary, and the overflow threshold.

For `k >= e`, the relative difference from a nonidentical midpoint has an
integer numerator and denominator `m*d < 2^49`, so it exceeds `2^-49`.
For `k < e`, compare integer n with `m*d*2^(e-k)` instead. If x is near y
(within a factor of two), that denominator is less than `2*n < 2^25`;
otherwise x is already far from y. Therefore a non-tie cannot approach an F32
midpoint closely enough for RN53 division, whose relative rounding error is
at most `2^-53`, to cross or land on it. An exact midpoint itself has at most
25 significant bits and is exactly representable in binary64. Consequently
RN32(x) equals RN32(RN53(x)), including ties-to-even.

Negative ratios follow by sign symmetry. Explicit zero inputs preserve the
division sign rules. Conversion to an F32 subnormal or signed zero is valid and
unchanged; overflow to F32 infinity still fails. This proof requires strict RN
operations and FTZ disabled, with no reciprocal approximation or fast-math
reassociation. Host qualification uses an independently rounded volatile float
quotient; CUDA uses `__fdiv_rn`.

## Qualification scope

`div_store_validate.cu` forces the option on against independently checked
`Number` division and free `math::store`, with bounded-check removals disabled.
The default corpus is one million deterministic random bit pairs plus every
pair of 1,088 boundary patterns. The boundaries cover all exponent/sign
combinations with extreme significands, subnormals, signed zeros, overflow,
nonfinite rejection, and near-midpoint quotients. Both platforms also test
sticky failures, exact and inexact integral numerators, generic mixed-width
operands, and wide doubles that cannot be narrowed first.

Every status and accepted result bit must match. Values after sticky rejection,
including NaN payload representations, are diagnostics outside the contract.
Native stage/game/search continuation and memory gates remain necessary before
profile admission; neither the argument nor a numeric test establishes speed.
