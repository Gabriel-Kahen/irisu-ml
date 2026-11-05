# Optional bounded binary64 store check removal

`IRISU_CUDA_NATIVE_BOUNDED_STORE` defaults to zero. It changes only device
`Context::store` when `IRISU_CUDA_NATIVE_BOUNDED_MATH` is also enabled. Host
Context, free `math::store`, division, square root, trig, and the old backend
are unchanged. The option preserves conversion through `float_rn`, the finite
binary32 result check, and sticky `unsupported_numeric` status. It does not
change an arithmetic operation, rounding point, signed zero, or store order.

This is an extension of the current-graph contract in
[bounded_math_contract.md](bounded_math_contract.md), replacing that document's
requirement to retain `normal_or_zero` at device Context stores. It is not a
generic finite-double store contract.

For the current graphs, successful store inputs have these origins:

* Binary32 fields/constants, or sums and differences of at most four binary32
  factors per monomial. Factory polygon integrals are the largest case;
  `inverse3` is explicitly binary32. Loops round accumulators to binary32 each
  iteration. The audited conservative nonzero lower grid is `2^-596`, with
  magnitude below `2^520`. Binary64 rounding never creates a finer grid.
  Therefore these values cannot be binary64 subnormals or silently underflow
  in binary64. Exact cancellation to signed zero remains valid.
* Checked division and square root results. Their `Number` checks already
  reject nonfinite/subnormal binary64 results and nonzero underflow, marking
  the Context before a store. Subsequent operations cannot clear failure.
  In the one sqrt-plus-radius graph both operands are nonnegative on admitted
  input, so cancellation cannot create a smaller result.

Thus `normal_or_zero(double_argument)` is true at each otherwise accepted
current-graph store. NaN or Inf on a rejected path remains rejected by the
finite-binary32 check. A finite normal binary64 value too large for binary32
also remains rejected. Conversion of a valid normal binary64 value to a
binary32 subnormal or zero remains intentional and bit-identical.

Arbitrary calls such as `Context.store(0x1p-1023)`, external raw-double
temporaries, longer product chains, or new unstored accumulators are outside
this option's domain. In particular, eight minimum-subnormal F32 factors can
underflow in binary64 and cannot be admitted merely because the final F32 is
zero. The checked free store continues to reject raw binary64 subnormals.
New call sites require auditing before enabling this option for them.

`bounded_store_validate.cu` compares ten composed graphs against independently
checked `Number` operations and free `math::store` on the device. Its default
corpus includes one million deterministic random four-tuples plus every pair
of 1,040 boundary bit patterns. It compares every status, and accepted F32
and raw F64 bits. Failure payload bits are diagnostic only. The graphs cover
cancellation, four-factor products, factory integrals, division, sqrt, store
overflow/underflow, and sticky failures. Raw double subnormal admission is
deliberately not asserted for fast Context.

This source argument and numeric qualifier do not replace native collision,
solver, factory, dynamic scene, full-game, snapshot, search, or memory checks.
No speed improvement is established by this document. Keep strict RN CUDA
flags and FTZ disabled; this option is independent of the F32-store experiment.
