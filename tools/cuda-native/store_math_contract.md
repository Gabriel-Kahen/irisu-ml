# Optional adjacent binary32 store operations

`IRISU_CUDA_NATIVE_F32_STORE` defaults to zero. `Context.mul_store`,
`add_store` and `sub_store` replace only an immediately stored binary operation.
The exact `float,float` overload uses the explicit RN binary32 operation when
both operands are finite. All other operand types and nonfinite float operands
retain the original double-operation/store graph. No unstored product, sum,
division, square root or trigonometric operation is narrowed or reassociated.

A product of two finite binary32 values has at most 48 significant bits, with
nonzero exponent between -298 and 256, so its binary64 product is exact and
normal. Rounding that product to binary32 equals the direct RN binary32 product.
For addition/subtraction, operands close enough to affect a binary32 rounding
boundary have an exact sum representable within binary64 precision; larger
exponent gaps cannot cause a harmful double rounding. Nonzero differences are
at least the minimum binary32 subnormal and remain normal in binary64. Overflow
of the binary32 result remains an unsupported context. Signed zeros and
subnormal inputs/results require FTZ disabled, as in the qualified build flags.

The status contract checks **every** result, including a previously failed
context. Value bits are required to match only when the context accepts the
result. No failed context may be accepted downstream. Rejected NaN payload/sign
bits and values computed after an already failed context are outside this
contract: C++ can commute NaN arithmetic even when the nominal double graph is
retained. The qualifier counts rejected-value differences separately rather
than accepting a status mismatch or an accepted-value mismatch.

`store_math_validate.cu` locally enables this option and disables bounded
intermediate arithmetic, selecting the original fully checked reference. Its
default corpus contains 1,000,000 reproducible random pairs plus 1,115,136
boundary pairs covering every exponent, signs, extreme mantissas, subnormals,
midpoints, cancellation, overflow, infinities and NaNs. It also tests sticky
failure and mixed float/double/integer fallback, including wide values that
cannot be narrowed before the operation. An argument up to 1,000,000,000 selects
a larger random corpus; `--cpu` selects the host qualifier. GPU and host each
compare against their own checked graph; rejected NaN representations need not
agree across architectures.
