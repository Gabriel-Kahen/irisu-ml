# Optional bounded CUDA arithmetic contract

This is a source audit of the current independent typed physics graphs, not a
proof for arbitrary `Context` expressions and not a performance result. The
checked implementation remains the default. `IRISU_CUDA_NATIVE_BOUNDED_MATH`
may opt CUDA device code into replacing only `Context.add/sub/mul`'s `Number`
checks with the same separately rounded binary64 operations. Host arithmetic,
the shared x87 implementation, and all other checks remain unchanged.

## Required invariant

Every consumed stage input is a finite binary32 value; each successful explicit
binary32 store reestablishes that invariant. Snapshots and external mutations
must satisfy the same contract. Counts, indices, shape kinds and other
structural checks remain required. Unconsumed padding/unused fields are not
numeric inputs.

All original binary32 store boundaries, evaluation order, signed-zero behavior,
PC53 round-to-nearest operations and branch comparisons remain intact. Keep
`__dadd_rn`, `__dsub_rn`, `__dmul_rn` (or their exact existing wrappers), rather
than allowing contraction, reassociation or reciprocal approximations. Retain
`--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` and the existing host
floating-point contract. This option is not CUDA `--use_fast_math`.

The invariant describes accepted executions. Arithmetic can continue after a
failed store, so subsequent operands may be Inf/NaN; sticky failure must prevent
accepting that stage or its partial writes. Do not infer that every instruction
executed on a rejected path has finite operands.

## Bounds between successful binary32 stores

A nonzero finite binary32 value has magnitude at least `2^-149` and less than
`2^128`, and is an integer multiple of `2^-149`. The following bounds include
subnormal binary32 values. They do not assume FTZ or a minimum normal input.

| Current graph | Unstored arithmetic | Bound / reason |
| --- | --- | --- |
| Collision dot/rotate, circle threshold, shape transforms | Sum/difference of two F32 products; F32 radius squared | Products are multiples of `2^-298` and have magnitude below `2^256`; the sum is below `2^257`. Cancellation cannot create a nonzero value below the common dyadic grid. |
| Solver effective mass and angular/world integration | `(F32 - F32*F32)*F32`, sums of these terms, or three F32 factors plus an F32 value | A common grid is `2^-447`; fewer than sixteen terms keep magnitudes below `2^388`. The actual loops store between iterations. |
| Factory polynomial integrals and circle mass/inertia | At most four F32 factors in a monomial, including stored F32 constants such as pi, `inverse3`, `.25`, `.5` | A conservative common grid is `2^-596`; fewer than 256 unstored monomial terms give a loose absolute bound below `2^520`. Factory polygon-loop accumulation is explicitly F32-stored each iteration. |
| SAP quantization | `(F32 - F32)*F32`, followed by explicit integer-domain check | Common grid `2^-298`, magnitude below `2^257`; preserve the `[0,2^31)` conversion check. |
| Division sites | F32/F32, `a/(a-b)` with F32 `a,b`, or `(width/2)/scale` | For nonzero finite denominators, a deliberately loose nonzero magnitude bound is `2^-279` to `2^279`. Clip only divides when `a*b<0`, so its finite opposite-sign operands imply a nonzero denominator and a fraction in `(0,1)`. |
| Square-root sites | Square root of a stored F32 dot/product/distance | For positive operands, magnitude is at least `2^-74.5` and less than `2^64`. Zero remains valid. Negative operands are outside the domain. The factory's subsequent sqrt-plus-radius uses a nonnegative radius. |

The polynomial bound `2^-600 <= abs(nonzero result) < 2^520` is intentionally
looser than the current graph-specific grids. It stays far inside binary64's
normal range, `2^-1022` through values below `2^1024`. Separately rounded
add/sub/mul cannot produce a finer nonzero grid point than these bounds: exact
operations on the dyadic grid remain on it, and binary64 rounding either leaves
the value exact or rounds to a coarser grid. Exact cancellation to zero is valid.

These bounds depend on the enumerated graph shapes. A new unstored product,
unrounded reciprocal, loop-carried double accumulator, different constant, or
new consumer of raw double temporaries requires a fresh audit. Trigonometric
polynomial internals are a separately qualified implementation and are not
covered by this substitution.

## Checks that must remain

* Every explicit F32 store keeps both `normal_or_zero(double_argument)` and
  `finite(converted_float)`, and records sticky `unsupported_numeric` failure.
  Finite stage inputs can still overflow an F32 store. For example,
  `FLT_MAX * 2` is finite binary64 but must not become an accepted F32 Inf.
  Underflow from a normal binary64 intermediate to F32 zero is intentional and
  must continue to be accepted when the current store accepts it.
* Division retains its existing operand/result range, nonzero-denominator and
  zero-underflow checks. `solver_build_constraints` divides by `normal_k` and
  `tangent_k` without an explicit nonzero guard. Two bodies with zero inverse
  mass/inertia and a one-point manifold are finite inputs that produce `1/0`.
  Positive physical inputs alone do not prove a strictly positive rounded
  effective mass; do not remove checks based on ideal real-number algebra.
* Square root retains its existing nonnegative-domain and range checks.
  `WorldPairCallbacks::PairAdded` computes `sqrt(store(friction1*friction2))`.
  The generic factory checks finite friction but does not reject negative
  friction, so finite opposite-sign inputs are a concrete negative-root case.
* Trig qualification remains unchanged: the existing reloaded-F32 contract and
  `abs(angle) < 2^19` admission limit are stronger than mere finiteness.
* Geometry validity, polygon area guards, positive wrapper magnification, valid
  world extents, SAP integer conversion checks and all existing error handling
  remain. Native ordering and numerical failure are distinct requirements.

## Why this is not a generic unchecked Context

Let `q = 2^-149`, the smallest positive F32 subnormal. Eight successive
multiplications with no intervening F32 store compute `q^8 = 2^-1192` in a
PC53 format with the x87 exponent range. Binary64 becomes subnormal at the
seventh factor (`q^7 = 2^-1043`) and underflows to zero at the eighth. If all
intermediate checks were removed, a final `store(0)` would pass. The checked
implementation rejects the seventh multiplication. The final F32 value alone
cannot reconstruct that lost range failure. This chain is outside the audited
current graphs and must not be accepted as a fast-mode contract test.

The analogous large-product/reciprocal construction can overflow binary64 and
then turn its Inf into a finite zero. Retaining division checks prevents that
particular masking path, but does not establish generic graph safety.

## Required qualification before timing or admission

`bounded_math_validate.cu` compares the selected device mode with checked host
arithmetic for bounded store-to-store graphs, raw binary64 results, exact F32
results, status codes, actual circle collision, and constraint construction.
It includes minimum subnormals, cancellation, large finite inputs, F32 overflow,
zero denominators, negative sqrt operands, and trig admission boundaries. It
does not bless the out-of-contract eight-product chain.

Build and run that qualifier in both checked and opt-in modes. Then rerun pinned
native collision corpus families, solver stages, dynamic world/SAP lifecycle
histories, full-game frame parity including the unchanged canonical record,
snapshot continuation/isolation, and actual search candidate/selection parity.
Keep error/status gates and actual-device memory checks. A numeric micro-test
cannot replace these independent native proofs. Record compile mode, source and
binary hashes in separate receipts; do not relabel older checked receipts.

Measure only after correctness gates, with an uncontended paired workload and
the same setup/transfer/output scope on both sides. A check-removal hypothesis
is not evidence that these checks dominate runtime or that a speed goal was met.
