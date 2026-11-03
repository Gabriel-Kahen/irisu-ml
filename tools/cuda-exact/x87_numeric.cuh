#pragma once

/* Correctly-rounded sincos of binary32 value.

Copyright (c) 2024-2025 Alexei Sibidanov

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace irisu::cuda_exact {

__host__ __device__ inline unsigned long long bits_double(double x) {
  union { double f; unsigned long long u; } bits; bits.f = x; return bits.u;
}
__host__ __device__ inline unsigned bits_float(float x) {
  union { float f; unsigned u; } bits; bits.f = x; return bits.u;
}
#define IRISU_RN_BINARY(name, type, intrinsic, expression) \
__host__ __device__ inline type name(type a, type b) { \
  IRISU_RN_BODY(intrinsic(a,b), expression) \
}
#ifdef __CUDA_ARCH__
#define IRISU_RN_BODY(device, host) return device;
#else
// Volatile makes each host operation a separate rounding point even when the
// parent translation unit permits contraction. Host requires FE_TONEAREST.
#define IRISU_RN_BODY(device, host) volatile auto rounded = (host); return rounded;
#endif
IRISU_RN_BINARY(add_rn, double, __dadd_rn, a+b)
IRISU_RN_BINARY(sub_rn, double, __dsub_rn, a-b)
IRISU_RN_BINARY(mul_rn, double, __dmul_rn, a*b)
IRISU_RN_BINARY(div_rn, double, __ddiv_rn, a/b)
IRISU_RN_BINARY(fmul_rn, float, __fmul_rn, a*b)
IRISU_RN_BINARY(fadd_rn, float, __fadd_rn, a+b)
__host__ __device__ inline double sqrt_rn(double x) {
  IRISU_RN_BODY(__dsqrt_rn(x), sqrt(x))
}
__host__ __device__ inline float float_rn(double x) {
  IRISU_RN_BODY(__double2float_rn(x), static_cast<float>(x))
}
#undef IRISU_RN_BINARY
#undef IRISU_RN_BODY


// Numeric values only: these helpers do not emulate x87 status/exception bits.
// PC53 has a wider exponent than binary64. Fail closed when a result would
// require that range, instead of treating CUDA underflow/overflow as exact.
struct Number {
  double value;
  bool supported;
};

__host__ __device__ inline bool normal_or_zero(double x) {
  const auto absolute = static_cast<unsigned long long>(bits_double(x)) &
                        0x7fffffffffffffffULL;
  const auto exponent = absolute & 0x7ff0000000000000ULL;
  return absolute == 0 || (exponent != 0 && exponent != 0x7ff0000000000000ULL);
}

__host__ __device__ inline Number add(double a, double b) {
  const double r = add_rn(a, b);
  return {r, normal_or_zero(a) && normal_or_zero(b) && normal_or_zero(r) &&
                 (r != 0 || a == -b)};
}
__host__ __device__ inline Number sub(double a, double b) { return add(a, -b); }
__host__ __device__ inline Number mul(double a, double b) {
  const double r = mul_rn(a, b);
  return {r, normal_or_zero(a) && normal_or_zero(b) && normal_or_zero(r) &&
                 (r != 0 || a == 0 || b == 0)};
}
__host__ __device__ inline Number div(double a, double b) {
  const double r = div_rn(a, b);
  return {r, normal_or_zero(a) && normal_or_zero(b) && b != 0 &&
                 normal_or_zero(r) && (r != 0 || a == 0)};
}
__host__ __device__ inline Number square_root(double a) {
  const double r = sqrt_rn(a);
  return {r, normal_or_zero(a) && a >= 0 && normal_or_zero(r)};
}

struct TrigPair {
  float sine;
  float cosine;
  bool supported;
};

inline constexpr double trig_table_host[32] = {
0x0p+0, 0x1.8f8b83c69a60bp-3, 0x1.87de2a6aea963p-2, 0x1.1c73b39ae68c8p-1,
      0x1.6a09e667f3bcdp-1, 0x1.a9b66290ea1a3p-1, 0x1.d906bcf328d46p-1, 0x1.f6297cff75cbp-1,
      0x1p+0, 0x1.f6297cff75cbp-1, 0x1.d906bcf328d46p-1, 0x1.a9b66290ea1a3p-1,
      0x1.6a09e667f3bcdp-1, 0x1.1c73b39ae68c8p-1, 0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60bp-3,
      0x0p+0, -0x1.8f8b83c69a60bp-3, -0x1.87de2a6aea963p-2, -0x1.1c73b39ae68c8p-1,
      -0x1.6a09e667f3bcdp-1, -0x1.a9b66290ea1a3p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1,
      -0x1p+0, -0x1.f6297cff75cbp-1, -0x1.d906bcf328d46p-1, -0x1.a9b66290ea1a3p-1,
      -0x1.6a09e667f3bcdp-1, -0x1.1c73b39ae68c8p-1, -0x1.87de2a6aea963p-2, -0x1.8f8b83c69a60bp-3};
static __device__ __constant__ double trig_table_device[32] = {
0x0p+0, 0x1.8f8b83c69a60bp-3, 0x1.87de2a6aea963p-2, 0x1.1c73b39ae68c8p-1,
      0x1.6a09e667f3bcdp-1, 0x1.a9b66290ea1a3p-1, 0x1.d906bcf328d46p-1, 0x1.f6297cff75cbp-1,
      0x1p+0, 0x1.f6297cff75cbp-1, 0x1.d906bcf328d46p-1, 0x1.a9b66290ea1a3p-1,
      0x1.6a09e667f3bcdp-1, 0x1.1c73b39ae68c8p-1, 0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60bp-3,
      0x0p+0, -0x1.8f8b83c69a60bp-3, -0x1.87de2a6aea963p-2, -0x1.1c73b39ae68c8p-1,
      -0x1.6a09e667f3bcdp-1, -0x1.a9b66290ea1a3p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1,
      -0x1p+0, -0x1.f6297cff75cbp-1, -0x1.d906bcf328d46p-1, -0x1.a9b66290ea1a3p-1,
      -0x1.6a09e667f3bcdp-1, -0x1.1c73b39ae68c8p-1, -0x1.87de2a6aea963p-2, -0x1.8f8b83c69a60bp-3};
__host__ __device__ inline double trig_table(unsigned i) {
#ifdef __CUDA_ARCH__
  return trig_table_device[i & 31U];
#else
  return trig_table_host[i & 31U];
#endif
}

// Device adaptation of the bounded path in ../x87-trig-differential/
// core_math_sincosf.c, CORE-MATH commit 07cf01e12a42b82cc478341982936cad7f3f9bdc.
// Copyright (c) 2024-2025 Alexei Sibidanov. MIT license: see that source file.
// The CPU's exhaustive result applies to final binary32 values, not the
// extended intermediates of FSIN/FCOS. Always round before consuming a result.
__host__ __device__ inline TrigPair sincos_float(float x) {
  const std::uint32_t absolute = bits_float(x) & 0x7fffffffU;
  if (absolute >= 0x49000000U) return {0, 0, false}; // |x| >= 2^19 or NaN/Inf
  if (absolute == 0) return {x, 1.0f, true};
  if (absolute < 0x39800000U) {
    if (absolute < 0x33000000U) return {x, 1.0f, true};
    const float xx = fmul_rn(x, x);
    const float s = fadd_rn(fmul_rn(fmul_rn(-0x1.555556p-3f, x), xx), x);
    const float c = fadd_rn(fmul_rn(fmul_rn(-0x1p-1f, x), x), 1.0f);
    return {s, c, true};
  }
  // These two hard-to-round inputs are CORE-MATH's bounded database entries.
  if (absolute == 0x4096cbe4U) {
    return {copysignf(1.0f, -x), 0x1.99bc5cp-27f, true};
  }
  if (absolute == 0x46199998U) {
    return {copysignf(0x1.63f4bap-2f, -x), -0x1.e01216p-1f, true};
  }
  const double argument = x;
  double reduced;
  double integral;
  if (absolute < 0x4116cbe4U) {
    const double high = mul_rn(0x1.45f306dc9c883p+2, argument);
    integral = nearbyint(high);
    reduced = sub_rn(high, integral);
  } else {
    const double low = mul_rn(-0x1.b1bbead603d8bp-29, argument);
    const double high = mul_rn(0x1.45f306ep+2, argument);
    integral = nearbyint(high);
    reduced = add_rn(sub_rn(high, integral), low);
  }
  const unsigned quadrant = static_cast<unsigned>(bits_double(add_rn(0x1.8p52, integral)) & 31U);
  const double z2 = mul_rn(reduced, reduced);
  const double z4 = mul_rn(z2, z2);
  double aa = add_rn(
      add_rn(0x1.921fb54442d17p-3, mul_rn(z2, -0x1.4abbce6256a39p-10)),
      mul_rn(z4, add_rn(0x1.466bc5a518c16p-19,
                            mul_rn(z2, -0x1.32bdc61074ff6p-29))));
  double bb = add_rn(
      add_rn(0x1.3bd3cc9be45dcp-6, mul_rn(z2, -0x1.03c1f081b0833p-14)),
      mul_rn(z4, add_rn(0x1.55d3c6fc9ac1fp-24,
                            mul_rn(z2, -0x1.e1d3ff281b40dp-35))));
  aa = mul_rn(aa, reduced);
  bb = mul_rn(bb, z2);
  const double s0 = trig_table(quadrant);
  const double c0 = trig_table(quadrant + 8);
  const double s = add_rn(s0, sub_rn(mul_rn(aa, c0), mul_rn(bb, s0)));
  const double c = sub_rn(c0, add_rn(mul_rn(aa, s0), mul_rn(bb, c0)));
  return {float_rn(s), float_rn(c), true};
}

__host__ __device__ inline TrigPair sincos_reloaded_float(double operand) {
  const float x = float_rn(operand);
  if (static_cast<double>(x) != operand) return {0, 0, false};
  return sincos_float(x);
}

} // namespace irisu::cuda_exact
