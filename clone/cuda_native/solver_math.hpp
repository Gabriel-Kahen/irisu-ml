#pragma once
// Numeric reuse only: no Guest, register machine, dispatch or fallback.
#include "types.hpp"
#include "../../tools/cuda-exact/x87_numeric.cuh"
#ifndef IRISU_CUDA_NATIVE_BOUNDED_MATH
#define IRISU_CUDA_NATIVE_BOUNDED_MATH 0
#endif
namespace irisu::cuda_native::math {
using Number=irisu::cuda_exact::Number;
using TrigPair=irisu::cuda_exact::TrigPair;
NATIVE_HD inline Number add(double a,double b){return irisu::cuda_exact::add(a,b);}
NATIVE_HD inline Number sub(double a,double b){return irisu::cuda_exact::sub(a,b);}
NATIVE_HD inline Number mul(double a,double b){return irisu::cuda_exact::mul(a,b);}
NATIVE_HD inline Number div(double a,double b){return irisu::cuda_exact::div(a,b);}
NATIVE_HD inline Number sqrt(double a){return irisu::cuda_exact::square_root(a);}
NATIVE_HD inline bool finite(float a){return (irisu::cuda_exact::bits_float(a)&0x7f800000U)!=0x7f800000U;}
struct Store {float value;bool supported;};
NATIVE_HD inline Store store(double a){
 const float value=irisu::cuda_exact::float_rn(a);
 return {value,irisu::cuda_exact::normal_or_zero(a)&&finite(value)};
}
NATIVE_HD inline TrigPair sincos(float a){return irisu::cuda_exact::sincos_reloaded_float(a);}
// Each operation rounds independently to PC53; store() marks the observed
// native binary32 rounding points. Callers must reject a failed context.
struct Context {
 Status status{Status::ok};
 NATIVE_HD bool ok() const{return status==Status::ok;}
 NATIVE_HD double take(Number n){if(!n.supported)status=Status::unsupported_numeric;return n.value;}
#if IRISU_CUDA_NATIVE_BOUNDED_MATH && defined(__CUDA_ARCH__)
 // Experimental current-graph domain only; not a generic checked Context.
 // Keep store/div/sqrt/trig checks. See bounded_math_contract.md for bounds
 // and the unsupported unstored-product counterexample. Checked is default.
 NATIVE_HD double add(double a,double b){return irisu::cuda_exact::add_rn(a,b);}
 NATIVE_HD double sub(double a,double b){return irisu::cuda_exact::sub_rn(a,b);}
 NATIVE_HD double mul(double a,double b){return irisu::cuda_exact::mul_rn(a,b);}
#else
 NATIVE_HD double add(double a,double b){return take(math::add(a,b));}
 NATIVE_HD double sub(double a,double b){return take(math::sub(a,b));}
 NATIVE_HD double mul(double a,double b){return take(math::mul(a,b));}
#endif
 NATIVE_HD double div(double a,double b){return take(math::div(a,b));}
 NATIVE_HD double sqrt(double a){return take(math::sqrt(a));}
 NATIVE_HD float store(double a){auto n=math::store(a);if(!n.supported)status=Status::unsupported_numeric;return n.value;}
};
}
