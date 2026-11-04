#pragma once
// Numeric reuse only: no Guest, register machine, dispatch or fallback.
#include "types.hpp"
#include "../../tools/cuda-exact/x87_numeric.cuh"
#ifndef IRISU_CUDA_NATIVE_BOUNDED_MATH
#define IRISU_CUDA_NATIVE_BOUNDED_MATH 0
#endif
#ifndef IRISU_CUDA_NATIVE_F32_STORE
#define IRISU_CUDA_NATIVE_F32_STORE 0
#endif
#ifndef IRISU_CUDA_NATIVE_F32_DIV_STORE
#define IRISU_CUDA_NATIVE_F32_DIV_STORE 0
#endif
#ifndef IRISU_CUDA_NATIVE_BOUNDED_STORE
#define IRISU_CUDA_NATIVE_BOUNDED_STORE 0
#endif
#ifndef IRISU_CUDA_NATIVE_CACHED_ROTATION
#define IRISU_CUDA_NATIVE_CACHED_ROTATION 0
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
// Exact admission predicate of sincos_reloaded_float for an already binary32
// operand. Cache hits must not bypass the original range/NaN/Inf rejection.
NATIVE_HD inline bool sincos_input_supported(float a){return (irisu::cuda_exact::bits_float(a)&0x7fffffffU)<0x49000000U;}
// Each operation rounds independently to PC53; store() marks the observed
// native binary32 rounding points. Callers must reject a failed context.
struct Context {
 Status status{Status::ok};
 NATIVE_HD bool ok() const{return status==Status::ok;}
 NATIVE_HD double take(Number n){if(!n.supported)status=Status::unsupported_numeric;return n.value;}
#if IRISU_CUDA_NATIVE_BOUNDED_MATH && defined(__CUDA_ARCH__)
 // Experimental current-graph domain only; not a generic checked Context.
 // Keep F32 finite/div/sqrt/trig checks. See bounded_math_contract.md for bounds
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
 NATIVE_HD float store(double a){
#if IRISU_CUDA_NATIVE_BOUNDED_STORE && IRISU_CUDA_NATIVE_BOUNDED_MATH && defined(__CUDA_ARCH__)
  // Current bounded graphs only: no nonzero subnormal binary64 store input.
  // Free math::store and host Context remain checked; see bounded_store_contract.md.
  const float value=irisu::cuda_exact::float_rn(a);
  if(!math::finite(value))status=Status::unsupported_numeric;
  return value;
#else
  auto n=math::store(a);if(!n.supported)status=Status::unsupported_numeric;return n.value;
#endif
 }
 // Only adjacent binary32 stores may use these overloads. Float products are
 // exact in binary64; float add/sub have no harmful double-round to binary32.
 // Nonfinite inputs retain the original graph; rejected payloads are unspecified.
 NATIVE_HD float mul_store(float a,float b){
#if IRISU_CUDA_NATIVE_F32_STORE
  if(math::finite(a)&&math::finite(b)){const auto v=irisu::cuda_exact::fmul_rn(a,b);if(!math::finite(v))status=Status::unsupported_numeric;return v;}
#endif
  return store(mul(double(a),double(b)));
 }
 NATIVE_HD float add_store(float a,float b){
#if IRISU_CUDA_NATIVE_F32_STORE
  if(math::finite(a)&&math::finite(b)){const auto v=irisu::cuda_exact::fadd_rn(a,b);if(!math::finite(v))status=Status::unsupported_numeric;return v;}
#endif
  return store(add(double(a),double(b)));
 }
 NATIVE_HD float sub_store(float a,float b){
#if IRISU_CUDA_NATIVE_F32_STORE
  if(math::finite(a)&&math::finite(b)){
#ifdef __CUDA_ARCH__
   const float v=__fsub_rn(a,b);
#else
   volatile float v=a-b;
#endif
   if(!math::finite(v))status=Status::unsupported_numeric;return v;
  }
#endif
  return store(sub(double(a),double(b)));
 }
 template<class A,class B>NATIVE_HD float mul_store(A a,B b){return store(mul(double(a),double(b)));}
 template<class A,class B>NATIVE_HD float add_store(A a,B b){return store(add(double(a),double(b)));}
 template<class A,class B>NATIVE_HD float sub_store(A a,B b){return store(sub(double(a),double(b)));}
 // Adjacent quotient stores only. Finite F32 ratios cannot double-round
 // differently through RN53; keep zero/nonfinite inputs on the checked path.
 NATIVE_HD float div_store(float a,float b){
#if IRISU_CUDA_NATIVE_F32_DIV_STORE
  if(math::finite(a)&&math::finite(b)&&b!=0){
#ifdef __CUDA_ARCH__
   const float value=__fdiv_rn(a,b);
#else
   volatile float value=a/b;
#endif
   if(!math::finite(value))status=Status::unsupported_numeric;return value;
  }
#endif
  return store(div(double(a),double(b)));
 }
 template<class I,std::enable_if_t<std::is_integral_v<I>,int> =0>NATIVE_HD float div_store(I a,float b){
#if IRISU_CUDA_NATIVE_F32_DIV_STORE
  const float numerator=irisu::cuda_exact::float_rn(double(a));
  if(double(numerator)==double(a))return div_store(numerator,b);
#endif
  return store(div(double(a),double(b)));
 }
 template<class A,class B>NATIVE_HD float div_store(A a,B b){return store(div(double(a),double(b)));}

};
}
