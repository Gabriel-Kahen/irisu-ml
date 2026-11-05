// Approximate device profiles versus the existing exact finite-angle helper.
#ifndef IRISU_CUDA_NATIVE_DEVICE_TRIG
#define IRISU_CUDA_NATIVE_DEVICE_TRIG 1
#endif
#include "../../clone/cuda_native/solver_math.hpp"
#include <cfenv>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace n=irisu::cuda_native;
namespace x=irisu::cuda_exact;
using U=unsigned long long;
constexpr double absolute_limit=0x1p-20,norm_limit=0x1p-18;
constexpr unsigned ulp_limit=8,boundary_count=1056;
struct Metrics {
 U samples{},accepted{},rejected{},admission_mismatches{},nonfinite{},zero_mismatches{},hook_mismatches{},bit_differences{},bound_failures{};
 U max_sine_error{},max_cosine_error{},max_norm_error{},first_bad{~U(0)};
 unsigned max_sine_ulp{},max_cosine_ulp{};
};
NATIVE_HD U mix(U v){v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
NATIVE_HD float f32(unsigned word){union{unsigned u;float f;}v{word};return v.f;}
double f64(U word){union{U u;double f;}v{word};return v.f;}
NATIVE_HD double absolute(double v){return v==0?0:v<0?-v:v;}
NATIVE_HD unsigned ulps(float a,float b){unsigned left=x::bits_float(a),right=x::bits_float(b);left=(left&0x80000000U)?~left:left^0x80000000U;right=(right&0x80000000U)?~right:right^0x80000000U;return left>right?left-right:right-left;}
NATIVE_HD float angle(U index,U random){
 if(index<1024)return f32(((unsigned(index)&1U)<<31)|(((unsigned(index)>>2)&255U)<<23)|((index&2U)?0x7fffffU:0));
 if(index<boundary_count){
  const unsigned extra[]={1,2,3,0x33000000,0x39800000,0x4096cbe4,0x46199998,0x48fffffe,0x48ffffff,0x49000000,0x49000001,0x3fc90fda,0x3fc90fdb,0x3fc90fdc,0x40490fda,0x40490fdb};
  const unsigned i=unsigned(index)-1024;return f32(extra[i&15U]|((i&16U)?0x80000000U:0));
 }
 index-=boundary_count;const auto kind=index/random;const U bits=mix(index%random);
 if(kind==0)return f32(unsigned(bits));
 if(kind==1)return x::float_rn(double(static_cast<std::int32_t>(unsigned(bits)))*0x1p-12);
 const int quadrant=int(bits%667545)-333772;
 const float center=x::float_rn(double(quadrant)*0x1.921fb54442d18p0);
 return f32(x::bits_float(center)+unsigned(bits%3)-1);
}
NATIVE_HD x::TrigPair candidate(float a,unsigned profile){
#ifdef __CUDA_ARCH__
 if(!n::math::sincos_input_supported(a))return {0,0,false};
 if((x::bits_float(a)&0x7fffffffU)==0)return {a,1,true};
 float sine,cosine;if(profile==1)::sincosf(a,&sine,&cosine);else ::__sincosf(a,&sine,&cosine);
 return {sine,cosine,n::math::finite(sine)&&n::math::finite(cosine)};
#else
 return x::sincos_float(a);
#endif
}
NATIVE_HD bool same(x::TrigPair a,x::TrigPair b){return a.supported==b.supported&&x::bits_float(a.sine)==x::bits_float(b.sine)&&x::bits_float(a.cosine)==x::bits_float(b.cosine);}
NATIVE_HD void sample(Metrics& m,float a,unsigned profile){
 ++m.samples;const auto ref=x::sincos_float(a),out=candidate(a,profile);
 bool bad=ref.supported!=out.supported;m.admission_mismatches+=bad;
 if(profile==IRISU_CUDA_NATIVE_DEVICE_TRIG||(IRISU_CUDA_NATIVE_DEVICE_TRIG==0&&profile==1)){
  const auto hooked=n::math::sincos(a);const bool mismatch=!same(hooked,IRISU_CUDA_NATIVE_DEVICE_TRIG==0?ref:out);m.hook_mismatches+=mismatch;bad|=mismatch;
 }
 if(!ref.supported){++m.rejected;}else{
  ++m.accepted;
  if(!n::math::finite(out.sine)||!n::math::finite(out.cosine)){++m.nonfinite;bad=true;}
  else{
   const auto su=ulps(ref.sine,out.sine),cu=ulps(ref.cosine,out.cosine);if(su>m.max_sine_ulp)m.max_sine_ulp=su;if(cu>m.max_cosine_ulp)m.max_cosine_ulp=cu;
   const double se=absolute(double(out.sine)-ref.sine),ce=absolute(double(out.cosine)-ref.cosine);
   const double norm=absolute(x::add_rn(x::mul_rn(out.sine,out.sine),x::mul_rn(out.cosine,out.cosine))-1);
   const U sb=x::bits_double(se),cb=x::bits_double(ce),nb=x::bits_double(norm);
   if(sb>m.max_sine_error)m.max_sine_error=sb;if(cb>m.max_cosine_error)m.max_cosine_error=cb;if(nb>m.max_norm_error)m.max_norm_error=nb;
   m.bit_differences+=su||cu;
   const bool outside=su>ulp_limit||cu>ulp_limit||se>absolute_limit||ce>absolute_limit||norm>norm_limit;m.bound_failures+=outside;bad|=outside;
   if((x::bits_float(a)&0x7fffffffU)==0&&!same(ref,out)){++m.zero_mismatches;bad=true;}
  }
 }
 if(bad&&m.first_bad==~U(0))m.first_bad=x::bits_float(a);
}
#ifdef __CUDACC__
__global__ void qualify(Metrics* results,U random){Metrics local[2];const U total=boundary_count+3*random;
 for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=U(blockDim.x)*gridDim.x){const float a=angle(i,random);sample(local[0],a,1);sample(local[1],a,2);}
 for(unsigned mode=0;mode<2;++mode){auto& m=local[mode];auto& r=results[mode];
#define ADD(field) atomicAdd(&r.field,m.field)
  ADD(samples);ADD(accepted);ADD(rejected);ADD(admission_mismatches);ADD(nonfinite);ADD(zero_mismatches);ADD(hook_mismatches);ADD(bit_differences);ADD(bound_failures);
#undef ADD
  atomicMax(&r.max_sine_error,m.max_sine_error);atomicMax(&r.max_cosine_error,m.max_cosine_error);atomicMax(&r.max_norm_error,m.max_norm_error);atomicMax(&r.max_sine_ulp,m.max_sine_ulp);atomicMax(&r.max_cosine_ulp,m.max_cosine_ulp);
  if(m.first_bad!=~U(0))atomicCAS(&r.first_bad,~U(0),m.first_bad);
 }
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
bool passed(const Metrics& m,bool host){return !m.admission_mismatches&&!m.nonfinite&&!m.zero_mismatches&&!m.hook_mismatches&&!m.bound_failures&&(!host||!m.bit_differences);}
int main(int argc,char** argv){try{
 U random=1000000;bool cpu=false;for(int i=1;i<argc;++i)if(std::string(argv[i])=="--cpu")cpu=true;else random=std::stoull(argv[i]);
 if(random>100000000ULL)throw std::runtime_error("maximum 100 million per input family");
 if(std::fesetenv(FE_DFL_ENV))throw std::runtime_error("cannot select round nearest");
#if defined(__i386__)||defined(__x86_64__)
 const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
#endif
 Metrics m[2];const U total=boundary_count+3*random;
 if(cpu){for(U i=0;i<total;++i){const float a=angle(i,random);sample(m[0],a,1);sample(m[1],a,2);}}else{
#ifdef __CUDACC__
  Metrics* device{};check(cudaMalloc(&device,sizeof(m)));check(cudaMemcpy(device,m,sizeof(m),cudaMemcpyHostToDevice));qualify<<<128,256>>>(device,random);check(cudaGetLastError());check(cudaMemcpy(m,device,sizeof(m),cudaMemcpyDeviceToHost));check(cudaFree(device));
#else
  throw std::runtime_error("host build requires --cpu");
#endif
 }
 std::printf("{\"schema\":\"irisu-device-trig-sampled-error-v1\",\"backend\":\"%s\",\"selected_profile\":%d,\"random_per_family\":%llu,\"samples_per_mode\":%llu,\"absolute_limit\":%.17g,\"ulp_limit\":%u,\"norm_limit\":%.17g,\"modes\":[",cpu?"host":"cuda",IRISU_CUDA_NATIVE_DEVICE_TRIG,random,total,absolute_limit,ulp_limit,norm_limit);
 for(unsigned i=0;i<2;++i){const auto& r=m[i];std::printf("%s{\"profile\":%u,\"name\":\"%s\",\"passed_sample_bounds\":%s,\"samples\":%llu,\"accepted\":%llu,\"rejected\":%llu,\"admission_mismatches\":%llu,\"nonfinite_outputs\":%llu,\"zero_mismatches\":%llu,\"hook_mismatches\":%llu,\"different_result_bits\":%llu,\"bound_failures\":%llu,\"max_sine_ulp\":%u,\"max_cosine_ulp\":%u,\"max_sine_absolute\":%.17g,\"max_cosine_absolute\":%.17g,\"max_unit_norm_error\":%.17g,\"first_bad_angle_bits\":%llu}",i?",":"",i+1,i?"intrinsic-__sincosf":"library-sincosf",passed(r,cpu)?"true":"false",r.samples,r.accepted,r.rejected,r.admission_mismatches,r.nonfinite,r.zero_mismatches,r.hook_mismatches,r.bit_differences,r.bound_failures,r.max_sine_ulp,r.max_cosine_ulp,f64(r.max_sine_error),f64(r.max_cosine_error),f64(r.max_norm_error),r.first_bad);}
 std::puts("],\"scope\":\"Sampled local trig error only; host must remain bit exact; no trajectory error bound, native parity, memory or performance admission\"}");
 const unsigned selected=IRISU_CUDA_NATIVE_DEVICE_TRIG==2?1:0;return passed(m[selected],cpu)&&m[selected].samples==total?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"device-trig-validate: %s\n",e.what());return 2;}}
