// Adjacent stored-F32 division versus the fully checked original quotient.
#ifdef IRISU_CUDA_NATIVE_F32_DIV_STORE
#undef IRISU_CUDA_NATIVE_F32_DIV_STORE
#endif
#define IRISU_CUDA_NATIVE_F32_DIV_STORE 1
#ifdef IRISU_CUDA_NATIVE_BOUNDED_MATH
#undef IRISU_CUDA_NATIVE_BOUNDED_MATH
#endif
#define IRISU_CUDA_NATIVE_BOUNDED_MATH 0
#ifdef IRISU_CUDA_NATIVE_BOUNDED_STORE
#undef IRISU_CUDA_NATIVE_BOUNDED_STORE
#endif
#define IRISU_CUDA_NATIVE_BOUNDED_STORE 0
#include "../../clone/cuda_native/solver_math.hpp"
#include <cfenv>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace n=irisu::cuda_native;
namespace x=irisu::cuda_exact;
using U=unsigned long long;
struct Outcome {unsigned bits{},status{};};
struct Metrics {U pairs{},comparisons{},mismatches{},status_mismatches{},accepted_bit_mismatches{},rejected_bit_differences{},first{~U(0)};unsigned a{},b{},kind{},expected{},actual{},expected_status{},actual_status{};};
NATIVE_HD float value(unsigned bits){union{unsigned u;float f;} v{bits};return v.f;}
NATIVE_HD U mix(U v){v+=0x9e3779b97f4a7c15ULL;v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
constexpr unsigned boundary_count=1088;
NATIVE_HD unsigned boundary(unsigned i){
 if(i<1024)return ((i&1U)<<31)|(((i>>2)&255U)<<23)|((i&2U)?0x7fffffU:0);
 const unsigned extra[]={1,2,3,0x7ffffe,0x800001,0x800002,0x3f7ffffe,0x3f7fffff,0x3f800001,0x3f800002,0x3ffffffe,0x3fffffff,0x40000001,0x4b7ffffd,0x4b7ffffe,0x4b7fffff,0x4b800000,0x4b800001,0x337fffff,0x33800000,0x33800001,0x00800000,0x007fffff,0x7f7ffffe,0x7f800001,0x7fc12345,0x7ffffffe,0x00010001,0x3eaaaaab,0x3f000001,0x3effffff,0x477fff00};
 i-=1024;return extra[i&31U]|((i&32U)?0x80000000U:0);
}
template<class A,class B>NATIVE_HD Outcome checked(A a,B b,bool failed=false){
 const auto quotient=n::math::div(double(a),double(b));const auto stored=n::math::store(quotient.value);
 const auto status=failed||!quotient.supported||!stored.supported?n::Status::unsupported_numeric:n::Status::ok;
 return {x::bits_float(stored.value),unsigned(status)};
}
template<class A,class B>NATIVE_HD Outcome selected(A a,B b,bool failed=false){
 n::math::Context m;if(failed)m.status=n::Status::unsupported_numeric;const auto stored=m.div_store(a,b);return {x::bits_float(stored),unsigned(m.status)};
}
NATIVE_HD void compare(Metrics& m,U index,unsigned a,unsigned b,unsigned kind,Outcome ref,Outcome out){
 ++m.comparisons;const bool status=ref.status!=out.status,accepted=ref.status==unsigned(n::Status::ok)&&out.status==unsigned(n::Status::ok),different=ref.bits!=out.bits;
 m.status_mismatches+=status;m.accepted_bit_mismatches+=accepted&&different;m.rejected_bit_differences+=!accepted&&different;
 if(status||(accepted&&different)){++m.mismatches;if(m.first==~U(0)){m.first=index;m.a=a;m.b=b;m.kind=kind;m.expected=ref.bits;m.actual=out.bits;m.expected_status=ref.status;m.actual_status=out.status;}}
}
NATIVE_HD void test_pair(Metrics& m,U i){
 constexpr U boundaries=U(boundary_count)*boundary_count;const unsigned ab=i<boundaries?boundary(unsigned(i/boundary_count)):unsigned(mix(i-boundaries));
 const unsigned bb=i<boundaries?boundary(unsigned(i%boundary_count)):unsigned(mix((i-boundaries)^0xfedcba9876543210ULL)>>32);
 const float a=value(ab),b=value(bb);++m.pairs;compare(m,i,ab,bb,0,checked(a,b),selected(a,b));
 if(i<4096){
  compare(m,i,ab,bb,1,checked(a,b,true),selected(a,b,true));
  compare(m,i,ab,bb,2,checked(1,b),selected(1,b));
  compare(m,i,ab,bb,3,checked(65535,b),selected(65535,b));
  compare(m,i,ab,bb,4,checked(16777217,b),selected(16777217,b));
  compare(m,i,ab,bb,5,checked(-16777216,b),selected(-16777216,b));
  compare(m,i,ab,bb,6,checked(double(a),b),selected(double(a),b));
  compare(m,i,ab,bb,7,checked(a,double(b)),selected(a,double(b)));
  compare(m,i,ab,bb,8,checked(a,3),selected(a,3));
 }
}
NATIVE_HD void test_wide(Metrics& m,U index){
 const double wide[]={0x1.0000000000001p0,-0x1.0000000000001p0,0x1p-1000,0x1.fffffffffffffp1023};
 const float divisors[]={-1.0f,0.0f,0x1p-100f,0x1.fffffep127f};
 for(double a:wide)for(float b:divisors){
  compare(m,index,0,0,9,checked(a,b),selected(a,b));compare(m,index,0,0,10,checked(b,a),selected(b,a));
 }
}
#ifdef __CUDACC__
__global__ void qualify(Metrics* result,U total){Metrics m;
 for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=U(blockDim.x)*gridDim.x)test_pair(m,i);
 if(!blockIdx.x&&!threadIdx.x)test_wide(m,total);
#define ADD(field) atomicAdd(&result->field,m.field)
 ADD(pairs);ADD(comparisons);ADD(mismatches);ADD(status_mismatches);ADD(accepted_bit_mismatches);ADD(rejected_bit_differences);
#undef ADD
 if(m.first!=~U(0)&&atomicCAS(&result->first,~U(0),m.first)==~U(0)){result->a=m.a;result->b=m.b;result->kind=m.kind;result->expected=m.expected;result->actual=m.actual;result->expected_status=m.expected_status;result->actual_status=m.actual_status;}
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
int main(int argc,char** argv){try{
 U random=1000000;bool cpu=false;for(int i=1;i<argc;++i)if(std::string(argv[i])=="--cpu")cpu=true;else random=std::stoull(argv[i]);
 if(random>1000000000ULL)throw std::runtime_error("maximum one billion random pairs");
 if(std::fesetenv(FE_DFL_ENV))throw std::runtime_error("cannot select round nearest");
#if defined(__i386__)||defined(__x86_64__)
 const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
#endif
 constexpr U boundaries=U(boundary_count)*boundary_count;const U total=boundaries+random;Metrics m;
 if(cpu){for(U i=0;i<total;++i)test_pair(m,i);test_wide(m,total);}else{
#ifdef __CUDACC__
  Metrics* device{};check(cudaMalloc(&device,sizeof(Metrics)));check(cudaMemcpy(device,&m,sizeof(Metrics),cudaMemcpyHostToDevice));qualify<<<128,256>>>(device,total);check(cudaGetLastError());check(cudaMemcpy(&m,device,sizeof(Metrics),cudaMemcpyDeviceToHost));check(cudaFree(device));
#else
  throw std::runtime_error("host build requires --cpu");
#endif
 }
 std::printf("{\"schema\":\"irisu-f32-div-store-qualifier-v1\",\"passed\":%s,\"backend\":\"%s\",\"random_pairs\":%llu,\"boundary_pairs\":%llu,\"pairs\":%llu,\"comparisons\":%llu,\"mismatches\":%llu,\"status_mismatches\":%llu,\"accepted_bit_mismatches\":%llu,\"rejected_bit_differences\":%llu,\"first\":%llu,\"a_bits\":%u,\"b_bits\":%u,\"kind\":%u,\"expected_bits\":%u,\"actual_bits\":%u,\"expected_status\":%u,\"actual_status\":%u,\"scope\":\"Adjacent F32 quotients and exact integral numerators; checked generic fallbacks, all statuses, accepted result bits; no native-world or performance admission\"}\n",m.mismatches?"false":"true",cpu?"host":"cuda",random,boundaries,m.pairs,m.comparisons,m.mismatches,m.status_mismatches,m.accepted_bit_mismatches,m.rejected_bit_differences,m.first,m.a,m.b,m.kind,m.expected,m.actual,m.expected_status,m.actual_status);
 return m.mismatches||m.pairs!=total?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"div-store-validate: %s\n",e.what());return 2;}}
