// Standalone qualifier deliberately selects the new stored-float overloads
// against the fully checked original Context graph, not bounded intermediates.
#ifdef IRISU_CUDA_NATIVE_F32_STORE
#undef IRISU_CUDA_NATIVE_F32_STORE
#endif
#define IRISU_CUDA_NATIVE_F32_STORE 1
#ifdef IRISU_CUDA_NATIVE_BOUNDED_MATH
#undef IRISU_CUDA_NATIVE_BOUNDED_MATH
#endif
#define IRISU_CUDA_NATIVE_BOUNDED_MATH 0
#include "../../clone/cuda_native/solver_math.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
namespace n=irisu::cuda_native;
using U=unsigned long long;
struct Outcome {unsigned bits{},status{};};
struct Metrics {U pairs{},comparisons{},mismatches{},status_mismatches{},value_mismatches{},rejected_value_mismatches{},first{~U(0)};unsigned a{},b{},operation{},expected{},actual{},expected_status{},actual_status{};};
NATIVE_HD float value(unsigned bits){union{unsigned u;float f;}v{bits};return v.f;}
NATIVE_HD U mix(U x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
constexpr unsigned boundary_count=1056;
NATIVE_HD unsigned boundary(unsigned i){
 if(i<1024)return ((i&1U)<<31)|(((i>>2)&255U)<<23)|((i&2U)?0x7fffffU:0);
 const unsigned extra[]={1,2,3,0x800001,0x800002,0x3f800001,0x3f800002,0x3f7fffff,0x337fffff,0x33800000,0x33800001,0x25000000,0x7f7ffffe,0x7f800001,0x7fc12345,0x7ffffffe};
 i-=1024;return extra[i&15U]|((i&16U)?0x80000000U:0);
}
NATIVE_HD void pair_at(U index,unsigned& a,unsigned& b){
 constexpr U boundaries=U(boundary_count)*boundary_count;
 if(index<boundaries){a=boundary(unsigned(index/boundary_count));b=boundary(unsigned(index%boundary_count));}
 else{index-=boundaries;a=unsigned(mix(index^0x123456789abcdef0ULL));b=unsigned(mix(index^0xfedcba9876543210ULL)>>32);}
}
template<class A,class B>NATIVE_HD Outcome checked(A a,B b,unsigned op,bool failed=false){
 n::math::Context m;if(failed)m.status=n::Status::unsupported_numeric;
 const float x=op==0?m.store(m.mul(double(a),double(b))):op==1?m.store(m.add(double(a),double(b))):m.store(m.sub(double(a),double(b)));
 return {irisu::cuda_exact::bits_float(x),unsigned(m.status)};
}
template<class A,class B>NATIVE_HD Outcome candidate(A a,B b,unsigned op,bool failed=false){
 n::math::Context m;if(failed)m.status=n::Status::unsupported_numeric;
 const float x=op==0?m.mul_store(a,b):op==1?m.add_store(a,b):m.sub_store(a,b);
 return {irisu::cuda_exact::bits_float(x),unsigned(m.status)};
}
NATIVE_HD void compare(Metrics& m,U index,unsigned a,unsigned b,unsigned op,Outcome reference,Outcome result){
 ++m.comparisons;const bool bad_status=reference.status!=result.status;
 const bool accepted=reference.status==unsigned(n::Status::ok)&&result.status==unsigned(n::Status::ok);
 const bool different=reference.bits!=result.bits,bad_value=accepted&&different;
 m.status_mismatches+=bad_status;m.value_mismatches+=bad_value;m.rejected_value_mismatches+=!accepted&&different;
 if(bad_status||bad_value){++m.mismatches;if(m.first==~U(0)){m.first=index;m.a=a;m.b=b;m.operation=op;m.expected=reference.bits;m.actual=result.bits;m.expected_status=reference.status;m.actual_status=result.status;}}
}
NATIVE_HD void test_pair(Metrics& m,U index){unsigned ab,bb;pair_at(index,ab,bb);const float a=value(ab),b=value(bb);++m.pairs;
 for(unsigned op=0;op<3;++op){compare(m,index,ab,bb,op,checked(a,b,op),candidate(a,b,op));
  if(index<4096){compare(m,index,ab,bb,op+3,checked(a,b,op,true),candidate(a,b,op,true));
   compare(m,index,ab,bb,op+6,checked(double(a),b,op),candidate(double(a),b,op));
   compare(m,index,ab,bb,op+9,checked(a,double(b),op),candidate(a,double(b),op));
   compare(m,index,ab,bb,op+12,checked(16777217,b,op),candidate(16777217,b,op));}}
}
#ifdef __CUDACC__
__global__ void qualify(Metrics* totals,U count){Metrics local;
 for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=U(blockDim.x)*gridDim.x)test_pair(local,i);
 atomicAdd(&totals->pairs,local.pairs);atomicAdd(&totals->comparisons,local.comparisons);atomicAdd(&totals->mismatches,local.mismatches);atomicAdd(&totals->status_mismatches,local.status_mismatches);atomicAdd(&totals->value_mismatches,local.value_mismatches);atomicAdd(&totals->rejected_value_mismatches,local.rejected_value_mismatches);
 if(local.first!=~U(0)&&atomicCAS(&totals->first,~U(0),local.first)==~U(0)){totals->a=local.a;totals->b=local.b;totals->operation=local.operation;totals->expected=local.expected;totals->actual=local.actual;totals->expected_status=local.expected_status;totals->actual_status=local.actual_status;}
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
#endif
int main(int argc,char** argv){try{
 U random=1000000;bool cpu=false;for(int i=1;i<argc;++i){if(std::string(argv[i])=="--cpu")cpu=true;else random=std::stoull(argv[i]);}
 if(random>1000000000ULL)throw std::runtime_error("maximum one billion random pairs");
 if(std::fesetenv(FE_DFL_ENV))throw std::runtime_error("cannot select round nearest");
#if defined(__i386__)||defined(__x86_64__)
 const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
#endif
 constexpr U boundaries=U(boundary_count)*boundary_count;const U total=boundaries+random;Metrics m;
 if(cpu){for(U i=0;i<total;++i)test_pair(m,i);}
 else{
#ifdef __CUDACC__
  Metrics* device{};check(cudaMalloc(&device,sizeof(Metrics)));check(cudaMemcpy(device,&m,sizeof(m),cudaMemcpyHostToDevice));qualify<<<128,256>>>(device,total);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(&m,device,sizeof(m),cudaMemcpyDeviceToHost));check(cudaFree(device));
#else
  throw std::runtime_error("host build requires --cpu");
#endif
 }
 // Mixed-width values that cannot be narrowed to float before the operation.
 const double wide[]={0x1.0000000000001p0,-0x1.0000000000001p0,0x1p-1000,0x1.fffffffffffffp1023};
 for(double a:wide)for(float b:{-1.0f,0.0f,0x1p-100f,0x1.fffffep127f})for(unsigned op=0;op<3;++op)compare(m,total,0,0,op+20,checked(a,b,op),candidate(a,b,op));
 std::printf("{\"profile\":\"f32-adjacent-store-v1\",\"backend\":\"%s\",\"random_pairs\":%llu,\"boundary_pairs\":%llu,\"pairs\":%llu,\"comparisons\":%llu,\"mismatches\":%llu,\"status_mismatches\":%llu,\"value_mismatches\":%llu,\"rejected_value_mismatches\":%llu,\"value_contract\":\"accepted-Context-results-only; all-statuses-checked\",\"first\":%llu,\"a_bits\":%u,\"b_bits\":%u,\"operation\":%u,\"expected_bits\":%u,\"actual_bits\":%u,\"expected_status\":%u,\"actual_status\":%u,\"nonfinite_inputs\":\"original-checked-graph-fallback\",\"ftz_required\":false}\n",cpu?"host":"cuda",random,boundaries,m.pairs,m.comparisons,m.mismatches,m.status_mismatches,m.value_mismatches,m.rejected_value_mismatches,m.first,m.a,m.b,m.operation,m.expected,m.actual,m.expected_status,m.actual_status);
 return m.mismatches||m.pairs!=total?2:0;
}catch(const std::exception& e){std::fprintf(stderr,"store_math_validate: %s\n",e.what());return 1;}}
