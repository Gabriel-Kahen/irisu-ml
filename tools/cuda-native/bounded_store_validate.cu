// Bounded graph qualification, not a generic unchecked-double admission test.
#ifndef IRISU_CUDA_NATIVE_BOUNDED_MATH
#define IRISU_CUDA_NATIVE_BOUNDED_MATH 1
#endif
#ifndef IRISU_CUDA_NATIVE_BOUNDED_STORE
#define IRISU_CUDA_NATIVE_BOUNDED_STORE 1
#endif
#include "../../clone/cuda_native/solver_math.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
namespace n=irisu::cuda_native;
namespace x=irisu::cuda_exact;
struct Checked {
 n::Status status{n::Status::ok};
 __host__ __device__ double take(x::Number a){if(!a.supported)status=n::Status::unsupported_numeric;return a.value;}
 __host__ __device__ double add(double a,double b){return take(x::add(a,b));}
 __host__ __device__ double sub(double a,double b){return take(x::sub(a,b));}
 __host__ __device__ double mul(double a,double b){return take(x::mul(a,b));}
 __host__ __device__ double div(double a,double b){return take(x::div(a,b));}
 __host__ __device__ double sqrt(double a){return take(x::square_root(a));}
 __host__ __device__ float store(double a){auto v=n::math::store(a);if(!v.supported)status=n::Status::unsupported_numeric;return v.value;}
};
struct Result {n::Status status;unsigned value;unsigned long long raw;};
template<class M> __host__ __device__ Result graph(unsigned kind,float a,float b,float c,float d){
 M m;double raw=0;
 switch(kind){
 case 0:raw=m.add(m.mul(a,b),m.mul(c,d));break;
 case 1:raw=m.sub(m.mul(a,b),m.mul(a,b));break;
 case 2:raw=m.add(m.mul(m.sub(a,m.mul(b,c)),d),a);break;
 case 3:raw=m.mul(m.mul(m.mul(a,b),c),d);break;
 case 4:raw=m.add(m.mul(0x1.555556p-2f,m.mul(.25f,m.add(m.add(m.mul(a,a),m.mul(b,a)),m.mul(b,b)))),m.mul(m.mul(.5f,c),c));break;
 case 5:raw=m.div(a,m.sub(a,b));break;
 case 6:raw=m.add(m.sqrt(a),b);break;
 case 7:raw=m.div(m.div(a,2),b);break;
 case 8:raw=m.mul(m.div(1,a),0);break; // sticky rejected division cannot be masked by a finite store
 case 9:raw=m.add(m.sqrt(a),0);m.status=n::Status::unsupported_numeric;break;
 }
 const float value=m.store(raw);return {m.status,x::bits_float(value),x::bits_double(raw)};
}
struct Counts {unsigned long long comparisons{},accepted{},rejected{},status_mismatches{},accepted_bit_mismatches{},rejected_bit_differences{};};
__host__ __device__ unsigned mix(unsigned a){a^=a>>16;a*=0x7feb352dU;a^=a>>15;a*=0x846ca68bU;return a^(a>>16);}
__host__ __device__ float bits(unsigned a){union{unsigned u;float f;}v{a};return v.f;}
__global__ void qualify(const unsigned* boundary,unsigned boundary_count,unsigned random_count,Counts* total){
 Counts local{};const unsigned long long boundary_cases=static_cast<unsigned long long>(boundary_count)*boundary_count;
 const auto count=boundary_cases+random_count;
 for(unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){
  float a,b,c,d;
  if(i<boundary_cases){a=bits(boundary[i/boundary_count]);b=bits(boundary[i%boundary_count]);c=a;d=b;}
  else {const unsigned seed=unsigned(i-boundary_cases);a=bits(mix(seed*4));b=bits(mix(seed*4+1));c=bits(mix(seed*4+2));d=bits(mix(seed*4+3));}
  for(unsigned kind=0;kind<10;++kind){
   const auto ref=graph<Checked>(kind,a,b,c,d),out=graph<n::math::Context>(kind,a,b,c,d);
   ++local.comparisons;local.status_mismatches+=ref.status!=out.status;
   const bool different=ref.value!=out.value||ref.raw!=out.raw;
   if(ref.status==n::Status::ok&&out.status==n::Status::ok){++local.accepted;local.accepted_bit_mismatches+=different;}
   else {++local.rejected;local.rejected_bit_differences+=different;}
  }
 }
#define ACCUMULATE(field) atomicAdd(&total->field,local.field)
 ACCUMULATE(comparisons);ACCUMULATE(accepted);ACCUMULATE(rejected);ACCUMULATE(status_mismatches);ACCUMULATE(accepted_bit_mismatches);ACCUMULATE(rejected_bit_differences);
#undef ACCUMULATE
}
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
int main(int argc,char** argv){try{
 const unsigned random_count=argc>1?unsigned(std::stoul(argv[1])):1000000;
 std::vector<unsigned> boundary;
 for(unsigned e=0;e<256;++e)for(unsigned s=0;s<2;++s){boundary.push_back((s<<31)|(e<<23));boundary.push_back((s<<31)|(e<<23)|0x7fffff);}
 for(auto v:{1U,2U,0x007ffffeU,0x00800001U,0x3f7fffffU,0x3f800001U,0x7f7ffffeU,0x7fc00001U}){boundary.push_back(v);boundary.push_back(v|0x80000000U);}
 // The free host checker must still reject raw subnormal doubles.
 if(n::math::store(0x1p-1023).supported)throw std::runtime_error("checked store accepted subnormal double");
 unsigned* values{};Counts* counts{};check(cudaMalloc(&values,boundary.size()*sizeof(unsigned)));check(cudaMalloc(&counts,sizeof(Counts)));check(cudaMemset(counts,0,sizeof(Counts)));
 check(cudaMemcpy(values,boundary.data(),boundary.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
 qualify<<<128,256>>>(values,unsigned(boundary.size()),random_count,counts);check(cudaGetLastError());Counts result{};check(cudaMemcpy(&result,counts,sizeof(result),cudaMemcpyDeviceToHost));
 check(cudaFree(counts));check(cudaFree(values));
 const bool passed=!result.status_mismatches&&!result.accepted_bit_mismatches;
 std::printf("{\"schema\":\"irisu-bounded-store-qualifier-v1\",\"passed\":%s,\"bounded_math\":%d,\"bounded_store\":%d,\"random_inputs\":%u,\"boundary_inputs\":%zu,\"comparisons\":%llu,\"accepted\":%llu,\"rejected\":%llu,\"status_mismatches\":%llu,\"accepted_bit_mismatches\":%llu,\"rejected_bit_differences\":%llu,\"scope\":\"Ten bounded composed graphs, checked device reference, accepted F32 and raw F64 bits; no native-world admission\"}\n",passed?"true":"false",IRISU_CUDA_NATIVE_BOUNDED_MATH,IRISU_CUDA_NATIVE_BOUNDED_STORE,random_count,boundary.size()*boundary.size(),result.comparisons,result.accepted,result.rejected,result.status_mismatches,result.accepted_bit_mismatches,result.rejected_bit_differences);
 return passed?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"bounded-store-validate: %s\n",e.what());return 1;}}
