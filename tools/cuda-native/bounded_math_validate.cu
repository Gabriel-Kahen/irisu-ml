// Numerical qualification only; no native-world/search admission or timing claim.
#include "../../clone/cuda_native/collision.hpp"
#include "../../clone/cuda_native/solver.hpp"
#include <cuda_runtime.h>
#include <cfenv>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace n=irisu::cuda_native;
namespace x=irisu::cuda_exact;
struct Case {unsigned kind{};float v[8]{};};
struct Result {n::Status status{n::Status::ok};unsigned first{},second{};unsigned long long raw{};};
// This reference never calls the optionally substituted Context arithmetic.
struct Checked {
  n::Status status{n::Status::ok};
  __host__ __device__ double take(x::Number value){if(!value.supported)status=n::Status::unsupported_numeric;return value.value;}
  __host__ __device__ double add(double a,double b){return take(x::add(a,b));}
  __host__ __device__ double sub(double a,double b){return take(x::sub(a,b));}
  __host__ __device__ double mul(double a,double b){return take(x::mul(a,b));}
  __host__ __device__ double div(double a,double b){return take(x::div(a,b));}
  __host__ __device__ double sqrt(double a){return take(x::square_root(a));}
  __host__ __device__ float store(double value){auto out=n::math::store(value);if(!out.supported)status=n::Status::unsupported_numeric;return out.value;}
};
__host__ __device__ unsigned long long manifold_hash(const n::Manifold& m){
  auto* bytes=reinterpret_cast<const unsigned char*>(&m);unsigned long long hash=14695981039346656037ULL;
  for(unsigned i=0;i<sizeof(m);++i){hash^=bytes[i];hash*=1099511628211ULL;}return hash;
}
template<class Math> __host__ __device__ Result evaluate(const Case& c){
  Math m;double raw=0;const auto* v=c.v;
  switch(c.kind){
    case 0:raw=m.add(v[0],v[1]);break;
    case 1:raw=m.mul(v[0],v[1]);break;
    case 2:raw=m.add(m.mul(v[0],v[0]),m.mul(v[1],v[1]));break;
    case 3:raw=m.add(m.mul(m.mul(v[0],v[1]),v[2]),v[3]);break;
    case 4:raw=m.mul(m.mul(m.mul(v[0],v[1]),v[2]),v[3]);break;
    case 5:raw=m.add(m.mul(m.sub(v[0],m.mul(v[1],v[1])),v[2]),v[3]);break;
    case 6:raw=m.div(v[0],m.sub(v[0],v[1]));break;
    case 7:raw=m.div(m.div(v[0],2),v[1]);break;
    case 8:raw=m.sqrt(v[0]);break;
    case 9:raw=m.div(1,v[0]);break;
    case 10:{const auto pair=n::math::sincos(v[0]);return {pair.supported?n::Status::ok:n::Status::unsupported_numeric,x::bits_float(pair.sine),x::bits_float(pair.cosine),0};}
    case 20:{
      n::BodyState bodies[2]{};bodies[0].inv_mass=v[0];bodies[1].inv_mass=v[1];
      bodies[0].inv_i=v[2];bodies[1].inv_i=v[3];
      n::Manifold manifold{};manifold.point_count=1;manifold.normal={0,1};
      n::SolverContact contact{0,1,0,.2f,0};n::ContactConstraint constraint{};unsigned count=0;
      const auto result=n::solver_build_constraints(bodies,2,&manifold,1,&contact,1,&constraint,1,count);
      return {result.status,x::bits_float(constraint.points[0].normal_mass),x::bits_float(constraint.points[0].tangent_mass),count};
    }
    case 21:{
      n::ShapeState a{},b{};a.kind=b.kind=n::ShapeKind::circle;
      a.position={v[0],v[1]};b.position={v[2],v[3]};a.radius=v[4];b.radius=v[5];
      n::Manifold manifold{};const auto result=n::collide_circle(manifold,a,b);
      return {result.status,unsigned(manifold.point_count),0,manifold_hash(manifold)};
    }
    default:return {n::Status::invalid_snapshot};
  }
  const auto stored=m.store(raw);return {m.status,x::bits_float(stored),0,x::bits_double(raw)};
}
__global__ void qualify(const Case* cases,Result* selected,Result* checked,unsigned count){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){selected[i]=evaluate<n::math::Context>(cases[i]);checked[i]=evaluate<Checked>(cases[i]);}
}
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
bool same(Result a,Result b){
  return a.status==b.status&&(a.status!=n::Status::ok||
    (a.first==b.first&&a.second==b.second&&a.raw==b.raw));
}
int main(){try{
  if(std::fesetround(FE_TONEAREST))throw std::runtime_error("cannot install round-to-nearest");
  constexpr float q=0x1p-149f,large=0x1.fffffep127f;
  std::vector<Case> cases;std::vector<std::string> labels;
  auto add=[&](const char* label,unsigned kind,float a=0,float b=0,float c=0,float d=0,float e=0,float f=0){cases.push_back({kind,{a,b,c,d,e,f,0,0}});labels.emplace_back(label);};
  add("signed-zero-add",0,-0.0f,-0.0f);add("subnormal-add",0,q,q);
  add("f32-add-overflow",0,large,large);add("f32-product-overflow",1,large,2);
  add("subnormal-square-to-f32-zero",1,q,q);add("subnormal-dot-to-f32-zero",2,q,q);
  add("normal-dot",2,.125f,-.25f);add("three-factor-grid",3,q,q,q,0);
  add("four-factor-grid",4,q,q,q,q);add("four-factor-f32-overflow",4,large,large,large,large);
  add("rotational-cancellation",5,1,0x1.000002p0f,0x1p22f,1);
  add("rotational-exact-zero",5,1,1.5f,4,5);
  add("clip-subnormal-ratio",6,q,-q);add("clip-large-ratio",6,large,-large);
  add("clip-zero-denominator",6,1,1);add("wrapper-small",7,q,large);
  add("wrapper-f32-overflow",7,large,q);add("wrapper-zero-scale",7,1,0);
  add("sqrt-zero",8,0);add("sqrt-negative-zero",8,-0.0f);add("sqrt-subnormal",8,q);
  add("sqrt-maximum",8,large);add("sqrt-negative",8,-1);
  add("reciprocal-one",9,1);add("reciprocal-zero",9,0);add("reciprocal-negative-zero",9,-0.0f);
  add("reciprocal-subnormal-f32-overflow",9,q);add("reciprocal-max-subnormal-result",9,large);
  add("trig-subnormal",10,q);add("trig-negative-zero",10,-0.0f);
  add("trig-largest-admitted",10,0x1.fffffep18f);add("trig-limit",10,0x1p19f);
  add("trig-negative-limit",10,-0x1p19f);add("trig-finite-outside-domain",10,large);
  add("solver-positive-effective-mass",20,1,1,0,0);add("solver-zero-effective-mass",20,0,0,0,0);
  add("solver-effective-mass-f32-overflow",20,large,large,0,0);
  add("solver-reciprocal-f32-overflow",20,q,0,0,0);
  add("circle-coincident",21,0,0,0,0,1,1);add("circle-subnormal",21,0,0,q,0,q,q);
  add("circle-tangent",21,0,0,2,0,1,1);add("circle-negative-radius",21,0,0,1,0,-1,1);
  add("circle-difference-overflow",21,-large,0,large,0,1,1);
  add("circle-distance-square-overflow",21,0,0,0x1p65f,0,1,1);
  add("circle-radius-sum-overflow",21,0,0,0,0,large,large);
  add("sqrt-nan",8,std::numeric_limits<float>::quiet_NaN());
  add("reciprocal-inf",9,std::numeric_limits<float>::infinity());
  add("circle-nan-input",21,0,0,0,0,std::numeric_limits<float>::quiet_NaN(),1);
  // Deterministic bounded polynomial inputs, not eight-factor generic chains.
  unsigned state=661003;
  for(unsigned i=0;i<1024;++i){float v[4];for(auto& f:v){state=1664525U*state+1013904223U;
      const unsigned bits=((state&1U)<<31)|((107U+(state%41U))<<23)|((state>>1)&0x7fffffU);
      union{unsigned u;float f;} value{bits};f=value.f;}
    add("bounded-polynomial",i%6,v[0],v[1],v[2],v[3]);}
  std::vector<Result> expected(cases.size()),selected(cases.size()),checked(cases.size());
  for(unsigned i=0;i<cases.size();++i)expected[i]=evaluate<Checked>(cases[i]);
  const char* numeric_rejections[]={"f32-add-overflow","f32-product-overflow","four-factor-f32-overflow",
    "clip-zero-denominator","wrapper-f32-overflow","wrapper-zero-scale","sqrt-negative",
    "reciprocal-zero","reciprocal-negative-zero","reciprocal-subnormal-f32-overflow",
    "trig-limit","trig-negative-limit","trig-finite-outside-domain","solver-zero-effective-mass",
    "solver-effective-mass-f32-overflow","solver-reciprocal-f32-overflow","circle-difference-overflow",
    "circle-distance-square-overflow","circle-radius-sum-overflow","sqrt-nan","reciprocal-inf","circle-nan-input"};
  for(const auto* label:numeric_rejections){bool found=false;
    for(unsigned i=0;i<labels.size();++i)if(labels[i]==label){found=true;
      if(expected[i].status!=n::Status::unsupported_numeric)throw std::runtime_error(std::string("negative fixture did not reject: ")+label);}
    if(!found)throw std::runtime_error("missing negative fixture");
  }
  for(unsigned i=0;i<labels.size();++i){
    if(labels[i]=="circle-negative-radius"&&expected[i].status!=n::Status::invalid_shape)
      throw std::runtime_error("negative radius fixture did not reject");
    if(labels[i]=="reciprocal-max-subnormal-result"&&
       (expected[i].status!=n::Status::ok||expected[i].first!=0x00200000U))
      throw std::runtime_error("subnormal store fixture did not retain exact value");
  }
  Case* dc{};Result *ds{},*dr{};check(cudaMalloc(&dc,cases.size()*sizeof(Case)));
  check(cudaMalloc(&ds,cases.size()*sizeof(Result)));check(cudaMalloc(&dr,cases.size()*sizeof(Result)));
  check(cudaMemcpy(dc,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice));
  qualify<<<(cases.size()+63)/64,64>>>(dc,ds,dr,unsigned(cases.size()));check(cudaGetLastError());
  check(cudaMemcpy(selected.data(),ds,selected.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(checked.data(),dr,checked.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  check(cudaFree(dr));check(cudaFree(ds));check(cudaFree(dc));
  unsigned passed=0,rejected=0;
  for(unsigned i=0;i<cases.size();++i){
    if(!same(expected[i],checked[i])||!same(expected[i],selected[i])){
      std::fprintf(stderr,"case %u (%s), kind %u: host/checked-device/selected status %u/%u/%u; output %08x/%08x/%08x; raw %016llx/%016llx/%016llx\n",
        i,labels[i].c_str(),cases[i].kind,unsigned(expected[i].status),unsigned(checked[i].status),unsigned(selected[i].status),
        expected[i].first,checked[i].first,selected[i].first,expected[i].raw,checked[i].raw,selected[i].raw);
      return 1;
    }
    ++passed;rejected+=expected[i].status!=n::Status::ok;
  }
#if defined(IRISU_CUDA_NATIVE_BOUNDED_MATH) && IRISU_CUDA_NATIVE_BOUNDED_MATH
  constexpr bool bounded=true;
#else
  constexpr bool bounded=false;
#endif
  std::printf("{\"schema\":\"irisu-bounded-math-qualifier-v1\",\"passed\":true,\"bounded_math_compiled\":%s,\"cases\":%u,\"rejected_cases\":%u,\"accepted_cases\":%u,\"host_checked_and_device_checked_equal\":true,\"selected_device_equal\":true,\"failure_output_bits_compared\":false,\"scope\":\"Numerical boundary graphs plus circle and constraint construction; no full native physics, game, search or performance admission\"}\n",
    bounded?"true":"false",passed,rejected,passed-rejected);
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"bounded-math-validate: %s\n",e.what());return 1;}}
