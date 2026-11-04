// Compare the new temporary layout with unchanged solver stages, byte for byte.
#include "../../clone/cuda_native/solver_interleaved.hpp"
#include "solver_wire.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace n=irisu::cuda_native;
namespace s=n::interleaved_solver;
constexpr unsigned body_capacity=32,constraint_capacity=64;
struct Case {unsigned stage{},warm{},body_count{},constraint_count{};float beta{};n::BodyState bodies[body_capacity];n::ContactConstraint constraints[constraint_capacity];};
struct Outcome {n::SolverResult reference{},selected{};unsigned body_difference{n::null_index},constraint_difference{n::null_index};};
NATIVE_HD bool same(n::SolverResult a,n::SolverResult b){return a.status==b.status&&a.constraint==b.constraint&&a.point==b.point&&a.converged==b.converged;}
template<bool interleaved>NATIVE_HD n::SolverResult run(Case& c,s::View view,unsigned iterations){
 std::uint32_t order[body_capacity];for(unsigned i=0;i<c.body_count;++i)order[i]=c.body_count-1-i;
 if constexpr(interleaved){
  if(c.stage==1)return s::iterate<s::Phase::presolve>(view,c.bodies,c.body_count,order,c.constraints,1,c.beta,c.warm);
  if(c.stage==2)return s::iterate<s::Phase::velocity>(view,c.bodies,c.body_count,order,c.constraints,iterations);
  return s::iterate<s::Phase::position>(view,c.bodies,c.body_count,order,c.constraints,iterations,c.beta);
 }
 n::SolverResult result;
 for(unsigned i=0;i<(c.stage==1?1:iterations);++i){
  if constexpr(!interleaved){
   if(c.stage==1)result=n::solver_presolve(c.bodies,c.body_count,c.constraints,c.constraint_count,c.warm);
   else if(c.stage==2)result=n::solver_velocity(c.bodies,c.body_count,c.constraints,c.constraint_count);
   else result=n::solver_position(c.bodies,c.body_count,c.constraints,c.constraint_count,c.beta);
  }
  if(!result||(c.stage==3&&result.converged))break;
 }
 return result;
}
NATIVE_HD Outcome compare(Case& reference,Case& selected,s::View view,unsigned iterations){
 Outcome out;out.reference=run<false>(reference,{},iterations);out.selected=run<true>(selected,view,iterations);
 const auto* a=reinterpret_cast<const unsigned char*>(reference.bodies);const auto* b=reinterpret_cast<const unsigned char*>(selected.bodies);
 for(unsigned i=0;i<sizeof(reference.bodies);++i)if(a[i]!=b[i]){out.body_difference=i;break;}
 a=reinterpret_cast<const unsigned char*>(reference.constraints);b=reinterpret_cast<const unsigned char*>(selected.constraints);
 for(unsigned i=0;i<sizeof(reference.constraints);++i)if(a[i]!=b[i]){out.constraint_difference=i;break;}
 return out;
}
#ifdef __CUDACC__
__global__ void qualify(Case* reference,Case* selected,std::uint32_t* workspace,Outcome* outputs,unsigned count,unsigned iterations){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
 auto& c=selected[lane];s::View view{workspace,count,lane,body_capacity,constraint_capacity,c.body_count,c.constraint_count};outputs[lane]=compare(reference[lane],c,view,iterations);
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
void read(std::ifstream& input,void* out,std::size_t bytes){if(bytes&&!input.read(static_cast<char*>(out),bytes))throw std::runtime_error("short solver fixture");}
float f32(unsigned word){float result;std::memcpy(&result,&word,4);return result;}
std::vector<Case> load(const char* path){
 std::ifstream input(path,std::ios::binary);unsigned header[3];read(input,header,12);
 if(header[0]!=SOLVER_INPUT_MAGIC||header[1]!=1||header[2]>100000)throw std::runtime_error("solver fixture identity");
 std::vector<Case> cases;
 for(unsigned i=0;i<header[2];++i){SolverWireCase meta;read(input,&meta,sizeof(meta));
  if(meta.bodies>n::max_bodies||meta.constraints>n::max_contacts||meta.manifolds>n::max_contacts||meta.contacts>n::max_contacts)throw std::runtime_error("fixture counts");
  Case c;std::memset(&c,0xa5,sizeof(c));for(auto& b:c.bodies)b.occupied=b.pending_destroy=false;
  c.stage=meta.stage;c.warm=meta.warm;c.body_count=meta.bodies;c.constraint_count=meta.constraints;c.beta=f32(meta.beta);
  const bool admitted=meta.stage>=1&&meta.stage<=3&&meta.bodies<=body_capacity&&meta.constraints<=constraint_capacity;
  for(unsigned j=0;j<meta.bodies;++j){unsigned w[12];read(input,w,48);if(admitted){auto& b=c.bodies[j];b.position={f32(w[0]),f32(w[1])};b.rotation=f32(w[2]);b.R={{f32(w[3]),f32(w[4])},{f32(w[5]),f32(w[6])}};b.velocity={f32(w[7]),f32(w[8])};b.angular_velocity=f32(w[9]);b.inv_mass=f32(w[10]);b.inv_i=f32(w[11]);}}
  for(unsigned j=0;j<meta.constraints;++j){unsigned w[30];read(input,w,120);if(admitted){auto& cst=c.constraints[j];std::memcpy(&cst,w,120);cst.manifold=w[24];cst.body1=w[25];cst.body2=w[26];}}
  input.seekg(std::streamoff(meta.manifolds)*60+std::streamoff(meta.contacts)*20,std::ios::cur);if(!input)throw std::runtime_error("short solver fixture tail");
  if(admitted)cases.push_back(c);
 }
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing solver fixture");
 return cases;
}
int main(int argc,char** argv){try{
 if(argc<2||argc>4)throw std::runtime_error("usage: solver-interleaved-validate cases.bin [--gpu] [iterations]");
 bool gpu=false;unsigned iterations=8;for(int i=2;i<argc;++i)if(std::string(argv[i])=="--gpu")gpu=true;else iterations=unsigned(std::stoul(argv[i]));
 if(!iterations||iterations>1000)throw std::runtime_error("iteration bound");
 auto reference=load(argv[1]);if(reference.empty())throw std::runtime_error("no admitted solver fixtures");
 const unsigned native_cases=unsigned(reference.size());
 for(unsigned i=0;i<native_cases;++i)if(reference[i].constraint_count){
  auto failure=reference[i];const auto first=failure.constraints[0].body1;
  if(first>=failure.body_count)throw std::runtime_error("invalid captured body index");
  if(failure.stage==3){failure.bodies[first].rotation=0x1p20f;reference.push_back(failure);}
  else if(failure.stage==2){failure.bodies[first].velocity={0x1.fffffep127f,0x1.fffffep127f};failure.bodies[first].angular_velocity=0x1.fffffep127f;reference.push_back(failure);}
 }
 auto selected=reference;const unsigned count=unsigned(reference.size());std::vector<Outcome> outcomes(count);
 if(gpu){
#ifdef __CUDACC__
  Case *dr{},*ds{};Outcome* outputs{};std::uint32_t* words{};check(cudaMalloc(&dr,count*sizeof(Case)));check(cudaMalloc(&ds,count*sizeof(Case)));check(cudaMalloc(&outputs,count*sizeof(Outcome)));check(cudaMalloc(&words,s::words_required(count,body_capacity,constraint_capacity)*4));
  check(cudaMemcpy(dr,reference.data(),count*sizeof(Case),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,selected.data(),count*sizeof(Case),cudaMemcpyHostToDevice));
  qualify<<<(count+63)/64,64>>>(dr,ds,words,outputs,count,iterations);check(cudaGetLastError());check(cudaMemcpy(outcomes.data(),outputs,count*sizeof(Outcome),cudaMemcpyDeviceToHost));
  check(cudaFree(words));check(cudaFree(outputs));check(cudaFree(ds));check(cudaFree(dr));
#else
  throw std::runtime_error("CUDA unavailable");
#endif
 }else{
  std::vector<std::uint32_t> words(s::words_required(count,body_capacity,constraint_capacity));
  for(unsigned lane=0;lane<count;++lane){auto& c=selected[lane];s::View view{words.data(),count,lane,body_capacity,constraint_capacity,c.body_count,c.constraint_count};outcomes[lane]=compare(reference[lane],c,view,iterations);}
 }
 unsigned failures=0,rejections=0;
 for(unsigned i=0;i<count;++i){const auto& o=outcomes[i];rejections+=o.reference.status!=n::Status::ok;
  if(!same(o.reference,o.selected)||o.body_difference!=n::null_index||o.constraint_difference!=n::null_index){++failures;std::fprintf(stderr,"case %u stage %u: status %u/%u, body byte %u, constraint byte %u\n",i,reference[i].stage,unsigned(o.reference.status),unsigned(o.selected.status),o.body_difference,o.constraint_difference);}
 }
 if(!rejections)throw std::runtime_error("numeric failure fixtures did not reject");
 std::printf("{\"schema\":\"irisu-interleaved-solver-qualifier-v1\",\"passed\":%s,\"backend\":\"%s\",\"cases\":%u,\"native_captured_cases\":%u,\"iterations\":%u,\"rejections\":%u,\"mismatches\":%u,\"body_capacity\":%u,\"constraint_capacity\":%u,\"full_body_and_constraint_bytes_compared\":true,\"native_oracle_directly_compared\":false,\"scope\":\"Temporary layout versus unchanged solver stages on native-captured and numeric failure inputs; no rollout performance claim\"}\n",failures?"false":"true",gpu?"cuda":"host",count,native_cases,iterations,rejections,failures,body_capacity,constraint_capacity);
 return failures?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"solver-interleaved-validate: %s\n",e.what());return 2;}}
