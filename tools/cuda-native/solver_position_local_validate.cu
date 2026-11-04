// Original and local position solvers on native-captured and failure fixtures.
#include "../../clone/cuda_native/solver_position_local.hpp"
#include "solver_wire.h"
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
constexpr unsigned body_capacity=32,constraint_capacity=64;
struct Case {unsigned body_count{},constraint_count{};float beta{};n::BodyState bodies[body_capacity];n::ContactConstraint constraints[constraint_capacity];};
struct Outcome {n::SolverResult reference{},selected{};unsigned body_difference{n::null_index},constraint_difference{n::null_index};};
NATIVE_HD bool same(n::SolverResult a,n::SolverResult b){return a.status==b.status&&a.constraint==b.constraint&&a.point==b.point&&a.converged==b.converged;}
NATIVE_HD Outcome compare(Case& reference,Case& selected,unsigned iterations,bool canonical){
 Outcome out;
 for(unsigned i=0;i<iterations;++i){out.reference=n::solver_position(reference.bodies,reference.body_count,reference.constraints,reference.constraint_count,reference.beta,canonical);if(!out.reference||out.reference.converged)break;}
 for(unsigned i=0;i<iterations;++i){out.selected=n::solver_position_local(selected.bodies,selected.body_count,selected.constraints,selected.constraint_count,selected.beta,canonical);if(!out.selected||out.selected.converged)break;}
 const auto* a=reinterpret_cast<const unsigned char*>(reference.bodies);const auto* b=reinterpret_cast<const unsigned char*>(selected.bodies);
 for(unsigned i=0;i<sizeof(reference.bodies);++i)if(a[i]!=b[i]){out.body_difference=i;break;}
 a=reinterpret_cast<const unsigned char*>(reference.constraints);b=reinterpret_cast<const unsigned char*>(selected.constraints);
 for(unsigned i=0;i<sizeof(reference.constraints);++i)if(a[i]!=b[i]){out.constraint_difference=i;break;}
 return out;
}
#ifdef __CUDACC__
__global__ void qualify(Case* reference,Case* selected,Outcome* outputs,unsigned count,unsigned iterations,bool canonical){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count)outputs[lane]=compare(reference[lane],selected[lane],iterations,canonical);
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
  c.body_count=meta.bodies;c.constraint_count=meta.constraints;c.beta=f32(meta.beta);
  const bool admitted=meta.stage==3&&meta.bodies<=body_capacity&&meta.constraints<=constraint_capacity;
  for(unsigned j=0;j<meta.bodies;++j){unsigned w[12];read(input,w,48);if(admitted){auto& b=c.bodies[j];b.position={f32(w[0]),f32(w[1])};b.rotation=f32(w[2]);b.R={{f32(w[3]),f32(w[4])},{f32(w[5]),f32(w[6])}};b.velocity={f32(w[7]),f32(w[8])};b.angular_velocity=f32(w[9]);b.inv_mass=f32(w[10]);b.inv_i=f32(w[11]);}}
  for(unsigned j=0;j<meta.constraints;++j){unsigned w[30];read(input,w,120);if(admitted){auto& cst=c.constraints[j];std::memcpy(&cst,w,120);cst.manifold=w[24];cst.body1=w[25];cst.body2=w[26];}}
  input.seekg(std::streamoff(meta.manifolds)*60+std::streamoff(meta.contacts)*20,std::ios::cur);if(!input)throw std::runtime_error("short solver fixture tail");
  if(admitted)cases.push_back(c);
 }
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing solver fixture");
 return cases;
}
int main(int argc,char** argv){try{
 if(argc<2||argc>5)throw std::runtime_error("usage: solver-position-local-validate cases.bin [--gpu] [--canonical] [iterations]");
 bool gpu=false,canonical=false;unsigned iterations=10;
 for(int i=2;i<argc;++i){const std::string arg=argv[i];if(arg=="--gpu")gpu=true;else if(arg=="--canonical")canonical=true;else iterations=unsigned(std::stoul(arg));}
 if(!iterations||iterations>1000)throw std::runtime_error("iteration bound");
 auto reference=load(argv[1]);if(reference.empty())throw std::runtime_error("no admitted solver fixtures");
 const unsigned native_cases=unsigned(reference.size());
 for(unsigned i=0;i<native_cases;++i)if(reference[i].constraint_count){
  const auto first=reference[i].constraints[0].body1;
  if(first>=reference[i].body_count)throw std::runtime_error("invalid captured body index");
  auto rotation=reference[i];rotation.bodies[first].rotation=0x1p20f;reference.push_back(rotation);
  auto anchor=reference[i];anchor.bodies[first].R.col1.x=0x1.fffffep127f;anchor.constraints[0].points[0].local_anchor1.x=0x1.fffffep127f;reference.push_back(anchor);
  auto invalid=reference[i];invalid.constraints[0].body1=invalid.body_count;reference.push_back(invalid);
  auto alias=reference[i];alias.constraints[0].body2=alias.constraints[0].body1;reference.push_back(alias);
  auto malformed=reference[i];malformed.constraints[0].point_count=3;reference.push_back(malformed);
  auto zero=reference[i];zero.constraints[0].point_count=0;reference.push_back(zero);
  auto partial=reference[i];partial.bodies[first].inv_i=0x1.fffffep127f;partial.constraints[0].points[0].separation=-1;reference.push_back(partial);
  auto late=reference[i];late.constraints[late.constraint_count-1].point_count=3;reference.push_back(late);
 }
 auto selected=reference;const unsigned count=unsigned(reference.size());std::vector<Outcome> outcomes(count);
 if(gpu){
#ifdef __CUDACC__
  Case *dr{},*ds{};Outcome* outputs{};check(cudaMalloc(&dr,count*sizeof(Case)));check(cudaMalloc(&ds,count*sizeof(Case)));check(cudaMalloc(&outputs,count*sizeof(Outcome)));
  check(cudaMemcpy(dr,reference.data(),count*sizeof(Case),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,selected.data(),count*sizeof(Case),cudaMemcpyHostToDevice));
  qualify<<<(count+63)/64,64>>>(dr,ds,outputs,count,iterations,canonical);check(cudaGetLastError());check(cudaMemcpy(outcomes.data(),outputs,count*sizeof(Outcome),cudaMemcpyDeviceToHost));
  check(cudaFree(outputs));check(cudaFree(ds));check(cudaFree(dr));
#else
  throw std::runtime_error("CUDA unavailable");
#endif
 }else for(unsigned lane=0;lane<count;++lane)outcomes[lane]=compare(reference[lane],selected[lane],iterations,canonical);
 unsigned failures=0,rejections=0;
 for(unsigned i=0;i<count;++i){const auto& o=outcomes[i];rejections+=o.reference.status!=n::Status::ok;
  if(!same(o.reference,o.selected)||o.body_difference!=n::null_index||o.constraint_difference!=n::null_index){++failures;std::fprintf(stderr,"case %u: status %u/%u, body byte %u, constraint byte %u\n",i,unsigned(o.reference.status),unsigned(o.selected.status),o.body_difference,o.constraint_difference);}
 }
 if(!rejections)throw std::runtime_error("missing rejection witnesses");
 std::printf("{\"schema\":\"irisu-local-position-qualifier-v1\",\"passed\":%s,\"backend\":\"%s\",\"canonical_rotation\":%s,\"cases\":%u,\"native_captured_cases\":%u,\"iterations\":%u,\"rejections\":%u,\"mismatches\":%u,\"full_body_and_constraint_bytes_compared\":true,\"scope\":\"Local position versus unchanged repeated stage on native-captured and invalid/numeric-failure inputs; no full-world admission\"}\n",failures?"false":"true",gpu?"cuda":"host",canonical?"true":"false",count,native_cases,iterations,rejections,failures);
 return failures?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"solver-position-local-validate: %s\n",e.what());return 2;}}
