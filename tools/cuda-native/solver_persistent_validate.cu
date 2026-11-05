// Native-captured stage inputs plus exact complete-island/failure differentials.
#include "../../clone/cuda_native/solver_persistent.hpp"
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
namespace n=irisu::cuda_native;namespace p=n::persistent_solver;
constexpr unsigned body_capacity=32,constraint_capacity=64;
struct Case {
 unsigned stage{},warm{},body_slots{},body_count{},constraint_count{};float beta{},dt{0x1.111112p-6f};
 unsigned order[body_capacity]{};n::BodyState bodies[body_capacity];n::ContactConstraint constraints[constraint_capacity];n::Manifold manifolds[constraint_capacity];
};
struct Outcome {
 n::SolverResult stage_reference{},stage_selected{};n::StageResult island_reference{},island_selected{};
 unsigned stage_body{n::null_index},stage_constraint{n::null_index},island_body{n::null_index},island_constraint{n::null_index},island_manifold{n::null_index};
 bool stage_used{},island_used{};
};
NATIVE_HD bool same(n::SolverResult a,n::SolverResult b){return a.status==b.status&&a.constraint==b.constraint&&a.point==b.point&&a.converged==b.converged;}
NATIVE_HD unsigned difference(const void* a,const void* b,unsigned bytes){
 const auto* x=static_cast<const unsigned char*>(a);const auto* y=static_cast<const unsigned char*>(b);
 for(unsigned i=0;i<bytes;++i)if(x[i]!=y[i])return i;return n::null_index;
}
NATIVE_HD n::SolverResult stage_original(Case& c,unsigned iterations){
 n::SolverResult result;
 for(unsigned i=0;i<(c.stage==1?1:iterations);++i){
  if(c.stage==1)result=n::solver_presolve(c.bodies,c.body_slots,c.constraints,c.constraint_count,c.warm);
  else if(c.stage==2)result=n::solver_velocity(c.bodies,c.body_slots,c.constraints,c.constraint_count);
  else result=n::solver_position(c.bodies,c.body_slots,c.constraints,c.constraint_count,c.beta);
  if(!result||(c.stage==3&&result.converged))break;
 }return result;
}
NATIVE_HD n::SolverResult stage_selected(Case& c,p::View view,unsigned iterations,bool& used){
 used=p::pack(view,c.bodies,c.body_slots,c.order,c.body_count,c.constraints,c.constraint_count);
 if(!used)return stage_original(c,iterations);
 n::SolverResult result;
 for(unsigned i=0;i<(c.stage==1?1:iterations);++i){
  if(c.stage==1)result=p::solve<p::Phase::presolve>(view,c.constraint_count,c.beta,c.warm);
  else if(c.stage==2)result=p::solve<p::Phase::velocity>(view,c.constraint_count);
  else result=p::solve<p::Phase::position>(view,c.constraint_count,c.beta);
  if(!result||(c.stage==3&&result.converged))break;
 }
 p::flush(view,c.bodies,c.order,c.body_count,c.constraints,c.constraint_count);return result;
}
NATIVE_HD n::StageResult island_original(Case& c,unsigned iterations,bool canonical){
 auto result=n::solver_presolve(c.bodies,c.body_slots,c.constraints,c.constraint_count,true);if(!result)return {result.status};
 for(unsigned i=0;i<iterations;++i){result=n::solver_velocity(c.bodies,c.body_slots,c.constraints,c.constraint_count);if(!result)return {result.status};}
 for(unsigned i=0;i<c.body_count;++i){const auto integrated=n::integrate_position(c.bodies[c.order[i]],c.dt,canonical);if(!integrated)return integrated;}
 for(unsigned i=0;i<iterations;++i){result=n::solver_position(c.bodies,c.body_slots,c.constraints,c.constraint_count,.2f,canonical);if(!result)return {result.status};if(result.converged)break;}
 result=n::solver_postsolve(c.constraints,c.constraint_count,c.manifolds,c.constraint_count);return {result.status};
}
NATIVE_HD void normalize_manifolds(Case& c){
 for(unsigned i=0;i<c.constraint_count;++i)c.constraints[i].manifold=i;
}
NATIVE_HD Outcome compare(Case& reference,Case& selected,Case& island_reference,Case& island_selected,p::View view,unsigned iterations,bool canonical){
 Outcome out;out.stage_reference=stage_original(reference,iterations);out.stage_selected=stage_selected(selected,view,iterations,out.stage_used);
 out.stage_body=difference(reference.bodies,selected.bodies,sizeof(reference.bodies));out.stage_constraint=difference(reference.constraints,selected.constraints,sizeof(reference.constraints));
 normalize_manifolds(island_reference);normalize_manifolds(island_selected);
 out.island_reference=island_original(island_reference,iterations,canonical);
 out.island_used=p::pack(view,island_selected.bodies,island_selected.body_slots,island_selected.order,island_selected.body_count,island_selected.constraints,island_selected.constraint_count);
 out.island_selected=out.island_used?p::island(view,island_selected.bodies,island_selected.order,island_selected.body_count,island_selected.constraints,island_selected.constraint_count,island_selected.manifolds,island_selected.constraint_count,island_selected.dt,iterations,canonical):island_original(island_selected,iterations,canonical);
 out.island_body=difference(island_reference.bodies,island_selected.bodies,sizeof(reference.bodies));out.island_constraint=difference(island_reference.constraints,island_selected.constraints,sizeof(reference.constraints));out.island_manifold=difference(island_reference.manifolds,island_selected.manifolds,sizeof(reference.manifolds));return out;
}
#ifdef __CUDACC__
__global__ void qualify(Case* reference,Case* selected,Case* island_reference,Case* island_selected,std::uint32_t* words,Outcome* out,unsigned count,unsigned iterations,bool canonical){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count)out[lane]=compare(reference[lane],selected[lane],island_reference[lane],island_selected[lane],{words,lane,count,body_capacity,constraint_capacity},iterations,canonical);
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
void read(std::ifstream& input,void* output,std::size_t bytes){if(bytes&&!input.read(static_cast<char*>(output),bytes))throw std::runtime_error("short native solver fixture");}
float f32(unsigned bits){float value;std::memcpy(&value,&bits,4);return value;}
std::vector<Case> load(const char* path){
 std::ifstream input(path,std::ios::binary);unsigned header[3];read(input,header,12);if(header[0]!=SOLVER_INPUT_MAGIC||header[1]!=1||header[2]>100000)throw std::runtime_error("solver fixture identity");
 std::vector<Case> cases;
 for(unsigned i=0;i<header[2];++i){SolverWireCase m;read(input,&m,sizeof(m));if(m.bodies>n::max_bodies||m.constraints>n::max_contacts||m.manifolds>n::max_contacts||m.contacts>n::max_contacts)throw std::runtime_error("fixture count bounds");
  Case c;std::memset(&c,0xa5,sizeof(c));c.stage=m.stage;c.warm=m.warm;c.body_slots=c.body_count=m.bodies;c.constraint_count=m.constraints;c.beta=f32(m.beta);c.dt=0x1.111112p-6f;
  for(unsigned j=0;j<body_capacity;++j){c.bodies[j].occupied=c.bodies[j].pending_destroy=false;c.bodies[j].id=1000+j;c.order[j]=j<c.body_count?c.body_count-j-1:0;}for(auto& manifold:c.manifolds)manifold=n::Manifold{};
  const bool admitted=m.stage>=1&&m.stage<=3&&m.bodies<=body_capacity&&m.constraints<=constraint_capacity;
  for(unsigned j=0;j<m.bodies;++j){unsigned w[12];read(input,w,48);if(admitted){auto& b=c.bodies[j];b.position={f32(w[0]),f32(w[1])};b.rotation=f32(w[2]);b.R={{f32(w[3]),f32(w[4])},{f32(w[5]),f32(w[6])}};b.velocity={f32(w[7]),f32(w[8])};b.angular_velocity=f32(w[9]);b.inv_mass=f32(w[10]);b.inv_i=f32(w[11]);}}
  for(unsigned j=0;j<m.constraints;++j){unsigned w[30];read(input,w,120);if(admitted){auto& con=c.constraints[j];std::memcpy(&con,w,120);con.manifold=w[24];con.body1=w[25];con.body2=w[26];}}
  input.seekg(std::streamoff(m.manifolds)*60+std::streamoff(m.contacts)*20,std::ios::cur);if(!input)throw std::runtime_error("short fixture tail");if(admitted)cases.push_back(c);
 }if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing fixture bytes");return cases;
}
int main(int argc,char** argv){try{
 if(argc<2||argc>5)throw std::runtime_error("usage: solver-persistent-validate cases.bin [--gpu] [--canonical] [iterations]");
 bool gpu=false,canonical=false;unsigned iterations=8;for(int i=2;i<argc;++i)if(std::string(argv[i])=="--gpu")gpu=true;else if(std::string(argv[i])=="--canonical")canonical=true;else iterations=unsigned(std::stoul(argv[i]));if(iterations>1000)throw std::runtime_error("iteration bound");
 auto reference=load(argv[1]);if(reference.empty())throw std::runtime_error("no native-captured inputs");const unsigned native=reference.size();
 for(unsigned i=0;i<native;++i){const auto original=reference[i];if(!original.constraint_count)continue;const unsigned first=original.constraints[0].body1;if(first>=original.body_slots)throw std::runtime_error("malformed captured participant");
  auto c=original;c.constraints[0].body1=c.body_slots;reference.push_back(c);c=original;c.constraints[0].body2=first;reference.push_back(c);c=original;c.constraints[0].point_count=3;reference.push_back(c);
  c=original;c.bodies[first].velocity={0x1.fffffep127f,0x1.fffffep127f};c.bodies[first].angular_velocity=0x1.fffffep127f;reference.push_back(c);
  c=original;c.bodies[first].rotation=0x1p20f;reference.push_back(c);c=original;c.bodies[first].R.col1.x=0x1.fffffep127f;c.constraints[0].points[0].local_anchor1.x=0x1.fffffep127f;reference.push_back(c);
  c=original;for(unsigned j=0;j<c.constraint_count;++j)c.constraints[j].point_count=0;reference.push_back(c);c=original;c.constraint_count=0;reference.push_back(c);
  if(original.constraint_count>1){c=original;c.constraints[c.constraint_count-1].body1=c.body_slots;reference.push_back(c);}
  c=original;for(unsigned j=0;j<original.body_slots;++j)c.bodies[body_capacity-1-j]=original.bodies[j];for(unsigned j=0;j<c.body_count;++j)c.order[j]=body_capacity-1-original.order[j];for(unsigned j=0;j<c.constraint_count;++j){c.constraints[j].body1=body_capacity-1-original.constraints[j].body1;c.constraints[j].body2=body_capacity-1-original.constraints[j].body2;}c.body_slots=body_capacity;reference.push_back(c);
 }
 auto selected=reference,island_reference=reference,island_selected=reference;const unsigned count=reference.size();std::vector<Outcome> outcomes(count);
 if(gpu){
#ifdef __CUDACC__
  Case *a{},*b{},*c{},*d{};Outcome* output{};unsigned* words{};check(cudaMalloc(&a,count*sizeof(Case)));check(cudaMalloc(&b,count*sizeof(Case)));check(cudaMalloc(&c,count*sizeof(Case)));check(cudaMalloc(&d,count*sizeof(Case)));check(cudaMalloc(&output,count*sizeof(Outcome)));check(cudaMalloc(&words,p::words_required(count,body_capacity,constraint_capacity)*4));
  // Struct assignment need not write ABI padding; the complete readback must be initialized.
  check(cudaMemset(output,0,count*sizeof(Outcome)));
  check(cudaMemcpy(a,reference.data(),count*sizeof(Case),cudaMemcpyHostToDevice));check(cudaMemcpy(b,selected.data(),count*sizeof(Case),cudaMemcpyHostToDevice));check(cudaMemcpy(c,island_reference.data(),count*sizeof(Case),cudaMemcpyHostToDevice));check(cudaMemcpy(d,island_selected.data(),count*sizeof(Case),cudaMemcpyHostToDevice));
  qualify<<<(count+63)/64,64>>>(a,b,c,d,words,output,count,iterations,canonical);check(cudaGetLastError());check(cudaMemcpy(outcomes.data(),output,count*sizeof(Outcome),cudaMemcpyDeviceToHost));check(cudaFree(words));check(cudaFree(output));check(cudaFree(d));check(cudaFree(c));check(cudaFree(b));check(cudaFree(a));
#else
  throw std::runtime_error("CUDA unavailable");
#endif
 }else{std::vector<unsigned> words(p::words_required(count,body_capacity,constraint_capacity));for(unsigned lane=0;lane<count;++lane)outcomes[lane]=compare(reference[lane],selected[lane],island_reference[lane],island_selected[lane],{words.data(),lane,count,body_capacity,constraint_capacity},iterations,canonical);}
 unsigned failures=0,rejections=0,used=0;
 for(unsigned i=0;i<count;++i){const auto& o=outcomes[i];used+=o.island_used;rejections+=o.island_reference.status!=n::Status::ok;
  if(!same(o.stage_reference,o.stage_selected)||o.island_reference.status!=o.island_selected.status||o.island_reference.body!=o.island_selected.body||o.stage_body!=n::null_index||o.stage_constraint!=n::null_index||o.island_body!=n::null_index||o.island_constraint!=n::null_index||o.island_manifold!=n::null_index){++failures;std::fprintf(stderr,"case%u stage%u: stageStatus%u/%u islandStatus%u/%u offsets%u,%u,%u,%u,%u\n",i,reference[i].stage,unsigned(o.stage_reference.status),unsigned(o.stage_selected.status),unsigned(o.island_reference.status),unsigned(o.island_selected.status),o.stage_body,o.stage_constraint,o.island_body,o.island_constraint,o.island_manifold);}
 }
 if(!used||used==count||!rejections)throw std::runtime_error("missing mirror/fallback/rejection witnesses");
 std::printf("{\"schema\":\"irisu-persistent-solver-qualifier-v1\",\"passed\":%s,\"backend\":\"%s\",\"native_captured_cases\":%u,\"cases\":%u,\"iterations\":%u,\"canonical_rotation\":%s,\"rejections\":%u,\"mirror_cases\":%u,\"fallback_cases\":%u,\"mismatches\":%u,\"full_body_constraint_manifold_bytes_compared\":true,\"scope\":\"Stable-slot tiled mirror versus original native-captured stage graphs and composed island sequence; no full-game/performance admission\"}\n",failures?"false":"true",gpu?"cuda":"host",native,count,iterations,canonical?"true":"false",rejections,used,count-used,failures);return failures?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"solver-persistent-validate: %s\n",e.what());return 2;}}
