// Independent typed solver packet runner. Guest conversion is not linked.
#include "../../clone/cuda_native/solver.hpp"
#include "solver_wire.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace n=irisu::cuda_native;
struct Plan {SolverWireCase meta;std::size_t body,constraint,manifold,contact;};
struct View {SolverWireCase meta;n::BodyState* bodies;n::ContactConstraint* constraints;n::Manifold* manifolds;n::SolverContact* contacts;};
NATIVE_HD inline SolverWireResult solve(View v){
 auto& m=v.meta;n::SolverResult result;std::uint32_t count=m.constraints;union{std::uint32_t u;float f;}beta{m.beta};
 if(m.stage==0)result=n::solver_build_constraints(v.bodies,m.bodies,v.manifolds,m.manifolds,v.contacts,m.contacts,v.constraints,m.constraints,count);
 if(m.stage==1)result=n::solver_presolve(v.bodies,m.bodies,v.constraints,m.constraints,m.warm);
 if(m.stage==2)result=n::solver_velocity(v.bodies,m.bodies,v.constraints,m.constraints);
 if(m.stage==3)result=n::solver_position(v.bodies,m.bodies,v.constraints,m.constraints,beta.f);
 if(m.stage==4)result=n::solver_postsolve(v.constraints,m.constraints,v.manifolds,m.manifolds);
 return {unsigned(result.status),count,result.converged};
}
#ifdef __CUDACC__
__global__ void run_solver(View* cases,SolverWireResult* results,unsigned count){const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)results[i]=solve(cases[i]);}
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,std::max<std::size_t>(1,v.size())*sizeof(T)));if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>void download(std::vector<T>& v,T* p){if(!v.empty())check(cudaMemcpy(v.data(),p,v.size()*sizeof(T),cudaMemcpyDeviceToHost));}
#endif
static float float_word(std::uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
static std::uint32_t word_float(float f){std::uint32_t u;std::memcpy(&u,&f,4);return u;}
static void read(std::ifstream& f,void* destination,std::size_t bytes){if(bytes&&!f.read(static_cast<char*>(destination),bytes))throw std::runtime_error("short solver packet");}
static void words(const void* p,std::size_t count){if(count&&std::fwrite(p,4,count,stdout)!=count)throw std::runtime_error("solver output failure");}
static n::BodyState body(const std::uint32_t* p){n::BodyState b{};b.position={float_word(p[0]),float_word(p[1])};b.rotation=float_word(p[2]);b.R={{float_word(p[3]),float_word(p[4])},{float_word(p[5]),float_word(p[6])}};b.velocity={float_word(p[7]),float_word(p[8])};b.angular_velocity=float_word(p[9]);b.inv_mass=float_word(p[10]);b.inv_i=float_word(p[11]);return b;}
static void body_words(const n::BodyState& b){std::uint32_t data[12];unsigned i=0;for(float f:{b.position.x,b.position.y,b.rotation,b.R.col1.x,b.R.col1.y,b.R.col2.x,b.R.col2.y,b.velocity.x,b.velocity.y,b.angular_velocity,b.inv_mass,b.inv_i})data[i++]=word_float(f);words(data,12);}
static n::ContactConstraint constraint(const std::uint32_t* p){static_assert(sizeof(n::ContactConstraint)==120);n::ContactConstraint c;std::memcpy(&c,p,120);c.manifold=p[24];c.body1=p[25];c.body2=p[26];return c;}
static void constraint_words(const n::ContactConstraint& c){std::uint32_t p[30];std::memcpy(p,&c,120);p[24]=c.manifold;p[25]=c.body1;p[26]=c.body2;words(p,30);}
int main(int argc,char** argv){try{
 if(argc<2||argc>3)throw std::runtime_error("usage: solver-runner cases.bin [--gpu]");const bool gpu=argc==3&&std::strcmp(argv[2],"--gpu")==0;
 std::ifstream input(argv[1],std::ios::binary);std::uint32_t header[3];read(input,header,12);if(header[0]!=SOLVER_INPUT_MAGIC||header[1]!=1||!header[2]||header[2]>100000)throw std::runtime_error("solver packet identity");
 std::vector<Plan> plans;std::vector<n::BodyState>bodies;std::vector<n::ContactConstraint>constraints;std::vector<n::Manifold>manifolds;std::vector<n::SolverContact>contacts;
 for(unsigned i=0;i<header[2];++i){Plan p;read(input,&p.meta,sizeof(p.meta));const auto& m=p.meta;if(m.stage>4||m.bodies>n::max_bodies||m.constraints>n::max_contacts||m.manifolds>n::max_contacts||m.contacts>n::max_contacts||m.warm>1)throw std::runtime_error("solver packet capacity/stage");p.body=bodies.size();p.constraint=constraints.size();p.manifold=manifolds.size();p.contact=contacts.size();
  for(unsigned j=0;j<m.bodies;++j){std::uint32_t w[12];read(input,w,48);bodies.push_back(body(w));}
  for(unsigned j=0;j<m.constraints;++j){std::uint32_t w[30];read(input,w,120);constraints.push_back(constraint(w));}
  static_assert(sizeof(n::Manifold)==60);for(unsigned j=0;j<m.manifolds;++j){n::Manifold value;read(input,&value,60);manifolds.push_back(value);}
  for(unsigned j=0;j<m.contacts;++j){std::uint32_t w[5];read(input,w,20);contacts.push_back({w[0],w[1],w[2],float_word(w[3]),float_word(w[4])});}plans.push_back(p);
 }
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing solver packet");std::vector<View>views;std::vector<SolverWireResult>results(plans.size());float kernel_ms=0;const auto begin=std::chrono::steady_clock::now();
 if(gpu){
#ifdef __CUDACC__
  auto* db=upload(bodies);auto* dc=upload(constraints);auto* dm=upload(manifolds);auto* di=upload(contacts);
  for(const auto& p:plans)views.push_back({p.meta,db+p.body,dc+p.constraint,dm+p.manifold,di+p.contact});auto* dv=upload(views);auto* dr=upload(results);
  cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));check(cudaEventRecord(start));run_solver<<<(views.size()+63)/64,64>>>(dv,dr,views.size());check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&kernel_ms,start,stop));
  download(results,dr);download(bodies,db);download(constraints,dc);download(manifolds,dm);check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaFree(db));check(cudaFree(dc));check(cudaFree(dm));check(cudaFree(di));check(cudaFree(dv));check(cudaFree(dr));
#else
  throw std::runtime_error("CUDA unavailable in host runner");
#endif
 }else{for(const auto& p:plans)views.push_back({p.meta,bodies.data()+p.body,constraints.data()+p.constraint,manifolds.data()+p.manifold,contacts.data()+p.contact});for(unsigned i=0;i<views.size();++i)results[i]=solve(views[i]);}
 const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();unsigned failures=0;header[0]=SOLVER_OUTPUT_MAGIC;words(header,3);
 for(unsigned i=0;i<plans.size();++i){const auto& p=plans[i];failures+=results[i].status!=0;words(&results[i],3);for(unsigned j=0;j<p.meta.bodies;++j)body_words(bodies[p.body+j]);for(unsigned j=0;j<p.meta.constraints;++j)constraint_words(constraints[p.constraint+j]);for(unsigned j=0;j<p.meta.manifolds;++j)words(&manifolds[p.manifold+j],15);}
 std::fprintf(stderr,"{\"backend\":\"%s\",\"stage_cases\":%zu,\"failed_cases\":%u,\"kernel_ms\":%.9f,\"elapsed_ms\":%.9f,\"guest_linked\":false}\n",gpu?"cuda-native":"host-native",plans.size(),failures,kernel_ms,elapsed);return failures||std::fflush(stdout)?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"solver-runner: %s\n",e.what());return 2;}}
