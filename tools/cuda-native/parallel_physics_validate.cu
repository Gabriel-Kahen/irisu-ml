#include "scene_dynamic_wire.hpp"
#include "../../clone/cuda_native/parallel_physics.cuh"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace p=irisu::cuda_native;namespace parallel=p::parallel_physics;
struct Frame {std::uint64_t hash{};unsigned status{};};
struct Corpus {std::vector<p::SceneCase> cases;std::vector<std::uint64_t> native;unsigned ticks{},iterations{};float dt{};};
Corpus load(const char* path){
 std::ifstream input(path,std::ios::binary);auto read=[&](){unsigned value;if(!input.read(reinterpret_cast<char*>(&value),4))throw std::runtime_error("short dynamic packet");return value;};
 if(read()!=p::dynamic_magic||read()!=p::dynamic_version)throw std::runtime_error("dynamic packet identity");
 Corpus c;const auto count=read();c.ticks=read();if(read()!=p::dynamic_case_words)throw std::runtime_error("dynamic case stride");const auto dt=read();std::memcpy(&c.dt,&dt,4);c.iterations=read();if(read()||!count||count>1024||!c.ticks||std::uint64_t(count)*c.ticks>1000000)throw std::runtime_error("dynamic packet bounds");
 c.cases.resize(count);c.native.resize(std::size_t(count)*c.ticks);
 for(unsigned i=0;i<count;++i){if(!input.read(reinterpret_cast<char*>(&c.cases[i]),sizeof(p::SceneCase)))throw std::runtime_error("short dynamic case");for(unsigned j=0;j<c.ticks;++j){const auto low=read(),high=read();c.native[std::size_t(i)*c.ticks+j]=std::uint64_t(low)|(std::uint64_t(high)<<32);}}
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing dynamic packet");return c;
}
#ifdef __CUDACC__
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
struct Memory {std::vector<void*> pointers;~Memory(){for(auto* p:pointers)cudaFree(p);}template<class T>T* allocate(unsigned count){T* p{};check(cudaMalloc(&p,std::size_t(count)*sizeof(T)));pointers.push_back(p);return p;}};
__global__ void initialize(p::ScenePipelineState* scenes,p::SceneActors* actors,const p::SceneCase* cases,unsigned count,unsigned* errors){const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count&&!p::initialize_scene_case(scenes[lane],actors[lane],cases[lane]))atomicAdd(errors,1U);}
__global__ void reference(p::ScenePipelineState* scenes,p::SceneActors* actors,const p::SceneCase* cases,unsigned count,unsigned ticks,float dt,unsigned iterations,Frame* output){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;p::StageResult status{};
 for(unsigned tick=0;tick<ticks;++tick){if(status)status=p::apply_scene_ops(scenes[lane],actors[lane],cases[lane],tick);if(status)status=scenes[lane].step(dt,iterations);output[std::size_t(lane)*ticks+tick]={p::dynamic_scene_hash(scenes[lane]),unsigned(status.status)};}
}
__global__ void cooperative(p::ScenePipelineState* scenes,p::SceneActors* actors,const p::SceneCase* cases,unsigned ticks,float dt,unsigned iterations,Frame* output){
 const unsigned lane=blockIdx.x;if(!threadIdx.x)parallel::begin_slot(lane,&scenes[lane].world);__syncthreads();
 if(!threadIdx.x){p::StageResult status{};for(unsigned tick=0;tick<ticks;++tick){if(status)status=p::apply_scene_ops(scenes[lane],actors[lane],cases[lane],tick);if(status)status=scenes[lane].step(dt,iterations);output[std::size_t(lane)*ticks+tick]={p::dynamic_scene_hash(scenes[lane]),unsigned(status.status)};}parallel::stop_slot(lane);}
 else if(threadIdx.x>=32)parallel::run_worker(lane);
}
__global__ void compare_state(const p::ScenePipelineState* a,const p::SceneActors* aa,const p::ScenePipelineState* b,const p::SceneActors* ba,unsigned count,unsigned* errors){
 const unsigned lane=blockIdx.x;if(lane>=count)return;const auto* x=reinterpret_cast<const unsigned char*>(a+lane);const auto* y=reinterpret_cast<const unsigned char*>(b+lane);
 for(unsigned i=threadIdx.x;i<offsetof(p::ScenePipelineState,scratch);i+=blockDim.x)if(x[i]!=y[i])atomicAdd(errors,1U);
 x=reinterpret_cast<const unsigned char*>(aa+lane);y=reinterpret_cast<const unsigned char*>(ba+lane);for(unsigned i=threadIdx.x;i<sizeof(p::SceneActors);i+=blockDim.x)if(x[i]!=y[i])atomicAdd(errors,1U);
}
struct FailureCase {p::ScenePipelineState scene{};unsigned kind{},order[4]{0,1,2,3};};
__device__ p::StageResult execute_failure(FailureCase& c,bool selected){
 auto& world=c.scene.world;p::StageResult result{};bool handled=false;
 if(c.kind==0){if(selected)parallel::try_collide(world,result,handled);if(!handled)result=p::collide_world(world);}
 else if(c.kind==1){if(selected)parallel::try_integrate_velocity(world,c.order,4,1.0f,result,handled);if(!handled)for(unsigned i=0;i<4;++i){result=p::integrate_velocity(world.bodies[c.order[i]],1.0f,world.gravity);if(!result)break;}}
 else if(c.kind==2){if(selected)parallel::try_integrate_position(world,c.order,4,1.0f,true,result,handled);if(!handled)for(unsigned i=0;i<4;++i){result=p::integrate_position(world.bodies[c.order[i]],1.0f,true);if(!result)break;}}
 else {if(selected)parallel::prepare_shape_bounds(world,c.order,4,true);for(unsigned i=0;i<4;++i){const auto b=c.order[i];if(world.bodies[b].inv_mass==0)continue;if(!p::math::sincos_input_supported(world.bodies[b].rotation)){result=p::rotation_matrix(world.bodies[b]);if(!result)break;}result=p::sync_body_shapes(world,c.scene.scratch,b,nullptr,false,true);if(!result)break;world.bodies[b].force={};world.bodies[b].torque=0;}}
 return result;
}
__global__ void failure_reference(FailureCase* cases,p::StageResult* results,unsigned count){const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count)results[lane]=execute_failure(cases[lane],false);}
__global__ void failure_cooperative(FailureCase* cases,p::StageResult* results){const unsigned lane=blockIdx.x;if(!threadIdx.x)parallel::begin_slot(lane,&cases[lane].scene.world);__syncthreads();if(!threadIdx.x){results[lane]=execute_failure(cases[lane],true);parallel::stop_slot(lane);}else if(threadIdx.x>=32)parallel::run_worker(lane);}
std::vector<FailureCase> failure_cases(){
 std::vector<FailureCase> cases;
 for(unsigned kind=0;kind<4;++kind)for(unsigned variant=0;variant<6;++variant){FailureCase c;std::memset(&c,0,sizeof(c));::new(&c)FailureCase();c.kind=kind;c.scene.world.bounds={{-100,-100},{100,100}};c.scene.world.gravity={0,2};c.scene.world.body_count=4;c.scene.world.shape_count=4;c.scene.world.contact_count=3;c.scene.world.contact_head=0;
  for(unsigned i=0;i<4;++i){auto& b=c.scene.world.bodies[i];b.id=i+1;b.occupied=true;b.inv_mass=1;b.inv_i=1;b.velocity={.25f,.5f};b.position={float(i),0};b.shape_head=i;auto& s=c.scene.world.shapes[i];s.kind=p::ShapeKind::circle;s.occupied=true;s.body=i;s.radius=1;s.position={float(i),0};s.proxy_id=0xffff;}
  for(unsigned i=0;i<3;++i){auto& contact=c.scene.world.contacts[i];contact.occupied=true;contact.id=i+1;contact.shape1=i;contact.shape2=i+1;contact.next=i==2?p::null_index:i+1;}
  if(kind==0){if(variant==1)c.scene.world.shapes[2].position.x=__builtin_inff();if(variant==2)c.scene.world.contacts[1].occupied=false;if(variant==3)c.scene.world.contacts[1].flags=2;if(variant==4)c.scene.world.contacts[1].shape1=p::max_shapes;if(variant==5)for(auto& b:c.scene.world.bodies)b.flags|=p::sleeping_body;}
  else if(kind==1){if(variant==1){c.scene.world.bodies[2].force={0x1.fffffep127f,0};c.scene.world.bodies[2].inv_mass=0x1.fffffep127f;}if(variant==2)c.scene.world.bodies[2].angular_velocity=__builtin_inff();if(variant==3)c.order[2]=1;if(variant==4)c.scene.world.bodies[1].flags|=p::static_body;if(variant==5)c.scene.world.bodies[2].linear_damping=0x1.fffffep127f;}
  else if(kind==2){if(variant==1)c.scene.world.bodies[2].rotation=0x1.fffffep127f;if(variant==2)c.scene.world.bodies[2].velocity.x=__builtin_inff();if(variant==3)c.order[2]=1;if(variant==4)c.scene.world.bodies[2].inv_mass=0;if(variant==5)c.scene.world.bodies[2].angular_velocity=0x1.fffffep127f;}
  else {if(variant==1)c.scene.world.bodies[2].rotation0=__builtin_inff();if(variant==2){c.scene.world.bodies[2].position={0x1.fffffep127f,0};c.scene.world.shapes[2].radius=0x1.fffffep127f;}if(variant==3)c.scene.world.shapes[2].local_position.x=__builtin_inff();if(variant==4)c.scene.world.shapes[2].kind=p::ShapeKind(99);if(variant==5)c.scene.world.bodies[2].inv_mass=0;}
  cases.push_back(c);
 }
 return cases;
}
#endif
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: parallel-physics-validate native-dynamic-packets.bin");
#if !defined(__CUDACC__) || !IRISU_CUDA_NATIVE_PARALLEL_PHASES
 throw std::runtime_error("parallel physics qualifier requires CUDA and PARALLEL_PHASES=ON");
#else
 const auto corpus=load(argv[1]);const unsigned count=unsigned(corpus.cases.size()),frames=count*corpus.ticks;Memory memory;
 auto* cases=memory.allocate<p::SceneCase>(count);auto* original=memory.allocate<p::ScenePipelineState>(count);auto* selected=memory.allocate<p::ScenePipelineState>(count);auto* oa=memory.allocate<p::SceneActors>(count);auto* sa=memory.allocate<p::SceneActors>(count);auto* reference_frames=memory.allocate<Frame>(frames);auto* selected_frames=memory.allocate<Frame>(frames);auto* errors=memory.allocate<unsigned>(1);auto* channels=memory.allocate<parallel::Channel>(count);
 check(cudaMemset(original,0,count*sizeof(*original)));check(cudaMemset(selected,0,count*sizeof(*selected)));check(cudaMemset(oa,0,count*sizeof(*oa)));check(cudaMemset(sa,0,count*sizeof(*sa)));check(cudaMemset(errors,0,4));check(cudaMemset(channels,0,count*sizeof(*channels)));check(cudaMemcpy(cases,corpus.cases.data(),count*sizeof(*cases),cudaMemcpyHostToDevice));
 // Frame stores populate semantic fields; initialize their trailing ABI padding for readback.
 check(cudaMemset(reference_frames,0,frames*sizeof(Frame)));check(cudaMemset(selected_frames,0,frames*sizeof(Frame)));
 initialize<<<(count+31)/32,32>>>(original,oa,cases,count,errors);initialize<<<(count+31)/32,32>>>(selected,sa,cases,count,errors);
 reference<<<(count+31)/32,32>>>(original,oa,cases,count,corpus.ticks,corpus.dt,corpus.iterations,reference_frames);check(cudaDeviceSynchronize());
 check(parallel::configure({selected,sizeof(*selected),count,channels,true,true,true,2}));cooperative<<<count,parallel::block_threads>>>(selected,sa,cases,corpus.ticks,corpus.dt,corpus.iterations,selected_frames);check(cudaGetLastError());check(cudaDeviceSynchronize());
 compare_state<<<count,128>>>(original,oa,selected,sa,count,errors);check(cudaDeviceSynchronize());std::vector<Frame> a(frames),b(frames);check(cudaMemcpy(a.data(),reference_frames,frames*sizeof(Frame),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),selected_frames,frames*sizeof(Frame),cudaMemcpyDeviceToHost));unsigned failure{};check(cudaMemcpy(&failure,errors,4,cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<frames;++i)if(a[i].status||b[i].status||a[i].hash!=b[i].hash||a[i].hash!=corpus.native[i]){++failure;std::fprintf(stderr,"frame%u status%u/%u hashes%llx/%llx/native%llx\n",i,a[i].status,b[i].status,(unsigned long long)a[i].hash,(unsigned long long)b[i].hash,(unsigned long long)corpus.native[i]);}
 std::vector<parallel::Stats> stats(count);check(cudaMemcpy2D(stats.data(),sizeof(stats[0]),channels,sizeof(*channels),sizeof(stats[0]),count,cudaMemcpyDeviceToHost));parallel::Stats total{};for(const auto& s:stats){total.collision_calls+=s.collision_calls;total.velocity_calls+=s.velocity_calls;total.position_calls+=s.position_calls;total.bounds_calls+=s.bounds_calls;total.collision_jobs+=s.collision_jobs;total.body_jobs+=s.body_jobs;total.shape_jobs+=s.shape_jobs;}check(parallel::configure({}));
 if(!total.collision_calls||!total.velocity_calls||!total.position_calls||!total.bounds_calls)throw std::runtime_error("missing active parallel phase witnesses");
 auto failures=failure_cases();const unsigned nf=unsigned(failures.size());auto* fr=memory.allocate<FailureCase>(nf);auto* fs=memory.allocate<FailureCase>(nf);auto* sr=memory.allocate<p::StageResult>(nf);auto* ss=memory.allocate<p::StageResult>(nf);auto* fc=memory.allocate<parallel::Channel>(nf);check(cudaMemcpy(fr,failures.data(),nf*sizeof(*fr),cudaMemcpyHostToDevice));check(cudaMemcpy(fs,failures.data(),nf*sizeof(*fs),cudaMemcpyHostToDevice));check(cudaMemset(fc,0,nf*sizeof(*fc)));
 failure_reference<<<1,32>>>(fr,sr,nf);check(cudaDeviceSynchronize());check(parallel::configure({fs,sizeof(*fs),nf,fc,true,true,true,2}));failure_cooperative<<<nf,parallel::block_threads>>>(fs,ss);check(cudaGetLastError());check(cudaDeviceSynchronize());check(parallel::configure({}));
 std::vector<FailureCase> first(nf),second(nf);std::vector<p::StageResult> r(nf),s(nf);check(cudaMemcpy(first.data(),fr,nf*sizeof(*fr),cudaMemcpyDeviceToHost));check(cudaMemcpy(second.data(),fs,nf*sizeof(*fs),cudaMemcpyDeviceToHost));check(cudaMemcpy(r.data(),sr,nf*sizeof(*sr),cudaMemcpyDeviceToHost));check(cudaMemcpy(s.data(),ss,nf*sizeof(*ss),cudaMemcpyDeviceToHost));unsigned rejections=0;
 for(unsigned i=0;i<nf;++i){rejections+=!r[i];if(r[i].status!=s[i].status||r[i].body!=s[i].body||std::memcmp(&first[i].scene,&second[i].scene,offsetof(p::ScenePipelineState,scratch))||std::memcmp(first[i].scene.scratch.swept,second[i].scene.scratch.swept,sizeof(first[i].scene.scratch.swept))){++failure;std::fprintf(stderr,"failurecase%u kind%u status%u/%u body%u/%u\n",i,first[i].kind,unsigned(r[i].status),unsigned(s[i].status),r[i].body,s[i].body);}}
 std::printf("{\"passed\":%s,\"native_frames\":%u,\"full_persistent_bytes_equal\":%s,\"failure_cases\":%u,\"rejections\":%u,\"mismatches\":%u,\"collision_calls\":%llu,\"velocity_calls\":%llu,\"position_calls\":%llu,\"bounds_calls\":%llu,\"collision_jobs\":%llu,\"body_jobs\":%llu,\"shape_jobs\":%llu,\"channel_bytes\":%zu}\n",failure?"false":"true",frames,failure?"false":"true",nf,rejections,failure,total.collision_calls,total.velocity_calls,total.position_calls,total.bounds_calls,total.collision_jobs,total.body_jobs,total.shape_jobs,sizeof(parallel::Channel));return failure?1:0;
#endif
}catch(const std::exception& e){std::fprintf(stderr,"parallel-physics-validate: %s\n",e.what());return 2;}}
