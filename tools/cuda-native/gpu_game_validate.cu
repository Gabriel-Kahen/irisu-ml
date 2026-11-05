// Continuous native-digest qualification with each GPU experiment configured.
#include "../../clone/cuda_native/game_world.cuh"
#include "../../clone/cuda_native/solver_persistent.hpp"
#include "../../clone/cuda_native/parallel_physics.cuh"
#include "../../clone/cuda_native/collision_memo.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace g=irisu::cuda_game;namespace p=irisu::cuda_native;
using World=g::NativeGameWorld;using Scene=p::ScenePipelineState;using Frame=irisu::game_digest::Frame;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {
 std::vector<void*> pointers;~Memory(){for(auto* p:pointers)cudaFree(p);}
 template<class T>T* allocate(std::size_t count=1){T* p{};check(cudaMalloc(&p,count*sizeof(T)));pointers.push_back(p);return p;}
};
__global__ void record_kernel(World* world,Scene* scene,unsigned seed,const unsigned* inputs,unsigned steps,Frame* frames){
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 if(!threadIdx.x)p::parallel_physics::begin_slot(0,&scene->world);
 __syncthreads();
 if(threadIdx.x>=32){p::parallel_physics::run_worker(0);return;}
#endif
 if(threadIdx.x)return;
 world->initialize(scene,seed);frames[0]=world->frame();
 for(unsigned i=0;i<steps&&!world->failed();++i)frames[i+1]=world->frame(world->step(g::native_replay_action(inputs[i],i)));
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 p::parallel_physics::stop_slot(0);
#endif
}
unsigned word(const std::vector<unsigned char>& bytes,std::size_t at){unsigned value;std::memcpy(&value,bytes.data()+at,4);return value;}
int main(int argc,char** argv){try{
 if(argc<2)throw std::runtime_error("usage: gpu-game-validate replay.rpy [steps] --gpu [--cache-mib N]");
 std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("cannot open replay");
 const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
 if(bytes.size()<=52||(bytes.size()-52)%4||word(bytes,16))throw std::runtime_error("invalid padded normal replay");
 const unsigned available=(bytes.size()-52)/4;unsigned steps=available,cache_mib=64;bool gpu=false;
 for(int i=2;i<argc;++i){const std::string arg=argv[i];if(arg=="--gpu")gpu=true;else if(arg=="--cache-mib"&&i+1<argc)cache_mib=std::stoul(argv[++i]);else steps=std::stoul(arg);}
 if(!gpu||steps>available||cache_mib>4096)throw std::runtime_error("GPU mode and bounded steps/cache required");
 const auto begin=std::chrono::steady_clock::now();Memory memory;check(cudaDeviceSetLimit(cudaLimitStackSize,16*1024));
 auto host=std::make_unique<World>();auto scene=std::make_unique<Scene>();std::memset(host.get(),0,sizeof(World));std::memset(scene.get(),0,sizeof(Scene));
 ::new(static_cast<void*>(host.get()))World();::new(static_cast<void*>(scene.get()))Scene();
 auto* dw=memory.allocate<World>();auto* ds=memory.allocate<Scene>();auto* di=memory.allocate<unsigned>(std::max(1U,steps));auto* df=memory.allocate<Frame>(steps+1);
 check(cudaMemcpy(dw,host.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));
 if(steps)check(cudaMemcpy(di,bytes.data()+52,steps*4,cudaMemcpyHostToDevice));check(cudaMemset(df,0,(steps+1)*sizeof(Frame)));
 unsigned long long* mirror_counters{};p::parallel_physics::Channel* channels{};p::collision_memo::Counters* cache_counters{};
 unsigned cache_capacity=0;
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
 auto* mirror=memory.allocate<unsigned>(p::persistent_solver::words_required(1));mirror_counters=memory.allocate<unsigned long long>(p::persistent_solver::counter_count);
 check(cudaMemset(mirror_counters,0,p::persistent_solver::counter_count*sizeof(unsigned long long)));
 check(p::persistent_solver::configure({mirror,reinterpret_cast<std::uintptr_t>(ds),sizeof(Scene),1,p::max_bodies,p::max_contacts,mirror_counters}));
#endif
#if IRISU_CUDA_NATIVE_PARALLEL_PHASES
 channels=memory.allocate<p::parallel_physics::Channel>();check(cudaMemset(channels,0,sizeof(*channels)));
 check(p::parallel_physics::configure({ds,sizeof(Scene),1,channels,true,true,true,2}));
#endif
#if IRISU_CUDA_NATIVE_COLLISION_CACHE
 if(cache_mib){while(std::size_t(cache_capacity?2*cache_capacity:4)*sizeof(p::collision_memo::Entry)<=std::size_t(cache_mib)*1024*1024)cache_capacity=cache_capacity?2*cache_capacity:4;
  auto* entries=memory.allocate<p::collision_memo::Entry>(cache_capacity);cache_counters=memory.allocate<p::collision_memo::Counters>(32);
  check(cudaMemset(entries,0,std::size_t(cache_capacity)*sizeof(*entries)));check(cudaMemset(cache_counters,0,32*sizeof(*cache_counters)));
  check(p::collision_memo::configure({entries,cache_capacity,cache_counters,32}));
 }
#endif
 cudaEvent_t a{},b{};check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a));
 record_kernel<<<1,IRISU_CUDA_NATIVE_PARALLEL_PHASES?64:1>>>(dw,ds,word(bytes,0),di,steps,df);check(cudaGetLastError());check(cudaEventRecord(b));check(cudaEventSynchronize(b));
 float kernel_ms{};check(cudaEventElapsedTime(&kernel_ms,a,b));std::vector<Frame> frames(steps+1);check(cudaMemcpy(frames.data(),df,frames.size()*sizeof(Frame),cudaMemcpyDeviceToHost));
 check(cudaMemcpy(host.get(),dw,sizeof(World),cudaMemcpyDeviceToHost));check(cudaMemcpy(scene.get(),ds,sizeof(Scene),cudaMemcpyDeviceToHost));
 const auto body_offset=reinterpret_cast<const unsigned char*>(host->simulator.bodies_.data())-reinterpret_cast<const unsigned char*>(host.get());
 if(!host->rebind(scene.get(),reinterpret_cast<std::uintptr_t>(dw)+body_offset))throw std::runtime_error("final world rebind failed");
 std::vector<unsigned long long> counts(p::persistent_solver::counter_count);p::parallel_physics::Stats parallel{};p::collision_memo::Counters cache{};
 if(mirror_counters)check(cudaMemcpy(counts.data(),mirror_counters,counts.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
 if(channels)check(cudaMemcpy(&parallel,channels,sizeof(parallel),cudaMemcpyDeviceToHost));
 if(cache_counters){std::vector<p::collision_memo::Counters> values(32);check(cudaMemcpy(values.data(),cache_counters,values.size()*sizeof(values[0]),cudaMemcpyDeviceToHost));for(const auto& c:values){
  cache.calls+=c.calls;cache.hits+=c.hits;cache.misses+=c.misses;cache.inserts+=c.inserts;cache.busy+=c.busy;cache.full+=c.full;cache.bypass+=c.bypass;}}
 const auto& s=host->simulator;
 std::fprintf(stderr,"{\"backend\":\"gpu-first-full-game\",\"steps\":%u,\"tick\":%llu,\"failed\":%s,\"physics_status\":%u,\"live_score\":%lld,\"recorded_final_score\":%lld,\"level\":%u,\"chain\":%u,\"kernel_ms_including_frame_hashes\":%.6f,\"elapsed_seconds\":%.9f,\"host_physics_calls\":0,\"guest_calls\":0,\"fallback_calls\":0,\"mirror_islands\":%llu,\"mirror_unavailable\":%llu,\"mirror_rejections\":%llu,\"parallel_collision_jobs\":%llu,\"parallel_body_jobs\":%llu,\"parallel_shape_jobs\":%llu,\"cache_capacity\":%u,\"cache_calls\":%llu,\"cache_hits\":%llu,\"cache_misses\":%llu,\"cache_full\":%llu}\n",
  steps,static_cast<unsigned long long>(s.tick_),host->failed()?"true":"false",unsigned(scene->world.status),static_cast<long long>(s.score_),static_cast<long long>(s.recorded_final_score_),s.level_,s.highest_chain_,kernel_ms,
  std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(),counts[p::persistent_solver::islands],counts[p::persistent_solver::unavailable],counts[p::persistent_solver::pack_rejections],parallel.collision_jobs,parallel.body_jobs,parallel.shape_jobs,cache_capacity,cache.calls,cache.hits,cache.misses,cache.full);
 check(cudaEventDestroy(a));check(cudaEventDestroy(b));if(host->failed())return 2;
 if(std::fwrite(frames.data(),sizeof(Frame),frames.size(),stdout)!=frames.size()||std::fflush(stdout))throw std::runtime_error("cannot emit frames");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"gpu-game-validate: %s\n",e.what());return 1;}}
