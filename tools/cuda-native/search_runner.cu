// Independent typed complete-game futures. Shared wire/helpers are read-only;
// this executable does not link the guest translator or native oracle library.
#include "../../clone/cuda_native/game_world.cuh"
#include "../cuda-exact/search_io.hpp"
#include "search_accel.cuh"
#include "prefix_guard.cuh"
#include "../../clone/cuda_native/compact_game_snapshot.cuh"
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace game=irisu::cuda_game;namespace physics=irisu::cuda_native;namespace wire=irisu::cuda_search;
using World=game::NativeGameWorld;using Scene=physics::ScenePipelineState;
struct Options {
 std::uint32_t seed{},prefix{},horizon{},keep{2},threads{32},stack_kib{16};
 std::int64_t target{4393102};bool host_prefix{},trace{true},validate{true},cpu{},isolated{},copy_u64{},split_rebind{},reuse_prefix{},physical_snapshot{},stream_copy{},merge_rank{},tiled_selection{},compact_clone{},parallel_guard{};
 unsigned warmups{1},repeats{3},scratch_poison{};
};
struct Sample {double host_prefix_ms{},prefix_ms{},branch_ms{},rollout_ms{},rank_compact_ms{},validation_ms{},total_ms{};};
struct Validation {unsigned errors{},first_section{},first_offset{~0U},prefix_errors{};};
constexpr std::size_t physical_scene_bytes=offsetof(Scene,scratch);
static_assert(physical_scene_bytes%8==0);
constexpr std::size_t prefix_guard_bytes=sizeof(World)+sizeof(Scene)+sizeof(wire::Result);
GAME_HD inline std::uintptr_t body_base(const World& w){return reinterpret_cast<std::uintptr_t>(w.simulator.bodies_.data());}
GAME_HD void prepare_prefix(World* world,Scene* scene,const Options& options,const std::uint32_t* replay,wire::Result* result){
 world->initialize(scene,options.seed);game::StepResult step{};
 for(unsigned i=0;i<options.prefix&&!world->failed();++i){if(step.terminated||step.truncated)break;step=world->step(game::native_replay_action(replay[i],i));}
 *result=wire::Result{};wire::fold(*result,world->frame(step));result->failed=world->failed()||step.terminated||step.truncated;
 result->guest_error=unsigned(scene->world.status);result->fault_pc=0;
}
GAME_HD void future(World& world,Scene& scene,const Options& options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate& candidate,const wire::Result& prefix,wire::Result& result){
 result=prefix;game::StepResult step{};
 if(!result.failed)for(unsigned offset=0;offset<options.horizon&&!world.failed();++offset){
  const unsigned tick=options.prefix+offset,incumbent=tick<replay_size?replay[tick]:0;
  step=world.step(game::native_replay_action(wire::input_at(candidate,offset,incumbent),tick));
  if(options.trace)wire::fold(result,world.frame(step));++result.simulated_ticks;if(step.terminated||step.truncated)break;
 }
 if(!options.trace&&result.simulated_ticks)wire::fold(result,world.frame(step));
 result.clears=world.simulator.qualifying_clear_count_;result.natural=step.terminated&&!step.truncated;
 result.canonical_level=step.diagnostics.recorded_final_level;result.canonical_score=step.diagnostics.recorded_final_score;
 result.failed|=result.natural&&!step.diagnostics.terminal_metadata_recorded;result.failed|=world.failed();
 result.guest_error=unsigned(scene.world.status);result.fault_pc=0;
 wire::rank_keys(result,std::uint64_t(options.prefix)+options.horizon,options.target);
}
// Host-only default construction creates EMPTY storage, not a simulated
// prefix. Keeping giant aggregate constructors out of device code avoids
// optimizer IR expansion; all initialize/reset/physics/replay remain device.
template<class T>std::unique_ptr<T> cold_object(){auto* p=static_cast<T*>(::operator new(sizeof(T)));std::memset(p,0,sizeof(T));::new(static_cast<void*>(p))T();return std::unique_ptr<T>(p);}
void cold_storage(World* world,Scene* scene){std::memset(world,0,sizeof(World));std::memset(scene,0,sizeof(Scene));::new(static_cast<void*>(world))World();::new(static_cast<void*>(scene))Scene();}
struct Environment {
 std::fenv_t previous;
 Environment(){if(std::fegetenv(&previous)||std::fesetenv(FE_DFL_ENV))throw std::runtime_error("cannot set host rounding environment");
#if defined(__i386__) || defined(__x86_64__)
  const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));
#endif
 }
 ~Environment(){std::fesetenv(&previous);}
};
void host_clone(World& destination,Scene& scene,const World& source,const Scene& source_scene,std::size_t scene_bytes=sizeof(Scene)){
 std::memcpy(&destination,&source,sizeof(World));std::memcpy(&scene,&source_scene,scene_bytes);
 if(!destination.rebind(&scene,body_base(source)))throw std::runtime_error("host snapshot pointer rebind failed");
}
// Normalize only address representations. Every byte of both objects, including
// all scratch and inactive container storage, is retained and compared.
bool host_equal(World& a,Scene& sa,World& b,Scene& sb,std::size_t scene_bytes=sizeof(Scene)){
 constexpr std::uintptr_t canonical=0x100000;bool valid=a.relocate_body_pointers(body_base(a),canonical)&&b.relocate_body_pointers(body_base(b),canonical);
 a.bind(nullptr);b.bind(nullptr);const bool equal=!std::memcmp(&a,&b,sizeof(World))&&!std::memcmp(&sa,&sb,scene_bytes);
 valid=a.rebind(&sa,canonical)&&b.rebind(&sb,canonical)&&valid;return valid&&equal;
}
void host_poison_scratch(Scene& scene,unsigned kind,unsigned lane=0){
 auto* bytes=reinterpret_cast<unsigned char*>(&scene)+physical_scene_bytes;
 for(std::size_t i=0;i<(sizeof(Scene)-physical_scene_bytes)/4;++i){const std::uint32_t word=(kind==2?0x7fc08000U:0xa5a58000U)|((unsigned(i)+17*lane)&0x7fffU);std::memcpy(bytes+4*i,&word,4);}
}
#ifdef __CUDACC__
__global__ void poison_scratch_kernel(Scene* scenes,unsigned count,const unsigned* selection,unsigned kind){
 const unsigned lane=blockIdx.x;if(lane>=count)return;const unsigned index=selection?selection[lane]:lane;
 auto* bytes=reinterpret_cast<unsigned char*>(scenes+index)+physical_scene_bytes;
 static_assert((sizeof(Scene)-physical_scene_bytes)%4==0);
 for(std::size_t i=threadIdx.x;i<(sizeof(Scene)-physical_scene_bytes)/4;i+=blockDim.x){const std::uint32_t word=(kind==2?0x7fc08000U:0xa5a58000U)|((unsigned(i)+17*index)&0x7fffU);std::memcpy(bytes+4*i,&word,4);}
}
__global__ void prefix_kernel(World* world,Scene* scene,const World* empty_world,const Scene* empty_scene,Options options,const std::uint32_t* replay,wire::Result* result){
 auto* w=reinterpret_cast<unsigned char*>(world);auto* s=reinterpret_cast<unsigned char*>(scene);const auto* ew=reinterpret_cast<const unsigned char*>(empty_world);const auto* es=reinterpret_cast<const unsigned char*>(empty_scene);
 for(std::size_t i=threadIdx.x;i<sizeof(World);i+=blockDim.x)w[i]=ew[i];for(std::size_t i=threadIdx.x;i<sizeof(Scene);i+=blockDim.x)s[i]=es[i];__syncthreads();
 if(!threadIdx.x)prepare_prefix(world,scene,options,replay,result);
}
__global__ void rebind_prefix_kernel(World* world,Scene* scene,std::uintptr_t old_base,Validation* validation){if(!threadIdx.x&&!world->rebind(scene,old_base))atomicAdd(&validation->errors,1U);}
namespace compact=physics::compact_snapshot;
__global__ void compact_plan_kernel(const World* world,const Scene* scene,compact::GamePlan* game_plan,compact::Plan* scene_plan,Validation* validation){
 if(!threadIdx.x&&(!compact::build_plan(*scene,*scene_plan)||!compact::build_game_plan(*world,*game_plan)))atomicAdd(&validation->errors,1U);
}
__global__ void compact_clone_kernel(World* worlds,Scene* scenes,const World* source_world,const Scene* source_scene,unsigned count,const compact::GamePlan* game_plan,const compact::Plan* scene_plan,bool initialized,Validation* validation){
 const unsigned lane=blockIdx.x;if(lane>=count)return;__shared__ bool valid;
 if(!threadIdx.x){valid=!initialized||bool(compact::clear_reachable(scenes[lane]));if(!valid)atomicAdd(&validation->errors,1U);}__syncthreads();if(!valid)return;
 compact::copy_game(worlds[lane],*source_world,*game_plan,threadIdx.x,blockDim.x);
 compact::copy_active(scenes[lane],*source_scene,*scene_plan,threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(worlds[lane],scenes[lane],*game_plan))atomicAdd(&validation->errors,1U);
}
// Only CUDA-global object representations reach this helper. Alignment is
// guaranteed by cudaMalloc, whole-object strides and eight-byte offsets; PTX
// avoids C++ typed aliasing while memory clobbers order the following rebind.
template<bool streaming=false> __device__ __forceinline__ void copy_word64(unsigned char* destination,const unsigned char* source){
 std::uint64_t word;
 if constexpr(streaming){
  asm volatile("ld.global.cg.u64 %0, [%1];":"=l"(word):"l"(__cvta_generic_to_global(source)):"memory");
  asm volatile("st.global.cs.u64 [%0], %1;"::"l"(__cvta_generic_to_global(destination)),"l"(word):"memory");
 }else{
  asm volatile("ld.global.u64 %0, [%1];":"=l"(word):"l"(__cvta_generic_to_global(source)):"memory");
  asm volatile("st.global.u64 [%0], %1;"::"l"(__cvta_generic_to_global(destination)),"l"(word):"memory");
 }
}
template<bool wide,bool bind=true,bool streaming=false> __global__ void clone_kernel(World* worlds,Scene* scenes,const World* source_worlds,const Scene* source_scenes,unsigned count,const unsigned* selection,Validation* validation,std::size_t scene_bytes){
 const unsigned lane=blockIdx.x;if(lane>=count)return;const unsigned source=selection?selection[lane]:0;
 auto* w=reinterpret_cast<unsigned char*>(worlds+lane);const auto* sw=reinterpret_cast<const unsigned char*>(source_worlds+source);
 auto* s=reinterpret_cast<unsigned char*>(scenes+lane);const auto* ss=reinterpret_cast<const unsigned char*>(source_scenes+source);
 static_assert(alignof(World)>=8&&alignof(Scene)>=8&&sizeof(World)%8==0&&sizeof(Scene)%8==0);
 if constexpr(wide){
  for(std::size_t i=threadIdx.x;i<sizeof(World)/8;i+=blockDim.x)copy_word64<streaming>(w+8*i,sw+8*i);
  for(std::size_t i=threadIdx.x;i<scene_bytes/8;i+=blockDim.x)copy_word64<streaming>(s+8*i,ss+8*i);
 }else{
  for(std::size_t i=threadIdx.x;i<sizeof(World);i+=blockDim.x)w[i]=sw[i];
  for(std::size_t i=threadIdx.x;i<scene_bytes;i+=blockDim.x)s[i]=ss[i];
 }
 if constexpr(bind){
  __syncthreads();
  if(!threadIdx.x&&!worlds[lane].rebind(scenes+lane,body_base(source_worlds[source])))atomicAdd(&validation->errors,1U);
 }
}
__global__ void rebind_clones_kernel(World* worlds,Scene* scenes,const World* source_worlds,unsigned count,const unsigned* selection,Validation* validation){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
 const unsigned source=selection?selection[lane]:0;
 if(!worlds[lane].rebind(scenes+lane,body_base(source_worlds[source])))atomicAdd(&validation->errors,1U);
}
// Default-stream kernel ordering makes all copy stores visible before the
// same rebind API runs with packed worlds. Selected sources use their rank.
void launch_clones(World* worlds,Scene* scenes,const World* source_worlds,const Scene* source_scenes,unsigned count,const unsigned* selection,Validation* validation,const Options& options){
 if(selection&&options.tiled_selection){
  const auto status=physics::search_accel::copy_tiled_async(worlds,scenes,source_worlds,source_scenes,count,selection,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  rebind_clones_kernel<<<(count+127)/128,128>>>(worlds,scenes,source_worlds,count,selection,validation);return;
 }
 if(options.split_rebind){
  if(options.stream_copy)clone_kernel<true,false,true><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
  else if(options.copy_u64)clone_kernel<true,false><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
  else clone_kernel<false,false><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
  rebind_clones_kernel<<<(count+127)/128,128>>>(worlds,scenes,source_worlds,count,selection,validation);
 }else if(options.stream_copy)clone_kernel<true,true,true><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
 else if(options.copy_u64)clone_kernel<true><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
 else clone_kernel<false><<<count,256>>>(worlds,scenes,source_worlds,source_scenes,count,selection,validation,options.physical_snapshot?physical_scene_bytes:sizeof(Scene));
}
__device__ __forceinline__ void compare_prefix_region(const void* current,const void* saved,std::size_t bytes,unsigned section,Validation* validation){
 const auto* a=static_cast<const unsigned char*>(current);const auto* b=static_cast<const unsigned char*>(saved);unsigned first=~0U;
 for(std::size_t i=threadIdx.x;i<bytes;i+=blockDim.x)if(a[i]!=b[i])first=min(first,unsigned(i));
 if(first!=~0U){atomicAdd(&validation->errors,1U);atomicAdd(&validation->prefix_errors,1U);atomicCAS(&validation->first_section,0U,section);atomicMin(&validation->first_offset,first);}
}
// Read-only: even body-pointer representations and unused scratch must remain
// unchanged in the cached source; only destination clones ever rebind.
__global__ void verify_prefix_kernel(const World* world,const Scene* scene,const wire::Result* result,const World* saved_world,const Scene* saved_scene,const wire::Result* saved_result,Validation* validation){
 compare_prefix_region(world,saved_world,sizeof(World),3,validation);
 compare_prefix_region(scene,saved_scene,sizeof(Scene),4,validation);
 compare_prefix_region(result,saved_result,sizeof(wire::Result),5,validation);
}
__global__ void future_kernel(World* worlds,Scene* scenes,Options options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate* candidates,unsigned count,const wire::Result* prefix,wire::Result* results){
 const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;if(options.isolated&&(thread&31))return;const unsigned lane=options.isolated?thread/32:thread;if(lane<count)future(worlds[lane],scenes[lane],options,replay,replay_size,candidates[lane],*prefix,results[lane]);
}
__global__ void rank_kernel(const wire::Result* results,unsigned count,unsigned* ranks){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;unsigned rank=0;for(unsigned j=0;j<count;++j)rank+=wire::better(results[j],j,results[lane],lane);ranks[rank]=lane;
}
__global__ void verify_kernel(World* selected,Scene* selected_scenes,World* source,Scene* source_scenes,const unsigned* ranks,unsigned count,Validation* validation,std::size_t scene_bytes){
 const unsigned lane=blockIdx.x;if(lane>=count)return;auto& a=selected[lane];auto& b=source[ranks[lane]];constexpr std::uintptr_t canonical=0x100000;
 if(!threadIdx.x){if(!a.relocate_body_pointers(body_base(a),canonical)||!b.relocate_body_pointers(body_base(b),canonical))atomicAdd(&validation->errors,1U);a.bind(nullptr);b.bind(nullptr);}__syncthreads();
 const auto* ga=reinterpret_cast<const unsigned char*>(&a);const auto* gb=reinterpret_cast<const unsigned char*>(&b);
 for(std::size_t i=threadIdx.x;i<sizeof(World);i+=blockDim.x)if(ga[i]!=gb[i]){atomicAdd(&validation->errors,1U);atomicCAS(&validation->first_section,0U,1U);atomicMin(&validation->first_offset,unsigned(i));}
 const auto* sa=reinterpret_cast<const unsigned char*>(selected_scenes+lane);const auto* sb=reinterpret_cast<const unsigned char*>(source_scenes+ranks[lane]);
 for(std::size_t i=threadIdx.x;i<scene_bytes;i+=blockDim.x)if(sa[i]!=sb[i]){atomicAdd(&validation->errors,1U);atomicCAS(&validation->first_section,0U,2U);atomicMin(&validation->first_offset,unsigned(i));}__syncthreads();
 if(!threadIdx.x)if(!a.rebind(selected_scenes+lane,canonical)||!b.rebind(source_scenes+ranks[lane],canonical))atomicAdd(&validation->errors,1U);
}
__global__ void continue_kernel(World* selected,World* source,const unsigned* ranks,unsigned count,Validation* validation){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;auto& a=selected[lane];auto& b=source[ranks[lane]];
 const auto first=a.step({}),second=b.step({});const auto af=a.frame(first),bf=b.frame(second);const auto* ap=reinterpret_cast<const unsigned char*>(&af);const auto* bp=reinterpret_cast<const unsigned char*>(&bf);bool equal=true;
 for(unsigned i=0;i<sizeof(af);++i)equal&=ap[i]==bp[i];if(a.failed()||b.failed()||!equal)atomicAdd(&validation->errors,1U);
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {
 std::vector<void*> pointers;std::uint64_t bytes{};~Memory(){for(auto* p:pointers)cudaFree(p);}
 template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,count*sizeof(T)));pointers.push_back(p);bytes+=count*sizeof(T);return p;}
};
struct Event {cudaEvent_t value{};Event(){check(cudaEventCreate(&value));}~Event(){cudaEventDestroy(value);}void record(){check(cudaEventRecord(value));}};
double elapsed(const Event& a,const Event& b){float ms{};check(cudaEventElapsedTime(&ms,a.value,b.value));return ms;}
#endif
void print_sample(const Sample& s){std::fprintf(stderr,"{\"host_prefix_ms\":%.9f,\"prefix_ms\":%.9f,\"branch_ms\":%.9f,\"rollout_ms\":%.9f,\"rank_compact_ms\":%.9f,\"validation_ms\":%.9f,\"total_ms\":%.9f}",s.host_prefix_ms,s.prefix_ms,s.branch_ms,s.rollout_ms,s.rank_compact_ms,s.validation_ms,s.total_ms);}
int main(int argc,char** argv){try{
 const auto started=std::chrono::steady_clock::now();if(argc<5)throw std::runtime_error("usage: native-search replay specs prefix horizon [--host-prefix] [--final-only] [--repeats N] [--warmups N] [--keep N] [--threads N] [--stack-kib N] [--target N] [--skip-compaction-check] [--isolated-warps] [--copy-u64] [--stream-copy] [--merge-rank] [--tiled-selection] [--compact-clone] [--parallel-prefix-guard] [--split-rebind] [--reuse-prefix] [--physical-snapshot] [--scratch-poison 1|2] [--cpu]");
 auto input=wire::load_inputs(argv[1],argv[2],argv[3],argv[4]);Options options;options.seed=input.options.seed;options.prefix=input.options.prefix;options.horizon=input.options.horizon;const unsigned count=input.candidates.size();options.keep=std::min(2U,count);
 for(int i=5;i<argc;++i){const std::string arg=argv[i];if(arg=="--host-prefix")options.host_prefix=true;else if(arg=="--final-only")options.trace=false;else if(arg=="--skip-compaction-check")options.validate=false;else if(arg=="--isolated-warps")options.isolated=true;else if(arg=="--copy-u64")options.copy_u64=true;else if(arg=="--stream-copy"){options.stream_copy=true;options.copy_u64=true;}else if(arg=="--merge-rank")options.merge_rank=true;else if(arg=="--tiled-selection")options.tiled_selection=true;else if(arg=="--parallel-prefix-guard")options.parallel_guard=true;else if(arg=="--compact-clone"){options.compact_clone=true;options.physical_snapshot=true;}else if(arg=="--split-rebind")options.split_rebind=true;else if(arg=="--reuse-prefix")options.reuse_prefix=true;else if(arg=="--physical-snapshot")options.physical_snapshot=true;else if(arg=="--cpu")options.cpu=true;
  else if(arg=="--scratch-poison"&&i+1<argc)options.scratch_poison=std::stoul(argv[++i]);else if(arg=="--repeats"&&i+1<argc)options.repeats=std::stoul(argv[++i]);else if(arg=="--warmups"&&i+1<argc)options.warmups=std::stoul(argv[++i]);else if(arg=="--keep"&&i+1<argc)options.keep=std::stoul(argv[++i]);else if((arg=="--threads"||arg=="--block-threads")&&i+1<argc)options.threads=std::stoul(argv[++i]);else if(arg=="--stack-kib"&&i+1<argc)options.stack_kib=std::stoul(argv[++i]);else if(arg=="--target"&&i+1<argc)options.target=std::stoll(argv[++i]);else throw std::runtime_error("unknown native-search option");}
 if(!options.repeats||options.repeats>20||options.warmups>10||!options.keep||options.keep>count||!options.threads||options.threads>256||options.stack_kib<4||options.stack_kib>128||options.scratch_poison>2)throw std::runtime_error("invalid bounded search options");
#if IRISU_CUDA_NATIVE_DEVICE_TRIG == 1
 constexpr const char* device_numeric_profile="approximate-device-trig-library";
#elif IRISU_CUDA_NATIVE_DEVICE_TRIG == 2
 constexpr const char* device_numeric_profile="approximate-device-trig-intrinsic";
#elif defined(IRISU_CUDA_NATIVE_BOUNDED_MATH) && IRISU_CUDA_NATIVE_BOUNDED_MATH
 constexpr const char* device_numeric_profile="bounded-device-add-sub-mul";
#else
 constexpr const char* device_numeric_profile="checked";
#endif
 if(options.cpu&&options.compact_clone)throw std::runtime_error("compact clone is a GPU-only experiment");
 const std::size_t scene_copy_bytes=options.physical_snapshot?physical_scene_bytes:sizeof(Scene);
 std::vector<wire::Result> results(count),previous;std::vector<unsigned> ranks(count),previous_ranks;std::vector<Sample> samples;Validation validation;std::uint64_t allocation_bytes=0;unsigned registers=0,prefix_executions=0;std::size_t local_bytes=0,effective_stack_bytes=0;double cold_storage_ms=0;std::size_t clone_game_bytes=sizeof(World),clone_scene_bytes=scene_copy_bytes;
 if(options.cpu){
  Environment environment;auto prefix=std::make_unique<World>();auto prefix_scene=std::make_unique<Scene>();std::unique_ptr<World[]> worlds(new World[count]);std::unique_ptr<Scene[]> scenes(new Scene[count]);auto selected=std::make_unique<World>();auto selected_scene=std::make_unique<Scene>();
  wire::Result prefix_result;std::vector<unsigned char> prefix_guard(options.reuse_prefix?prefix_guard_bytes:0);
  for(unsigned run=0;run<options.warmups+options.repeats;++run){Sample sample;const auto begin=std::chrono::steady_clock::now();const bool execute_prefix=!options.reuse_prefix||!run;
   if(execute_prefix){cold_storage(prefix.get(),prefix_scene.get());prepare_prefix(prefix.get(),prefix_scene.get(),options,input.replay.data(),&prefix_result);++prefix_executions;
    if(options.reuse_prefix){std::memcpy(prefix_guard.data(),prefix.get(),sizeof(World));std::memcpy(prefix_guard.data()+sizeof(World),prefix_scene.get(),sizeof(Scene));std::memcpy(prefix_guard.data()+sizeof(World)+sizeof(Scene),&prefix_result,sizeof(prefix_result));}}
   const auto a=std::chrono::steady_clock::now();sample.prefix_ms=execute_prefix?std::chrono::duration<double,std::milli>(a-begin).count():0;
   for(unsigned i=0;i<count;++i){host_clone(worlds[i],scenes[i],*prefix,*prefix_scene,scene_copy_bytes);if(options.scratch_poison)host_poison_scratch(scenes[i],options.scratch_poison,i);}const auto b=std::chrono::steady_clock::now();sample.branch_ms=std::chrono::duration<double,std::milli>(b-a).count();
   for(unsigned i=0;i<count;++i)future(worlds[i],scenes[i],options,input.replay.data(),input.replay.size(),input.candidates[i],prefix_result,results[i]);const auto c=std::chrono::steady_clock::now();sample.rollout_ms=std::chrono::duration<double,std::milli>(c-b).count();std::iota(ranks.begin(),ranks.end(),0U);std::sort(ranks.begin(),ranks.end(),[&](unsigned a,unsigned b){return wire::better(results[a],a,results[b],b);});const auto d=std::chrono::steady_clock::now();sample.rank_compact_ms=std::chrono::duration<double,std::milli>(d-c).count();
   if(options.validate)for(unsigned k=0;k<options.keep;++k){const unsigned i=ranks[k];host_clone(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes);if(!host_equal(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes))++validation.errors;if(options.physical_snapshot){host_poison_scratch(*selected_scene,1);host_poison_scratch(scenes[i],2);}const auto sa=selected->step({}),sb=worlds[i].step({});const auto fa=selected->frame(sa),fb=worlds[i].frame(sb);if(selected->failed()||worlds[i].failed()||std::memcmp(&fa,&fb,sizeof(fa))||!host_equal(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes))++validation.errors;}
   if(options.reuse_prefix&&(std::memcmp(prefix_guard.data(),prefix.get(),sizeof(World))||std::memcmp(prefix_guard.data()+sizeof(World),prefix_scene.get(),sizeof(Scene))||std::memcmp(prefix_guard.data()+sizeof(World)+sizeof(Scene),&prefix_result,sizeof(prefix_result))))throw std::runtime_error("typed host reused checkpoint changed");
   const auto end=std::chrono::steady_clock::now();sample.validation_ms=std::chrono::duration<double,std::milli>(end-d).count();sample.total_ms=std::chrono::duration<double,std::milli>(end-begin).count();if(run>=options.warmups)samples.push_back(sample);
   if(!previous.empty()&&(std::memcmp(previous.data(),results.data(),count*sizeof(results[0]))||previous_ranks!=ranks))throw std::runtime_error("typed host repeated search differs");previous=results;previous_ranks=ranks;
  }
 }else{
#ifdef __CUDACC__
  check(cudaDeviceSetLimit(cudaLimitStackSize,std::size_t(options.stack_kib)*1024));check(cudaDeviceGetLimit(&effective_stack_bytes,cudaLimitStackSize));Memory memory;
  auto* replay=memory.allocate<std::uint32_t>(input.replay.size());auto* candidates=memory.allocate<wire::Candidate>(count);auto* prefix=memory.allocate<World>(1);auto* prefix_scene=memory.allocate<Scene>(1);auto* prefix_result=memory.allocate<wire::Result>(1);auto* worlds=memory.allocate<World>(count);auto* scenes=memory.allocate<Scene>(count);auto* output=memory.allocate<wire::Result>(count);auto* order=memory.allocate<unsigned>(count);auto* selected=memory.allocate<World>(options.keep);auto* selected_scenes=memory.allocate<Scene>(options.keep);auto* errors=memory.allocate<Validation>(1);
  compact::GamePlan* game_plan{};compact::Plan* scene_plan{};
  if(options.compact_clone){game_plan=memory.allocate<compact::GamePlan>(1);scene_plan=memory.allocate<compact::Plan>(1);check(cudaMemset(game_plan,0,sizeof(*game_plan)));check(cudaMemset(scene_plan,0,sizeof(*scene_plan)));check(cudaMemset(worlds,0,count*sizeof(World)));check(cudaMemset2D(scenes,sizeof(Scene),0,physical_scene_bytes,count));}
  World* empty_world{};Scene* empty_scene{};World* saved_prefix{};Scene* saved_prefix_scene{};wire::Result* saved_prefix_result{};
  if(options.reuse_prefix){saved_prefix=memory.allocate<World>(1);saved_prefix_scene=memory.allocate<Scene>(1);saved_prefix_result=memory.allocate<wire::Result>(1);}
  if(!options.host_prefix){const auto cold_begin=std::chrono::steady_clock::now();auto host=cold_object<World>();auto scene=cold_object<Scene>();empty_world=memory.allocate<World>(1);empty_scene=memory.allocate<Scene>(1);check(cudaMemcpy(empty_world,host.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(empty_scene,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));cold_storage_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_begin).count();}
  std::size_t rank_bytes{};unsigned char* rank_workspace{};if(options.merge_rank){check(physics::search_accel::rank_workspace_bytes(output,order,count,rank_bytes));rank_workspace=memory.allocate<unsigned char>(rank_bytes);}
  allocation_bytes=memory.bytes;
  check(cudaMemcpy(replay,input.replay.data(),input.replay.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(candidates,input.candidates.data(),count*sizeof(wire::Candidate),cudaMemcpyHostToDevice));cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,future_kernel));registers=attributes.numRegs;local_bytes=attributes.localSizeBytes;
  for(unsigned run=0;run<options.warmups+options.repeats;++run){const auto begin=std::chrono::steady_clock::now();Sample sample;validation={};check(cudaMemcpy(errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));std::uintptr_t old_base{};const bool execute_prefix=!options.reuse_prefix||!run;
   if(execute_prefix&&options.host_prefix){Environment environment;auto host=cold_object<World>();auto scene=cold_object<Scene>();wire::Result initial;prepare_prefix(host.get(),scene.get(),options,input.replay.data(),&initial);old_base=body_base(*host);check(cudaMemcpy(prefix,host.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_scene,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_result,&initial,sizeof(initial),cudaMemcpyHostToDevice));sample.host_prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
   Event a,b,c,d,e,f;a.record();
   if(execute_prefix){if(options.host_prefix)rebind_prefix_kernel<<<1,1>>>(prefix,prefix_scene,old_base,errors);else prefix_kernel<<<1,256>>>(prefix,prefix_scene,empty_world,empty_scene,options,replay,prefix_result);++prefix_executions;
    if(options.reuse_prefix){check(cudaMemcpyAsync(saved_prefix,prefix,sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_prefix_scene,prefix_scene,sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_prefix_result,prefix_result,sizeof(wire::Result),cudaMemcpyDeviceToDevice));}}
   if(execute_prefix&&options.compact_clone){compact_plan_kernel<<<1,1>>>(prefix,prefix_scene,game_plan,scene_plan,errors);check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("compact prefix plan rejected");}
   b.record();if(options.compact_clone)compact_clone_kernel<<<count,256>>>(worlds,scenes,prefix,prefix_scene,count,game_plan,scene_plan,run!=0,errors);else launch_clones(worlds,scenes,prefix,prefix_scene,count,nullptr,errors,options);if(options.scratch_poison)poison_scratch_kernel<<<count,256>>>(scenes,count,nullptr,options.scratch_poison);c.record();
   const unsigned threads=options.isolated?std::max(32U,options.threads):options.threads,total=count*(options.isolated?32U:1U);future_kernel<<<(total+threads-1)/threads,threads>>>(worlds,scenes,options,replay,input.replay.size(),candidates,count,prefix_result,output);d.record();if(options.merge_rank)check(physics::search_accel::rank_async(output,order,count,rank_workspace,rank_bytes));else rank_kernel<<<(count+127)/128,128>>>(output,count,order);launch_clones(selected,selected_scenes,worlds,scenes,options.keep,order,errors,options);e.record();
   if(options.validate){verify_kernel<<<options.keep,256>>>(selected,selected_scenes,worlds,scenes,order,options.keep,errors,scene_copy_bytes);if(options.physical_snapshot){poison_scratch_kernel<<<options.keep,256>>>(selected_scenes,options.keep,nullptr,1);poison_scratch_kernel<<<options.keep,256>>>(scenes,options.keep,order,2);}continue_kernel<<<(options.keep+31)/32,32>>>(selected,worlds,order,options.keep,errors);verify_kernel<<<options.keep,256>>>(selected,selected_scenes,worlds,scenes,order,options.keep,errors,scene_copy_bytes);}
   if(options.reuse_prefix){if(options.parallel_guard)check(physics::prefix_guard::check_prefix_async(prefix,prefix_scene,prefix_result,saved_prefix,saved_prefix_scene,saved_prefix_result,errors));else verify_prefix_kernel<<<1,256>>>(prefix,prefix_scene,prefix_result,saved_prefix,saved_prefix_scene,saved_prefix_result,errors);}
   check(cudaGetLastError());f.record();check(cudaEventSynchronize(f.value));check(cudaMemcpy(results.data(),output,count*sizeof(wire::Result),cudaMemcpyDeviceToHost));check(cudaMemcpy(ranks.data(),order,count*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));sample.prefix_ms=execute_prefix?elapsed(a,b):0;sample.branch_ms=elapsed(b,c);sample.rollout_ms=elapsed(c,d);sample.rank_compact_ms=elapsed(d,e);sample.validation_ms=elapsed(e,f);sample.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(run>=options.warmups)samples.push_back(sample);
   if(validation.errors)throw std::runtime_error("CUDA selected full-state check failed: errors="+std::to_string(validation.errors)+" section="+std::to_string(validation.first_section)+" offset="+std::to_string(validation.first_offset));
   if(!previous.empty()&&(std::memcmp(previous.data(),results.data(),count*sizeof(results[0]))||previous_ranks!=ranks))throw std::runtime_error("CUDA repeated search differs");previous=results;previous_ranks=ranks;
  }
  if(options.compact_clone){compact::GamePlan gp;compact::Plan sp;check(cudaMemcpy(&gp,game_plan,sizeof(gp),cudaMemcpyDeviceToHost));check(cudaMemcpy(&sp,scene_plan,sizeof(sp),cudaMemcpyDeviceToHost));clone_game_bytes=gp.bytes;clone_scene_bytes=compact::transport_bytes(sp);}
#else
  throw std::runtime_error("CUDA unavailable: use explicit --cpu for typed host validation");
#endif
 }
 unsigned failed=0;std::uint64_t ticks=0;for(const auto& result:results){failed+=bool(result.failed);ticks+=result.simulated_ticks;}
 std::fprintf(stderr,"{\"identity\":\"experimental-independent-typed-game-search-v1\",\"backend\":\"%s\",\"numeric_profile\":\"%s\",\"host_prefix_numeric_profile\":\"checked\",\"promotion_eligible\":false,\"candidates\":%u,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"failed_candidates\":%u,\"trace_every_tick\":%s,\"host_prefix\":%s,\"threads\":%u,\"isolated_warps\":%s,\"copy_u64\":%s,\"stream_copy\":%s,\"merge_rank\":%s,\"tiled_selection\":%s,\"compact_clone\":%s,\"parallel_prefix_guard\":%s,\"split_rebind\":%s,\"reuse_prefix\":%s,\"physical_snapshot\":%s,\"scene_transport_bytes\":%zu,\"scene_scratch_omitted_bytes\":%zu,\"scratch_poison\":%u,\"snapshot_comparison_scope\":\"%s\",\"prefix_executions\":%u,\"prefix_immutability_checks\":%s,\"prefix_immutability_errors\":%u,\"prefix_guard_bytes\":%zu,\"keep\":%u,\"warmups\":%u,\"repeats\":%u,\"selected_state_checks\":%s,\"compaction_errors\":%u,\"first_error_section\":%u,\"first_error_offset\":%u,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"cold_storage\":\"host-prepared-empty-template\",\"cold_storage_host_physics_calls\":0,\"cold_storage_ms\":%.9f,\"allocated_bytes\":%llu,\"registers\":%u,\"local_bytes\":%zu,\"requested_stack_kib\":%u,\"effective_stack_bytes\":%zu,\"guest_calls\":0,\"fallback_calls\":0,\"elapsed_before_stdout_ms\":%.9f,\"winner\":%u,\"samples\":[",options.cpu?"typed-host":"cuda-native",options.cpu?"checked":device_numeric_profile,count,options.prefix,options.horizon,(unsigned long long)ticks,failed,options.trace?"true":"false",options.host_prefix?"true":"false",options.threads,options.isolated?"true":"false",options.copy_u64?"true":"false",options.stream_copy?"true":"false",options.merge_rank?"true":"false",options.tiled_selection?"true":"false",options.compact_clone?"true":"false",options.parallel_guard?"true":"false",options.split_rebind?"true":"false",options.reuse_prefix?"true":"false",options.physical_snapshot?"true":"false",scene_copy_bytes,sizeof(Scene)-scene_copy_bytes,options.scratch_poison,options.physical_snapshot?"all-Game-and-all-World-BroadPhase-bytes":"all-Game-and-all-Scene-bytes",prefix_executions,options.reuse_prefix?"true":"false",validation.prefix_errors,options.reuse_prefix?prefix_guard_bytes:std::size_t(0),options.keep,options.warmups,options.repeats,options.validate?"true":"false",validation.errors,validation.first_section,validation.first_offset,sizeof(World),sizeof(Scene),cold_storage_ms,(unsigned long long)allocation_bytes,registers,local_bytes,options.stack_kib,effective_stack_bytes,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count(),ranks[0]);
 for(unsigned i=0;i<samples.size();++i){if(i)std::fputc(',',stderr);print_sample(samples[i]);}std::fprintf(stderr,"],\"clone_game_bytes\":%zu,\"clone_scene_bytes\":%zu,\"compile_profile\":{\"light_shape_sync\":%s,\"cached_rotation\":%s,\"f32_store\":%s,\"f32_div_store\":%s,\"device_trig\":%u,\"max_contacts\":%u,\"bounded_store\":%s,\"sparse_reset\":%s,\"inline_step\":%s,\"entry_index\":%s,\"wide_rebind\":%s}}\n",clone_game_bytes,clone_scene_bytes,IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC?"true":"false",IRISU_CUDA_NATIVE_CACHED_ROTATION?"true":"false",IRISU_CUDA_NATIVE_F32_STORE?"true":"false",IRISU_CUDA_NATIVE_F32_DIV_STORE?"true":"false",IRISU_CUDA_NATIVE_DEVICE_TRIG,physics::max_contacts,IRISU_CUDA_NATIVE_BOUNDED_STORE?"true":"false",IRISU_CUDA_NATIVE_SPARSE_RESET?"true":"false",IRISU_CUDA_NATIVE_INLINE_STEP?"true":"false",IRISU_CUDA_NATIVE_ENTRY_INDEX?"true":"false",IRISU_CUDA_NATIVE_WIDE_REBIND?"true":"false");
 if(failed||validation.errors)return 2;wire::emit_results(results,ranks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native-search: %s\n",e.what());return 1;}}
