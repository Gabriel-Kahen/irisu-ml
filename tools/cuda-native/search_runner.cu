// Independent typed complete-game futures. Shared wire/helpers are read-only;
// this executable does not link the guest translator or native oracle library.
#include "../../clone/cuda_native/game_world.cuh"
#include "../cuda-exact/search_io.hpp"
#include "search_accel.cuh"
#include "prefix_guard.cuh"
#include "shared_future_plan.hpp"
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
#ifndef IRISU_CUDA_NATIVE_CACHED_VELOCITY
#define IRISU_CUDA_NATIVE_CACHED_VELOCITY 0
#endif
#ifndef IRISU_CUDA_NATIVE_LOCAL_VELOCITY
#define IRISU_CUDA_NATIVE_LOCAL_VELOCITY 0
#endif
#ifndef IRISU_CUDA_NATIVE_LOCAL_POSITION
#define IRISU_CUDA_NATIVE_LOCAL_POSITION 0
#endif
#ifndef IRISU_CUDA_NATIVE_CONTACT_ANCHORS
#define IRISU_CUDA_NATIVE_CONTACT_ANCHORS 0
#endif
#ifndef IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT
#define IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT 0
#endif
namespace game=irisu::cuda_game;namespace physics=irisu::cuda_native;namespace wire=irisu::cuda_search;
using World=game::NativeGameWorld;using Scene=physics::ScenePipelineState;
struct Options {
 std::uint32_t seed{},prefix{},horizon{},keep{2},threads{32},stack_kib{16};
 std::int64_t target{4393102};bool host_prefix{},trace{true},validate{true},cpu{},isolated{},copy_u64{},split_rebind{},reuse_prefix{},physical_snapshot{},stream_copy{},merge_rank{},tiled_selection{},compact_clone{},parallel_guard{};
 unsigned warmups{1},repeats{3},scratch_poison{},shared_split{};bool share_futures{},adaptive_shared{},host_shared_checkpoints{},prefer_l1{};
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
// Carry the last StepResult across a shared checkpoint: terminal metadata and
// final-only hashing must behave exactly like one uninterrupted future().
struct SharedState {wire::Result result{};game::StepResult step{};};
GAME_HD void shared_segment(World& world,Scene& scene,const Options& options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate& candidate,unsigned begin,unsigned end,SharedState& state,bool finish){
 auto& result=state.result;auto& step=state.step;
 if(!result.failed&&!step.terminated&&!step.truncated)for(unsigned offset=begin;offset<end&&!world.failed();++offset){
  const unsigned tick=options.prefix+offset,incumbent=tick<replay_size?replay[tick]:0;
  step=world.step(game::native_replay_action(wire::input_at(candidate,offset,incumbent),tick));
  if(options.trace)wire::fold(result,world.frame(step));++result.simulated_ticks;if(step.terminated||step.truncated)break;
 }
 if(!finish)return;
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
__global__ void rebind_shared_host_kernel(World* worlds,Scene* scenes,const std::uintptr_t* old_bases,unsigned count,Validation* errors){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane<count&&!worlds[lane].rebind(scenes+lane,old_bases[lane]))atomicAdd(&errors->errors,1U);
}
__global__ void shared_prefix_futures_kernel(World* worlds,Scene* scenes,Options options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate* candidates,const unsigned* representatives,unsigned count,const unsigned* depths,const wire::Result* prefix,SharedState* states,unsigned long long* executed){
 const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;if(options.isolated&&(thread&31))return;const unsigned lane=options.isolated?thread/32:thread;if(lane>=count)return;
 SharedState state{};state.result=*prefix;if(lane)shared_segment(worlds[lane],scenes[lane],options,replay,replay_size,candidates[representatives[lane]],0,depths[lane],state,false);states[lane]=state;atomicAdd(executed,(unsigned long long)(state.result.simulated_ticks-prefix->simulated_ticks));
}
__global__ void shared_suffix_futures_kernel(World* worlds,Scene* scenes,Options options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate* candidates,unsigned count,const unsigned* depths,const unsigned* selection,const SharedState* states,wire::Result* output,unsigned long long* executed){
 const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;if(options.isolated&&(thread&31))return;const unsigned lane=options.isolated?thread/32:thread;if(lane>=count)return;
 SharedState state=states[selection[lane]];const auto before=state.result.simulated_ticks;shared_segment(worlds[lane],scenes[lane],options,replay,replay_size,candidates[lane],depths[selection[lane]],options.horizon,state,true);output[lane]=state.result;atomicAdd(executed,(unsigned long long)(state.result.simulated_ticks-before));
}
__global__ void shared_plans_kernel(const World* worlds,const Scene* scenes,unsigned count,compact::GamePlan* gp,compact::Plan* sp,Validation* errors){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;gp[lane].valid=false;if(!compact::build_plan(scenes[lane],sp[lane])||!compact::build_game_plan(worlds[lane],gp[lane]))atomicAdd(&errors->errors,1U);
}
__global__ void shared_compact_clones_kernel(World* worlds,Scene* scenes,const World* source,const Scene* source_scenes,unsigned count,const unsigned* selection,const compact::GamePlan* gp,const compact::Plan* sp,bool initialized,Validation* errors){
 const unsigned lane=blockIdx.x;if(lane>=count)return;const unsigned from=selection[lane];__shared__ bool valid;
 if(!threadIdx.x){valid=gp[from].valid&&(!initialized||bool(compact::clear_reachable(scenes[lane])));if(!valid)atomicAdd(&errors->errors,1U);}__syncthreads();if(!valid)return;
 compact::copy_game(worlds[lane],source[from],gp[from],threadIdx.x,blockDim.x);compact::copy_active(scenes[lane],source_scenes[from],sp[from],threadIdx.x,blockDim.x);__syncthreads();
 if(!threadIdx.x&&!compact::rebind_game(worlds[lane],scenes[lane],gp[from]))atomicAdd(&errors->errors,1U);
}
__global__ void independent_selected_futures_kernel(World* selected,Scene* scenes,Options options,const std::uint32_t* replay,unsigned replay_size,const wire::Candidate* candidates,const unsigned* ranks,unsigned keep,const wire::Result* prefix,const wire::Result* results,Validation* errors){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=keep)return;wire::Result result;future(selected[lane],scenes[lane],options,replay,replay_size,candidates[ranks[lane]],*prefix,result);
 const auto* a=reinterpret_cast<const unsigned char*>(&result);const auto* b=reinterpret_cast<const unsigned char*>(results+ranks[lane]);for(unsigned i=0;i<sizeof(result);++i)if(a[i]!=b[i]){atomicAdd(&errors->errors,1U);atomicCAS(&errors->first_section,0U,6U);atomicMin(&errors->first_offset,i);break;}
}
__global__ void semantic_selected_verify_kernel(World* selected,Scene* selected_scenes,World* source,Scene* source_scenes,const unsigned* ranks,unsigned keep,compact::GamePlan* plans,Validation* errors){
 const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=keep)return;auto& a=selected[lane];auto& b=source[ranks[lane]];auto& sa=selected_scenes[lane];auto& sb=source_scenes[ranks[lane]];
 if(!compact::build_game_plan(a,plans[lane])){atomicAdd(&errors->errors,1U);return;}
 constexpr std::uintptr_t canonical=0x100000;bool valid=a.relocate_body_pointers(body_base(a),canonical)&&b.relocate_body_pointers(body_base(b),canonical);a.bind(nullptr);b.bind(nullptr);
 const auto game_offset=compact::first_game_difference(a,b,plans[lane]),scene_offset=compact::first_difference(sa,sb);
 valid=a.rebind(&sa,canonical)&&b.rebind(&sb,canonical)&&valid;
 if(!valid||game_offset!=physics::null_index||scene_offset!=physics::null_index){atomicAdd(&errors->errors,1U);atomicCAS(&errors->first_section,0U,game_offset!=physics::null_index?7U:8U);atomicMin(&errors->first_offset,game_offset!=physics::null_index?game_offset:scene_offset);}
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
 const auto started=std::chrono::steady_clock::now();if(argc<5)throw std::runtime_error("usage: native-search replay specs prefix horizon [--host-prefix] [--final-only] [--repeats N] [--warmups N] [--keep N] [--threads N] [--stack-kib N] [--target N] [--skip-compaction-check] [--isolated-warps] [--copy-u64] [--stream-copy] [--merge-rank] [--tiled-selection] [--compact-clone] [--parallel-prefix-guard] [--split-rebind] [--reuse-prefix] [--physical-snapshot] [--scratch-poison 1|2] [--host-shared-checkpoints] [--adaptive-shared-prefixes] [--prefer-l1] [--share-future-prefixes] [--shared-split N] [--cpu]");
 auto input=wire::load_inputs(argv[1],argv[2],argv[3],argv[4]);Options options;options.seed=input.options.seed;options.prefix=input.options.prefix;options.horizon=input.options.horizon;const unsigned count=input.candidates.size();options.keep=std::min(2U,count);
 for(int i=5;i<argc;++i){const std::string arg=argv[i];if(arg=="--host-prefix")options.host_prefix=true;else if(arg=="--final-only")options.trace=false;else if(arg=="--skip-compaction-check")options.validate=false;else if(arg=="--isolated-warps")options.isolated=true;else if(arg=="--copy-u64")options.copy_u64=true;else if(arg=="--stream-copy"){options.stream_copy=true;options.copy_u64=true;}else if(arg=="--merge-rank")options.merge_rank=true;else if(arg=="--tiled-selection")options.tiled_selection=true;else if(arg=="--parallel-prefix-guard")options.parallel_guard=true;else if(arg=="--compact-clone"){options.compact_clone=true;options.physical_snapshot=true;}else if(arg=="--split-rebind")options.split_rebind=true;else if(arg=="--reuse-prefix")options.reuse_prefix=true;else if(arg=="--physical-snapshot")options.physical_snapshot=true;else if(arg=="--cpu")options.cpu=true;else if(arg=="--host-shared-checkpoints"){options.host_shared_checkpoints=true;options.share_futures=true;}else if(arg=="--adaptive-shared-prefixes"){options.adaptive_shared=true;options.share_futures=true;}else if(arg=="--prefer-l1")options.prefer_l1=true;else if(arg=="--share-future-prefixes")options.share_futures=true;else if(arg=="--shared-split"&&i+1<argc){const auto split=std::stoull(argv[++i]);if(!split||split>options.horizon)throw std::runtime_error("shared split outside horizon");options.shared_split=unsigned(split);options.share_futures=true;}
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
 if(options.host_shared_checkpoints&&(!options.host_prefix||options.cpu))throw std::runtime_error("host shared checkpoints require --host-prefix and CUDA execution");
 if(options.adaptive_shared&&options.shared_split)throw std::runtime_error("adaptive sharing does not take a fixed split");
 auto shared_plan=options.adaptive_shared?irisu::shared_future::build_adaptive(input):(options.share_futures?irisu::shared_future::build(input,options.shared_split):irisu::shared_future::Plan{});if(!options.share_futures)shared_plan.independent_ticks=shared_plan.planned_ticks=std::uint64_t(count)*options.horizon;
 const unsigned shared_count=shared_plan.representatives.size();std::uint64_t executed_future_ticks=0,measured_executed_future_ticks=0,host_shared_checkpoint_ticks=0;unsigned host_shared_checkpoint_executions=0,shared_checkpoint_immutability_errors=0,shared_checkpoint_guard_executions=0;
 const std::uint64_t shared_preparation_ticks=std::accumulate(shared_plan.depths.begin(),shared_plan.depths.end(),std::uint64_t(0)),planned_gpu_future_ticks_per_batch=options.cpu?0:shared_plan.planned_ticks-(options.host_shared_checkpoints?shared_preparation_ticks:0);
 const std::size_t scene_copy_bytes=options.physical_snapshot?physical_scene_bytes:sizeof(Scene);
 std::vector<wire::Result> results(count),previous;std::vector<unsigned> ranks(count),previous_ranks;std::vector<Sample> samples;Validation validation;std::uint64_t allocation_bytes=0;unsigned registers=0,prefix_executions=0;std::size_t local_bytes=0,effective_stack_bytes=0;double cold_storage_ms=0;std::size_t clone_game_bytes=sizeof(World),clone_scene_bytes=scene_copy_bytes;
 if(options.cpu){
  Environment environment;auto prefix=std::make_unique<World>();auto prefix_scene=std::make_unique<Scene>();std::unique_ptr<World[]> worlds(new World[count]);std::unique_ptr<Scene[]> scenes(new Scene[count]);auto selected=std::make_unique<World>();auto selected_scene=std::make_unique<Scene>();
  std::unique_ptr<World[]> shared_worlds(shared_count?new World[shared_count]:nullptr);std::unique_ptr<Scene[]> shared_scenes(shared_count?new Scene[shared_count]:nullptr);std::vector<SharedState> shared_states(shared_count);
  wire::Result prefix_result;std::vector<unsigned char> prefix_guard(options.reuse_prefix?prefix_guard_bytes:0);
  for(unsigned run=0;run<options.warmups+options.repeats;++run){Sample sample;const auto begin=std::chrono::steady_clock::now();const bool execute_prefix=!options.reuse_prefix||!run;
   if(execute_prefix){cold_storage(prefix.get(),prefix_scene.get());prepare_prefix(prefix.get(),prefix_scene.get(),options,input.replay.data(),&prefix_result);++prefix_executions;
    if(options.reuse_prefix){std::memcpy(prefix_guard.data(),prefix.get(),sizeof(World));std::memcpy(prefix_guard.data()+sizeof(World),prefix_scene.get(),sizeof(Scene));std::memcpy(prefix_guard.data()+sizeof(World)+sizeof(Scene),&prefix_result,sizeof(prefix_result));}}
   const auto a=std::chrono::steady_clock::now();sample.prefix_ms=execute_prefix?std::chrono::duration<double,std::milli>(a-begin).count():0;
   if(!shared_count)for(unsigned i=0;i<count;++i){host_clone(worlds[i],scenes[i],*prefix,*prefix_scene,scene_copy_bytes);if(options.scratch_poison)host_poison_scratch(scenes[i],options.scratch_poison,i);}const auto b=std::chrono::steady_clock::now();sample.branch_ms=std::chrono::duration<double,std::milli>(b-a).count();
   executed_future_ticks=0;
   if(shared_count){
    for(unsigned g=0;g<shared_count;++g){host_clone(shared_worlds[g],shared_scenes[g],*prefix,*prefix_scene,scene_copy_bytes);if(options.scratch_poison)host_poison_scratch(shared_scenes[g],options.scratch_poison,g);shared_states[g]={};shared_states[g].result=prefix_result;if(g)shared_segment(shared_worlds[g],shared_scenes[g],options,input.replay.data(),input.replay.size(),input.candidates[shared_plan.representatives[g]],0,shared_plan.depths[g],shared_states[g],false);executed_future_ticks+=shared_states[g].result.simulated_ticks-prefix_result.simulated_ticks;}
    for(unsigned i=0;i<count;++i){const auto g=shared_plan.selection[i];host_clone(worlds[i],scenes[i],shared_worlds[g],shared_scenes[g],scene_copy_bytes);if(options.scratch_poison)host_poison_scratch(scenes[i],options.scratch_poison,i);auto state=shared_states[g];const auto before=state.result.simulated_ticks;shared_segment(worlds[i],scenes[i],options,input.replay.data(),input.replay.size(),input.candidates[i],shared_plan.depths[g],options.horizon,state,true);results[i]=state.result;executed_future_ticks+=state.result.simulated_ticks-before;}
   }else for(unsigned i=0;i<count;++i){future(worlds[i],scenes[i],options,input.replay.data(),input.replay.size(),input.candidates[i],prefix_result,results[i]);executed_future_ticks+=results[i].simulated_ticks;}
   if(run>=options.warmups)measured_executed_future_ticks+=executed_future_ticks;
   const auto c=std::chrono::steady_clock::now();sample.rollout_ms=std::chrono::duration<double,std::milli>(c-b).count();std::iota(ranks.begin(),ranks.end(),0U);std::sort(ranks.begin(),ranks.end(),[&](unsigned a,unsigned b){return wire::better(results[a],a,results[b],b);});const auto d=std::chrono::steady_clock::now();sample.rank_compact_ms=std::chrono::duration<double,std::milli>(d-c).count();
   if(options.validate)for(unsigned k=0;k<options.keep;++k){const unsigned i=ranks[k];if(shared_count){host_clone(*selected,*selected_scene,*prefix,*prefix_scene,scene_copy_bytes);wire::Result independent;future(*selected,*selected_scene,options,input.replay.data(),input.replay.size(),input.candidates[i],prefix_result,independent);if(std::memcmp(&independent,&results[i],sizeof(independent)))++validation.errors;}else host_clone(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes);if(!host_equal(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes))++validation.errors;if(options.physical_snapshot){host_poison_scratch(*selected_scene,1);host_poison_scratch(scenes[i],2);}const auto sa=selected->step({}),sb=worlds[i].step({});const auto fa=selected->frame(sa),fb=worlds[i].frame(sb);if(selected->failed()||worlds[i].failed()||std::memcmp(&fa,&fb,sizeof(fa))||!host_equal(*selected,*selected_scene,worlds[i],scenes[i],scene_copy_bytes))++validation.errors;}
   if(options.reuse_prefix&&(std::memcmp(prefix_guard.data(),prefix.get(),sizeof(World))||std::memcmp(prefix_guard.data()+sizeof(World),prefix_scene.get(),sizeof(Scene))||std::memcmp(prefix_guard.data()+sizeof(World)+sizeof(Scene),&prefix_result,sizeof(prefix_result))))throw std::runtime_error("typed host reused checkpoint changed");
   const auto end=std::chrono::steady_clock::now();sample.validation_ms=std::chrono::duration<double,std::milli>(end-d).count();sample.total_ms=std::chrono::duration<double,std::milli>(end-begin).count();if(run>=options.warmups)samples.push_back(sample);
   if(!previous.empty()&&(std::memcmp(previous.data(),results.data(),count*sizeof(results[0]))||previous_ranks!=ranks))throw std::runtime_error("typed host repeated search differs");previous=results;previous_ranks=ranks;
  }
 }else{
#ifdef __CUDACC__
  if(options.prefer_l1){check(cudaFuncSetCacheConfig(future_kernel,cudaFuncCachePreferL1));check(cudaFuncSetCacheConfig(shared_prefix_futures_kernel,cudaFuncCachePreferL1));check(cudaFuncSetCacheConfig(shared_suffix_futures_kernel,cudaFuncCachePreferL1));}
  check(cudaDeviceSetLimit(cudaLimitStackSize,std::size_t(options.stack_kib)*1024));check(cudaDeviceGetLimit(&effective_stack_bytes,cudaLimitStackSize));Memory memory;
  auto* replay=memory.allocate<std::uint32_t>(input.replay.size());auto* candidates=memory.allocate<wire::Candidate>(count);auto* prefix=memory.allocate<World>(1);auto* prefix_scene=memory.allocate<Scene>(1);auto* prefix_result=memory.allocate<wire::Result>(1);auto* worlds=memory.allocate<World>(count);auto* scenes=memory.allocate<Scene>(count);auto* output=memory.allocate<wire::Result>(count);auto* order=memory.allocate<unsigned>(count);auto* selected=memory.allocate<World>(options.keep);auto* selected_scenes=memory.allocate<Scene>(options.keep);auto* errors=memory.allocate<Validation>(1);
  compact::GamePlan* game_plan{};compact::Plan* scene_plan{};
  if(options.compact_clone){game_plan=memory.allocate<compact::GamePlan>(1);scene_plan=memory.allocate<compact::Plan>(1);check(cudaMemset(game_plan,0,sizeof(*game_plan)));check(cudaMemset(scene_plan,0,sizeof(*scene_plan)));check(cudaMemset(worlds,0,count*sizeof(World)));check(cudaMemset2D(scenes,sizeof(Scene),0,physical_scene_bytes,count));}
  World* empty_world{};Scene* empty_scene{};World* saved_prefix{};Scene* saved_prefix_scene{};wire::Result* saved_prefix_result{};
  if(options.reuse_prefix){saved_prefix=memory.allocate<World>(1);saved_prefix_scene=memory.allocate<Scene>(1);saved_prefix_result=memory.allocate<wire::Result>(1);}
  if(!options.host_prefix){const auto cold_begin=std::chrono::steady_clock::now();auto host=cold_object<World>();auto scene=cold_object<Scene>();empty_world=memory.allocate<World>(1);empty_scene=memory.allocate<Scene>(1);check(cudaMemcpy(empty_world,host.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(empty_scene,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));cold_storage_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_begin).count();}
  std::size_t rank_bytes{};unsigned char* rank_workspace{};if(options.merge_rank){check(physics::search_accel::rank_workspace_bytes(output,order,count,rank_bytes));rank_workspace=memory.allocate<unsigned char>(rank_bytes);}
  World* shared_worlds{};Scene* shared_scenes{};SharedState* shared_states{};unsigned* shared_representatives{};unsigned* shared_selection{};unsigned* shared_depths{};compact::GamePlan* shared_game_plans{};compact::Plan* shared_scene_plans{};unsigned long long* shared_executed{};compact::GamePlan* validation_game_plans{};std::uintptr_t* shared_old_bases{};World* saved_shared_worlds{};Scene* saved_shared_scenes{};SharedState* saved_shared_states{};Validation* shared_cache_errors{};
  if(shared_count){
   if(options.validate)validation_game_plans=memory.allocate<compact::GamePlan>(options.keep);if(options.host_shared_checkpoints){shared_old_bases=memory.allocate<std::uintptr_t>(shared_count);saved_shared_worlds=memory.allocate<World>(shared_count);saved_shared_scenes=memory.allocate<Scene>(shared_count);saved_shared_states=memory.allocate<SharedState>(shared_count);shared_cache_errors=memory.allocate<Validation>(1);}
   shared_worlds=memory.allocate<World>(shared_count);shared_scenes=memory.allocate<Scene>(shared_count);shared_states=memory.allocate<SharedState>(shared_count);shared_representatives=memory.allocate<unsigned>(shared_count);shared_selection=memory.allocate<unsigned>(count);shared_depths=memory.allocate<unsigned>(shared_count);shared_executed=memory.allocate<unsigned long long>(1);
   check(cudaMemcpy(shared_depths,shared_plan.depths.data(),shared_count*sizeof(unsigned),cudaMemcpyHostToDevice));check(cudaMemcpy(shared_representatives,shared_plan.representatives.data(),shared_count*sizeof(unsigned),cudaMemcpyHostToDevice));check(cudaMemcpy(shared_selection,shared_plan.selection.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));
   if(options.compact_clone){shared_game_plans=memory.allocate<compact::GamePlan>(shared_count);shared_scene_plans=memory.allocate<compact::Plan>(shared_count);check(cudaMemset(shared_worlds,0,shared_count*sizeof(World)));check(cudaMemset2D(shared_scenes,sizeof(Scene),0,physical_scene_bytes,shared_count));}
  }
  allocation_bytes=memory.bytes;
  check(cudaMemcpy(replay,input.replay.data(),input.replay.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(candidates,input.candidates.data(),count*sizeof(wire::Candidate),cudaMemcpyHostToDevice));cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,future_kernel));registers=attributes.numRegs;local_bytes=attributes.localSizeBytes;
  for(unsigned run=0;run<options.warmups+options.repeats;++run){const auto begin=std::chrono::steady_clock::now();Sample sample;validation={};check(cudaMemcpy(errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));if(shared_cache_errors)check(cudaMemcpy(shared_cache_errors,&validation,sizeof(validation),cudaMemcpyHostToDevice));std::uintptr_t old_base{};const bool execute_prefix=!options.reuse_prefix||!run;
   if(execute_prefix&&options.host_prefix){Environment environment;auto host=cold_object<World>();auto scene=cold_object<Scene>();wire::Result initial;prepare_prefix(host.get(),scene.get(),options,input.replay.data(),&initial);old_base=body_base(*host);check(cudaMemcpy(prefix,host.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_scene,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));check(cudaMemcpy(prefix_result,&initial,sizeof(initial),cudaMemcpyHostToDevice));
    if(options.host_shared_checkpoints&&shared_count){std::vector<std::uintptr_t> old_bases(shared_count);for(unsigned g=0;g<shared_count;++g){auto checkpoint=cold_object<World>();auto checkpoint_scene=cold_object<Scene>();host_clone(*checkpoint,*checkpoint_scene,*host,*scene,scene_copy_bytes);if(options.scratch_poison)host_poison_scratch(*checkpoint_scene,options.scratch_poison,g);SharedState state{};state.result=initial;if(g)shared_segment(*checkpoint,*checkpoint_scene,options,input.replay.data(),input.replay.size(),input.candidates[shared_plan.representatives[g]],0,shared_plan.depths[g],state,false);host_shared_checkpoint_ticks+=state.result.simulated_ticks-initial.simulated_ticks;old_bases[g]=body_base(*checkpoint);check(cudaMemcpy(shared_worlds+g,checkpoint.get(),sizeof(World),cudaMemcpyHostToDevice));check(cudaMemcpy(shared_scenes+g,checkpoint_scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));check(cudaMemcpy(shared_states+g,&state,sizeof(state),cudaMemcpyHostToDevice));}check(cudaMemcpy(shared_old_bases,old_bases.data(),shared_count*sizeof(std::uintptr_t),cudaMemcpyHostToDevice));++host_shared_checkpoint_executions;}
    sample.host_prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
   Event a,b,c,d,e,f;a.record();
   if(execute_prefix){if(options.host_prefix)rebind_prefix_kernel<<<1,1>>>(prefix,prefix_scene,old_base,errors);else prefix_kernel<<<1,256>>>(prefix,prefix_scene,empty_world,empty_scene,options,replay,prefix_result);if(options.host_shared_checkpoints&&shared_count){rebind_shared_host_kernel<<<(shared_count+31)/32,32>>>(shared_worlds,shared_scenes,shared_old_bases,shared_count,errors);check(cudaMemcpyAsync(saved_shared_worlds,shared_worlds,shared_count*sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_shared_scenes,shared_scenes,shared_count*sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_shared_states,shared_states,shared_count*sizeof(SharedState),cudaMemcpyDeviceToDevice));}++prefix_executions;
    if(options.reuse_prefix){check(cudaMemcpyAsync(saved_prefix,prefix,sizeof(World),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_prefix_scene,prefix_scene,sizeof(Scene),cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(saved_prefix_result,prefix_result,sizeof(wire::Result),cudaMemcpyDeviceToDevice));}}
   if(execute_prefix&&options.compact_clone){compact_plan_kernel<<<1,1>>>(prefix,prefix_scene,game_plan,scene_plan,errors);check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("compact prefix plan rejected");}
   b.record();const unsigned initial_count=shared_count?shared_count:count;auto* initial_worlds=shared_count?shared_worlds:worlds;auto* initial_scenes=shared_count?shared_scenes:scenes;
   if(!options.host_shared_checkpoints||!shared_count){if(options.compact_clone)compact_clone_kernel<<<initial_count,256>>>(initial_worlds,initial_scenes,prefix,prefix_scene,initial_count,game_plan,scene_plan,run!=0,errors);else launch_clones(initial_worlds,initial_scenes,prefix,prefix_scene,initial_count,nullptr,errors,options);if(options.scratch_poison)poison_scratch_kernel<<<initial_count,256>>>(initial_scenes,initial_count,nullptr,options.scratch_poison);}c.record();
   const unsigned threads=options.isolated?std::max(32U,options.threads):options.threads,total=count*(options.isolated?32U:1U);
   if(shared_count){
    check(cudaMemsetAsync(shared_executed,0,sizeof(unsigned long long)));const unsigned shared_total=shared_count*(options.isolated?32U:1U);
    if(!options.host_shared_checkpoints)shared_prefix_futures_kernel<<<(shared_total+threads-1)/threads,threads>>>(shared_worlds,shared_scenes,options,replay,input.replay.size(),candidates,shared_representatives,shared_count,shared_depths,prefix_result,shared_states,shared_executed);
    if(options.compact_clone){if(!options.host_shared_checkpoints||execute_prefix){shared_plans_kernel<<<(shared_count+31)/32,32>>>(shared_worlds,shared_scenes,shared_count,shared_game_plans,shared_scene_plans,errors);check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));if(validation.errors)throw std::runtime_error("shared intermediate checkpoint rejected");}shared_compact_clones_kernel<<<count,256>>>(worlds,scenes,shared_worlds,shared_scenes,count,shared_selection,shared_game_plans,shared_scene_plans,run!=0,errors);}
    else launch_clones(worlds,scenes,shared_worlds,shared_scenes,count,shared_selection,errors,options);
    if(options.scratch_poison)poison_scratch_kernel<<<count,256>>>(scenes,count,nullptr,options.scratch_poison);
    shared_suffix_futures_kernel<<<(total+threads-1)/threads,threads>>>(worlds,scenes,options,replay,input.replay.size(),candidates,count,shared_depths,shared_selection,shared_states,output,shared_executed);
   }else future_kernel<<<(total+threads-1)/threads,threads>>>(worlds,scenes,options,replay,input.replay.size(),candidates,count,prefix_result,output);
   d.record();if(options.merge_rank)check(physics::search_accel::rank_async(output,order,count,rank_workspace,rank_bytes));else rank_kernel<<<(count+127)/128,128>>>(output,count,order);if(shared_count&&options.validate)launch_clones(selected,selected_scenes,prefix,prefix_scene,options.keep,nullptr,errors,options);else launch_clones(selected,selected_scenes,worlds,scenes,options.keep,order,errors,options);e.record();
   if(options.validate){
    if(shared_count){independent_selected_futures_kernel<<<(options.keep+31)/32,32>>>(selected,selected_scenes,options,replay,input.replay.size(),candidates,order,options.keep,prefix_result,output,errors);semantic_selected_verify_kernel<<<(options.keep+31)/32,32>>>(selected,selected_scenes,worlds,scenes,order,options.keep,validation_game_plans,errors);}
    else verify_kernel<<<options.keep,256>>>(selected,selected_scenes,worlds,scenes,order,options.keep,errors,scene_copy_bytes);
    if(options.physical_snapshot){poison_scratch_kernel<<<options.keep,256>>>(selected_scenes,options.keep,nullptr,1);poison_scratch_kernel<<<options.keep,256>>>(scenes,options.keep,order,2);}continue_kernel<<<(options.keep+31)/32,32>>>(selected,worlds,order,options.keep,errors);
    if(shared_count)semantic_selected_verify_kernel<<<(options.keep+31)/32,32>>>(selected,selected_scenes,worlds,scenes,order,options.keep,validation_game_plans,errors);else verify_kernel<<<options.keep,256>>>(selected,selected_scenes,worlds,scenes,order,options.keep,errors,scene_copy_bytes);
   }
   if(options.reuse_prefix){if(options.parallel_guard)check(physics::prefix_guard::check_prefix_async(prefix,prefix_scene,prefix_result,saved_prefix,saved_prefix_scene,saved_prefix_result,errors));else verify_prefix_kernel<<<1,256>>>(prefix,prefix_scene,prefix_result,saved_prefix,saved_prefix_scene,saved_prefix_result,errors);}
   if(shared_cache_errors){for(unsigned g=0;g<shared_count;++g)check(physics::prefix_guard::check_prefix_async(shared_worlds+g,shared_scenes+g,shared_states+g,saved_shared_worlds+g,saved_shared_scenes+g,saved_shared_states+g,shared_cache_errors));++shared_checkpoint_guard_executions;}
   check(cudaGetLastError());f.record();check(cudaEventSynchronize(f.value));check(cudaMemcpy(results.data(),output,count*sizeof(wire::Result),cudaMemcpyDeviceToHost));check(cudaMemcpy(ranks.data(),order,count*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&validation,errors,sizeof(validation),cudaMemcpyDeviceToHost));if(shared_cache_errors){Validation guard;check(cudaMemcpy(&guard,shared_cache_errors,sizeof(guard),cudaMemcpyDeviceToHost));shared_checkpoint_immutability_errors+=guard.errors;if(guard.errors)throw std::runtime_error("shared immutable source checkpoint changed");}sample.prefix_ms=execute_prefix?elapsed(a,b):0;sample.branch_ms=elapsed(b,c);sample.rollout_ms=elapsed(c,d);sample.rank_compact_ms=elapsed(d,e);sample.validation_ms=elapsed(e,f);sample.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(run>=options.warmups)samples.push_back(sample);
   if(shared_count){unsigned long long actual{};check(cudaMemcpy(&actual,shared_executed,sizeof(actual),cudaMemcpyDeviceToHost));executed_future_ticks=actual;}else{executed_future_ticks=0;for(const auto& r:results)executed_future_ticks+=r.simulated_ticks;}
   if(run>=options.warmups)measured_executed_future_ticks+=executed_future_ticks;
   if(validation.errors)throw std::runtime_error("CUDA selected full-state check failed: errors="+std::to_string(validation.errors)+" section="+std::to_string(validation.first_section)+" offset="+std::to_string(validation.first_offset));
   if(!previous.empty()&&(std::memcmp(previous.data(),results.data(),count*sizeof(results[0]))||previous_ranks!=ranks))throw std::runtime_error("CUDA repeated search differs");previous=results;previous_ranks=ranks;
  }
  if(options.compact_clone){compact::GamePlan gp;compact::Plan sp;check(cudaMemcpy(&gp,game_plan,sizeof(gp),cudaMemcpyDeviceToHost));check(cudaMemcpy(&sp,scene_plan,sizeof(sp),cudaMemcpyDeviceToHost));clone_game_bytes=gp.bytes;clone_scene_bytes=compact::transport_bytes(sp);}
#else
  throw std::runtime_error("CUDA unavailable: use explicit --cpu for typed host validation");
#endif
 }
 unsigned failed=0;std::uint64_t ticks=0;for(const auto& result:results){failed+=bool(result.failed);ticks+=result.simulated_ticks;}
 std::fprintf(stderr,"{\"identity\":\"experimental-independent-typed-game-search-v1\",\"backend\":\"%s\",\"numeric_profile\":\"%s\",\"host_prefix_numeric_profile\":\"checked\",\"promotion_eligible\":false,\"candidates\":%u,\"prefix_ticks\":%u,\"horizon\":%u,\"simulated_ticks\":%llu,\"failed_candidates\":%u,\"trace_every_tick\":%s,\"host_prefix\":%s,\"threads\":%u,\"isolated_warps\":%s,\"copy_u64\":%s,\"stream_copy\":%s,\"merge_rank\":%s,\"tiled_selection\":%s,\"compact_clone\":%s,\"parallel_prefix_guard\":%s,\"split_rebind\":%s,\"reuse_prefix\":%s,\"physical_snapshot\":%s,\"scene_transport_bytes\":%zu,\"scene_scratch_omitted_bytes\":%zu,\"scratch_poison\":%u,\"snapshot_comparison_scope\":\"%s\",\"prefix_executions\":%u,\"prefix_immutability_checks\":%s,\"prefix_immutability_errors\":%u,\"prefix_guard_bytes\":%zu,\"keep\":%u,\"warmups\":%u,\"repeats\":%u,\"selected_state_checks\":%s,\"compaction_errors\":%u,\"first_error_section\":%u,\"first_error_offset\":%u,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"cold_storage\":\"host-prepared-empty-template\",\"cold_storage_host_physics_calls\":0,\"cold_storage_ms\":%.9f,\"allocated_bytes\":%llu,\"registers\":%u,\"local_bytes\":%zu,\"requested_stack_kib\":%u,\"effective_stack_bytes\":%zu,\"guest_calls\":0,\"fallback_calls\":0,\"elapsed_before_stdout_ms\":%.9f,\"winner\":%u,\"samples\":[",options.cpu?"typed-host":"cuda-native",options.cpu?"checked":device_numeric_profile,count,options.prefix,options.horizon,(unsigned long long)ticks,failed,options.trace?"true":"false",options.host_prefix?"true":"false",options.threads,options.isolated?"true":"false",options.copy_u64?"true":"false",options.stream_copy?"true":"false",options.merge_rank?"true":"false",options.tiled_selection?"true":"false",options.compact_clone?"true":"false",options.parallel_guard?"true":"false",options.split_rebind?"true":"false",options.reuse_prefix?"true":"false",options.physical_snapshot?"true":"false",scene_copy_bytes,sizeof(Scene)-scene_copy_bytes,options.scratch_poison,(shared_count&&options.validate)?"live-Game-values-active-World-and-full-SAP":(options.physical_snapshot?"all-Game-and-all-World-BroadPhase-bytes":"all-Game-and-all-Scene-bytes"),prefix_executions,options.reuse_prefix?"true":"false",validation.prefix_errors,options.reuse_prefix?prefix_guard_bytes:std::size_t(0),options.keep,options.warmups,options.repeats,options.validate?"true":"false",validation.errors,validation.first_section,validation.first_offset,sizeof(World),sizeof(Scene),cold_storage_ms,(unsigned long long)allocation_bytes,registers,local_bytes,options.stack_kib,effective_stack_bytes,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count(),ranks[0]);
 for(unsigned i=0;i<samples.size();++i){if(i)std::fputc(',',stderr);print_sample(samples[i]);}std::fprintf(stderr,"],\"prefer_l1\":%s,\"adaptive_shared_prefixes\":%s,\"shared_future_prefixes\":%s,\"shared_split\":%u,\"shared_checkpoint_count\":%u,\"independent_selected_replays_per_batch\":%u,\"host_shared_checkpoints\":%s,\"host_shared_checkpoint_ticks\":%llu,\"host_shared_checkpoint_executions\":%u,\"shared_checkpoint_preparation_ticks_per_load\":%llu,\"planned_gpu_future_ticks_per_batch\":%llu,\"shared_checkpoint_immutability_checks\":%s,\"shared_checkpoint_guard_executions\":%u,\"shared_checkpoint_immutability_errors\":%u,\"shared_checkpoint_guard_bytes_per_batch\":%llu,\"planned_independent_future_ticks\":%llu,\"planned_executed_future_ticks\":%llu,\"executed_future_ticks\":%llu,\"measured_executed_future_ticks\":%llu,\"gpu_executed_future_ticks\":%llu,\"measured_gpu_executed_future_ticks\":%llu,\"clone_game_bytes\":%zu,\"clone_scene_bytes\":%zu,\"compile_profile\":{\"light_shape_sync\":%s,\"cached_rotation\":%s,\"f32_store\":%s,\"f32_div_store\":%s,\"device_trig\":%u,\"max_contacts\":%u,\"bounded_store\":%s,\"sparse_reset\":%s,\"inline_step\":%s,\"entry_index\":%s,\"wide_rebind\":%s,\"cached_velocity\":%s,\"local_velocity\":%s,\"local_position\":%s,\"contact_anchors\":%s,\"split_construct\":%s}}\n",options.prefer_l1?"true":"false",options.adaptive_shared?"true":"false",options.share_futures?"true":"false",shared_plan.split,shared_count,(shared_count&&options.validate)?options.keep:0U,options.host_shared_checkpoints?"true":"false",(unsigned long long)host_shared_checkpoint_ticks,host_shared_checkpoint_executions,(unsigned long long)shared_preparation_ticks,(unsigned long long)planned_gpu_future_ticks_per_batch,(options.host_shared_checkpoints&&shared_count)?"true":"false",shared_checkpoint_guard_executions,shared_checkpoint_immutability_errors,(unsigned long long)((options.host_shared_checkpoints?shared_count:0)*(sizeof(World)+sizeof(Scene)+sizeof(SharedState))),(unsigned long long)shared_plan.independent_ticks,(unsigned long long)shared_plan.planned_ticks,(unsigned long long)executed_future_ticks,(unsigned long long)measured_executed_future_ticks,(unsigned long long)(options.cpu?0:executed_future_ticks),(unsigned long long)(options.cpu?0:measured_executed_future_ticks),clone_game_bytes,clone_scene_bytes,IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC?"true":"false",IRISU_CUDA_NATIVE_CACHED_ROTATION?"true":"false",IRISU_CUDA_NATIVE_F32_STORE?"true":"false",IRISU_CUDA_NATIVE_F32_DIV_STORE?"true":"false",IRISU_CUDA_NATIVE_DEVICE_TRIG,physics::max_contacts,IRISU_CUDA_NATIVE_BOUNDED_STORE?"true":"false",IRISU_CUDA_NATIVE_SPARSE_RESET?"true":"false",IRISU_CUDA_NATIVE_INLINE_STEP?"true":"false",IRISU_CUDA_NATIVE_ENTRY_INDEX?"true":"false",IRISU_CUDA_NATIVE_WIDE_REBIND?"true":"false",IRISU_CUDA_NATIVE_CACHED_VELOCITY?"true":"false",IRISU_CUDA_NATIVE_LOCAL_VELOCITY?"true":"false",IRISU_CUDA_NATIVE_LOCAL_POSITION?"true":"false",IRISU_CUDA_NATIVE_CONTACT_ANCHORS?"true":"false",IRISU_CUDA_NATIVE_SPLIT_CONSTRUCT?"true":"false");
 if(failed||validation.errors)return 2;wire::emit_results(results,ranks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native-search: %s\n",e.what());return 1;}}
