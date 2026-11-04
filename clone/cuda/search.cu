#include "search.hpp"
#include "search_session.hpp"
#include "world.cuh"
#include "../../tools/cuda-exact/image_identity.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cfenv>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace irisu::cuda_search {
namespace {
using cuda_game::World;
struct GuestCache {
  World& world;cuda_guest::Guest local{};bool active{};
  GAME_HD GuestCache(World& w,bool enabled):world(w),active(enabled){
    if(active){local=world.guest;world.simulator.physics_.bind(&local);}
  }
  GAME_HD void finish(){if(active){world.guest=local;world.bind(local.memory);active=false;}}
};
GAME_HD void prepare_prefix(World* world,unsigned char* arena,Options options,
 const unsigned char* image,unsigned image_size,const std::uint32_t* replay,Result* prefix) {
  ::new(static_cast<void*>(world)) World();
  world->initialize(arena,options.arena_bytes,image,image_size,options.seed);
  GuestCache cache(*world,options.cache_guest);
  cuda_game::StepResult step{};
  for(unsigned tick=0;tick<options.prefix&&!world->failed();++tick) {
    if(step.terminated||step.truncated){
      if(options.cache_guest)cache.local.fail(cuda_guest::aborted);
      else world->guest.fail(cuda_guest::aborted);
      break;
    }
    step=world->step(cuda_game::replay_action(replay[tick],tick));
  }
  cache.finish();*prefix=Result{};fold(*prefix,world->frame(step));
  prefix->failed=world->failed()||step.terminated||step.truncated;
  prefix->guest_error=world->guest.error;prefix->fault_pc=world->guest.fault_pc;
}
__global__ void prefix_kernel(World* world,unsigned char* arena,Options options,
 const unsigned char* image,unsigned image_size,const std::uint32_t* replay,Result* prefix) {
  if(threadIdx.x||blockIdx.x)return;
  prepare_prefix(world,arena,options,image,image_size,replay,prefix);
}
__global__ void bind_prefix_kernel(World* world,unsigned char* arena) {if(!threadIdx.x&&!blockIdx.x)world->bind(arena);}

// One block copies one complete relocatable checkpoint. Game containers are
// inline; only guest memory and its owner pointer require rebinding.
__global__ void clone_kernel(World* worlds,unsigned char* arenas,const World* source,
 const unsigned char* source_arena,unsigned arena_bytes,unsigned count,
 const std::uint32_t* selection) {
  const unsigned index=blockIdx.x;if(index>=count)return;
  const unsigned parent=selection?selection[index]:0;
  auto* out=reinterpret_cast<unsigned char*>(worlds+index);
  const auto* in=reinterpret_cast<const unsigned char*>(source+parent);
  for(std::size_t i=threadIdx.x;i<sizeof(World);i+=blockDim.x)out[i]=in[i];
  for(std::size_t i=threadIdx.x;i<arena_bytes;i+=blockDim.x)
    arenas[std::size_t(index)*arena_bytes+i]=source_arena[std::size_t(parent)*arena_bytes+i];
  __syncthreads();
  if(!threadIdx.x)worlds[index].bind(arenas+std::size_t(index)*arena_bytes);
}

unsigned arena_storage_lanes(unsigned count,unsigned stride){return stride==32?(count+31)&~31U:count;}
GAME_HD std::size_t arena_lane_offset(unsigned lane,unsigned bytes,unsigned stride) {
  return stride==1?std::size_t(lane)*bytes:std::size_t(lane/32)*bytes*32+(lane%32)*4;
}
__global__ void clone_worlds_kernel(World* worlds,unsigned char* arenas,const World* source,
 unsigned bytes,unsigned count,const std::uint32_t* selection,unsigned stride) {
  const unsigned lane=blockIdx.x;if(lane>=count)return;
  const auto parent=selection?selection[lane]:0;
  auto* out=reinterpret_cast<unsigned char*>(worlds+lane);
  const auto* in=reinterpret_cast<const unsigned char*>(source+parent);
  for(std::size_t i=threadIdx.x;i<sizeof(World);i+=blockDim.x)out[i]=in[i];
  __syncthreads();
  if(!threadIdx.x){worlds[lane].guest.word_stride=stride;worlds[lane].bind(arenas+arena_lane_offset(lane,bytes,stride));}
}
// Each warp transports one logical word across32worlds. Parent beams remain
// dense; packed future cohorts can therefore share parent reads and stores.
__global__ void clone_arena_words_kernel(unsigned char* out,const unsigned char* in,
 unsigned bytes,unsigned count,const std::uint32_t* selection,unsigned source_stride,unsigned destination_stride) {
  const unsigned lane=blockIdx.x*32+(threadIdx.x&31);if(lane>=count)return;
  const auto parent=selection?selection[lane]:0;
  const auto* src=in+arena_lane_offset(parent,bytes,source_stride);
  auto* dst=out+arena_lane_offset(lane,bytes,destination_stride);
  for(std::size_t word=threadIdx.x/32;word<bytes/4;word+=blockDim.x/32)
    *reinterpret_cast<std::uint32_t*>(dst+word*4*destination_stride)=
      *reinterpret_cast<const std::uint32_t*>(src+word*4*source_stride);
}
void clone_layout(World* worlds,unsigned char* arenas,const World* source,
 const unsigned char* source_arenas,unsigned bytes,unsigned count,const std::uint32_t* selection,
 unsigned source_stride,unsigned destination_stride) {
  if(source_stride==1&&destination_stride==1)
    clone_kernel<<<count,256>>>(worlds,arenas,source,source_arenas,bytes,count,selection);
  else {
    clone_worlds_kernel<<<count,256>>>(worlds,arenas,source,bytes,count,selection,destination_stride);
    clone_arena_words_kernel<<<(count+31)/32,256>>>(arenas,source_arenas,bytes,count,selection,source_stride,destination_stride);
  }
}

__global__ void future_kernel(World* worlds,Options options,const std::uint32_t* replay,
 unsigned replay_size,const Candidate* candidates,unsigned count,const Result* prefix,
 Result* results,std::uint32_t* heaps) {
  const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;
  if(options.isolated_warps&&(thread&31))return;
  const unsigned lane=options.isolated_warps?thread/32:thread;if(lane>=count)return;
  auto& world=worlds[lane];auto result=*prefix;cuda_game::StepResult step{};
  GuestCache cache(world,options.cache_guest);
  if(!result.failed)for(unsigned offset=0;offset<options.horizon&&!world.failed();++offset) {
    const auto tick=options.prefix+offset;
    const auto base=tick<replay_size?replay[tick]:0;
    step=world.step(cuda_game::replay_action(input_at(candidates[lane],offset,base),tick));
    if(options.trace_every_tick)fold(result,world.frame(step));
    ++result.simulated_ticks;
    if(step.terminated||step.truncated)break;
  }
  cache.finish();
  if(!options.trace_every_tick&&result.simulated_ticks)fold(result,world.frame(step));
  const auto& sim=world.simulator;
  result.clears=sim.qualifying_clear_count_;
  result.natural=step.terminated&&!step.truncated;
  result.canonical_level=step.diagnostics.recorded_final_level;
  result.canonical_score=step.diagnostics.recorded_final_score;
  result.failed|=result.natural&&!step.diagnostics.terminal_metadata_recorded;
  result.failed|=world.failed();result.guest_error=world.guest.error;result.fault_pc=world.guest.fault_pc;
  rank_keys(result,std::uint64_t(options.prefix)+options.horizon,options.target_score);
  results[lane]=result;heaps[lane]=world.guest.heap;
}

__global__ void rank_kernel(const Result* results,unsigned count,std::uint32_t* ranks) {
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
  unsigned rank=0;
  for(unsigned other=0;other<count;++other)
    rank+=better(results[other],other,results[lane],lane);
  ranks[rank]=lane;
}
unsigned padded_rank_count(unsigned count){unsigned padded=1;while(padded<count)padded<<=1;return padded;}
__global__ void rank_indices_kernel(std::uint32_t* ranks,unsigned count,unsigned padded) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<padded)ranks[i]=i<count?i:UINT32_MAX;
}
__device__ bool rank_before(const Result* results,unsigned a,unsigned b) {
  if(a==UINT32_MAX||b==UINT32_MAX)return a<b;
  return better(results[a],a,results[b],b);
}
__global__ void rank_merge_kernel(const Result* results,std::uint32_t* ranks,
 unsigned padded,unsigned width,unsigned distance) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded)return;
  const auto other=i^distance;if(other<=i)return;
  const auto a=ranks[i],b=ranks[other];const bool forward=(i&width)==0;
  if(forward?rank_before(results,b,a):rank_before(results,a,b)){ranks[i]=b;ranks[other]=a;}
}
void rank_results(const Result* results,unsigned count,std::uint32_t* ranks) {
  if(count<=256){rank_kernel<<<(count+127)/128,128>>>(results,count,ranks);return;}
  const auto padded=padded_rank_count(count);
  rank_indices_kernel<<<(padded+127)/128,128>>>(ranks,count,padded);
  for(unsigned width=2;width<=padded;width<<=1)
    for(unsigned distance=width/2;distance;distance>>=1)
      rank_merge_kernel<<<(padded+127)/128,128>>>(results,ranks,padded,width,distance);
}

__global__ void verify_compact_kernel(const World* selected,const unsigned char* selected_arenas,
 const World* worlds,const unsigned char* arenas,const std::uint32_t* ranks,
 unsigned count,unsigned arena_bytes,unsigned* errors,unsigned source_stride=1,unsigned selected_stride=1) {
  const auto lane=blockIdx.x;if(lane>=count)return;
  const auto source=ranks[lane];
  const auto* a_bytes=selected_arenas+arena_lane_offset(lane,arena_bytes,selected_stride);
  const auto* b_bytes=arenas+arena_lane_offset(source,arena_bytes,source_stride);
  for(std::size_t i=threadIdx.x;i<arena_bytes;i+=blockDim.x) {
    const auto a_offset=(i&~std::size_t(3))*selected_stride+(i&3);
    const auto b_offset=(i&~std::size_t(3))*source_stride+(i&3);
    if(a_bytes[a_offset]!=b_bytes[b_offset])atomicAdd(errors,1U);
  }
  if(!threadIdx.x) {
    const auto& a=selected[lane];const auto& b=worlds[source];
    if(a.guest.memory!=a_bytes ||b.guest.memory!=b_bytes||a.guest.word_stride!=selected_stride||b.guest.word_stride!=source_stride||
       a.guest.heap!=b.guest.heap || a.simulator.tick_!=b.simulator.tick_ ||
       a.simulator.score_!=b.simulator.score_ || a.state_hash()!=b.state_hash())
      atomicAdd(errors,1U);
  }
}

// Execute real physics through each rebound selected pointer, and through its
// independent source. This mutates neither stored search results nor ranks.
// Running one more identical action detects a stale PhysicsWorld owner binding
// and accidentally shared heaps, unlike comparing public snapshot scalars.
__global__ void continue_compact_kernel(World* selected,World* worlds,
 const std::uint32_t* ranks,unsigned count,unsigned* errors,bool cache_guest) {
  const auto lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
  auto& a=selected[lane];auto& b=worlds[ranks[lane]];
  GuestCache cache_a(a,cache_guest),cache_b(b,cache_guest);
  const auto step_a=a.step({}),step_b=b.step({});
  cache_a.finish();cache_b.finish();
  Result ra{},rb{};fold(ra,a.frame(step_a));fold(rb,b.frame(step_b));
  if(a.failed()||b.failed()||ra.trace_hash!=rb.trace_hash||a.guest.heap!=b.guest.heap||
     a.state_hash()!=b.state_hash())atomicAdd(errors,1U);
}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Memory {
  std::vector<void*> allocations;
  ~Memory(){for(auto* p:allocations)cudaFree(p);}
  template<class T>T* allocate(std::size_t count) {
    T* p{};check(cudaMalloc(&p,count*sizeof(T)));allocations.push_back(p);return p;
  }
};
struct Event {
  cudaEvent_t value{};Event(){check(cudaEventCreate(&value));}~Event(){cudaEventDestroy(value);}
  void record(){check(cudaEventRecord(value));}
};
double elapsed(const Event& a,const Event& b){float value{};check(cudaEventElapsedTime(&value,a.value,b.value));return value;}
}

void run(const Options& options,const unsigned char* image,std::size_t image_bytes,
 const std::uint32_t* replay,std::uint32_t replay_size,const Candidate* candidates,
 std::uint32_t count,Result* results,std::uint32_t* ranks,Timings* timings) {
  if(!image||!image_bytes||image_bytes>0x20000||!replay||!replay_size||
     !candidates||!results||!ranks||!timings||!count||count>16384||!options.horizon||
     options.horizon>100000||options.prefix>replay_size||options.arena_bytes<0x40000||
     std::uint64_t(options.prefix)+options.horizon>UINT32_MAX||
     !options.block_threads||options.block_threads>256||
     (options.isolated_warps&&options.block_threads>32&&options.block_threads%32)||
     options.stack_bytes<(16U<<10)||options.stack_bytes>(128U<<10)||
     (options.arena_word_stride!=1&&options.arena_word_stride!=32)||
     (options.arena_word_stride==32&&(options.arena_bytes&3))||
     options.arena_bytes>(512U<<20)||!options.keep||options.keep>count)
    throw std::invalid_argument("invalid bounded search arguments");
  for(unsigned i=0;i<count;++i)if(!valid(candidates[i],options.horizon))
    throw std::invalid_argument("invalid candidate");
  cuda_exact::verify_image(image,image_bytes);
  const auto begin=std::chrono::steady_clock::now();
  check(cudaDeviceSetLimit(cudaLimitStackSize,options.stack_bytes));
  Memory memory;
  auto* image_gpu=memory.allocate<unsigned char>(image_bytes);
  auto* replay_gpu=memory.allocate<std::uint32_t>(replay_size);
  auto* specs_gpu=memory.allocate<Candidate>(count);
  auto* prefix_world=memory.allocate<World>(1);
  check(cudaMemset(prefix_world,0,sizeof(World)));
  auto* prefix_arena=memory.allocate<unsigned char>(options.arena_bytes);
  auto* prefix_result=memory.allocate<Result>(1);
  auto* worlds=memory.allocate<World>(count);
  auto* arenas=memory.allocate<unsigned char>(std::size_t(arena_storage_lanes(count,options.arena_word_stride))*options.arena_bytes);
  auto* results_gpu=memory.allocate<Result>(count);
  const auto rank_count=padded_rank_count(count);
  auto* ranks_gpu=memory.allocate<std::uint32_t>(rank_count);
  auto* heaps_gpu=memory.allocate<std::uint32_t>(count);
  auto* compact_worlds=memory.allocate<World>(options.keep);
  auto* compact_arenas=memory.allocate<unsigned char>(std::size_t(options.keep)*options.arena_bytes);
  auto* errors=memory.allocate<unsigned>(1);check(cudaMemset(errors,0,sizeof(unsigned)));
  check(cudaMemcpy(image_gpu,image,image_bytes,cudaMemcpyHostToDevice));
  check(cudaMemcpy(replay_gpu,replay,std::size_t(replay_size)*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(specs_gpu,candidates,std::size_t(count)*sizeof(Candidate),cudaMemcpyHostToDevice));
  double host_prefix_ms=0;
  if(options.host_prefix) {
    const auto prefix_begin=std::chrono::steady_clock::now();
    auto host_world=std::make_unique<World>();std::vector<unsigned char> host_arena(options.arena_bytes);
    // Unused inline container bytes are transported with snapshots too.
    // Initialize them so raw device copies remain initcheck-clean.
    std::memset(host_world.get(),0,sizeof(World));
    Result host_result;
    struct Environment {
      std::fenv_t prior;
      Environment(){
        if(std::fegetenv(&prior)||std::fesetenv(FE_DFL_ENV))
          throw std::runtime_error("cannot install host floating point environment");
#if defined(__i386__) || defined(__x86_64__)
        const std::uint16_t control=0x027f;
        __asm__ __volatile__("fldcw %0"::"m"(control));
#endif
      }
      ~Environment(){std::fesetenv(&prior);}
    } environment;
    prepare_prefix(host_world.get(),host_arena.data(),options,image,image_bytes,replay,&host_result);
    check(cudaMemcpy(prefix_world,host_world.get(),sizeof(World),cudaMemcpyHostToDevice));
    check(cudaMemcpy(prefix_arena,host_arena.data(),options.arena_bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(prefix_result,&host_result,sizeof(Result),cudaMemcpyHostToDevice));
    host_prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prefix_begin).count();
  }
  Event a,b,c,d,e,f;a.record();
  if(options.host_prefix)bind_prefix_kernel<<<1,1>>>(prefix_world,prefix_arena);
  else prefix_kernel<<<1,1>>>(prefix_world,prefix_arena,options,image_gpu,image_bytes,replay_gpu,prefix_result);
  b.record();
  clone_layout(worlds,arenas,prefix_world,prefix_arena,options.arena_bytes,count,nullptr,1,options.arena_word_stride);c.record();
  const auto future_threads=count*(options.isolated_warps?32U:1U);
  const auto block=options.isolated_warps?std::max(32U,options.block_threads):options.block_threads;
  future_kernel<<<(future_threads+block-1)/block,block>>>(worlds,options,replay_gpu,replay_size,specs_gpu,count,prefix_result,results_gpu,heaps_gpu);d.record();
  rank_results(results_gpu,count,ranks_gpu);
  clone_layout(compact_worlds,compact_arenas,worlds,arenas,options.arena_bytes,options.keep,ranks_gpu,options.arena_word_stride,1);
  e.record();
  if(options.validate_compaction) {
    verify_compact_kernel<<<options.keep,256>>>(compact_worlds,compact_arenas,worlds,arenas,ranks_gpu,options.keep,options.arena_bytes,errors,options.arena_word_stride,1);
    continue_compact_kernel<<<(options.keep+31)/32,32>>>(compact_worlds,worlds,ranks_gpu,options.keep,errors,options.cache_guest);
    verify_compact_kernel<<<options.keep,256>>>(compact_worlds,compact_arenas,worlds,arenas,ranks_gpu,options.keep,options.arena_bytes,errors,options.arena_word_stride,1);
  }
  check(cudaGetLastError());f.record();check(cudaEventSynchronize(f.value));
  check(cudaMemcpy(results,results_gpu,std::size_t(count)*sizeof(Result),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(ranks,ranks_gpu,std::size_t(count)*4,cudaMemcpyDeviceToHost));
  std::vector<std::uint32_t> heaps(count);unsigned compact_errors{};
  check(cudaMemcpy(heaps.data(),heaps_gpu,std::size_t(count)*4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&compact_errors,errors,sizeof(compact_errors),cudaMemcpyDeviceToHost));
  *timings={};timings->host_prefix_ms=host_prefix_ms;timings->prefix_ms=elapsed(a,b);timings->branch_ms=elapsed(b,c);
  timings->rollout_ms=elapsed(c,d);timings->rank_compact_ms=elapsed(d,e);
  timings->validation_ms=elapsed(e,f);
  timings->total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  timings->allocation_bytes=(std::uint64_t(count)+options.keep+1)*sizeof(World)+
    (std::uint64_t(arena_storage_lanes(count,options.arena_word_stride))+options.keep+1)*options.arena_bytes+
    image_bytes+std::uint64_t(replay_size)*4+std::uint64_t(count)*(sizeof(Candidate)+sizeof(Result)+4)+std::uint64_t(rank_count)*4+sizeof(Result)+4;
  for(auto heap:heaps)if(heap>timings->peak_guest_bytes)timings->peak_guest_bytes=heap;
  timings->compaction_errors=compact_errors;
}
#include "search_session.inc"
} // namespace irisu::cuda_search
