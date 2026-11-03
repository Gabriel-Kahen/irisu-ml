#include "search.hpp"
#include "world.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cfenv>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace irisu::cuda_search {
namespace {
using cuda_game::World;
GAME_HD void prepare_prefix(World* world,unsigned char* arena,Options options,
 const unsigned char* image,unsigned image_size,const std::uint32_t* replay,Result* prefix) {
  ::new(static_cast<void*>(world)) World();
  world->initialize(arena,options.arena_bytes,image,image_size,options.seed);
  cuda_game::StepResult step{};
  for(unsigned tick=0;tick<options.prefix&&!world->failed();++tick) {
    if(step.terminated||step.truncated){world->guest.fail(cuda_guest::aborted);break;}
    step=world->step(cuda_game::replay_action(replay[tick],tick));
  }
  *prefix=Result{};fold(*prefix,world->frame(step));
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

__global__ void future_kernel(World* worlds,Options options,const std::uint32_t* replay,
 unsigned replay_size,const Candidate* candidates,unsigned count,const Result* prefix,
 Result* results,std::uint32_t* heaps) {
  const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;
  if(options.isolated_warps&&(thread&31))return;
  const unsigned lane=options.isolated_warps?thread/32:thread;if(lane>=count)return;
  auto& world=worlds[lane];auto result=*prefix;cuda_game::StepResult step{};
  if(!result.failed)for(unsigned offset=0;offset<options.horizon&&!world.failed();++offset) {
    const auto tick=options.prefix+offset;
    const auto base=tick<replay_size?replay[tick]:0;
    step=world.step(cuda_game::replay_action(input_at(candidates[lane],offset,base),tick));
    fold(result,world.frame(step));++result.simulated_ticks;
    if(step.terminated||step.truncated)break;
  }
  const auto& sim=world.simulator;
  result.clears=sim.qualifying_clear_count_;
  result.natural=step.terminated&&!step.truncated;
  result.canonical_level=step.diagnostics.latest_final_level;
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

__global__ void verify_compact_kernel(const World* selected,const unsigned char* selected_arenas,
 const World* worlds,const unsigned char* arenas,const std::uint32_t* ranks,
 unsigned count,unsigned arena_bytes,unsigned* errors) {
  const auto lane=blockIdx.x;if(lane>=count)return;
  const auto source=ranks[lane];
  for(std::size_t i=threadIdx.x;i<arena_bytes;i+=blockDim.x)
    if(selected_arenas[std::size_t(lane)*arena_bytes+i]!=arenas[std::size_t(source)*arena_bytes+i])atomicAdd(errors,1U);
  if(!threadIdx.x) {
    const auto& a=selected[lane];const auto& b=worlds[source];
    if(a.guest.memory!=selected_arenas+std::size_t(lane)*arena_bytes ||
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
 const std::uint32_t* ranks,unsigned count,unsigned* errors) {
  const auto lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
  auto& a=selected[lane];auto& b=worlds[ranks[lane]];
  const auto step_a=a.step({}),step_b=b.step({});
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
     !candidates||!results||!ranks||!timings||!count||count>4096||!options.horizon||
     options.horizon>100000||options.prefix>replay_size||options.arena_bytes<0x40000||
     std::uint64_t(options.prefix)+options.horizon>UINT32_MAX||
     options.arena_bytes>(512U<<20)||!options.keep||options.keep>count)
    throw std::invalid_argument("invalid bounded search arguments");
  for(unsigned i=0;i<count;++i)if(!valid(candidates[i],options.horizon))
    throw std::invalid_argument("invalid candidate");
  const auto begin=std::chrono::steady_clock::now();
  check(cudaDeviceSetLimit(cudaLimitStackSize,128*1024));
  Memory memory;
  auto* image_gpu=memory.allocate<unsigned char>(image_bytes);
  auto* replay_gpu=memory.allocate<std::uint32_t>(replay_size);
  auto* specs_gpu=memory.allocate<Candidate>(count);
  auto* prefix_world=memory.allocate<World>(1);
  check(cudaMemset(prefix_world,0,sizeof(World)));
  auto* prefix_arena=memory.allocate<unsigned char>(options.arena_bytes);
  auto* prefix_result=memory.allocate<Result>(1);
  auto* worlds=memory.allocate<World>(count);
  auto* arenas=memory.allocate<unsigned char>(std::size_t(count)*options.arena_bytes);
  auto* results_gpu=memory.allocate<Result>(count);
  auto* ranks_gpu=memory.allocate<std::uint32_t>(count);
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
  clone_kernel<<<count,256>>>(worlds,arenas,prefix_world,prefix_arena,options.arena_bytes,count,nullptr);c.record();
  const auto future_threads=count*(options.isolated_warps?32U:1U);
  future_kernel<<<(future_threads+127)/128,128>>>(worlds,options,replay_gpu,replay_size,specs_gpu,count,prefix_result,results_gpu,heaps_gpu);d.record();
  rank_kernel<<<(count+127)/128,128>>>(results_gpu,count,ranks_gpu);
  clone_kernel<<<options.keep,256>>>(compact_worlds,compact_arenas,worlds,arenas,options.arena_bytes,options.keep,ranks_gpu);
  e.record();
  if(options.validate_compaction) {
    verify_compact_kernel<<<options.keep,256>>>(compact_worlds,compact_arenas,worlds,arenas,ranks_gpu,options.keep,options.arena_bytes,errors);
    continue_compact_kernel<<<(options.keep+31)/32,32>>>(compact_worlds,worlds,ranks_gpu,options.keep,errors);
    verify_compact_kernel<<<options.keep,256>>>(compact_worlds,compact_arenas,worlds,arenas,ranks_gpu,options.keep,options.arena_bytes,errors);
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
  timings->allocation_bytes=(std::uint64_t(count)+options.keep+1)*(sizeof(World)+options.arena_bytes)+
    image_bytes+std::uint64_t(replay_size)*4+std::uint64_t(count)*(sizeof(Candidate)+sizeof(Result)+8)+sizeof(Result)+4;
  for(auto heap:heaps)if(heap>timings->peak_guest_bytes)timings->peak_guest_bytes=heap;
  timings->compaction_errors=compact_errors;
}
} // namespace irisu::cuda_search
