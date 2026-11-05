// Diagnostic wrapper around the unchanged search runner. Counter collection
// preinitializes CUDA and perturbs register allocation/latency: never benchmark.
#include "../../clone/cuda_native/game_adapter.cuh"
#include "../../clone/cuda/game.cuh"
#include "../cuda-exact/game_digest.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <type_traits>

namespace irisu::cuda_native::instrumentation {
enum Phase : unsigned {physics,external_sync,game_step,digest_frame,phase_count};
__device__ unsigned long long counters[phase_count*2]{};
__device__ __forceinline__ unsigned long long begin(){
  asm volatile("":::"memory");return clock64();
}
__device__ __forceinline__ unsigned long long end(){
  asm volatile("":::"memory");return clock64();
}
__device__ __forceinline__ void record(Phase phase,unsigned long long cycles){
  // Both atomics are OUTSIDE this phase's measured interval. Outer Game timing
  // still includes nested counter overhead; its residual is not pure rule cost.
  atomicAdd(counters+2*phase,cycles);atomicAdd(counters+2*phase+1,1ULL);
}
}

namespace irisu::cuda_game {
using OriginalPhysics=NativePhysicsWorld;
using OriginalSimulator=Simulator<OriginalPhysics>;
struct TimedPhysics:OriginalPhysics {
  GAME_HD explicit TimedPhysics(MechanicsConfig config={}):OriginalPhysics(config){}
  GAME_HD __forceinline__ const vector<Contact>& step(vector<Body>& bodies){
#ifdef __CUDA_ARCH__
    namespace timer=irisu::cuda_native::instrumentation;
    const auto start=timer::begin();const auto& result=OriginalPhysics::step(bodies);
    const auto finish=timer::end();timer::record(timer::physics,finish-start);return result;
#else
    return OriginalPhysics::step(bodies);
#endif
  }
  GAME_HD __forceinline__ void synchronize(vector<Body>& bodies){
#ifdef __CUDA_ARCH__
    namespace timer=irisu::cuda_native::instrumentation;
    const auto start=timer::begin();OriginalPhysics::synchronize(bodies);
    const auto finish=timer::end();timer::record(timer::external_sync,finish-start);
#else
    OriginalPhysics::synchronize(bodies);
#endif
  }
};
template<class Physics> struct TimedSimulator:Simulator<Physics> {
  using Base=Simulator<Physics>;
  GAME_HD explicit TimedSimulator(MechanicsConfig config={}):Base(config){}
  GAME_HD __forceinline__ StepResult step(const Action& action){
#ifdef __CUDA_ARCH__
    namespace timer=irisu::cuda_native::instrumentation;
    const auto start=timer::begin();auto result=Base::step(action);
    const auto finish=timer::end();timer::record(timer::game_step,finish-start);return result;
#else
    return Base::step(action);
#endif
  }
};
static_assert(sizeof(TimedPhysics)==sizeof(OriginalPhysics)&&alignof(TimedPhysics)==alignof(OriginalPhysics));
static_assert(sizeof(TimedSimulator<TimedPhysics>)==sizeof(OriginalSimulator));
static_assert(std::is_trivially_copyable_v<TimedPhysics>&&std::is_trivially_copyable_v<TimedSimulator<TimedPhysics>>);
}

namespace irisu::game_digest {
template<class Range,class Observation,class Step>
__host__ __device__ __forceinline__ Frame timed_make_frame(const Range& bodies,const Observation& observation,
    std::uint64_t event_hash,std::uint64_t event_count,const Step& step){
#ifdef __CUDA_ARCH__
  namespace timer=irisu::cuda_native::instrumentation;
  const auto start=timer::begin();auto result=make_frame(bodies,observation,event_hash,event_count,step);
  const auto finish=timer::end();timer::record(timer::digest_frame,finish-start);return result;
#else
  return make_frame(bodies,observation,event_hash,event_count,step);
#endif
}
}

// Original definitions above are already protected by pragma once. These aliases
// instantiate the same generated rules with data-free wrappers. Keep type aliases
// active through compact snapshot helpers so their Simulator/Physics types agree.
#define NativePhysicsWorld TimedPhysics
#define Simulator TimedSimulator
#define make_frame timed_make_frame
#include "../../clone/cuda_native/game_world.cuh"
#undef make_frame
static_assert(sizeof(irisu::cuda_game::NativeGameWorld)==sizeof(irisu::cuda_game::OriginalSimulator));
#define main instrumentation_prior_entry
#include "search_runner.cu"
#undef main
#undef Simulator
#undef NativePhysicsWorld

int main(int argc,char** argv){
  namespace timer=irisu::cuda_native::instrumentation;
  for(int i=1;i<argc;++i)if(!std::strcmp(argv[i],"--cpu")){
    std::fprintf(stderr,"instrumented-runner: device diagnostic requires GPU execution\n");return 1;
  }
  unsigned long long values[timer::phase_count*2]{};
  auto status=cudaMemcpyToSymbol(timer::counters,values,sizeof(values));
  if(status!=cudaSuccess){std::fprintf(stderr,"instrumented-runner counter setup: %s\n",cudaGetErrorString(status));return 1;}
  const int original_returncode=instrumentation_prior_entry(argc,argv);
  status=cudaMemcpyFromSymbol(values,timer::counters,sizeof(values));
  if(status!=cudaSuccess){std::fprintf(stderr,"instrumented-runner counter readback: %s\n",cudaGetErrorString(status));return original_returncode?original_returncode:1;}
  const auto total=values[2*timer::game_step],physical=values[2*timer::physics],sync=values[2*timer::external_sync];
  const double physics_fraction=total?double(physical)/double(total):0;
  const double sync_fraction=total?double(sync)/double(total):0;
  const double digest_to_step=total?double(values[2*timer::digest_frame])/double(total):0;
  std::fprintf(stderr,"{\"identity\":\"diagnostic-typed-search-cycle-attribution-v1\",\"benchmark_eligible\":false,\"promotion_eligible\":false,\"original_returncode\":%d,\"cuda_context_preinitialized\":true,\"world_layout_unchanged\":true,\"normal_stdout_packet_retained\":true,\"scope\":\"Aggregated per-thread elapsed SM clocks, not kernel wall or device utilization; counters include all GPU prefix/warmup/continuation calls. Use host-prefix,warmups0,repeats1,skip-compaction-check for future attribution. Digest interval covers make_frame, excluding observation-summary construction and wire.fold. Nested atomic overhead remains in outer Game/residual.\",\"phases\":{",
    original_returncode);
  const char* names[]={"physics_step_including_internal_sync","external_game_synchronize","whole_game_step","digest_make_frame"};
  for(unsigned phase=0;phase<timer::phase_count;++phase)
    std::fprintf(stderr,"%s\"%s\":{\"cycles_sum\":%llu,\"calls\":%llu,\"cycles_per_call\":%.9f}",
      phase?",":"",names[phase],values[2*phase],values[2*phase+1],values[2*phase+1]?double(values[2*phase])/double(values[2*phase+1]):0);
  std::fprintf(stderr,"},\"physics_fraction_of_game_step_cycles\":%.9f,\"external_sync_fraction_of_game_step_cycles\":%.9f,\"rules_and_instrumentation_residual_fraction\":%.9f,\"digest_cycles_over_game_step_cycles\":%.9f}\n",
    physics_fraction,sync_fraction,1-physics_fraction-sync_fraction,digest_to_step);
  return original_returncode;
}
