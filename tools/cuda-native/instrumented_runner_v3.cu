// V3 diagnostic: nonempty original physics stages, private thread counters.
// Not a benchmark. Original programs/headers stay unchanged.
#ifdef IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
#undef IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
#endif
#define IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER 0
// Parse originals before redirecting later pipeline call sites. In particular,
// nested calls INSIDE these originals are not separately redirected/timed.
#include "../../clone/cuda_native/world.hpp"
#include "../../clone/cuda_native/world_shapes.hpp"
#include "../../clone/cuda_native/collision.hpp"
#include "../../clone/cuda_native/solver.hpp"
#include "../../clone/cuda_native/broadphase.hpp"
#include "../../clone/cuda_native/world_contacts.hpp"
#include "../../clone/cuda/game.cuh"
#include "../cuda-exact/game_digest.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <array>
#include <vector>

namespace irisu::cuda_native::instrumentation {
enum Phase : unsigned {physics,external_sync,game_step,digest_frame,
  solver_build_phase,solver_pre_phase,solver_velocity_phase,solver_position_phase,solver_post_phase,
  collision_phase,quick_sync_phase,circle_bounds_phase,polygon_bounds_phase,
  integrate_velocity_phase,integrate_position_phase,sleep_phase,phase_count};
inline constexpr unsigned maximum_threads=16384;
__device__ unsigned long long counters[maximum_threads][phase_count*2]{};
__device__ unsigned physics_active[maximum_threads]{};
__device__ __forceinline__ unsigned linear_thread(){
  const unsigned thread=blockIdx.x*blockDim.x+threadIdx.x;
  if(thread>=maximum_threads){asm volatile("trap;");return 0;}return thread;
}
__device__ __forceinline__ unsigned long long begin(){
  asm volatile("":::"memory");return clock64();
}
__device__ __forceinline__ unsigned long long end(){
  asm volatile("":::"memory");return clock64();
}
__device__ __forceinline__ void record(Phase phase,unsigned long long cycles){
  // One packed future thread owns each row. All participating kernels run
  // serially on the default stream. Updates are OUTSIDE this phase's interval;
  // outer Game timing still includes nested private-counter update overhead.
  const unsigned thread=linear_thread();
  counters[thread][2*phase]+=cycles;
  counters[thread][2*phase+1]+=1;
}

NATIVE_HD __forceinline__ unsigned long long stage_begin(){
#ifdef __CUDA_ARCH__
  if(physics_active[linear_thread()])return begin();
#endif
  return 0;
}
NATIVE_HD __forceinline__ void stage_end(Phase phase,unsigned long long start){
#ifdef __CUDA_ARCH__
  if(physics_active[linear_thread()]){const auto finish=end();record(phase,finish-start);}
#else
  (void)phase;(void)start;
#endif
}
}

namespace irisu::cuda_native {
// Match original parameter passing exactly. In particular, scalar counts must
// be BY VALUE: forwarding references ODR-use host constexpr globals in CUDA.
// Empty calls still execute originals so all output/reference effects remain.
#define IRISU_TIMED_STAGE(name,phase,parameters,arguments,skip) \
NATIVE_HD __forceinline__ auto timed_##name parameters -> decltype(name arguments){ \
  if(skip)return name arguments; \
  const auto start=instrumentation::stage_begin(); \
  auto result=name arguments; \
  instrumentation::stage_end(instrumentation::phase,start);return result; \
}
IRISU_TIMED_STAGE(solver_build_constraints,solver_build_phase,
  (const BodyState* bodies,std::uint32_t body_count,const Manifold* manifolds,std::uint32_t manifold_count,
   const SolverContact* contacts,std::uint32_t contact_count,ContactConstraint* output,std::uint32_t capacity,std::uint32_t& count),
  (bodies,body_count,manifolds,manifold_count,contacts,contact_count,output,capacity,count),contact_count==0)
IRISU_TIMED_STAGE(solver_presolve,solver_pre_phase,
  (BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,bool warm_start),
  (bodies,body_count,constraints,count,warm_start),count==0)
IRISU_TIMED_STAGE(solver_velocity,solver_velocity_phase,
  (BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count),
  (bodies,body_count,constraints,count),count==0)
IRISU_TIMED_STAGE(solver_position,solver_position_phase,
  (BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,float beta,bool canonical_rotation=false),
  (bodies,body_count,constraints,count,beta,canonical_rotation),count==0)
IRISU_TIMED_STAGE(solver_postsolve,solver_post_phase,
  (const ContactConstraint* constraints,std::uint32_t count,Manifold* manifolds,std::uint32_t manifold_count),
  (constraints,count,manifolds,manifold_count),count==0)
IRISU_TIMED_STAGE(collide,collision_phase,
  (Manifold& manifold,const ShapeState& first,const ShapeState& second,bool conservative=false),
  (manifold,first,second,conservative),false)
IRISU_TIMED_STAGE(shape_quick_sync,quick_sync_phase,
  (ShapeState& shape,Vec2 position,const Mat22& rotation),(shape,position,rotation),false)
IRISU_TIMED_STAGE(circle_swept_bounds,circle_bounds_phase,
  (AABB& bounds,const ShapeState& shape,Vec2 previous,const Mat22& previous_rotation),
  (bounds,shape,previous,previous_rotation),false)
IRISU_TIMED_STAGE(polygon_swept_bounds,polygon_bounds_phase,
  (AABB& bounds,const ShapeState& shape,Vec2 previous,const Mat22& previous_rotation,Vec2 position,const Mat22& rotation),
  (bounds,shape,previous,previous_rotation,position,rotation),false)
IRISU_TIMED_STAGE(integrate_velocity,integrate_velocity_phase,
  (BodyState& body,float dt,Vec2 gravity),(body,dt,gravity),body.inv_mass==0)
IRISU_TIMED_STAGE(integrate_position,integrate_position_phase,
  (BodyState& body,float dt,bool canonical_rotation=false),(body,dt,canonical_rotation),body.inv_mass==0)
IRISU_TIMED_STAGE(update_sleep,sleep_phase,
  (BodyState* bodies,const std::uint32_t* order,unsigned count,float dt),(bodies,order,count,dt),count==0)
#undef IRISU_TIMED_STAGE
}

#define solver_build_constraints timed_solver_build_constraints
#define solver_presolve timed_solver_presolve
#define solver_velocity timed_solver_velocity
#define solver_position timed_solver_position
#define solver_postsolve timed_solver_postsolve
#define collide timed_collide
#define shape_quick_sync timed_shape_quick_sync
#define circle_swept_bounds timed_circle_swept_bounds
#define polygon_swept_bounds timed_polygon_swept_bounds
#define integrate_velocity timed_integrate_velocity
#define integrate_position timed_integrate_position
#define update_sleep timed_update_sleep
#include "../../clone/cuda_native/game_adapter.cuh"
#undef solver_build_constraints
#undef solver_presolve
#undef solver_velocity
#undef solver_position
#undef solver_postsolve
#undef collide
#undef shape_quick_sync
#undef circle_swept_bounds
#undef polygon_swept_bounds
#undef integrate_velocity
#undef integrate_position
#undef update_sleep

namespace irisu::cuda_game {
using OriginalPhysics=NativePhysicsWorld;
using OriginalSimulator=Simulator<OriginalPhysics>;
struct TimedPhysics:OriginalPhysics {
  GAME_HD explicit TimedPhysics(MechanicsConfig config={}):OriginalPhysics(config){}
  GAME_HD __forceinline__ const vector<Contact>& step(vector<Body>& bodies){
#ifdef __CUDA_ARCH__
    namespace timer=irisu::cuda_native::instrumentation;
    const auto thread=timer::linear_thread();timer::physics_active[thread]=1;
    const auto start=timer::begin();const auto& result=OriginalPhysics::step(bodies);
    const auto finish=timer::end();timer::physics_active[thread]=0;
    timer::record(timer::physics,finish-start);return result;
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
  bool skip_selected=false;
  for(int i=1;i<argc;++i){
    if(!std::strcmp(argv[i],"--cpu")||!std::strcmp(argv[i],"--isolated-warps")){
      std::fprintf(stderr,"instrumented-v3: only packed GPU threads are supported\n");return 1;
    }
    if(!std::strcmp(argv[i],"--skip-compaction-check"))skip_selected=true;
  }
  if(!skip_selected){std::fprintf(stderr,"instrumented-v3: require --skip-compaction-check\n");return 1;}
  std::vector<unsigned long long> rows(timer::maximum_threads*timer::phase_count*2,0);
  std::array<unsigned long long,timer::phase_count*2> values{};
  const auto row_bytes=rows.size()*sizeof(rows[0]);
  auto status=cudaMemcpyToSymbol(timer::counters,rows.data(),row_bytes);
  if(status!=cudaSuccess){std::fprintf(stderr,"instrumented-runner counter setup: %s\n",cudaGetErrorString(status));return 1;}
  status=cudaMemcpyToSymbol(timer::physics_active,rows.data(),timer::maximum_threads*sizeof(unsigned));
  if(status!=cudaSuccess){std::fprintf(stderr,"instrumented-v3 scope setup: %s\n",cudaGetErrorString(status));return 1;}
  const int original_returncode=instrumentation_prior_entry(argc,argv);
  status=cudaMemcpyFromSymbol(rows.data(),timer::counters,row_bytes);
  if(status!=cudaSuccess){std::fprintf(stderr,"instrumented-runner counter readback: %s\n",cudaGetErrorString(status));return original_returncode?original_returncode:1;}
  unsigned active_rows=0;
  for(unsigned thread=0;thread<timer::maximum_threads;++thread){bool active=false;
    for(unsigned field=0;field<timer::phase_count*2;++field){
      const auto value=rows[thread*timer::phase_count*2+field];values[field]+=value;active|=value!=0;
    }
    active_rows+=active;
  }
  const auto total=values[2*timer::game_step],physical=values[2*timer::physics],sync=values[2*timer::external_sync];
  const double physics_fraction=total?double(physical)/double(total):0;
  const double sync_fraction=total?double(sync)/double(total):0;
  const double digest_to_step=total?double(values[2*timer::digest_frame])/double(total):0;
  std::fprintf(stderr,"{\"identity\":\"diagnostic-typed-search-cycle-attribution-v3\",\"benchmark_eligible\":false,\"promotion_eligible\":false,\"original_returncode\":%d,\"cuda_context_preinitialized\":true,\"counter_mode\":\"private-linear-CUDA-thread-no-atomics\",\"counter_rows\":16384,\"counter_storage_bytes\":4194304,\"physics_scope_flag_bytes\":65536,\"interleaved_solver\":false,\"empty_solver_calls_timed\":false,\"static_integration_calls_timed\":false,\"world_layout_unchanged\":true,\"normal_stdout_packet_retained\":true,\"scope\":\"Aggregated per-thread elapsed SM clocks, not kernel wall or device utilization; counters include all GPU prefix/warmup/continuation calls. Use host-prefix,warmups0,repeats1,skip-compaction-check for future attribution. Digest interval covers make_frame, excluding observation-summary construction and wire.fold. Private counter updates remain in outer Game/residual; no same-address atomic contention. Only packed GPU execution with selected checks disabled is supported. Stage counters accumulate only inside Base physics step; preloaded-original nested calls are included in their enclosing stage, not separately timed. Empty solver and static integration calls execute original code without timers. Stage records and scope checks perturb outer physics/residual.\",\"phases\":{",
    original_returncode);
  const char* names[]={"physics_step_including_internal_sync","external_game_synchronize","whole_game_step","digest_make_frame",
    "solver_build_nonempty","solver_presolve_nonempty","solver_velocity_nonempty","solver_position_nonempty","solver_postsolve_nonempty",
    "collide","shape_quick_sync","circle_swept_bounds","polygon_swept_bounds",
    "integrate_velocity_dynamic","integrate_position_dynamic","update_sleep_nonempty"};
  static_assert(sizeof(names)/sizeof(names[0])==timer::phase_count);
  for(unsigned phase=0;phase<timer::phase_count;++phase)
    std::fprintf(stderr,"%s\"%s\":{\"cycles_sum\":%llu,\"calls\":%llu,\"cycles_per_call\":%.9f}",
      phase?",":"",names[phase],values[2*phase],values[2*phase+1],values[2*phase+1]?double(values[2*phase])/double(values[2*phase+1]):0);
  unsigned long long stage_sum=0;for(unsigned phase=timer::solver_build_phase;phase<timer::phase_count;++phase)stage_sum+=values[2*phase];
  std::fprintf(stderr,"},\"active_counter_rows\":%u,\"physics_fraction_of_game_step_cycles\":%.9f,\"external_sync_fraction_of_game_step_cycles\":%.9f,\"rules_and_instrumentation_residual_fraction\":%.9f,\"digest_cycles_over_game_step_cycles\":%.9f,\"stages\":{",
    active_rows,physics_fraction,sync_fraction,1-physics_fraction-sync_fraction,digest_to_step);
  for(unsigned phase=timer::solver_build_phase;phase<timer::phase_count;++phase)
    std::fprintf(stderr,"%s\"%s\":%.9f",phase==timer::solver_build_phase?"":",",names[phase],physical?double(values[2*phase])/double(physical):0);
  std::fprintf(stderr,"},\"unwrapped_physics_and_instrumentation_residual_fraction\":%.9f}\n",physical?1-double(stage_sum)/double(physical):0);
  return original_returncode;
}
