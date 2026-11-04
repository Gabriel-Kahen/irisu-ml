#pragma once
#include "game_adapter.cuh"
#include "../cuda/game.cuh"
#include "../../tools/cuda-exact/game_digest.hpp"
#ifdef __CUDACC__
#define NATIVE_WORLD_CALL __noinline__
#else
#define NATIVE_WORLD_CALL
#endif
#if defined(__CUDACC__) && defined(IRISU_CUDA_NATIVE_INLINE_STEP) && IRISU_CUDA_NATIVE_INLINE_STEP
#define NATIVE_WORLD_STEP_CALL __forceinline__
#else
#define NATIVE_WORLD_STEP_CALL NATIVE_WORLD_CALL
#endif

namespace irisu::cuda_game {
struct NativeGameWorld {
  Simulator<NativePhysicsWorld> simulator{};
  GAME_HD void bind(irisu::cuda_native::ScenePipelineState* scene){simulator.physics_.bind(scene);}
  GAME_HD bool relocate_body_pointers(std::uintptr_t old_base,std::uintptr_t new_base,bool live_only=false){
    const bool valid=relocate_native_body_refs(simulator.actor_refs_scratch_,old_base,new_base,live_only)&&
      simulator.physics_.relocate_body_pointers(old_base,new_base,live_only);
    if(!valid)simulator.failed_=true;return valid;
  }
  GAME_HD bool rebind(irisu::cuda_native::ScenePipelineState* scene,std::uintptr_t old_body_base,bool live_only=false){
    bind(scene);return relocate_body_pointers(old_body_base,reinterpret_cast<std::uintptr_t>(simulator.bodies_.data()),live_only);
  }
  GAME_HD bool failed() const {return simulator.failed();}
  GAME_HD void initialize(irisu::cuda_native::ScenePipelineState* scene,std::uint32_t seed){
    bind(scene);for(auto& byte:simulator.actor_refs_scratch_.storage)byte=0;
    simulator.physics_.initialize_pointer_storage();simulator.reset(0);if(!failed())simulator.reset(seed);
  }
  GAME_HD NATIVE_WORLD_STEP_CALL StepResult step(const Action& action){return simulator.step(action);}
  GAME_HD NATIVE_WORLD_CALL game_digest::Frame frame(const StepResult& result={}) const {
    struct Count {std::uint64_t value{};GAME_HD std::uint64_t size() const{return value;}};
    struct Summary {
      std::uint64_t tick{};std::int64_t score{},gauge{};std::uint32_t level{};
      bool terminated{},truncated{};Count bodies{};
      double field_x{},field_y{},field_width{},field_height{},side_wall_top{},side_wall_bottom{};
      std::int64_t gauge_max{};std::uint32_t active_colors{},current_spawn_interval_ticks{};
      bool left_held{},right_held{};std::uint32_t highest_chain{};std::uint64_t qualifying_clear_count{};
    } observation;
    const auto& s=simulator;const auto& c=s.config_;
    observation.tick=s.tick_;observation.score=s.score_;observation.gauge=s.gauge_;
    observation.level=s.level_;observation.terminated=s.terminated_;observation.truncated=s.truncated_;
    for(const auto& body:s.bodies_)observation.bodies.value+=body.lifecycle!=Lifecycle::Deleted;
    observation.field_x=c.field_x;observation.field_y=c.field_y;observation.field_width=c.field_width;observation.field_height=c.field_height;
    observation.side_wall_top=c.side_wall_top;observation.side_wall_bottom=c.side_wall_bottom;observation.gauge_max=c.gauge_max;
    observation.active_colors=s.current_color_count();observation.current_spawn_interval_ticks=s.current_spawn_interval();
    observation.left_held=s.previous_left_level_;observation.right_held=s.previous_right_level_;
    observation.highest_chain=s.highest_chain_;observation.qualifying_clear_count=s.qualifying_clear_count_;
    return game_digest::make_frame(s.bodies_,observation,result.events.hash,result.events.count,result);
  }
};
static_assert(std::is_trivially_copyable_v<NativeGameWorld>);
GAME_HD inline Action native_replay_action(std::uint32_t word,std::uint64_t index){
  return {static_cast<ActionKind>(word&3U),static_cast<double>((word>>2U)&1023U),
    static_cast<double>((word>>12U)&511U),1,index<2};
}
#undef NATIVE_WORLD_CALL
#undef NATIVE_WORLD_STEP_CALL
}
