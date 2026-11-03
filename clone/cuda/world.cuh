#pragma once

#include "physics_game.cuh"
#include "game.cuh"
#include "../../tools/cuda-exact/game_digest.hpp"

namespace irisu::cuda_game {

// Inline game containers and guest-offset physics pointers make complete worlds
// relocatable by byte copy. The two semantic native pointers must be rebound
// afterwards; scratch Body pointers are always cleared and rebuilt before use.
// Arena bytes include the pinned program image, allocator, physics and stack.
struct World {
  cuda_guest::Guest guest{};
  Simulator<PhysicsWorld> simulator{};

  GAME_HD void bind(unsigned char* arena) {
    guest.memory=arena;
    simulator.physics_.bind(&guest);
  }
  GAME_HD bool failed() const { return guest.error || simulator.failed(); }
  GAME_HD void initialize(unsigned char* arena,std::uint32_t capacity,
                          const unsigned char* image,std::uint32_t image_size,
                          std::uint32_t seed) {
    guest={};
    bind(arena);
    guest.capacity=capacity;guest.heap=0x20000;guest.heap_limit=capacity-0x10000;
    if(capacity<0x40000 || image_size>0x20000 || image_size>capacity) {
      guest.fail(cuda_guest::bad_address);return;
    }
    // Caller constructs a fresh World and owns initialization of arena storage.
    for(std::uint32_t i=0;i<capacity;++i)arena[i]=0;
    for(std::uint32_t i=0;i<image_size;++i)arena[i]=image[i];
    simulator.reset(0); // authoritative native constructor's initial scene
    if(!failed())simulator.reset(seed);
  }
  GAME_HD StepResult step(const Action& action) { return simulator.step(action); }
  // Canonical scalar hashing excludes host/device binding pointers and unused
  // container storage. Optional arena hashing also covers native allocator,
  // physics caches, program globals and guest stack for checkpoint tests.
  GAME_HD std::uint64_t state_hash(bool include_arena=false) const {
    std::uint64_t hash=14695981039346656037ULL;
    const auto& s=simulator;
#define HASH(name) hash=fnv_scalar(hash,s.name)
    HASH(config_hash_);
    HASH(tick_);
    HASH(scene_frame_);
    HASH(next_body_id_);
    HASH(next_chain_id_);
    HASH(actor_pool_cursor_);
    HASH(next_event_sequence_);
    HASH(spawn_count_);
    HASH(score_);
    HASH(gauge_);
    HASH(level_);
    HASH(qualifying_clear_count_);
    HASH(next_special_clear_count_);
    HASH(level_shape_cutoff_);
    HASH(highest_chain_);
    HASH(finish_call_count_);
    HASH(terminal_metadata_recorded_);
    HASH(recorded_final_score_);
    HASH(recorded_final_highest_chain_);
    HASH(recorded_final_level_);
    HASH(recorded_final_clears_);
    HASH(latest_final_score_);
    HASH(latest_final_highest_chain_);
    HASH(latest_final_level_);
    HASH(latest_final_clears_);
    HASH(previous_left_level_);
    HASH(previous_right_level_);
    HASH(terminated_);
    HASH(truncated_);
    HASH(actor_counter_guard_required_);
    HASH(numeric_guard_required_);
    HASH(failed_);

#undef HASH
    for(auto value:s.rng_.state())hash=fnv_scalar(hash,value);
    hash=fnv_scalar(hash,s.rng_.index());
    for(auto value:s.actor_pool_colors_)hash=fnv_scalar(hash,value);
    hash=fnv_scalar(hash,s.bodies_.size());
    for(const auto& body:s.bodies_)hash=body_hash(body,hash);
    hash=fnv_scalar(hash,s.groups_.size());
    for(const auto& group:s.groups_) {
      hash=fnv_scalar(hash,group.id);hash=fnv_scalar(hash,group.chain);
      hash=fnv_scalar(hash,group.secondary_count);hash=fnv_scalar(hash,group.num);
    }
    hash=fnv_scalar(hash,s.active_contact_pairs_.size());
    for(auto value:s.active_contact_pairs_)hash=fnv_scalar(hash,value);
    hash=s.physics_.state_hash(hash);
    for(auto value:guest.r)hash=fnv_scalar(hash,value);
    for(auto value:guest.fp)hash=fnv_scalar(hash,value);
#define HASH(name) hash=fnv_scalar(hash,guest.name)
    HASH(top);HASH(depth);HASH(status);HASH(control);HASH(flags);HASH(pc);HASH(error);
    HASH(fault_pc);HASH(fault_value);HASH(heap);HASH(heap_limit);HASH(capacity);
    HASH(free_head);HASH(allocation_reuses);HASH(instructions);
#undef HASH
    if(include_arena)for(std::uint32_t i=0;i<guest.capacity;++i)hash=fnv_byte(hash,guest.memory[i]);
    return hash;
  }

  // Digest observation metadata without materializing the policy body vector.
  // The shared frame encoder hashes only the live count plus these scalars.
  GAME_HD game_digest::Frame frame(const StepResult& result={}) const {
    struct Count { std::uint64_t value{}; GAME_HD std::uint64_t size() const{return value;} };
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
    observation.field_x=c.field_x;observation.field_y=c.field_y;
    observation.field_width=c.field_width;observation.field_height=c.field_height;
    observation.side_wall_top=c.side_wall_top;observation.side_wall_bottom=c.side_wall_bottom;
    observation.gauge_max=c.gauge_max;observation.active_colors=s.current_color_count();
    observation.current_spawn_interval_ticks=s.current_spawn_interval();
    observation.left_held=s.previous_left_level_;observation.right_held=s.previous_right_level_;
    observation.highest_chain=s.highest_chain_;observation.qualifying_clear_count=s.qualifying_clear_count_;
    return game_digest::make_frame(s.bodies_,observation,result.events.hash,result.events.count,result);
  }
};

static_assert(std::is_trivially_copyable<World>::value, "world checkpoints require representation copying");

GAME_HD inline Action replay_action(std::uint32_t word,std::uint64_t index) {
  return {static_cast<ActionKind>(word&3U),static_cast<double>((word>>2U)&1023U),
          static_cast<double>((word>>12U)&511U),1,index<2};
}
} // namespace irisu::cuda_game
