#include "../../clone/cuda/game.cuh"
#include "irisu/simulator.hpp"
#include "irisu/floating_point.hpp"

#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace game = irisu::cuda_game;

// Both sides use the same CPU physics implementation; this differential isolates
// the source-derived game rules/container/event transformation from the physics
// translation. Full GPU/exact replay parity is a separate integration test.
struct PhysicsAdapter {
  irisu::PhysicsWorld world;
  std::vector<irisu::Body> native;

  explicit PhysicsAdapter(game::MechanicsConfig config) : world(convert_config(config)) {}
  static irisu::MechanicsConfig convert_config(game::MechanicsConfig config) {
    static_assert(sizeof(config) == sizeof(irisu::MechanicsConfig));
    irisu::MechanicsConfig result;
    std::memcpy(&result, &config, sizeof(result));
    return result;
  }
  void reset() { world.reset(); }
  void initialize_mass(game::Body& body) {
    static_assert(sizeof(body) == sizeof(irisu::Body));
    irisu::Body value;
    std::memcpy(&value, &body, sizeof(value));
    world.initialize_mass(value);
    std::memcpy(&body, &value, sizeof(body));
  }
  void load(const game::vector<game::Body>& bodies) {
    native.resize(bodies.size());
    for (std::size_t i = 0; i < native.size(); ++i)
      std::memcpy(&native[i], &bodies[i], sizeof(native[i]));
  }
  void save(game::vector<game::Body>& bodies) {
    for (std::size_t i = 0; i < native.size(); ++i)
      std::memcpy(&bodies[i], &native[i], sizeof(native[i]));
  }
  void synchronize(game::vector<game::Body>& bodies) { load(bodies); world.synchronize(native); save(bodies); }
  void queue_destroy(game::BodyId id) { world.queue_destroy(id); }
  game::vector<game::Contact> step(game::vector<game::Body>& bodies) {
    load(bodies);
    const auto contacts = world.step(native);
    save(bodies);
    game::vector<game::Contact> result;
    static_assert(sizeof(game::Contact) == sizeof(irisu::Contact));
    for (const auto& contact : contacts) {
      game::Contact converted;
      std::memcpy(&converted, &contact, sizeof(converted));
      result.push_back(converted);
    }
    return result;
  }
  game::Vec2 raw_velocity(game::BodyId id) {
    const auto velocity = world.raw_velocity(id);
    return {velocity.x, velocity.y};
  }
};

using Port = game::Simulator<PhysicsAdapter>;

void equal(bool predicate, const char* detail, std::uint64_t tick) {
  if (!predicate) {
    std::fprintf(stderr,"game rule mismatch at tick %llu: %s\n",static_cast<unsigned long long>(tick),detail);
    throw std::runtime_error(detail);
  }
}

void compare(const irisu::Snapshot& expected, const Port& actual) {
  const auto tick = expected.tick;
  equal(!actual.failed(), "port failure", tick);
  equal(expected.tick == actual.tick_ && expected.scene_frame == actual.scene_frame_, "frame", tick);
  equal(expected.score == actual.score_ && expected.gauge == actual.gauge_ && expected.level == actual.level_, "score/gauge/level", tick);
  equal(expected.next_body_id == actual.next_body_id_ && expected.next_chain_id == actual.next_chain_id_, "ids", tick);
  equal(expected.actor_pool_cursor == actual.actor_pool_cursor_, "actor cursor", tick);
  equal(expected.spawn_count == actual.spawn_count_ && expected.next_event_sequence == actual.next_event_sequence_, "spawn/events", tick);
  equal(expected.qualifying_clear_count == actual.qualifying_clear_count_ && expected.next_special_clear_count == actual.next_special_clear_count_, "clear scheduler", tick);
  equal(expected.highest_chain == actual.highest_chain_ && expected.level_shape_cutoff == actual.level_shape_cutoff_, "chain/shape", tick);
  equal(expected.terminated == actual.terminated_ && expected.truncated == actual.truncated_, "termination", tick);
  equal(expected.finish_call_count == actual.finish_call_count_ && expected.terminal_metadata_recorded == actual.terminal_metadata_recorded_, "finish metadata", tick);
  equal(expected.recorded_final_score == actual.recorded_final_score_ && expected.latest_final_score == actual.latest_final_score_, "final score", tick);
  equal(expected.recorded_final_highest_chain == actual.recorded_final_highest_chain_ && expected.latest_final_highest_chain == actual.latest_final_highest_chain_, "final chain", tick);
  equal(expected.recorded_final_level == actual.recorded_final_level_ && expected.latest_final_level == actual.latest_final_level_, "final level", tick);
  equal(expected.recorded_final_clears == actual.recorded_final_clears_ && expected.latest_final_clears == actual.latest_final_clears_, "final clears", tick);
  equal(expected.previous_left_level == actual.previous_left_level_ && expected.previous_right_level == actual.previous_right_level_, "held history", tick);
  equal(expected.rng_index == actual.rng_.index(), "rng index", tick);
  for (unsigned i = 0; i < 624; ++i)
    equal(expected.rng_state[i] == actual.rng_.state()[i], "rng state", tick);
  for (unsigned i = 0; i < 200; ++i)
    equal(expected.actor_pool_colors[i] == actual.actor_pool_colors_[i], "retained actor color", tick);
  equal(expected.bodies.size() == actual.bodies_.size(), "body count", tick);
  for (std::size_t i = 0; i < expected.bodies.size(); ++i) {
    if (game::body_hash(expected.bodies[i]) != game::body_hash(actual.bodies_[i])) {
      std::fprintf(stderr,"body index=%zu id=%u expected hash=%llx actual hash=%llx\n",i,expected.bodies[i].id,
                   static_cast<unsigned long long>(game::body_hash(expected.bodies[i])),
                   static_cast<unsigned long long>(game::body_hash(actual.bodies_[i])));
      equal(false, "body fields", tick);
    }
  }
  equal(expected.groups.size() == actual.groups_.size(), "group count", tick);
  for (std::size_t i = 0; i < expected.groups.size(); ++i) {
    const auto& a = expected.groups[i];
    const auto& b = actual.groups_[i];
    equal(a.id == b.id && a.num == b.num && a.chain == b.chain && a.secondary_count == b.secondary_count, "group fields", tick);
  }
  equal(expected.active_contact_pairs.size() == actual.active_contact_pairs_.size(), "active contact count", tick);
  for (std::size_t i = 0; i < expected.active_contact_pairs.size(); ++i)
    equal(expected.active_contact_pairs[i] == actual.active_contact_pairs_[i], "active contacts", tick);
}

game::Events event_digest(const std::vector<irisu::Event>& original) {
  game::Events result;
  if (!original.empty()) result.sequence_base=original.front().sequence;
  for (const auto& event : original)
    result.push_back({event.tick,static_cast<game::EventKind>(event.kind),event.a,event.b,
                      event.value,event.detail.c_str(),event.sequence});
  return result;
}

int main(int argc, char** argv) {
  try {
    unsigned ticks = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 4096;
    std::ifstream file("irisu-high-score-4393101.rpy",std::ios::binary);
    std::vector<std::uint32_t> replay((388928-52)/4);
    file.seekg(52);
    file.read(reinterpret_cast<char*>(replay.data()),static_cast<std::streamsize>(replay.size()*4));
    if(!file) throw std::runtime_error("cannot read published replay");
    std::uint64_t compared=0,events=0;
    const std::uint32_t seeds[]={1298144938U,3939967453U,1196846281U,4000000151U};
    for (const auto seed : seeds) {
      irisu::Simulator reference;
      auto port = std::make_unique<Port>();
      equal(port->config_hash() == reference.config_hash(), "config hash", 0);
      port->reset(0); // original constructor's initial scene
      reference.reset(seed);
      port->reset(seed);
      compare(reference.clone_state(),*port);
      for(unsigned index=0;index<ticks && index<replay.size();++index) {
        const auto word=replay[index];
        irisu::Action input;
        input.kind=static_cast<irisu::ActionKind>(word&3U);
        input.cursor_x=(word>>2)&1023U;
        input.cursor_y=(word>>12)&511U;
        input.suppress_fresh_edges=index<2;
        game::Action converted;
        std::memcpy(&converted,&input,sizeof(converted));
        const irisu::ScopedFloatingPointEnvironment fp;
        const auto expected=reference.step(input);
        const auto actual=port->step(converted);
        compare(reference.clone_state(),*port);
        const auto digest=event_digest(expected.events);
        equal(expected.reward==actual.reward && digest.count==actual.events.count && digest.hash==actual.events.hash,"reward/event digest",index+1);
        events+=digest.count;
        ++compared;
        if(expected.terminated||expected.truncated)break;
      }
    }
    std::printf("{\"schema\":\"irisu-cuda-game-rules-v1\",\"status\":\"passed\",\"scope\":\"source-derived-rules-host-reference-same-portable-physics\",\"gpu_physics_parity\":false,\"seeds\":4,\"steps\":%llu,\"events\":%llu}\n",static_cast<unsigned long long>(compared),static_cast<unsigned long long>(events));
    return 0;
  } catch(const std::exception& error) {
    std::fprintf(stderr,"game validation failed: %s\n",error.what()); return 1;
  }
}
