#include "irisu/metsu/simulator.hpp"
#include "irisu/floating_point.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace irisu::metsu {
struct ReplayTraceAccess {
  static std::uint32_t bits(double value) {
    return std::bit_cast<std::uint32_t>(static_cast<float>(value));
  }

  static void write(const Simulator& sim, std::uint64_t frame,
                    std::uint64_t rng_calls, std::uint32_t input_count) {
    const ScopedFloatingPointEnvironment guard;
    std::cout << "{\"type\":\"frame\",\"phase\":\"" << (frame ? "tick" : "reset")
      << "\",\"frame\":" << frame << ",\"tick\":" << sim.scene_frame_
      << ",\"input_index\":" << frame << ",\"input_count\":" << input_count
      << ",\"score\":" << sim.score_ << ",\"gauge\":" << sim.gauge_
      << ",\"level\":" << sim.level_ << ",\"qualifying_clears\":" << sim.qualifying_clear_count_
      << ",\"highest_chain\":" << sim.highest_chain_ << ",\"rng_calls\":" << rng_calls
      << ",\"rng_index\":" << sim.rng_.index() << ",\"pool_cursor\":" << sim.actor_pool_cursor_
      << ",\"terminated\":" << (sim.terminated_ ? "true" : "false")
      << ",\"truncated\":" << (sim.truncated_ ? "true" : "false")
      << ",\"finish_call_count\":" << sim.finish_call_count_
      << ",\"terminal_metadata_recorded\":" << (sim.terminal_metadata_recorded_ ? "true" : "false")
      << ",\"recorded_final_score\":" << sim.recorded_final_score_
      << ",\"recorded_final_level\":" << sim.recorded_final_level_
      << ",\"recorded_final_chain\":" << sim.recorded_final_highest_chain_
      << ",\"recorded_final_clears\":" << sim.recorded_final_clears_
      << ",\"pool_colors\":[";
    for (std::size_t index = 0; index < sim.actor_pool_colors_.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << sim.actor_pool_colors_[index];
    }
    std::cout << "],\"actors\":[";
    std::vector<const Body*> actors;
    for (const auto& body : sim.bodies_) {
      if (body.lifecycle != Lifecycle::Deleted) actors.push_back(&body);
    }
    std::sort(actors.begin(), actors.end(), [](auto* left, auto* right) {
      return left->actor_slot < right->actor_slot;
    });
    bool first = true;
    for (const auto* body : actors) {
      if (!first) std::cout << ',';
      first = false;
      const auto found = std::find_if(sim.groups_.begin(), sim.groups_.end(),
        [&](const auto& group) { return group.id == body->chain_id; });
      const GroupState* group = found == sim.groups_.end() ? nullptr : &*found;
      std::cout << "{\"slot\":" << body->actor_slot
        << ",\"role\":" << (body->kind == BodyKind::Projectile ? 5 : 4)
        << ",\"ex\":" << unsigned(body->rule_guard_f0)
        << ",\"ex_timer\":" << sim.ex_timers_[body->actor_slot]
        << ",\"color\":" << body->color << ",\"size_slot\":" << body->size_slot
        << ",\"size_bits\":" << bits(body->size)
        << ",\"x_bits\":" << bits(body->position.x) << ",\"y_bits\":" << bits(body->position.y)
        << ",\"angle_bits\":" << bits(body->angle)
        << ",\"vx_bits\":" << bits(body->velocity.x) << ",\"vy_bits\":" << bits(body->velocity.y)
        << ",\"physics_updates\":" << body->physics_update_count
        << ",\"age\":" << body->age_ticks << ",\"remaining_lifetime\":" << body->remaining_lifetime
        << ",\"freshness\":" << unsigned(body->freshness_state)
        << ",\"grouped\":" << unsigned(body->grouped)
        << ",\"physics_owned\":" << unsigned(body->physics_owned)
        << ",\"special\":" << unsigned(body->special)
        << ",\"rot_timer\":" << body->rot_timer << ",\"projectile_hits\":" << body->projectile_hits
        << ",\"clear_pending\":" << unsigned(body->successful_clear_pending)
        << ",\"delete_marked\":" << unsigned(body->delete_marked)
        << ",\"non_wall_contacts\":" << body->non_wall_contacts
        << ",\"gate_seen\":" << unsigned(body->top_contact_pending)
        << ",\"gate_enabled\":" << unsigned(body->top_contact_enabled)
        << ",\"group_chain\":" << (group ? group->chain : 0)
        << ",\"group_id\":" << body->chain_id
        << ",\"group_shadow\":" << (group ? group->secondary_count : 0)
        << ",\"group_num\":" << (group ? group->num : 0) << '}';
    }
    std::cout << "]}\n";
  }

  static std::uint64_t match_rng(const Simulator& sim, DxRandom& tracker) {
    std::uint64_t count = 0;
    while (tracker.index() != sim.rng_.index() || tracker.state() != sim.rng_.state()) {
      if (++count > 1'000'000) throw std::runtime_error("RNG state cannot be reached within the trace bound");
      (void)tracker.raw_u32();
    }
    return count;
  }
};
}  // namespace irisu::metsu

namespace {
std::uint32_t word(const std::vector<unsigned char>& bytes, std::size_t offset) {
  return std::uint32_t(bytes.at(offset)) | (std::uint32_t(bytes.at(offset + 1)) << 8U)
    | (std::uint32_t(bytes.at(offset + 2)) << 16U) | (std::uint32_t(bytes.at(offset + 3)) << 24U);
}
std::uint16_t control_word() {
  std::uint16_t value{};
#if defined(__i386__) || defined(__x86_64__)
  __asm__ __volatile__("fnstcw %0" : "=m"(value));
#endif
  return value;
}
}

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) throw std::invalid_argument("usage: irisu-metsu-trace REPLAY.rpy SHA256 [MAX_INPUTS]");
    const std::string sha = argv[2];
    if (sha.size() != 64 || sha.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::invalid_argument("a lowercase replay SHA-256 is required");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open replay");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 52 || (bytes.size() - 52) % 4 || word(bytes, 16) != 1)
      throw std::invalid_argument("expected an original-format v2.03 mode-1 replay");
    const auto count = (bytes.size() - 52) / 4;
    if (!count || count > 5'000'000) throw std::invalid_argument("unsupported replay input count");
    std::uint64_t maximum = count;
    if (argc == 4) {
      const std::string value = argv[3];
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), maximum);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !maximum || maximum > count)
        throw std::invalid_argument("MAX_INPUTS must be in [1,replay input count]");
    }
    irisu::detail::install_canonical_floating_point_environment();
    irisu::metsu::Simulator simulator;
    const auto seed = word(bytes, 0);
    simulator.reset(seed);
    irisu::DxRandom rng_tracker;
    rng_tracker.seed(seed);
    auto rng_calls = irisu::metsu::ReplayTraceAccess::match_rng(simulator, rng_tracker);
    std::cout << "{\"type\":\"metadata\",\"schema\":\"irisu-metsu-state-trace-v1\",\"source\":\"native\","
      "\"mode\":1,\"boundary\":\"after-actor-pass\",\"physics_backend\":\"" IRISU_METSU_TRACE_PHYSICS
      "\",\"seed\":" << seed << ",\"config_hash\":\"" << simulator.config_hash()
      << "\",\"replay_sha256\":\"" << sha << "\",\"replay_input_count\":" << count
      << ",\"requested_inputs\":" << maximum << ",\"x87_cw\":" << control_word() << "}\n";
    irisu::metsu::ReplayTraceAccess::write(simulator, 0, rng_calls, static_cast<std::uint32_t>(count));
    for (std::uint64_t index = 0; index < maximum; ++index) {
      const auto packed = word(bytes, 52 + static_cast<std::size_t>(index) * 4);
      irisu::Action action;
      action.kind = static_cast<irisu::ActionKind>(packed & 3U);
      action.cursor_x = (packed >> 2U) & 1023U;
      action.cursor_y = (packed >> 12U) & 511U;
      action.suppress_fresh_edges = index < 2;
      const auto result = simulator.step(action);
      rng_calls += irisu::metsu::ReplayTraceAccess::match_rng(simulator, rng_tracker);
      irisu::metsu::ReplayTraceAccess::write(simulator, index + 1, rng_calls, static_cast<std::uint32_t>(count));
      if (result.terminated || result.truncated) break;
    }
  } catch (const std::exception& error) {
    std::cerr << "Metsu replay trace: " << error.what() << '\n';
    return 1;
  }
}
