#include "irisu/metsu/simulator.hpp"
#include "irisu/simulator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using irisu::ActionKind;
using irisu::EventKind;
using irisu::metsu::Simulator;

std::size_t events(const irisu::StepResult& result, EventKind kind) {
  return static_cast<std::size_t>(std::count_if(result.events.begin(), result.events.end(),
      [kind](const auto& event) { return event.kind == kind; }));
}

const irisu::Body& body(const irisu::metsu::Snapshot& state, irisu::BodyId id) {
  const auto found = std::find_if(state.bodies.begin(), state.bodies.end(),
      [id](const auto& item) { return item.id == id; });
  assert(found != state.bodies.end());
  return *found;
}

irisu::MechanicsConfig empty_config() {
  auto config = irisu::metsu::default_config();
  config.initial_falling_count = 0;
  config.spawn_interval_ticks = 100'000;
  return config;
}

void skip_first_cadence(Simulator& simulator) {
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  simulator.restore_state(state);
}

void exact_level_fixture(const char* path) {
  std::ifstream input(path);
  assert(input.is_open());
  std::string header;
  std::getline(input, header);
  std::uint32_t level, drain, rot, colors, interval, ex_interval, reward, cutoff;
  std::string descent, rise, scale;
  std::uint32_t rows = 0;
  while (input >> level >> drain >> descent >> rise >> rot >> colors >> interval
               >> ex_interval >> scale >> reward >> cutoff) {
    assert(level == ++rows);
    const auto p = irisu::metsu::level_parameters(level);
    assert(p.passive_drain_unit == drain && p.rot_penalty == rot);
    assert(p.maximum_color_id == colors && p.spawn_interval_frames == interval);
    assert(p.clear_reward_unit == reward && cutoff == 100);
    assert(std::bit_cast<std::uint32_t>(static_cast<float>(p.scripted_descent_per_update)) ==
           std::stoul(descent, nullptr, 16));
    assert(std::bit_cast<std::uint32_t>(static_cast<float>(irisu::metsu::ex_scripted_rise)) ==
           std::stoul(rise, nullptr, 16));
    assert(std::bit_cast<std::uint64_t>(p.score_scale) == std::stoull(scale, nullptr, 16));
    assert(irisu::metsu::ex_spawn_interval(level) == ex_interval);
  }
  assert(input.eof() && rows == 99);
  for (const auto bad : {0U, 100U, std::numeric_limits<std::uint32_t>::max()}) {
    bool rejected = false;
    try { (void)irisu::metsu::level_parameters(bad); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
}

void reject_incompatible_shape_rng_override() {
  for (const auto maximum : {99U, 101U}) {
    auto config = irisu::metsu::default_config();
    config.shape_random_max = maximum;
    bool rejected = false;
    try { Simulator invalid(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
}

void reset_profile_and_dual_cadence() {
  Simulator simulator;
  const auto observation = simulator.reset(41);
  const auto state = simulator.clone_state();
  assert(observation.tick == 0 && observation.score == 0 && observation.gauge == 1'000);
  assert(observation.active_colors == 6 && observation.current_spawn_interval_ticks == 110);
  assert(observation.field_x == 94 && observation.field_width == 420);
  assert(state.bodies.size() == 10 && state.rng_index == 51 && state.level_shape_cutoff == 100);
  for (const auto& item : state.bodies) {
    assert(item.kind == irisu::BodyKind::Piece && item.shape == irisu::Shape::Box);
    assert(!item.rule_guard_f0 && !item.physics_owned && item.freshness_state == 2);
    assert(item.age_ticks == 1 && item.remaining_lifetime == 9'999);
    assert(std::abs(item.position.y - 60.13) < 0.00001);
  }
  const auto first = simulator.step({ActionKind::Wait, 0, 0, 1});
  assert(events(first, EventKind::Spawned) == 2);
  std::vector<std::string> order;
  for (const auto& event : first.events) if (event.kind == EventKind::Spawned) order.push_back(event.detail);
  assert((order == std::vector<std::string>{"Metsu ordinary spawn", "Metsu EX spawn"}));
}

void gauge_and_opposite_scripted_motion() {
  Simulator simulator(empty_config());
  simulator.reset(7);
  skip_first_cadence(simulator);
  const auto ordinary = simulator.spawn_piece(irisu::Shape::Box, 0, 30, {200, 100});
  const auto ex = simulator.spawn_ex(1, 42, {400, 300});
  const auto result = simulator.step({ActionKind::Wait, 0, 0, 20});
  const auto state = simulator.clone_state();
  assert(simulator.observation().tick == 20 && simulator.observation().gauge == 1'000);
  assert(events(result, EventKind::GaugeChanged) == 0);
  assert(std::abs(body(state, ordinary).position.y - 102.6) < 0.0005);
  assert(std::abs(body(state, ex).position.y - 299.0) < 0.0005);
  assert(body(state, ex).rule_guard_f0 == 1 && body(state, ex).remaining_lifetime < -1);
  assert(simulator.ex_timer(ex) == -1);
}

void ex_countdown_and_snapshot_future() {
  Simulator simulator(empty_config());
  simulator.reset(9);
  skip_first_cadence(simulator);
  const auto ex = simulator.spawn_ex(2, 42, {300, 200});
  auto state = simulator.clone_state();
  auto& item = state.bodies.front();
  item.physics_owned = true;
  item.freshness_state = 2;
  item.lifecycle = irisu::Lifecycle::DynamicFresh;
  item.chain_id = 1;
  state.groups = {{1, 1, 1, 0}};
  state.next_chain_id = 2;
  const auto slot = item.actor_slot;
  state.ex_timers[slot] = 2;
  simulator.restore_state(state);
  simulator.step({ActionKind::Wait, 0, 0, 1});
  assert(simulator.ex_timer(ex) == 1);
  const auto halfway = simulator.serialize_snapshot();
  simulator.step({ActionKind::Wait, 0, 0, 1});
  assert(simulator.ex_timer(ex) == 0);
  assert(body(simulator.clone_state(), ex).grouped);
  const auto future = simulator.serialize_snapshot();
  simulator.restore_snapshot(halfway);
  assert(simulator.ex_timer(ex) == 1);
  simulator.step({ActionKind::Wait, 0, 0, 1});
  assert(simulator.serialize_snapshot() == future);
  state = simulator.clone_state();
  state.ex_timers[slot] = 121;
  const auto stable = simulator.state_hash();
  bool rejected = false;
  try { simulator.restore_state(state); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && simulator.state_hash() == stable);
}

void ordinary_group_activates_ex_contact() {
  Simulator simulator(empty_config());
  simulator.reset(12);
  skip_first_cadence(simulator);
  const auto ordinary = simulator.spawn_piece(irisu::Shape::Box, 0, 30, {260, 200});
  const auto ex = simulator.spawn_ex(0, 42, {290, 200});
  auto state = simulator.clone_state();
  auto& source = state.bodies.front();
  assert(source.id == ordinary);
  source.physics_owned = true;
  source.freshness_state = 2;
  source.grouped = true;
  source.chain_id = 1;
  source.lifecycle = irisu::Lifecycle::Confirmed;
  state.groups = {{1, 1, 1, 0}};
  state.next_chain_id = 2;
  simulator.restore_state(state);
  const auto result = simulator.step({ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(events(result, EventKind::Contact) > 0);
  assert(body(state, ex).physics_owned && !body(state, ex).grouped);
  assert(body(state, ex).chain_id == 1 && simulator.ex_timer(ex) == 119);
  assert(state.groups.front().chain == 2);
  const auto bytes = simulator.serialize_snapshot();
  simulator.restore_snapshot(bytes);
  assert(simulator.ex_timer(ex) == 119 && simulator.serialize_snapshot() == bytes);
}

void level_reentry_keeps_contact_participants_alive() {
  auto config = irisu::metsu::default_config();
  config.initial_falling_count = 0;
  Simulator simulator(config);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30, {240, 410});
  simulator.spawn_piece(irisu::Shape::Box, 1, 30, {400, 100});
  auto state = simulator.clone_state();
  auto& landing = state.bodies.front();
  landing.physics_owned = true;
  landing.grouped = true;
  landing.chain_id = 1;
  landing.freshness_state = 2;
  landing.lifecycle = irisu::Lifecycle::Confirmed;
  landing.top_contact_enabled = true;
  landing.age_ticks = 200;
  state.groups = {{1, 1, 1, 0}};
  state.next_chain_id = 2;
  state.qualifying_clear_count = 5;
  simulator.restore_state(state);
  // Initial dual cadence fills vector capacity; the sixth clear immediately
  // re-enters both generators while the same native contact is being handled.
  const auto result = simulator.step({ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(state.level == 2 && state.qualifying_clear_count == 6);
  assert(state.next_body_id == 7 && events(result, EventKind::Spawned) == 4);
  assert(events(result, EventKind::Confirmed) == 1 && simulator.observation().score == 6);
  const auto checkpoint = simulator.serialize_snapshot();
  simulator.step({ActionKind::Wait, 0, 0, 3});
  const auto future = simulator.serialize_snapshot();
  simulator.restore_snapshot(checkpoint);
  simulator.step({ActionKind::Wait, 0, 0, 3});
  assert(simulator.serialize_snapshot() == future);
}

void separate_snapshot_identity() {
  Simulator metsu;
  irisu::Simulator normal;
  assert(normal.config_hash() == 0xec0e8463feaf2670ULL);
  assert(metsu.config_hash() != normal.config_hash());
  const auto normal_bytes = normal.serialize_snapshot();
  const auto metsu_bytes = metsu.serialize_snapshot();
  const auto normal_hash = normal.state_hash(), metsu_hash = metsu.state_hash();
  bool rejected = false;
  try { metsu.restore_snapshot(normal_bytes); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && metsu.state_hash() == metsu_hash);
  rejected = false;
  try { normal.restore_snapshot(metsu_bytes); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && normal.state_hash() == normal_hash);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  exact_level_fixture(argv[1]);
  reject_incompatible_shape_rng_override();
  reset_profile_and_dual_cadence();
  gauge_and_opposite_scripted_motion();
  ex_countdown_and_snapshot_future();
  ordinary_group_activates_ex_contact();
  level_reentry_keeps_contact_participants_alive();
  separate_snapshot_identity();
  std::cout << "Metsu tests passed\n";
}
