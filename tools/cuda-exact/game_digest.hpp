#pragma once

// Explicit scalar encoding shared by the independent CPU oracle and GPU probe.
// FNV-1a is a compact regression digest, not a collision-resistant attestation.
#include <cstdint>
#include <cstdio>
#include <type_traits>

#ifdef __CUDACC__
#define IRISU_DIGEST_HD __host__ __device__
#else
#define IRISU_DIGEST_HD
#endif

namespace irisu::game_digest {
inline constexpr std::uint64_t basis = 14695981039346656037ULL;

struct Hash {
  std::uint64_t value{basis};
  IRISU_DIGEST_HD void byte(unsigned char x) { value = (value ^ x) * 1099511628211ULL; }
  template<class T> IRISU_DIGEST_HD void integer(T x) {
    using U = typename std::make_unsigned<T>::type;
    const auto bits = static_cast<U>(x);
    for (unsigned i = 0; i < sizeof(T); ++i) byte(static_cast<unsigned char>(bits >> (8*i)));
  }
  template<class T> IRISU_DIGEST_HD void scalar(T x) {
    if constexpr (std::is_enum<T>::value) integer(static_cast<typename std::underlying_type<T>::type>(x));
    else if constexpr (std::is_same<T, bool>::value) byte(x ? 1 : 0);
    else if constexpr (std::is_same<T, double>::value) {
      std::uint64_t bits{};
      auto* out = reinterpret_cast<unsigned char*>(&bits);
      const auto* in = reinterpret_cast<const unsigned char*>(&x);
      for (unsigned i = 0; i < sizeof(x); ++i) out[i] = in[i];
      integer(bits);
    } else if constexpr (std::is_same<T, float>::value) {
      std::uint32_t bits{};
      auto* out = reinterpret_cast<unsigned char*>(&bits);
      const auto* in = reinterpret_cast<const unsigned char*>(&x);
      for (unsigned i = 0; i < sizeof(x); ++i) out[i] = in[i];
      integer(bits);
    } else integer(x);
  }
};

template<class Body,class Visitor> IRISU_DIGEST_HD void visit_body(const Body& b,Visitor visit) {
#define FIELD(name) visit(#name,b.name)
  FIELD(id); FIELD(kind); FIELD(shape); FIELD(lifecycle); FIELD(color);
  FIELD(position.x); FIELD(position.y); FIELD(velocity.x); FIELD(velocity.y);
  FIELD(angle); FIELD(angular_velocity);
  FIELD(native_position.x); FIELD(native_position.y);
  FIELD(native_center.x); FIELD(native_center.y);
  FIELD(native_velocity.x); FIELD(native_velocity.y);
  FIELD(native_angle); FIELD(native_angular_velocity);
  FIELD(native_state_valid); FIELD(native_center_valid);
  FIELD(size); FIELD(density); FIELD(friction); FIELD(restitution);
  FIELD(inverse_mass); FIELD(inverse_inertia);
  FIELD(chain_id); FIELD(actor_slot); FIELD(size_slot); FIELD(projectile_hits);
  FIELD(age_ticks); FIELD(remaining_lifetime);
  FIELD(scripted_velocity.x); FIELD(scripted_velocity.y); FIELD(rot_timer);
  FIELD(physics_owned); FIELD(special); FIELD(freshness_state); FIELD(grouped);
  FIELD(successful_clear_pending); FIELD(non_wall_contacts);
  FIELD(top_contact_pending); FIELD(top_contact_enabled); FIELD(physics_update_count);
  FIELD(rule_guard_f0); FIELD(delete_marked); FIELD(pending_delete);
  FIELD(sleeping); FIELD(sleep_time);
#undef FIELD
}

struct BodyHashVisitor {
  Hash* hash;
  template<class T> IRISU_DIGEST_HD void operator()(const char*,T value) const { hash->scalar(value); }
};
template<class Body> IRISU_DIGEST_HD void body(Hash& h,const Body& b) {
  visit_body(b,BodyHashVisitor{&h});
}

template<class Body> void dump_body(std::FILE* out,std::uint64_t tick,const Body& b) {
  visit_body(b,[&](const char* name,auto value){
    Hash h;h.scalar(value);
    std::fprintf(out,"%llu %u %s %016llx\n",static_cast<unsigned long long>(tick),b.id,name,
      static_cast<unsigned long long>(h.value));
  });
}

template<class Range> IRISU_DIGEST_HD std::uint64_t bodies(const Range& range) {
  Hash h;
  h.integer(static_cast<std::uint64_t>(range.size()));
  for (const auto& b : range) body(h, b);
  return h.value;
}

template<class Observation> IRISU_DIGEST_HD std::uint64_t observation(const Observation& o) {
  Hash h;
#define FIELD(name) h.scalar(o.name)
  FIELD(tick); FIELD(score); FIELD(gauge); FIELD(level); FIELD(terminated); FIELD(truncated);
  h.integer(static_cast<std::uint64_t>(o.bodies.size()));
  FIELD(field_x); FIELD(field_y); FIELD(field_width); FIELD(field_height);
  FIELD(side_wall_top); FIELD(side_wall_bottom); FIELD(gauge_max);
  FIELD(active_colors); FIELD(current_spawn_interval_ticks);
  FIELD(left_held); FIELD(right_held); FIELD(highest_chain); FIELD(qualifying_clear_count);
#undef FIELD
  return h.value;
}

template<class Step> IRISU_DIGEST_HD std::uint64_t transition(const Step& s) {
  Hash h;
  h.scalar(s.reward); h.scalar(s.terminated); h.scalar(s.truncated);
  const auto& d = s.diagnostics;
#define FIELD(name) h.scalar(d.name)
  FIELD(config_hash); FIELD(finish_call_count); FIELD(terminal_metadata_recorded);
  FIELD(recorded_final_score); FIELD(recorded_final_highest_chain); FIELD(recorded_final_level);
  FIELD(recorded_final_clears); FIELD(latest_final_score); FIELD(latest_final_highest_chain);
  FIELD(latest_final_level); FIELD(latest_final_clears);
#undef FIELD
  return h.value;
}

// Native event strings are independently encoded to match cuda_game::Events.
template<class Range> std::uint64_t native_events(const Range& events) {
  Hash h;
  for (const auto& event : events) {
    Hash detail;
    for (unsigned char c : event.detail) detail.byte(c);
    h.scalar(event.tick); h.scalar(static_cast<std::uint8_t>(event.kind));
    h.scalar(event.a); h.scalar(event.b); h.scalar(event.value);
    h.scalar(detail.value); h.scalar(event.sequence);
  }
  return h.value;
}

// Wire format v1: these twelve words are emitted separately in little-endian
// order. The struct itself has no native padding and may also be copied by CUDA.
struct Frame {
  std::uint64_t tick{}, body_hash{}, observation_hash{}, event_hash{}, event_count{};
  std::uint64_t transition_hash{}, body_count{}, score{}, gauge{}, level{};
  // Legacy v1 field name: this word encodes latest_final_score. Canonical RPY
  // export uses the FIRST recorded_final_score, separately hashed in transition.
  std::uint64_t highest_chain{}, canonical_score{};
};
static_assert(sizeof(Frame) == 12 * sizeof(std::uint64_t));

template<class Range, class Observation, class Step>
IRISU_DIGEST_HD Frame make_frame(const Range& b, const Observation& o,
                                std::uint64_t event_hash, std::uint64_t event_count,
                                const Step& s) {
  return {o.tick, bodies(b), observation(o), event_hash, event_count, transition(s),
          static_cast<std::uint64_t>(b.size()), static_cast<std::uint64_t>(o.score),
          static_cast<std::uint64_t>(o.gauge), o.level, o.highest_chain,
          static_cast<std::uint64_t>(s.diagnostics.latest_final_score)};
}
}  // namespace irisu::game_digest

#undef IRISU_DIGEST_HD
