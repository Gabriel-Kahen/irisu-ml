#pragma once

#include <cstddef>
#include <cstdint>
#include "../../tools/cuda-exact/game_digest.hpp"

#ifdef __CUDACC__
#define IRISU_SEARCH_HD __host__ __device__
#else
#define IRISU_SEARCH_HD
#endif

namespace irisu::cuda_search {

// Experimental full-game search, deliberately separate from admitted CPU
// training identities. Inputs and outputs use explicit, fixed-width scalars.
inline constexpr const char* identity = "experimental-cuda-game-search-v2";
inline constexpr std::uint32_t result_magic=0x32535249U, result_version=2;
inline constexpr unsigned maximum_mutations = 4;
struct Mutation { std::uint32_t offset{}, input{}; };
struct Candidate {
  std::uint32_t count{}, reserved{};
  Mutation mutations[maximum_mutations]{};
};
static_assert(sizeof(Candidate) == 40);

struct Result {
  game_digest::Frame final{};
  std::uint64_t trace_hash{game_digest::basis}, simulated_ticks{}, clears{};
  std::uint64_t canonical_level{}, natural{}, failed{}, guest_error{}, fault_pc{};
  // The first terminal metadata is authoritative for replay/scoring. Frame's
  // last word separately preserves latest_final_score for regression parity.
  std::uint64_t canonical_score{};
  std::int64_t objective[11]{};
};
static_assert(sizeof(Result) == 256);

IRISU_SEARCH_HD inline bool valid(const Candidate& c, std::uint32_t horizon) {
  if (c.count > maximum_mutations || c.reserved) return false;
  for (unsigned i=0; i<c.count; ++i) {
    if (c.mutations[i].offset >= horizon || (c.mutations[i].input >> 21)) return false;
    for (unsigned j=0; j<i; ++j)
      if (c.mutations[i].offset == c.mutations[j].offset) return false;
  }
  return true;
}

IRISU_SEARCH_HD inline std::uint32_t input_at(const Candidate& c,
    std::uint32_t offset, std::uint32_t incumbent) {
  for (unsigned i=0; i<c.count; ++i)
    if (c.mutations[i].offset == offset) return c.mutations[i].input;
  return incumbent;
}

IRISU_SEARCH_HD inline void fold(Result& result, const game_digest::Frame& frame) {
  game_digest::Hash hash{result.trace_hash};
  hash.scalar(frame.tick); hash.scalar(frame.body_hash);
  hash.scalar(frame.observation_hash); hash.scalar(frame.event_hash);
  hash.scalar(frame.event_count); hash.scalar(frame.transition_hash);
  hash.scalar(frame.body_count); hash.scalar(frame.score); hash.scalar(frame.gauge);
  hash.scalar(frame.level); hash.scalar(frame.highest_chain); hash.scalar(frame.canonical_score);
  result.trace_hash=hash.value; result.final=frame;
}

IRISU_SEARCH_HD inline void rank_keys(Result& r, std::uint64_t end_tick,
                                    std::int64_t target) {
  const bool completed=r.natural && r.canonical_level>=100;
  const auto score=static_cast<std::int64_t>(r.natural?r.canonical_score:r.final.score);
  const auto tick=static_cast<std::int64_t>(r.final.tick);
  const auto gauge=static_cast<std::int64_t>(r.final.gauge);
  const auto clears=static_cast<std::int64_t>(r.clears);
  const std::int64_t keys[]={completed && score>=target,completed,
    completed?score:0,completed?clears:0,completed?gauge:0,
    completed?-tick:static_cast<std::int64_t>(r.final.tick>=end_tick),
    completed?0:tick,score,clears,gauge,static_cast<std::int64_t>(r.natural)};
  for (unsigned i=0;i<11;++i) r.objective[i]=keys[i];
}

IRISU_SEARCH_HD inline bool better(const Result& a, std::uint32_t ai,
                                   const Result& b, std::uint32_t bi) {
  if (bool(a.failed)!=bool(b.failed)) return !a.failed;
  for (unsigned i=0;i<11;++i)
    if (a.objective[i]!=b.objective[i]) return a.objective[i]>b.objective[i];
  return ai<bi;
}

struct Options {
  std::uint32_t seed{}, prefix{}, horizon{}, arena_bytes{2U<<20}, keep{1};
  std::int64_t target_score{4393102};
  bool validate_compaction{true};
  bool host_prefix{false};
  bool isolated_warps{false};
  bool trace_every_tick{true}, cache_guest{true};
  std::uint32_t block_threads{4}, stack_bytes{16U<<10};
};
struct Timings {
  double host_prefix_ms{}, prefix_ms{}, branch_ms{}, rollout_ms{}, rank_compact_ms{}, validation_ms{}, total_ms{};
  std::uint64_t allocation_bytes{}, peak_guest_bytes{};
  std::uint64_t compaction_errors{};
};

// Runs initialization/prefix, snapshot fanout, complete futures, ranking and
// selected-state compaction on the device. No host synchronization per tick.
// Caller owns all input/output host buffers. Throws on CUDA/runtime errors.
void run(const Options& options, const unsigned char* image, std::size_t image_bytes,
         const std::uint32_t* replay, std::uint32_t replay_size,
         const Candidate* candidates, std::uint32_t count,
         Result* results, std::uint32_t* ranks, Timings* timings);

} // namespace irisu::cuda_search
#undef IRISU_SEARCH_HD
