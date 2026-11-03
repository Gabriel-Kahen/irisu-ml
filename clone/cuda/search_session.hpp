#pragma once
#include "search.hpp"
#include <vector>

namespace irisu::cuda_search {
struct SessionOptions {
  Options game{};
  std::uint32_t depths{3}, beam{1};
};
struct SessionOutput {
  // Rounds are flattened in order. Each contains round_counts[d] results and
  // ranks. A chosen child ordinal encodes parent rank * fanout + candidate.
  std::vector<std::uint32_t> round_counts, round_kept, ranks;
  std::vector<Result> results;
  std::vector<std::uint32_t> winner_candidates, winner_words;
  std::uint64_t winner_canonical_chain{}, actual_simulated_ticks{}, actual_simulated_branches{};
  Timings timings{};
};

// One prefix and reusable device parent/child pools across dependent depths.
// Every depth, selection and compaction stays on the GPU; downloads occur only
// after the final depth. Spec mutations are relative to each segment start.
void run_session(const SessionOptions& options,const unsigned char* image,
    std::size_t image_bytes,const std::uint32_t* replay,std::uint32_t replay_size,
    const Candidate* candidates,std::uint32_t fanout,SessionOutput* output);
} // namespace irisu::cuda_search
