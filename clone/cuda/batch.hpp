#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace irisu::cuda {

// These buffers hold pointer-relative guest heaps, not native C++ snapshots.
// This transport/scheduling API makes no physics-parity or promotion claim.
enum class LaneStatus : std::uint32_t { empty, ready, terminal, error };
enum class LaneError : std::uint32_t {
  none,
  parent_out_of_range,
  parent_inactive,
  invalid_mutation,
  invalid_selection,
};

struct Objective {
  std::int64_t keys[4]{};
};

struct Lane {
  Objective objective{};
  std::uint64_t lineage{};
  std::uint32_t parent{};
  LaneStatus status{LaneStatus::empty};
  LaneError error{LaneError::none};
  std::uint32_t reserved{};
};

struct Arena {
  unsigned char* states{};
  Lane* lanes{};
  std::uint32_t capacity{};
  std::uint32_t stride{};
};

inline constexpr std::uint32_t no_mutation = UINT32_MAX;
inline constexpr std::uint32_t no_lane = UINT32_MAX;

struct Branch {
  std::uint32_t parent{};
  std::uint32_t mutation_offset{no_mutation};
  std::uint32_t mutation_value{};
  std::uint32_t reserved{};
  std::uint64_t lineage{};
};

// Allocation is synchronous and initializes all lanes to empty. All other
// operations enqueue work on stream and never read device results on the CPU.
// Free only after all streams using the arena have completed.
cudaError_t allocate_arena(Arena* arena, std::uint32_t capacity,
                           std::uint32_t bytes_per_lane);
cudaError_t free_arena(Arena* arena);

// Source/destination storage must not overlap. Arguments and capacities are
// validated on the host; invalid individual specs become zeroed error lanes.
// A 32-bit mutation is optional, aligned, and confined to one complete lane.
// Branches can only start from ready parents; compact also accepts terminals.
cudaError_t branch_async(Arena destination, Arena source,
                         const Branch* device_specs, std::uint32_t count,
                         std::uint32_t parent_count,
                         cudaStream_t stream = nullptr);
cudaError_t checkpoint_async(Arena destination, Arena source,
                             std::uint32_t count,
                             cudaStream_t stream = nullptr);
cudaError_t compact_async(Arena destination, Arena source,
                          const std::uint32_t* device_indices,
                          std::uint32_t count, std::uint32_t source_count,
                          cudaStream_t stream = nullptr);

// Caller supplies rank_capacity(count) device indices. Result: descending
// lexicographic integer objective; equal keys preserve original lane index.
// Error/empty lanes sort after every ready/terminal lane. Padded slots contain
// no_lane and are never selectable. No allocations occur during ranking.
std::uint32_t rank_capacity(std::uint32_t count);
cudaError_t rank_async(Arena source, std::uint32_t count,
                       std::uint32_t* device_indices,
                       std::uint32_t index_capacity,
                       cudaStream_t stream = nullptr);

}  // namespace irisu::cuda
