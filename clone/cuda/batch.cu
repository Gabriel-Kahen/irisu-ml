#include "batch.hpp"

#include <limits>

namespace irisu::cuda {
namespace {

constexpr std::uint32_t tile_size = 256;

std::size_t state_bytes(Arena arena) {
  return std::size_t{arena.capacity} * arena.stride;
}

std::size_t lane_bytes(Arena arena) {
  return std::size_t{arena.capacity} * sizeof(Lane);
}

bool overlaps(const void* a, std::size_t an, const void* b, std::size_t bn) {
  const auto ap = reinterpret_cast<std::uintptr_t>(a);
  const auto bp = reinterpret_cast<std::uintptr_t>(b);
  // Subtraction avoids address-end overflow on malformed views.
  return an && bn && (ap <= bp ? bp - ap < an : ap - bp < bn);
}

bool valid(Arena arena) {
  return arena.states && arena.lanes && arena.capacity && arena.stride &&
         arena.capacity < no_lane &&
         arena.capacity <= std::numeric_limits<std::size_t>::max() /
                               arena.stride &&
         !overlaps(arena.states, state_bytes(arena), arena.lanes,
                   lane_bytes(arena));
}

bool disjoint(Arena a, Arena b) {
  return !overlaps(a.states, state_bytes(a), b.states, state_bytes(b)) &&
         !overlaps(a.states, state_bytes(a), b.lanes, lane_bytes(b)) &&
         !overlaps(a.lanes, lane_bytes(a), b.states, state_bytes(b)) &&
         !overlaps(a.lanes, lane_bytes(a), b.lanes, lane_bytes(b));
}

bool outside(Arena arena, const void* data, std::size_t bytes) {
  return data && !overlaps(arena.states, state_bytes(arena), data, bytes) &&
         !overlaps(arena.lanes, lane_bytes(arena), data, bytes);
}

bool compatible(Arena destination, Arena source, std::uint32_t count,
                std::uint32_t source_count) {
  return valid(destination) && valid(source) &&
         destination.stride == source.stride &&
         count <= destination.capacity && source_count <= source.capacity &&
         disjoint(destination, source);
}

__device__ bool selectable(const Lane& lane) {
  return lane.error == LaneError::none &&
         (lane.status == LaneStatus::ready ||
          lane.status == LaneStatus::terminal);
}

__device__ Lane branch_lane(Arena source, const Branch& spec,
                            std::uint32_t parent_count) {
  Lane result{};
  result.status = LaneStatus::error;
  result.lineage = spec.lineage;
  result.parent = spec.parent;
  if (spec.parent >= parent_count) {
    result.error = LaneError::parent_out_of_range;
  } else if (source.lanes[spec.parent].status != LaneStatus::ready ||
             source.lanes[spec.parent].error != LaneError::none) {
    result.error = LaneError::parent_inactive;
  } else if (spec.mutation_offset != no_mutation &&
             (source.stride < sizeof(std::uint32_t) ||
              spec.mutation_offset > source.stride - sizeof(std::uint32_t) ||
              spec.mutation_offset % alignof(std::uint32_t))) {
    result.error = LaneError::invalid_mutation;
  } else {
    result = source.lanes[spec.parent];
    result.parent = spec.parent;
    result.lineage = spec.lineage;
  }
  return result;
}

template<bool Branching>
__global__ void copy_lanes(Arena destination, Arena source,
                           const Branch* specs, const std::uint32_t* indices,
                           std::uint32_t count, std::uint32_t source_count) {
  const auto index = static_cast<std::uint32_t>(blockIdx.x);
  if (index >= count) return;
  __shared__ Lane lane;
  __shared__ std::uint32_t parent;
  if (threadIdx.x == 0) {
    if constexpr (Branching) {
      parent = specs[index].parent;
      lane = branch_lane(source, specs[index], source_count);
    } else {
      parent = indices[index];
      if (parent >= source_count) {
        lane = Lane{};
        lane.status = LaneStatus::error;
        lane.error = LaneError::invalid_selection;
      } else if (!selectable(source.lanes[parent])) {
        lane = Lane{};
        lane.status = LaneStatus::error;
        lane.error = LaneError::parent_inactive;
      } else {
        lane = source.lanes[parent];
      }
    }
    destination.lanes[index] = lane;
  }
  __syncthreads();
  const std::size_t output = std::size_t{index} * destination.stride;
  const bool accepted = lane.error == LaneError::none;
  const std::size_t input = accepted ? std::size_t{parent} * source.stride : 0;
  if (source.stride % sizeof(uint4) == 0 &&
      reinterpret_cast<std::uintptr_t>(source.states) % alignof(uint4) == 0 &&
      reinterpret_cast<std::uintptr_t>(destination.states) % alignof(uint4) == 0) {
    auto* target = reinterpret_cast<uint4*>(destination.states + output);
    const auto* origin = reinterpret_cast<const uint4*>(source.states + input);
    for (std::size_t offset = threadIdx.x; offset < source.stride / sizeof(uint4);
         offset += blockDim.x) {
      target[offset] = accepted ? origin[offset] : uint4{0, 0, 0, 0};
    }
  } else {
    for (std::size_t offset = threadIdx.x; offset < source.stride;
         offset += blockDim.x) {
      destination.states[output + offset] = accepted ? source.states[input + offset] : 0U;
    }
  }
  if constexpr (Branching) {
    const auto mutation = specs[index].mutation_offset;
    if (accepted && mutation != no_mutation) {
      __syncthreads();
      if (threadIdx.x < sizeof(std::uint32_t))
        destination.states[output + mutation + threadIdx.x] =
            (specs[index].mutation_value >> (threadIdx.x * 8)) & 0xffU;
    }
  }
}

__device__ bool before(std::uint32_t a, std::uint32_t b, const Lane* lanes,
                       std::uint32_t count) {
  if (a >= count || b >= count) return a < b;
  const bool av = selectable(lanes[a]), bv = selectable(lanes[b]);
  if (av != bv) return av;
  if (av) {
    for (unsigned key = 0; key < 4; ++key) {
      if (lanes[a].objective.keys[key] != lanes[b].objective.keys[key])
        return lanes[a].objective.keys[key] > lanes[b].objective.keys[key];
    }
  }
  return a < b;
}

__global__ void initial_rank(std::uint32_t* indices, const Lane* lanes,
                             std::uint32_t count, std::uint32_t padded) {
  __shared__ std::uint32_t ranks[tile_size];
  const auto local = static_cast<std::uint32_t>(threadIdx.x);
  const auto global = static_cast<std::uint32_t>(blockIdx.x) * tile_size + local;
  ranks[local] = global < count ? global : no_lane;
  __syncthreads();
  for (std::uint32_t width = 2; width <= tile_size; width <<= 1) {
    for (std::uint32_t distance = width / 2; distance; distance >>= 1) {
      const auto other = local ^ distance;
      if (other > local) {
        const bool forward = (global & width) == 0;
        const auto a = ranks[local], b = ranks[other];
        if (forward ? before(b, a, lanes, count) : before(a, b, lanes, count)) {
          ranks[local] = b;
          ranks[other] = a;
        }
      }
      __syncthreads();
    }
  }
  if (global < padded) indices[global] = ranks[local];
}

__global__ void global_merge(std::uint32_t* indices, const Lane* lanes,
                             std::uint32_t count, std::uint32_t padded,
                             std::uint32_t width, std::uint32_t distance) {
  const auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                                 threadIdx.x);
  if (index >= padded) return;
  const auto other = index ^ distance;
  if (other <= index) return;
  const auto a = indices[index], b = indices[other];
  const bool forward = (index & width) == 0;
  if (forward ? before(b, a, lanes, count) : before(a, b, lanes, count)) {
    indices[index] = b;
    indices[other] = a;
  }
}

__global__ void tile_merge(std::uint32_t* indices, const Lane* lanes,
                           std::uint32_t count, std::uint32_t width) {
  __shared__ std::uint32_t ranks[tile_size];
  const auto local = static_cast<std::uint32_t>(threadIdx.x);
  const auto index = static_cast<std::uint32_t>(blockIdx.x) * tile_size + local;
  ranks[local] = indices[index];
  __syncthreads();
  for (std::uint32_t distance = tile_size / 2; distance; distance >>= 1) {
    const auto other = local ^ distance;
    if (other > local) {
      const auto a = ranks[local], b = ranks[other];
      const bool forward = (index & width) == 0;
      if (forward ? before(b, a, lanes, count) : before(a, b, lanes, count)) {
        ranks[local] = b;
        ranks[other] = a;
      }
    }
    __syncthreads();
  }
  indices[index] = ranks[local];
}

}  // namespace

cudaError_t allocate_arena(Arena* arena, std::uint32_t capacity,
                           std::uint32_t bytes_per_lane) {
  if (!arena || arena->states || arena->lanes || !capacity || !bytes_per_lane ||
      capacity == no_lane ||
      capacity > std::numeric_limits<std::size_t>::max() / bytes_per_lane)
    return cudaErrorInvalidValue;
  Arena allocated{};
  allocated.capacity = capacity;
  allocated.stride = bytes_per_lane;
  auto status = cudaMalloc(&allocated.states, state_bytes(allocated));
  if (status != cudaSuccess) return status;
  status = cudaMalloc(&allocated.lanes, lane_bytes(allocated));
  if (status == cudaSuccess)
    status = cudaMemset(allocated.lanes, 0, lane_bytes(allocated));
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    cudaFree(allocated.lanes);
    cudaFree(allocated.states);
    return status;
  }
  *arena = allocated;
  return cudaSuccess;
}

cudaError_t free_arena(Arena* arena) {
  if (!arena) return cudaErrorInvalidValue;
  const auto states_status = cudaFree(arena->states);
  const auto lanes_status = cudaFree(arena->lanes);
  *arena = Arena{};
  return states_status == cudaSuccess ? lanes_status : states_status;
}

cudaError_t branch_async(Arena destination, Arena source,
                         const Branch* device_specs, std::uint32_t count,
                         std::uint32_t parent_count, cudaStream_t stream) {
  if (!compatible(destination, source, count, parent_count) ||
      (count && !outside(destination, device_specs, std::size_t{count} *
                                                      sizeof(Branch))))
    return cudaErrorInvalidValue;
  if (!count) return cudaSuccess;
  copy_lanes<true><<<count, 128, 0, stream>>>(destination, source, device_specs,
                                           nullptr, count, parent_count);
  return cudaGetLastError();
}

cudaError_t checkpoint_async(Arena destination, Arena source,
                             std::uint32_t count, cudaStream_t stream) {
  if (!compatible(destination, source, count, count)) return cudaErrorInvalidValue;
  if (!count) return cudaSuccess;
  auto status = cudaMemcpyAsync(destination.states, source.states,
                                std::size_t{count} * source.stride,
                                cudaMemcpyDeviceToDevice, stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(destination.lanes, source.lanes,
                             std::size_t{count} * sizeof(Lane),
                             cudaMemcpyDeviceToDevice, stream);
  return status;
}

cudaError_t compact_async(Arena destination, Arena source,
                          const std::uint32_t* device_indices,
                          std::uint32_t count, std::uint32_t source_count,
                          cudaStream_t stream) {
  if (!compatible(destination, source, count, source_count) ||
      (count && !outside(destination, device_indices,
                         std::size_t{count} * sizeof(std::uint32_t))))
    return cudaErrorInvalidValue;
  if (!count) return cudaSuccess;
  copy_lanes<false><<<count, 128, 0, stream>>>(destination, source, nullptr,
                                            device_indices, count, source_count);
  return cudaGetLastError();
}

std::uint32_t rank_capacity(std::uint32_t count) {
  if (!count) return 0;
  if (count > (1U << 30)) return 0;
  std::uint32_t capacity = tile_size;
  while (capacity < count) capacity <<= 1;
  return capacity;
}

cudaError_t rank_async(Arena source, std::uint32_t count,
                       std::uint32_t* device_indices,
                       std::uint32_t index_capacity, cudaStream_t stream) {
  const auto padded = rank_capacity(count);
  if (!valid(source) || count > source.capacity || (count && !padded) ||
      index_capacity < padded ||
      (count && !outside(source, device_indices,
                         std::size_t{padded} * sizeof(std::uint32_t))))
    return cudaErrorInvalidValue;
  if (!count) return cudaSuccess;
  initial_rank<<<padded / tile_size, tile_size, 0, stream>>>(
      device_indices, source.lanes, count, padded);
  for (std::uint32_t width = tile_size * 2; width <= padded; width <<= 1) {
    for (std::uint32_t distance = width / 2; distance >= tile_size; distance >>= 1)
      global_merge<<<(padded + 255) / 256, 256, 0, stream>>>(
          device_indices, source.lanes, count, padded, width, distance);
    tile_merge<<<padded / tile_size, tile_size, 0, stream>>>(
        device_indices, source.lanes, count, width);
  }
  return cudaGetLastError();
}

}  // namespace irisu::cuda
