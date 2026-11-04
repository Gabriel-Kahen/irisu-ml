#include "../../clone/cuda/batch.hpp"

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace batch = irisu::cuda;
using Clock = std::chrono::steady_clock;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class T> struct Device {
  T* data{};
  explicit Device(std::size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

struct Buffer {
  batch::Arena arena{};
  Buffer(std::uint32_t count, std::uint32_t stride) {
    check(batch::allocate_arena(&arena, count, stride));
  }
  ~Buffer() { batch::free_arena(&arena); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

struct HostArena {
  std::uint32_t count, stride;
  std::vector<unsigned char> states;
  std::vector<batch::Lane> lanes;

  HostArena(std::uint32_t n, std::uint32_t bytes)
      : count(n), stride(bytes), states(std::size_t{n} * bytes), lanes(n) {}
};

bool eligible(const batch::Lane& lane) {
  return lane.error == batch::LaneError::none &&
         (lane.status == batch::LaneStatus::ready ||
          lane.status == batch::LaneStatus::terminal);
}

void initialize(HostArena& arena) {
  for (std::uint32_t i = 0; i < arena.count; ++i) {
    for (std::uint32_t j = 0; j < arena.stride; ++j)
      arena.states[std::size_t{i} * arena.stride + j] = (i * 37U + j * 13U) & 255U;
    arena.lanes[i].status = batch::LaneStatus::ready;
    arena.lanes[i].lineage = i;
    arena.lanes[i].parent = i;
    arena.lanes[i].objective.keys[0] = i % 5;
  }
}

void upload(batch::Arena device, const HostArena& host) {
  check(cudaMemcpy(device.states, host.states.data(), host.states.size(),
                   cudaMemcpyHostToDevice));
  check(cudaMemcpy(device.lanes, host.lanes.data(), host.lanes.size() * sizeof(batch::Lane),
                   cudaMemcpyHostToDevice));
}

HostArena download(batch::Arena device, std::uint32_t count) {
  HostArena host(count, device.stride);
  check(cudaMemcpy(host.states.data(), device.states, host.states.size(),
                   cudaMemcpyDeviceToHost));
  check(cudaMemcpy(host.lanes.data(), device.lanes, count * sizeof(batch::Lane),
                   cudaMemcpyDeviceToHost));
  return host;
}

void equal(const HostArena& a, const HostArena& b, const char* detail) {
  require(a.states == b.states, detail);
  require(a.lanes.size() == b.lanes.size(), detail);
  for (std::size_t i = 0; i < a.lanes.size(); ++i)
    require(std::memcmp(&a.lanes[i], &b.lanes[i], sizeof(batch::Lane)) == 0, detail);
}

void host_branch(HostArena& out, const HostArena& source,
                 const std::vector<batch::Branch>& specs) {
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const auto& spec = specs[i];
    batch::Lane lane{};
    lane.status = batch::LaneStatus::error;
    lane.parent = spec.parent;
    lane.lineage = spec.lineage;
    if (spec.parent >= source.count)
      lane.error = batch::LaneError::parent_out_of_range;
    else if (source.lanes[spec.parent].status != batch::LaneStatus::ready ||
             source.lanes[spec.parent].error != batch::LaneError::none)
      lane.error = batch::LaneError::parent_inactive;
    else if (spec.mutation_offset != batch::no_mutation &&
             (source.stride < 4 || spec.mutation_offset > source.stride - 4 ||
              spec.mutation_offset % 4))
      lane.error = batch::LaneError::invalid_mutation;
    else {
      lane = source.lanes[spec.parent];
      lane.parent = spec.parent;
      lane.lineage = spec.lineage;
    }
    auto* output = out.states.data() + i * out.stride;
    if (lane.error == batch::LaneError::none) {
      std::memcpy(output, source.states.data() + std::size_t{spec.parent} * source.stride,
                  source.stride);
      if (spec.mutation_offset != batch::no_mutation)
        std::memcpy(output + spec.mutation_offset, &spec.mutation_value, 4);
    } else {
      std::memset(output, 0, out.stride);
    }
    out.lanes[i] = lane;
  }
}

std::vector<std::uint32_t> host_rank(const HostArena& source) {
  std::vector<std::uint32_t> indices(source.count);
  std::iota(indices.begin(), indices.end(), 0U);
  std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) {
    const bool av = eligible(source.lanes[a]), bv = eligible(source.lanes[b]);
    if (av != bv) return av;
    if (av) {
      for (unsigned key = 0; key < 4; ++key) {
        const auto x = source.lanes[a].objective.keys[key];
        const auto y = source.lanes[b].objective.keys[key];
        if (x != y) return x > y;
      }
    }
    return false;
  });
  return indices;
}

void host_compact(HostArena& out, const HostArena& source,
                  const std::vector<std::uint32_t>& indices) {
  #pragma omp parallel for schedule(static)
  for (std::uint32_t i = 0; i < out.count; ++i) {
    const auto parent = indices[i];
    if (parent < source.count && eligible(source.lanes[parent])) {
      std::memcpy(out.states.data() + std::size_t{i} * out.stride,
                  source.states.data() + std::size_t{parent} * source.stride, out.stride);
      out.lanes[i] = source.lanes[parent];
    } else {
      std::memset(out.states.data() + std::size_t{i} * out.stride, 0, out.stride);
      out.lanes[i] = batch::Lane{};
      out.lanes[i].status = batch::LaneStatus::error;
      out.lanes[i].error = parent >= source.count ? batch::LaneError::invalid_selection
                                                 : batch::LaneError::parent_inactive;
    }
  }
}

void validate_copy(std::uint32_t stride) {
  constexpr std::uint32_t n = 9;
  Buffer source(n, stride), destination(n, stride), snapshot(n, stride);
  HostArena host(n, stride);
  initialize(host);
  host.lanes[4].status = batch::LaneStatus::terminal;
  host.lanes[5].status = batch::LaneStatus::error;
  host.lanes[5].error = batch::LaneError::invalid_mutation;
  host.lanes[6].status = batch::LaneStatus::empty;
  upload(source.arena, host);
  check(batch::checkpoint_async(snapshot.arena, source.arena, n));
  equal(download(snapshot.arena, n), host, "snapshot mismatch");
  const std::vector<batch::Branch> specs = {
      {0, batch::no_mutation, 0, 0, 100}, {0, 0, 0x12345678, 0, 101},
      {1, (stride - 4) & ~3U, 0xff01aa55, 0, 102}, {n, batch::no_mutation, 0, 0, 103},
      {4, batch::no_mutation, 0, 0, 104}, {5, batch::no_mutation, 0, 0, 105},
      {6, batch::no_mutation, 0, 0, 106}, {7, 1, 12, 0, 107},
      {7, stride - 1, 0, 0, 108}};
  Device<batch::Branch> device_specs(n);
  check(cudaMemcpy(device_specs.data, specs.data(), n * sizeof(batch::Branch),
                   cudaMemcpyHostToDevice));
  HostArena expected(n, stride);
  host_branch(expected, host, specs);
  check(batch::branch_async(destination.arena, source.arena, device_specs.data, n, n));
  equal(download(destination.arena, n), expected, "branch mismatch");
  equal(download(source.arena, n), host, "branch changed parent");
  equal(download(snapshot.arena, n), host, "branch changed checkpoint");
  require(expected.states[stride] != expected.states[0], "mutation missing");

  require(batch::branch_async(source.arena, source.arena, device_specs.data, n, n) ==
              cudaErrorInvalidValue, "branch alias accepted");
  require(batch::checkpoint_async(source.arena, source.arena, n) ==
              cudaErrorInvalidValue, "checkpoint alias accepted");
  require(batch::branch_async(destination.arena, source.arena, device_specs.data, n + 1, n) ==
              cudaErrorInvalidValue, "capacity overflow accepted");
  require(batch::branch_async(destination.arena, source.arena, device_specs.data, n, n + 1) ==
              cudaErrorInvalidValue, "parent count overflow accepted");
  auto overlapping = destination.arena;
  overlapping.states = source.arena.states + 1;
  require(batch::checkpoint_async(overlapping, source.arena, n) ==
              cudaErrorInvalidValue, "partial overlap accepted");
  auto different_stride = destination.arena;
  --different_stride.stride;
  require(batch::checkpoint_async(different_stride, source.arena, n) ==
              cudaErrorInvalidValue, "stride mismatch accepted");
  require(batch::branch_async(destination.arena, source.arena,
                              reinterpret_cast<batch::Branch*>(destination.arena.states),
                              n, n) == cudaErrorInvalidValue, "spec/output overlap accepted");

  std::vector<std::uint32_t> choices{2, 1, 0, n, batch::no_lane, 3, 4, 5, 6};
  Device<std::uint32_t> device_choices(n);
  check(cudaMemcpy(device_choices.data, choices.data(), n * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice));
  host_compact(expected, host, choices);
  check(batch::compact_async(destination.arena, source.arena, device_choices.data, n, n));
  equal(download(destination.arena, n), expected, "compaction mismatch");
  require(batch::compact_async(source.arena, source.arena, device_choices.data, n, n) ==
              cudaErrorInvalidValue, "compact alias accepted");
  check(batch::checkpoint_async(destination.arena, snapshot.arena, n));
  equal(download(destination.arena, n), host, "checkpoint restore mismatch");
  check(batch::branch_async(destination.arena, source.arena, nullptr, 0, n));
  check(batch::compact_async(destination.arena, source.arena, nullptr, 0, n));
}

void validate_rank() {
  for (const auto n : {1U, 7U, 255U, 256U, 257U, 511U, 512U, 513U,
                       1023U, 1024U, 1025U, 4095U, 4096U, 4097U, 10007U}) {
    HostArena host(n, 4);
    initialize(host);
    for (std::uint32_t i = 0; i < n; ++i) {
      host.lanes[i].objective = {{static_cast<std::int64_t>(i % 13) - 6,
                                  static_cast<std::int64_t>((i * 17) % 7),
                                  static_cast<std::int64_t>(i % 3),
                                  static_cast<std::int64_t>(i % 2)}};
      if (i % 71 == 9) host.lanes[i].status = batch::LaneStatus::terminal;
      if (i % 73 == 11) host.lanes[i].status = batch::LaneStatus::error;
      if (i % 67 == 7) host.lanes[i].status = batch::LaneStatus::empty;
      if (i % 61 == 6) host.lanes[i].error = batch::LaneError::invalid_mutation;
    }
    host.lanes.front().objective.keys[0] = std::numeric_limits<std::int64_t>::min();
    host.lanes.back().objective.keys[0] = std::numeric_limits<std::int64_t>::max();
    Buffer device(n, 4);
    upload(device.arena, host);
    const auto padded = batch::rank_capacity(n);
    Device<std::uint32_t> indices(padded);
    check(batch::rank_async(device.arena, n, indices.data, padded));
    std::vector<std::uint32_t> actual(padded);
    check(cudaMemcpy(actual.data(), indices.data, padded * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost));
    const auto expected = host_rank(host);
    require(std::equal(expected.begin(), expected.end(), actual.begin()), "ranking mismatch");
    require(std::all_of(actual.begin() + n, actual.end(), [](auto i) {
      return i == batch::no_lane;
    }), "rank padding not sentinel");
    require(batch::rank_async(device.arena, n, indices.data, padded - 1) ==
                cudaErrorInvalidValue, "rank scratch overflow accepted");
    require(batch::rank_async(device.arena, n,
                              reinterpret_cast<std::uint32_t*>(device.arena.states), padded) ==
                cudaErrorInvalidValue, "rank/source alias accepted");
    check(batch::rank_async(device.arena, 0, nullptr, 0));
  }
}

// Synthetic integer scoring exercises device scheduling and branch independence.
// It is intentionally not an Irisu physics/search throughput measurement.
__host__ __device__ void evaluate_lane(unsigned char* state, batch::Lane& lane) {
  if (lane.status != batch::LaneStatus::ready || lane.error != batch::LaneError::none) return;
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 64; ++i) value = value * 1664525U + state[i] + 1013904223U;
  state[63] ^= static_cast<unsigned char>(value >> 24);
  lane.objective.keys[0] += value % 251;
  lane.objective.keys[1] = value % 17;
  lane.objective.keys[2] = value % 11;
  ++lane.objective.keys[3];
}

__global__ void evaluate(batch::Arena arena, std::uint32_t count) {
  const auto lane = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (lane < count)
    evaluate_lane(arena.states + std::size_t{lane} * arena.stride, arena.lanes[lane]);
}

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto n = samples.size();
  return n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
}

struct BenchResult {
  double cpu_ms, gpu_wall_ms, gpu_event_ms, graph_wall_ms, graph_event_ms;
  std::vector<double> cpu_samples, gpu_samples, gpu_events, graph_samples, graph_events;
  std::size_t allocated_bytes;
};

BenchResult benchmark(std::uint32_t parents, std::uint32_t fanout,
                       std::uint32_t stride, unsigned depths, unsigned repeats) {
  const auto count = parents * fanout;
  HostArena initial(parents, stride);
  initialize(initial);
  std::vector<batch::Branch> specs(count);
  for (std::uint32_t i = 0; i < count; ++i)
    specs[i] = {i / fanout, 0, (i + 1) * 2654435761U, 0, i};
  Buffer source(parents, stride), destination(count, stride);
  const auto padded = batch::rank_capacity(count);
  Device<batch::Branch> device_specs(count);
  Device<std::uint32_t> indices(padded);
  check(cudaMemcpy(device_specs.data, specs.data(), count * sizeof(batch::Branch),
                   cudaMemcpyHostToDevice));
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto pipeline = [&] {
    for (unsigned depth = 0; depth < depths; ++depth) {
      check(batch::branch_async(destination.arena, source.arena, device_specs.data,
                                count, parents, stream));
      evaluate<<<(count + 255) / 256, 256, 0, stream>>>(destination.arena, count);
      check(cudaGetLastError());
      check(batch::rank_async(destination.arena, count, indices.data, padded, stream));
      check(batch::compact_async(source.arena, destination.arena, indices.data,
                                 parents, count, stream));
    }
  };
  HostArena expected = initial, host_destination(count, stride);
  auto cpu_pipeline = [&] {
    for (unsigned depth = 0; depth < depths; ++depth) {
      host_branch(host_destination, expected, specs);
      #pragma omp parallel for schedule(static)
      for (std::uint32_t i = 0; i < count; ++i)
        evaluate_lane(host_destination.states.data() + std::size_t{i} * stride,
                       host_destination.lanes[i]);
      host_compact(expected, host_destination, host_rank(host_destination));
    }
  };

  upload(source.arena, initial);
  pipeline();
  check(cudaStreamSynchronize(stream));
  cpu_pipeline();
  const auto reference = expected;
  equal(download(source.arena, parents), reference, "resident pipeline parity mismatch");

  upload(source.arena, initial);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  pipeline();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  check(cudaGraphLaunch(executable, stream));
  check(cudaStreamSynchronize(stream));
  equal(download(source.arena, parents), reference, "graph pipeline parity mismatch");

  cudaEvent_t start_event{}, end_event{};
  check(cudaEventCreate(&start_event));
  check(cudaEventCreate(&end_event));
  BenchResult result{};
  for (unsigned repeat = 0; repeat < repeats; ++repeat) {
    expected = initial;
    auto start = Clock::now();
    cpu_pipeline();
    result.cpu_samples.push_back(milliseconds(start));

    for (unsigned mode = 0; mode < 2; ++mode) {
      upload(source.arena, initial);
      start = Clock::now();
      check(cudaEventRecord(start_event, stream));
      if (mode) check(cudaGraphLaunch(executable, stream)); else pipeline();
      check(cudaEventRecord(end_event, stream));
      check(cudaEventSynchronize(end_event));
      const auto wall = milliseconds(start);
      float elapsed{};
      check(cudaEventElapsedTime(&elapsed, start_event, end_event));
      (mode ? result.graph_samples : result.gpu_samples).push_back(wall);
      (mode ? result.graph_events : result.gpu_events).push_back(elapsed);
    }
  }
  equal(download(source.arena, parents), reference, "repeated pipeline mismatch");
  result.cpu_ms = median(result.cpu_samples);
  result.gpu_wall_ms = median(result.gpu_samples);
  result.gpu_event_ms = median(result.gpu_events);
  result.graph_wall_ms = median(result.graph_samples);
  result.graph_event_ms = median(result.graph_events);
  result.allocated_bytes = std::size_t{parents + count} * (stride + sizeof(batch::Lane)) +
                           std::size_t{count} * sizeof(batch::Branch) +
                           std::size_t{padded} * sizeof(std::uint32_t);
  check(cudaEventDestroy(start_event));
  check(cudaEventDestroy(end_event));
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  return result;
}

void samples(const std::vector<double>& values) {
  std::printf("[");
  for (std::size_t i = 0; i < values.size(); ++i)
    std::printf("%s%.6f", i ? "," : "", values[i]);
  std::printf("]");
}

int main(int argc, char** argv) {
  try {
    std::uint32_t parents = 256, fanout = 4, stride = 65536;
    unsigned depths = 16, repeats = 5, cpu_threads = 1;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      const auto value = std::stoull(argv[++i]);
      require(value && value <= UINT32_MAX, "option out of range");
      if (option == "--parents") parents = static_cast<std::uint32_t>(value);
      else if (option == "--fanout") fanout = static_cast<std::uint32_t>(value);
      else if (option == "--stride") stride = static_cast<std::uint32_t>(value);
      else if (option == "--depths") depths = static_cast<unsigned>(value);
      else if (option == "--repeats") repeats = static_cast<unsigned>(value);
      else if (option == "--cpu-threads") cpu_threads = static_cast<unsigned>(value);
      else throw std::runtime_error("unknown option");
    }
    require(stride >= 64 && parents <= (1U << 20) / fanout, "invalid workload size");
    require(depths <= 1024 && repeats <= 101, "workload exceeds bounded limit");
    require(cpu_threads <= 256, "CPU thread count exceeds bounded limit");
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(cpu_threads));
    validate_copy(69);
    validate_copy(256);
    validate_rank();
    const auto result = benchmark(parents, fanout, stride, depths, repeats);
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    std::printf("{\"schema\":\"irisu-cuda-batch-v1\",\"scope\":\"checkpoint-copy-ranking-synthetic-integer-score-only\","
                "\"physics_parity\":false,\"promotable\":false,\"validation\":\"passed\","
                "\"gpu\":\"%s\",\"compute_capability\":\"%d.%d\",\"parents\":%u,\"fanout\":%u,"
                "\"state_bytes\":%u,\"depths\":%u,\"repeats\":%u,\"cpu_threads\":%u,\"allocated_bytes\":%zu,"
                "\"median_cpu_ms\":%.6f,\"median_gpu_wall_ms\":%.6f,\"median_gpu_event_ms\":%.6f,"
                "\"median_graph_wall_ms\":%.6f,\"median_graph_event_ms\":%.6f,"
                "\"graph_copy_branches_per_second\":%.3f,\"cpu_samples_ms\":",
                device.name, device.major, device.minor, parents, fanout, stride, depths, repeats, cpu_threads,
                result.allocated_bytes, result.cpu_ms, result.gpu_wall_ms, result.gpu_event_ms,
                result.graph_wall_ms, result.graph_event_ms,
                static_cast<double>(parents) * fanout * depths * 1000 / result.graph_wall_ms);
    samples(result.cpu_samples);
    std::printf(",\"gpu_samples_ms\":"); samples(result.gpu_samples);
    std::printf(",\"gpu_event_samples_ms\":"); samples(result.gpu_events);
    std::printf(",\"graph_samples_ms\":"); samples(result.graph_samples);
    std::printf(",\"graph_event_samples_ms\":"); samples(result.graph_events);
    std::printf("}\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "batch validation failed: %s\n", error.what());
    return 1;
  }
}
