#include "x87_numeric.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace num = irisu::cuda_exact;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct TrigOutput { num::TrigPair candidate; float native_sine, native_cosine; };
struct ArithmeticInput { double a, b; };
struct ArithmeticOutput { num::Number value[5]; };

__host__ __device__ unsigned boundary_checks() {
  unsigned mask = 0;
  mask |= !num::sincos_reloaded_float(0.1).supported ? 1U : 0;
  mask |= num::sincos_reloaded_float(0.125).supported ? 2U : 0;
  mask |= !num::sincos_reloaded_float(0x1p19).supported ? 4U : 0;
  mask |= num::sincos_reloaded_float(-0.0).supported ? 8U : 0;
  mask |= !num::mul(0x1p1023, 2).supported ? 16U : 0;
  mask |= !num::mul(0x1p-1022, 0.5).supported ? 32U : 0;
  mask |= !num::div(1, 0).supported ? 64U : 0;
  mask |= !num::square_root(-1).supported ? 128U : 0;
  return mask;
}

__global__ void boundary_kernel(unsigned* output) { *output = boundary_checks(); }

__global__ void trig_kernel(const std::uint32_t* keys, TrigOutput* outputs,
                            std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float x = __uint_as_float(keys[i]);
  outputs[i].candidate = num::sincos_float(x);
  sincosf(x, &outputs[i].native_sine, &outputs[i].native_cosine);
}

__global__ void arithmetic_kernel(const ArithmeticInput* inputs,
                                  ArithmeticOutput* outputs, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto input = inputs[i];
  outputs[i].value[0] = num::add(input.a, input.b);
  outputs[i].value[1] = num::sub(input.a, input.b);
  outputs[i].value[2] = num::mul(input.a, input.b);
  outputs[i].value[3] = num::div(input.a, input.b);
  outputs[i].value[4] = num::square_root(input.a);
}

std::array<float, 2> oracle_trig(float x) {
  std::array<float, 2> result;
  asm volatile("flds %2; fsin; fstps %0; flds %2; fcos; fstps %1"
               : "=m"(result[0]), "=m"(result[1]) : "m"(x) : "st");
  return result;
}

double oracle_arithmetic(unsigned op, double a, double b) {
  double r;
  switch (op) {
    case 0: asm volatile("fldl %1; faddl %2; fstpl %0" : "=m"(r) : "m"(a), "m"(b) : "st"); break;
    case 1: asm volatile("fldl %1; fsubl %2; fstpl %0" : "=m"(r) : "m"(a), "m"(b) : "st"); break;
    case 2: asm volatile("fldl %1; fmull %2; fstpl %0" : "=m"(r) : "m"(a), "m"(b) : "st"); break;
    case 3: asm volatile("fldl %1; fdivl %2; fstpl %0" : "=m"(r) : "m"(a), "m"(b) : "st"); break;
    default: asm volatile("fldl %1; fsqrt; fstpl %0" : "=m"(r) : "m"(a) : "st"); break;
  }
  return r;
}

template<class T> T* device_alloc(std::size_t count) {
  T* result;
  check(cudaMalloc(&result, count * sizeof(T)));
  return result;
}

int main(int argc, char** argv) try {
  std::size_t count = 1000000;
  int repeats = 7;
  const bool trig_only = argc > 1 && std::string(argv[1]) == "--trig-sweep";
  std::uint32_t sweep_start = 0;
  if (trig_only) {
    if (argc != 4) throw std::runtime_error("usage: arithmetic_validate --trig-sweep START COUNT");
    sweep_start = static_cast<std::uint32_t>(std::stoull(argv[2], nullptr, 0));
    count = std::stoull(argv[3]); repeats = 1;
  }
  if (!trig_only && argc > 1) count = std::stoull(argv[1]);
  if (!trig_only && argc > 2) repeats = std::stoi(argv[2]);
  if (count < 100 || count > 100000000 || repeats < 1 || repeats > 100)
    throw std::runtime_error("usage: arithmetic_validate [samples 100..100000000] [repeats 1..100]");
  std::fenv_t saved_environment;
  std::fegetenv(&saved_environment);
  std::fesetround(FE_TONEAREST);
  const std::uint16_t control = 0x027f;
  asm volatile("fldcw %0" :: "m"(control));
  std::uint16_t observed;
  asm volatile("fnstcw %0" : "=m"(observed));
  if (observed != control) throw std::runtime_error("PC53 control word not installed");
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 7) throw std::runtime_error("requires CUDA compute capability >=7");
  auto* device_boundary = device_alloc<unsigned>(1);
  boundary_kernel<<<1,1>>>(device_boundary);
  unsigned boundary = 0;
  check(cudaMemcpy(&boundary, device_boundary, sizeof(boundary), cudaMemcpyDeviceToHost));
  check(cudaFree(device_boundary));
  if (boundary != 255 || boundary_checks() != 255)
    throw std::runtime_error("a host/device numeric support boundary check failed");
  std::mt19937_64 rng(0x8743753);
  std::vector<std::uint32_t> keys = {
      0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
      0x33000000, 0x39800000, 0x4096cbe3, 0x4096cbe4, 0x4096cbe5,
      0x4116cbe4, 0x46199998, 0xc6199998, 0x48ffffff, 0xc8ffffff,
      0x49000000, 0xc9000000, 0x5f000000, 0x7f7fffff, 0x7f800000,
      0xff800000, 0x7fc00000, 0x7f800001};
  // Half uniform raw keys exercise rejection; half sample the bounded domain.
  while (keys.size() < count) {
    const auto bits = static_cast<std::uint32_t>(rng());
    keys.push_back(keys.size() & 1 ? bits : (bits & 0x80000000U) | ((bits & 0x7fffffffU) % 0x49000000U));
  }
  if (trig_only) {
    for(std::size_t i=0;i<count;++i) keys[i] = sweep_start+static_cast<std::uint32_t>(i);
  }
  std::vector<ArithmeticInput> inputs = {
      {0, 0}, {-0.0, 1}, {1, -1}, {1, 0}, {0x1p1023, 2}, {0x1p-1022, 0.5},
      {0x1p-1074, 1}, {0x1p-1022, -0x1.fffffffffffffp-1023},
      {0x1.fffffffffffffp1023, 2}, {1, 0x1p-53}, {1, 0x1.8p-52}};
  const std::size_t arithmetic_count = trig_only ? 0 : count;
  while (inputs.size() < arithmetic_count) {
    const double a = std::bit_cast<double>(rng());
    const double b = std::bit_cast<double>(rng());
    inputs.push_back({a,b});
  }
  auto* device_keys = device_alloc<std::uint32_t>(count);
  auto* device_trig = device_alloc<TrigOutput>(count);
  auto* device_inputs = device_alloc<ArithmeticInput>(arithmetic_count ? arithmetic_count : 1);
  auto* device_outputs = device_alloc<ArithmeticOutput>(arithmetic_count ? arithmetic_count : 1);
  check(cudaMemcpy(device_keys, keys.data(), count*sizeof(keys[0]), cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_inputs, inputs.data(), arithmetic_count*sizeof(inputs[0]), cudaMemcpyHostToDevice));
  const unsigned blocks = static_cast<unsigned>((count + 255)/256);
  trig_kernel<<<blocks, 256>>>(device_keys, device_trig, count);
  if (arithmetic_count) arithmetic_kernel<<<blocks, 256>>>(device_inputs, device_outputs, arithmetic_count);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  std::vector<TrigOutput> trig(count);
  std::vector<ArithmeticOutput> arithmetic(arithmetic_count);
  check(cudaMemcpy(trig.data(), device_trig, count*sizeof(trig[0]), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(arithmetic.data(), device_outputs, arithmetic_count*sizeof(arithmetic[0]), cudaMemcpyDeviceToHost));
  std::size_t trig_supported=0, trig_rejected=0, trig_mismatches=0, native_mismatches=0;
  std::size_t host_mismatches=0;
  std::vector<std::array<std::uint32_t, 5>> trig_samples;
  const auto cpu_start = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<count; ++i) {
    const auto candidate = trig[i].candidate;
    if (!candidate.supported) { ++trig_rejected; continue; }
    ++trig_supported;
    const float x = std::bit_cast<float>(keys[i]);
    const auto reference = oracle_trig(x);
    const auto host = num::sincos_float(x);
    const auto s = std::bit_cast<std::uint32_t>(reference[0]);
    const auto c = std::bit_cast<std::uint32_t>(reference[1]);
    const auto ds = std::bit_cast<std::uint32_t>(candidate.sine);
    const auto dc = std::bit_cast<std::uint32_t>(candidate.cosine);
    if (s != ds || c != dc) {
      ++trig_mismatches;
      if (trig_samples.size()<8) trig_samples.push_back({keys[i],s,ds,c,dc});
    }
    if (s != std::bit_cast<std::uint32_t>(host.sine) || c != std::bit_cast<std::uint32_t>(host.cosine)) ++host_mismatches;
    if (s != std::bit_cast<std::uint32_t>(trig[i].native_sine) || c != std::bit_cast<std::uint32_t>(trig[i].native_cosine)) ++native_mismatches;
  }
  const double cpu_compare_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-cpu_start).count();
  std::array<std::size_t,5> accepted{}, rejected{}, mismatches{};
  for (std::size_t i=0; i<arithmetic_count; ++i) for (unsigned op=0; op<5; ++op) {
    const auto candidate = arithmetic[i].value[op];
    if (!candidate.supported) { ++rejected[op]; continue; }
    ++accepted[op];
    const double reference = oracle_arithmetic(op,inputs[i].a,inputs[i].b);
    if (std::bit_cast<std::uint64_t>(candidate.value) != std::bit_cast<std::uint64_t>(reference)) ++mismatches[op];
  }
  std::vector<float> elapsed;
  cudaEvent_t start, end;
  check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
  for (int repeat=0; repeat<repeats; ++repeat) {
    check(cudaEventRecord(start));
    trig_kernel<<<blocks, 256>>>(device_keys, device_trig, count);
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms,start,end)); elapsed.push_back(ms);
  }
  std::sort(elapsed.begin(),elapsed.end());
  std::cout << std::setprecision(10) << "{\n  \"schema\": 1,\n  \"scope\": \"numeric_boundary_only_not_physics_parity\",\n"
            << "  \"gpu\": \"" << properties.name << "\",\n  \"compute_capability\": \"" << properties.major << '.' << properties.minor << "\",\n"
            << "  \"samples\": " << count << ",\n  \"boundary_checks_passed\": 8,\n  \"seed\": 141834067,\n  \"control_word\": " << observed << ",\n"
            << "  \"trig\": {\"supported\": " << trig_supported << ", \"rejected\": " << trig_rejected
            << ", \"gpu_mismatches\": " << trig_mismatches << ", \"host_mismatches\": " << host_mismatches
            << ", \"cuda_libm_mismatches_on_supported\": " << native_mismatches
            << ", \"comparison_seconds_including_host_helper\": " << cpu_compare_seconds << ", \"counterexamples\": [";
  for(std::size_t i=0;i<trig_samples.size();++i) {
    if(i)std::cout << ',';
    std::cout << '[';
    for(unsigned j=0;j<5;++j){if(j)std::cout << ',';std::cout << trig_samples[i][j];}
    std::cout << ']';
  }
  std::cout << "]},\n  \"contiguous_range\": ";
  if(trig_only) std::cout << "{\"start\":" << sweep_start << ",\"count\":" << count << "}";
  else std::cout << "null";
  std::cout << ",\n  \"arithmetic\": {";
  const char* names[] = {"add","sub","mul","div","sqrt"};
  for(unsigned op=0;op<5;++op) {
    if(op)std::cout << ',';
    std::cout << '\"' << names[op] << "\": {\"supported\": " << accepted[op] << ", \"rejected\": " << rejected[op] << ", \"mismatches\": " << mismatches[op] << '}';
  }
  std::cout << "},\n  \"warmed_combined_trig_kernel_ms\": [";
  for(std::size_t i=0;i<elapsed.size();++i){if(i)std::cout << ',';std::cout << elapsed[i];}
  std::cout << "],\n  \"allocated_device_bytes\": " << count*(sizeof(std::uint32_t)+sizeof(TrigOutput))+arithmetic_count*(sizeof(ArithmeticInput)+sizeof(ArithmeticOutput))
            << ",\n  \"pass\": " << ((trig_mismatches==0 && host_mismatches==0 && std::all_of(mismatches.begin(),mismatches.end(),[](auto n){return n==0;})) ? "true" : "false") << "\n}\n";
  check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
  check(cudaFree(device_keys)); check(cudaFree(device_trig));
  check(cudaFree(device_inputs)); check(cudaFree(device_outputs));
  std::fesetenv(&saved_environment);
  return trig_mismatches || host_mismatches || std::any_of(mismatches.begin(),mismatches.end(),[](auto n){return n!=0;}) ? 1:0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n'; return 2;
}
