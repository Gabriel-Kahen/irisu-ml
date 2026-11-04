// Actual independent GPU circle-world qualifier. No Guest or CPU physics in
// the device path; native-created snapshots enter only before the first step.
#include "scene_wire.hpp"
#include "../../clone/cuda_native/world_pipeline.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace irisu::cuda_native;
namespace {
struct Step {float dt;std::uint32_t iterations;};
struct Frame {std::uint64_t hash;std::uint32_t status;};
__global__ void scenes(World* worlds,SceneScratch* scratch,const Step* steps,Frame* output,unsigned count,unsigned ticks){
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=count)return;
  StageResult result{};
  for(unsigned tick=0;tick<ticks;++tick){
    const auto index=std::size_t(lane)*ticks+tick;
    if(result)result=step_fixed_circle_pairs(worlds[lane],scratch[lane],steps[index].dt,steps[index].iterations);
    output[index]={scene_hash(worlds[lane]),std::uint32_t(result.status)};
  }
}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Memory {
  std::vector<void*> allocations;
  ~Memory(){for(auto p:allocations)cudaFree(p);}
  template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,count*sizeof(T)));allocations.push_back(p);return p;}
};
}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("usage: scene-runner native-circle-packets.bin");
  std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("cannot open scene packets");
  const auto read=[&](){std::uint32_t w{};if(!input.read(reinterpret_cast<char*>(&w),4))throw std::runtime_error("truncated scene packet");return w;};
  if(read()!=scene_wire_magic||read()!=scene_wire_version)throw std::runtime_error("scene wire identity mismatch");
  const unsigned count=read(),ticks=read();
  if(!count||count>1024||!ticks||ticks>100000||std::uint64_t(count)*ticks>1000000)throw std::runtime_error("unbounded scene workload");
  auto worlds=std::make_unique<World[]>(count);std::vector<Step> steps(std::size_t(count)*ticks);
  std::vector<std::uint64_t> expected(steps.size());
  for(unsigned lane=0;lane<count;++lane){
    const auto size=read();if(size>1000000)throw std::runtime_error("oversize scene snapshot");
    std::vector<std::uint32_t> words(size);for(auto& w:words)w=read();decode_scene(worlds[lane],words);
    for(unsigned tick=0;tick<ticks;++tick){
      const auto index=std::size_t(lane)*ticks+tick;union{std::uint32_t u;float f;}bits{read()};
      steps[index]={bits.f,read()};if(steps[index].iterations>1000)throw std::runtime_error("unbounded solver iterations");
      const auto low=read(),high=read();expected[index]=std::uint64_t(low)|(std::uint64_t(high)<<32);
    }
  }
  if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing scene packet data");
  check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));Memory memory;
  auto* gpu_worlds=memory.allocate<World>(count);auto* scratch=memory.allocate<SceneScratch>(count);
  auto* gpu_steps=memory.allocate<Step>(steps.size());auto* gpu_frames=memory.allocate<Frame>(steps.size());
  check(cudaMemcpy(gpu_worlds,worlds.get(),std::size_t(count)*sizeof(World),cudaMemcpyHostToDevice));
  check(cudaMemset(scratch,0,std::size_t(count)*sizeof(SceneScratch)));
  check(cudaMemcpy(gpu_steps,steps.data(),steps.size()*sizeof(Step),cudaMemcpyHostToDevice));
  cudaEvent_t before{},after{};check(cudaEventCreate(&before));check(cudaEventCreate(&after));
  check(cudaEventRecord(before));scenes<<<(count+31)/32,32>>>(gpu_worlds,scratch,gpu_steps,gpu_frames,count,ticks);
  check(cudaGetLastError());check(cudaEventRecord(after));check(cudaEventSynchronize(after));
  float elapsed{};check(cudaEventElapsedTime(&elapsed,before,after));cudaEventDestroy(before);cudaEventDestroy(after);
  std::vector<Frame> actual(expected.size());check(cudaMemcpy(actual.data(),gpu_frames,actual.size()*sizeof(Frame),cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<actual.size();++i)if(actual[i].status||actual[i].hash!=expected[i]){
    std::fprintf(stderr,"scene%u tick%u status%u gpu%016llx native%016llx\n",i/ticks,i%ticks,actual[i].status,
      static_cast<unsigned long long>(actual[i].hash),static_cast<unsigned long long>(expected[i]));
    throw std::runtime_error("native full-step GPU scene mismatch");
  }
  cudaFuncAttributes attributes{};check(cudaFuncGetAttributes(&attributes,scenes));
  std::printf("{\"pass\":true,\"backend\":\"independent-typed-cuda\",\"scenes\":%u,\"ticks_each\":%u,\"native_fullstep_frames\":%zu,\"all_semantic_hashes_equal\":true,\"fallback_calls\":0,\"instrumented_kernel_ms\":%.6f,\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"bytes_per_world\":%zu,\"scratch_bytes_per_world\":%zu,\"fixed_pair_topology_only\":true,\"performance_benchmark\":false}\n",
    count,ticks,actual.size(),elapsed,attributes.numRegs,attributes.localSizeBytes,sizeof(World),sizeof(SceneScratch));
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"scene-runner: %s\n",e.what());return 1;}}
