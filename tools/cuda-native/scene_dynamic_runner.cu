#include "scene_dynamic_wire.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace irisu::cuda_native;
namespace {
struct Frame {std::uint64_t hash{};std::uint32_t status{};};
__global__ void initialize(ScenePipelineState* scenes,SceneActors* actors,const SceneCase* cases,unsigned count,unsigned cases_count,std::uint32_t* status){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)status[i]=std::uint32_t(initialize_scene_case(scenes[i],actors[i],cases[i%cases_count]).status);
}
template<bool trace>__global__ void advance(ScenePipelineState* scenes,SceneActors* actors,const SceneCase* cases,
  unsigned count,unsigned cases_count,unsigned ticks,float dt,unsigned iterations,Frame* frames,std::uint32_t* status){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  StageResult result{Status(status[i])};
  for(unsigned tick=0;tick<ticks;++tick){
    if(result)result=apply_scene_ops(scenes[i],actors[i],cases[i%cases_count],tick);
    if(result)result=scenes[i].step(dt,iterations);
    if constexpr(trace)frames[std::size_t(i)*ticks+tick]={dynamic_scene_hash(scenes[i]),std::uint32_t(result.status)};
  }
  status[i]=std::uint32_t(result.status);
}
__global__ void hash_final(const ScenePipelineState* scenes,const std::uint32_t* status,Frame* output,unsigned count){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]={dynamic_scene_hash(scenes[i]),status[i]};
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
struct Memory {
  std::vector<void*> data;~Memory(){for(auto p:data)cudaFree(p);}
  template<class T>T* allocate(std::size_t count){T* p{};check(cudaMalloc(&p,count*sizeof(T)));data.push_back(p);return p;}
};
struct Event {cudaEvent_t event{};Event(){check(cudaEventCreate(&event));}~Event(){cudaEventDestroy(event);}void record(){check(cudaEventRecord(event));}};
float elapsed(const Event& a,const Event& b){float value{};check(cudaEventElapsedTime(&value,a.event,b.event));return value;}
using Clock=std::chrono::steady_clock;
}
int main(int argc,char** argv){try{
  const auto process_start=Clock::now();
  if(argc<2)throw std::runtime_error("usage: scene-dynamic-runner packets.bin [--benchmark --worlds N --warmups N --repeats N]");
  bool benchmark=false;unsigned worlds=0,warmups=1,repeats=3;
  for(int i=2;i<argc;++i){const std::string flag=argv[i];
    if(flag=="--benchmark")benchmark=true;
    else if(flag=="--worlds"&&i+1<argc)worlds=std::stoul(argv[++i]);
    else if(flag=="--warmups"&&i+1<argc)warmups=std::stoul(argv[++i]);
    else if(flag=="--repeats"&&i+1<argc)repeats=std::stoul(argv[++i]);
    else throw std::runtime_error("unknown dynamic runner argument");
  }
  std::ifstream file(argv[1],std::ios::binary);if(!file)throw std::runtime_error("cannot open dynamic packets");
  const auto read=[&](){std::uint32_t w{};if(!file.read(reinterpret_cast<char*>(&w),4))throw std::runtime_error("truncated dynamic packet");return w;};
  if(read()!=dynamic_magic||read()!=dynamic_version)throw std::runtime_error("dynamic packet identity mismatch");
  const unsigned count=read(),ticks=read();if(read()!=dynamic_case_words)throw std::runtime_error("dynamic case stride mismatch");
  union{std::uint32_t u;float f;}dt{read()};const unsigned iterations=read();if(read())throw std::runtime_error("dynamic reserved field nonzero");
  if(!count||count>1024||!ticks||ticks>100000||std::uint64_t(count)*ticks>1000000||iterations>1000)throw std::runtime_error("unbounded dynamic corpus");
  std::vector<SceneCase> cases(count);std::vector<std::uint64_t> expected(std::size_t(count)*ticks);
  for(unsigned i=0;i<count;++i){std::uint32_t words[dynamic_case_words];for(auto& word:words)word=read();std::memcpy(&cases[i],words,sizeof(SceneCase));
    if(cases[i].actor_count>8||cases[i].op_count>16)throw std::runtime_error("dynamic case capacity exceeded");
    for(unsigned j=0;j<cases[i].op_count;++j)if(cases[i].ops[j].tick>=ticks||unsigned(cases[i].ops[j].kind)>3)throw std::runtime_error("dynamic operation invalid");
    for(unsigned j=0;j<ticks;++j){const auto lo=read(),hi=read();expected[std::size_t(i)*ticks+j]=std::uint64_t(lo)|(std::uint64_t(hi)<<32);}
  }
  if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing dynamic packet data");
  if(!worlds)worlds=count;if(!worlds||worlds>8192||warmups>10||!repeats||repeats>20)throw std::runtime_error("unbounded dynamic benchmark");
  if(!benchmark){worlds=count;warmups=0;repeats=1;}
  check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));Memory memory;
  auto* device_cases=memory.allocate<SceneCase>(count);auto* scenes=memory.allocate<ScenePipelineState>(worlds);
  auto* actors=memory.allocate<SceneActors>(worlds);auto* status=memory.allocate<std::uint32_t>(worlds);
  auto* frames=memory.allocate<Frame>(benchmark?worlds:std::size_t(worlds)*ticks);
  check(cudaMemcpy(device_cases,cases.data(),cases.size()*sizeof(SceneCase),cudaMemcpyHostToDevice));
  std::vector<Frame> result(benchmark?worlds:std::size_t(worlds)*ticks);
  std::vector<float> kernel_samples,setup_samples;std::vector<double> wall_samples;
  for(unsigned run=0;run<warmups+repeats;++run){
    Event init_before,init_after,before,after;init_before.record();
    initialize<<<(worlds+31)/32,32>>>(scenes,actors,device_cases,worlds,count,status);check(cudaGetLastError());
    init_after.record();check(cudaEventSynchronize(init_after.event));
    const auto begin=Clock::now();before.record();
    if(benchmark)advance<false><<<(worlds+127)/128,128>>>(scenes,actors,device_cases,worlds,count,ticks,dt.f,iterations,nullptr,status);
    else advance<true><<<(worlds+31)/32,32>>>(scenes,actors,device_cases,worlds,count,ticks,dt.f,iterations,frames,status);
    check(cudaGetLastError());after.record();check(cudaEventSynchronize(after.event));
    const auto end=Clock::now();
    if(benchmark){hash_final<<<(worlds+31)/32,32>>>(scenes,status,frames,worlds);check(cudaGetLastError());check(cudaDeviceSynchronize());}
    check(cudaMemcpy(result.data(),frames,result.size()*sizeof(Frame),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<result.size();++i){const auto world=benchmark?i:i/ticks,tick=benchmark?ticks-1:i%ticks;
      const auto wanted=expected[std::size_t(world%count)*ticks+tick];
      if(result[i].status||result[i].hash!=wanted){std::fprintf(stderr,"world%u tick%u status%u gpu%016llx native%016llx\n",world,tick,result[i].status,
          static_cast<unsigned long long>(result[i].hash),static_cast<unsigned long long>(wanted));throw std::runtime_error("native dynamic GPU scene mismatch");}
    }
    if(run>=warmups){kernel_samples.push_back(elapsed(before,after));setup_samples.push_back(elapsed(init_before,init_after));wall_samples.push_back(std::chrono::duration<double,std::milli>(end-begin).count());}
  }
  cudaFuncAttributes a{};if(benchmark)check(cudaFuncGetAttributes(&a,advance<false>));else check(cudaFuncGetAttributes(&a,advance<true>));
  std::printf("{\"pass\":true,\"benchmark\":%s,\"worlds\":%u,\"ticks_each\":%u,\"attempted_world_steps\":%llu,\"all_native_hashes_equal\":true,\"trace_every_tick\":%s,\"warmups\":%u,\"repeats\":%u,\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"scene_bytes\":%zu,\"rollout_kernel_ms\":[",benchmark?"true":"false",worlds,ticks,static_cast<unsigned long long>(worlds)*ticks,benchmark?"false":"true",warmups,repeats,a.numRegs,a.localSizeBytes,sizeof(ScenePipelineState));
  for(unsigned i=0;i<kernel_samples.size();++i)std::printf("%s%.6f",i?",":"",kernel_samples[i]);
  std::printf("],\"resident_launch_sync_ms\":[");for(unsigned i=0;i<wall_samples.size();++i)std::printf("%s%.6f",i?",":"",wall_samples[i]);
  std::printf("],\"cold_scene_setup_kernel_ms\":[");for(unsigned i=0;i<setup_samples.size();++i)std::printf("%s%.6f",i?",":"",setup_samples[i]);
  std::printf("],\"process_total_ms\":%.6f,\"timing_scope\":\"resident sequential physics+recorded lifecycle operations; setup separately reported; benchmark final hash/download outside rollout timer\",\"fallback_calls\":0}\n",std::chrono::duration<double,std::milli>(Clock::now()-process_start).count());return 0;
}catch(const std::exception& e){std::fprintf(stderr,"scene-dynamic-runner: %s\n",e.what());return 1;}}
