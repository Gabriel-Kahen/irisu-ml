// Checkpoint-segmented replay differential. Each GPU lane starts from a host
// translated checkpoint, validates every frame and the complete endpoint state.
// This is explicitly not a measurement of one continuous GPU replay.
#include "../../clone/cuda/world.cuh"
#include "image_identity.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace game=irisu::cuda_game;
using Frame=irisu::game_digest::Frame;
struct Endpoint { std::uint64_t initial_hash{},final_hash{},tick{};unsigned error{},game_failed{}; };

__global__ void segments(game::World* worlds,unsigned char* memory,unsigned capacity,
 const std::uint32_t* words,unsigned steps,unsigned stride,unsigned count,
 Frame* frames,Endpoint* endpoints) {
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=count)return;
  auto& world=worlds[lane];world.bind(memory+std::size_t(lane)*capacity);
  auto& result=endpoints[lane];result.initial_hash=world.state_hash(true);
  auto local_guest=world.guest;world.simulator.physics_.bind(&local_guest);
  const unsigned start=lane*stride,stop=start+stride<steps?start+stride:steps;
  if(!start)frames[0]=world.frame();
  for(unsigned tick=start;tick<stop&&!world.failed();++tick) {
    const auto step=world.step(game::replay_action(words[tick],tick));
    frames[tick+1]=world.frame(step);
  }
  world.guest=local_guest;world.bind(memory+std::size_t(lane)*capacity);
  result.final_hash=world.state_hash(true);result.tick=world.simulator.tick_;
  result.error=world.guest.error;result.game_failed=world.failed();
}
void check(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
std::vector<unsigned char> read(const char* path) {
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error(std::string("cannot read ")+path);
  return {(std::istreambuf_iterator<char>(input)),{}};
}
unsigned word(const std::vector<unsigned char>& bytes,std::size_t at) {
  return unsigned(bytes[at])|(unsigned(bytes[at+1])<<8)|(unsigned(bytes[at+2])<<16)|(unsigned(bytes[at+3])<<24);
}
std::uint64_t word64(const unsigned char* bytes) {
  std::uint64_t result=0;for(unsigned i=0;i<8;++i)result|=std::uint64_t(bytes[i])<<(8*i);return result;
}
void equal_frame(const Frame& actual,const Frame& expected,unsigned row,const char* backend) {
  const auto* a=reinterpret_cast<const unsigned char*>(&actual);
  const auto* b=reinterpret_cast<const unsigned char*>(&expected);
  const char* fields[]={"tick","bodies","observation","events","event_count","transition","body_count","score","gauge","level","highest_chain","canonical_score"};
  for(unsigned i=0;i<12;++i)if(word64(a+i*8)!=word64(b+i*8)) {
    std::fprintf(stderr,"%s row%u field%s expected%llx actual%llx\n",backend,row,fields[i],
      (unsigned long long)word64(b+i*8),(unsigned long long)word64(a+i*8));
    throw std::runtime_error("per-frame differential failed");
  }
}
int main(int argc,char** argv) {
 try {
  if(argc<4)throw std::runtime_error("usage: game-segments image.bin replay.rpy native.digest [--segment-ticks N] [--max-steps N] [--arena-mib N] [--threads N]");
  const auto image=read(argv[1]),replay=read(argv[2]),native=read(argv[3]);
  irisu::cuda_exact::verify_image(image.data(),image.size());
  if(image.empty()||image.size()>0x20000||replay.size()<56||(replay.size()-52)%4||word(replay,16))throw std::runtime_error("invalid image/replay");
  unsigned steps=(replay.size()-52)/4,stride=256,capacity=1024*1024,threads=4;
  for(int i=4;i<argc;++i) {
    const std::string arg=argv[i];if(i+1>=argc)throw std::runtime_error("missing argument");
    if(arg=="--segment-ticks")stride=std::stoul(argv[++i]);
    else if(arg=="--max-steps")steps=std::stoul(argv[++i]);
    else if(arg=="--arena-mib")capacity=std::stoul(argv[++i])*1024*1024;
    else if(arg=="--threads")threads=std::stoul(argv[++i]);
    else throw std::runtime_error("unknown argument");
  }
  if(!threads||threads>128||!steps||steps>(replay.size()-52)/4||!stride||capacity<0x40000||capacity>32*1024*1024||native.size()<std::size_t(steps+1)*96)throw std::runtime_error("invalid bounds/native trajectory");
  const unsigned count=(steps+stride-1)/stride;
  if(std::uint64_t(count)*(sizeof(game::World)+capacity)>16ULL*1024*1024*1024)throw std::runtime_error("checkpoint batch exceeds16GiB bound");
  std::vector<std::uint32_t> words(steps);for(unsigned i=0;i<steps;++i)words[i]=word(replay,52+4*i);
  std::vector<Frame> reference(steps+1),actual(steps+1);std::memcpy(reference.data(),native.data(),reference.size()*96);
  std::vector<game::World> checkpoints(count);std::vector<unsigned char> arenas(std::size_t(count)*capacity);
  std::vector<std::uint64_t> initial_hash(count),final_hash(count),expected_tick(count);
  auto host=std::make_unique<game::World>();std::vector<unsigned char> memory(capacity);
  const std::uint16_t cw=0x027f;__asm__ __volatile__("fldcw %0"::"m"(cw));
  const auto start=std::chrono::steady_clock::now();
  host->initialize(memory.data(),capacity,image.data(),image.size(),word(replay,0));
  if(host->failed())throw std::runtime_error("host checkpoint initialization failed");
  equal_frame(host->frame(),reference[0],0,"host");
  for(unsigned segment=0;segment<count;++segment) {
    std::memcpy(&checkpoints[segment],host.get(),sizeof(game::World));
    std::memcpy(arenas.data()+std::size_t(segment)*capacity,memory.data(),capacity);
    initial_hash[segment]=host->state_hash(true);
    const auto stop=std::min(steps,(segment+1)*stride);
    for(unsigned tick=segment*stride;tick<stop;++tick) {
      const auto result=host->step(game::replay_action(words[tick],tick));
      if(host->failed())throw std::runtime_error("host checkpoint trajectory failed");
      equal_frame(host->frame(result),reference[tick+1],tick+1,"host");
    }
    final_hash[segment]=host->state_hash(true);expected_tick[segment]=host->simulator.tick_;
  }
  const double host_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  game::World* device_worlds{};unsigned char* device_arenas{};std::uint32_t* device_words{};
  Frame* device_frames{};Endpoint* device_endpoints{};std::vector<Endpoint> endpoints(count);
  check(cudaDeviceSetLimit(cudaLimitStackSize,16*1024));
  check(cudaMalloc(&device_worlds,checkpoints.size()*sizeof(game::World)));
  check(cudaMalloc(&device_arenas,arenas.size()));check(cudaMalloc(&device_words,words.size()*4));
  check(cudaMalloc(&device_frames,actual.size()*sizeof(Frame)));check(cudaMalloc(&device_endpoints,count*sizeof(Endpoint)));
  check(cudaMemcpy(device_worlds,checkpoints.data(),checkpoints.size()*sizeof(game::World),cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_arenas,arenas.data(),arenas.size(),cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_words,words.data(),words.size()*4,cudaMemcpyHostToDevice));
  check(cudaMemset(device_frames,0,actual.size()*sizeof(Frame)));check(cudaMemset(device_endpoints,0,count*sizeof(Endpoint)));
  cudaEvent_t begin{},end{};check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaEventRecord(begin));
  segments<<<(count+threads-1)/threads,threads>>>(device_worlds,device_arenas,capacity,device_words,steps,stride,count,device_frames,device_endpoints);
  check(cudaGetLastError());check(cudaEventRecord(end));check(cudaEventSynchronize(end));float kernel_ms{};check(cudaEventElapsedTime(&kernel_ms,begin,end));
  check(cudaMemcpy(actual.data(),device_frames,actual.size()*sizeof(Frame),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(endpoints.data(),device_endpoints,count*sizeof(Endpoint),cudaMemcpyDeviceToHost));
  for(unsigned segment=0;segment<count;++segment) {
    const auto& endpoint=endpoints[segment];
    if(endpoint.initial_hash!=initial_hash[segment]||endpoint.final_hash!=final_hash[segment]||endpoint.tick!=expected_tick[segment]||endpoint.error||endpoint.game_failed) {
      std::fprintf(stderr,"segment%u endpoint mismatch initial%llx/%llx final%llx/%llx tick%llu error%u game%u\n",segment,
        (unsigned long long)endpoint.initial_hash,(unsigned long long)initial_hash[segment],(unsigned long long)endpoint.final_hash,(unsigned long long)final_hash[segment],
        (unsigned long long)endpoint.tick,endpoint.error,endpoint.game_failed);
      throw std::runtime_error("checkpoint endpoint differential failed");
    }
  }
  for(unsigned row=0;row<=steps;++row)equal_frame(actual[row],reference[row],row,"cuda");
  check(cudaFree(device_endpoints));check(cudaFree(device_frames));check(cudaFree(device_words));check(cudaFree(device_arenas));check(cudaFree(device_worlds));check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
  std::printf("{\"schema\":\"irisu-cuda-game-segments-v1\",\"status\":\"passed\",\"scope\":\"checkpoint-segmented GPU replay; every frame plus canonical hidden/complete arena endpoint hashes\",\"continuous_gpu_replay_observed\":false,\"exact_training_admitted\":false,\"replay_words\":%u,\"rows\":%u,\"segments\":%u,\"segment_ticks\":%u,\"arena_bytes\":%u,\"threads_per_block\":%u,\"host_checkpoint_seconds\":%.9f,\"gpu_kernel_ms\":%.6f,\"elapsed_seconds\":%.9f,\"final_tick\":%llu,\"canonical_score\":%lld,\"live_score\":%lld,\"level\":%u,\"chain\":%u}\n",
    steps,steps+1,count,stride,capacity,threads,host_seconds,kernel_ms,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),
    (unsigned long long)host->simulator.tick_,(long long)host->simulator.latest_final_score_,(long long)host->simulator.score_,host->simulator.level_,host->simulator.highest_chain_);
  return 0;
 }catch(const std::exception& error){std::fprintf(stderr,"game-segments: %s\n",error.what());return 1;}
}
