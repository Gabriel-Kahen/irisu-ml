// Independent typed physics + existing generated game rules; no guest image.
// Binary stdout uses the existing independent native game-digest-v1 oracle.
#include "../../clone/cuda_native/game_world.cuh"
#include "scene_wire.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace game=irisu::cuda_game;namespace physics=irisu::cuda_native;
using Frame=irisu::game_digest::Frame;
GAME_HD void trajectory(game::NativeGameWorld& world,const std::uint32_t* words,unsigned steps,Frame* frames){
  frames[0]=world.frame();
  for(unsigned i=0;i<steps&&!world.failed();++i)frames[i+1]=world.frame(world.step(game::native_replay_action(words[i],i)));
}
#ifdef __CUDACC__
__global__ void run(game::NativeGameWorld* world,physics::ScenePipelineState* scene,
  unsigned seed,const std::uint32_t* words,unsigned steps,Frame* frames){
  if(threadIdx.x||blockIdx.x)return;
  world->initialize(scene,seed);trajectory(*world,words,steps,frames);
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
#endif
std::uint32_t word(const std::vector<unsigned char>& bytes,std::size_t at){
  return bytes[at]|(unsigned(bytes[at+1])<<8)|(unsigned(bytes[at+2])<<16)|(unsigned(bytes[at+3])<<24);
}
int main(int argc,char** argv){try{
  if(argc<2)throw std::runtime_error("usage: native-game-runner replay.rpy [steps] [--gpu] [--clone-at N]");
  std::ifstream file(argv[1],std::ios::binary);if(!file)throw std::runtime_error("cannot open replay");
  const std::vector<unsigned char> replay((std::istreambuf_iterator<char>(file)),{});
  if(replay.size()<56||(replay.size()-52)%4||word(replay,16)!=0)throw std::runtime_error("invalid padded normal-mode replay");
  const unsigned available=(replay.size()-52)/4;unsigned steps=available,clone_at=~0U;bool gpu=false,clone_verified=false;
  for(int i=2;i<argc;++i){const std::string arg=argv[i];if(arg=="--gpu")gpu=true;
    else if(arg=="--clone-at"&&i+1<argc)clone_at=std::stoul(argv[++i]);else steps=std::stoul(argv[i]);}
  if(steps>available)throw std::runtime_error("requested steps exceed replay");
  if(clone_at!=~0U&&(gpu||clone_at>=steps))throw std::runtime_error("clone requires host and clone-at<steps");
  std::vector<std::uint32_t> words(steps);for(unsigned i=0;i<steps;++i)words[i]=word(replay,52+4*i);
  std::vector<Frame> frames(steps+1);auto world=std::make_unique<game::NativeGameWorld>();
  auto scene=std::make_unique<physics::ScenePipelineState>();double kernel_ms=0;
  const auto start=std::chrono::steady_clock::now();
  if(gpu){
#ifdef __CUDACC__
    // Empty object representations are input preparation, not host physics.
    // Keep giant aggregate constructors out of device compilation; all reset,
    // construction of physical bodies and replay execution still run on GPU.
    std::memset(world.get(),0,sizeof(*world));::new(static_cast<void*>(world.get()))game::NativeGameWorld();
    std::memset(scene.get(),0,sizeof(*scene));::new(static_cast<void*>(scene.get()))physics::ScenePipelineState();
    game::NativeGameWorld* dw{};physics::ScenePipelineState* ds{};std::uint32_t* di{};Frame* df{};
    check(cudaDeviceSetLimit(cudaLimitStackSize,16*1024));check(cudaMalloc(&dw,sizeof(*dw)));check(cudaMalloc(&ds,sizeof(*ds)));
    check(cudaMemcpy(dw,world.get(),sizeof(*dw),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,scene.get(),sizeof(*ds),cudaMemcpyHostToDevice));
    check(cudaMalloc(&di,std::max<std::size_t>(4,words.size()*4)));if(!words.empty())check(cudaMemcpy(di,words.data(),words.size()*4,cudaMemcpyHostToDevice));
    check(cudaMalloc(&df,frames.size()*sizeof(Frame)));check(cudaMemset(df,0,frames.size()*sizeof(Frame)));
    cudaEvent_t begin{},end{};check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaEventRecord(begin));
    run<<<1,1>>>(dw,ds,word(replay,0),di,steps,df);check(cudaGetLastError());check(cudaEventRecord(end));check(cudaEventSynchronize(end));
    float ms{};check(cudaEventElapsedTime(&ms,begin,end));kernel_ms=ms;
    check(cudaMemcpy(world.get(),dw,sizeof(*dw),cudaMemcpyDeviceToHost));check(cudaMemcpy(scene.get(),ds,sizeof(*ds),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(frames.data(),df,frames.size()*sizeof(Frame),cudaMemcpyDeviceToHost));
    const auto body_offset=reinterpret_cast<const unsigned char*>(world->simulator.bodies_.data())-reinterpret_cast<const unsigned char*>(world.get());
    if(!world->rebind(scene.get(),reinterpret_cast<std::uintptr_t>(dw)+body_offset))throw std::runtime_error("device game pointer relocation failed");
    check(cudaFree(df));check(cudaFree(di));check(cudaFree(ds));check(cudaFree(dw));check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
#else
    throw std::runtime_error("CUDA was not compiled into this host qualifier");
#endif
  }else{
    const std::uint16_t cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
    world->initialize(scene.get(),word(replay,0));frames[0]=world->frame();
    auto* path=std::getenv("IRISU_GAME_BODY_DUMP_FILE");auto* dump=path?std::fopen(path,"w"):nullptr;
    if(path&&!dump)throw std::runtime_error("cannot open body dump");
    std::unique_ptr<game::NativeGameWorld> clone;std::unique_ptr<physics::ScenePipelineState> cloned_scene;
    for(unsigned i=0;i<=steps&&!world->failed();++i){
      if(dump)for(const auto& b:world->simulator.bodies_)irisu::game_digest::dump_body(dump,world->simulator.tick_,b);
      if(i==clone_at){
        clone=std::make_unique<game::NativeGameWorld>();cloned_scene=std::make_unique<physics::ScenePipelineState>();
        std::memcpy(clone.get(),world.get(),sizeof(*world));std::memcpy(cloned_scene.get(),scene.get(),sizeof(*scene));
        if(!clone->rebind(cloned_scene.get(),reinterpret_cast<std::uintptr_t>(world->simulator.bodies_.data())))
          throw std::runtime_error("host game pointer relocation failed");
      }
      if(i<steps){
        frames[i+1]=world->frame(world->step(game::native_replay_action(words[i],i)));
        if(clone){const auto result=clone->step(game::native_replay_action(words[i],i));const auto frame=clone->frame(result);
          if(clone->failed()||std::memcmp(&frame,&frames[i+1],sizeof(Frame))||
             physics::scene_hash(scene->world)!=physics::scene_hash(cloned_scene->world)||
             std::memcmp(&scene->broadphase,&cloned_scene->broadphase,sizeof(scene->broadphase)))
            throw std::runtime_error("typed game clone/rebind suffix diverged");
          clone_verified=true;
        }
      }
    }
    if(dump)std::fclose(dump);
  }
  const auto& s=world->simulator;
  std::fprintf(stderr,"{\"backend\":\"%s\",\"steps\":%u,\"tick\":%llu,\"failed\":%s,\"physics_status\":%u,\"live_score\":%lld,\"recorded_final_score\":%lld,\"level\":%u,\"chain\":%u,\"clone_verified\":%s,\"kernel_ms_including_frame_hashes\":%.6f,\"elapsed_seconds\":%.9f,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"guest_calls\":0,\"fallback_calls\":0}\n",
    gpu?"cuda-native-full-game":"typed-host-full-game",steps,static_cast<unsigned long long>(s.tick_),world->failed()?"true":"false",
    unsigned(scene->world.status),static_cast<long long>(s.score_),static_cast<long long>(s.recorded_final_score_),s.level_,s.highest_chain_,clone_verified?"true":"false",kernel_ms,
    std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),sizeof(*world),sizeof(*scene));
  if(world->failed())return 1;
  for(const auto& frame:frames)if(std::fwrite(&frame,sizeof(frame),1,stdout)!=1)throw std::runtime_error("cannot write game digest");
  return std::fflush(stdout)?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"native-game-runner: %s\n",e.what());return 1;}}
