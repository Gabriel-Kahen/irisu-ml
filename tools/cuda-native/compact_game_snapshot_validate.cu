// Independent full-copy versus compact-copy game continuation qualification.
// Host prefix preparation is explicit; no performance claim is made.
#include "../../clone/cuda_native/compact_game_snapshot.cuh"
#include <cuda_runtime.h>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace p=irisu::cuda_native;namespace c=p::compact_snapshot;namespace g=irisu::cuda_game;
using Game=c::Game;using Scene=p::ScenePipelineState;using Frame=irisu::game_digest::Frame;
namespace {
struct Result {
  unsigned failure{},phase{},offset{},snapshots{},native_frames{},warm{},pending{},sleeping{},frozen{};
  std::uint64_t game_bytes{},scene_bytes{},padding_ignored{};
};
__device__ bool compare_games(Game& a,Scene& sa,Game& b,Scene& sb,c::GamePlan& plan,Result& result,unsigned phase){
  if(!c::build_game_plan(a,plan)){result.failure=1;result.phase=phase;return false;}
  constexpr std::uintptr_t canonical=0x100000;
  bool valid=a.relocate_body_pointers(reinterpret_cast<std::uintptr_t>(a.simulator.bodies_.data()),canonical)&&
    b.relocate_body_pointers(reinterpret_cast<std::uintptr_t>(b.simulator.bodies_.data()),canonical);
  a.bind(nullptr);b.bind(nullptr);const auto offset=c::first_game_difference(a,b,plan,&result.padding_ignored);
  valid=a.rebind(&sa,canonical)&&b.rebind(&sb,canonical)&&valid;
  if(!valid||offset!=p::null_index){result.failure=1;result.phase=phase;result.offset=offset;return false;}
  return true;
}
__global__ void qualify(Game* initial,Scene* initial_scene,Game* full,Scene* full_scene,Game* compact,Scene* compact_scene,
                        std::uintptr_t host_base,const std::uint32_t* replay,const Frame* native,unsigned prefix,unsigned ticks,
                        bool virgin,c::GamePlan* game_plan,c::Plan* scene_plan,Result* result){
  __shared__ unsigned valid;
  if(!threadIdx.x){valid=initial->rebind(initial_scene,host_base);if(!valid){result->failure=1;result->phase=1;}}
  __syncthreads();if(!valid)return;
  for(unsigned tick=0;tick<ticks;++tick){
    if(!threadIdx.x){
      valid=c::build_game_plan(*initial,*game_plan)&&bool(c::build_plan(*initial_scene,*scene_plan));
      if(valid&&tick)valid=bool(c::clear_reachable(*compact_scene));
      if(!valid){result->failure=1;result->phase=2;}
      result->game_bytes+=game_plan->bytes;result->scene_bytes+=c::transport_bytes(*scene_plan);
    }
    __syncthreads();if(!valid)return;
    // Poison all omitted Game payloads before copy; pointer tails must then be
    // explicitly zero-restored by copy_game before existing rebind scans them.
    if(!virgin){
      auto* bytes=reinterpret_cast<unsigned char*>(compact);
      for(std::size_t i=threadIdx.x;i<sizeof(Game);i+=blockDim.x)bytes[i]=static_cast<unsigned char>(0xa5U+(i+tick*17U)%83U);
      auto* scratch=reinterpret_cast<unsigned char*>(&compact_scene->scratch);
      for(std::size_t i=threadIdx.x;i<sizeof(p::SceneScratch)/4;i+=blockDim.x){
        const unsigned value=0x7fc08000U|((unsigned(i)+tick*13U)&0x7fffU);
        for(unsigned byte=0;byte<4;++byte)scratch[4*i+byte]=static_cast<unsigned char>(value>>(8*byte));
      }
    }
    __syncthreads();
    c::copy_bytes(reinterpret_cast<unsigned char*>(full),reinterpret_cast<const unsigned char*>(initial),sizeof(Game),threadIdx.x,blockDim.x);
    c::copy_bytes(reinterpret_cast<unsigned char*>(full_scene),reinterpret_cast<const unsigned char*>(initial_scene),sizeof(Scene),threadIdx.x,blockDim.x);
    if(!tick)c::clear_missing(*compact_scene,*scene_plan,threadIdx.x,blockDim.x);
    c::copy_active(*compact_scene,*initial_scene,*scene_plan,threadIdx.x,blockDim.x);
    c::copy_game(*compact,*initial,*game_plan,threadIdx.x,blockDim.x);
    __syncthreads();
    if(!threadIdx.x){
      valid=full->rebind(full_scene,reinterpret_cast<std::uintptr_t>(initial->simulator.bodies_.data()))&&
        c::rebind_game(*compact,*compact_scene,*game_plan);
      if(!valid){result->failure=1;result->phase=3;}
      const auto before=c::first_difference(*full_scene,*compact_scene);
      if(valid&&before!=p::null_index){valid=0;result->failure=1;result->phase=4;result->offset=before;}
      if(valid)valid=compare_games(*full,*full_scene,*compact,*compact_scene,*game_plan,*result,5);
      if(valid){
        ++result->snapshots;result->pending+=full_scene->world.destroy_head!=p::null_index;
        for(auto contact=full_scene->world.contact_head;contact!=p::null_index;contact=full_scene->world.contacts[contact].next){
          const auto& entry=full_scene->world.contacts[contact];
          if(entry.manifold_count)for(int i=0;i<entry.manifold.point_count;++i)
            result->warm+=entry.manifold.points[i].normal_impulse!=0||entry.manifold.points[i].tangent_impulse!=0;
        }
        const auto action=g::native_replay_action(replay[prefix+tick],prefix+tick);
        const auto first=full->step(action),second=compact->step(action);
        const auto a=full->frame(first),b=compact->frame(second);
        const auto* aa=reinterpret_cast<const unsigned char*>(&a);const auto* bb=reinterpret_cast<const unsigned char*>(&b);
        const auto* nn=reinterpret_cast<const unsigned char*>(native+tick);
        for(unsigned i=0;i<sizeof(Frame);++i)if(aa[i]!=bb[i]||aa[i]!=nn[i]){
          valid=0;result->failure=1;result->phase=6;result->offset=i;break;
        }
        if(full->failed()||compact->failed()){valid=0;result->failure=1;result->phase=7;}
        if(valid){++result->native_frames;const auto after=c::first_difference(*full_scene,*compact_scene);
          if(after!=p::null_index){valid=0;result->failure=1;result->phase=8;result->offset=after;}}
        if(valid)valid=compare_games(*full,*full_scene,*compact,*compact_scene,*game_plan,*result,9);
        for(auto body=full_scene->world.body_head;body!=p::null_index;body=full_scene->world.bodies[body].next){
          result->sleeping+=(full_scene->world.bodies[body].flags&p::sleeping_body)!=0;
          result->frozen+=(full_scene->world.bodies[body].flags&p::frozen_body)!=0;
        }
      }
    }
    __syncthreads();if(!valid)return;
    auto* old=initial;initial=full;full=old;auto* old_scene=initial_scene;initial_scene=full_scene;full_scene=old_scene;
  }
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
struct Memory {
  std::vector<void*> pointers;~Memory(){for(auto* pointer:pointers)cudaFree(pointer);}
  template<class T>T* allocate(std::size_t count=1){T* pointer{};check(cudaMalloc(&pointer,count*sizeof(T)));pointers.push_back(pointer);return pointer;}
};
std::vector<unsigned char> read(const char* path){std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open input");return {(std::istreambuf_iterator<char>(file)),{}};}
unsigned word(const std::vector<unsigned char>& bytes,std::size_t at){return bytes[at]|(unsigned(bytes[at+1])<<8)|(unsigned(bytes[at+2])<<16)|(unsigned(bytes[at+3])<<24);}
template<class T>std::unique_ptr<T> empty_object(){auto* pointer=static_cast<T*>(::operator new(sizeof(T)));std::memset(pointer,0,sizeof(T));::new(static_cast<void*>(pointer))T();return std::unique_ptr<T>(pointer);}
}
int main(int argc,char** argv){try{
  if(argc<5||argc>6||(argc==6&&std::string(argv[5])!="--virgin"))throw std::runtime_error("usage: compact-game-snapshot-validate replay.rpy actual-native.digest prefix steps [--virgin]");
  const auto replay=read(argv[1]),native=read(argv[2]);const unsigned prefix=std::stoul(argv[3]),ticks=std::stoul(argv[4]);const bool virgin=argc==6;
  if(replay.size()<56||(replay.size()-52)%4||word(replay,16)||!ticks||ticks>4096||
     std::uint64_t(prefix)+ticks>(replay.size()-52)/4||native.size()%sizeof(Frame)||
     (std::uint64_t(prefix)+ticks+1)*sizeof(Frame)>native.size())throw std::runtime_error("invalid bounded replay/digest range");
  std::fesetenv(FE_DFL_ENV);const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));
  auto source=empty_object<Game>();auto scene=empty_object<Scene>();source->initialize(scene.get(),word(replay,0));
  std::vector<std::uint32_t> words(prefix+ticks);for(unsigned i=0;i<words.size();++i)words[i]=word(replay,52+4*i);
  for(unsigned i=0;i<prefix&&!source->failed();++i)source->step(g::native_replay_action(words[i],i));
  if(source->failed())throw std::runtime_error("typed host prefix failed");
  check(cudaDeviceSetLimit(cudaLimitStackSize,16U<<10));Memory memory;
  auto* current=memory.allocate<Game>();auto* full=memory.allocate<Game>();auto* compact=memory.allocate<Game>();
  auto* current_scene=memory.allocate<Scene>();auto* full_scene=memory.allocate<Scene>();auto* compact_scene=memory.allocate<Scene>();
  auto* game_plan=memory.allocate<c::GamePlan>();auto* scene_plan=memory.allocate<c::Plan>();auto* result=memory.allocate<Result>();
  auto* inputs=memory.allocate<std::uint32_t>(words.size());auto* expected=memory.allocate<Frame>(ticks);
  check(cudaMemcpy(current,source.get(),sizeof(Game),cudaMemcpyHostToDevice));check(cudaMemcpy(current_scene,scene.get(),sizeof(Scene),cudaMemcpyHostToDevice));
  // Match the runner's real cold storage. Initcheck tests virgin SceneScratch,
  // not uninitialized persistent padding consumed by representation checks.
  // Default poison mode still overwrites the whole compact Game each snapshot.
  check(cudaMemset(compact,0,sizeof(Game)));check(cudaMemset(compact_scene,0,offsetof(Scene,scratch)));
  check(cudaMemcpy(inputs,words.data(),words.size()*4,cudaMemcpyHostToDevice));
  check(cudaMemcpy(expected,native.data()+(std::size_t(prefix)+1)*sizeof(Frame),ticks*sizeof(Frame),cudaMemcpyHostToDevice));
  check(cudaMemset(result,0,sizeof(Result)));
  qualify<<<1,256>>>(current,current_scene,full,full_scene,compact,compact_scene,reinterpret_cast<std::uintptr_t>(source->simulator.bodies_.data()),
    inputs,expected,prefix,ticks,virgin,game_plan,scene_plan,result);check(cudaGetLastError());check(cudaDeviceSynchronize());
  Result observed{};check(cudaMemcpy(&observed,result,sizeof(observed),cudaMemcpyDeviceToHost));
  if(observed.failure||observed.snapshots!=ticks||observed.native_frames!=ticks){std::fprintf(stderr,"phase%u offset%u snapshots%u native_frames%u\n",observed.phase,observed.offset,observed.snapshots,observed.native_frames);throw std::runtime_error("compact game continuation mismatch");}
  std::printf("{\"pass\":true,\"performance_claim\":false,\"prefix_backend\":\"typed-host-checked\",\"prefix\":%u,\"snapshots\":%u,\"actual_native_frame_checks\":%u,\"virgin\":%s,\"virgin_scope\":\"%s\",\"all_persistent_storage_initialized_once\":true,\"warm_point_witnesses\":%u,\"pending_delete_snapshots\":%u,\"sleeping_body_witnesses\":%u,\"frozen_body_witnesses\":%u,\"game_transport_bytes_sum\":%llu,\"scene_transport_bytes_sum\":%llu,\"live_game_value_fields_and_active_scene_equal_before_after\":true,\"live_element_padding_mismatches_ignored\":%llu,\"pointer_tails_zero_restored\":true,\"inactive_payload_equality\":false,\"full_SAP_bytes_preserved\":true,\"fallback_calls\":0}\n",
    prefix,observed.snapshots,observed.native_frames,virgin?"true":"false",virgin?"SceneScratch only; all persistent Game/Scene storage initialized once":"none; Game and SceneScratch poisoned each snapshot",observed.warm,observed.pending,observed.sleeping,observed.frozen,
    static_cast<unsigned long long>(observed.game_bytes),static_cast<unsigned long long>(observed.scene_bytes),static_cast<unsigned long long>(observed.padding_ignored));
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"compact-game-snapshot-validate: %s\n",error.what());return 1;}}
