#include "../../clone/cuda_native/game_world.cuh"
#include "../../clone/cuda_native/compact_snapshot.cuh"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
namespace p=irisu::cuda_native;namespace g=irisu::cuda_game;
namespace {
std::vector<unsigned char> read(const char* path){std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open replay/digest");return {(std::istreambuf_iterator<char>(file)),{}};}
unsigned word(const std::vector<unsigned char>& bytes,std::size_t offset){return bytes[offset]|(unsigned(bytes[offset+1])<<8)|(unsigned(bytes[offset+2])<<16)|(unsigned(bytes[offset+3])<<24);}
void compare(const irisu::game_digest::Frame& frame,const std::vector<unsigned char>& native,unsigned tick){
  if(std::memcmp(&frame,native.data()+std::size_t(tick)*sizeof(frame),sizeof(frame))){std::fprintf(stderr,"native frame mismatch tick%u\n",tick);throw std::runtime_error("typed capacity replay differs from actual-native Frame");}
}
void capacity_rejection(){
  auto world=std::make_unique<p::World>();world->bodies[0].inv_mass=world->bodies[1].inv_mass=1;
  for(unsigned i=0;i<2;++i){world->shapes[i].occupied=true;world->shapes[i].body=i;world->shapes[i].category_bits=1;world->shapes[i].mask_bits=0xffff;}
  p::WorldPairCallbacks callback{world.get()};
  // Invoke the admitted allocator directly with distinct retained slots; pair
  // deduplication belongs to SAP, while this unit checks its capacity boundary.
  for(unsigned i=0;i<p::max_contacts;++i)if(callback.PairAdded(1,2)!=i+1)throw std::runtime_error("contact allocator rejected before configured capacity");
  if(callback.PairAdded(1,2)||world->status!=p::Status::capacity||world->contact_count!=p::max_contacts)
    throw std::runtime_error("contact allocator did not reject overflow explicitly");
}
}
int main(int argc,char** argv){try{
  if(argc!=3)throw std::runtime_error("usage: contact-capacity-validate replay.rpy actual-native.digest");
  const auto replay=read(argv[1]),native=read(argv[2]);
  if(replay.size()<56||(replay.size()-52)%4||word(replay,16))throw std::runtime_error("invalid normal replay");
  const unsigned ticks=(replay.size()-52)/4;
  if(native.size()!=(std::size_t(ticks)+1)*sizeof(irisu::game_digest::Frame))throw std::runtime_error("native digest frame count mismatch");
  std::fesetenv(FE_DFL_ENV);const std::uint16_t control=0x027f;asm volatile("fldcw %0"::"m"(control));
  capacity_rejection();auto game=std::make_unique<g::NativeGameWorld>();auto scene=std::make_unique<p::ScenePipelineState>();
  game->initialize(scene.get(),word(replay,0));compare(game->frame(),native,0);
  unsigned contact_peak=0,pair_peak=0,pending_boundaries=0,contact_tick=0;
  const auto observe=[&](unsigned tick){
    if(scene->world.contact_count>contact_peak){contact_peak=scene->world.contact_count;contact_tick=tick;}
    if(unsigned(scene->broadphase.m_pairManager.m_pairCount)>pair_peak)pair_peak=scene->broadphase.m_pairManager.m_pairCount;
    pending_boundaries+=scene->world.destroy_head!=p::null_index;
  };
  observe(0);
  for(unsigned tick=0;tick<ticks;++tick){const auto step=game->step(g::native_replay_action(word(replay,52+4*tick),tick));
    if(game->failed()){std::fprintf(stderr,"failed tick%u status%u contacts%u\n",tick,unsigned(scene->world.status),scene->world.contact_count);throw std::runtime_error("typed capacity replay failed");}
    compare(game->frame(step),native,tick+1);observe(tick+1);
  }
  std::printf("{\"pass\":true,\"backend\":\"checked-typed-host\",\"contact_capacity\":%u,\"native_SAP_pair_capacity\":4096,\"game_contact_vector_capacity\":%zu,\"actual_native_frame_checks\":%u,\"typed_boundary_contact_peak\":%u,\"first_contact_peak_tick\":%u,\"typed_boundary_SAP_pair_peak\":%u,\"pending_delete_boundaries\":%u,\"peak_scope\":\"typed call-boundary occupancy; not native hidden or transient allocation high-water\",\"capacity_overflow_rejected\":true,\"world_bytes\":%zu,\"scene_bytes\":%zu,\"scratch_bytes\":%zu,\"physical_prefix_bytes\":%zu,\"game_bytes\":%zu,\"performance_claim\":false}\n",
    p::max_contacts,g::vector<g::Contact>::capacity,ticks+1,contact_peak,contact_tick,pair_peak,pending_boundaries,
    sizeof(p::World),sizeof(p::ScenePipelineState),sizeof(p::SceneScratch),offsetof(p::ScenePipelineState,scratch),sizeof(g::NativeGameWorld));
  return 0;
}catch(const std::exception& error){std::fprintf(stderr,"contact-capacity-validate: %s\n",error.what());return 1;}}
