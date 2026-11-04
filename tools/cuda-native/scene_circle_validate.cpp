#include "scene_native.hpp"
#include "scene_wire.hpp"
#include "../../clone/cuda_native/world_pipeline.hpp"
#include <cfenv>
#include <fstream>
#include <memory>

using namespace irisu::cuda_native;
int main(int argc,char** argv){try{
  if(argc!=2&&argc!=3)throw std::runtime_error("usage: scene-circle-validate pinned-native-library [GPU-packet-file]");
  std::fesetround(FE_TONEAREST);const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
  scene_oracle::Api api(argv[1]);auto world=std::make_unique<World>();auto expected=std::make_unique<World>();
  auto scratch=std::make_unique<SceneScratch>();unsigned frames{};
  std::ofstream packet;
  if(argc==3){packet.open(argv[2],std::ios::binary);if(!packet)throw std::runtime_error("cannot open scene packet");}
  const auto write=[&](std::uint32_t value){packet.write(reinterpret_cast<const char*>(&value),4);};
  if(packet){write(scene_wire_magic);write(scene_wire_version);write(8);write(64);}
  for(unsigned scene=0;scene<8;++scene){
    void* handle=api.create(-1e6f,-1e6f,1e6f,1e6f,scene%2?-0.25f:0,1);
    if(!handle)throw std::runtime_error("native circle world creation failed");
    const auto fixed=api.circle(handle,2,0,0,scene%4<2?0:1,.5f,0);
    const auto moving=api.circle(handle,1,scene%3==0?.1f:0,2.8f,1,.75f,0);
    if(!fixed||!moving)throw std::runtime_error("native circle creation failed");
    const auto mapping=scene_oracle::import_scene(*world,handle);
    if(world->contact_count!=1)throw std::runtime_error("native initial pair set unexpected");
    auto initial=encode_scene(*world);
    if(packet){write(initial.size());for(auto word:initial)write(word);}
    for(unsigned tick=0;tick<64;++tick){
      const float dt=.016f;
      const auto result=step_fixed_circle_pairs(*world,*scratch,dt,8);
      if(!result){std::fprintf(stderr,"scene%u tick%u typed status%u\n",scene,tick,unsigned(result.status));throw std::runtime_error("typed circle scene step rejected");}
      api.step(handle,dt,8);scene_oracle::import_scene(*expected,handle,tick+1);
      const auto actual_words=encode_scene(*world),expected_words=encode_scene(*expected);
      if(actual_words!=expected_words){
        for(unsigned i=0;i<actual_words.size()&&i<expected_words.size();++i)if(actual_words[i]!=expected_words[i]){
          std::fprintf(stderr,"scene%u tick%u word%u typed%08x native%08x\n",scene,tick,i,actual_words[i],expected_words[i]);break;
        }
        throw std::runtime_error("independent native circle step mismatch");
      }
      if(packet){union{float f;std::uint32_t u;}bits{dt};write(bits.u);write(8);const auto hash=scene_hash(*expected);write(std::uint32_t(hash));write(std::uint32_t(hash>>32));}
      ++frames;
    }
    api.destroy(handle);
  }
  if(packet&&!packet.good())throw std::runtime_error("scene packet write failed");
  std::printf("{\"pass\":true,\"circle_scenes\":8,\"steps_each\":64,\"native_fullstep_frames\":%u,\"all_semantic_words_equal\":true,\"pair_topology\":\"fixed-imported-native-pair\",\"fallback_calls\":0,\"full_broadphase_backend\":false}\n",frames);
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"scene-circle-validate: %s\n",e.what());return 1;}}
