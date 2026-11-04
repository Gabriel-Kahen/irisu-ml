#include "scene_native.hpp"
#include "factory_wire.hpp"
#include "../../clone/cuda_native/factory.hpp"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>

using namespace irisu::cuda_native;
namespace {
using Input=FactoryInput;
unsigned comparisons{},cases{};
void describe(const Input& s){
  std::fprintf(stderr,"kind%u width%a height%a x%a y%a angle%a density%a friction%a restitution%a scale%a\n",
    s.kind,s.width,s.height,s.x,s.y,s.angle,s.density,s.friction,s.restitution,s.scale);
}
template<class T>void equal(const char* field,const T& actual,const T& native,const Input& input){
  ++comparisons;if(std::memcmp(&actual,&native,sizeof(T))==0)return;
  const auto* a=reinterpret_cast<const unsigned char*>(&actual);const auto* n=reinterpret_cast<const unsigned char*>(&native);
  for(unsigned i=0;i<sizeof(T);i+=4){std::uint32_t x{},y{};std::memcpy(&x,a+i,sizeof(T)-i<4?sizeof(T)-i:4);std::memcpy(&y,n+i,sizeof(T)-i<4?sizeof(T)-i:4);
    if(x!=y){describe(input);std::fprintf(stderr,"field%s word%u typed%08x native%08x\n",field,i/4,x,y);break;}}
  throw std::runtime_error("native cold factory mismatch");
}
void compare(const BodyState& b,const ShapeState& s,const BodyState& nb,const ShapeState& ns,const Input& input){
#define B(x) equal("body." #x,b.x,nb.x,input)
  B(flags);B(position);B(rotation);B(R);B(position0);B(rotation0);B(velocity);B(angular_velocity);
  B(force);B(torque);B(center);B(mass);B(inv_mass);B(inertia);B(inv_i);B(linear_damping);B(angular_damping);B(sleep_time);B(user_data);
#undef B
#define S(x) equal("shape." #x,s.x,ns.x,input)
  S(kind);S(R);S(position);S(friction);S(restitution);S(max_radius);S(category_bits);S(mask_bits);S(group_index);S(user_data);
  if(s.kind==ShapeKind::circle){S(local_position);S(radius);}
  else {S(local_centroid);S(vertex_count);S(local_obb_R);S(local_obb_center);S(local_obb_extents);
    for(unsigned i=0;i<s.vertex_count;++i){equal("shape.vertices",s.vertices[i],ns.vertices[i],input);equal("shape.normals",s.normals[i],ns.normals[i],input);equal("shape.core_vertices",s.core_vertices[i],ns.core_vertices[i],input);}}
#undef S
}
std::uint32_t random(std::uint32_t& r){r^=r<<13;r^=r>>17;r^=r<<5;return r;}
}
int main(int argc,char** argv){try{
  if(argc!=2&&argc!=3)throw std::runtime_error("usage: factory-validate pinned-native-library [GPU-packets.bin]");
  std::fesetround(FE_TONEAREST);const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
  scene_oracle::Api api(argv[1]);auto expected=std::make_unique<World>();std::uint32_t rng=0xaea3c15d;
  std::ofstream packet;if(argc==3){packet.open(argv[2],std::ios::binary);if(!packet)throw std::runtime_error("cannot open factory packets");}
  const auto write=[&](std::uint32_t value){packet.write(reinterpret_cast<const char*>(&value),4);};
  if(packet){write(factory_magic);write(factory_version);write(1536);write(factory_case_words);}
  for(unsigned kind=0;kind<3;++kind)for(unsigned test=0;test<512;++test){
    Input in{kind,12+float(random(rng)%5000)*.01f,12+float(random(rng)%5000)*.01f,
      float(int(random(rng)%100001)-50000)*.01f,float(int(random(rng)%100001)-50000)*.01f,
      float(int(random(rng)%20001)-10000)*.001f,test%7?float(random(rng)%800)*.01f:0,
      float(random(rng)%400)*.01f,float(random(rng)%200)*.01f,test%2?30.f:1.f};
    if(test<4){in.width=30;in.height=30;in.x=test%2?-0.f:0.f;in.y=test%2?0.f:-0.f;in.angle=test%2?-0.f:0.f;in.density=test%2?1:0;}
    void* world=api.create(-1e6f,-1e6f,1e6f,1e6f,0,in.scale);if(!world)throw std::runtime_error("native factory world failed");
    void* actor=kind==0?api.circle(world,in.width,in.x,in.y,in.density,in.friction,in.restitution):
      (kind==1?api.box:api.triangle)(world,in.width,in.height,in.x,in.y,in.angle,in.density,in.friction,in.restitution);
    if(!actor)throw std::runtime_error("native factory actor failed");
    const auto map=scene_oracle::import_scene(*expected,world);const auto slot=scene_oracle::Mapping::index(map.bodies,reinterpret_cast<std::uintptr_t>(actor));
    const auto shape=expected->bodies[slot].shape_head;if(shape==null_index)throw std::runtime_error("native factory shape absent");
    BodyDefinition bd{};ShapeDefinition sd{};auto result=wrapper_definition(bd,sd,kind,in.width,in.height,in.x,in.y,in.angle,in.density,in.friction,in.restitution,in.scale);
    if(!result){describe(in);throw std::runtime_error("wrapper definition rejected");}
    BodyState body{};ShapeState typed_shape{};result=construct_body_shape(body,typed_shape,bd,sd);
    if(!result){describe(in);throw std::runtime_error("typed factory rejected");}
    compare(body,typed_shape,expected->bodies[slot],expected->shapes[shape],in);++cases;
    if(packet){std::uint32_t words[10];std::memcpy(words,&in,sizeof(in));for(auto word:words)write(word);
      const auto hash=factory_hash(expected->bodies[slot],expected->shapes[shape]);write(std::uint32_t(hash));write(std::uint32_t(hash>>32));}
    api.destroy(world);
  }
  if(packet&&!packet.good())throw std::runtime_error("factory packet write failed");
  std::printf("{\"pass\":true,\"native_wrapper_factory_cases\":%u,\"hidden_field_comparisons\":%u,\"circle_box_triangle\":true,\"consumed_and_active_core_geometry\":true,\"proxy_allocation_compared\":false}\n",cases,comparisons);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"factory-validate: %s\n",e.what());return 1;}}
