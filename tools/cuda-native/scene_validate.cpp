// Independent shape-free scene differential against the pinned native ELF32.
// Build this oracle in32-bit mode; no Guest/translated dispatcher is involved.
#include "../../clone/cuda_native/world.hpp"
#include "../../clone/cuda_native/world_shapes.hpp"
#include "oracle_native.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace irisu::cuda_native;
namespace {
struct NativeBody {
  std::uint32_t flags;
  Vec2 position;float rotation;Mat22 R;Vec2 position0;float rotation0;
  Vec2 velocity;float angular_velocity;Vec2 force;float torque;Vec2 center;
  std::uint32_t world,prev,next,shape_list;std::int32_t shape_count;
  std::uint32_t joint_list,contact_list;
  float mass,inv_mass,inertia,inv_i,linear_damping,angular_damping,sleep_time;
  std::uint32_t user_data;
};
static_assert(sizeof(void*)==4,"native oracle requires -m32");
static_assert(sizeof(NativeBody)==136);
static_assert(offsetof(NativeBody,inv_mass)==108);
static_assert(offsetof(NativeBody,sleep_time)==128);
struct Island {void* allocator;NativeBody** bodies;void* contacts;void* joints;
  std::int32_t body_count,joint_count,contact_count,body_capacity,contact_capacity,joint_capacity;};
static_assert(sizeof(Island)==40);
struct Api {
  NativeOracle pinned{};
  void* library{};std::uintptr_t base{};
  void* (*create)(float,float,float,float,float,float){};
  void (*destroy)(void*){};void (*step)(void*,float,int){};void (*destroy_body)(void*,void*){};
  using DefCtor=void(__attribute__((thiscall))*)(void*);
  using CreateBody=NativeBody*(__attribute__((thiscall))*)(void*,const void*);
  using Sleep=void(__attribute__((thiscall))*)(Island*,float);
  using Solve=void(__attribute__((thiscall))*)(Island*,const TimeStep*,const Vec2*);
  using QuickSync=void(__attribute__((thiscall))*)(void*,const Vec2*,const Mat22*);
  DefCtor definition{};CreateBody body{};Sleep sleep{};Solve solve{};
  QuickSync quick_circle{},quick_poly{};
  explicit Api(const char* path){
    pinned=oracle_open(path);library=pinned.library;
    create=reinterpret_cast<decltype(create)>(dlsym(library,"b2d_world_create"));
    destroy=reinterpret_cast<decltype(destroy)>(dlsym(library,"b2d_world_destroy"));
    step=reinterpret_cast<decltype(step)>(dlsym(library,"b2d_world_step"));
    destroy_body=reinterpret_cast<decltype(destroy_body)>(dlsym(library,"b2d_world_destroy_body"));
    void* anchor=dlsym(library,"msvc_b2d_world_create");
    if(!create||!destroy||!step||!destroy_body||!anchor)throw std::runtime_error("native API unavailable");
    Dl_info info{};if(!dladdr(anchor,&info))throw std::runtime_error("native image base unavailable");
    base=reinterpret_cast<std::uintptr_t>(info.dli_fbase);
    if(reinterpret_cast<std::uintptr_t>(anchor)-base!=0x14a30)throw std::runtime_error("native image entrypoint mismatch");
    definition=reinterpret_cast<DefCtor>(base+79600);
    body=reinterpret_cast<CreateBody>(base+79904);
    sleep=reinterpret_cast<Sleep>(base+43520);solve=reinterpret_cast<Solve>(base+42640);
    quick_circle=reinterpret_cast<QuickSync>(base+72320);quick_poly=reinterpret_cast<QuickSync>(base+72544);
  }
  ~Api(){if(library)oracle_close(&pinned);}
};
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
std::uint32_t random_word(std::uint32_t& state){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
float value(std::uint32_t& state,float scale){return float(int(random_word(state)%20001)-10000)*scale;}
void set(NativeBody& n,const BodyState& b){
#define COPY(x) n.x=b.x
  COPY(flags);COPY(position);COPY(rotation);COPY(R);COPY(position0);COPY(rotation0);COPY(velocity);
  COPY(angular_velocity);COPY(force);COPY(torque);COPY(center);COPY(mass);COPY(inv_mass);COPY(inertia);
  COPY(inv_i);COPY(linear_damping);COPY(angular_damping);COPY(sleep_time);COPY(user_data);
#undef COPY
}
unsigned comparisons{};
void equal(const NativeBody& n,const BodyState& b,unsigned scene,unsigned tick){
  const auto check=[&](const char* field,const auto& a,const auto& c){
    ++comparisons;if(std::memcmp(&a,&c,sizeof(a))){
      std::fprintf(stderr,"scene%u tick%u body%u field%s differs\n",scene,tick,b.id,field);
      throw std::runtime_error("native hidden body mismatch");
    }
  };
#define CHECK(x) check(#x,n.x,b.x)
  CHECK(flags);CHECK(position);CHECK(rotation);CHECK(R);CHECK(position0);CHECK(rotation0);CHECK(velocity);
  CHECK(angular_velocity);CHECK(force);CHECK(torque);CHECK(center);CHECK(mass);CHECK(inv_mass);CHECK(inertia);
  CHECK(inv_i);CHECK(linear_damping);CHECK(angular_damping);CHECK(sleep_time);CHECK(user_data);
#undef CHECK
}
BodyState sample(std::uint32_t& rng,unsigned index){
  BodyState b{};b.position={value(rng,.01f),value(rng,.01f)};b.rotation=value(rng,.001f);
  b.velocity={value(rng,.001f),value(rng,.001f)};b.angular_velocity=value(rng,.001f);
  b.inv_mass=index%7?float(index%5+1)*.125f:0;b.inv_i=b.inv_mass*.25f;
  b.mass=b.inv_mass?1/b.inv_mass:0;b.inertia=b.inv_i?1/b.inv_i:0;
  b.force={value(rng,.001f),value(rng,.001f)};b.torque=value(rng,.001f);
  b.linear_damping=float(random_word(rng)%5)*.25f;b.angular_damping=float(random_word(rng)%5)*.25f;
  b.flags=(b.inv_mass?0U:static_body)|(index%3?allow_sleep:0U);b.user_data=index+16;
  b.position0={value(rng,.01f),value(rng,.01f)};b.rotation0=value(rng,.001f);
  require(bool(rotation_matrix(b)),"sample rotation unsupported");return b;
}
void scene_gate(Api& api){
  std::uint32_t rng=0x3ea1c927;
  auto world=std::make_unique<World>();
  for(unsigned scene=0;scene<128;++scene){
    const float gy=scene%4?value(rng,.001f):0;
    world->reset({0,gy});void* handle=api.create(-1e6f,-1e6f,1e6f,1e6f,gy,1);
    require(handle,"native world creation failed");void* native_world=*static_cast<void**>(handle);
    std::array<NativeBody*,8> native{};std::array<unsigned,8> slots{};
    const unsigned count=scene%8+1;
    for(unsigned i=0;i<count;++i){
      std::array<unsigned char,320> definition{};api.definition(definition.data());
      native[i]=api.body(native_world,definition.data());require(native[i],"native body creation failed");
      auto b=sample(rng,i+scene);require(bool(world->create_body(b,slots[i])),"typed body creation failed");set(*native[i],world->bodies[slots[i]]);
    }
    for(unsigned tick=0;tick<64;++tick){
      const float dt=tick%5==0?0:tick%3==0?0.03125f:0.01600000075995922f;
      api.step(handle,dt,8);require(bool(world->step_shape_free(dt)),"typed scene step failed");
      for(unsigned i=0;i<count;++i)if(native[i])equal(*native[i],world->bodies[slots[i]],scene,tick);
      if(tick==15){
        auto snapshot=std::make_unique<World>(*world);
        require(bool(snapshot->step_shape_free(dt)),"snapshot continuation failed");
        auto sibling=std::make_unique<World>();require(bool(sibling->restore(*world)),"snapshot restore failed");
        require(bool(sibling->step_shape_free(dt)),"sibling continuation failed");
        require(std::memcmp(snapshot.get(),sibling.get(),sizeof(World))==0,"snapshot continuation diverged");
      }
      if(tick==31&&count>1){api.destroy_body(handle,native[0]);native[0]=nullptr;require(bool(world->destroy(slots[0])),"typed deferred destroy failed");}
      if(tick==47&&count>1){
        std::array<unsigned char,320> definition{};api.definition(definition.data());
        native[0]=api.body(native_world,definition.data());require(native[0],"native replacement creation failed");
        const auto old_id=world->next_body_id;auto replacement=sample(rng,scene);
        require(bool(world->create_body(replacement,slots[0])),"typed replacement creation failed");
        require(world->bodies[slots[0]].id==old_id,"replacement stable ID reused");
        set(*native[0],world->bodies[slots[0]]);
      }
    }
    api.destroy(handle);
  }
}
void island_gate(Api& api){
  std::uint32_t rng=0x5fa20d83;
  for(unsigned test=0;test<2048;++test){
    const unsigned count=test%8+1;std::array<BodyState,8> bodies{};std::array<NativeBody,8> native{};
    std::array<NativeBody*,8> pointers{};std::array<unsigned,8> order{};
    for(unsigned i=0;i<count;++i){bodies[i]=sample(rng,i+test);bodies[i].id=i;
      if(test%3==0){bodies[i].velocity={.005f,.005f};bodies[i].angular_velocity=.005f;bodies[i].sleep_time=float(test%70)*.01f;}
      if(test%7==0){bodies[i].velocity={std::nextafter(.01f,test%2?0.f:1.f),0};bodies[i].angular_velocity=0;}
      set(native[i],bodies[i]);order[i]=count-i-1;pointers[i]=&native[count-i-1];}
    Island island{};island.bodies=pointers.data();island.body_count=count;
    const float dt=test%2?.016f:0;
    api.sleep(&island,dt);require(bool(update_sleep(bodies.data(),order.data(),count,dt)),"typed sleep rejected valid case");
    for(unsigned i=0;i<count;++i)equal(native[i],bodies[i],128+test,0);
  }
}
void boundary_gate(){
  auto world=std::make_unique<World>();BodyState b{};unsigned slot{};
  require(bool(world->create_body(b,slot)),"boundary fixture create failed");
  auto invalid=std::make_unique<World>(*world);invalid->body_head=max_bodies;
  require(world->restore(*invalid).status==Status::invalid_snapshot,"invalid snapshot accepted");
  require(world->body_head==slot,"invalid restore mutated destination");
  world->shape_count=1;
  require(world->step_shape_free(.016f).status==Status::bridge_required&&world->tick==0,"collision stage silently skipped");
  b.inv_mass=1;b.rotation=0x1p19f;
  require(rotation_matrix(b).status==Status::unsupported_numeric,"unsupported trig domain accepted");
}
void shape_gate(Api& api){
  std::uint32_t rng=0xf1ae37d9;
  for(unsigned test=0;test<4096;++test){
    std::array<unsigned char,128> raw{};raw.fill(0x55);ShapeState shape{};
    shape.kind=test%2?ShapeKind::circle:ShapeKind::polygon;
    shape.local_position=shape.local_centroid={value(rng,.001f),value(rng,.001f)};
    const Vec2 position{value(rng,.01f),value(rng,.01f)};
    const Mat22 R{{value(rng,.0001f),value(rng,.0001f)},{value(rng,.0001f),value(rng,.0001f)}};
    std::memcpy(raw.data()+64,&shape.local_position,sizeof(Vec2));
    const auto before=raw;
    (test%2?api.quick_circle:api.quick_poly)(raw.data(),&position,&R);
    require(bool(shape_quick_sync(shape,position,R)),"typed QuickSync failed");
    require(std::memcmp(raw.data()+8,&shape.R,sizeof(Mat22))==0,"QuickSync R differs");
    require(std::memcmp(raw.data()+24,&shape.position,sizeof(Vec2))==0,"QuickSync position differs");
    auto normalized=raw;std::memcpy(normalized.data()+8,before.data()+8,24);
    require(normalized==before,"native QuickSync unexpectedly wrote other shape fields");
  }
}
}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("usage: scene-validate pinned-native-library");
  std::fesetround(FE_TONEAREST);const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
  Api api(argv[1]);scene_gate(api);island_gate(api);shape_gate(api);boundary_gate();
  std::printf("{\"pass\":true,\"native_scenes\":128,\"steps_per_scene\":64,\"native_sleep_islands\":2048,\"shape_quick_sync_cases\":4096,\"hidden_field_comparisons\":%u,\"backend\":\"independent-typed-shape-free\",\"full_collision_backend\":false}\n",comparisons);
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"scene-validate: %s\n",e.what());return 1;}}
