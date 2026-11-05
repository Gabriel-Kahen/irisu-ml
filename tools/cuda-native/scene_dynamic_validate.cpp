#include "scene_native.hpp"
#include "scene_dynamic_native.hpp"
#include "../../clone/cuda_native/world_scene.hpp"
#include <cfenv>
#include <memory>
#include <fstream>

using namespace irisu::cuda_native;
namespace {
unsigned scene_index{},tick_index{},comparisons{},frames{},frozen_origin_calls{},partial_proxy_creations{};
template<class T>void equal(const char* name,const T& a,const T& b){
  ++comparisons;if(!std::memcmp(&a,&b,sizeof(T)))return;
  for(unsigned i=0;i<sizeof(T);i+=4){std::uint32_t x{},y{};const auto n=sizeof(T)-i<4?sizeof(T)-i:4;
    std::memcpy(&x,reinterpret_cast<const char*>(&a)+i,n);std::memcpy(&y,reinterpret_cast<const char*>(&b)+i,n);
    if(x!=y){std::fprintf(stderr,"scene%u tick%u field%s word%u typed%08x native%08x\n",scene_index,tick_index,name,i/4,x,y);break;}}
  throw std::runtime_error("native dynamic scene mismatch");
}
void body_equal(const BodyState& a,const BodyState& b){
#define F(x) equal("body." #x,a.x,b.x)
  F(user_data);F(flags);F(position);F(rotation);F(R);F(position0);F(rotation0);F(velocity);F(angular_velocity);
  F(force);F(torque);F(center);F(mass);F(inv_mass);F(inertia);F(inv_i);F(linear_damping);F(angular_damping);F(sleep_time);
#undef F
}
void shape_equal(const ShapeState& a,const ShapeState& b){
#define F(x) equal("shape." #x,a.x,b.x)
  F(kind);F(R);F(position);F(friction);F(restitution);F(max_radius);F(proxy_id);F(category_bits);F(mask_bits);F(group_index);
  if(a.kind==ShapeKind::circle){F(local_position);F(radius);}
  else {F(local_centroid);F(vertex_count);F(local_obb_R);F(local_obb_center);F(local_obb_extents);
    for(unsigned i=0;i<a.vertex_count;++i){equal("shape.vertex",a.vertices[i],b.vertices[i]);equal("shape.normal",a.normals[i],b.normals[i]);equal("shape.core",a.core_vertices[i],b.core_vertices[i]);}}
#undef F
}
void compare_world(const World& a,const World& b){
  equal("body_count",a.body_count,b.body_count);equal("shape_count",a.shape_count,b.shape_count);equal("contact_count",a.contact_count,b.contact_count);
  auto body1=a.body_head,body2=b.body_head;
  while(body1!=null_index&&body2!=null_index){
    body_equal(a.bodies[body1],b.bodies[body2]);auto s1=a.bodies[body1].shape_head,s2=b.bodies[body2].shape_head;
    while(s1!=null_index&&s2!=null_index){shape_equal(a.shapes[s1],b.shapes[s2]);s1=a.shapes[s1].next;s2=b.shapes[s2].next;}
    equal("shape_list_end",s1,s2);body1=a.bodies[body1].next;body2=b.bodies[body2].next;
  }
  equal("body_list_end",body1,body2);
  std::array<unsigned,max_contacts> ordinal1{},ordinal2{};unsigned ordinal=0;
  auto c1=a.contact_head,c2=b.contact_head;
  while(c1!=null_index&&c2!=null_index){
    const auto& x=a.contacts[c1];const auto& y=b.contacts[c2];ordinal1[c1]=ordinal2[c2]=ordinal++;
    equal("contact.shape1tag",a.bodies[a.shapes[x.shape1].body].user_data,b.bodies[b.shapes[y.shape1].body].user_data);
    equal("contact.shape2tag",a.bodies[a.shapes[x.shape2].body].user_data,b.bodies[b.shapes[y.shape2].body].user_data);
    equal("contact.flags",x.flags,y.flags);equal("contact.friction",x.friction,y.friction);equal("contact.restitution",x.restitution,y.restitution);
    equal("contact.manifold_count",x.manifold_count,y.manifold_count);equal("contact.point_count",x.manifold.point_count,y.manifold.point_count);
    if(x.manifold_count){equal("manifold.normal",x.manifold.normal,y.manifold.normal);
      for(int p=0;p<x.manifold.point_count;++p)equal("manifold.active_point",x.manifold.points[p],y.manifold.points[p]);}
    c1=x.next;c2=y.next;
  }
  equal("contact_list_end",c1,c2);
  for(body1=a.body_head,body2=b.body_head;body1!=null_index;body1=a.bodies[body1].next,body2=b.bodies[body2].next){
    auto e1=a.bodies[body1].contact_head,e2=b.bodies[body2].contact_head;
    while(e1!=null_index&&e2!=null_index){
      const auto& x=a.contacts[e1/2].edges[e1&1];const auto& y=b.contacts[e2/2].edges[e2&1];
      equal("edge.contact_order",ordinal1[e1/2],ordinal2[e2/2]);equal("edge.other_tag",a.bodies[x.other].user_data,b.bodies[y.other].user_data);
      e1=x.next;e2=y.next;
    }
    equal("edge_list_end",e1,e2);
  }
}
void compare_broadphase(const ScenePipelineState& scene,void* handle,const scene_oracle::Mapping& map){
  const auto& actual=scene.broadphase;void* nw=*static_cast<void**>(handle);
  const auto* pointer=scene_oracle::pointer<unsigned char>(scene_oracle::at<std::uint32_t>(nw,102868));
  const auto& native=*reinterpret_cast<const BroadPhase*>(pointer);
  equal("bp.proxy_count",actual.m_proxyCount,native.m_proxyCount);equal("bp.free_proxy",actual.m_freeProxy,native.m_freeProxy);
  equal("bp.query_count",actual.m_queryResultCount,native.m_queryResultCount);equal("bp.timestamp",actual.m_timeStamp,native.m_timeStamp);
  equal("bp.bounds",actual.m_worldAABB,native.m_worldAABB);equal("bp.quantization",actual.m_quantizationFactor,native.m_quantizationFactor);
  for(unsigned axis=0;axis<2;++axis)for(int i=0;i<2*actual.m_proxyCount;++i)equal("bp.live_bound",actual.m_bounds[axis][i],native.m_bounds[axis][i]);
  for(int i=0;i<actual.m_queryResultCount;++i)equal("bp.live_query",actual.m_queryResults[i],native.m_queryResults[i]);
  for(unsigned i=0;i<512;++i){const auto& a=actual.m_proxyPool[i];const auto& n=native.m_proxyPool[i];
    equal("bp.proxy_valid",a.IsValid(),n.IsValid());
    if(n.IsValid()){equal("bp.lower",a.lowerBounds,n.lowerBounds);equal("bp.upper",a.upperBounds,n.upperBounds);equal("bp.overlap",a.overlapCount,n.overlapCount);equal("bp.proxy_stamp",a.timeStamp,n.timeStamp);
      if(!a.userData||a.userData>max_shapes)throw std::runtime_error("invalid typed proxy token");
      const auto native_slot=scene_oracle::Mapping::index(map.shapes,n.userData);
      const auto native_body=map.shapes[native_slot]->body;
      equal("bp.proxy_actor",scene.world.bodies[scene.world.shapes[a.userData-1].body].user_data,scene_oracle::pointer<scene_oracle::Body>(native_body)->user_data);
    }
  }
  for(auto p=actual.m_freeProxy;p!=65535;p=actual.m_proxyPool[p].GetNext())equal("bp.proxy_free_next",actual.m_proxyPool[p].GetNext(),native.m_proxyPool[p].GetNext());
  const auto& a=actual.m_pairManager;const auto& n=native.m_pairManager;
  equal("pair.count",a.m_pairCount,n.m_pairCount);equal("pair.free",a.m_freePair,n.m_freePair);equal("pair.buffer_count",a.m_pairBufferCount,n.m_pairBufferCount);
  for(unsigned i=0;i<4096;++i)equal("pair.hash",a.m_hashTable[i],n.m_hashTable[i]);
  for(unsigned i=0;i<4096;++i){const auto& x=a.m_pairs[i];const auto& y=n.m_pairs[i];
    equal("pair.proxy1",x.proxyId1,y.proxyId1);equal("pair.proxy2",x.proxyId2,y.proxyId2);equal("pair.next",x.next,y.next);
    if(y.proxyId1==65535)continue;equal("pair.status",x.status,y.status);
    if(x.userData){if(x.userData>max_contacts||!scene.world.contacts[x.userData-1].occupied)throw std::runtime_error("invalid typed pair token");
      const auto target=scene_oracle::Mapping::index(map.contacts,y.userData);unsigned typed_order=0,native_order=target;
      for(auto c=scene.world.contact_head;c!=x.userData-1;c=scene.world.contacts[c].next)++typed_order;
      equal("pair.contact_token_order",typed_order,native_order);
    }else {const auto null_contact=reinterpret_cast<std::uintptr_t>(nw)+102876;equal("pair.null_token",std::uintptr_t(y.userData),null_contact);}
  }
}
struct Actor {void* native{};std::uint32_t slot{},tag{};};
}
int main(int argc,char** argv){try{
  if(argc!=2&&argc!=3)throw std::runtime_error("usage: scene-dynamic-validate pinned-native-library [GPU-packets.bin]");
  std::fesetround(FE_TONEAREST);const unsigned short cw=0x027f;asm volatile("fldcw %0"::"m"(cw));
  scene_oracle::Api api(argv[1]);auto scene=std::make_unique<ScenePipelineState>();auto expected=std::make_unique<World>();
  auto expected_bp=std::make_unique<BroadPhase>();std::ofstream packet;
  if(argc==3){packet.open(argv[2],std::ios::binary);if(!packet)throw std::runtime_error("cannot open dynamic packets");}
  const auto write=[&](std::uint32_t v){packet.write(reinterpret_cast<const char*>(&v),4);};
  if(packet){write(dynamic_magic);write(dynamic_version);write(16);write(192);write(dynamic_case_words);write(0x3c83126f);write(8);write(0);}
  for(scene_index=0;scene_index<16;++scene_index){
    const float scale=scene_index%2?30:1;math::Context m;
    const bool boundary=scene_index>=12;const float extent=boundary?20:1000,gravity_y=boundary?0:300;
    const AABB bounds{{m.store(m.div(-extent,scale)),m.store(m.div(-extent,scale))},{m.store(m.div(extent,scale)),m.store(m.div(extent,scale))}};
    const Vec2 gravity{0,m.store(m.div(gravity_y,scale))};
    SceneCase program{};program.extent=extent;program.gravity_y=gravity_y;program.scale=scale;
    bool initial_setup=true;std::vector<std::uint64_t> hashes;
    void* handle=api.create(-extent,-extent,extent,extent,gravity_y,scale);if(!handle)throw std::runtime_error("native dynamic world failed");
    if(!scene->initialize(bounds,gravity))throw std::runtime_error("typed dynamic initialization failed");
    BodyState ground{};ground.flags=static_body|allow_sleep;rotation_matrix(ground);std::uint32_t ground_slot{};
    if(!scene->world.create_body(ground,ground_slot))throw std::runtime_error("typed ground creation failed");
    const auto create=[&](unsigned kind,float width,float height,float x,float y,float angle,float density,unsigned tag){
      const SceneActor descriptor{{kind,width,height,x,y,angle,density,.6f,0,scale},tag};
      if(initial_setup)program.actors[program.actor_count++]=descriptor;
      else {auto& op=program.ops[program.op_count++];op.tick=tick_index;op.kind=SceneOpKind::create;op.actor=descriptor;}
      Actor actor{};actor.tag=tag;actor.native=kind==0?api.circle(handle,width,x,y,density,.6f,0):
        (kind==1?api.box:api.triangle)(handle,width,height,x,y,angle,density,.6f,0);
      if(!actor.native)throw std::runtime_error("native dynamic actor failed");api.set_tag(handle,actor.native,reinterpret_cast<void*>(std::uintptr_t(tag)));
      BodyDefinition bd{};ShapeDefinition sd{};
      if(!wrapper_definition(bd,sd,kind,width,height,x,y,angle,density,.6f,0,scale))throw std::runtime_error("typed dynamic definition failed");bd.user_data=tag;
      if(!scene->create_body_shape(bd,sd,actor.slot))throw std::runtime_error("typed dynamic actor failed");return actor;
    };
    const auto floor=create(1,600,20,0,300,0,0,1);(void)floor;
    std::array<Actor,4> actors{};
    for(unsigned i=0;i<4;++i){const auto kind=(scene_index+i)%3;
      if(boundary){const float x[4]={18.5f,19.8f,21.8f,0};actors[i]=create(0,1,2,x[i],0,0,1,16+i);}
      else actors[i]=create(kind,kind?30:15,30,float(int(i)*24-36),float(100-int(i%2)*24),float(int(i)-1)*.17f,1,16+i);}
    if(boundary){
      if(scene->world.shapes[scene->world.bodies[actors[1].slot].shape_head].proxy_id==0xffff||
         !(scene->world.bodies[actors[2].slot].flags&frozen_body))throw std::runtime_error("boundary fixture admission failed");
      ++partial_proxy_creations;
    }
    initial_setup=false;
    auto& velocity_op=program.ops[program.op_count++];velocity_op.tick=16;velocity_op.kind=SceneOpKind::velocity;velocity_op.actor.tag=actors[0].tag;velocity_op.x=3*scale;velocity_op.y=-2*scale;
    if(boundary){auto& op=program.ops[program.op_count++];op.tick=32;op.kind=SceneOpKind::origin;op.actor.tag=actors[2].tag;op.angle=.1f;}
    auto& origin_op=program.ops[program.op_count++];origin_op.tick=40;origin_op.kind=SceneOpKind::origin;origin_op.actor.tag=actors[1].tag;origin_op.x=origin_op.y=80;origin_op.angle=.37f;
    auto& destroy_op=program.ops[program.op_count++];destroy_op.tick=64;destroy_op.kind=SceneOpKind::destroy;destroy_op.actor.tag=actors[2].tag;
    for(tick_index=0;tick_index<192;++tick_index){
      if(tick_index==16){api.set_velocity(handle,actors[0].native,3*scale,-2*scale);if(!scene->set_velocity(actors[0].slot,{3,-2}))throw std::runtime_error("typed velocity failed");}
      if(boundary&&tick_index==32){api.set_origin(handle,actors[2].native,0,0,.1f);
        if(!scene->set_origin(actors[2].slot,{0,0},.1f))throw std::runtime_error("typed frozen origin failed");++frozen_origin_calls;}
      if(tick_index==40){api.set_origin(handle,actors[1].native,80,80,.37f);math::Context n;
        if(!scene->set_origin(actors[1].slot,{n.store(n.div(80,scale)),n.store(n.div(80,scale))},.37f))throw std::runtime_error("typed origin failed");}
      if(tick_index==64){api.destroy_body(handle,actors[2].native);if(!scene->queue_destroy(actors[2].slot))throw std::runtime_error("typed destroy failed");}
      if(tick_index==80)actors[2]=create((scene_index+2)%3,20,35,-10,50,-.21f,2,24);
      api.step(handle,.016f,8);const auto result=scene->step(.016f,8);
      if(!result){std::fprintf(stderr,"scene%u tick%u status%u\n",scene_index,tick_index,unsigned(result.status));throw std::runtime_error("typed dynamic step failed");}
      const auto map=scene_oracle::import_scene(*expected,handle,tick_index+1);compare_world(scene->world,*expected);compare_broadphase(*scene,handle,map);++frames;
      scene_oracle::import_broadphase(*expected_bp,handle,map);
      const auto hash=dynamic_scene_hash(*expected,*expected_bp);
      if(hash!=dynamic_scene_hash(*scene))throw std::runtime_error("canonical native scene digest mismatch");
      hashes.push_back(hash);
    }
    if(packet){std::uint32_t words[dynamic_case_words];std::memcpy(words,&program,sizeof(program));for(auto word:words)write(word);
      for(auto hash:hashes){write(std::uint32_t(hash));write(std::uint32_t(hash>>32));}}
    api.destroy(handle);
  }
  if(packet&&!packet.good())throw std::runtime_error("dynamic packet write failed");
  std::printf("{\"pass\":true,\"dynamic_scenes\":16,\"steps_each\":192,\"native_fullstep_frames\":%u,\"field_comparisons\":%u,\"all_shape_pairs\":true,\"pair_add_remove\":true,\"destroy_reuse\":true,\"origin_velocity_setters\":true,\"partial_proxy_creations\":%u,\"frozen_origin_calls\":%u,\"broadphase_live_state_and_free_lists\":true,\"fallback_calls\":0}\n",frames,comparisons,partial_proxy_creations,frozen_origin_calls);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"scene-dynamic-validate: %s\n",e.what());return 1;}}
