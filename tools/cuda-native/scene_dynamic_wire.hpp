#pragma once
#include "factory_wire.hpp"
#include "../../clone/cuda_native/world_scene.hpp"
namespace irisu::cuda_native {
inline constexpr std::uint32_t dynamic_magic=0x444e5349,dynamic_version=1;
struct SceneActor {FactoryInput shape{};std::uint32_t tag{};};
enum class SceneOpKind:std::uint32_t {create,destroy,velocity,origin};
struct SceneOp {std::uint32_t tick{};SceneOpKind kind{};SceneActor actor{};float x{},y{},angle{};};
struct SceneCase {
  float extent{},gravity_y{},scale{1};
  std::uint32_t actor_count{},op_count{};
  SceneActor actors[8]{};SceneOp ops[16]{};
};
static_assert(sizeof(SceneCase)==1396&&alignof(SceneCase)==4);
inline constexpr std::uint32_t dynamic_case_words=sizeof(SceneCase)/4;
struct SceneActors {
  std::uint32_t tags[16]{},slots[16]{},count{};
  NATIVE_HD std::uint32_t find(std::uint32_t tag)const{for(unsigned i=0;i<count;++i)if(tags[i]==tag)return slots[i];return null_index;}
  NATIVE_HD bool put(std::uint32_t tag,std::uint32_t slot){for(unsigned i=0;i<count;++i)if(tags[i]==tag){slots[i]=slot;return true;}
    if(count==16)return false;tags[count]=tag;slots[count++]=slot;return true;}
};
NATIVE_HD inline StageResult create_scene_actor(ScenePipelineState& scene,SceneActors& actors,const SceneActor& input){
  if(actors.count==16&&actors.find(input.tag)==null_index)return {Status::capacity};
  const auto& s=input.shape;BodyDefinition bd{};ShapeDefinition sd{};
  auto r=wrapper_definition(bd,sd,s.kind,s.width,s.height,s.x,s.y,s.angle,s.density,s.friction,s.restitution,s.scale);
  if(!r)return r;bd.user_data=input.tag;std::uint32_t slot{};r=scene.create_body_shape(bd,sd,slot);
  if(r&&!actors.put(input.tag,slot))return {Status::capacity};return r;
}
NATIVE_HD inline StageResult initialize_scene_case(ScenePipelineState& scene,SceneActors& actors,const SceneCase& input){
  math::Context m;const AABB bounds{{m.store(m.div(-input.extent,input.scale)),m.store(m.div(-input.extent,input.scale))},
    {m.store(m.div(input.extent,input.scale)),m.store(m.div(input.extent,input.scale))}};
  auto r=scene.initialize(bounds,{0,m.store(m.div(input.gravity_y,input.scale))});if(!r)return r;
  BodyState ground{};ground.flags=static_body|allow_sleep;r=rotation_matrix(ground);if(!r)return r;std::uint32_t slot{};
  r=scene.world.create_body(ground,slot);if(!r)return r;actors.count=0;
  for(unsigned i=0;i<input.actor_count;++i){r=create_scene_actor(scene,actors,input.actors[i]);if(!r)return r;}
  return {m.status};
}
NATIVE_HD inline StageResult apply_scene_ops(ScenePipelineState& scene,SceneActors& actors,const SceneCase& input,unsigned tick){
  for(unsigned i=0;i<input.op_count;++i){const auto& op=input.ops[i];if(op.tick!=tick)continue;
    if(op.kind==SceneOpKind::create){const auto r=create_scene_actor(scene,actors,op.actor);if(!r)return r;continue;}
    const auto slot=actors.find(op.actor.tag);if(slot==null_index)return {Status::invalid_body};math::Context m;
    StageResult r{};
    if(op.kind==SceneOpKind::destroy)r=scene.queue_destroy(slot);
    else if(op.kind==SceneOpKind::velocity)r=scene.set_velocity(slot,{m.store(m.div(op.x,input.scale)),m.store(m.div(op.y,input.scale))});
    else if(op.kind==SceneOpKind::origin)r=scene.set_origin(slot,{m.store(m.div(op.x,input.scale)),m.store(m.div(op.y,input.scale))},op.angle);
    else return {Status::invalid_snapshot};
    if(!r)return r;if(!m.ok())return {m.status};
  }
  return {};
}
NATIVE_HD inline std::uint32_t contact_ordinal(const World& world,std::uint32_t slot){
  std::uint32_t ordinal=0;for(auto c=world.contact_head;c!=slot;c=world.contacts[c].next)++ordinal;return ordinal;
}
// Initialized physical state and order, independent of pool slots/trace IDs.
// Hashing is qualification-only; timed physics kernels never call this visitor.
NATIVE_HD inline std::uint64_t dynamic_scene_hash(const World& w,const BroadPhase& bp){
  SceneHash h;h(w.body_count);h(w.shape_count);h(w.contact_count);
  h(w.gravity.x);h(w.gravity.y);h(w.bounds.min.x);h(w.bounds.min.y);h(w.bounds.max.x);h(w.bounds.max.y);
  for(auto b=w.body_head;b!=null_index;b=w.bodies[b].next){
    const auto& body=w.bodies[b];const auto s=body.shape_head;
    if(s==null_index){const ShapeState empty{};visit_factory(body,empty,h);h(0U);}
    else {for(auto i=s;i!=null_index;i=w.shapes[i].next){visit_factory(body,w.shapes[i],h);h(w.shapes[i].proxy_id);}h(1U);}
  }
  for(auto c=w.contact_head;c!=null_index;c=w.contacts[c].next){const auto& x=w.contacts[c];
    h(w.bodies[w.shapes[x.shape1].body].user_data);h(w.bodies[w.shapes[x.shape2].body].user_data);
    h(x.flags);h(x.friction);h(x.restitution);h(x.manifold_count);h(x.manifold.point_count);
    if(x.manifold_count){h(x.manifold.normal.x);h(x.manifold.normal.y);
      for(int i=0;i<x.manifold.point_count;++i){const auto& p=x.manifold.points[i];h(p.position.x);h(p.position.y);h(p.separation);h(p.normal_impulse);h(p.tangent_impulse);h(p.id);}}
  }
  for(auto b=w.body_head;b!=null_index;b=w.bodies[b].next){unsigned count=0;
    for(auto e=w.bodies[b].contact_head;e!=null_index;e=w.contacts[e/2].edges[e&1].next){const auto& x=w.contacts[e/2].edges[e&1];h(contact_ordinal(w,e/2));h(w.bodies[x.other].user_data);++count;}h(count);
  }
  h(bp.m_proxyCount);h(bp.m_freeProxy);h(bp.m_queryResultCount);h(bp.m_timeStamp);
  h(bp.m_worldAABB.min.x);h(bp.m_worldAABB.min.y);h(bp.m_worldAABB.max.x);h(bp.m_worldAABB.max.y);h(bp.m_quantizationFactor.x);h(bp.m_quantizationFactor.y);
  for(unsigned axis=0;axis<2;++axis)for(int i=0;i<2*bp.m_proxyCount;++i){const auto& b=bp.m_bounds[axis][i];h(b.value);h(b.proxyId);h(b.stabbingCount);}
  for(int i=0;i<bp.m_queryResultCount;++i)h(bp.m_queryResults[i]);
  for(unsigned i=0;i<512;++i){const auto& p=bp.m_proxyPool[i];h(p.IsValid());if(!p.IsValid())continue;
    h(p.lowerBounds[0]);h(p.lowerBounds[1]);h(p.upperBounds[0]);h(p.upperBounds[1]);h(p.overlapCount);h(p.timeStamp);h(w.bodies[w.shapes[p.userData-1].body].user_data);}
  for(auto p=bp.m_freeProxy;p!=65535;p=bp.m_proxyPool[p].GetNext())h(bp.m_proxyPool[p].GetNext());
  const auto& pm=bp.m_pairManager;h(pm.m_pairCount);h(pm.m_freePair);h(pm.m_pairBufferCount);
  for(unsigned i=0;i<4096;++i)h(pm.m_hashTable[i]);
  for(unsigned i=0;i<4096;++i){const auto& p=pm.m_pairs[i];h(p.proxyId1);h(p.proxyId2);h(p.next);if(p.proxyId1==65535)continue;
    h(p.status);h(p.userData?contact_ordinal(w,p.userData-1)+1:0);
  }
  return h.value;
}
NATIVE_HD inline std::uint64_t dynamic_scene_hash(const ScenePipelineState& scene){return dynamic_scene_hash(scene.world,scene.broadphase);}
} // namespace irisu::cuda_native
