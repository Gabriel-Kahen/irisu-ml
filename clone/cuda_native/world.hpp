#pragma once
#include "types.hpp"
#include "solver_math.hpp"

namespace irisu::cuda_native {

// Store boundaries below are from the pinned MSVC b2Island::Solve (0xa690),
// not an algebraic rewrite of the portable Box2D source.
NATIVE_HD inline StageResult integrate_velocity(BodyState& b,float dt,Vec2 gravity) {
  if(b.inv_mass==0)return {};
  math::Context m;
  const auto fx=m.mul_store(b.inv_mass,b.force.x);
  const auto fy=m.mul_store(b.inv_mass,b.force.y);
  const auto ax=m.add_store(gravity.x,fx);
  const auto ay=m.add_store(gravity.y,fy);
  const auto dx=m.mul_store(dt,ax),dy=m.mul_store(dt,ay);
  const auto vx=m.add_store(b.velocity.x,dx),vy=m.add_store(b.velocity.y,dy);
  const auto w=m.add_store(b.angular_velocity,m.mul(m.mul(b.inv_i,dt),b.torque));
  const Vec2 velocity{m.mul_store(vx,b.linear_damping),m.mul_store(b.linear_damping,vy)};
  const auto angular=m.mul_store(b.angular_damping,w);
  if(!m.ok())return {Status::unsupported_numeric,b.id};
  b.velocity=velocity;b.angular_velocity=angular;
  b.position0=b.position;b.rotation0=b.rotation;
  return {};
}
NATIVE_HD inline StageResult rotation_matrix(BodyState& b) {
  const auto pair=math::sincos(b.rotation);
  if(!pair.supported)return {Status::unsupported_numeric,b.id};
  b.R={{pair.cosine,pair.sine},{-pair.sine,pair.cosine}};
  return {};
}
NATIVE_HD inline StageResult integrate_position(BodyState& b,float dt,bool canonical_rotation=false) {
  if(b.inv_mass==0)return {};
  math::Context m;
  const auto dx=m.mul_store(dt,b.velocity.x),dy=m.mul_store(dt,b.velocity.y);
  const Vec2 position{m.add_store(b.position.x,dx),m.add_store(b.position.y,dy)};
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
  const auto old_rotation_bits=irisu::cuda_exact::bits_float(b.rotation);
#endif
  const auto angle=m.add_store(b.rotation,m.mul(b.angular_velocity,dt));
  if(!m.ok())return {Status::unsupported_numeric,b.id};
  b.position=position;b.rotation=angle;
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
  if(canonical_rotation&&math::sincos_input_supported(angle)&&irisu::cuda_exact::bits_float(angle)==old_rotation_bits)return {};
#endif
  return rotation_matrix(b);
}
NATIVE_HD inline StageResult finish_integration(BodyState& b) {
  if(b.inv_mass==0)return {};
  const auto result=rotation_matrix(b);if(!result)return result;
  b.force={};b.torque=0;return {};
}
// The caller supplies native DFS island order. Static bodies participate in
// the final flag loop, even though their sleep timers do not enter the minimum.
NATIVE_HD inline StageResult update_sleep(BodyState* bodies,const std::uint32_t* order,
                                         unsigned count,float dt) {
  math::Context m;
  const auto linear=m.mul_store(0x1.47ae14p-7f,0x1.47ae14p-7f);
  const auto angular=m.mul_store(0x1.6c16c2p-7f,0x1.6c16c2p-7f);
  float minimum=0x1.fffffep127f;
  for(unsigned i=0;i<count;++i){
    auto& b=bodies[order[i]];if(b.inv_mass==0)continue;
    bool moving=!(b.flags&allow_sleep);
    if(!moving)moving=m.mul(b.angular_velocity,b.angular_velocity)>angular;
    // b2Dot returns through a binary32 stack temporary in this build.
    if(!moving)moving=m.add_store(m.mul(b.velocity.y,b.velocity.y),m.mul(b.velocity.x,b.velocity.x))>linear;
    if(moving){b.sleep_time=0;minimum=0;}
    else {b.sleep_time=m.add_store(b.sleep_time,dt);if(b.sleep_time<minimum)minimum=b.sleep_time;}
  }
  if(!m.ok())return {Status::unsupported_numeric};
  if(minimum>=0.5f)for(unsigned i=0;i<count;++i)bodies[order[i]].flags|=sleeping_body;
  return {};
}

// Pointer-free state, suitable for lane-local scenes and byte-copy snapshots.
// Intrusive indices preserve the native head-insertion and deferred-delete
// order independently from stable user IDs and reusable storage slots.
struct World {
  BodyState bodies[max_bodies]{};
  ShapeState shapes[max_shapes]{};
  ContactState contacts[max_contacts]{};
  Vec2 gravity{};
  AABB bounds{{-1e6f,-1e6f},{1e6f,1e6f}};
  std::uint32_t body_head{null_index},destroy_head{null_index},contact_head{null_index};
  std::uint32_t body_count{},shape_count{},contact_count{},next_body_id{1},next_shape_id{1},next_contact_id{1};
  alignas(8) std::uint64_t tick{};
  bool allow_sleeping{true};
  bool destroy_immediate{};
  Status status{Status::ok};

  NATIVE_HD void reset(Vec2 g={},bool sleeping=true) {
#if defined(IRISU_CUDA_NATIVE_SPARSE_RESET) && IRISU_CUDA_NATIVE_SPARSE_RESET
    for(auto& b:bodies)b.occupied=false;
    for(auto& s:shapes)s.occupied=false;
    for(auto& c:contacts)c.occupied=false;
#else
    for(auto& b:bodies)b=BodyState{};
    for(auto& s:shapes)s=ShapeState{};
    for(auto& c:contacts)c=ContactState{};
#endif
    gravity=g;body_head=destroy_head=contact_head=null_index;
    body_count=shape_count=contact_count=0;next_body_id=next_shape_id=next_contact_id=1;tick=0;
    allow_sleeping=sleeping;destroy_immediate=false;status=Status::ok;
  }
  NATIVE_HD StageResult validate_snapshot() const {
    if(body_count>max_bodies||shape_count>max_shapes||contact_count>max_contacts)
      return {Status::invalid_snapshot};
    bool seen[max_bodies]{};unsigned count=0;auto previous=null_index;
    for(auto i=body_head;i!=null_index;){
      if(i>=max_bodies||seen[i]||!bodies[i].occupied||bodies[i].pending_destroy||bodies[i].prev!=previous)
        return {Status::invalid_snapshot,i};
      seen[i]=true;++count;previous=i;i=bodies[i].next;
    }
    if(count!=body_count)return {Status::invalid_snapshot};
    for(auto i=destroy_head;i!=null_index;){
      if(i>=max_bodies||seen[i]||!bodies[i].occupied||!bodies[i].pending_destroy)
        return {Status::invalid_snapshot,i};
      seen[i]=true;i=bodies[i].next;
    }
    for(unsigned i=0;i<max_bodies;++i)if(bodies[i].occupied!=seen[i])return {Status::invalid_snapshot,i};
    return {};
  }
  NATIVE_HD StageResult restore(const World& source) {
    const auto result=source.validate_snapshot();if(!result)return result;
    *this=source;return {};
  }
  NATIVE_HD StageResult create_body(const BodyState& initial,std::uint32_t& slot) {
    if(status!=Status::ok)return {status};
    for(unsigned i=0;i<max_bodies;++i)if(!bodies[i].occupied){
      slot=i;bodies[i]=initial;auto& b=bodies[i];b.occupied=true;b.pending_destroy=false;
      b.id=next_body_id++;b.prev=null_index;b.next=body_head;
      b.shape_head=b.contact_head=null_index;
      if(body_head!=null_index)bodies[body_head].prev=i;
      body_head=i;++body_count;return {};
    }
    status=Status::capacity;return {status};
  }
  NATIVE_HD StageResult destroy(std::uint32_t slot) {
    if(slot>=max_bodies||!bodies[slot].occupied)return {Status::invalid_body};
    auto& b=bodies[slot];if(b.pending_destroy)return {};
    if(b.prev!=null_index)bodies[b.prev].next=b.next;
    if(b.next!=null_index)bodies[b.next].prev=b.prev;
    if(body_head==slot)body_head=b.next;
    b.pending_destroy=true;b.flags|=destroy_body;--body_count;
    b.prev=null_index;b.next=destroy_head;destroy_head=slot;return {};
  }
  NATIVE_HD StageResult clean_shape_free_bodies() {
    for(auto i=destroy_head;i!=null_index;){
      auto& b=bodies[i];if(b.shape_head!=null_index||b.contact_head!=null_index)return {Status::bridge_required,i};
      const auto next=b.next;b=BodyState{};i=next;
    }
    destroy_head=null_index;return {};
  }
  // Independent shape-free scene path. This entry deliberately rejects shape
  // or contact work instead of silently omitting broadphase/collision stages.
  NATIVE_HD StageResult step_shape_free(float dt) {
    if(status!=Status::ok)return {status};
    if(shape_count||contact_count)return {Status::bridge_required};
    auto result=clean_shape_free_bodies();if(!result){status=result.status;return result;}
    for(auto& b:bodies)if(b.occupied)b.flags&=~island_body;
    for(auto i=body_head;i!=null_index;i=bodies[i].next){
      auto& b=bodies[i];if(b.flags&(static_body|frozen_body|sleeping_body|island_body))continue;
      b.flags|=island_body;b.flags&=~sleeping_body;
      result=integrate_velocity(b,dt,gravity);if(result)result=integrate_position(b,dt);
      if(result)result=finish_integration(b);
      if(result&&allow_sleeping)result=update_sleep(bodies,&i,1,dt);
      if(!result){status=result.status;return result;}
    }
    ++tick;return {};
  }
};
static_assert(std::is_trivially_copyable_v<World>);
} // namespace irisu::cuda_native
