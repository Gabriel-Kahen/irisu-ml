#pragma once
#include "world_pipeline.hpp"
#include "factory.hpp"

namespace irisu::cuda_native {
struct ScenePipelineState {
  World world{};
  BroadPhase broadphase{};
  SceneScratch scratch{};

  NATIVE_HD StageResult initialize(AABB bounds,Vec2 gravity,bool sleeping=true){
    world.reset(gravity,sleeping);world.bounds=bounds;
    world.status=broadphase.Initialize(bounds);return {world.status};
  }
  NATIVE_HD StageResult accept(StageResult result){if(!result)world.status=result.status;return result;}
  NATIVE_HD StageResult check(){
    if(world.status!=Status::ok)return {world.status};
    return accept({broadphase.error});
  }
  NATIVE_HD StageResult create_body_shape(const BodyDefinition& bd,const ShapeDefinition& sd,std::uint32_t& slot){
    if(!check())return {world.status};
    BodyState body{};ShapeState shape{};auto result=construct_body_shape(body,shape,bd,sd);if(!result)return accept(result);
    unsigned shape_slot=0;while(shape_slot<max_shapes&&world.shapes[shape_slot].occupied)++shape_slot;
    if(shape_slot==max_shapes)return accept({Status::capacity});
    result=world.create_body(body,slot);if(!result)return accept(result);
    shape.body=slot;shape.id=world.next_shape_id++;shape.occupied=true;
    world.shapes[shape_slot]=shape;world.bodies[slot].shape_head=shape_slot;++world.shape_count;
    AABB bounds{};result=shape_creation_bounds(bounds,shape);if(!result)return accept(result);
    if(broadphase.InRange(bounds)){
      WorldPairCallbacks callbacks{&world};
      world.shapes[shape_slot].proxy_id=broadphase.CreateProxy(bounds,shape_slot+1,callbacks);
    }
    if(world.shapes[shape_slot].proxy_id==0xffff){result=freeze_body(world,broadphase,slot);if(!result)return accept(result);}
    return check();
  }
  NATIVE_HD StageResult queue_destroy(std::uint32_t slot){return accept(world.destroy(slot));}
  NATIVE_HD StageResult clean_contacts(){
    for(auto c=world.contact_head;c!=null_index;){const auto next=world.contacts[c].next;
      if(world.contacts[c].flags&2)destroy_contact(world,c);c=next;
    }
    return check();
  }
  NATIVE_HD StageResult clean_bodies(){
    world.destroy_immediate=true;WorldPairCallbacks callbacks{&world};
    for(auto b=world.destroy_head;b!=null_index;){
      const auto next_body=world.bodies[b].next;
      for(auto s=world.bodies[b].shape_head;s!=null_index;){
        const auto next_shape=world.shapes[s].next;
        if(world.shapes[s].proxy_id!=0xffff)broadphase.DestroyProxy(world.shapes[s].proxy_id,callbacks);
        world.shapes[s].occupied=false;--world.shape_count;s=next_shape;
        if(!check()){world.destroy_immediate=false;return {world.status};}
      }
      world.bodies[b].occupied=false;b=next_body;
    }
    world.destroy_head=null_index;world.destroy_immediate=false;return check();
  }
  NATIVE_HD StageResult step(float dt,unsigned iterations){
    if(!check())return {world.status};
    auto result=clean_contacts();if(result)result=clean_bodies();
    if(result)result=step_pairs_impl(world,scratch,dt,iterations,&broadphase);
    if(!result)return accept(result);
    WorldPairCallbacks callbacks{&world};broadphase.Commit(callbacks);return check();
  }
  NATIVE_HD StageResult set_velocity(std::uint32_t slot,Vec2 value){
    if(slot>=max_bodies||!world.bodies[slot].occupied)return accept({Status::invalid_body});
    if(!math::finite(value.x)||!math::finite(value.y))return accept({Status::unsupported_numeric});
    world.bodies[slot].velocity=value;return check(); // Native r58 does not wake here.
  }
  // Wrapper-equivalent origin setter, including its unconditional zero velocity.
  NATIVE_HD StageResult set_origin(std::uint32_t slot,Vec2 origin,float angle){
    if(slot>=max_bodies||!world.bodies[slot].occupied)return accept({Status::invalid_body});
    auto& b=world.bodies[slot];StageResult result{};
    if(!(b.flags&frozen_body)){
      b.rotation=angle;result=rotation_matrix(b);if(!result)return accept(result);math::Context m;
      const auto offset=shape_rotate(m,b.R,b.center);
      b.position={m.store(m.add(origin.x,offset.x)),m.store(m.add(origin.y,offset.y))};
      b.position0=b.position;b.rotation0=b.rotation;
      if(!m.ok())return accept({m.status});
      result=sync_body_shapes(world,scratch,slot,&broadphase,true);if(!result)return accept(result);
      WorldPairCallbacks callbacks{&world};broadphase.Commit(callbacks);
    }
    b.velocity={};return check();
  }
};
static_assert(std::is_trivially_copyable_v<ScenePipelineState>);
} // namespace irisu::cuda_native
