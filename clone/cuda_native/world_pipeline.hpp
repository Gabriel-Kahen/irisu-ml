#pragma once
#include "world.hpp"
#include "world_shapes.hpp"
#include "collision.hpp"
#include "solver.hpp"
#include "broadphase.hpp"
#include "world_contacts.hpp"
#ifndef IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
#define IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER 0
#endif
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
#include "solver_interleaved.hpp"
#endif

namespace irisu::cuda_native {
struct SceneScratch {
  std::uint32_t stack[max_bodies]{},body_order[max_bodies]{},contact_order[max_contacts]{};
  SolverContact solver_contacts[max_contacts]{};
  Manifold manifolds[max_contacts]{};
  ContactConstraint constraints[max_contacts]{};
  AABB swept[max_shapes]{};
  std::uint32_t shape_order[max_shapes]{},shape_count{};
};
NATIVE_HD inline StageResult collide_world(World& world) {
  for(auto index=world.contact_head;index!=null_index;index=world.contacts[index].next){
    if(index>=max_contacts)return {Status::invalid_snapshot};
    auto& c=world.contacts[index];
    if(!c.occupied||c.shape1>=max_shapes||c.shape2>=max_shapes)return {Status::invalid_snapshot};
    if(c.flags&2)return {Status::bridge_required}; // deferred broadphase removal
    const auto& a=world.shapes[c.shape1];const auto& b=world.shapes[c.shape2];
    if(a.body>=max_bodies||b.body>=max_bodies)return {Status::invalid_body};
    if((world.bodies[a.body].flags&sleeping_body)&&(world.bodies[b.body].flags&sleeping_body))continue;
    const auto old=c.manifold_count;const auto previous=c.manifold;
    const auto result=collide(c.manifold,a,b);if(!result)return result;
    if(a.kind==ShapeKind::polygon&&b.kind==ShapeKind::polygon){
      bool matched[2]{};
      for(int i=0;i<c.manifold.point_count;++i){
        auto& point=c.manifold.points[i];point.normal_impulse=point.tangent_impulse=0;
        for(int j=0;j<previous.point_count;++j)if(!matched[j]&&point.id==previous.points[j].id){
          matched[j]=true;point.normal_impulse=previous.points[j].normal_impulse;
          point.tangent_impulse=previous.points[j].tangent_impulse;break;
        }
      }
    }
    c.manifold_count=c.manifold.point_count?1:0;
    if(!old&&c.manifold_count)connect_contact(world,index);
    else if(old&&!c.manifold_count)disconnect_contact(world,index);
  }
  return {};
}
NATIVE_HD inline StageResult freeze_body(World& world,BroadPhase& broadphase,std::uint32_t body){
  auto& b=world.bodies[body];b.flags|=frozen_body;b.velocity={};b.angular_velocity=0;
  WorldPairCallbacks callbacks{&world};
  for(auto s=b.shape_head;s!=null_index;s=world.shapes[s].next)if(world.shapes[s].proxy_id!=0xffff){
    broadphase.DestroyProxy(world.shapes[s].proxy_id,callbacks);world.shapes[s].proxy_id=0xffff;
    if(broadphase.error!=Status::ok)return {broadphase.error};if(world.status!=Status::ok)return {world.status};
  }
  return {};
}
NATIVE_HD inline StageResult sync_body_shapes(World& world,SceneScratch& scratch,std::uint32_t body,BroadPhase* broadphase=nullptr,bool force_static=false,bool canonical_rotation=false) {
  auto& b=world.bodies[body];if(b.inv_mass==0&&!force_static)return {};
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
  Mat22 previous_R;
  if(canonical_rotation&&math::sincos_input_supported(b.rotation)&&irisu::cuda_exact::bits_float(b.rotation0)==irisu::cuda_exact::bits_float(b.rotation))previous_R=b.R;
  else {
    const auto previous_trig=math::sincos(b.rotation0);
    if(!previous_trig.supported)return {Status::unsupported_numeric,b.id};
    previous_R={{previous_trig.cosine,previous_trig.sine},{-previous_trig.sine,previous_trig.cosine}};
  }
  StageResult result{};
#elif IRISU_CUDA_NATIVE_LIGHT_SHAPE_SYNC
  // rotation_matrix previously updated a complete local BodyState copy.
  // Its only consumed output is R; keep the same trig input/stores/status.
  const auto previous_trig=math::sincos(b.rotation0);
  if(!previous_trig.supported)return {Status::unsupported_numeric,b.id};
  const Mat22 previous_R{{previous_trig.cosine,previous_trig.sine},{-previous_trig.sine,previous_trig.cosine}};
  StageResult result{};
#else
  auto previous=b;previous.rotation=b.rotation0;
  auto result=rotation_matrix(previous);if(!result)return result;
  const auto& previous_R=previous.R;
#endif
  for(auto i=b.shape_head;i!=null_index;i=world.shapes[i].next){
    if(i>=max_shapes||!world.shapes[i].occupied)return {Status::invalid_shape};
    auto& s=world.shapes[i];
    result=shape_quick_sync(s,b.position,b.R);if(!result)return result;
    if(broadphase&&s.proxy_id==0xffff)continue;
    if(s.kind==ShapeKind::circle)result=circle_swept_bounds(scratch.swept[i],s,b.position0,previous_R);
    else if(s.kind==ShapeKind::polygon)result=polygon_swept_bounds(scratch.swept[i],s,b.position0,previous_R,b.position,b.R);
    else return {Status::unsupported_collision};
    if(!result)return result;
    const auto& bounds=scratch.swept[i];
    if(broadphase){
      if(broadphase->InRange(bounds))broadphase->MoveProxy(s.proxy_id,bounds);
      else {result=freeze_body(world,*broadphase,body);if(!result)return result;}
      if(broadphase->error!=Status::ok)return {broadphase->error};
    }else if(bounds.min.x<=world.bounds.min.x||bounds.min.y<=world.bounds.min.y||
             bounds.max.x>=world.bounds.max.x||bounds.max.y>=world.bounds.max.y)return {Status::bridge_required};
  }
  return {};
}
NATIVE_HD inline bool shape_pair_allowed(const World& world,unsigned i,unsigned j) {
  const auto& a=world.shapes[i];const auto& b=world.shapes[j];
  if(a.body==b.body||(world.bodies[a.body].inv_mass==0&&world.bodies[b.body].inv_mass==0))return false;
  if(a.group_index==b.group_index&&a.group_index)return a.group_index>0;
  return (a.mask_bits&b.category_bits)&&(a.category_bits&b.mask_bits);
}
// Explicit interim broadphase boundary. Accepted steps have the same complete
// eligible pair set throughout the swept AABBs. Any potential pair transition
// requires the separately qualified integer SAP/PairManager implementation.
NATIVE_HD inline StageResult verify_fixed_pairs(const World& world,const SceneScratch& scratch) {
  for(unsigned a=0;a<scratch.shape_count;++a)
    for(unsigned b=a+1;b<scratch.shape_count;++b){
      const auto i=scratch.shape_order[a],j=scratch.shape_order[b];if(!shape_pair_allowed(world,i,j))continue;
      bool existing=false;
      for(auto c=world.contact_head;c!=null_index;c=world.contacts[c].next){
        const auto& pair=world.contacts[c];
        if((pair.shape1==i&&pair.shape2==j)||(pair.shape1==j&&pair.shape2==i)){existing=true;break;}
      }
      if(existing!=aabb_overlap(scratch.swept[i],scratch.swept[j]))return {Status::bridge_required};
    }
  return {};
}
NATIVE_HD inline StageResult step_pairs_impl(World& world,SceneScratch& scratch,float dt,unsigned iterations,BroadPhase* broadphase=nullptr) {
  if(world.status!=Status::ok)return {world.status};
  if(world.body_count>max_bodies||world.shape_count>max_shapes||world.contact_count>max_contacts)return {Status::invalid_snapshot};
  StageResult result{};
  const bool canonical_rotation=IRISU_CUDA_NATIVE_CACHED_ROTATION&&broadphase!=nullptr;
  if(world.destroy_head!=null_index)return {Status::bridge_required};
  scratch.shape_count=0;
  if(!broadphase){
  for(auto body=world.body_head;body!=null_index;body=world.bodies[body].next){
    if(body>=max_bodies)return {Status::invalid_body};
    for(auto i=world.bodies[body].shape_head;i!=null_index;i=world.shapes[i].next){
    if(i>=max_shapes||scratch.shape_count==max_shapes)return {Status::invalid_shape};
    scratch.shape_order[scratch.shape_count++]=i;
    const auto& shape=world.shapes[i];
    if(shape.body>=max_bodies||!world.bodies[shape.body].occupied)return {Status::invalid_body};
    // Static, sleeping and non-island shapes retain their current bounds.
    if(shape.kind==ShapeKind::circle){math::Context m;scratch.swept[i]={{m.sub_store(shape.position.x,shape.radius),m.sub_store(shape.position.y,shape.radius)},
      {m.add_store(shape.radius,shape.position.x),m.add_store(shape.radius,shape.position.y)}};
      if(!m.ok())return {m.status};}
    else if(shape.kind==ShapeKind::polygon){result=polygon_bounds(scratch.swept[i],shape,world.bodies[body].position,world.bodies[body].R);if(!result)return result;}
    else return {Status::unsupported_collision};
    }
  }
  }
  result=collide_world(world);if(!result)return result;
  for(auto i=world.body_head;i!=null_index;i=world.bodies[i].next)world.bodies[i].flags&=~island_body;
  for(auto i=world.contact_head;i!=null_index;i=world.contacts[i].next)world.contacts[i].flags&=~1U;
  for(auto seed=world.body_head;seed!=null_index;seed=world.bodies[seed].next){
    if(world.bodies[seed].flags&(static_body|island_body|sleeping_body|frozen_body))continue;
    unsigned stack_count=1,body_count=0,contact_count=0;scratch.stack[0]=seed;world.bodies[seed].flags|=island_body;
    while(stack_count){
      const auto body=scratch.stack[--stack_count];auto& b=world.bodies[body];
      scratch.body_order[body_count++]=body;b.flags&=~sleeping_body;
      if(b.flags&static_body)continue;
      for(auto edge=b.contact_head;edge!=null_index;edge=contact_edge(world,edge).next){
        if(edge/2>=max_contacts)return {Status::invalid_snapshot};
        auto& contact=world.contacts[edge/2];if(contact.flags&1)continue;
        if(contact_count==max_contacts)return {Status::capacity};
        scratch.contact_order[contact_count++]=edge/2;contact.flags|=1;
        const auto other=contact_edge(world,edge).other;
        if(other>=max_bodies)return {Status::invalid_body};
        if(world.bodies[other].flags&island_body)continue;
        if(stack_count==max_bodies)return {Status::capacity};
        scratch.stack[stack_count++]=other;world.bodies[other].flags|=island_body;
      }
    }
    for(unsigned i=0;i<body_count;++i){result=integrate_velocity(world.bodies[scratch.body_order[i]],dt,world.gravity);if(!result)return result;}
    for(unsigned i=0;i<contact_count;++i){
      const auto& c=world.contacts[scratch.contact_order[i]];scratch.manifolds[i]=c.manifold;
      scratch.solver_contacts[i]={world.shapes[c.shape1].body,world.shapes[c.shape2].body,i,c.friction,c.restitution};
    }
    unsigned constraint_count{};
    auto solve=solver_build_constraints(world.bodies,max_bodies,scratch.manifolds,contact_count,scratch.solver_contacts,contact_count,scratch.constraints,max_contacts,constraint_count);
    if(!solve)return {solve.status};
    solve=solver_presolve(world.bodies,max_bodies,scratch.constraints,constraint_count,true);if(!solve)return {solve.status};
    bool interleaved_velocity=false;
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
    solve=interleaved_solver::try_iterate<interleaved_solver::Phase::velocity>(&world,world.bodies,max_bodies,
      scratch.body_order,body_count,scratch.constraints,constraint_count,iterations,interleaved_velocity);
    if(!solve)return {solve.status};
#endif
    if(!interleaved_velocity)for(unsigned i=0;i<iterations;++i){solve=solver_velocity(world.bodies,max_bodies,scratch.constraints,constraint_count);if(!solve)return {solve.status};}
    for(unsigned i=0;i<body_count;++i){result=integrate_position(world.bodies[scratch.body_order[i]],dt,canonical_rotation);if(!result)return result;}
    bool interleaved_position=false;
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
    solve=interleaved_solver::try_iterate<interleaved_solver::Phase::position>(&world,world.bodies,max_bodies,
      scratch.body_order,body_count,scratch.constraints,constraint_count,iterations,interleaved_position,.2f,canonical_rotation);
    if(!solve)return {solve.status};
#endif
    if(!interleaved_position)for(unsigned i=0;i<iterations;++i){solve=solver_position(world.bodies,max_bodies,scratch.constraints,constraint_count,.2f,canonical_rotation);if(!solve)return {solve.status};if(solve.converged)break;}
    solve=solver_postsolve(scratch.constraints,constraint_count,scratch.manifolds,contact_count);if(!solve)return {solve.status};
    for(unsigned i=0;i<contact_count;++i)world.contacts[scratch.contact_order[i]].manifold=scratch.manifolds[i];
    for(unsigned i=0;i<body_count;++i){
      const auto body=scratch.body_order[i];if(world.bodies[body].inv_mass==0)continue;
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
      // Integration establishes R, and every successful position update keeps
      // it canonical. PostSolve does not change body rotation or R.
      if(!canonical_rotation||!math::sincos_input_supported(world.bodies[body].rotation)){
#endif
      result=rotation_matrix(world.bodies[body]);if(!result)return result;
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
      }
#endif
      result=sync_body_shapes(world,scratch,body,broadphase,false,canonical_rotation);if(!result)return result;
      world.bodies[body].force={};world.bodies[body].torque=0;
    }
    if(world.allow_sleeping){result=update_sleep(world.bodies,scratch.body_order,body_count,dt);if(!result)return result;}
    for(unsigned i=0;i<body_count;++i)if(world.bodies[scratch.body_order[i]].flags&static_body)world.bodies[scratch.body_order[i]].flags&=~island_body;
  }
  if(!broadphase){result=verify_fixed_pairs(world,scratch);if(!result)return result;}
  ++world.tick;return {};
}
NATIVE_HD inline StageResult step_fixed_circle_pairs(World& world,SceneScratch& scratch,float dt,unsigned iterations) {
  for(auto b=world.body_head;b!=null_index;b=world.bodies[b].next)
    for(auto s=world.bodies[b].shape_head;s!=null_index;s=world.shapes[s].next)
      if(world.shapes[s].kind!=ShapeKind::circle){world.status=Status::unsupported_collision;return {world.status};}
  const auto result=step_pairs_impl(world,scratch,dt,iterations);
  if(!result)world.status=result.status;
  return result;
}
NATIVE_HD inline StageResult step_fixed_shape_pairs(World& world,SceneScratch& scratch,float dt,unsigned iterations) {
  const auto result=step_pairs_impl(world,scratch,dt,iterations);
  if(!result)world.status=result.status;
  return result;
}
} // namespace irisu::cuda_native
