#pragma once
#include "world.hpp"
#include "world_shapes.hpp"
#include "collision.hpp"
#include "solver.hpp"

namespace irisu::cuda_native {
struct SceneScratch {
  std::uint32_t stack[max_bodies]{},body_order[max_bodies]{},contact_order[max_contacts]{};
  SolverContact solver_contacts[max_contacts]{};
  Manifold manifolds[max_contacts]{};
  ContactConstraint constraints[max_contacts]{};
  AABB swept[max_shapes]{};
};
NATIVE_HD inline ContactEdge& contact_edge(World& world,std::uint32_t edge) {
  return world.contacts[edge/2].edges[edge&1];
}
NATIVE_HD inline void connect_contact(World& world,std::uint32_t index) {
  auto& contact=world.contacts[index];
  const std::uint32_t bodies[2]={world.shapes[contact.shape1].body,world.shapes[contact.shape2].body};
  for(unsigned side=0;side<2;++side){
    auto& edge=contact.edges[side];auto& body=world.bodies[bodies[side]];
    edge.other=bodies[1-side];edge.prev=null_index;edge.next=body.contact_head;
    if(edge.next!=null_index)contact_edge(world,edge.next).prev=2*index+side;
    body.contact_head=2*index+side;
  }
}
NATIVE_HD inline void disconnect_contact(World& world,std::uint32_t index) {
  auto& contact=world.contacts[index];
  const std::uint32_t bodies[2]={world.shapes[contact.shape1].body,world.shapes[contact.shape2].body};
  for(unsigned side=0;side<2;++side){
    auto& edge=contact.edges[side];auto& body=world.bodies[bodies[side]];
    if(edge.prev!=null_index)contact_edge(world,edge.prev).next=edge.next;
    if(edge.next!=null_index)contact_edge(world,edge.next).prev=edge.prev;
    if(body.contact_head==2*index+side)body.contact_head=edge.next;
    edge.prev=edge.next=null_index;
  }
}
NATIVE_HD inline StageResult collide_world_circles(World& world) {
  for(auto index=world.contact_head;index!=null_index;index=world.contacts[index].next){
    if(index>=max_contacts)return {Status::invalid_snapshot};
    auto& c=world.contacts[index];
    if(!c.occupied||c.shape1>=max_shapes||c.shape2>=max_shapes)return {Status::invalid_snapshot};
    if(c.flags&2)return {Status::bridge_required}; // deferred broadphase removal
    const auto& a=world.shapes[c.shape1];const auto& b=world.shapes[c.shape2];
    if(a.body>=max_bodies||b.body>=max_bodies)return {Status::invalid_body};
    if((world.bodies[a.body].flags&sleeping_body)&&(world.bodies[b.body].flags&sleeping_body))continue;
    const auto old=c.manifold_count;const auto result=collide_circle(c.manifold,a,b);if(!result)return result;
    c.manifold_count=c.manifold.point_count?1:0;
    if(!old&&c.manifold_count)connect_contact(world,index);
    else if(old&&!c.manifold_count)disconnect_contact(world,index);
  }
  return {};
}
NATIVE_HD inline StageResult sync_circle_shapes(World& world,SceneScratch& scratch,std::uint32_t body) {
  auto& b=world.bodies[body];if(b.inv_mass==0)return {};
  auto previous=b;previous.rotation=b.rotation0;
  auto result=rotation_matrix(previous);if(!result)return result;
  for(auto i=b.shape_head;i!=null_index;i=world.shapes[i].next){
    if(i>=max_shapes||!world.shapes[i].occupied)return {Status::invalid_shape};
    auto& s=world.shapes[i];if(s.kind!=ShapeKind::circle)return {Status::unsupported_collision};
    result=shape_quick_sync(s,b.position,b.R);if(!result)return result;
    result=circle_swept_bounds(scratch.swept[i],s,b.position0,previous.R);if(!result)return result;
    const auto& bounds=scratch.swept[i];
    if(bounds.min.x<=world.bounds.min.x||bounds.min.y<=world.bounds.min.y||
       bounds.max.x>=world.bounds.max.x||bounds.max.y>=world.bounds.max.y)return {Status::bridge_required};
  }
  return {};
}
NATIVE_HD inline bool circle_pair_allowed(const World& world,unsigned i,unsigned j) {
  const auto& a=world.shapes[i];const auto& b=world.shapes[j];
  if(a.body==b.body||(world.bodies[a.body].inv_mass==0&&world.bodies[b.body].inv_mass==0))return false;
  if(a.group_index==b.group_index&&a.group_index)return a.group_index>0;
  return (a.mask_bits&b.category_bits)&&(a.category_bits&b.mask_bits);
}
// Explicit interim broadphase boundary. Accepted steps have the same complete
// eligible pair set throughout the swept AABBs. Any potential pair transition
// requires the separately qualified integer SAP/PairManager implementation.
NATIVE_HD inline StageResult verify_fixed_circle_pairs(const World& world,const SceneScratch& scratch) {
  for(unsigned i=0;i<max_shapes;++i)if(world.shapes[i].occupied)
    for(unsigned j=i+1;j<max_shapes;++j)if(world.shapes[j].occupied&&circle_pair_allowed(world,i,j)){
      bool existing=false;
      for(auto c=world.contact_head;c!=null_index;c=world.contacts[c].next){
        const auto& pair=world.contacts[c];
        if((pair.shape1==i&&pair.shape2==j)||(pair.shape1==j&&pair.shape2==i)){existing=true;break;}
      }
      if(existing!=aabb_overlap(scratch.swept[i],scratch.swept[j]))return {Status::bridge_required};
    }
  return {};
}
NATIVE_HD inline StageResult step_fixed_circle_pairs_impl(World& world,SceneScratch& scratch,float dt,unsigned iterations) {
  if(world.status!=Status::ok)return {world.status};
  auto result=world.validate_snapshot();if(!result)return result;
  if(world.destroy_head!=null_index)return {Status::bridge_required};
  for(unsigned i=0;i<max_shapes;++i)if(world.shapes[i].occupied){
    const auto& shape=world.shapes[i];
    if(shape.kind!=ShapeKind::circle)return {Status::unsupported_collision};
    if(shape.body>=max_bodies||!world.bodies[shape.body].occupied)return {Status::invalid_body};
    // Static, sleeping and non-island shapes retain their current bounds.
    math::Context m;scratch.swept[i]={{m.store(m.sub(shape.position.x,shape.radius)),m.store(m.sub(shape.position.y,shape.radius))},
      {m.store(m.add(shape.radius,shape.position.x)),m.store(m.add(shape.radius,shape.position.y))}};
    if(!m.ok())return {m.status};
  }
  result=collide_world_circles(world);if(!result)return result;
  for(auto& b:world.bodies)if(b.occupied)b.flags&=~island_body;
  for(auto& c:world.contacts)if(c.occupied)c.flags&=~1U;
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
    for(unsigned i=0;i<iterations;++i){solve=solver_velocity(world.bodies,max_bodies,scratch.constraints,constraint_count);if(!solve)return {solve.status};}
    for(unsigned i=0;i<body_count;++i){result=integrate_position(world.bodies[scratch.body_order[i]],dt);if(!result)return result;}
    for(unsigned i=0;i<iterations;++i){solve=solver_position(world.bodies,max_bodies,scratch.constraints,constraint_count,.2f);if(!solve)return {solve.status};if(solve.converged)break;}
    solve=solver_postsolve(scratch.constraints,constraint_count,scratch.manifolds,contact_count);if(!solve)return {solve.status};
    for(unsigned i=0;i<contact_count;++i)world.contacts[scratch.contact_order[i]].manifold=scratch.manifolds[i];
    for(unsigned i=0;i<body_count;++i){
      const auto body=scratch.body_order[i];result=finish_integration(world.bodies[body]);if(!result)return result;
      result=sync_circle_shapes(world,scratch,body);if(!result)return result;
    }
    if(world.allow_sleeping){result=update_sleep(world.bodies,scratch.body_order,body_count,dt);if(!result)return result;}
    for(unsigned i=0;i<body_count;++i)if(world.bodies[scratch.body_order[i]].flags&static_body)world.bodies[scratch.body_order[i]].flags&=~island_body;
  }
  result=verify_fixed_circle_pairs(world,scratch);if(!result)return result;
  ++world.tick;return {};
}
NATIVE_HD inline StageResult step_fixed_circle_pairs(World& world,SceneScratch& scratch,float dt,unsigned iterations) {
  const auto result=step_fixed_circle_pairs_impl(world,scratch,dt,iterations);
  if(!result)world.status=result.status;
  return result;
}
} // namespace irisu::cuda_native
