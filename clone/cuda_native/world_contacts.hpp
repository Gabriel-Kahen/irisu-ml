#pragma once
#include "world.hpp"

namespace irisu::cuda_native {
NATIVE_HD inline ContactEdge& contact_edge(World& world,std::uint32_t edge){return world.contacts[edge/2].edges[edge&1];}
NATIVE_HD inline void wake_body(BodyState& body){body.flags&=~sleeping_body;body.sleep_time=0;}
NATIVE_HD inline void connect_contact(World& world,std::uint32_t index){
  auto& contact=world.contacts[index];
  const std::uint32_t bodies[2]={world.shapes[contact.shape1].body,world.shapes[contact.shape2].body};
  for(unsigned side=0;side<2;++side){auto& edge=contact.edges[side];auto& body=world.bodies[bodies[side]];
    edge.other=bodies[1-side];edge.prev=null_index;edge.next=body.contact_head;
    if(edge.next!=null_index)contact_edge(world,edge.next).prev=2*index+side;
    body.contact_head=2*index+side;
  }
}
NATIVE_HD inline void disconnect_contact(World& world,std::uint32_t index){
  auto& contact=world.contacts[index];
  const std::uint32_t bodies[2]={world.shapes[contact.shape1].body,world.shapes[contact.shape2].body};
  for(unsigned side=0;side<2;++side){auto& edge=contact.edges[side];auto& body=world.bodies[bodies[side]];
    if(edge.prev!=null_index)contact_edge(world,edge.prev).next=edge.next;
    if(edge.next!=null_index)contact_edge(world,edge.next).prev=edge.prev;
    if(body.contact_head==2*index+side)body.contact_head=edge.next;
    edge.prev=edge.next=null_index;
  }
}
NATIVE_HD inline void destroy_contact(World& world,std::uint32_t index){
  auto& c=world.contacts[index];
  if(c.prev!=null_index)world.contacts[c.prev].next=c.next;
  if(c.next!=null_index)world.contacts[c.next].prev=c.prev;
  if(world.contact_head==index)world.contact_head=c.next;
  if(c.manifold_count){
    wake_body(world.bodies[world.shapes[c.shape1].body]);wake_body(world.bodies[world.shapes[c.shape2].body]);
    disconnect_contact(world,index);
  }
  c.occupied=false;--world.contact_count;
}
struct WorldPairCallbacks {
  World* world;
  NATIVE_HD std::uint32_t PairAdded(std::uint32_t first,std::uint32_t second){
    auto& w=*world;if(!first||!second||first>max_shapes||second>max_shapes){w.status=Status::invalid_shape;return 0;}
    auto a=first-1,b=second-1;
    if(!w.shapes[a].occupied||!w.shapes[b].occupied){w.status=Status::invalid_shape;return 0;}
    const auto body1=w.shapes[a].body,body2=w.shapes[b].body;
    if(body1==body2||(w.bodies[body1].inv_mass==0&&w.bodies[body2].inv_mass==0))return 0;
    const auto& s1=w.shapes[a];const auto& s2=w.shapes[b];
    if(s1.group_index==s2.group_index&&s1.group_index){if(s1.group_index<0)return 0;}
    else if(!(s1.mask_bits&s2.category_bits)||!(s1.category_bits&s2.mask_bits))return 0;
    if(w.bodies[body2].inv_mass==0){const auto t=a;a=b;b=t;}
    // The native contact registry canonicalizes only polygon/circle order.
    if(w.shapes[a].kind==ShapeKind::circle&&w.shapes[b].kind==ShapeKind::polygon){const auto t=a;a=b;b=t;}
    for(unsigned i=0;i<max_contacts;++i)if(!w.contacts[i].occupied){
      auto& c=w.contacts[i];c=ContactState{};c.occupied=true;c.id=w.next_contact_id++;
      c.shape1=a;c.shape2=b;c.next=w.contact_head;
      if(c.next!=null_index)w.contacts[c.next].prev=i;
      w.contact_head=i;++w.contact_count;
      math::Context m;const auto product=m.store(m.mul(w.shapes[a].friction,w.shapes[b].friction));
      c.friction=m.store(m.sqrt(product));
      c.restitution=w.shapes[a].restitution>w.shapes[b].restitution?w.shapes[a].restitution:w.shapes[b].restitution;
      if(!m.ok())w.status=m.status;
      return i+1;
    }
    w.status=Status::capacity;return 0;
  }
  NATIVE_HD void PairRemoved(std::uint32_t,std::uint32_t,std::uint32_t token){
    auto& w=*world;if(!token)return;
    if(token>max_contacts||!w.contacts[token-1].occupied){w.status=Status::invalid_snapshot;return;}
    if(w.destroy_immediate)destroy_contact(w,token-1);else w.contacts[token-1].flags|=2;
  }
};
} // namespace irisu::cuda_native
