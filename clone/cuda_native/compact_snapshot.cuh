#pragma once
#include "world_scene.hpp"
#include <cstddef>

namespace irisu::cuda_native::compact_snapshot {
// Experimental semantic checkpoint: preserve every occupied slot, including
// pending bodies and deferred contacts, and the complete SAP/history metadata.
// Inactive slot payloads and SceneScratch are outside this representation.
template<unsigned Capacity> struct Slots {
  std::uint32_t indices[Capacity],mask[(Capacity+31)/32],count;
  NATIVE_HD void clear(){count=0;for(auto& word:mask)word=0;}
  NATIVE_HD void add(unsigned slot){indices[count++]=slot;mask[slot/32]|=1U<<(slot%32);}
  NATIVE_HD bool contains(unsigned slot)const{return (mask[slot/32]>>(slot%32))&1U;}
};
struct Plan {Slots<max_bodies> bodies;Slots<max_shapes> shapes;Slots<max_contacts> contacts;};
inline constexpr std::size_t metadata_begin=offsetof(ScenePipelineState,world)+offsetof(World,gravity);
inline constexpr std::size_t metadata_end=offsetof(ScenePipelineState,scratch);
static_assert(std::is_standard_layout_v<ScenePipelineState>&&std::is_standard_layout_v<World>);
static_assert(metadata_begin==offsetof(World,contacts)+sizeof(World::contacts));
static_assert(metadata_end>=offsetof(ScenePipelineState,broadphase)+sizeof(BroadPhase));

// Caller owns global/host plan storage; no giant local aggregate construction.
// The plan expires when the source occupancy changes. Immutable prefixes can
// reuse one plan for all future clones. Selected leaves need fresh plans.
NATIVE_HD inline StageResult build_plan(const ScenePipelineState& source,Plan& plan){
  if(source.world.status!=Status::ok)return {source.world.status};
  if(source.broadphase.error!=Status::ok)return {source.broadphase.error};
  plan.bodies.clear();plan.shapes.clear();plan.contacts.clear();
  for(unsigned i=0;i<max_bodies;++i)if(source.world.bodies[i].occupied)plan.bodies.add(i);
  for(unsigned i=0;i<max_shapes;++i)if(source.world.shapes[i].occupied)plan.shapes.add(i);
  for(unsigned i=0;i<max_contacts;++i)if(source.world.contacts[i].occupied)plan.contacts.add(i);
  return {};
}
NATIVE_HD inline std::size_t transport_bytes(const Plan& plan){
  return metadata_end-metadata_begin+plan.bodies.count*sizeof(BodyState)+
    plan.shapes.count*sizeof(ShapeState)+plan.contacts.count*sizeof(ContactState);
}
// Safe even for virgin destination storage. Every occupied flag becomes
// initialized; inactive object payloads remain untouched and unconsumed.
NATIVE_HD inline void clear_missing(ScenePipelineState& destination,const Plan& plan,unsigned worker=0,unsigned workers=1){
  for(unsigned i=worker;i<max_bodies;i+=workers)if(!plan.bodies.contains(i))destination.world.bodies[i].occupied=false;
  for(unsigned i=worker;i<max_shapes;i+=workers)if(!plan.shapes.contains(i))destination.world.shapes[i].occupied=false;
  for(unsigned i=worker;i<max_contacts;i+=workers)if(!plan.contacts.contains(i))destination.world.contacts[i].occupied=false;
}
// Faster dirty-lane reuse is valid only after a successful complete call and
// an initialized World whose occupied slots are reachable from these lists.
// Invoke once, then synchronize cooperating workers before copy_active().
// Source and destination must differ. No callback/unlink operation is wanted:
// all links, proxy history and counters are subsequently replaced by source.
NATIVE_HD inline StageResult clear_reachable(ScenePipelineState& destination){
  auto& w=destination.world;if(w.status!=Status::ok)return {w.status};
  unsigned bodies=0,shapes=0,contacts=0;
  for(unsigned list=0;list<2;++list)for(auto b=list?w.destroy_head:w.body_head;b!=null_index;){
    if(b>=max_bodies||++bodies>max_bodies||!w.bodies[b].occupied)return {Status::invalid_snapshot};
    auto& body=w.bodies[b];const auto next=body.next;
    for(auto s=body.shape_head;s!=null_index;){
      if(s>=max_shapes||++shapes>max_shapes||!w.shapes[s].occupied)return {Status::invalid_snapshot};
      const auto next_shape=w.shapes[s].next;w.shapes[s].occupied=false;s=next_shape;
    }
    body.occupied=false;b=next;
  }
  for(auto c=w.contact_head;c!=null_index;){
    if(c>=max_contacts||++contacts>max_contacts||!w.contacts[c].occupied)return {Status::invalid_snapshot};
    const auto next=w.contacts[c].next;w.contacts[c].occupied=false;c=next;
  }
  return {};
}
NATIVE_HD inline void copy_bytes(unsigned char* destination,const unsigned char* source,std::size_t size,unsigned worker,unsigned workers){
  for(std::size_t i=worker;i<size;i+=workers)destination[i]=source[i];
}
template<class T,unsigned Capacity>NATIVE_HD inline void copy_slots(T* destination,const T* source,const Slots<Capacity>& slots,unsigned worker,unsigned workers){
  const auto bytes=std::size_t(slots.count)*sizeof(T);
  for(std::size_t i=worker;i<bytes;i+=workers){const auto slot=slots.indices[i/sizeof(T)];const auto offset=i%sizeof(T);
    reinterpret_cast<unsigned char*>(destination+slot)[offset]=reinterpret_cast<const unsigned char*>(source+slot)[offset];}
}
// First clear_missing() OR clear_reachable(), then copy_active(). All workers
// must finish before any scene operation. No SceneScratch initialization needed.
NATIVE_HD inline void copy_active(ScenePipelineState& destination,const ScenePipelineState& source,
                                  const Plan& plan,unsigned worker=0,unsigned workers=1){
  copy_slots(destination.world.bodies,source.world.bodies,plan.bodies,worker,workers);
  copy_slots(destination.world.shapes,source.world.shapes,plan.shapes,worker,workers);
  copy_slots(destination.world.contacts,source.world.contacts,plan.contacts,worker,workers);
  copy_bytes(reinterpret_cast<unsigned char*>(&destination)+metadata_begin,
    reinterpret_cast<const unsigned char*>(&source)+metadata_begin,metadata_end-metadata_begin,worker,workers);
}
NATIVE_HD inline std::uint32_t first_difference(const ScenePipelineState& a,const ScenePipelineState& b){
  const auto* aa=reinterpret_cast<const unsigned char*>(&a);const auto* bb=reinterpret_cast<const unsigned char*>(&b);
  for(std::size_t i=metadata_begin;i<metadata_end;++i)if(aa[i]!=bb[i])return std::uint32_t(i);
#define COMPARE_POOL(member,capacity) \
  for(unsigned slot=0;slot<capacity;++slot){ \
    if(a.world.member[slot].occupied!=b.world.member[slot].occupied)return std::uint32_t(offsetof(World,member)+slot*sizeof(a.world.member[slot])); \
    if(a.world.member[slot].occupied){const auto begin=offsetof(World,member)+slot*sizeof(a.world.member[slot]); \
      for(std::size_t i=begin;i<begin+sizeof(a.world.member[slot]);++i)if(aa[i]!=bb[i])return std::uint32_t(i);}}
  COMPARE_POOL(bodies,max_bodies)
  COMPARE_POOL(shapes,max_shapes)
  COMPARE_POOL(contacts,max_contacts)
#undef COMPARE_POOL
  return null_index;
}
} // namespace irisu::cuda_native::compact_snapshot
