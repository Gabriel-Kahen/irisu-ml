#pragma once
#include "compact_snapshot.cuh"
#include "game_world.cuh"

namespace irisu::cuda_native::compact_snapshot {
using Game=irisu::cuda_game::NativeGameWorld;
struct ByteRange {std::uint32_t begin{},size{};};
struct GamePlan {
  // Nine current vectors; spare capacity makes additions fail explicitly
  // rather than silently overflowing the checkpoint descriptor.
  ByteRange holes[16],spans[17],pointer_tails[4];
  std::uint32_t hole_count{},span_count{},pointer_tail_count{},bytes{};
  std::uintptr_t old_body_base{};bool valid{};
};
struct VectorHoles {
  GamePlan& plan;const unsigned char* base;
  template<class T>NATIVE_HD void operator()(const irisu::cuda_game::vector<T>& vector){
    if(!plan.valid)return;
    if(vector.count>vector.capacity||plan.hole_count==16){plan.valid=false;return;}
    const auto live=std::size_t(vector.count)*sizeof(T),size=sizeof(vector.storage)-live;
    if(!size)return;
    const ByteRange hole{std::uint32_t(vector.storage-base+live),std::uint32_t(size)};
    plan.holes[plan.hole_count++]=hole;
    if constexpr(std::is_pointer_v<T>){
      if(plan.pointer_tail_count==4){plan.valid=false;return;}
      plan.pointer_tails[plan.pointer_tail_count++]=hole;
    }
  }
};
// All scalar state, RNG, configuration, vector headers and LIVE vector values
// survive. Only unused vector storage is omitted. No live scratch is discarded.
// Pointer tails are zero-restored because the unchanged rebind API scans their
// entire capacity. Other omitted tails need no initialization: push/resize
// placement-construct elements before they become live.
NATIVE_HD inline bool build_game_plan(const Game& source,GamePlan& plan){
  plan.hole_count=plan.span_count=plan.pointer_tail_count=plan.bytes=0;plan.valid=true;
  plan.old_body_base=reinterpret_cast<std::uintptr_t>(source.simulator.bodies_.data());
  VectorHoles visit{plan,reinterpret_cast<const unsigned char*>(&source)};
  const auto& s=source.simulator;s.physics_.snapshot_vectors(visit);
  visit(s.bodies_);visit(s.groups_);visit(s.active_contact_pairs_);
  visit(s.scripted_origins_scratch_);visit(s.current_contacts_scratch_);visit(s.actor_refs_scratch_);
  if(!plan.valid)return false;
  for(unsigned i=1;i<plan.hole_count;++i){const auto value=plan.holes[i];unsigned j=i;
    while(j&&plan.holes[j-1].begin>value.begin){plan.holes[j]=plan.holes[j-1];--j;}plan.holes[j]=value;
  }
  std::uint32_t position=0;
  for(unsigned i=0;i<plan.hole_count;++i){const auto& hole=plan.holes[i];
    if(hole.begin<position||std::uint64_t(hole.begin)+hole.size>sizeof(Game)){plan.valid=false;return false;}
    if(hole.begin!=position){const ByteRange span{position,hole.begin-position};plan.spans[plan.span_count++]=span;plan.bytes+=span.size;}
    position=hole.begin+hole.size;
  }
  if(position<sizeof(Game)){const ByteRange span{position,std::uint32_t(sizeof(Game)-position)};plan.spans[plan.span_count++]=span;plan.bytes+=span.size;}
  return true;
}
NATIVE_HD inline void copy_game(Game& destination,const Game& source,const GamePlan& plan,unsigned worker=0,unsigned workers=1){
  auto* out=reinterpret_cast<unsigned char*>(&destination);const auto* in=reinterpret_cast<const unsigned char*>(&source);
  for(unsigned range=0;range<plan.span_count;++range){const auto span=plan.spans[range];
    copy_bytes(out+span.begin,in+span.begin,span.size,worker,workers);
  }
  for(unsigned range=0;range<plan.pointer_tail_count;++range){const auto tail=plan.pointer_tails[range];
    for(std::size_t i=worker;i<tail.size;i+=workers)out[tail.begin+i]=0;
  }
}
// Call only after copy_game has zero-restored inactive pointer tails and all
// copy workers finish. That precondition allows skipping the zero tails here;
// all live pointer range/alignment checks remain. Default/full copies and
// address normalizations retain the old capacity-wide scan.
NATIVE_HD inline bool rebind_game(Game& destination,ScenePipelineState& scene,const GamePlan& plan){
  return destination.rebind(&scene,plan.old_body_base,true);
}
// Only live element padding is dispensable. Inactive poisoned storage can
// become live through a memberwise copy constructor which leaves ABI padding
// untouched. Enumerate every value field, retaining all bytes of nested Vec2.
// Unknown element types default to byte-exact comparison, never silent omission.
template<class T>NATIVE_HD constexpr bool element_value_byte(std::size_t byte){
#define VALUE(field) if(byte>=offsetof(T,field)&&byte<offsetof(T,field)+sizeof(((T*)nullptr)->field))return true;
  if constexpr(std::is_same_v<T,irisu::cuda_game::Body>){
    VALUE(id) VALUE(kind) VALUE(shape) VALUE(lifecycle) VALUE(color)
    VALUE(position) VALUE(velocity) VALUE(angle) VALUE(angular_velocity)
    VALUE(native_position) VALUE(native_center) VALUE(native_velocity)
    VALUE(native_angle) VALUE(native_angular_velocity) VALUE(native_state_valid) VALUE(native_center_valid)
    VALUE(size) VALUE(density) VALUE(friction) VALUE(restitution) VALUE(inverse_mass) VALUE(inverse_inertia)
    VALUE(chain_id) VALUE(actor_slot) VALUE(size_slot) VALUE(projectile_hits) VALUE(age_ticks) VALUE(remaining_lifetime)
    VALUE(scripted_velocity) VALUE(rot_timer) VALUE(physics_owned) VALUE(special) VALUE(freshness_state)
    VALUE(grouped) VALUE(successful_clear_pending) VALUE(non_wall_contacts) VALUE(top_contact_pending)
    VALUE(top_contact_enabled) VALUE(physics_update_count) VALUE(rule_guard_f0) VALUE(delete_marked)
    VALUE(pending_delete) VALUE(sleeping) VALUE(sleep_time)
    return false;
  }else if constexpr(std::is_same_v<T,irisu::cuda_game::Contact>){
    VALUE(a) VALUE(b) VALUE(boundary) VALUE(point) VALUE(normal) VALUE(penetration)
    return false;
  }else if constexpr(std::is_same_v<T,irisu::cuda_game::ScriptedOrigin>){
    VALUE(id) VALUE(position) VALUE(angle)
    return false;
  }else if constexpr(std::is_same_v<T,irisu::cuda_game::NativePhysicsWorld::Entry>){
    VALUE(id) VALUE(slot) VALUE(shape) VALUE(size) VALUE(density) VALUE(friction) VALUE(restitution)
    VALUE(position) VALUE(velocity) VALUE(angle) VALUE(pending)
    return false;
  }else return true; // GroupState, uint64 and pointer elements have no padding.
#undef VALUE
}
struct LivePadding {
  const unsigned char* base;std::size_t offset;bool padding{};
  template<class T>NATIVE_HD void operator()(const irisu::cuda_game::vector<T>& vector){
    const auto begin=std::size_t(vector.storage-base),end=begin+std::size_t(vector.count)*sizeof(T);
    if(offset>=begin&&offset<end)padding=!element_value_byte<T>((offset-begin)%sizeof(T));
  }
};
NATIVE_HD inline bool live_padding_at(const Game& game,std::size_t offset){
  LivePadding visit{reinterpret_cast<const unsigned char*>(&game),offset};const auto& s=game.simulator;
  s.physics_.snapshot_vectors(visit);visit(s.bodies_);visit(s.groups_);visit(s.active_contact_pairs_);
  visit(s.scripted_origins_scratch_);visit(s.current_contacts_scratch_);visit(s.actor_refs_scratch_);return visit.padding;
}
static_assert(element_value_byte<irisu::cuda_game::Contact>(offsetof(irisu::cuda_game::Contact,boundary)));
static_assert(!element_value_byte<irisu::cuda_game::Contact>(offsetof(irisu::cuda_game::Contact,boundary)+1));
// Both games must first have equal scene bindings and address-normalized live
// body references. Inactive pointer tails need not match; restore makes them0.
// Every scalar/header byte remains exact; only listed LIVE element padding is
// ignored. The optional counter makes that narrower comparison explicit.
NATIVE_HD inline std::uint32_t first_game_difference(const Game& a,const Game& b,const GamePlan& plan,std::uint64_t* padding_ignored=nullptr){
  const auto* first=reinterpret_cast<const unsigned char*>(&a);const auto* second=reinterpret_cast<const unsigned char*>(&b);
  for(unsigned range=0;range<plan.span_count;++range){const auto span=plan.spans[range];
    for(std::size_t i=span.begin;i<std::size_t(span.begin)+span.size;++i)if(first[i]!=second[i]){
      if(!live_padding_at(a,i))return std::uint32_t(i);
      if(padding_ignored)++*padding_ignored;
    }
  }
  return null_index;
}
} // namespace irisu::cuda_native::compact_snapshot
