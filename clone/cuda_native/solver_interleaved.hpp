#pragma once
// Experimental solver workspace. Original solver mathematics and native order
// are reused; only temporary storage across independent worlds is interleaved.
#include "solver.hpp"
#include "solver_hot.hpp"
#include <cstddef>
#include <cstring>
#ifndef IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER
#define IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER 0
#endif

namespace irisu::cuda_native::interleaved_solver {
inline constexpr unsigned hot_body_words=offsetof(BodyState,force)/sizeof(std::uint32_t);
inline constexpr unsigned constraint_words=sizeof(ContactConstraint)/sizeof(std::uint32_t);
static_assert(hot_body_words==12&&constraint_words==30);
static_assert(std::is_standard_layout_v<BodyState>);
static_assert(sizeof(ConstraintPoint)==44&&offsetof(ConstraintPoint,normal_impulse)==16&&
 offsetof(ConstraintPoint,tangent_impulse)==20&&offsetof(ConstraintPoint,position_impulse)==24);

// Allocate words_required(lanes,bodies,contacts) words outside the hot call.
// All participating worlds must use the same capacities and distinct lanes.
// Views cannot alias the original body/constraint arrays or another live view.
NATIVE_HD constexpr std::size_t words_required(unsigned lanes,unsigned bodies=32,unsigned contacts=64){
 return std::size_t(lanes)*(bodies*(hot_body_words+1)+contacts*constraint_words);
}
struct View {
 std::uint32_t* words{};
 unsigned lanes{},lane{},body_capacity{},constraint_capacity{},body_count{},constraint_count{};
 NATIVE_HD std::uint32_t& body_word(unsigned slot,unsigned field)const{return words[(std::size_t(field)*body_capacity+slot)*lanes+lane];}
 NATIVE_HD std::uint32_t& constraint_word(unsigned slot,unsigned field)const{
  return words[(std::size_t(hot_body_words)*body_capacity+std::size_t(field)*constraint_capacity+slot)*lanes+lane];
 }
 NATIVE_HD std::uint32_t& body_id(unsigned slot)const{
  return words[(std::size_t(hot_body_words)*body_capacity+std::size_t(constraint_words)*constraint_capacity+slot)*lanes+lane];
 }
 NATIVE_HD bool valid()const{return words&&lanes&&lane<lanes&&body_count<=body_capacity&&constraint_count<=constraint_capacity;}
 NATIVE_HD unsigned find(unsigned original)const{for(unsigned i=0;i<body_count;++i)if(body_id(i)==original)return i;return null_index;}
 template<class Body>NATIVE_HD void load_body(unsigned slot,Body& body)const{
  static_assert(sizeof(Body)>=hot_body_words*4);
  for(unsigned field=0;field<hot_body_words;++field){const auto word=body_word(slot,field);std::memcpy(reinterpret_cast<unsigned char*>(&body)+field*4,&word,4);}
 }
 template<class Body>NATIVE_HD void store_body_field(unsigned slot,unsigned field,const Body& body)const{
  std::uint32_t word;std::memcpy(&word,reinterpret_cast<const unsigned char*>(&body)+field*4,4);body_word(slot,field)=word;
 }
 template<class Body>NATIVE_HD void store_body(unsigned slot,const Body& body)const{
  static_assert(sizeof(Body)>=hot_body_words*4);
  for(unsigned field=0;field<hot_body_words;++field){std::uint32_t word;std::memcpy(&word,reinterpret_cast<const unsigned char*>(&body)+field*4,4);body_word(slot,field)=word;}
 }
 NATIVE_HD ContactConstraint load_constraint(unsigned slot)const{
  ContactConstraint result;
  for(unsigned field=0;field<constraint_words;++field){const auto word=constraint_word(slot,field);std::memcpy(reinterpret_cast<unsigned char*>(&result)+field*4,&word,4);}
  return result;
 }
 NATIVE_HD void store_constraint(unsigned slot,const ContactConstraint& constraint)const{
  for(unsigned field=0;field<constraint_words;++field){std::uint32_t word;std::memcpy(&word,reinterpret_cast<const unsigned char*>(&constraint)+field*4,4);constraint_word(slot,field)=word;}
 }
 NATIVE_HD void store_constraint_field(unsigned slot,unsigned field,const ContactConstraint& constraint)const{
  std::uint32_t word;std::memcpy(&word,reinterpret_cast<const unsigned char*>(&constraint)+field*4,4);constraint_word(slot,field)=word;
 }
};

// Host initializes this descriptor once for a pool of adjacent Scene objects.
// Only worlds beginning exactly at a pool stride are admitted. Prefixes and
// separately allocated selected-state buffers continue through the old path.
enum CounterIndex : unsigned {velocity_islands,position_islands,capacity_fallbacks,outside_pool_fallbacks,errors,counter_count};
struct Counters {unsigned long long velocity_islands{},position_islands{},capacity_fallbacks{},outside_pool_fallbacks{},errors{};};
static_assert(sizeof(Counters)==counter_count*sizeof(unsigned long long));
struct Descriptor {
 std::uint32_t* words{};
 std::uintptr_t scene_base{};
 std::size_t scene_stride{};
 unsigned lanes{},body_capacity{32},constraint_capacity{64};
 unsigned long long* counters{}; // velocity, position, capacity, outside pool, error
 NATIVE_HD View view(const void* world,unsigned bodies,unsigned contacts)const{
  const auto address=reinterpret_cast<std::uintptr_t>(world);
  if(!words||!scene_stride||address<scene_base)return {};
  const auto difference=address-scene_base;
#ifdef __CUDA_ARCH__
  // Packed rollout kernels assign the matching pool lane to global thread ID.
  // Verify its complete address before avoiding integer remainder/division;
  // isolated layouts and separately allocated buffers retain the general path.
  const std::size_t thread=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(thread<lanes&&difference==thread*scene_stride)
   return {words,lanes,unsigned(thread),body_capacity,constraint_capacity,bodies,contacts};
#endif
  if(difference%scene_stride||difference/scene_stride>=lanes)return {};
  return {words,lanes,unsigned(difference/scene_stride),body_capacity,constraint_capacity,bodies,contacts};
 }
};
#ifdef __CUDACC__
// Per-consumer translation unit: the runner configures the exact symbol it
// includes. This header does not introduce an RDC or shared-library dependency.
static __device__ __constant__ Descriptor device_workspace;
#endif

// Capacity/structural failure happens before any source mutation. The caller
// may select the original exact solver for islands outside this workspace.
NATIVE_HD inline SolverResult pack(View view,const BodyState* bodies,unsigned body_slots,
 const std::uint32_t* order,const ContactConstraint* constraints){
 if(!view.valid()||!bodies||!order||!constraints)return {Status::capacity};
 for(unsigned i=0;i<view.body_count;++i){
  if(order[i]>=body_slots)return {Status::invalid_body,i};
  for(unsigned j=0;j<i;++j)if(order[i]==order[j])return {Status::invalid_snapshot,i};
 }
 for(unsigned i=0;i<view.constraint_count;++i){const auto& c=constraints[i];
  bool first=false,second=false;
  for(unsigned j=0;j<view.body_count;++j){first|=order[j]==c.body1;second|=order[j]==c.body2;}
  if(!first||!second)return {Status::invalid_body,i};
  if(c.body1==c.body2||c.point_count<0||c.point_count>2)return {Status::invalid_snapshot,i};
 }
 for(unsigned i=0;i<view.body_count;++i){view.body_id(i)=order[i];view.store_body(i,bodies[order[i]]);}
 for(unsigned i=0;i<view.constraint_count;++i){auto c=constraints[i];c.body1=view.find(c.body1);c.body2=view.find(c.body2);view.store_constraint(i,c);}
 return {};
}
NATIVE_HD inline SolverResult unpack(View view,BodyState* bodies,unsigned body_slots,ContactConstraint* constraints){
 if(!view.valid())return {Status::invalid_snapshot};
 for(unsigned i=0;i<view.body_count;++i)if(view.body_id(i)>=body_slots)return {Status::invalid_body,i};
 for(unsigned i=0;i<view.constraint_count;++i){const auto c=view.load_constraint(i);if(c.body1>=view.body_count||c.body2>=view.body_count)return {Status::invalid_snapshot,i};}
 for(unsigned i=0;i<view.body_count;++i)view.load_body(i,bodies[view.body_id(i)]);
 for(unsigned i=0;i<view.constraint_count;++i){auto c=view.load_constraint(i);c.body1=view.body_id(c.body1);c.body2=view.body_id(c.body2);constraints[i]=c;}
 return {};
}

enum class Phase {presolve,velocity,position};
template<Phase phase>NATIVE_HD inline void commit_body(View view,unsigned slot,const hot::BodyHot& body){
 if constexpr(phase==Phase::position){
  view.store_body_field(slot,0,body);view.store_body_field(slot,1,body);view.store_body_field(slot,4,body);
  view.store_body_field(slot,8,body);view.store_body_field(slot,9,body);view.store_body_field(slot,10,body);view.store_body_field(slot,11,body);
 }else{
  view.store_body_field(slot,2,body);view.store_body_field(slot,3,body);view.store_body_field(slot,5,body);
 }
}
template<Phase phase>NATIVE_HD inline void commit_constraint(View view,unsigned slot,const ContactConstraint& c){
 for(int point=0;point<c.point_count;++point){const unsigned base=unsigned(point)*11;
  if constexpr(phase!=Phase::position){view.store_constraint_field(slot,base+4,c);view.store_constraint_field(slot,base+5,c);}
  if constexpr(phase!=Phase::velocity)view.store_constraint_field(slot,base+6,c);
 }
}
// One constraint at a time preserves Gauss-Seidel dependence and both point
// passes. Register temporaries carry only the two participants; every exit
// commits their partial writes before reporting the original error position.
template<Phase phase>NATIVE_HD inline SolverResult solve(View view,float beta=.2f,bool option=false){
 if(!view.valid())return {Status::invalid_snapshot};
 bool converged=true;
 for(unsigned i=0;i<view.constraint_count;++i){auto c=view.load_constraint(i);
  if(c.body1>=view.body_count||c.body2>=view.body_count)return {Status::invalid_body,i};
  if(c.body1==c.body2||c.point_count<0||c.point_count>2)return {Status::invalid_snapshot,i};
  const auto first=c.body1,second=c.body2;
  hot::BodyHot local[2];view.load_body(first,local[0]);view.load_body(second,local[1]);c.body1=0;c.body2=1;
  SolverResult result;
  if constexpr(phase==Phase::velocity)result=hot::solver_velocity(local,2,&c,1);
  else if constexpr(phase==Phase::presolve)result=hot::solver_presolve(local,2,&c,1,option);
  else result=hot::solver_position(local,2,&c,1,beta,option);
  commit_body<phase>(view,first,local[0]);commit_body<phase>(view,second,local[1]);commit_constraint<phase>(view,i,c);
  if(!result){result.constraint=i;return result;}
  converged&=result.converged;
 }
 return {Status::ok,null_index,null_index,converged};
}

template<Phase phase>NATIVE_HD inline SolverResult iterate(View view,BodyState* bodies,unsigned body_slots,
 const std::uint32_t* order,ContactConstraint* constraints,unsigned iterations,float beta=.2f,bool canonical_rotation=false){
 auto result=pack(view,bodies,body_slots,order,constraints);if(!result)return result;
 for(unsigned i=0;i<iterations;++i){result=solve<phase>(view,beta,canonical_rotation);
  if(!result||(phase==Phase::position&&result.converged))break;
 }
 const auto committed=unpack(view,bodies,body_slots,constraints);return committed?result:committed;
}

// Optional bridge used at the existing velocity/position iteration boundary.
// Ineligible islands explicitly leave used=false so callers run the original
// exact implementation. No approximate or CPU/native-library fallback exists.
template<Phase phase>NATIVE_HD inline SolverResult try_iterate(const void* world,BodyState* bodies,unsigned body_slots,
 const std::uint32_t* order,unsigned body_count,ContactConstraint* constraints,unsigned constraint_count,
 unsigned iterations,bool& used,float beta=.2f,bool canonical_rotation=false){
 used=false;
#if IRISU_CUDA_NATIVE_INTERLEAVED_SOLVER && defined(__CUDA_ARCH__)
 const auto descriptor=device_workspace;
 if(!descriptor.words||!constraint_count||!iterations)return {};
 auto view=descriptor.view(world,body_count,constraint_count);
 if(!view.words){if(descriptor.counters)atomicAdd(descriptor.counters+outside_pool_fallbacks,1ULL);return {};}
 if(!view.valid()){if(descriptor.counters)atomicAdd(descriptor.counters+capacity_fallbacks,1ULL);return {};}
 used=true;if(descriptor.counters)atomicAdd(descriptor.counters+(phase==Phase::velocity?velocity_islands:position_islands),1ULL);
 auto result=iterate<phase>(view,bodies,body_slots,order,constraints,iterations,beta,canonical_rotation);
 if(!result&&descriptor.counters)atomicAdd(descriptor.counters+errors,1ULL);
 return result;
#else
 return {};
#endif
}
} // namespace irisu::cuda_native::interleaved_solver
