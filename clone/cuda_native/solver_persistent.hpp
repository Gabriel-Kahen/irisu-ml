#pragma once
// Experimental island-lifetime hot mirror. Native body slots and contact order
// stay unchanged; the existing AoS objects remain authoritative at API edges.
#include "world.hpp"
#include "solver_hot.hpp"
#include "solver_persistent_velocity.hpp"
#include <cstddef>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#ifndef IRISU_CUDA_NATIVE_PERSISTENT_SOLVER
#define IRISU_CUDA_NATIVE_PERSISTENT_SOLVER 0
#endif

namespace irisu::cuda_native::persistent_solver {
namespace hot=::irisu::cuda_native::interleaved_solver::hot;
inline constexpr unsigned tile_width=32,body_words=12,constraint_words=30;
static_assert(sizeof(hot::BodyHot)==body_words*4&&sizeof(ContactConstraint)==constraint_words*4);
NATIVE_HD constexpr std::size_t words_per_lane(unsigned bodies=max_bodies,unsigned contacts=max_contacts){
 return std::size_t(body_words)*bodies+std::size_t(constraint_words)*contacts+(bodies+31)/32;
}
NATIVE_HD constexpr std::size_t words_required(unsigned lanes,unsigned bodies=max_bodies,unsigned contacts=max_contacts){
 return std::size_t((lanes+tile_width-1)/tile_width)*tile_width*words_per_lane(bodies,contacts);
}
struct View {
 std::uint32_t* words{};unsigned lane{},lanes{},body_capacity{},constraint_capacity{};
 NATIVE_HD bool valid()const{return words&&lanes&&lane<lanes&&body_capacity&&constraint_capacity&&body_capacity<=max_bodies&&constraint_capacity<=max_contacts;}
 NATIVE_HD std::uint32_t& word(std::size_t field)const{
  return words[std::size_t(lane/tile_width)*tile_width*words_per_lane(body_capacity,constraint_capacity)+field*tile_width+lane%tile_width];
 }
 NATIVE_HD std::uint32_t& body_word(unsigned slot,unsigned field)const{return word(std::size_t(field)*body_capacity+slot);}
 NATIVE_HD std::uint32_t& constraint_word(unsigned slot,unsigned field)const{return word(std::size_t(body_words)*body_capacity+std::size_t(field)*constraint_capacity+slot);}
 NATIVE_HD std::uint32_t& members(unsigned index)const{return word(std::size_t(body_words)*body_capacity+std::size_t(constraint_words)*constraint_capacity+index);}
 NATIVE_HD bool contains(unsigned slot)const{return slot<body_capacity&&(members(slot/32)&(1U<<(slot%32)));}
 NATIVE_HD hot::BodyHot load_body(unsigned slot)const{
  hot::BodyHot body;
#ifdef __CUDACC__
#pragma unroll
#endif
  for(unsigned field=0;field<body_words;++field){const auto bits=body_word(slot,field);std::memcpy(reinterpret_cast<unsigned char*>(&body)+4*field,&bits,4);}
  return body;
 }
 template<class Body>NATIVE_HD void store_body(unsigned slot,const Body& body)const{
#ifdef __CUDACC__
#pragma unroll
#endif
  for(unsigned field=0;field<body_words;++field){std::uint32_t bits;std::memcpy(&bits,reinterpret_cast<const unsigned char*>(&body)+4*field,4);body_word(slot,field)=bits;}
 }
 NATIVE_HD void store_body_field(unsigned slot,unsigned field,const hot::BodyHot& body)const{
  std::uint32_t bits;std::memcpy(&bits,reinterpret_cast<const unsigned char*>(&body)+4*field,4);body_word(slot,field)=bits;
 }
 NATIVE_HD ContactConstraint load_constraint(unsigned slot)const{
  ContactConstraint c;
#ifdef __CUDACC__
#pragma unroll
#endif
  for(unsigned field=0;field<constraint_words;++field){const auto bits=constraint_word(slot,field);std::memcpy(reinterpret_cast<unsigned char*>(&c)+4*field,&bits,4);}
  return c;
 }
 NATIVE_HD void store_constraint(unsigned slot,const ContactConstraint& c)const{
#ifdef __CUDACC__
#pragma unroll
#endif
  for(unsigned field=0;field<constraint_words;++field){std::uint32_t bits;std::memcpy(&bits,reinterpret_cast<const unsigned char*>(&c)+4*field,4);constraint_word(slot,field)=bits;}
 }
 NATIVE_HD void store_constraint_field(unsigned slot,unsigned field,const ContactConstraint& c)const{
  std::uint32_t bits;std::memcpy(&bits,reinterpret_cast<const unsigned char*>(&c)+4*field,4);constraint_word(slot,field)=bits;
 }
};
enum CounterIndex : unsigned {islands,body_records,constraint_records,unavailable,pack_rejections,failures,counter_count};
struct Descriptor {
 std::uint32_t* words{};std::uintptr_t scene_base{};std::size_t scene_stride{};
 unsigned lanes{},body_capacity{max_bodies},constraint_capacity{max_contacts};unsigned long long* counters{};
 NATIVE_HD View view(const void* world)const{
  const auto address=reinterpret_cast<std::uintptr_t>(world);
  if(!words||!scene_stride||address<scene_base)return {};
  const auto difference=address-scene_base;
#ifdef __CUDA_ARCH__
  const std::size_t thread=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(thread<lanes&&difference==thread*scene_stride)return {words,unsigned(thread),lanes,body_capacity,constraint_capacity};
  if(blockIdx.x<lanes&&difference==std::size_t(blockIdx.x)*scene_stride)return {words,unsigned(blockIdx.x),lanes,body_capacity,constraint_capacity};
#endif
  if(difference%scene_stride||difference/scene_stride>=lanes)return {};
  return {words,unsigned(difference/scene_stride),lanes,body_capacity,constraint_capacity};
 }
};
#ifdef __CUDACC__
static __device__ __constant__ Descriptor device_workspace;
inline cudaError_t configure(const Descriptor& descriptor){return cudaMemcpyToSymbol(device_workspace,&descriptor,sizeof(descriptor));}
#endif

// Linear validation/transport; no dense renumbering or body-by-contact searches.
// Rejection mutates only private workspace, so the caller can retain its exact
// original path, including any later stage's native partial failure writes.
NATIVE_HD inline bool pack(View view,const BodyState* bodies,unsigned body_slots,const unsigned* order,unsigned body_count,
                           const ContactConstraint* constraints,unsigned count){
 if(!view.valid()||!bodies||!order||!constraints||body_count>view.body_capacity||count>view.constraint_capacity)return false;
 for(unsigned i=0;i<(view.body_capacity+31)/32;++i)view.members(i)=0;
 for(unsigned i=0;i<body_count;++i){const auto slot=order[i];
  if(slot>=body_slots||slot>=view.body_capacity||view.contains(slot))return false;
  view.members(slot/32)|=1U<<(slot%32);view.store_body(slot,bodies[slot]);
 }
 for(unsigned i=0;i<count;++i){const auto& c=constraints[i];
  if(c.body1>=body_slots||c.body2>=body_slots||c.body1==c.body2||c.point_count<0||c.point_count>2||!view.contains(c.body1)||!view.contains(c.body2))return false;
  view.store_constraint(i,c);
 }
 return true;
}
enum class Phase {presolve,velocity,position};
template<Phase phase>NATIVE_HD inline void commit_body(View view,unsigned slot,const hot::BodyHot& body){
 if constexpr(phase==Phase::position){
  view.store_body_field(slot,0,body);view.store_body_field(slot,1,body);view.store_body_field(slot,4,body);
  view.store_body_field(slot,8,body);view.store_body_field(slot,9,body);view.store_body_field(slot,10,body);view.store_body_field(slot,11,body);
 }
 else {view.store_body_field(slot,2,body);view.store_body_field(slot,3,body);view.store_body_field(slot,5,body);}
}
template<Phase phase>NATIVE_HD inline void commit_constraint(View view,unsigned slot,const ContactConstraint& c){
 for(int j=0;j<c.point_count;++j){const unsigned base=unsigned(j)*11;
  if constexpr(phase!=Phase::position){view.store_constraint_field(slot,base+4,c);view.store_constraint_field(slot,base+5,c);}
  if constexpr(phase!=Phase::velocity)view.store_constraint_field(slot,base+6,c);
 }
}
template<Phase phase>NATIVE_HD inline SolverResult solve(View view,unsigned count,float beta=.2f,bool option=false){
 bool converged=true;
 for(unsigned i=0;i<count;++i){auto c=view.load_constraint(i);const auto first=c.body1,second=c.body2;
  hot::BodyHot local[2]={view.load_body(first),view.load_body(second)};c.body1=0;c.body2=1;SolverResult result;
  if constexpr(phase==Phase::presolve)result=hot::solver_presolve(local,2,&c,1,option);
  else if constexpr(phase==Phase::velocity)result=hot_velocity::solver_velocity_contact(local,2,&c,1);
  else result=hot::solver_position(local,2,&c,1,beta,option);
  commit_body<phase>(view,first,local[0]);commit_body<phase>(view,second,local[1]);commit_constraint<phase>(view,i,c);
  if(!result){result.constraint=i;return result;}converged&=result.converged;
 }
 return {Status::ok,null_index,null_index,converged};
}
// The same separately rounded graph as integrate_position/rotation_matrix.
// IDs remain in the canonical body; only its actual numerical hot fields move.
NATIVE_HD inline StageResult integrate_position_hot(hot::BodyHot& b,float dt,bool canonical_rotation,unsigned id){
 if(b.inv_mass==0)return {};math::Context m;
 const auto dx=m.mul_store(dt,b.velocity.x),dy=m.mul_store(dt,b.velocity.y);
 const Vec2 position{m.add_store(b.position.x,dx),m.add_store(b.position.y,dy)};
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
 const auto old_rotation_bits=irisu::cuda_exact::bits_float(b.rotation);
#endif
 const auto angle=m.add_store(b.rotation,m.mul(b.angular_velocity,dt));
 if(!m.ok())return {Status::unsupported_numeric,id};b.position=position;b.rotation=angle;
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
 if(canonical_rotation&&math::sincos_input_supported(angle)&&irisu::cuda_exact::bits_float(angle)==old_rotation_bits)return {};
#endif
 const auto pair=math::sincos(b.rotation);if(!pair.supported)return {Status::unsupported_numeric,id};
 b.R={{pair.cosine,pair.sine},{-pair.sine,pair.cosine}};return {};
}
NATIVE_HD inline void flush(View view,BodyState* bodies,const unsigned* order,unsigned body_count,ContactConstraint* constraints,unsigned count){
 for(unsigned i=0;i<body_count;++i){const auto slot=order[i];auto& body=bodies[slot];
  for(unsigned field=0;field<body_words;++field)if(field!=6&&field!=7){const auto bits=view.body_word(slot,field);std::memcpy(reinterpret_cast<unsigned char*>(&body)+field*4,&bits,4);}
 }
 for(unsigned i=0;i<count;++i){auto& c=constraints[i];
  for(int j=0;j<c.point_count;++j)for(unsigned field=4;field<=6;++field){const auto bits=view.constraint_word(i,unsigned(j)*11+field);std::memcpy(reinterpret_cast<unsigned char*>(&c)+4*(unsigned(j)*11+field),&bits,4);}
 }
}
// pack() must have succeeded. Every exit flushes exactly the mutable fields;
// canonical topology, inactive payloads, and native scratch padding stay intact.
NATIVE_HD inline StageResult island(View view,BodyState* bodies,const unsigned* order,unsigned body_count,
 ContactConstraint* constraints,unsigned count,Manifold* manifolds,unsigned manifold_count,float dt,unsigned iterations,bool canonical_rotation){
 StageResult result{};auto solved=solve<Phase::presolve>(view,count,.2f,true);
 if(!solved)result={solved.status};
 if(result)for(unsigned i=0;i<iterations;++i){solved=solve<Phase::velocity>(view,count);if(!solved){result={solved.status};break;}}
 if(result)for(unsigned i=0;i<body_count;++i){const auto slot=order[i];auto b=view.load_body(slot);result=integrate_position_hot(b,dt,canonical_rotation,bodies[slot].id);commit_body<Phase::position>(view,slot,b);if(!result)break;}
 if(result)for(unsigned i=0;i<iterations;++i){solved=solve<Phase::position>(view,count,.2f,canonical_rotation);if(!solved){result={solved.status};break;}if(solved.converged)break;}
 if(result)for(unsigned i=0;i<count;++i){const auto c=view.load_constraint(i);solved=solver_postsolve(&c,1,manifolds,manifold_count);if(!solved){result={solved.status};break;}}
 flush(view,bodies,order,body_count,constraints,count);return result;
}
NATIVE_HD inline StageResult try_island(const void* world,BodyState* bodies,unsigned body_slots,const unsigned* order,unsigned body_count,
 ContactConstraint* constraints,unsigned count,Manifold* manifolds,unsigned manifold_count,float dt,unsigned iterations,bool canonical_rotation,bool& used){
 used=false;
#if IRISU_CUDA_NATIVE_PERSISTENT_SOLVER && defined(__CUDA_ARCH__)
 const auto descriptor=device_workspace;if(!descriptor.words||!count)return {};
 const auto view=descriptor.view(world);
 if(!view.valid()){if(descriptor.counters)atomicAdd(descriptor.counters+unavailable,1ULL);return {};}
 if(!pack(view,bodies,body_slots,order,body_count,constraints,count)){if(descriptor.counters)atomicAdd(descriptor.counters+pack_rejections,1ULL);return {};}
 used=true;if(descriptor.counters){atomicAdd(descriptor.counters+islands,1ULL);atomicAdd(descriptor.counters+body_records,static_cast<unsigned long long>(body_count));atomicAdd(descriptor.counters+constraint_records,static_cast<unsigned long long>(count));}
 const auto result=island(view,bodies,order,body_count,constraints,count,manifolds,manifold_count,dt,iterations,canonical_rotation);
 if(!result&&descriptor.counters)atomicAdd(descriptor.counters+failures,1ULL);return result;
#else
 return {};
#endif
}
} // namespace irisu::cuda_native::persistent_solver
