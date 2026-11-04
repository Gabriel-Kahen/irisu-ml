#pragma once
/*
* Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com
*
* This software is provided 'as-is', without any express or implied
* warranty. In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered generated specialization: 48-byte hot bodies, unchanged arithmetic.
// Source solver.hpp SHA256: ee00e16b69741b59b28aebcba45594a2fa2cf522849e95ca5a35b6c01b845f12
// Regenerate: python tools/cuda-native/generate_hot_solver.py
#include "solver.hpp"
#include <cstddef>
namespace irisu::cuda_native::interleaved_solver::hot {
struct BodyHot {
 Vec2 position,velocity;
 float rotation,angular_velocity,inv_mass,inv_i;
 Mat22 R;
};
static_assert(sizeof(BodyHot)==48&&std::is_standard_layout_v<BodyHot>);
static_assert(offsetof(BodyHot,position)==offsetof(BodyState,position));
static_assert(offsetof(BodyHot,velocity)==offsetof(BodyState,velocity));
static_assert(offsetof(BodyHot,rotation)==offsetof(BodyState,rotation));
static_assert(offsetof(BodyHot,angular_velocity)==offsetof(BodyState,angular_velocity));
static_assert(offsetof(BodyHot,inv_mass)==offsetof(BodyState,inv_mass));
static_assert(offsetof(BodyHot,inv_i)==offsetof(BodyState,inv_i));
static_assert(offsetof(BodyHot,R)==offsetof(BodyState,R));
namespace solver_detail {
using ::irisu::cuda_native::solver_detail::anchor;
using ::irisu::cuda_native::solver_detail::dot_store;
using ::irisu::cuda_native::solver_detail::cross_store;
NATIVE_HD inline Vec2 relative_velocity(math::Context& m,const BodyHot& a,const BodyHot& b,Vec2 r1,Vec2 r2){
 const Vec2 cross1{m.mul_store(-double(a.angular_velocity),r1.y),m.mul_store(a.angular_velocity,r1.x)};
 const Vec2 cross2{m.mul_store(-double(b.angular_velocity),r2.y),m.mul_store(b.angular_velocity,r2.x)};
 const Vec2 at2{m.add_store(b.velocity.x,cross2.x),m.add_store(b.velocity.y,cross2.y)};
 const Vec2 subtract1{m.sub_store(at2.x,a.velocity.x),m.sub_store(at2.y,a.velocity.y)};
 return {m.sub_store(subtract1.x,cross1.x),m.sub_store(subtract1.y,cross1.y)};
}
NATIVE_HD inline void apply_velocity(math::Context& m,BodyHot& a,BodyHot& b,Vec2 r1,Vec2 r2,Vec2 P,float inv_mass1,float inv_i1,float inv_mass2,float inv_i2){
 // Native stores both scaled linear components before updating either body.
 const Vec2 scaled1{m.mul_store(P.x,inv_mass1),m.mul_store(P.y,inv_mass1)};
 a.velocity.x=m.sub_store(a.velocity.x,scaled1.x);
 a.velocity.y=m.sub_store(a.velocity.y,scaled1.y);
 const float cross1=cross_store(m,r1,P);
 a.angular_velocity=m.sub_store(a.angular_velocity,m.mul(cross1,inv_i1));
 const Vec2 scaled2{m.mul_store(P.x,inv_mass2),m.mul_store(P.y,inv_mass2)};
 b.velocity.x=m.add_store(b.velocity.x,scaled2.x);
 b.velocity.y=m.add_store(scaled2.y,b.velocity.y);
 const float cross2=cross_store(m,r2,P);
 b.angular_velocity=m.add_store(m.mul(cross2,inv_i2),b.angular_velocity);
}
}
NATIVE_HD inline SolverResult solver_velocity(BodyHot* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count){
 for(std::uint32_t i=0;i<count;++i){
  auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  auto& a=bodies[c.body1];auto& b=bodies[c.body2];math::Context m;
  const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;
  const Vec2 normal=c.normal,tangent{normal.y,m.mul_store(-1,normal.x)};
  for(std::int32_t j=0;j<c.point_count;++j){
   auto& p=c.points[j];const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);
   const Vec2 dv=solver_detail::relative_velocity(m,a,b,r1,r2);
   const float vn=solver_detail::dot_store(m,dv,normal);
   const float lambda=m.mul_store(m.sub(vn,p.velocity_bias),-double(p.normal_mass));
   const float accumulated=m.add_store(lambda,p.normal_impulse);
   const float new_impulse=accumulated>0?accumulated:0.0f;
   const float delta=m.sub_store(new_impulse,p.normal_impulse);
   const Vec2 P{m.mul_store(normal.x,delta),m.mul_store(delta,normal.y)};
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
   solver_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
   p.normal_impulse=new_impulse;
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
  for(std::int32_t j=0;j<c.point_count;++j){
   auto& p=c.points[j];const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);
   const Vec2 dv=solver_detail::relative_velocity(m,a,b,r1,r2);
   const float max_friction=m.mul_store(p.normal_impulse,c.friction);
   const float vt=solver_detail::dot_store(m,dv,tangent);
   const float lambda=m.mul_store(-double(vt),p.tangent_mass);
   const float accumulated=m.add_store(lambda,p.tangent_impulse);
   const float upper=accumulated<max_friction?accumulated:max_friction;
   const float lower=-max_friction;
   const float new_impulse=upper<lower?lower:upper;
   const float delta=m.sub_store(new_impulse,p.tangent_impulse);
   const Vec2 P{m.mul_store(tangent.x,delta),m.mul_store(tangent.y,delta)};
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
   solver_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
   p.tangent_impulse=new_impulse;
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
  if(!m.ok())return {m.status,i};
 }
 return {};
}
NATIVE_HD inline SolverResult solver_presolve(BodyHot* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,bool warm_start){
 for(std::uint32_t i=0;i<count;++i){auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  auto& a=bodies[c.body1];auto& b=bodies[c.body2];math::Context m;
  const float inv_mass1=a.inv_mass,inv_mass2=b.inv_mass,inv_i1=a.inv_i,inv_i2=b.inv_i;
  const Vec2 tangent{c.normal.y,m.mul_store(c.normal.x,-1)};
  for(std::int32_t j=0;j<c.point_count;++j){auto& p=c.points[j];
   if(warm_start){
    const Vec2 tangent_P{m.mul_store(c.normal.y,p.tangent_impulse),m.mul_store(p.tangent_impulse,tangent.y)};
    const Vec2 normal_P{m.mul_store(c.normal.x,p.normal_impulse),m.mul_store(c.normal.y,p.normal_impulse)};
    const Vec2 P{m.add_store(normal_P.x,tangent_P.x),m.add_store(normal_P.y,tangent_P.y)};
    const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);
    const float cross1=solver_detail::cross_store(m,r1,P);
    a.angular_velocity=m.sub_store(a.angular_velocity,m.mul(cross1,inv_i1));
    const Vec2 scaled1{m.mul_store(P.x,inv_mass1),m.mul_store(P.y,inv_mass1)};
    a.velocity.x=m.sub_store(a.velocity.x,scaled1.x);a.velocity.y=m.sub_store(a.velocity.y,scaled1.y);
    const float cross2=solver_detail::cross_store(m,r2,P);
    b.angular_velocity=m.add_store(m.mul(cross2,inv_i2),b.angular_velocity);
    const Vec2 scaled2{m.mul_store(P.x,inv_mass2),m.mul_store(P.y,inv_mass2)};
    b.velocity.x=m.add_store(b.velocity.x,scaled2.x);b.velocity.y=m.add_store(scaled2.y,b.velocity.y);
   }else{p.normal_impulse=0.0f;p.tangent_impulse=0.0f;}
   p.position_impulse=0.0f;
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
  if(!m.ok())return {m.status,i};
 }
 return {};
}
NATIVE_HD inline SolverResult solver_position(BodyHot* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,float beta,bool canonical_rotation=false){
 constexpr float linear_slop=0x1.47ae14p-8f,max_linear_correction=0x1.99999ap-3f;
 float min_separation=0.0f;
 for(std::uint32_t i=0;i<count;++i){auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  auto& a=bodies[c.body1];auto& b=bodies[c.body2];math::Context m;
  const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;
  for(std::int32_t j=0;j<c.point_count;++j){auto& p=c.points[j];
   const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);
   const Vec2 p1{m.add_store(a.position.x,r1.x),m.add_store(a.position.y,r1.y)};
   const Vec2 p2{m.add_store(b.position.x,r2.x),m.add_store(b.position.y,r2.y)};
   const Vec2 dp{m.sub_store(p2.x,p1.x),m.sub_store(p2.y,p1.y)};
   const float dot=solver_detail::dot_store(m,dp,c.normal);
   const float separation=m.add_store(dot,p.separation);
   if(separation<=min_separation)min_separation=separation;
   const float slop=m.add_store(separation,linear_slop);
   const float upper=slop<0?slop:0.0f;
   const float clamped=upper< -max_linear_correction? -max_linear_correction:upper;
   const float C=m.mul_store(clamped,beta);
   const float d_impulse=m.mul_store(C,-double(p.normal_mass));
   const float old_impulse=p.position_impulse;
   const float accumulated=m.add_store(d_impulse,old_impulse);
   p.position_impulse=accumulated>0?accumulated:0.0f;
   const float delta=m.sub_store(p.position_impulse,old_impulse);
   const Vec2 P{m.mul_store(delta,c.normal.x),m.mul_store(delta,c.normal.y)};
   const Vec2 scaled1{m.mul_store(P.x,inv_mass1),m.mul_store(P.y,inv_mass1)};
   a.position.x=m.sub_store(a.position.x,scaled1.x);a.position.y=m.sub_store(a.position.y,scaled1.y);
   const float cross1=solver_detail::cross_store(m,r1,P);
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   const auto old_rotation1=irisu::cuda_exact::bits_float(a.rotation);
#endif
   a.rotation=m.sub_store(a.rotation,m.mul(cross1,inv_i1));
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   if(!canonical_rotation||!math::sincos_input_supported(a.rotation)||irisu::cuda_exact::bits_float(a.rotation)!=old_rotation1){
#endif
   const auto trig1=math::sincos(a.rotation);if(!trig1.supported)return {Status::unsupported_numeric,i,std::uint32_t(j)};
   a.R={{trig1.cosine,trig1.sine},{-trig1.sine,trig1.cosine}};
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   }
#endif
   const Vec2 scaled2{m.mul_store(P.x,inv_mass2),m.mul_store(P.y,inv_mass2)};
   b.position.x=m.add_store(b.position.x,scaled2.x);b.position.y=m.add_store(b.position.y,scaled2.y);
   const float cross2=solver_detail::cross_store(m,r2,P);
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   const auto old_rotation2=irisu::cuda_exact::bits_float(b.rotation);
#endif
   b.rotation=m.add_store(m.mul(cross2,inv_i2),b.rotation);
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   if(!canonical_rotation||!math::sincos_input_supported(b.rotation)||irisu::cuda_exact::bits_float(b.rotation)!=old_rotation2){
#endif
   const auto trig2=math::sincos(b.rotation);if(!trig2.supported)return {Status::unsupported_numeric,i,std::uint32_t(j)};
   b.R={{trig2.cosine,trig2.sine},{-trig2.sine,trig2.cosine}};
#if IRISU_CUDA_NATIVE_CACHED_ROTATION
   }
#endif
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
 }
 return {Status::ok,null_index,null_index,min_separation>=-linear_slop};
}
} // namespace irisu::cuda_native::interleaved_solver::hot
