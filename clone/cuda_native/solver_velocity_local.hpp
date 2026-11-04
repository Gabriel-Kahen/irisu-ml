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
// Altered generated specialization: defer velocity writeback to contact exit.
// Source solver.hpp SHA256: b18dbf0356d32fe798be74ab9e62bb34747064554ee9bbe983e6f3ce0ce05e60
// Regenerate: python tools/cuda-native/generate_local_velocity.py
#include "solver_velocity_cached.hpp"
namespace irisu::cuda_native {
namespace local_velocity_detail {
struct VelocityState {Vec2 velocity;float angular_velocity;};
// Contacts have distinct body indices. Commit on every exit, including numeric
// rejection, to retain the reference's complete partial-write state.
struct VelocityBinding {
 BodyState& first;BodyState& second;VelocityState a,b;
 NATIVE_HD VelocityBinding(BodyState& x,BodyState& y):first(x),second(y),a{x.velocity,x.angular_velocity},b{y.velocity,y.angular_velocity}{}
 NATIVE_HD ~VelocityBinding(){first.velocity=a.velocity;first.angular_velocity=a.angular_velocity;second.velocity=b.velocity;second.angular_velocity=b.angular_velocity;}
};
using ::irisu::cuda_native::solver_detail::cross_store;
NATIVE_HD inline Vec2 relative_velocity(math::Context& m,const VelocityState& a,const VelocityState& b,Vec2 r1,Vec2 r2){
 const Vec2 cross1{m.mul_store(-double(a.angular_velocity),r1.y),m.mul_store(a.angular_velocity,r1.x)};
 const Vec2 cross2{m.mul_store(-double(b.angular_velocity),r2.y),m.mul_store(b.angular_velocity,r2.x)};
 const Vec2 at2{m.add_store(b.velocity.x,cross2.x),m.add_store(b.velocity.y,cross2.y)};
 const Vec2 subtract1{m.sub_store(at2.x,a.velocity.x),m.sub_store(at2.y,a.velocity.y)};
 return {m.sub_store(subtract1.x,cross1.x),m.sub_store(subtract1.y,cross1.y)};
}
NATIVE_HD inline void apply_velocity(math::Context& m,VelocityState& a,VelocityState& b,Vec2 r1,Vec2 r2,Vec2 P,float inv_mass1,float inv_i1,float inv_mass2,float inv_i2){
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
} // namespace local_velocity_detail
template<bool cached>NATIVE_HD inline SolverResult solver_velocity_local(BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,const VelocityAnchor* anchors=nullptr){
 for(std::uint32_t i=0;i<count;++i){
  auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  local_velocity_detail::VelocityBinding binding(bodies[c.body1],bodies[c.body2]);auto& a=binding.a;auto& b=binding.b;math::Context m;
  const float inv_mass1=bodies[c.body1].inv_mass,inv_i1=bodies[c.body1].inv_i,inv_mass2=bodies[c.body2].inv_mass,inv_i2=bodies[c.body2].inv_i;
  const Vec2 normal=c.normal,tangent{normal.y,m.mul_store(-1,normal.x)};
  for(std::int32_t j=0;j<c.point_count;++j){
   auto& p=c.points[j];const Vec2 r1=cached?anchors[2*i+j].first:solver_detail::anchor(m,bodies[c.body1].R,p.local_anchor1),r2=cached?anchors[2*i+j].second:solver_detail::anchor(m,bodies[c.body2].R,p.local_anchor2);
   const Vec2 dv=local_velocity_detail::relative_velocity(m,a,b,r1,r2);
   const float vn=solver_detail::dot_store(m,dv,normal);
   const float lambda=m.mul_store(m.sub(vn,p.velocity_bias),-double(p.normal_mass));
   const float accumulated=m.add_store(lambda,p.normal_impulse);
   const float new_impulse=accumulated>0?accumulated:0.0f;
   const float delta=m.sub_store(new_impulse,p.normal_impulse);
   const Vec2 P{m.mul_store(normal.x,delta),m.mul_store(delta,normal.y)};
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
   local_velocity_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
   p.normal_impulse=new_impulse;
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
  for(std::int32_t j=0;j<c.point_count;++j){
   auto& p=c.points[j];const Vec2 r1=cached?anchors[2*i+j].first:solver_detail::anchor(m,bodies[c.body1].R,p.local_anchor1),r2=cached?anchors[2*i+j].second:solver_detail::anchor(m,bodies[c.body2].R,p.local_anchor2);
   const Vec2 dv=local_velocity_detail::relative_velocity(m,a,b,r1,r2);
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
   local_velocity_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
   p.tangent_impulse=new_impulse;
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
  if(!m.ok())return {m.status,i};
 }
 return {};
}
} // namespace irisu::cuda_native
