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
// Altered generated specialization: exact contact anchors in register hot bodies.
// Source solver.hpp SHA256: b18dbf0356d32fe798be74ab9e62bb34747064554ee9bbe983e6f3ce0ce05e60
// Regenerate: python tools/cuda-native/generate_persistent_velocity.py
#include "solver_hot.hpp"
namespace irisu::cuda_native::persistent_solver::hot_velocity {
using BodyHot=::irisu::cuda_native::interleaved_solver::hot::BodyHot;
namespace solver_detail=::irisu::cuda_native::interleaved_solver::hot::solver_detail;
struct Anchor {Vec2 first,second;};
NATIVE_HD inline SolverResult solver_velocity_contact(BodyHot* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count){
 for(std::uint32_t i=0;i<count;++i){
  auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  auto& a=bodies[c.body1];auto& b=bodies[c.body2];Anchor anchors[2];math::Context m;
  const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;
  const Vec2 normal=c.normal,tangent{normal.y,m.mul_store(-1,normal.x)};
  #ifdef __CUDACC__
#pragma unroll
#endif
  for(std::int32_t j=0;j<2;++j){if(j>=c.point_count)break;
   auto& p=c.points[j];const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);anchors[j]={r1,r2};
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
  #ifdef __CUDACC__
#pragma unroll
#endif
  for(std::int32_t j=0;j<2;++j){if(j>=c.point_count)break;
   auto& p=c.points[j];const Vec2 r1=anchors[j].first,r2=anchors[j].second;
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
} // namespace irisu::cuda_native::persistent_solver::hot_velocity
