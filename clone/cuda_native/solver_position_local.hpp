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
// Altered generated specialization: defer position writeback to contact exit.
// Source solver.hpp SHA256: b18dbf0356d32fe798be74ab9e62bb34747064554ee9bbe983e6f3ce0ce05e60
// Regenerate: python tools/cuda-native/generate_local_position.py
#include "solver.hpp"
namespace irisu::cuda_native {
namespace local_position_detail {
struct PositionState {Vec2 position;float rotation;Mat22 R;};
// Distinct contact participants are checked before construction. Every exit
// commits the reference's position/rotation partial writes, including errors.
struct PositionBinding {
 BodyState& first;BodyState& second;PositionState a,b;
 NATIVE_HD PositionBinding(BodyState& x,BodyState& y):first(x),second(y),a{x.position,x.rotation,x.R},b{y.position,y.rotation,y.R}{}
 NATIVE_HD ~PositionBinding(){first.position=a.position;first.rotation=a.rotation;first.R=a.R;second.position=b.position;second.rotation=b.rotation;second.R=b.R;}
};
} // namespace local_position_detail
NATIVE_HD inline SolverResult solver_position_local(BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,float beta,bool canonical_rotation=false){
 constexpr float linear_slop=0x1.47ae14p-8f,max_linear_correction=0x1.99999ap-3f;
 float min_separation=0.0f;
 for(std::uint32_t i=0;i<count;++i){auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count)return {Status::invalid_body,i};
  if(c.point_count<0||c.point_count>2||c.body1==c.body2)return {Status::invalid_snapshot,i};
  local_position_detail::PositionBinding binding(bodies[c.body1],bodies[c.body2]);auto& a=binding.a;auto& b=binding.b;math::Context m;
  const float inv_mass1=bodies[c.body1].inv_mass,inv_i1=bodies[c.body1].inv_i,inv_mass2=bodies[c.body2].inv_mass,inv_i2=bodies[c.body2].inv_i;
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
} // namespace irisu::cuda_native
