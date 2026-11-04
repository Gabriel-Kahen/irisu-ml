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
// Altered implementation: typed independent CUDA/host solver. Mathematical
// algorithm derives from legacy Box2D; explicit rounding/store order follows
// pinned MSVC SolveVelocityConstraints 0x7610..0x7b60, not host C++ evaluation.
#include "solver_math.hpp"
namespace irisu::cuda_native {
struct SolverContact {std::uint32_t body1{},body2{},manifold{};float friction{},restitution{};};
struct SolverResult {
 Status status{Status::ok};std::uint32_t constraint{null_index},point{null_index};bool converged{true};
 NATIVE_HD explicit operator bool() const{return status==Status::ok;}
};
namespace solver_detail {
NATIVE_HD inline Vec2 anchor(math::Context& m,const Mat22& R,Vec2 a){
 return {m.add_store(m.mul(R.col2.x,a.y),m.mul(a.x,R.col1.x)),
         m.add_store(m.mul(R.col1.y,a.x),m.mul(R.col2.y,a.y))};
}
NATIVE_HD inline Vec2 relative_velocity(math::Context& m,const BodyState& a,const BodyState& b,Vec2 r1,Vec2 r2){
 const Vec2 cross1{m.mul_store(-double(a.angular_velocity),r1.y),m.mul_store(a.angular_velocity,r1.x)};
 const Vec2 cross2{m.mul_store(-double(b.angular_velocity),r2.y),m.mul_store(b.angular_velocity,r2.x)};
 const Vec2 at2{m.add_store(b.velocity.x,cross2.x),m.add_store(b.velocity.y,cross2.y)};
 const Vec2 subtract1{m.sub_store(at2.x,a.velocity.x),m.sub_store(at2.y,a.velocity.y)};
 return {m.sub_store(subtract1.x,cross1.x),m.sub_store(subtract1.y,cross1.y)};
}
NATIVE_HD inline float dot_store(math::Context& m,Vec2 a,Vec2 b){return m.add_store(m.mul(a.x,b.x),m.mul(a.y,b.y));}
NATIVE_HD inline float cross_store(math::Context& m,Vec2 a,Vec2 b){return m.sub_store(m.mul(a.x,b.y),m.mul(a.y,b.x));}
NATIVE_HD inline void apply_velocity(math::Context& m,BodyState& a,BodyState& b,Vec2 r1,Vec2 r2,Vec2 P,float inv_mass1,float inv_i1,float inv_mass2,float inv_i2){
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
// Constraints and points are traversed in supplied native order. Each world
// owns its arrays; parallelize worlds, not dependent contact impulses. Failure
// invalidates the stage; callers must not accept its possibly partial writes.
NATIVE_HD inline SolverResult solver_velocity(BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count){
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
// Input order is native island/contact traversal order, supplied by caller.
// Preserve unconsumed point slots/positionImpulse: native constructor does not
// initialize them; PreSolve establishes the consumed position impulse state.
NATIVE_HD inline SolverResult solver_build_constraints(const BodyState* bodies,std::uint32_t body_count,const Manifold* manifolds,std::uint32_t manifold_count,const SolverContact* contacts,std::uint32_t contact_count,ContactConstraint* output,std::uint32_t capacity,std::uint32_t& count){
 count=0;
 for(std::uint32_t i=0;i<contact_count;++i){const auto& input=contacts[i];
  if(input.body1>=body_count||input.body2>=body_count)return {Status::invalid_body,i};
  if(input.manifold>=manifold_count||input.body1==input.body2)return {Status::invalid_snapshot,i};
  const auto& manifold=manifolds[input.manifold];
  if(manifold.point_count<0||manifold.point_count>2)return {Status::invalid_snapshot,i};
  if(!manifold.point_count)continue;
  if(count>=capacity)return {Status::capacity,i};
  auto& c=output[count++];c.body1=input.body1;c.body2=input.body2;c.manifold=input.manifold;c.normal=manifold.normal;c.friction=input.friction;c.restitution=input.restitution;c.point_count=manifold.point_count;
  const auto& a=bodies[c.body1];const auto& b=bodies[c.body2];math::Context m;
  const Vec2 tangent{c.normal.y,m.mul_store(-1,c.normal.x)};
  for(std::int32_t j=0;j<c.point_count;++j){const auto& source=manifold.points[j];auto& p=c.points[j];
   p.normal_impulse=source.normal_impulse;p.tangent_impulse=source.tangent_impulse;p.separation=source.separation;
   const Vec2 r1{m.sub_store(source.position.x,a.position.x),m.sub_store(source.position.y,a.position.y)};
   const Vec2 r2{m.sub_store(source.position.x,b.position.x),m.sub_store(source.position.y,b.position.y)};
   p.local_anchor1={m.add_store(m.mul(a.R.col1.y,r1.y),m.mul(a.R.col1.x,r1.x)),m.add_store(m.mul(a.R.col2.y,r1.y),m.mul(a.R.col2.x,r1.x))};
   p.local_anchor2={m.add_store(m.mul(b.R.col1.y,r2.y),m.mul(b.R.col1.x,r2.x)),m.add_store(m.mul(b.R.col2.y,r2.y),m.mul(b.R.col2.x,r2.x))};
   const float r1_sqr=solver_detail::dot_store(m,r1,r1),r2_sqr=solver_detail::dot_store(m,r2,r2);
   const float rn1=solver_detail::dot_store(m,c.normal,r1),rn2=solver_detail::dot_store(m,c.normal,r2);
   const float mass_sum=m.add_store(b.inv_mass,a.inv_mass);
   const double rotational1=m.mul(m.sub(r1_sqr,m.mul(rn1,rn1)),a.inv_i);
   const double rotational2=m.mul(m.sub(r2_sqr,m.mul(rn2,rn2)),b.inv_i);
   const float normal_k=m.add_store(m.add(rotational1,rotational2),mass_sum);
   p.normal_mass=m.store(m.div(1,normal_k));
   const float rt1=solver_detail::dot_store(m,tangent,r1),rt2=solver_detail::dot_store(m,tangent,r2);
   const double tangent1=m.mul(m.sub(r1_sqr,m.mul(rt1,rt1)),a.inv_i);
   const double tangent2=m.mul(m.sub(r2_sqr,m.mul(rt2,rt2)),b.inv_i);
   const float tangent_k=m.add_store(m.add(tangent1,tangent2),mass_sum);
   p.tangent_mass=m.store(m.div(1,tangent_k));p.velocity_bias=0.0f;
   if(p.separation>0)p.velocity_bias=m.mul_store(p.separation,-60);
   const Vec2 dv=solver_detail::relative_velocity(m,a,b,r1,r2);
   const float relative=solver_detail::dot_store(m,c.normal,dv);
   if(relative< -1.0f)p.velocity_bias=m.sub_store(p.velocity_bias,m.mul(relative,c.restitution));
   if(!m.ok())return {m.status,i,std::uint32_t(j)};
  }
 }
 return {};
}
NATIVE_HD inline SolverResult solver_presolve(BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,bool warm_start){
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
NATIVE_HD inline SolverResult solver_postsolve(const ContactConstraint* constraints,std::uint32_t count,Manifold* manifolds,std::uint32_t manifold_count){
 for(std::uint32_t i=0;i<count;++i){const auto& c=constraints[i];
  if(c.manifold>=manifold_count||c.point_count<0||c.point_count>2)return {Status::invalid_snapshot,i};
  auto& manifold=manifolds[c.manifold];
  for(std::int32_t j=0;j<c.point_count;++j){const auto& p=c.points[j];
   if(!math::finite(p.normal_impulse)||!math::finite(p.tangent_impulse))return {Status::unsupported_numeric,i,std::uint32_t(j)};
   manifold.points[j].normal_impulse=p.normal_impulse;manifold.points[j].tangent_impulse=p.tangent_impulse;
  }
 }
 return {};
}
NATIVE_HD inline SolverResult solver_position(BodyState* bodies,std::uint32_t body_count,ContactConstraint* constraints,std::uint32_t count,float beta,bool canonical_rotation=false){
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

}
