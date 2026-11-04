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
// Altered independent typed construction. It computes physical state only;
// World owns slots, intrusive order and separately qualified proxy creation.
#include "world.hpp"
#include "collision.hpp"

namespace irisu::cuda_native {
struct ShapeDefinition {
  ShapeKind kind{ShapeKind::circle};
  Vec2 local_position{},vertices[8]{};
  float local_rotation{},radius{},density{1},friction{.2f},restitution{};
  std::uint32_t vertex_count{};
  std::uint16_t category_bits{1},mask_bits{0xffff};std::int16_t group_index{};
};
struct BodyDefinition {
  Vec2 position{},velocity{};float rotation{},angular_velocity{},linear_damping{},angular_damping{};
  bool allow_sleep{true},prevent_rotation{},is_sleeping{};std::uint32_t user_data{};
};
struct MassData {float mass{};Vec2 center{};float inertia{};};
namespace factory_detail {
using namespace collision_detail;
NATIVE_HD inline float cross(math::Context& m,Vec2 a,Vec2 b){return m.store(m.sub(m.mul(a.x,b.y),m.mul(a.y,b.x)));}
NATIVE_HD inline MassData polygon_mass(math::Context& m,const ShapeDefinition& s){
  MassData out;float area=0,inertia=0;Vec2 center{};constexpr float inverse3=0x1.555556p-2f;
  for(unsigned i=0;i<s.vertex_count;++i){
    const Vec2 p1{},p2=s.vertices[i],p3=s.vertices[i+1<s.vertex_count?i+1:0];
    const auto e1=subtract(m,p2,p1),e2=subtract(m,p3,p1);
    const float D=cross(m,e1,e2),triangle_area=m.store(m.mul(.5f,D));
    area=m.store(m.add(area,triangle_area));
    center=add(m,center,scale(m,m.store(m.mul(triangle_area,inverse3)),add(m,add(m,p1,p2),p3)));
    const auto integral=[&](float p,float a,float b){
      const double squares=m.add(m.add(m.mul(a,a),m.mul(b,a)),m.mul(b,b));
      const double first=m.mul(inverse3,m.add(m.mul(.25f,squares),m.add(m.mul(p,a),m.mul(p,b))));
      return m.store(m.add(first,m.mul(m.mul(.5f,p),p)));
    };
    const float x2=integral(p1.x,e1.x,e2.x),y2=integral(p1.y,e1.y,e2.y);
    inertia=m.store(m.add(inertia,m.mul(D,m.add(x2,y2))));
  }
  if(!(area>0x1p-23f)){m.status=Status::invalid_shape;return {};}
  out.mass=m.store(m.mul(s.density,area));center=scale(m,m.store(m.div(1,area)),center);out.center=center;
  out.inertia=m.store(m.mul(s.density,m.sub(inertia,m.mul(area,dot(m,center,center)))));return out;
}
NATIVE_HD inline Vec2 polygon_centroid(math::Context& m,const ShapeDefinition& s){
  float area=0;Vec2 center{};constexpr float inverse3=0x1.555556p-2f;
  for(unsigned i=0;i<s.vertex_count;++i){
    const Vec2 zero{},a=subtract(m,s.vertices[i],zero),b=subtract(m,s.vertices[i+1<s.vertex_count?i+1:0],zero);
    const float triangle_area=m.store(m.mul(.5f,cross(m,a,b)));area=m.store(m.add(area,triangle_area));
    center=add(m,center,scale(m,m.store(m.mul(triangle_area,inverse3)),add(m,add(m,zero,s.vertices[i]),s.vertices[i+1<s.vertex_count?i+1:0])));
  }
  if(!(area>0x1p-23f)){m.status=Status::invalid_shape;return {};}
  return scale(m,m.store(m.div(1,area)),center);
}
NATIVE_HD inline bool finite_definition(const ShapeDefinition& s,const BodyDefinition& b){
  if(!finite(s.local_position)||!math::finite(s.local_rotation)||!math::finite(s.density)||
     !math::finite(s.friction)||!math::finite(s.restitution)||!finite(b.position)||!finite(b.velocity)||
     !math::finite(b.rotation)||!math::finite(b.angular_velocity)||!math::finite(b.linear_damping)||!math::finite(b.angular_damping))return false;
  return true;
}
}

NATIVE_HD inline StageResult compute_mass(MassData& result,const ShapeDefinition& s){
  if(!math::finite(s.density)||s.density<0)return {Status::unsupported_numeric};math::Context m;
  if(s.kind==ShapeKind::circle){
    if(!math::finite(s.radius)||s.radius<0)return {Status::invalid_shape};
    result.mass=m.store(m.mul(m.mul(m.mul(s.density,0x1.921fb6p1f),s.radius),s.radius));result.center={};
    result.inertia=m.store(m.mul(m.mul(m.mul(.5f,result.mass),s.radius),s.radius));
  }else if(s.kind==ShapeKind::polygon){
    if(s.vertex_count<3||s.vertex_count>8)return {Status::invalid_shape};
    for(unsigned i=0;i<s.vertex_count;++i)if(!factory_detail::finite(s.vertices[i]))return {Status::unsupported_numeric};
    result=factory_detail::polygon_mass(m,s);
  }else return {Status::invalid_shape};
  return {m.status};
}

// One native shape per body is the public Irisu wrapper's construction domain.
// Proxy creation/freeze is a separate stage; no implicit host or old-backend call.
NATIVE_HD inline StageResult construct_body_shape(BodyState& body,ShapeState& shape,
                                                const BodyDefinition& bd,const ShapeDefinition& sd){
  using namespace factory_detail;
  if(!finite_definition(sd,bd))return {Status::unsupported_numeric};
  MassData mass;auto result=compute_mass(mass,sd);if(!result)return result;math::Context m;
  BodyState b{};ShapeState s{};b.position=b.position0=bd.position;b.rotation=b.rotation0=bd.rotation;
  result=rotation_matrix(b);if(!result)return result;
  const float linear=m.store(m.sub(1,bd.linear_damping)),angular=m.store(m.sub(1,bd.angular_damping));
  b.linear_damping=linear<0?0:linear>1?1:linear;b.angular_damping=angular<0?0:angular>1?1:angular;
  b.mass=m.store(m.add(0,mass.mass));b.center=add(m,{},scale(m,mass.mass,add(m,sd.local_position,mass.center)));
  b.flags=0;
  if(b.mass>0){b.center=scale(m,m.store(m.div(1,b.mass)),b.center);b.position=add(m,b.position,rotate(m,b.R,b.center));}
  else b.flags|=static_body;
  const auto r=subtract(m,add(m,sd.local_position,mass.center),b.center);
  b.inertia=m.store(m.add(m.store(m.add(0,mass.inertia)),m.mul(mass.mass,dot(m,r,r))));
  b.inv_mass=b.mass>0?m.store(m.div(1,b.mass)):0;
  if(b.inertia>0&&!bd.prevent_rotation)b.inv_i=m.store(m.div(1,b.inertia));else b.inertia=b.inv_i=0;
  const Vec2 angular_velocity{m.store(m.mul(-double(bd.angular_velocity),b.center.y)),m.store(m.mul(bd.angular_velocity,b.center.x))};
  b.velocity=add(m,bd.velocity,angular_velocity);b.angular_velocity=bd.angular_velocity;
  b.flags|=bd.allow_sleep?allow_sleep:0;b.flags|=bd.is_sleeping?sleeping_body:0;b.user_data=bd.user_data;
  if((b.flags&sleeping_body)||b.inv_mass==0){b.velocity={};b.angular_velocity=0;}
  s.kind=sd.kind;s.friction=sd.friction;s.restitution=sd.restitution;s.category_bits=sd.category_bits;
  s.mask_bits=sd.mask_bits;s.group_index=sd.group_index;s.R=b.R;
  if(sd.kind==ShapeKind::circle){
    s.radius=sd.radius;s.local_position=subtract(m,sd.local_position,b.center);
    const auto offset=rotate(m,b.R,s.local_position);s.position=add(m,b.position,offset);
    s.max_radius=m.store(m.add(m.sqrt(dot(m,offset,offset)),s.radius));
  }else{
    BodyState local{};local.rotation=sd.local_rotation;result=rotation_matrix(local);if(!result)return result;
    const auto centroid=polygon_centroid(m,sd);s.local_centroid=subtract(m,add(m,sd.local_position,rotate(m,local.R,centroid)),b.center);
    s.vertex_count=sd.vertex_count;Vec2 minimum{0x1.fffffep127f,0x1.fffffep127f},maximum{-0x1.fffffep127f,-0x1.fffffep127f};
    for(unsigned i=0;i<sd.vertex_count;++i){
      const auto v=rotate(m,local.R,subtract(m,sd.vertices[i],centroid));s.vertices[i]=v;
      auto unit=v;const float length=m.store(m.sqrt(dot(m,v,v)));if(length>0x1p-23f)unit=scale(m,m.store(m.div(1,length)),unit);
      s.core_vertices[i]=subtract(m,v,scale(m,0.01f,unit));
      minimum={minimum.x<v.x?minimum.x:v.x,minimum.y<v.y?minimum.y:v.y};
      maximum={maximum.x>v.x?maximum.x:v.x,maximum.y>v.y?maximum.y:v.y};
      if(length>s.max_radius)s.max_radius=length;
    }
    s.local_obb_R={};s.local_obb_center=scale(m,.5f,add(m,minimum,maximum));s.local_obb_extents=scale(m,.5f,subtract(m,maximum,minimum));
    for(unsigned i=0;i<sd.vertex_count;++i){const auto edge=subtract(m,s.vertices[i+1<sd.vertex_count?i+1:0],s.vertices[i]);
      s.normals[i]={edge.y,-edge.x};normalize(m,s.normals[i]);}
    s.position=add(m,b.position,rotate(m,b.R,s.local_centroid));
  }
  if(!m.ok())return {m.status};body=b;shape=s;return {};
}

// Build exactly the public wrapper's circle, rectangle or right triangle defs.
NATIVE_HD inline StageResult wrapper_definition(BodyDefinition& body,ShapeDefinition& shape,unsigned kind,
  float width,float height,float x,float y,float rotation,float density,float friction,float restitution,float magnification){
  math::Context m;if(!(magnification>0)||!math::finite(magnification)||kind>2)return {Status::invalid_shape};
  body={};shape={};body.position={m.store(m.div(x,magnification)),m.store(m.div(y,magnification))};
  body.rotation=kind?rotation:0;shape.density=density;shape.friction=friction;shape.restitution=restitution;
  if(kind==0){shape.kind=ShapeKind::circle;shape.radius=m.store(m.div(width,magnification));}
  else{
    shape.kind=ShapeKind::polygon;const float hx=m.store(m.div(m.div(width,2),magnification)),hy=m.store(m.div(m.div(height,2),magnification));
    shape.vertices[0]={-hx,hy};shape.vertices[1]={-hx,-hy};
    if(kind==1){shape.vertex_count=4;shape.vertices[2]={hx,-hy};shape.vertices[3]={hx,hy};}
    else{shape.vertex_count=3;shape.vertices[2]={hx,hy};}
  }
  return {m.status};
}
} // namespace irisu::cuda_native
