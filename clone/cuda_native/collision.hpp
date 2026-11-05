#pragma once
/*
* Copyright (c) 2007 Erin Catto http://www.gphysics.com
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
// Altered GPU/host implementation. The operation graph and binary32 stores
// follow pinned MSVC b2CollideCircle at 0x45f0..0x4725, not C++ float evaluation.
#include "solver_math.hpp"

namespace irisu::cuda_native {
namespace collision_detail {
NATIVE_HD inline Vec2 subtract(math::Context& m,Vec2 a,Vec2 b){
  return {m.sub_store(a.x,b.x),m.sub_store(a.y,b.y)};
}
NATIVE_HD inline Vec2 add(math::Context& m,Vec2 a,Vec2 b){
  return {m.add_store(a.x,b.x),m.add_store(a.y,b.y)};
}
NATIVE_HD inline Vec2 scale(math::Context& m,float a,Vec2 b){
  return {m.mul_store(a,b.x),m.mul_store(a,b.y)};
}
NATIVE_HD inline float dot(math::Context& m,Vec2 a,Vec2 b){
  return m.add_store(m.mul(a.x,b.x),m.mul(a.y,b.y));
}
NATIVE_HD inline Vec2 rotate(math::Context& m,const Mat22& R,Vec2 v){
  return {m.add_store(m.mul(R.col1.x,v.x),m.mul(R.col2.x,v.y)),
          m.add_store(m.mul(R.col1.y,v.x),m.mul(R.col2.y,v.y))};
}
NATIVE_HD inline Vec2 inverse_rotate(math::Context& m,const Mat22& R,Vec2 v){
  return {dot(m,v,R.col1),dot(m,v,R.col2)};
}
NATIVE_HD inline float normalize(math::Context& m,Vec2& v){
  const float length=m.store(m.sqrt(dot(m,v,v)));
  if(length<0x1p-23f)return 0.0f;
  const float inverse=m.div_store(1,length);v=scale(m,inverse,v);return length;
}
NATIVE_HD inline bool finite(Vec2 v){return math::finite(v.x)&&math::finite(v.y);}
NATIVE_HD inline void point(math::Context& m,Manifold& out,const ShapeState& circle,float separation){
  out.point_count=1;out.points[0].separation=separation;
  out.points[0].position=subtract(m,circle.position,scale(m,circle.radius,out.normal));
}
struct ClipVertex {Vec2 position{};std::uint32_t id{};};
NATIVE_HD inline unsigned clip(math::Context& m,ClipVertex* out,const ClipVertex* in,Vec2 normal,float offset){
  const float a=m.sub_store(dot(m,normal,in[0].position),offset);
  const float b=m.sub_store(dot(m,normal,in[1].position),offset);
  unsigned count=0;if(a<=0)out[count++]=in[0];if(b<=0)out[count++]=in[1];
  if(m.mul(a,b)<0){
    const float fraction=m.div_store(a,m.sub(a,b));
    out[count].position=add(m,in[0].position,scale(m,fraction,subtract(m,in[1].position,in[0].position)));
    out[count++].id=a>0?in[0].id:in[1].id;
  }
  return count;
}
NATIVE_HD inline float edge_separation(math::Context& m,const ShapeState& a,unsigned edge,const ShapeState& b){
  const auto normal=rotate(m,a.R,a.normals[edge]),local=inverse_rotate(m,b.R,normal);
  unsigned vertex=0;float minimum=0x1.fffffep127f;
  for(unsigned i=0;i<b.vertex_count;++i){const float value=dot(m,b.vertices[i],local);if(value<minimum){minimum=value;vertex=i;}}
  const auto first=add(m,a.position,rotate(m,a.R,a.vertices[edge]));
  const auto second=add(m,b.position,rotate(m,b.R,b.vertices[vertex]));
  return dot(m,subtract(m,second,first),normal);
}
NATIVE_HD inline float max_separation(math::Context& m,unsigned& selected,const ShapeState& a,const ShapeState& b,bool conservative){
  const auto local=inverse_rotate(m,a.R,subtract(m,b.position,a.position));
  unsigned edge=0;float maximum=-0x1.fffffep127f;
  for(unsigned i=0;i<a.vertex_count;++i){const float value=dot(m,a.normals[i],local);if(value>maximum){maximum=value;edge=i;}}
  float current=edge_separation(m,a,edge,b);if(current>0&&!conservative)return current;
  const auto previous=edge?edge-1:a.vertex_count-1,next=edge+1<a.vertex_count?edge+1:0;
  const float before=edge_separation(m,a,previous,b);if(before>0&&!conservative)return before;
  const float after=edge_separation(m,a,next,b);if(after>0&&!conservative)return after;
  int direction=0;unsigned best=edge;float separation=current;
  if(before>current&&before>after){direction=-1;best=previous;separation=before;}
  else if(after>current){direction=1;best=next;separation=after;}
  else{selected=edge;return current;}
  for(unsigned step=0;step<a.vertex_count;++step){
    edge=direction<0?(best?best-1:a.vertex_count-1):(best+1<a.vertex_count?best+1:0);
    current=edge_separation(m,a,edge,b);if(current>0&&!conservative)return current;
    if(current>separation){best=edge;separation=current;}
    else{selected=best;return separation;}
  }
  m.status=Status::invalid_shape;return separation;
}
NATIVE_HD inline void incident_edge(math::Context& m,ClipVertex* out,const ShapeState& a,unsigned edge,const ShapeState& b){
  auto direction=subtract(m,a.vertices[edge+1<a.vertex_count?edge+1:0],a.vertices[edge]);
  Vec2 normal{direction.y,-direction.x};normalize(m,normal);
  const auto local=inverse_rotate(m,b.R,rotate(m,a.R,normal));
  unsigned first=0,second=0;float minimum=0x1.fffffep127f;
  for(unsigned i=0;i<b.vertex_count;++i){
    const auto next=i+1<b.vertex_count?i+1:0;
    direction=subtract(m,b.vertices[next],b.vertices[i]);normal={direction.y,-direction.x};normalize(m,normal);
    const float value=dot(m,normal,local);if(value<minimum){minimum=value;first=i;second=next;}
  }
  out[0]={add(m,b.position,rotate(m,b.R,b.vertices[first])),edge|(first<<8)|(first<<16)};
  out[1]={add(m,b.position,rotate(m,b.R,b.vertices[second])),edge|(first<<8)|(second<<16)};
}
}
NATIVE_HD inline StageResult collide_circle(Manifold& manifold,const ShapeState& a,
                                           const ShapeState& b,bool conservative=false) {
  if(a.kind!=ShapeKind::circle||b.kind!=ShapeKind::circle)
    return {Status::unsupported_collision};
  if(!math::finite(a.position.x)||!math::finite(a.position.y)||
     !math::finite(b.position.x)||!math::finite(b.position.y)||
     !math::finite(a.radius)||!math::finite(b.radius))return {Status::unsupported_numeric};
  if(a.radius<0||b.radius<0)return {Status::invalid_shape};
  auto out=manifold;out.point_count=0;math::Context m;
  const float dx=m.sub_store(b.position.x,a.position.x);
  const float dy=m.sub_store(b.position.y,a.position.y);
  const float distance_squared=m.add_store(m.mul(dx,dx),m.mul(dy,dy));
  const float radius_sum=m.add_store(a.radius,b.radius);
  // This comparison consumes the unrounded PC53 radius product.
  const double threshold=m.mul(radius_sum,radius_sum);
  if(!m.ok())return {m.status};
  if(distance_squared>threshold&&!conservative){manifold=out;return {};}
  float separation;
  if(distance_squared<0x1p-23f){
    separation=-radius_sum;out.normal={0,1};
  }else{
    const float distance=m.store(m.sqrt(distance_squared));
    separation=m.sub_store(distance,radius_sum);
    const float inverse=m.div_store(1,distance);
    out.normal={m.mul_store(inverse,dx),m.mul_store(inverse,dy)};
  }
  // Native vector multiplication stores both components before subtraction.
  const Vec2 radius_normal{m.mul_store(b.radius,out.normal.x),
                           m.mul_store(b.radius,out.normal.y)};
  out.points[0].position={m.sub_store(b.position.x,radius_normal.x),
                          m.sub_store(b.position.y,radius_normal.y)};
  out.points[0].separation=separation;out.points[0].id=0;out.point_count=1;
  if(!m.ok())return {m.status};
  manifold=out;return {};
}

// Pinned b2CollidePolyAndCircle 0x4730..0x4bb0. Vertex and normal arrays are
// already in native centroid-local coordinates, and R/position are stored caches.
NATIVE_HD inline StageResult collide_poly_circle(Manifold& manifold,const ShapeState& poly,
                                                const ShapeState& circle,bool=false){
  using namespace collision_detail;
  if(poly.kind!=ShapeKind::polygon||circle.kind!=ShapeKind::circle)return {Status::unsupported_collision};
  if(!poly.vertex_count||poly.vertex_count>8||circle.radius<0)return {Status::invalid_shape};
  if(!finite(poly.position)||!finite(circle.position)||!finite(poly.R.col1)||!finite(poly.R.col2)||
     !math::finite(circle.radius))return {Status::unsupported_numeric};
  for(unsigned i=0;i<poly.vertex_count;++i)if(!finite(poly.vertices[i])||!finite(poly.normals[i]))return {Status::unsupported_numeric};
  auto out=manifold;out.point_count=0;math::Context m;
  const Vec2 local=inverse_rotate(m,poly.R,subtract(m,circle.position,poly.position));
  unsigned normal_index=0;float separation=-0x1.fffffep127f;
  for(unsigned i=0;i<poly.vertex_count;++i){
    const float value=dot(m,poly.normals[i],subtract(m,local,poly.vertices[i]));
    if(!m.ok())return {m.status};
    if(value>circle.radius){manifold=out;return {};}
    if(value>separation){separation=value;normal_index=i;}
  }
  if(separation<0x1p-23f){
    out.normal=rotate(m,poly.R,poly.normals[normal_index]);
    out.points[0].id=0x00ff00ffU|(normal_index<<8);
    point(m,out,circle,m.sub_store(separation,circle.radius));
  }else{
    const auto next=normal_index+1<poly.vertex_count?normal_index+1:0;
    Vec2 edge=subtract(m,poly.vertices[next],poly.vertices[normal_index]);
    const float length=normalize(m,edge);
    Vec2 direction;
    if(length<0x1p-23f){
      direction=subtract(m,local,poly.vertices[normal_index]);
      const float distance=normalize(m,direction);
      if(!m.ok())return {m.status};
      if(distance>circle.radius){manifold=out;return {};}
      out.normal=rotate(m,poly.R,direction);out.points[0].id=0x0000ffffU|(normal_index<<16);
      point(m,out,circle,m.sub_store(distance,circle.radius));
    }else{
      const float projection=dot(m,subtract(m,local,poly.vertices[normal_index]),edge);
      Vec2 closest;out.points[0].id=0x00ffffffU;
      if(projection<=0){closest=poly.vertices[normal_index];out.points[0].id=0x0000ffffU|(normal_index<<16);}
      else if(projection>=length){closest=poly.vertices[next];out.points[0].id=0x0000ffffU|(next<<16);}
      else{closest=add(m,poly.vertices[normal_index],scale(m,projection,edge));out.points[0].id=0x00ff00ffU|(normal_index<<8);}
      direction=subtract(m,local,closest);const float distance=normalize(m,direction);
      if(!m.ok())return {m.status};
      if(distance>circle.radius){manifold=out;return {};}
      out.normal=rotate(m,poly.R,direction);point(m,out,circle,m.sub_store(distance,circle.radius));
    }
  }
  if(!m.ok())return {m.status};
  manifold=out;return {};
}

// Original r58 SAT edge choice and clipping order, with native float stores.
NATIVE_HD inline StageResult collide_poly(Manifold& manifold,const ShapeState& a,const ShapeState& b,bool conservative=false){
  using namespace collision_detail;
  if(a.kind!=ShapeKind::polygon||b.kind!=ShapeKind::polygon)return {Status::unsupported_collision};
  const ShapeState* polygons[2]={&a,&b};
  for(const auto* p:polygons){
    if(!p->vertex_count||p->vertex_count>8)return {Status::invalid_shape};
    if(!finite(p->position)||!finite(p->R.col1)||!finite(p->R.col2))return {Status::unsupported_numeric};
    for(unsigned i=0;i<p->vertex_count;++i)if(!finite(p->vertices[i])||!finite(p->normals[i]))return {Status::unsupported_numeric};
  }
  auto out=manifold;out.point_count=0;math::Context m;unsigned edge_a=0,edge_b=0;
  const float separation_a=max_separation(m,edge_a,a,b,conservative);
  if(!m.ok())return {m.status};
  if(separation_a>0&&!conservative){manifold=out;return {};}
  const float separation_b=max_separation(m,edge_b,b,a,conservative);
  if(!m.ok())return {m.status};
  if(separation_b>0&&!conservative){manifold=out;return {};}
  // Native b2CollidePoly stores the reference-face threshold to binary32
  // before comparing (0x5785); retaining the PC53 sum changes near ties.
  const float reference_threshold=m.add_store(m.mul(0.98f,separation_a),0.001f);
  const bool flip=separation_b>reference_threshold;
  const auto& first=flip?b:a;const auto& second=flip?a:b;const auto edge=flip?edge_b:edge_a;
  ClipVertex incident[2]{},clipped1[2]{},clipped2[2]{};incident_edge(m,incident,first,edge,second);
  auto v1=first.vertices[edge],v2=first.vertices[edge+1<first.vertex_count?edge+1:0];
  auto side=rotate(m,first.R,subtract(m,v2,v1));normalize(m,side);const Vec2 front{side.y,-side.x};
  v1=add(m,first.position,rotate(m,first.R,v1));v2=add(m,first.position,rotate(m,first.R,v2));
  const float front_offset=dot(m,front,v1),side1=-dot(m,side,v1),side2=dot(m,side,v2);
  if(clip(m,clipped1,incident,{-side.x,-side.y},side1)<2){if(!m.ok())return {m.status};manifold=out;return {};}
  if(clip(m,clipped2,clipped1,side,side2)<2){if(!m.ok())return {m.status};manifold=out;return {};}
  out.normal=flip?Vec2{-front.x,-front.y}:front;
  for(unsigned i=0;i<2;++i){
    const float separation=m.sub_store(dot(m,front,clipped2[i].position),front_offset);
    if(separation<=0||conservative){
      auto& point=out.points[out.point_count++];point.separation=separation;point.position=clipped2[i].position;
      point.id=(clipped2[i].id&0x00ffffffU)|(std::uint32_t(flip)<<24);
    }
  }
  if(!m.ok())return {m.status};manifold=out;return {};
}
NATIVE_HD inline StageResult collide(Manifold& manifold,const ShapeState& a,const ShapeState& b,bool conservative=false){
  if(a.kind==ShapeKind::circle)return collide_circle(manifold,a,b,conservative);
  if(b.kind==ShapeKind::circle)return collide_poly_circle(manifold,a,b,conservative);
  return collide_poly(manifold,a,b,conservative);
}
static_assert(sizeof(Manifold)==60);
} // namespace irisu::cuda_native
