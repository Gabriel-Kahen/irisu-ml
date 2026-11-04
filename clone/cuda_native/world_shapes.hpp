#pragma once
#include "solver_math.hpp"

namespace irisu::cuda_native {
// Native circle and polygon QuickSync share this operation graph; their local
// position/centroid is stored at the same offsets64/68 in the pinned build.
NATIVE_HD inline StageResult shape_quick_sync(ShapeState& shape,Vec2 position,const Mat22& R) {
  math::Context m;const auto local=shape.kind==ShapeKind::circle?shape.local_position:shape.local_centroid;
  const Vec2 rotated{
    m.store(m.add(m.mul(R.col2.x,local.y),m.mul(local.x,R.col1.x))),
    m.store(m.add(m.mul(R.col2.y,local.y),m.mul(R.col1.y,local.x)))};
  const Vec2 result{m.store(m.add(position.x,rotated.x)),m.store(m.add(position.y,rotated.y))};
  if(!m.ok())return {m.status};
  shape.R=R;shape.position=result;return {};
}
NATIVE_HD inline StageResult circle_swept_bounds(AABB& bounds,const ShapeState& shape,
                                                Vec2 previous,const Mat22& previous_R) {
  if(shape.kind!=ShapeKind::circle)return {Status::unsupported_collision};
  auto old=shape;const auto result=shape_quick_sync(old,previous,previous_R);if(!result)return result;
  math::Context m;
  // Native b2Min/b2Max choose the second operand on equal values, including
  // differently signed zeros. Keep these comparisons rather than fmin/fmax.
  const Vec2 low{old.position.x<shape.position.x?old.position.x:shape.position.x,
                 old.position.y<shape.position.y?old.position.y:shape.position.y};
  const Vec2 high{old.position.x>shape.position.x?old.position.x:shape.position.x,
                  old.position.y>shape.position.y?old.position.y:shape.position.y};
  bounds.min={m.store(m.sub(low.x,shape.radius)),m.store(m.sub(low.y,shape.radius))};
  bounds.max={m.store(m.add(shape.radius,high.x)),m.store(m.add(shape.radius,high.y))};
  return {m.status};
}
NATIVE_HD inline bool aabb_overlap(const AABB& a,const AABB& b) {
  return a.max.x>=b.min.x&&b.max.x>=a.min.x&&a.max.y>=b.min.y&&b.max.y>=a.min.y;
}
NATIVE_HD inline float shape_abs(float f){union{float f;std::uint32_t u;}b{f};b.u&=0x7fffffffU;return b.f;}
NATIVE_HD inline Vec2 shape_rotate(math::Context& m,const Mat22& R,Vec2 v){
  return {m.store(m.add(m.mul(R.col2.x,v.y),m.mul(v.x,R.col1.x))),
          m.store(m.add(m.mul(R.col1.y,v.x),m.mul(R.col2.y,v.y)))};
}
NATIVE_HD inline StageResult polygon_bounds(AABB& bounds,const ShapeState& shape,Vec2 position,const Mat22& R){
  if(shape.kind!=ShapeKind::polygon)return {Status::unsupported_collision};
  math::Context m;
  const Mat22 obb{shape_rotate(m,R,shape.local_obb_R.col1),shape_rotate(m,R,shape.local_obb_R.col2)};
  const Mat22 absolute{{shape_abs(obb.col1.x),shape_abs(obb.col1.y)},{shape_abs(obb.col2.x),shape_abs(obb.col2.y)}};
  const auto half=shape_rotate(m,absolute,shape.local_obb_extents);
  const Vec2 local{m.store(m.add(shape.local_centroid.x,shape.local_obb_center.x)),
                   m.store(m.add(shape.local_centroid.y,shape.local_obb_center.y))};
  const auto rotated=shape_rotate(m,R,local);
  const Vec2 center{m.store(m.add(position.x,rotated.x)),m.store(m.add(position.y,rotated.y))};
  bounds.min={m.store(m.sub(center.x,half.x)),m.store(m.sub(center.y,half.y))};
  bounds.max={m.store(m.add(center.x,half.x)),m.store(m.add(center.y,half.y))};
  return {m.status};
}
NATIVE_HD inline StageResult polygon_swept_bounds(AABB& bounds,const ShapeState& shape,
  Vec2 previous,const Mat22& previous_R,Vec2 position,const Mat22& R){
  AABB a{},b{};auto result=polygon_bounds(a,shape,previous,previous_R);if(!result)return result;
  result=polygon_bounds(b,shape,position,R);if(!result)return result;
  bounds.min={a.min.x<b.min.x?a.min.x:b.min.x,a.min.y<b.min.y?a.min.y:b.min.y};
  bounds.max={a.max.x>b.max.x?a.max.x:b.max.x,a.max.y>b.max.y?a.max.y:b.max.y};
  return {};
}
NATIVE_HD inline StageResult shape_creation_bounds(AABB& bounds,const ShapeState& shape){
  math::Context m;
  if(shape.kind==ShapeKind::circle){
    bounds.min={m.store(m.sub(shape.position.x,shape.radius)),m.store(m.sub(shape.position.y,shape.radius))};
    bounds.max={m.store(m.add(shape.position.x,shape.radius)),m.store(m.add(shape.position.y,shape.radius))};
  }else if(shape.kind==ShapeKind::polygon){
    const Mat22 obb{shape_rotate(m,shape.R,shape.local_obb_R.col1),shape_rotate(m,shape.R,shape.local_obb_R.col2)};
    const Mat22 absolute{{shape_abs(obb.col1.x),shape_abs(obb.col1.y)},{shape_abs(obb.col2.x),shape_abs(obb.col2.y)}};
    const auto half=shape_rotate(m,absolute,shape.local_obb_extents),offset=shape_rotate(m,shape.R,shape.local_obb_center);
    const Vec2 center{m.store(m.add(shape.position.x,offset.x)),m.store(m.add(shape.position.y,offset.y))};
    bounds.min={m.store(m.sub(center.x,half.x)),m.store(m.sub(center.y,half.y))};
    bounds.max={m.store(m.add(center.x,half.x)),m.store(m.add(center.y,half.y))};
  }else return {Status::invalid_shape};
  return {m.status};
}
} // namespace irisu::cuda_native
