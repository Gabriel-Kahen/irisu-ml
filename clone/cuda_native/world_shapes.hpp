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
} // namespace irisu::cuda_native
