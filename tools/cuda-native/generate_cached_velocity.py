#!/usr/bin/env python3
"""Specialize the unchanged velocity solver to precomputed world anchors."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'clone/cuda_native/solver.hpp'
OUTPUT = ROOT / 'clone/cuda_native/solver_velocity_cached.hpp'


def main():
    source = SOURCE.read_text()
    start = source.index('NATIVE_HD inline SolverResult solver_velocity(')
    end = source.index('// Input order is native island/contact traversal order', start)
    graph = source[start:end].strip()
    graph = graph.replace('solver_velocity(', 'solver_velocity_cached(', 1)
    graph = graph.replace('std::uint32_t count){',
                          'std::uint32_t count,const VelocityAnchor* anchors){', 1)
    old = 'const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);'
    assert graph.count(old) == 2
    graph = graph.replace(old, 'const auto& cached=anchors[2*i+j];const Vec2 r1=cached.first,r2=cached.second;')
    license_text = source[:source.index('// Altered implementation:')]
    result = license_text + f'''// Altered generated specialization: reuse exact anchors during velocity only.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_cached_velocity.py
#include "solver.hpp"
namespace irisu::cuda_native {{
struct VelocityAnchor {{Vec2 first,second;}};
// Build before mutating bodies/constraints. A failed cache selects the original
// solver, preserving its validation and partial numeric-failure writes.
NATIVE_HD inline bool build_velocity_anchors(const BodyState* bodies,unsigned body_count,
 const ContactConstraint* constraints,unsigned count,VelocityAnchor* anchors){{
 for(unsigned i=0;i<count;++i){{const auto& c=constraints[i];
  if(c.body1>=body_count||c.body2>=body_count||c.body1==c.body2||c.point_count<0||c.point_count>2)return false;
  math::Context m;
  for(int j=0;j<c.point_count;++j){{const auto& p=c.points[j];
   anchors[2*i+j]={{solver_detail::anchor(m,bodies[c.body1].R,p.local_anchor1),
                   solver_detail::anchor(m,bodies[c.body2].R,p.local_anchor2)}};
   if(!m.ok())return false;
  }}
 }}
 return true;
}}
{graph}
}} // namespace irisu::cuda_native
'''
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == '__main__':
    main()
