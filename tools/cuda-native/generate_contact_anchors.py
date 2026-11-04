#!/usr/bin/env python3
"""Reuse each contact's normal-pass anchors in its tangent pass."""
import hashlib
from pathlib import Path
from generate_local_velocity import extract

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'clone/cuda_native/solver.hpp'
OUTPUT = ROOT / 'clone/cuda_native/solver_velocity_contact.hpp'


def main():
    source = SOURCE.read_text()
    graph = extract(source, 'solver_velocity')
    graph = graph.replace('NATIVE_HD inline SolverResult solver_velocity(',
                          'template<bool local>NATIVE_HD inline SolverResult solver_velocity_contact(', 1)
    graph = graph.replace('auto& a=bodies[c.body1];auto& b=bodies[c.body2];',
                          'using Binding=std::conditional_t<local,local_velocity_detail::VelocityBinding,contact_anchor_detail::DirectBinding>;Binding binding(bodies[c.body1],bodies[c.body2]);auto& a=binding.a;auto& b=binding.b;VelocityAnchor anchors[2];')
    graph = graph.replace('const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;',
                          'const float inv_mass1=bodies[c.body1].inv_mass,inv_i1=bodies[c.body1].inv_i,inv_mass2=bodies[c.body2].inv_mass,inv_i2=bodies[c.body2].inv_i;')
    old = 'const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);'
    assert graph.count(old) == 2
    graph = graph.replace(old, 'const Vec2 r1=solver_detail::anchor(m,bodies[c.body1].R,p.local_anchor1),r2=solver_detail::anchor(m,bodies[c.body2].R,p.local_anchor2);anchors[j]={r1,r2};', 1)
    graph = graph.replace(old, 'const Vec2 r1=anchors[j].first,r2=anchors[j].second;', 1)
    # The original validates point_count in [0,2]. A fixed bound allows both
    # small anchor records to reside in registers without dynamic local arrays.
    loop = 'for(std::int32_t j=0;j<c.point_count;++j){'
    assert graph.count(loop) == 2
    graph = graph.replace(loop, '#ifdef __CUDACC__\n#pragma unroll\n#endif\n  for(std::int32_t j=0;j<2;++j){if(j>=c.point_count)break;')
    for name in ('relative_velocity', 'apply_velocity'):
        graph = graph.replace('solver_detail::' + name, 'contact_anchor_detail::' + name + '<local>')
    license_text = source[:source.index('// Altered implementation:')]
    result = license_text + f'''// Altered generated specialization: contact-local velocity anchor reuse.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_contact_anchors.py
#include "solver_velocity_local.hpp"
namespace irisu::cuda_native {{
namespace contact_anchor_detail {{
struct DirectBinding {{BodyState& a;BodyState& b;NATIVE_HD DirectBinding(BodyState& x,BodyState& y):a(x),b(y){{}}}};
template<bool local,class Body>NATIVE_HD inline Vec2 relative_velocity(math::Context& m,const Body& a,const Body& b,Vec2 r1,Vec2 r2){{
 if constexpr(local)return local_velocity_detail::relative_velocity(m,a,b,r1,r2);
 else return solver_detail::relative_velocity(m,a,b,r1,r2);
}}
template<bool local,class Body>NATIVE_HD inline void apply_velocity(math::Context& m,Body& a,Body& b,Vec2 r1,Vec2 r2,Vec2 P,float inv_mass1,float inv_i1,float inv_mass2,float inv_i2){{
 if constexpr(local)local_velocity_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
 else solver_detail::apply_velocity(m,a,b,r1,r2,P,inv_mass1,inv_i1,inv_mass2,inv_i2);
}}
}} // namespace contact_anchor_detail
{graph}
}} // namespace irisu::cuda_native
'''
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == '__main__':
    main()
