#!/usr/bin/env python3
"""Keep per-contact velocity state local without changing the solver graph."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'clone/cuda_native/solver.hpp'
OUTPUT = ROOT / 'clone/cuda_native/solver_velocity_local.hpp'


def extract(source, name):
    start = source.index('NATIVE_HD inline ', source.index('namespace solver_detail'))
    while name + '(' not in source[start:source.index('{', start)]:
        start = source.index('NATIVE_HD inline ', start + 1)
    opening = source.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    source = SOURCE.read_text()
    helpers = '\n'.join(extract(source, name).replace('BodyState', 'VelocityState')
                        for name in ('relative_velocity', 'apply_velocity'))
    graph = extract(source, 'solver_velocity')
    graph = graph.replace('NATIVE_HD inline SolverResult solver_velocity(',
                          'template<bool cached>NATIVE_HD inline SolverResult solver_velocity_local(', 1)
    graph = graph.replace('std::uint32_t count){',
                          'std::uint32_t count,const VelocityAnchor* anchors=nullptr){', 1)
    graph = graph.replace('auto& a=bodies[c.body1];auto& b=bodies[c.body2];',
                          'local_velocity_detail::VelocityBinding binding(bodies[c.body1],bodies[c.body2]);auto& a=binding.a;auto& b=binding.b;')
    graph = graph.replace('const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;',
                          'const float inv_mass1=bodies[c.body1].inv_mass,inv_i1=bodies[c.body1].inv_i,inv_mass2=bodies[c.body2].inv_mass,inv_i2=bodies[c.body2].inv_i;')
    old = 'const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);'
    assert graph.count(old) == 2
    graph = graph.replace(old, 'const Vec2 r1=cached?anchors[2*i+j].first:solver_detail::anchor(m,bodies[c.body1].R,p.local_anchor1),r2=cached?anchors[2*i+j].second:solver_detail::anchor(m,bodies[c.body2].R,p.local_anchor2);')
    for name in ('relative_velocity', 'apply_velocity'):
        graph = graph.replace('solver_detail::' + name, 'local_velocity_detail::' + name)
    license_text = source[:source.index('// Altered implementation:')]
    result = license_text + f'''// Altered generated specialization: defer velocity writeback to contact exit.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_local_velocity.py
#include "solver_velocity_cached.hpp"
namespace irisu::cuda_native {{
namespace local_velocity_detail {{
struct VelocityState {{Vec2 velocity;float angular_velocity;}};
// Contacts have distinct body indices. Commit on every exit, including numeric
// rejection, to retain the reference's complete partial-write state.
struct VelocityBinding {{
 BodyState& first;BodyState& second;VelocityState a,b;
 NATIVE_HD VelocityBinding(BodyState& x,BodyState& y):first(x),second(y),a{{x.velocity,x.angular_velocity}},b{{y.velocity,y.angular_velocity}}{{}}
 NATIVE_HD ~VelocityBinding(){{first.velocity=a.velocity;first.angular_velocity=a.angular_velocity;second.velocity=b.velocity;second.angular_velocity=b.angular_velocity;}}
}};
using ::irisu::cuda_native::solver_detail::cross_store;
{helpers}
}} // namespace local_velocity_detail
{graph}
}} // namespace irisu::cuda_native
'''
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == '__main__':
    main()
