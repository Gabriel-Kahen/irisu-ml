#!/usr/bin/env python3
"""Keep per-contact position state local without changing the solver graph."""
import hashlib
from pathlib import Path
from generate_local_velocity import extract

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'clone/cuda_native/solver.hpp'
OUTPUT = ROOT / 'clone/cuda_native/solver_position_local.hpp'


def main():
    source = SOURCE.read_text()
    graph = extract(source, 'solver_position').replace('solver_position(', 'solver_position_local(', 1)
    graph = graph.replace('auto& a=bodies[c.body1];auto& b=bodies[c.body2];',
                          'local_position_detail::PositionBinding binding(bodies[c.body1],bodies[c.body2]);auto& a=binding.a;auto& b=binding.b;')
    graph = graph.replace('const float inv_mass1=a.inv_mass,inv_i1=a.inv_i,inv_mass2=b.inv_mass,inv_i2=b.inv_i;',
                          'const float inv_mass1=bodies[c.body1].inv_mass,inv_i1=bodies[c.body1].inv_i,inv_mass2=bodies[c.body2].inv_mass,inv_i2=bodies[c.body2].inv_i;')
    license_text = source[:source.index('// Altered implementation:')]
    result = license_text + f'''// Altered generated specialization: defer position writeback to contact exit.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_local_position.py
#include "solver.hpp"
namespace irisu::cuda_native {{
namespace local_position_detail {{
struct PositionState {{Vec2 position;float rotation;Mat22 R;}};
// Distinct contact participants are checked before construction. Every exit
// commits the reference's position/rotation partial writes, including errors.
struct PositionBinding {{
 BodyState& first;BodyState& second;PositionState a,b;
 NATIVE_HD PositionBinding(BodyState& x,BodyState& y):first(x),second(y),a{{x.position,x.rotation,x.R}},b{{y.position,y.rotation,y.R}}{{}}
 NATIVE_HD ~PositionBinding(){{first.position=a.position;first.rotation=a.rotation;first.R=a.R;second.position=b.position;second.rotation=b.rotation;second.R=b.R;}}
}};
}} // namespace local_position_detail
{graph}
}} // namespace irisu::cuda_native
'''
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == '__main__':
    main()
