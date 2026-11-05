#!/usr/bin/env python3
"""Retain exact contact-local anchor reuse in the GPU hot-body graph."""
import hashlib
from pathlib import Path
from generate_hot_solver import extract

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'clone/cuda_native/solver.hpp'
OUTPUT = ROOT / 'clone/cuda_native/solver_persistent_velocity.hpp'


def main():
    source = SOURCE.read_text()
    graph = extract(source, 'solver_velocity')
    graph = graph.replace('solver_velocity(', 'solver_velocity_contact(', 1)
    old = 'const Vec2 r1=solver_detail::anchor(m,a.R,p.local_anchor1),r2=solver_detail::anchor(m,b.R,p.local_anchor2);'
    assert graph.count(old) == 2
    graph = graph.replace('auto& a=bodies[c.body1];auto& b=bodies[c.body2];math::Context m;',
                          'auto& a=bodies[c.body1];auto& b=bodies[c.body2];Anchor anchors[2];math::Context m;')
    graph = graph.replace(old, '__PERSISTENT_FIRST_ANCHOR__', 1)
    graph = graph.replace(old, 'const Vec2 r1=anchors[j].first,r2=anchors[j].second;', 1)
    graph = graph.replace('__PERSISTENT_FIRST_ANCHOR__', old + 'anchors[j]={r1,r2};', 1)
    loop = 'for(std::int32_t j=0;j<c.point_count;++j){'
    assert graph.count(loop) == 2
    graph = graph.replace(loop, '#ifdef __CUDACC__\n#pragma unroll\n#endif\n  for(std::int32_t j=0;j<2;++j){if(j>=c.point_count)break;')
    result = source[:source.index('// Altered implementation:')] + f'''// Altered generated specialization: exact contact anchors in register hot bodies.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_persistent_velocity.py
#include "solver_hot.hpp"
namespace irisu::cuda_native::persistent_solver::hot_velocity {{
using BodyHot=::irisu::cuda_native::interleaved_solver::hot::BodyHot;
namespace solver_detail=::irisu::cuda_native::interleaved_solver::hot::solver_detail;
struct Anchor {{Vec2 first,second;}};
{graph}
}} // namespace irisu::cuda_native::persistent_solver::hot_velocity
'''
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == '__main__':
    main()
