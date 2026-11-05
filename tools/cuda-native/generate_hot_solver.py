#!/usr/bin/env python3
"""Retain the exact solver graph while specializing its body storage type."""
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "clone/cuda_native/solver.hpp"
OUTPUT = ROOT / "clone/cuda_native/solver_hot.hpp"


def extract(source, name):
    start = re.search(r"NATIVE_HD inline [^\n]*\b" + name + r"\(", source).start()
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return re.sub(r"\bBodyState\b", "BodyHot", source[start:end])


def main():
    source = SOURCE.read_text()
    license_text = source[:source.index("// Altered implementation:")]
    helpers = "\n".join(extract(source, name) for name in ("relative_velocity", "apply_velocity"))
    stages = "\n".join(extract(source, name) for name in ("solver_velocity", "solver_presolve", "solver_position"))
    offsets = "\n".join(f"static_assert(offsetof(BodyHot,{field})==offsetof(BodyState,{field}));" for field in
                        ("position", "velocity", "rotation", "angular_velocity", "inv_mass", "inv_i", "R"))
    result = license_text + f"""// Altered generated specialization: 48-byte hot bodies, unchanged arithmetic.
// Source solver.hpp SHA256: {hashlib.sha256(source.encode()).hexdigest()}
// Regenerate: python tools/cuda-native/generate_hot_solver.py
#include "solver.hpp"
#include <cstddef>
namespace irisu::cuda_native::interleaved_solver::hot {{
struct BodyHot {{
 Vec2 position,velocity;
 float rotation,angular_velocity,inv_mass,inv_i;
 Mat22 R;
}};
static_assert(sizeof(BodyHot)==48&&std::is_standard_layout_v<BodyHot>);
{offsets}
namespace solver_detail {{
using ::irisu::cuda_native::solver_detail::anchor;
using ::irisu::cuda_native::solver_detail::dot_store;
using ::irisu::cuda_native::solver_detail::cross_store;
{helpers}
}}
{stages}
}} // namespace irisu::cuda_native::interleaved_solver::hot
"""
    if not OUTPUT.exists() or OUTPUT.read_text() != result:
        OUTPUT.write_text(result)


if __name__ == "__main__":
    main()
