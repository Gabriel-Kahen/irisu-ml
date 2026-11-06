#!/usr/bin/env python3
"""Derive bounded CUDA normal-mode rules from the authoritative C++ source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "clone/cuda"


def source(name: str) -> str:
    return (ROOT / name).read_text()


def extract(text: str, start: str, end: str) -> str:
    return text[text.index(start):text.index(end)]


def convert(text: str) -> str:
    replacements = {
        "std::vector<Event>": "Events",
        "std::tuple<BodyId, Vec2, double>": "ScriptedOrigin",
        "std::array": "array", "std::vector": "vector", "std::string": "Detail",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    for name in ("bit_cast", "min", "max", "clamp", "abs", "isfinite", "sqrt",
                 "trunc", "move", "sort", "unique", "binary_search", "find_if",
                 "remove_if", "get", "swap"):
        text = text.replace("std::" + name, name)
    text = text.replace("  const ScopedFloatingPointEnvironment floating_point_environment;\n", "")
    return text


def spawn_angle_bits() -> list[int]:
    program = r'''
#include "irisu/normal_rules.hpp"
#include <bit>
#include <cstdio>
int main() { for (unsigned i=0;i<=1000;++i)
  std::printf("%08x\n",std::bit_cast<unsigned>(float(irisu::normal_spawn_angle(i)))); }
'''
    with tempfile.TemporaryDirectory(prefix="irisu-spawn-angle-") as temp:
        path = Path(temp)
        (path / "main.cpp").write_text(program)
        subprocess.run(["g++-14", "-std=c++20", "-O2", "-mfpmath=387", "-ffp-contract=off",
                        "-DIRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT=1", "-I" + str(ROOT / "clone/include"),
                        str(path / "main.cpp"), str(ROOT / "clone/core/normal_rules.cpp"),
                        "-o", str(path / "angles")], check=True)
        return [int(line, 16) for line in subprocess.check_output([str(path / "angles")], text=True).splitlines()]


def types() -> str:
    math = extract(source("clone/include/irisu/math.hpp"), "struct Vec2", "}  // namespace irisu")
    math = convert(math).replace("constexpr ", "GAME_HD constexpr ").replace("inline ", "GAME_HD inline ")
    # Trigonometric Vec2 rotation is never used by the normal-rule port; physics
    # owns its own reference-matched rotation boundary.
    math = re.sub(r"GAME_HD inline Vec2 rotate\(.*?\n}\n", "", math, flags=re.S)
    math = re.sub(r"GAME_HD inline double clamp\(.*?\n}\n", "", math, flags=re.S)
    original = extract(source("clone/include/irisu/types.hpp"), "using BodyId", "}  // namespace irisu")
    original = convert(original)
    body_text = extract(original, "struct Body {", "// Public body state")
    fields = re.findall(r"^\s+(BodyId|BodyKind|Shape|Lifecycle|std::\w+|Vec2|double|ChainId|bool)\s+(\w+)[{;]", body_text, re.M)
    body_hash = ["template<class B> GAME_HD std::uint64_t body_hash(const B& body,std::uint64_t hash=14695981039346656037ULL) {"]
    for kind, name in fields:
        for component in (".x", ".y") if kind == "Vec2" else ("",):
            body_hash.append(f"  hash=fnv_scalar(hash,body.{name}{component});")
    body_hash.append("  return hash;\n}")
    original += "\n" + "\n".join(body_hash) + "\n"
    events = r'''
struct Events {
  std::uint64_t count{};
  std::uint64_t hash{14695981039346656037ULL};
  std::uint64_t sequence_base{};
  bool overflow{};
#ifdef IRISU_CUDA_GAME_EVENT_CAPTURE
  Event* capture{};std::uint32_t capture_capacity{};bool capture_overflow{};
#endif
  GAME_HD std::uint64_t size() const { return count; }
  GAME_HD void push_back(Event event) {
    if (count == UINT64_MAX || sequence_base > UINT64_MAX-count) { overflow=true; return; }
    event.sequence=sequence_base+count++;
#ifdef IRISU_CUDA_GAME_EVENT_CAPTURE
    if(capture){if(count<=capture_capacity)capture[count-1]=event;else capture_overflow=true;}
#endif
    hash=fnv_integer(hash,event.tick);
    hash=fnv_integer(hash,static_cast<std::uint8_t>(event.kind));
    hash=fnv_integer(hash,event.a); hash=fnv_integer(hash,event.b);
    hash=fnv_integer(hash,event.value); hash=fnv_integer(hash,event.detail.hash);
    hash=fnv_integer(hash,event.sequence);
  }
};
'''
    original = original.replace("struct StepDiagnostics {", events + "\nstruct StepDiagnostics {")
    config = extract(source("clone/include/irisu/config.hpp"), "struct MechanicsConfig", "// Validates every runtime field")
    config = convert(config)
    rng = extract(source("clone/include/irisu/dx_random.hpp"), "class DxRandom", "}  // namespace irisu")
    rng = convert(rng)
    rng = re.sub(r"^(  )(void|std::uint32_t|const array|std::uint32_t index)", r"\1GAME_HD \2", rng, flags=re.M)
    rng = rng.replace("GAME_HD std::uint32_t index_{", "std::uint32_t index_{")
    rng_impl = extract(source("clone/core/dx_random.cpp"), "void DxRandom::seed", "}  // namespace irisu")
    rng_impl = convert(rng_impl)
    rng_impl = rng_impl.replace('throw std::invalid_argument("DxLib GetRand maximum exceeds signed int32");', "return 0;")
    rng_impl = rng_impl.replace('throw std::invalid_argument("DxLib RNG index is outside [0,624]");', "return;")
    rng_impl = re.sub(r"^(void|std::uint32_t) DxRandom::", r"GAME_HD inline \1 DxRandom::", rng_impl, flags=re.M)
    origin = r'''
struct ScriptedOrigin { std::uint32_t id{}; Vec2 position{}; double angle{}; };
template<unsigned I> GAME_HD decltype(auto) get(const ScriptedOrigin& origin) {
  if constexpr (I==0) return (origin.id);
  else if constexpr (I==1) return (origin.position);
  else return (origin.angle);
}
'''
    return ("#pragma once\n#include \"game_support.cuh\"\nnamespace irisu::cuda_game {\n" + math + origin +
            original + config + rng + "\ninline constexpr std::size_t kPeriodOffset=397;\n"
            "inline constexpr std::uint32_t kMatrixA=0x9908b0dfU;\n" + rng_impl + "\n}\n")


def tables(angles: list[int]) -> str:
    rules = source("clone/core/normal_rules.cpp")
    definitions = []
    for kind, name, size in (("std::uint32_t", "kScriptedDescentBits", 99),
                             ("std::uint64_t", "kScoreScaleBits", 99),
                             ("std::uint32_t", "kSpawnAngleBits", 1001)):
        if name == "kSpawnAngleBits":
            init = ",\n".join("    " + ", ".join(f"0x{v:08x}U" for v in angles[i:i+8])
                              for i in range(0, len(angles), 8))
        else:
            init = re.search(r"\b" + name + r"\{(.*?)\n};", rules, re.S).group(1)
        definitions += [f"inline constexpr array<{kind},{size}> {name}Host{{{init}\n}};",
                        "#ifdef __CUDACC__",
                        f"static __device__ __constant__ array<{kind},{size}> {name}Device{{{init}\n}};",
                        "#endif",
                        f"GAME_HD inline {kind} {name}(unsigned i) {{",
                        "#ifdef __CUDA_ARCH__", f"  return {name}Device[i];", "#else",
                        f"  return {name}Host[i];", "#endif", "}"]
    return "#pragma once\n#include \"game_types.cuh\"\nnamespace irisu::cuda_game {\n" + "\n".join(definitions) + "\n}\n"


def game() -> str:
    text = source("clone/core/simulator.cpp")
    methods = extract(text, "Observation Simulator::reset", "Snapshot Simulator::clone_state")
    methods = convert(methods)
    methods = methods.replace("Simulator::", "")
    # GPU callers observe explicitly; avoid a 64KiB discarded return buffer on
    # every reset while preserving all authoritative reset side effects.
    reset_end=methods.index("StepResult step(")
    reset=methods[:reset_end].replace("Observation reset(","void reset(").replace("return observation();","return;")
    methods=reset+methods[reset_end:]
    methods = re.sub(r"^(Observation|StepResult|StepDiagnostics|std::uint32_t|BodyId|void|Body\*|GroupState\*) ",
                     r"GAME_HD \1 ", methods, flags=re.M)
    # Source exceptions are replaced with lane failure + an early method return.
    # The two caught score-overflow paths retain their original saturation.
    numeric = extract(methods, "    if (numeric_guard_required_) {", "    // A step is capped")
    methods = methods.replace(numeric, "    if (numeric_guard_required_) { failed_=true; return result; }\n")
    methods = re.sub(r"        try \{\n          points = (normal_score_delta\(.*?);\n        } catch \(const std::overflow_error&\) \{\n          points = std::numeric_limits<std::int64_t>::max\(\);\n        }",
                     r"        points = \1;", methods, flags=re.S)
    methods = re.sub(r"void sequence_events\(Events& events\) \{.*?\n}\n", r'''void sequence_events(Events& events) {
  if (events.size() > UINT64_MAX-next_event_sequence_) { failed_=true; return; }
  next_event_sequence_ += events.size();
}
''', methods, flags=re.S)
    methods = methods.replace("StepResult step(const Action& action)", """StepResult step(const Action& action
#ifdef IRISU_CUDA_GAME_EVENT_CAPTURE
  , Event* capture=nullptr, std::uint32_t capture_capacity=0
#endif
  )""")
    methods = methods.replace("  StepResult result;", """  StepResult result;
  result.events.sequence_base=next_event_sequence_;
#ifdef IRISU_CUDA_GAME_EVENT_CAPTURE
  result.events.capture=capture;result.events.capture_capacity=capture_capacity;
#endif
  if(failed()) return result;""")
    methods = methods.replace("  rng_.seed(", "  failed_=false;\n  rng_.seed(", 1)
    # Per-method exception replacements preserve failure as a non-selectable
    # lane instead of throwing across a GPU kernel.
    spans = list(re.finditer(r"^GAME_HD ([^\n]+?) ([a-z_]+)\(", methods, re.M))
    for index in reversed(range(len(spans))):
        match = spans[index]
        stop = spans[index+1].start() if index+1 < len(spans) else len(methods)
        segment = methods[match.start():stop]
        return_type = match.group(1)
        fallback = "return;" if return_type == "void" else "return {};"
        segment = re.sub(r"throw std::(?:invalid_argument|logic_error|overflow_error)\([^;]*\);",
                         "{ failed_=true; " + fallback + " }", segment)
        methods = methods[:match.start()] + segment + methods[stop:]
    # Participant methods are local-class functions, not standalone HD methods.
    methods = re.sub(r"    (int|bool|std::uint8_t) (contact_class|physics_owned|grouped|rotten|guard)\(",
                     r"    GAME_HD \1 \2(", methods)
    methods = methods.replace("  sequence_events(result.events);", "  sequence_events(result.events);\n  failed_ |= result.events.overflow;")
    methods = methods.replace("  active_contact_pairs_ = move(current);", "  active_contact_pairs_.assign_live(current);")
    # Scratch is owned by each world, so CUDA does not reserve tens of KiB of
    # thread stack for fixed-capacity vectors. Each call clears before use;
    # scratch never participates in game state, RNG or event ordering.
    methods = methods.replace("  vector<ScriptedOrigin> scripted_origins;", "  auto& scripted_origins=scripted_origins_scratch_;scripted_origins.clear();")
    methods = methods.replace("  vector<std::uint64_t> current;", "  auto& current=current_contacts_scratch_;current.clear();")
    methods = methods.replace("  vector<Body*> actors;", "  auto& actors=actor_refs_scratch_;actors.clear();")
    methods = methods.replace("  const auto contacts = physics_.step(bodies_);", "  const auto& contacts = physics_.step(bodies_);\n  failed_ |= contacts.overflow;")
    methods = re.sub(r"(  validate_public_spawn_geometry\([\s\S]*?\);)",r"\1\n  if(failed_) return 0;",methods)
    methods = methods.replace("  const auto parameter_level = min<std::uint32_t>(level_, 99U);\n  const auto parameters", "  failed_ |= actors.overflow;\n  const auto parameter_level = min<std::uint32_t>(level_, 99U);\n  const auto parameters")
    for name in ("step", "tick_once", "process_contacts", "process_actor_updates", "reset"):
        methods = re.sub(r"GAME_HD ([^\n]+? " + name + r"\()",r"GAME_HD GAME_NOINLINE \1",methods)
    if "throw " in methods or "try {" in methods or "catch (" in methods:
        raise RuntimeError("untranslated exception path")
    members = extract(source("clone/include/irisu/simulator.hpp"), "  MechanicsConfig config_;", "};\n\n}  // namespace irisu")
    members = convert(members).replace("PhysicsWorld physics_;", "Physics physics_;")
    helpers = convert(extract(text, "std::uint64_t pair_key", "void serialize_body"))
    helpers = re.sub(r"^(std::uint64_t|bool|std::int64_t) ", r"GAME_HD \1 ", helpers, flags=re.M)
    helpers = helpers.replace("GAME_HD bool live", "GAME_HD static bool live")
    for function, fallback in (("checked_u64_add", "0"), ("checked_nonnegative_product", "0"),
                               ("checked_subtract", "0")):
        start = helpers.index(function)
        end = helpers.find("\nGAME_HD ", start)
        if end < 0: end = len(helpers)
        part = helpers[start:end]
        part = part.replace("throw std::overflow_error(detail);", f"failed_=true; return {fallback};")
        helpers = helpers[:start] + part + helpers[end:]
    numeric_guard = convert(extract(text, "bool body_requires_numeric_guard", "void validate_public_spawn_geometry"))
    numeric_guard = numeric_guard.replace("long double", "double").replace("kNumericGuardThreshold", "1.0e5")
    numeric_guard = "GAME_HD " + numeric_guard
    parameters = extract(source("clone/include/irisu/normal_rules.hpp"), "struct NormalLevelParameters", "// The executable bypasses")
    normal = convert(extract(source("clone/core/normal_rules.cpp"), "NormalLevelParameters normal_level_parameters", "double normal_spawn_angle"))
    normal = normal.replace("kScoreScaleBits.size()", "99U")
    normal = normal.replace("kScriptedDescentBits[level - 1U]", "kScriptedDescentBits(level - 1U)")
    normal = normal.replace("kScoreScaleBits[level - 1U]", "kScoreScaleBits(level - 1U)")
    normal = normal.replace('throw std::invalid_argument("normal level parameters exist only for 1..99");', "failed_=true; return {};")
    normal = normal.replace('throw std::overflow_error("normal score delta overflow");', "return INT64_MAX;")
    normal = re.sub(r"^(NormalLevelParameters|std::int64_t) ", r"GAME_HD \1 ", normal, flags=re.M)
    normal = normal.replace("normal_level_parameters(std::uint32_t level) {", "normal_level_parameters(std::uint32_t level) const {")
    config_hash = extract(text, "std::uint64_t Simulator::calculate_config_hash", "}  // namespace irisu")
    config_hash = convert(config_hash).replace("Simulator::", "")
    config_hash = config_hash.replace("vector<std::byte> bytes;", "std::uint64_t hash=14695981039346656037ULL;")
    config_hash = re.sub(r"    using T = decltype\(value\);[\s\S]*?    else append_integer\(bytes, value\);", "    hash=fnv_scalar(hash,value);", config_hash)
    config_hash = "GAME_HD " + config_hash.replace("return fnv1a(bytes);", "return hash;")
    prelude = r''' 
  GAME_HD explicit Simulator(MechanicsConfig config={}) : config_(config), physics_(config) { config_hash_=calculate_config_hash(); }
  GAME_HD bool failed() const {
    bool result=failed_ || bodies_.overflow || groups_.overflow || active_contact_pairs_.overflow;
    if constexpr(requires(const Physics& p){p.failed();}) result |= physics_.failed();
    return result;
  }
  GAME_HD std::uint64_t config_hash() const { return config_hash_; }
  GAME_HD double normal_spawn_angle(std::uint32_t ticket) {
    if(ticket>1000) { failed_=true; return 0; }
    return static_cast<double>(bit_cast<float>(kSpawnAngleBits(ticket)));
  }
  GAME_HD bool uses_nominal_scripted_fall(double value) {
    return bit_cast<std::uint64_t>(value)==0x3fc999999999999aULL;
  }
  GAME_HD double stored_float(double value) { return static_cast<double>(static_cast<float>(value)); }
  GAME_HD double stored_float_add(double a,double b) { return static_cast<double>(static_cast<float>(static_cast<float>(a)+static_cast<float>(b))); }
  GAME_HD void validate_public_spawn_geometry(const MechanicsConfig&,Shape,double size,Vec2 position,double density,double friction,double restitution,double velocity) {
    if(!isfinite(size)||size<=0||!isfinite(position.x)||!isfinite(position.y)||!isfinite(density)||density<0||!isfinite(friction)||friction<0||!isfinite(restitution)||!isfinite(velocity)) failed_=true;
  }
'''
    return ("#pragma once\n#include \"game_tables.cuh\"\nnamespace irisu::cuda_game {\n" + parameters +
            "template<class Physics> class Simulator {\npublic:\n" + prelude +
            "static constexpr BodyId kMaximumBodyId=0x7fffffffU;\nstatic constexpr std::size_t kMaximumSnapshotGroups=1000000;\n" +
            helpers + numeric_guard + normal + methods + config_hash + members + "\n  vector<ScriptedOrigin> scripted_origins_scratch_;\n  vector<std::uint64_t> current_contacts_scratch_;\n  vector<Body*> actor_refs_scratch_;\n  mutable bool failed_{};\n};\n}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    angles = spawn_angle_bits()
    outputs = {"game_types.cuh": types(), "game_tables.cuh": tables(angles), "game.cuh": game()}
    for name, text in outputs.items():
        (OUT / name).write_text("// Generated by tools/cuda-exact/game_codegen.py; do not edit.\n" + text)
    inputs = ["clone/core/simulator.cpp", "clone/core/normal_rules.cpp", "clone/core/dx_random.cpp",
              "clone/include/irisu/types.hpp", "clone/include/irisu/math.hpp", "clone/include/irisu/config.hpp",
              "clone/include/irisu/dx_random.hpp", "clone/include/irisu/normal_rules.hpp", "clone/include/irisu/simulator.hpp",
              "tools/exact-physics-prototype/physics_wrapper_forward.cpp", "tools/cuda-exact/game_codegen.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    (OUT / "game_source.json").write_text(json.dumps({
        "schema": "irisu-cuda-game-derived-v1", "inputs": hashes,
        "outputs": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in outputs},
        "spawn_angles": "1001 float32 stores generated by original normal_spawn_angle with x87 PC53",
        "scope": "normal-mode mechanics; exactness requires integration/differential validation",
        "unsupported": ["durable serialized snapshots", "numeric-guard/exotic body states", "rotation tickets above 1000"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
