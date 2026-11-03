# Recovering the CPU exact oracle

The original development checkout, local artifacts and MSVC9 toolchain were
absent on 2026-11-03. The project's published browser-runtime release still
contains the production MSVC physics host byte for byte. It supplies an
independent runnable CPU oracle for GPU development without rebuilding the
physics with a different compiler.

Recover the pinned files into a new local directory:

```sh
python3 tools/recover-exact-cpu-runtime.py --output .tmp/exact-cpu-oracle
```

The helper downloads the 5.5 MB
[project release](https://github.com/Gabriel-Kahen/irisu-ml/releases/tag/web-exact-runtime-virtio-v1-20260824),
validates archive SHA-256, copies only named regular files, validates both
executable hashes, and atomically exposes the completed directory. An existing
destination is refused. For an offline download, supply `--archive PATH`.
The Box2D license and published provenance/source-offer files are retained.

| Artifact | SHA-256 |
| --- | --- |
| Release archive | `0340660c5e5ccd1bbc47bd75eb60ec96349a0d046fa9cf1e5923ff78ee77838f` |
| MSVC physics host | `ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5` |
| Browser-release CPU worker | `8ef81521e81a5b2a764c305ac48dad997b28476bcd2fccbd1c9aed9603322854` |
| Historical training worker | `4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261` |

The browser worker was relinked for its guest ABI; its hash differs from the
historical training worker. The historical training identity config remains
unchanged. Diagnostic equivalence evidence does not silently authorize a
different worker for target-policy collection or promotion.

Build the current application against the recovered host with a GNU compiler
and working 32-bit C++ runtime:

```sh
cmake -S . -B build-exact-recovered -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_EXACT_BOX2D_LIBRARY="$PWD/.tmp/exact-cpu-oracle/guest/libirisu_box2d_msvc_exact_multiworld.so"
cmake --build build-exact-recovered -j 6
# One exact checkpoint test also checks that the portable backend refuses it.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
ctest --test-dir build-exact-recovered --output-on-failure -j 2
ctest --test-dir build --output-on-failure -j 2
```

Both recovered/rebuilt workers independently reproduce the published replay's
canonical **4,393,101** score, level 100 and chain 70 at tick 97,219. The
post-update live score is 4,393,699; it must not replace canonical terminal
metadata. Validate every state, event and diagnostic response byte between
workers:

```sh
python3 tools/cuda-exact/compare_cpu_workers.py \
  --left .tmp/exact-cpu-oracle/guest/irisu-exact-worker \
  --right build-exact-recovered/irisu-exact-worker \
  --output artifacts/cpu-worker-byte-differential.json
```

The 2026-11-03 run matched all 188,356,329 response bytes across 97,219 ticks,
with zero mismatches. The length-prefixed response transcript SHA-256 is
`1c0ec5f7b679fe23847ffdb38b8564ee9984d9311017d90e1a22a69e02fa53b7`.
All 12 available exact CTests and eight portable CTests passed. The four old
external replay/original-process oracle files are absent, so their CTests are
explicitly skipped at configuration; these results do not claim their rerun.
The checked-in summary is
[`exact-runtime-recovery-2026-11-03.json`](../benchmarks/results/exact-runtime-recovery-2026-11-03.json).

## Physics-level GPU differential input

`tools/cuda-exact/physics_oracle.c` exercises the real public MSVC host API
under x87 control word `0x027f`. It sets up a floor and a falling circle, box
and triangle, including contacts and nonzero rotations. Its default raw output
contains 15 little-endian uint32 words per tick: body-major x/y/rotation/vx/vy.

```sh
mkdir -p .tmp
gcc -m32 -O2 -fno-fast-math -Wall -Wextra -Werror \
  tools/cuda-exact/physics_oracle.c -ldl -o .tmp/physics-oracle
.tmp/physics-oracle \
  "$PWD/.tmp/exact-cpu-oracle/guest/libirisu_box2d_msvc_exact_multiworld.so" 600 \
  > .tmp/physics-600.raw
```

That baseline is 36,000 bytes, SHA-256
`4fcd94f803509317a3f81b9611c0c4c362374354becf04b5382252c6f2d603b9`.
`--jsonl` additionally records ordered contact user IDs (floor 1, circle 2,
box 3, triangle 4). `--mutations` sets circle velocity before tick 200 and
destroys/recreates the box before tick 300. These small collision fixtures are
component tests, not a replacement for full game-state/replay validation.
