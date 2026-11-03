# Experimental exact-host CUDA backend

This optional backend translates the pinned i386 MSVC physics image rather than
replacing it with a float32 Box2D implementation. Its admission boundary is
explicit: unsupported instructions, numeric domains, capacities or guest-memory
accesses invalidate the lane. A failed candidate must never participate in score
selection. Search wire format v2 carries the **first recorded terminal** score
explicitly for ranking and replay headers. The older Frame v1 regression wire
retains its legacy `canonical_score` field encoding of the latest diagnostic;
that field alone is not the ranking contract. The production CPU runtime and exact-training pins remain separate.

The development target is a single Tesla V100 PCIe 32 GB (`sm_70`). The observed
host has a Ryzen 7 3700X, NVIDIA driver 580.178.04, CUDA 12.9.86 and GCC 14 for
CUDA host compilation. These are recorded observations, not minimum requirements
for portable builds. CUDA 13 toolkit is not substituted for the installed 12.9
compiler merely because `nvidia-smi` reports driver CUDA capability 13.0.

## Architecture

- `tools/cuda-exact/translate.py` checks the original ELF SHA-256, resolves its
  exports/imports and lowers decoded instruction regions. Guest
  addresses are 32-bit offsets into a private lane arena.
- `clone/cuda/guest.hpp` implements integer registers/flags, bounded x87 stack,
  PC53 numeric operations and guest memory. Translated control flow preserves
  per-world update order. Worlds execute independently across CUDA threads.
- `x87_numeric.cuh` preserves admitted PC53 arithmetic and final-binary32 trig
  boundaries. The bounded trig result gate is described in [arithmetic.md](../tools/cuda-exact/arithmetic.md).
  This is not unrestricted x87 exception or extended-exponent emulation.
- `game_codegen.py` derives normal-mode rules, MT19937 RNG, spawns, scores and
  lifecycle logic from the owning C++ sources. `game_source.json` hashes inputs
  and outputs so changes require regeneration and new differential evidence.
- `physics_game.cuh` ports the exact CPU wrapper's ordering and conversion
  boundaries. The wrapper's C++/x87 promotion behavior matters in addition to
  the MSVC physics image's own operations.
- `World` stores inline game containers plus the guest state. Its arena contains
  the image, physics heap, allocator and stack. Copies must rebind both native
  pointers with `World::bind`; guest offsets remain unchanged.
- The search path uses device arena copies, independent candidate futures,
  deterministic integer objectives and survivor compaction. Depth and selection
  dependencies remain sequential. No Python callback is needed per tick.

`tools/cuda-exact/runtime_identity.py` records source, generated-image and binary
hashes in a distinct experimental identity. Verification fails if any captured
input changes. Its receipt does not grant production exact-training admission;
parity evidence must be attached separately for the actual workload tested.

## Build separation

The CPU exact oracle requires a 32-bit GNU build; CUDA requires a separate
64-bit build tree. Portable builds do not require CUDA, OpenSSL or translation packages.
CUDA image admission uses the host OpenSSL Crypto library; the optional build
checks for its development headers and library.
First recover the original runtime using [exact-runtime-recovery.md](exact-runtime-recovery.md).

```sh
# Recover only when the preserved runtime is absent (existing outputs are refused).
[ -d .local/exact-runtime-20260824/exact-runtime/guest ] || \
  python3 tools/recover-exact-cpu-runtime.py --output .local/exact-runtime-20260824/exact-runtime
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -r tools/cuda-exact/requirements.txt
cmake -S . -B build-project-exact -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_EXACT_BOX2D_LIBRARY="$PWD/.local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so"
cmake --build build-project-exact -j 3
cmake -S . -B build-project-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DIRISU_BUILD_CUDA_EXPERIMENTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv-gpu/bin/python" \
  -DIRISU_CUDA_EXACT_HOST="$PWD/.local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so"
```

The translation is split into separately compiled device regions so that CUDA
assembly optimization remains bounded in memory. The shared region library is
reused by the fixture, replay and search targets. `IRISU_CUDA_HOST_OPTIMIZATION`
controls `nvcc -O` host code; `IRISU_CUDA_PTXAS_OPTIMIZATION` controls assembly
optimization. Device frontend passes are controlled separately by
`IRISU_CUDA_GAME_FAST_COMPILE` (default `mid`).

`IRISU_CUDA_FAST_COMPILE=ON` selects the lowest consumer compile optimization
for iteration. Performance receipts must include
actual compiler flags and binary hashes; a fast-compile fixture is not evidence
for the performance of an optimized build. Strict rounding/denormal/FMA flags
are required regardless of optimization level. Do not start several large
translated-program compilations simultaneously on a 32 GB development host.

The inherited development environment hides CUDA with
`CUDA_VISIBLE_DEVICES=-1`. Expose the device only to each intended test/search
child, for example `env -u CUDA_VISIBLE_DEVICES COMMAND`; no global environment,
power limit or driver changes are needed.

## Evidence and interpretation

The original recovered work and checkpoint are preserved in
`GPU-HANDOFF-20261103.txt`. The project migration SHA-256 checked 2,845 regular
files and left `/home/gabe/Documents/irisu` untouched. Local migration/build
receipts live under `artifacts/gpu-exact-20261103/`; compact reviewable evidence
is checked in under `benchmarks/results/`.

Fresh project CPU tests pass 12 exact and eight portable CTests. The native
published-record oracle repeats all 97,220 regression digest rows byte for byte:
97,219 steps, 308,512 events, level 100, chain 70, canonical score **4,393,101**.
Its post-update live score is 4,393,699 and is not the search objective.

Archived commands from the stopped search chat identify the actual record
runner as `benchmarks/rl_combo_diverse_search.py` in the former `a40c` worktree:
event algorithm, game seed 1298144938, search seed 661003, root tick 96000,
horizon 3072, segment 32, beam width 24 and 48 candidates. See the compact
[historical provenance receipt](../benchmarks/results/record-search-historical-provenance-2026-11-03.json).
The former runner source and campaign manifests are absent. The new bounded
CUDA workload defines its own sparse proposals and must not be presented as a
reproduction of that historical event proposal algorithm.

Report physics microbenchmarks, synthetic arena scheduling and complete search
separately. For complete search, compare identical candidate futures and stable
selection, warmed repeats, physical ticks/s, completed branches/s, fixed-budget
wall time, VRAM and sampled device utilization. A shorter horizon or a faster
memory-copy kernel does not establish a full-game search speedup. Historical
CPU decisions/s in `docs/fidelity.md` describe another workload and are not a
denominator for these measurements.

## Run the implementation

Build the optional executables and native reference tools:

```sh
cmake --build build-project-cuda --target irisu-cuda-physics-probe \
  irisu-cuda-game-probe irisu-cuda-game-segments irisu-cuda-search \
  irisu-cuda-search-session -j 3
cmake --build build-project-exact --target irisu-game-oracle irisu-search-oracle -j 3
```

A continuous short GPU trajectory checks body, observation, event and diagnostic
regression digests against the independent native oracle:

```sh
python3 tools/cuda-exact/game_validate.py \
  --oracle build-project-exact/irisu-game-oracle \
  --probe build-project-cuda/clone/cuda/irisu-cuda-game-probe \
  --image build-project-cuda/clone/cuda/generated/guest_image.bin \
  --steps 128 --gpu --arena-mib 1 --output artifacts/cuda-game-smoke
```

To cover the complete record in parallel checkpoint segments, prepare its
independent native digest, then run the segmented validator. Host preparation
and GPU execution times are separate; this is validation, not a search benchmark.

```sh
mkdir -p artifacts
build-project-exact/irisu-game-oracle irisu-high-score-4393101.rpy \
  > artifacts/record-native.digest
env -u CUDA_VISIBLE_DEVICES build-project-cuda/clone/cuda/irisu-cuda-game-segments \
  build-project-cuda/clone/cuda/generated/guest_image.bin \
  irisu-high-score-4393101.rpy artifacts/record-native.digest \
  --segment-ticks 64 --threads 4 --arena-mib 1
```

The fixed-budget benchmark performs a warmup and identical native/GPU candidate
comparisons on every timed repetition. `--final-only` removes per-tick regression
hashing from **both** backends, retaining checkpoint/final comparisons; use the
default for per-step validation. A host-translated prefix is prepared once per
search process and included in its wall time. No host work occurs per future tick.

```sh
python3 tools/cuda-exact/search_validate.py \
  --oracle build-project-exact/irisu-search-oracle \
  --gpu build-project-cuda/clone/cuda/irisu-cuda-search \
  --image build-project-cuda/clone/cuda/generated/guest_image.bin \
  --prefix 100 --horizon 8 --candidates 512 --arena-mib 1 --keep 8 \
  --workers 1 8 16 --repeats 3 --host-prefix --final-only \
  --output benchmarks/results/cuda-search-local.json
```

The resident beam queues dependent depths with no intermediate state download,
retains selected device checkpoints, and records stable parent/candidate lineage.
The following small differential independently checks every leaf against native
forks and verifies the exported winning trace from reset. Its flattened native
reference deliberately repeats earlier history for correctness and must not be
used as an optimized CPU multi-depth speed benchmark.

```sh
python3 tools/cuda-exact/search_session_validate.py \
  --oracle build-project-exact/irisu-search-oracle \
  --game-oracle build-project-exact/irisu-game-oracle \
  --gpu build-project-cuda/clone/cuda/irisu-cuda-search-session \
  --image build-project-cuda/clone/cuda/generated/guest_image.bin \
  --replay irisu-high-score-4393101.rpy --prefix 100 --horizon 2 \
  --fanout 3 --beam 2 --depths 3 --arena-mib 1 --host-prefix \
  --output artifacts/cuda-resident-session.json
```

The session runner writes a candidate replay and an experimental lineage manifest
with `native_replay_verified=false`. The independent validator adds the replay
hash, verification scope, and successful native reset-replay check. A bounded
nonterminal trace is labeled as such; it is not a completed score record.
Production training admission remains false even after this diagnostic check.

Create a runtime receipt once sources, build flags and binaries are stable:

```sh
python3 tools/cuda-exact/runtime_identity.py create \
  --binary build-project-cuda/clone/cuda/irisu-cuda-search \
  --generated build-project-cuda/clone/cuda/generated \
  --host .local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so \
  --build-command 'cmake --build build-project-cuda --target irisu-cuda-search -j 3' \
  --output artifacts/cuda-search-identity.json
python3 tools/cuda-exact/runtime_identity.py verify artifacts/cuda-search-identity.json
```

The receipt includes CMake cache, compiler flags and link commands when available.
The runtime also checks the actual image SHA-256 before allocating GPU state;
modified, truncated or wrong-sized images are rejected.
