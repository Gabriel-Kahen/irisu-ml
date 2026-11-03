# Metsu mode

Metsu is an optional, separate v2.03 gameplay engine. Normal remains the default
build and keeps its existing rules, snapshots, worker identity, and browser
runtime. Metsu uses its own `irisu::metsu::Simulator`, targets, mode-1 replays,
and static client. The shared components are the physics engine, RNG, numeric
helpers, and IPC transport.

## Headless simulator

```sh
cmake -S . -B build-metsu -DCMAKE_BUILD_TYPE=Release -DIRISU_BUILD_METSU=ON
cmake --build build-metsu -j 6
printf 'wait 20\nstrong 300 360\nwait 200\nquit\n' | build-metsu/irisu-metsu-headless --seed 41
ctest --test-dir build-metsu --output-on-failure -j 2
```

For the exact backend, first recover the pinned CPU runtime following
[exact-runtime-recovery.md](exact-runtime-recovery.md), then configure:

```sh
cmake -S . -B build-metsu-exact -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_BUILD_METSU=ON \
  -DIRISU_EXACT_BOX2D_LIBRARY="$PWD/.tmp/exact-cpu-oracle/guest/libirisu_box2d_msvc_exact_multiworld.so"
cmake --build build-metsu-exact -j 6
IRISU_CLONE_LIBRARY="$PWD/build-metsu/libirisu_clone.so" \
  ctest --test-dir build-metsu-exact --output-on-failure -j 2
```

Use the distinct Python entry point with an explicit worker path:

```python
from irisu_metsu import MetsuSimulator

with MetsuSimulator("build-metsu-exact/irisu-metsu-exact-worker") as game:
    game.reset(41)
    game.step(2, 300, 360, 1)  # strong shot
    game.step(0, 0, 0, 200)  # release and wait
    print(game.observation()["score"])
```

The worker owns one episode per process. Python resets replace the process;
durable snapshots replay the seed and inputs, and Linux fork checkpoints keep
the Metsu class and worker identity. Normal and Metsu clients reject each
other's workers and snapshots. Portable C++ snapshots also use a distinct
magic/schema and preserve EX countdowns explicitly.

## Browser client

Follow [the separate web build guide](../apps/web/METSU.md) to prepare the
Metsu runtime and stage it under `/metsu/`. The normal static build and runtime
stay at `/`. Metsu has its own cache namespace and content-derived runtime
identity, nine colors, EX-circle markers, and mode-1 replay loading/saving.
Runtime preparation verifies native/v86 response parity before marking the
artifact ready. Local builds do not publish or replace the deployed normal site.

## Gameplay and evidence

The implementation includes the larger field, ten ordinary reset blocks, two
spawn cadences, ordinary descent and EX ascent, eight-step propagation threshold
and 120-update activation countdown, mode-specific contacts, gauge recovery,
rot, score, six-clear level progression, bonus clears, and level-100 finish.
It preserves the original EX sampler's use of the ordinary 70-ticket bound.

All 99 level parameter rows were measured by executing the pinned original
setter, including its power/sine routines and float stores. The extraction
first reproduces all 99 recorded normal score and descent rows. Separate
original-machine-code component probes cover EX contacts and countdowns.
See [the audit](../reference/metsu-rules-analysis.md) for addresses, reproduction
commands, stubs, hashes, and scope.

These are component and implementation checks. A complete original-game
mode-1 replay comparison has not been demonstrated. Browser/native parity
establishes that both execute the same Metsu implementation; it does not by
itself establish unrestricted fidelity to the original game. Portable physics
retains the same long-horizon numerical limitations as normal's portable build.

The [implementation validation receipt](../benchmarks/results/metsu-implementation-validation-2026-11-03.json)
records passing native/client tests, sanitizer scope, original component checks,
and the unchanged 97,219-tick normal replay. The
[browser/native report](../benchmarks/results/metsu-web-native-v86-2026-11-03.json)
compares three complete seeded episodes plus terminal tails: 5,663 responses
and 214,754,473 bytes, with no mismatches.

## Protect normal

Run [the pinned normal regression guard](metsu-normal-regression.md) alongside
both modes' tests. It compares the published 97,219-tick normal replay against
the released worker, rather than comparing two newly modified workers.
The [development separation contract](metsu-development.md) records ownership
of the mode boundaries.
