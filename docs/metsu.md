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
stay at `/`, with `/normal/` as a direct alias. The header links switch between
normal and Metsu. Metsu has its own cache namespace and content-derived runtime
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
setter, including its power/sine routines and float stores. Metsu rules
revision 2 selects the stores measured at the actual original process's
`0x137f` control word; the earlier PC53 table is retained as calibration
history. Normal's calibration and tables remain unchanged. Separate
original-machine-code component probes cover EX contacts and countdowns.
See [the audit](../reference/metsu-rules-analysis.md) for addresses, reproduction
commands, stubs, hashes, and scope.

Actual original-game qualification now covers three complete recorded inputs:

| Input origin | Inputs | Reset/update boundaries checked | Canonical score / level / chain |
|---|---:|---:|---|
| Actual browser Save download | 3,942 | 3,943 | 40 / 1 / 2 |
| Actual original-game recorder output | 2,994 | 2,995 | 0 / 1 / 0 |
| Deterministic active-input coverage policy | 4,652 | 4,653 | 426 / 4 / 3 |

Each original/native comparison checks 8 scalar/pool fields, 28 actor fields
including raw float32 positions and velocities, live actor-slot identity,
normalized group sharing including pending EX links, and first-finish metadata.
All measured states match. Natural termination and saved headers are checked
separately from input exhaustion. The original-recorder and controlled files
were also loaded through the real browser file chooser and visibly completed
without error.
The [interoperability report](../benchmarks/results/metsu-replay-interoperability-2026-11-03.json)
binds the browser actions, original-process receipts, input hashes, and
per-run comparisons.

The controlled replay reaches level 4, three level transitions, held weak/strong
shots, and EX chains. Reproduce its exact bytes with
`build-metsu-exact/irisu-metsu-generate-controlled 1 OUTPUT.rpy`. Stable inputs
are under [`tests/fixtures/metsu`](../tests/fixtures/metsu/README.md); the
[comparison guide](../tools/metsu/REPLAY_INTERCHANGE.md) gives the strict commands.

This is finite observational qualification for those inputs and fields, not
proof that every possible run or rare contact state matches. The full-run
special-orb scheduler and levels beyond 4 are not covered by these completed
replay samples. Component probes retain their smaller scopes. Original-game
trajectory qualification applies to the exact backend; portable physics retains
normal's long-horizon numerical limitations.

The current
[native/v86 transport report](../benchmarks/results/metsu-replay-browser-native-v86-2026-11-03.json)
compares three complete seeded episodes plus terminal tails: 5,834 responses
and 223,000,797 bytes, with no mismatches. It establishes that the packaged
browser engine executes those native workloads identically; it does not replace
the separate actual-original evidence above. Historical component and initial
implementation receipts are retained, including the rejected premature browser
export that matched a final score while the original game was still active.

## Protect normal

Run [the pinned normal regression guard](metsu-normal-regression.md) alongside
both modes' tests. It compares the published 97,219-tick normal replay against
the released worker, rather than comparing two newly modified workers.
The [development separation contract](metsu-development.md) records ownership
of the mode boundaries.

The [replay-qualification checkpoint's normal gate](../benchmarks/results/metsu-final-normal-regression-2026-11-03.json)
passed all 97,219 ticks and 188,356,329 response bytes with zero mismatches;
its canonical result remains 4,393,101 points, level 100, chain 70. The final
normal worker is byte-identical to the pristine pre-Metsu baseline build. The
[protected-source check](../benchmarks/results/metsu-protected-normal-source-2026-11-03.json)
confirmed that normal gameplay/physics sources, headers, browser assets and
normal web build/runtime scripts were unchanged from `13023da` at that checkpoint.
Subsequent shared-site routing adds header links and styles to both clients,
a normal alias and combined deployment packaging. It does not change normal
gameplay, replay handling, worker bytes, or the pinned runtime.
