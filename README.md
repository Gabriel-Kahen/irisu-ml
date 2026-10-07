# IriSu ML

A headless simulator, ML environment, and
[playable web client](https://irisu.online/) for **IriSu Syndrome! v2.03
normal mode**. The project is complete; its final verified run scored
**4,393,101** points.

[Play in your browser](https://irisu.online/) ·
[Download the final replay](irisu-high-score-4393101.rpy) ·
[Read the fidelity evidence](docs/fidelity.md)

## Final run

| Result | Value |
| --- | ---: |
| Score | **4,393,101** |
| Level | 100 |
| Highest chain | 70 |
| Seed | `1298144938` |

The [replay](irisu-high-score-4393101.rpy) is an original-format `.rpy` file.
Open it with **play replay** in the web client or with the original v2.03 game.
Its SHA-256 is
`81fe723b6b825e62afb471d7517355fef2b8bf0fb1dbe636069c97c70ce2ca79`.
An independent exact-runtime replay check accepted the score, level, and chain
recorded in its header.

The run demonstrates a verified score on a selected seed. It does not establish
that a learned policy transfers to the original game or performs at this level
across arbitrary seeds.

## Headless simulator

[`clone/`](clone) is a deterministic C++20 simulator for the game's normal
puzzle mode, with a command-line interface and C API. It models input timing,
seeded spawning, scoring, gauge, chains, and level progression without shipping
the original game's assets. The physics adapter uses the zlib-licensed Box2D
1.4.3 engine.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
printf 'wait 20\nweak 250 350\nwait 20\nstate\nquit\n' | build/irisu-headless --seed 42
```

The portable build is the default. An optional 32-bit exact-MSVC physics host
supports trajectory-exact replay work; it requires separately generated local
dependencies. Four eligible instrumented v2.03 playbacks matched the exact
backend at every observed score and rotation event, with the longest trace
matching through 47,019 updates. See the [fidelity report](docs/fidelity.md),
[physics provenance](docs/physics-source.md), and
[exact host setup](reference/native-box2d/multiworld/README.md) for the methods
and limits.

## Machine-learning environment

[`python/irisu_env/`](python/irisu_env) wraps the native simulator in a
dependency-free, Gymnasium-shaped Python API. It exposes observations, rewards,
typed actions, state snapshots, and independent vector environments. The exact
backend runs each episode in an isolated worker process.
[`python/irisu_rl/`](python/irisu_rl) contains the research encoders, training,
evaluation, and search tools built on that API.

The Python package does **not** bundle the native library. Build it first, then
install the package and point the interface at that library:

```bash
python3 -m pip install .
export IRISU_CLONE_LIBRARY="$PWD/build/libirisu_clone.so"
python3 - <<'PY'
from irisu_env import Action, IrisuEnv

with IrisuEnv() as env:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(
        Action.strong(300, 360)
    )
    print(observation["score"], terminated)
PY
```

Gymnasium and training packages are optional extras; the core interface needs
neither. The [RL documentation](RL.md),
[exact training notes](docs/exact-training.md), and
[benchmark guide](benchmarks/README.md) record the research contracts and
measured limits. The published replay is a search result, not evidence of
general policy transfer.

## Playable web client

[**Play IriSu online**](https://irisu.online/) in the static browser client at
[`apps/web/`](apps/web). It runs the exact i386 worker locally under v86 inside
a Web Worker. GitHub Pages serves the files; gameplay and physics run in the
browser. The client supports keyboard, mouse, and touch controls, replay
loading and saving, seeking, and playback from 1× to 8× speed.

Use **play replay** to load the [final run](irisu-high-score-4393101.rpy). The
[web client guide](apps/web/README.md) covers controls and local builds;
[runtime provenance](apps/web/EXACT_RUNTIME.md) explains the pinned exact
runtime.

## Validate and explore

Run the native, Python, and web checks from the repository root:

```bash
uv run --all-extras python tools/validate.py
```

The [validation guide](docs/validation.md) describes prerequisites and
options. At finalization on October 7, 2026, the existing native build and all
**207** validation jobs passed with `--jobs 8 --no-build`.

| Path | Contents |
| --- | --- |
| [`clone/`](clone) and [`third_party/`](third_party) | Simulator, public C API, and licensed physics source |
| [`python/`](python) | Environment and learning/search code |
| [`apps/web/`](apps/web) | Static browser client and exact runtime build tools |
| [`tests/`](tests) | Native, Python, web, and fidelity tests |
| [`configs/`](configs) | Measured mechanics and research configurations |
| [`docs/`](docs), [`reference/`](reference/README.md), [`benchmarks/`](benchmarks/README.md) | Methods, evidence, and historical measurements |

Generated builds, search outputs, original-game binaries, recordings, and other
reference-only material stay outside version control. The final replay is
published here so the result can be inspected directly.
