# Playable web app

This asset-free browser client renders the real headless simulator at 50 Hz.
Its open-bottom U-shaped well follows the measured v2.03 mode-0 geometry.

## Modes and site routes

The header's Normal / Metsu links switch between independent game clients:

- `/` always opens normal mode.
- `/normal` opens normal mode directly (static hosts add the trailing slash).
- `/metsu` opens Metsu directly (static hosts add the trailing slash).

The normal alias shares the root's scripts, exact runtime, soundtrack and cache;
Metsu keeps its own scripts, runtime and cache under `/metsu/`. Mode selection
does not change the default landing page or either game's HUD.

Build the complete site from the prepared runtimes:

```sh
IRISU_EXACT_RUNTIME_DIR=/path/to/prepared-normal-runtime \
IRISU_METSU_RUNTIME_DIR=/path/to/verified-metsu-runtime \
  apps/web/build-site.sh build-metsu-web/combined-site
python3 -m http.server 8768 --bind 127.0.0.1 \
  --directory build-metsu-web/combined-site
```

See [the Metsu guide](METSU.md) to prepare its runtime. The combined build
stages both clients before replacing an existing generated site, so a failed
Metsu build preserves the previous normal site.

GitHub Pages uses this combined build. It downloads the unchanged pinned
normal release and separately builds Metsu on Ubuntu 24.04, checks its recorded
states against original-game golden digests, then verifies native/v86 parity.
The verified Metsu runtime is cached by engine, fixture, packaging and workflow
inputs. Only `main`, `codex/github-pages`, or an explicit workflow dispatch
deploys; building a local branch does not publish it.

## Controls

- Left click or `W`: weak shot
- Right click or `S`: strong shot
- Shift + click: both shots
- Mouse wheel down or hold Space: fast-forward (wheel up or releasing Space stops)
- Enter: pause/resume
- In replay mode, Left/Right: jump backward/forward five seconds without changing play state
- Replay speed: 1x, 2x, 4x, or 8x
- `R`: restart with a new random seed
- **play replay**: open an original v2.03 normal-mode `.rpy` file
- **save replay**: download the completed run from the game-over dialog

Touch taps fire weak shots; desktop players can right-click or press `S` for a
strong shot.

## Spotify soundtrack

A compact 80-pixel Spotify player is centered directly below the game. The first three original
soundtrack songs play in album order and repeat automatically, starting with a random
one of the three on each new run. A loss interrupts them with **Game Over 1**. There is
no song selector or full-album mode. Score changes, game pauses, replay exhaustion,
and level completion do not trigger the loss cue.

Gameplay starts immediately and requests Spotify playback automatically. Browsers
may block sound until the first gameplay click or keypress. Selecting the loss cue
always requests playback even if the previous song has stopped. If the browser
blocks a playback request, subsequent gameplay interactions can retry; Spotify's play
button remains available. Manual pauses persist until the next run or loss cue.
All audio stays in Spotify's official embed; full-track availability depends on
Spotify and the browser. Restricted previews do not automatically cycle or loop.
The isolated iFrame API falls back to a plain Spotify embed if unavailable.

See [music research](../../reference/music-rules.md) for the original game's
rules and track mappings. This site's automatic three-song rotation is an
intentional convenience; the original game uses a manually selected song.

## Build the standalone normal app

Build the existing 32-bit exact worker and provide its MSVC9 r58 host, then run:

```bash
IRISU_EXACT_WORKER=/path/to/irisu-exact-worker \
IRISU_EXACT_HOST=/path/to/libirisu_box2d_msvc_exact_multiworld.so \
  apps/web/build-static.sh
```

Runtime preparation performs a linker-only restage of that build's pinned
worker object and `libirisu_core.a` against the guest toolchain. If those files
do not share the worker's standard CMake build directory, set
`IRISU_EXACT_WORKER_OBJECT` and `IRISU_EXACT_CORE_ARCHIVE` explicitly.

This writes the standalone normal artifact to `apps/web/dist`. The static site boots
the exact i386 worker under v86 inside a Web Worker and requires no application
server or cross-origin isolation. Dependencies are pinned and cached in
`build-web/downloads`; generated guest binaries remain outside source control.
The client overlaps emulator and guest-engine downloads, uses v86 fast boot,
reuses parity-verified loader and C runtime libraries already in the guest
image, and shows
the current startup phase until the first exact state is available.
The workspace's standard exact-worker paths are used when the two environment
variables are omitted. The build reproducibly compiles the pinned v86
CORE-MATH patch; `IRISU_V86_WASM` may point at a prebuilt artifact with the
required hash. It also builds the pinned project-owned Linux guest; a prebuilt
copy with the required hash may be supplied as `IRISU_GUEST_BZIMAGE`. See
`EXACT_RUNTIME.md` for provenance and licensing.

Serve the completed directory with any ordinary static file host for local
testing, for example:

```bash
python3 -m http.server 8000 --directory apps/web/dist
```

Open <http://127.0.0.1:8000>. Gameplay and physics remain entirely in the
browser; this process only serves immutable files.

Saved replays use the original v2.03 52-byte header and per-20 ms input-word
format. The exact actions accepted by the simulator are recorded, including
release and idle ticks, and terminal metadata comes from the first recorded
finish. Replay playback applies the original two-frame startup edge rule.
The transport can pause, jump five seconds, scrub without interrupting playback,
and play at 1x, 2x, 4x, or 8x; long backward seeks are rebuilt from
the seed and immutable input stream when their exact observations have left the
bounded in-memory cache.

GitHub Pages downloads the hash-pinned runtime from the
[`web-exact-runtime-virtio-v1-20260824`](https://github.com/Gabriel-Kahen/irisu-ml/releases/tag/web-exact-runtime-virtio-v1-20260824)
release with `fetch-exact-runtime.sh`, verifies both the archive and its embedded
runtime manifest, and passes the prepared directory through
`IRISU_EXACT_RUNTIME_DIR`.
