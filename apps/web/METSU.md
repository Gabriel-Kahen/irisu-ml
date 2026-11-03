# Separate Metsu web client

Metsu uses `metsu-static/`, a separate mode-1 exact worker, and a separate
runtime manifest/cache. The normal `static/`, build scripts, pinned runtime,
and deployment workflow are unchanged. This is a local build path; it does
not publish the site.

The client keeps normal's controls, run recording, original-format replay
transport, seeking, pause/restart, and fast-forward. It accepts and saves only
mode-1 replays. Its field rendering uses Metsu's recovered fixture geometry,
nine distinct colors, and a ring identifying EX circles. Normal-mode replays
must be opened in the normal client. No normal-mode soundtrack rotation is
applied to Metsu.

The client uses normal mode’s original in-field HUD and interface. EX circles
have amber rings; diagonal stripes identify rotten blocks without changing their color.
Trails, clear bursts, and animated ring accents respect reduced-motion settings.

## Build and preview

First build `irisu-metsu-exact-worker` as described in `docs/metsu.md` and
obtain the unchanged pinned normal browser runtime:

```sh
apps/web/fetch-exact-runtime.sh .local/normal-web-runtime
apps/web/prepare-metsu-runtime.sh build-metsu-web/runtime \
  --baseline .local/normal-web-runtime \
  --worker build-metsu-exact/irisu-metsu-exact-worker \
  --steps 3500 --seeds 0,41,123

IRISU_METSU_RUNTIME_DIR="$PWD/build-metsu-web/runtime" \
  apps/web/build-metsu-static.sh build-metsu-web/site/metsu
python3 -m http.server 8001 --directory build-metsu-web/site
```

Open `http://127.0.0.1:8001/metsu/`. To preview both clients together, build
normal into the parent site directory first, then stage Metsu beneath it:

```sh
IRISU_EXACT_RUNTIME_DIR="$PWD/.local/normal-web-runtime" \
  apps/web/build-static.sh build-metsu-web/combined-site
IRISU_METSU_RUNTIME_DIR="$PWD/build-metsu-web/runtime" \
  apps/web/build-metsu-static.sh build-metsu-web/combined-site/metsu
python3 -m http.server 8001 --directory build-metsu-web/combined-site
```

Normal remains at `/`; Metsu is at `/metsu/`. Rebuilding the parent site
requires staging Metsu again afterwards. Neither command changes the source
of the other client.

## Runtime verification and provenance

Runtime preparation rejects normal workers before downloading dependencies.
It checks the distinct `exact-msvc9-r58-metsu-forward` backend, protocol/ABI,
configuration hash, and pinned physics-host identity. It verifies the pinned
v86/kernel/BIOS files and packages hash-locked Debian i386 C runtime libraries.

The unchanged guest kernel launches a small shell wrapper. That wrapper runs
the packaged loader with an explicit library path, avoiding any dependence on
the older guest's libc ABI. The executable's incorrect host x86-v3 policy note
is removed; the actual i386 code remains intact. A native/v86 byte comparison
must pass before the runtime receives its verified marker. The check includes
seeded shot/release inputs, body states, events, configuration diagnostics,
EX interactions, and terminal metadata when reached. Worker changes during
preparation are rejected.

The generated `metsu-runtime.json` records all worker/dependency identities
and source routes. `PROVENANCE.md`, copyright/license files, and the preserved
baseline source offer identify the packaged components. The runtime cache key
includes every served guest file (including the launcher), emulator, kernel,
and BIOS hash. Changing any of those bytes changes the key. Cleanup only
touches the Metsu cache namespace; normal's cache and runtime remain separate.

`native-v86-parity.json` establishes equivalence between the packaged browser
engine and the native Metsu implementation for the recorded workload. Original
game fidelity is a separate claim; see `reference/metsu-rules-analysis.md` and
`docs/metsu.md` for that evidence and its limits.

Run the targeted client/packaging tests with:

```sh
node --test apps/web/tests/metsu/*.test.mjs
```
