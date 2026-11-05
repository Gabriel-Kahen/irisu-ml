# Actual original-game state trace

`original_state_proxy.c` observes the complete pinned v2.03 executable and its
unmodified shipped Box2D DLL under Wine. It does not run the component harness
and does not substitute clone rules or clone physics.

Build with `tools/metsu/build_original_state_proxy.sh`. Deploy only to a fresh
disposable copy of the authenticated original game: rename its authentic
`data/dll/Box2D.dll` to `Box2D.real.dll`, then install the observer as `Box2D.dll`.
The original executable and authentic sibling DLL must retain their manifest
hashes. The deployment/runner receipt binds these files, the observer source,
observer DLL, Wine runtime, replay, and trace outputs.

## Observer hook and boundary

The observer includes the existing transparent Box2D forwarding source and
checks the five-byte original CALL at `0x40666d`. It changes that mapped CALL's
destination to a wrapper which still calls the authentic `0x4031ec` actor-pool
update exactly once. The executable file is not rewritten. Observer callbacks
save/restore all general registers, EFLAGS, and FXSAVE machine state. The512-byte
FXSAVE region is aligned **up** inside a528-byte reserved area, keeping call
arguments below it. The state callback logs integer/raw-bit values only.

The post-call boundary is after the complete original actor pass and before
the next input poll. Reset is frame0; frameN follows the Nth gameplay input.
The observer records actual `Input+0x94` index separately, so alignment cannot
be manufactured by relabeling a skipped input word. Frames are emitted when
the original scene counter advances, including a terminal update.

The static API forwarder separately logs every authentic physics call and
body-ordinal/user-data association to `data/dll/box2d-trace.jsonl`. State JSONL
is `data/dll/metsu-state.jsonl`. Each state includes all live role4/5 actors
in original pool order and all200 retained pool colors, including dead slots.
Raw float32 fields are eight-digit hexadecimal strings. Group pointers are
process-local diagnostics; comparisons must compare group counters and sharing,
not pointer addresses across processes.

## Controlled launcher

The observer offers explicit diagnostic selection through environment variables:

- `IRISU_METSU_REPLAY`: original replay filename, relative to the disposable
  game's working directory or an explicit Wine path.
- `IRISU_METSU_REPLAY_SHA256`: runner-verified replay-copy hash, attached to
  metadata; this attestation supplements observed loader count/seed/mode.
- `IRISU_METSU_MAX_FRAME`: optional finite capture limit. The original process
  exits after recording that boundary. Omit for natural game completion.
- `IRISU_METSU_LAUNCH=live`: without a replay filename, select a genuine live
  original run, using the original mouse polling, time-derived seed, recorder,
  and replay writer. The runner must distinguish these files from supplied
  replay playback.

Selection waits for the actual title scene (`vtable 0x435c10`) and initialized
Input/SaveData objects. It calls the original mode setter `0x408594`, profile
and ticket builders `0x412268`, class allocator `0x418ec8` with ScenePlay class
metadata `0x4347d0`, and the original ScenePlay initializer `0x40941c`.
Playback sets the original scene's replay flag and filename selection; the
initializer itself invokes the authentic `Input.load` at `0x410d20`, reads the
52-byte header and input records, and seeds the authentic RNG. It does not
inject parsed words into game memory or bypass the loader.

The original title teardown `0x40eeec` and active-scene transition are completed
before the initial actor pass, matching the original menu-selected transition
at `0x406632–0x40663f`. Merely setting the title's next-scene pointer after that
outer-loop transition point consumes an extra title-only input; early smoke001
was diagnostic only for this reason. Smoke002/003 also precede the robust
FXSAVE-area alignment fix. Smoke004 is the first validated fixed-hook smoke.

The launcher bypasses manual menu/file-dialog selection. Its mode, filename,
and scene selection are controlled test setup; gameplay initialization,
replay decoding, RNG, physics, rule updates, scoring, termination, and writer
are original code. A live recorder run uses original sampled input and actual
writer output; a supplied playback file must never be described as a new
original-recorded replay.

## Owning addresses and observations

| State | Original location |
|---|---|
| Actor-pool D array length / pointer | `0x442620` / `0x442624` |
| Input object | pointer at `0x4428a4` |
| SaveData object | pointer at `0x4428d0`; data pointer at object+8; mode at data+0x1c |
| Scene frame | `0x4428e0` |
| Score / highest chain / level / qualifying clears | `0x4428c0`, `0x4428c4`, `0x4428c8`, `0x4428cc` |
| Gauge | `0x4428e4` |
| Seeded RNG call counter | `0x442854`, zeroed by original `SRand` wrapper `0x40ffb0`; incremented by original `GetRand` wrapper `0x40ffc4` |
| Replay header / loaded or recorded word count / playback index / playback flag | `Input+0x4c`, `+0x80`, `+0x94`, `+0x98` |
| Live scene seed / scene play-state | `ScenePlay+0x38` / `+0x30` |

Actor offsets and contact meanings are in `reference/metsu-rules-analysis.md`.
The final JSONL `complete` record distinguishes `frame-limit` from
`original-finished`; the latter is emitted after the original termination code
has run (and, in a recording run, after its replay writer has been called).
A capture limit alone is not evidence of an original completed run.

The observer captures the actual x87 control word, including its value before
lazy observer DLL loading, and restores the previously observed CW around that
loading. It never forces a canonical precision mode. Wine11.21 smoke004 uses
`0x137f` throughout, whereas historical normal qualification used `0x027f`.
This environment difference must be preserved in receipts and comparisons;
component-oracle success is not a substitute for full-process comparison.

Run only on the explicitly private Xvfb display with `WAYLAND_DISPLAY` and
`WAYLAND_SOCKET` unset and Wine Wayland driver disabled. This keeps the test
away from the user's signed-in display, physical pointer, and open apps.

## Title warmup calibration

The current launcher waits60 ordinary title actor passes before selection, so
normal title rendering and lazy graphics initialization can run first. Override
only for a labeled diagnostic with `IRISU_METSU_TITLE_WARMUP=N`. A separate
`bootstrap` record preserves the first title CW and the warmed title CW.
`IRISU_ORIGINAL_MODE=0` selects a normal replay control; default mode is1.
Frame rows always read actual SaveData mode and current level globals rather
than trusting this requested mode.

Wine11.21 smoke005 reports61 observed title passes, first/warm CW both137f,
and actual `ex_spawn_interval=175` at reset and frames1..3. Warming the title
therefore does not change this runtime's precision/cadence behavior. The native
original setter probe run with that observed control word also yields175; the
PC53 table yields176. This component observation explains the first full-state
spawn divergence but is not a replacement for the actual-process trace.

The first genuine browser export capture consumes all3788 original input words
and retains actual score40, gauge520, and active scene state0 at the capture
limit. The browser recorded termination on its3788th input. Thus matching the
replay header's score40 does **not** demonstrate state or terminal parity;
the full comparison must retain that failure. Receipt and raw original trace
are under `reference/captures/metsu-browser-original-20261103-001/` and
`reference/runs/metsu-browser-original-20261103-001/` respectively.

## Active-input completed replay qualification

`generate_controlled_replay.cpp` produces a deterministic **native-policy**
coverage run; it must not be labeled a browser export or original recording.
It aims at young ungrouped EX actors, alternates weak and strong shots every24
updates, holds each button for3 updates, packs integer cursor coordinates before
stepping, and writes only a naturally completed replay's canonical first-finish
header. Seed1 with Metsu config `1958987580749614460` produces4652 inputs,
score426, level4, highest chain3, and18 qualifying clears. EX chains occur;
this run does not reach the special-orb scheduler threshold.

The actual original executable consumed that complete replay with the same
natural finish. The strict comparison of reset plus all4652 updates passes:
4653 frames,271122 actor states,28 actor fields (including raw position,
rotation, and velocity bits), scalar/RNG/pool state, normalized live group
sharing on all 4,653 frames, and 6 first-finish metadata
fields. The original finish hook directly observes canonical426/4/3 at input4652
with pre-clamp gauge-320; the post-actor state has gauge1. Both original and
native preserve that first-finish tuple and the saved header agrees.

Evidence:

- `reference/captures/metsu-controlled-policy-20261103-001/controlled.rpy`
  SHA-256 `c76a26c0f1b5a8526efaf53f41d704e5275c608077d07cb1d6ca4d782d632fa8`.
- Same capture directory: `source-metadata.json`, `native-state.jsonl`, and
  strict `comparison.json` (`--require-natural-terminal`).
- Actual original receipt:
  `reference/captures/metsu-controlled-original-20261103-001/receipt.json`.
- Raw original state/physics traces:
  `reference/runs/metsu-controlled-original-20261103-001/data/dll/`.

This is finite end-to-end evidence through multiple level transitions, input
edges/held levels, and EX chains. It does not prove unbounded trajectory parity,
all rare contact combinations, or special-orb scheduling during a full run.
The retained failed old-parameter browser capture above remains diagnostic
history; it must not be substituted for a newly qualified browser export.

## Canonical finish and replay interchange

The observer also replaces the mapped ScenePlay virtual `+0x68` slot
`0x434888` with a transparent wrapper around authentic finish entry `0x40adc4`.
It preserves registers, flags, and aligned FXSAVE state before tail-jumping to
the original routine. Each `finish` record captures the actual entry score,
level, highest chain, qualifying clears, scene frame, and input index. Frame
rows retain the first observed tuple and total finish-call count. This
distinguishes canonical saved metadata from later changes in the same update.

`compare_state_traces.py` validates raw trace/replay hashes against the runner
receipt. It compares 8 scalar/pool fields, 28 actor fields as raw float32 bits
where applicable, all live actor slots, and the partition of non-null group
links. The latter compares original pointers with native group IDs by which
live slots share each record, including EX actors whose countdown has not yet
set their grouped flag. Equal group counters alone are insufficient.

The genuine original-written fixture
`tests/fixtures/metsu/original-generated.rpy` contains 2,994 inputs. Its live
recording and its subsequent authentic original self-playback both match the
native engine at all 2,995 boundaries and 41,999 actor states, including group
sharing. The self-playback captures a real first finish at scene frame 2,993 /
input 2,994, canonical score 0 / level 1 / chain 0; the final gauge is 1.
Strict receipts are
`benchmarks/results/metsu-original-writer-native-2026-11-03.json` and
`benchmarks/results/metsu-original-writer-playback-native-2026-11-03.json`.

The active-input fixture is reproducible without the original executable:

```sh
cmake --build build-metsu-exact --target irisu-metsu-generate-controlled
build-metsu-exact/irisu-metsu-generate-controlled 1 OUTPUT.rpy
```

Its stable repository input is `tests/fixtures/metsu/controlled-level4.rpy`;
the strict original-process report is
`benchmarks/results/metsu-controlled-original-native-2026-11-03.json`.
Actual browser Save/Load actions are recorded separately from native replay
execution in `benchmarks/results/metsu-replay-interoperability-2026-11-03.json`.
See `REPLAY_INTERCHANGE.md` for capture and comparison commands.

The corrected actual browser Save fixture
`tests/fixtures/metsu/browser-export.rpy` has SHA-256
`18d630363610072e123182421353db9ef79a2df2fee4f9473c72d4499a545e9e`.
The authentic original loads its 3,942 inputs and naturally finishes with the
same canonical 40 / 1 / 2 header. The strict comparison checks all 3,943
boundaries, 57,778 actor states, all group partitions, and canonical finish
fields without a mismatch. The original finish entry is observed at scene
frame 3,941 / input 3,942; its gauge is -80 on entry and 1 after the update.
Report: `benchmarks/results/metsu-browser-export-original-2026-11-03.json`.
This new run supersedes the old premature export as the positive browser route,
while retaining that old file's failed terminal qualification.

These are finite observations of the recorded runs. The controlled run reaches
level 4 and EX chains; it does not cover every level, the full-run special-orb
scheduler, every allocator/contact combination, or every possible input stream.
Parameter/component probes and browser/native checks retain their distinct
scopes; none alone establishes arbitrary-run original-game equivalence.
