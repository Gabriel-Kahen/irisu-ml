# Metsu replay interchange and original-game state checks

These tools compare the actual v2.03 game with the separate native Metsu
engine. Native/v86 agreement is a different gate and cannot replace this one.
Original executable, DLL, replay captures, raw traces, and receipts remain in
ignored reference directories; tracked reports contain their hashes.

## Capture the native side

Build the optional exact Metsu tracer and capture a mode-1 replay:

```sh
cmake --build build-metsu-exact --target irisu-metsu-trace -j 4
python3 tools/metsu/trace_replay.py \
  --tracer build-metsu-exact/irisu-metsu-trace \
  --replay INPUT.rpy --output NATIVE.jsonl
```

The wrapper computes replay and executed-tracer hashes, rejects portable
physics, and refuses a replay or binary that changes during capture. The
underlying C++ tool resets from the recorded seed, applies one input word per
update, and preserves the original two-frame replay edge suppression. It
emits reset plus every post-actor-pass state through natural termination or
input exhaustion. It does not request unsupported exact physics snapshots.

Actor identity is the original 200-slot pool index, not the clone's body ID.
The trace preserves raw float32 coordinates, rotations, velocities, sizes,
lifecycle/contact flags, EX timers, group counters, stale pool colors, and
score/gauge/level/chain/clear counters. RNG calls are measured by advancing an
independent generator until its full state and index match the engine, rather
than estimating calls from a wrapping index.

## Compare with an actual original capture

Use the raw original JSONL and hash-binding receipt produced by
`run_original_replay.py`; see `ORIGINAL_STATE_TRACE.md` for that process.
Do not rewrite raw observer output to insert missing metadata.

```sh
python3 tools/metsu/compare_state_traces.py \
  --original ORIGINAL.jsonl --candidate NATIVE.jsonl \
  --replay INPUT.rpy --receipt ORIGINAL-RECEIPT.json \
  --output STATE-REPORT.json
```

This checks complete update coverage and every required scalar and actor
field, comparing float bits without rounding or tolerance. Duplicate slots,
missing fields, missing completion records, truncated matching prefixes,
stale receipts, and empty body coverage cannot pass. Bootstrap and transparent
finish-entry records are preserved. Static boundary roles are excluded from
the dynamic actor list explicitly. Original group pointers and native group
IDs are compared by their normalized partition of live slots, including
pending EX links, rather than by numeric identity or equal counters alone.

The default result qualifies **state parity for the recorded inputs**.
Matching states at input exhaustion does not establish that a saved replay
contains a completed game.

## Qualify a saved replay

Add `--require-natural-terminal` for saved-replay interchange:

```sh
python3 tools/metsu/compare_state_traces.py \
  --original ORIGINAL.jsonl --candidate NATIVE.jsonl \
  --replay INPUT.rpy --receipt ORIGINAL-RECEIPT.json \
  --require-natural-terminal --output INTERCHANGE-REPORT.json
```

Both engines must naturally finish at the matched endpoint. Header score,
level, and chain must equal the native's canonical first finish and the
original's actual first-finish hook values. A hash-bound receipt proving the
original engine itself wrote the input replay can supply that canonical
tuple when an older capture lacks the finish hook. The latest post-update
score is never substituted for first-finish metadata.

Actual browser Save and file-chooser Load actions are separate required
receipts for browser interoperability. A successful native check alone is
not evidence that either browser UI action occurred.

`--through-frame N --diagnostic` can locate an early divergence while a
capture is in progress; its report always remains diagnostic and cannot pass
an interchange gate. Numerical-environment metadata is recorded separately
from measured state equality. In particular, actual original Wine captures
can report `137f` while the native physics profile reports `027f`.

Comparator regression checks:

```sh
python3 tests/test_metsu_state_comparison.py
```
