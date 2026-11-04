# Capacity512 qualification commands

These are queued GPU commands, not executed results. Run after the coordinator
grants the GPU and freezes the selected profile and its source. Do not rebuild
the shared cap directory while any gate runs. The512 profile must retain SAP
and Game vector capacities4096. Raw full-scene v1/v2 packets from4096 builds are
incompatible; the dynamic packet below is capacity-independent.

The current cap directory has also been used for division and batching pilots.
A CMake cache alone does not prove which profile a binary contains. Before
using these commands, the coordinator must build the listed consumers from the
same final profile and record its compile definitions plus binary/source pins.
In particular, do not call old DIVOFF binaries DIVON or apply an older full4096
GPU record receipt to512. A later trig or division change requires its own
component proof and final integrated gates.

## Paths and pins

Run from any shell after setting these task-specific variables:

```sh
IRISU_CAP_ROOT=/home/gabe/Documents/ChatGPT/irisu
IRISU_CAP_BIN="$IRISU_CAP_ROOT/build-cuda-native-efficiency-cap/clone/cuda_native"
IRISU_CAP_OUT="$IRISU_CAP_ROOT/artifacts/gpu-efficiency2-20261104/cap512-final-gates"
IRISU_CAP_REPLAY="$IRISU_CAP_ROOT/irisu-high-score-4393101.rpy"
IRISU_CAP_LIBRARY="$IRISU_CAP_ROOT/.local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so"
IRISU_CAP_PACKETS="$IRISU_CAP_ROOT/artifacts/cuda-native-20261103/dynamic-scene/packets.bin"
IRISU_CAP_SPECS="$IRISU_CAP_ROOT/artifacts/gpu-efficiency-20261104/search-gates"
IRISU_CAP_SANITIZER=/opt/cuda/compute-sanitizer/compute-sanitizer
mkdir -p "$IRISU_CAP_OUT"
sha256sum "$IRISU_CAP_BIN/irisu-cuda-native-game-runner" \
  "$IRISU_CAP_BIN/irisu-cuda-native-search-runner" \
  "$IRISU_CAP_BIN/irisu-cuda-native-dynamic-scene-runner" \
  > "$IRISU_CAP_OUT/binaries-before.sha256"
```

The replay SHA must be
`81fe723b6b825e62afb471d7517355fef2b8bf0fb1dbe636069c97c70ce2ca79`;
the native library must be
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`.
The helper scripts below enforce those applicable pins. Save the final profile
cache/compile definitions separately; source-at-invocation hashes in
`qualify_game.py` are explicitly not a build provenance receipt.

For the original cap-only profile, immutable executable archives are under
`artifacts/gpu-efficiency-20261104/contact-capacity/*-initial`. Their SHA pins
are in `contact_capacity_qualification.json`. They have DIVOFF and represent
the81b0900 build, not subsequent division/trig changes.

## Full continuous GPU replay

First catch integration failures with512steps:

```sh
python3 "$IRISU_CAP_ROOT/tools/cuda-native/qualify_game.py" \
  --native-oracle "$IRISU_CAP_ROOT/build-project-exact/irisu-game-oracle" \
  --runner "$IRISU_CAP_BIN/irisu-cuda-native-game-runner" \
  --library "$IRISU_CAP_LIBRARY" --replay "$IRISU_CAP_REPLAY" \
  --steps 512 --gpu --timeout 120 \
  --output "$IRISU_CAP_OUT/prefix512.json"
```

Then execute all97,219 records continuously on GPU, including initial frame:

```sh
python3 "$IRISU_CAP_ROOT/tools/cuda-native/qualify_game.py" \
  --native-oracle "$IRISU_CAP_ROOT/build-project-exact/irisu-game-oracle" \
  --runner "$IRISU_CAP_BIN/irisu-cuda-native-game-runner" \
  --library "$IRISU_CAP_LIBRARY" --replay "$IRISU_CAP_REPLAY" \
  --steps 97219 --gpu --timeout 900 \
  --output "$IRISU_CAP_OUT/full-record.json"
```

Require passed=true; exactly97,220 rows of96bytes; native and typed exit0;
all output bytes equal; unchanged inputs/sources; failed=false and physics
status0. Expect308,512 native events, live score4,393,699, recorded canonical
score4,393,101, level100 and chain70. This proves continuous Game-digest wire
parity for the published replay, not raw hidden native-state equality or a
throughput ratio. The native oracle independently checks the terminal header.

## Dynamic physical/SAP state

```sh
env -u CUDA_VISIBLE_DEVICES \
  "$IRISU_CAP_BIN/irisu-cuda-native-dynamic-scene-runner" "$IRISU_CAP_PACKETS" \
  > "$IRISU_CAP_OUT/dynamic.json" 2> "$IRISU_CAP_OUT/dynamic.stderr"
env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_SANITIZER" --tool memcheck --error-exitcode 77 \
  --log-file "$IRISU_CAP_OUT/dynamic-memcheck.sanitizer" \
  "$IRISU_CAP_BIN/irisu-cuda-native-dynamic-scene-runner" "$IRISU_CAP_PACKETS" \
  > "$IRISU_CAP_OUT/dynamic-memcheck.json" 2> "$IRISU_CAP_OUT/dynamic-memcheck.log"
```

Require16worlds×192steps=3,072 exact native canonical physical/SAP hashes,
all statuses0 and memcheck0errors. This packet exercises all collision families,
creation/deletion/reuse, warm caches, setters, partial bounds and freezing.

The selected profile must also build `irisu-native-compact-snapshot-validate`
and `irisu-native-compact-game-validate`; the initial capacity build deliberately
stopped before those targets. Once available, execute:

```sh
env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_BIN/irisu-native-compact-snapshot-validate" \
  "$IRISU_CAP_PACKETS" > "$IRISU_CAP_OUT/compact-scene.json"
env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_SANITIZER" --tool initcheck --error-exitcode 77 \
  --log-file "$IRISU_CAP_OUT/compact-scene-initcheck.sanitizer" \
  "$IRISU_CAP_BIN/irisu-native-compact-snapshot-validate" \
  "$IRISU_CAP_PACKETS" --virgin-scratch \
  > "$IRISU_CAP_OUT/compact-scene-initcheck.json" 2> "$IRISU_CAP_OUT/compact-scene-initcheck.log"
for IRISU_CAP_PREFIX in 100 96000 97150; do
  IRISU_CAP_STEPS=192
  if [ "$IRISU_CAP_PREFIX" = 97150 ]; then IRISU_CAP_STEPS=69; fi
  env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_BIN/irisu-native-compact-game-validate" \
    "$IRISU_CAP_REPLAY" "$IRISU_CAP_OUT/full-record.native.frames.bin" \
    "$IRISU_CAP_PREFIX" "$IRISU_CAP_STEPS" \
    > "$IRISU_CAP_OUT/compact-game-$IRISU_CAP_PREFIX.json"
done
env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_SANITIZER" --tool initcheck --error-exitcode 77 \
  --log-file "$IRISU_CAP_OUT/compact-game-initcheck.sanitizer" \
  "$IRISU_CAP_BIN/irisu-native-compact-game-validate" \
  "$IRISU_CAP_REPLAY" "$IRISU_CAP_OUT/full-record.native.frames.bin" 100 192 --virgin \
  > "$IRISU_CAP_OUT/compact-game-initcheck.json" 2> "$IRISU_CAP_OUT/compact-game-initcheck.log"
```

Require12,288 scene snapshots and9,216 native frame checks; zero-iteration warm
impulse, pending-delete, origin, sleeping and frozen witnesses. Game gates must
compare each continued frame to native plus all live value fields and active
Scene/SAP state before/after. Padding is explicitly ignored only for listed
Game element types. Initcheck's virgin scope is **SceneScratch only**, with all
persistent Game/Scene storage initialized once, matching the actual fast path.
Default poisoned Game tails independently exercise their semantic irrelevance.

## Search integration and launch groups

This scalar runner gate creates distinct specs, reruns the pinned native oracle,
and checks full Result/rank packets, compact selections/continuations, immutable
prefix guards, trace/final-only modes, GPU/host prefixes and workspace poisons:

```sh
cd "$IRISU_CAP_ROOT"
python3 tools/cuda-native/qualify_search_efficiency.py \
  --native "$IRISU_CAP_ROOT/build-project-exact/irisu-search-oracle" \
  --runner "$IRISU_CAP_BIN/irisu-cuda-native-search-runner" \
  --library "$IRISU_CAP_LIBRARY" --replay "$IRISU_CAP_REPLAY" \
  --output "$IRISU_CAP_OUT/scalar-search/qualification.json"
```

Require all8cases pass, including8and512candidates and12,288 genuinely
distinct futures. Selected-state checks and guards must be on, prefix execution
exactly1, and all failure/compaction/prefix errors, Guest and fallback counts0.
This is correctness evidence, not a campaign timing command.

If the selected runner uses two concurrent batches, separately qualify its
independent Result/rank partitions and odd final batch. The following8-case
command keeps all8selected leaves, runs2+1 batches and compares the final packet
to the actual native packet produced by the scalar qualification above:

```sh
env -u CUDA_VISIBLE_DEVICES "$IRISU_CAP_BIN/irisu-cuda-native-batched-search-runner" \
  "$IRISU_CAP_REPLAY" "$IRISU_CAP_OUT/scalar-search/specs-8-100.bin" 100 4 \
  --concurrent-batches 2 --host-prefix --final-only --warmups 0 --repeats 3 --keep 8 \
  > "$IRISU_CAP_OUT/batched8.bin" 2> "$IRISU_CAP_OUT/batched8.log"
cmp "$IRISU_CAP_OUT/batched8.bin" "$IRISU_CAP_OUT/scalar-search/native-8-100-0.bin"
```

Repeat this admitted launch-group shape for trace mode (omit --final-only), GPU
prefix (omit --host-prefix), and512/12,288counts using matching native packet
and generated specs in the scalar gate directory. For12,288final-only,3batches
must report147,456 actual future ticks, two independent rank partitions and
the remainder batch. The final20-batch campaign must report983,040 GPU future
ticks; sharing one CPU100-tick prefix is an explicit hybrid mode. Do not replace
these full-work checks with recycled result packets or a warm phase ratio.

Run one8-candidate selected-continuation launch group under memcheck and
initcheck using the exact same flags, with outputs and native comparison saved.
Use `--log-file` with a separate sanitizer log for every binary packet
comparison: Compute Sanitizer otherwise writes its banner and summary to
stdout. Compare the complete clean child packet; require zero errors in that
separate log. Record batch partition/selection guards, not merely equality
of the final packet. Batched binaries require their own before/after hash pin in addition to
the three setup binaries above. Finally verify those initial binaries were not
replaced during the gates:

```sh
sha256sum -c "$IRISU_CAP_OUT/binaries-before.sha256"
```

## CPU evidence already available

`contact_capacity_qualification.json` records full97,220-frame native parity
and explicit overflow rejection under both4096 and512, with observed typed
call-boundary contact/SAP pair peaks64 at tick97,161. This is not a hidden
native or transient allocator high-water measurement. `b197620`'s
`div_store_qualification.json` additionally records DIVON/cap512 checked-host
full-record parity and scalar division qualifications. Reuse those scoped
receipts; neither substitutes for final-profile GPU integration. Legacy CPU
tests are unaffected by these opt-in typed-only options, so a wholesale repeat
is unnecessary unless subsequent shared/default code changes.
