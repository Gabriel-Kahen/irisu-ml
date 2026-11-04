# Metsu mode: original-binary evidence

Audit date: 2026-11-03. This ledger began as a bounded planning checkpoint and
now includes contact rules, original-component receipts, and measured actual
original-process replay comparisons. Qualification is finite and covers the
recorded inputs and fields described below, not every possible game state.
Addresses below refer to the original v2.03 engine.

## Original recovered and identified

The former `/home/gabe/Games/Irisu Syndrome` installation and the ignored
workspace reference installation were absent. The pristine archive was recovered
from the Internet Archive item already recorded in `manifest.md`:

- Retrieval: `https://archive.org/download/irisu-syndrome/irisu203%20%28original%29.zip`.
- Ignored archive: `reference/archives/irisu203.zip`.
- Archive SHA-256:
  `3fcc57ff0bbc4af1d262813173956ecc1563f26a39f1e59119643cf07400f2a5`.
- Ignored original executable:
  `reference/game/irisu-v2.03-jp/irisu203/irisu.exe`.
- Executable SHA-256:
  `0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255`.

Both hashes match the existing pinned manifest. The base Japanese executable is
the same engine as the English-patched research installation. The archive,
executable, and original assets remain ignored. The initial recovery checkpoint
used static `objdump -D -Mintel` inspection; later actual-process runs are
documented below. Local full
disassembly is cached in `.tmp/metsu-original-disassembly.txt`.

## Architectural distinction: scene mode and EX actor flag

**Confirmed/static:** scene mode selects a complete level-parameter setter,
not only a configuration table. The dispatch at `0x415600–0x415621` calls
`0x415624` for normal mode and `0x415800` for Metsu.

**Confirmed/static:** `Block+0xf0` is a per-actor EX flag. It must not simply
be set on every actor in a Metsu scene. The generator initializes its local EX
flag to zero at `0x405b1a`; only the EX spawn branch sets it to one at
`0x405e44`. The flag is copied to the actor at `0x405eab`. Thus ordinary blocks
in Metsu retain the ordinary actor path, while EX blocks use extra contact,
activation, and timer behavior. Earlier normal-only descriptions calling this
field a normal-versus-Metsu mode flag are too broad.

**Confirmed/static:** a nonterminal level change re-enters `Field.update`
before installing the next parameter set (`0x40af7c–0x40af99`). This existing
ordering matters with Metsu's second cadence as well as its ordinary cadence.
Reusing only the table while retaining normal level formulas is incorrect.

## Reset and cadence

**Confirmed/static:** at `0x4053a7–0x4053f9`, scene mode skips the normal ten
rotten blocks at Y=200. Both modes then execute the ten ordinary falling-block
calls at Y=60 (`0x4053f9–0x40540c`). The special scheduler is still called by
the shared field constructor; verify complete initial RNG ordering before
claiming reset parity.

**Confirmed/static:** field update first runs the ordinary cadence test and
ordinary generator (`0x4055e7–0x405605`), then, only in a nonzero scene mode,
runs the EX cadence test and generator (`0x405605–0x405631`). Both use the
same scene counter, so both tests pass at counter zero. Generator call arguments
are `(ex=false, initial=false, y=-50)` for ordinary cadence and
`(ex=true, initial=false, y=0)` for EX cadence.

The last argument is ignored by the EX branch, which computes its own Y.
Actor-pool allocation failures and RNG consumption need explicit probes.

## Metsu level formulas

The following are **confirmed/static** from `0x415800–0x4159f0`. Here `L` is
positive nonterminal level, `q=trunc(L/25)`, and `f32` denotes the executable's
single-precision store. Formulas describe operations; they are not a promise
that host libm reproduces the x87 operation sequence.

| Global | Role | Metsu expression / source |
|---|---|---|
| `0x436a10` | passive drain parameter | `0` (`0x415809`) |
| `0x436a14` | ordinary scripted descent | `1.0` if `L%24==0`; else float constant `0.6` if `L%6==0`; else `f32((L%6)*float(0.03)+float(0.1))` (`0x415813–0x415866`) |
| `0x436a18` | EX scripted ascent | selected `ex_block_speed`, shipped `-0.05` (`0x41586c–0x4158a6`) |
| `0x436a24` | ordinary spawn interval | `120-10*(L%8)` (`0x4158ac–0x4158ca`) |
| `0x436a28` | EX spawn interval | `200-trunc(abs(sin(L*2*pi/12))*50)` (`0x4158d0–0x415916`) |
| `0x436a1c` | rot penalty parameter | `600+50*q` (`0x41591c–0x415931`) |
| `0x436a20` | inclusive color maximum | `5+q` (`0x415937–0x41593a`) |
| `0x436a38` | clear reward unit | selected `game_life_plus_unit`, shipped `120` (`0x415940–0x41597e`) |
| `0x436a30` | score scale | `(4+q)*pow(L,0.7)` (`0x415984–0x4159b6`) |
| `0x436a3c` | triangle threshold | `100`, without a random draw (`0x4159bc`) |

The sine uses x87 `fsin`; the power uses the same original routine as normal.
The float constants' binary32 encodings, rather than exact decimal values,
are operands. EX-cadence values near integer boundaries and score-scale bits
need original-runtime dumps. The shared ordinary reward hook at
`0x4098fc–0x409927` multiplies group count by `0x436a38` when the clearing
block is not rotten. Other reward callers still need a complete mode audit.

The existing `game-rules-analysis.md` parameter comparison remains useful for
field dimensions, gauge limits, lifetimes, and the six-clear divisor. Its
statement that level formulas/reward are unchanged must not be extended to
Metsu: this separately selected setter proves otherwise.

## Ordinary and EX generation

**Confirmed/static:** ordinary Metsu X placement uses
`field_x + 30 + GetRand(field_width - 40)` at `0x405bf9–0x405c6f`, hence
`124+GetRand(380)` for shipped values. This corrects the older ledger's broad
claim that the nonzero-mode X maximum was `404`; `width-thickness` belongs to
the normal scene branch. Caller Y is retained for ordinary blocks.

**Confirmed/static:** EX placement at `0x405c80–0x405e44` is:

```text
x = field_x + ex_block_x_adj
    + GetRand(field_width - field_thick - 2*ex_block_x_adj)
y = ex_block_y1 + GetRand(ex_block_y2)
```

For shipped values this is `x=144+GetRand(304)`, `y=500+GetRand(0)`.
A zero-maximum RNG call is still a call and must not be optimized away.

After allocation, both paths draw rotation with `GetRand(1000)` and color
with `GetRand(color_max)` (`0x405e74–0x405ebb`). The EX path still consumes the
`GetRand(100)` draw at `0x406046`, then creates a circle through virtual
`+0x6c` using EX material keys (`0x406052–0x406118`). Its scripted Y velocity
is the EX parameter (`0x4062bf–0x4062c5`). EX blocks skip the ordinary
`block_life_span` assignment (`0x4062e0–0x406321`).

The size/slot lookup has separate arrays selected by the EX call argument
(`0x405a19–0x405aac`). The exact ticket-count state at `0x4428f0`, truncation
behavior, and array-building order have **not yet been fully traced**. Do not
infer EX sampling by normalizing the INI weights without auditing that builder.

The shipped INI provides these EX leads, not yet validated as a complete
runtime distribution: sizes `42,58,74,90,104,120`, weights
`20,40,50,50,25,15`; density `25`, friction `1`, restitution `0`.
Ordinary Metsu triangle cutoff is 100, so the ordinary shape draw cannot exceed
it; the selected ordinary path creates boxes. The special-orb threshold remains
in shared generation, excludes EX actors (`0x405eda–0x405ee3`), and requires
mode-aware reward validation.

## Contact/state-machine leads

These are **confirmed/static branch observations**, not a complete EX contact
specification:

- `0x40a2df` and `0x40a2f4` reject certain same-color rotten/EX pairings before
  the directional grouping helper.
- `0x40a369–0x40a3d9` handles two EX actors separately. It can attach an EX
  actor whose `+0xf4==-1` to a source with `0 < +0xf4 <=112`, increment the
  shared chain record, set target `+0xf4=120`, and set `+0xc4=1`.
- `0x40a449–0x40a477` also sets `+0xf4=120` and dynamic ownership for an EX
  target joining an already grouped source; the ordinary path sets `+0xd4`.
- The shared clear-source eligibility helper accepts EX actors independently
  of the ordinary `rot_timer>0 || bottom` condition (`0x40a4a4–0x40a4d3`).
- The EX rot-start branch adds a `+0xf8>=5` guard (`0x40a6bf–0x40a6e5`).
- Top-gate logic branches on the EX flag near `0x40a716`; it has not been
  completely decoded at this checkpoint.

The meaning and update/deletion semantics of `+0xf4` require the full actor
update audit. Implementing it as an ordinary grouped boolean would be premature.

## Initial checkpoint next steps (superseded where completed below)

1. Trace every `Block+0xf0`, `+0xf4`, and `+0xf8` read/write, and every scene
   mode branch in actor updates, contacts, clear-color, and terminal handling.
2. Decode both ticket builders and confirm reset/cadence RNG call sequences.
3. Dump all 99 Metsu level parameter bit patterns from the original runtime,
   especially score scale and EX sine cadence; retain provenance and hashes.
4. Produce controlled mode-1 replays and original traces for reset, ordinary/EX
   spawns, EX activation/propagation, rot, bonus, score, gauge, and ending.
5. Add mode identity to simulator config/snapshots/replays and browser worker
   admission only after the mechanics model is established. Keep normal-mode
   golden behavior and exact runtime identity checks intact.

At that initial checkpoint no complete original-game mode-1 replay parity or
Metsu simulator was claimed. Subsequent implementation and measured results
are recorded below; retain the distinction between those scopes.

## Implementation audit continuation (2026-11-03)

The following supersedes the bounded checkpoint's unresolved static mechanics
where explicit pseudocode is now supplied. Evidence remains original-machine-code
inspection unless an execution receipt is named. Actual-process replay evidence
is documented in the final qualification section.

### World and terminal scene behavior

`0x409467–0x409487` unconditionally calls:

```text
b2d_init(0, -200, 640, 480, 160, 10)
```

The five stored constants at `0x4344c0..0x4344d0` decode to `10,160,480,640,-200`;
there is no mode branch around this call. EX center Y=500 does not justify
expanding the broadphase AABB: a minimum EX radius21 extends to Y479 and
therefore still overlaps the world bound.

Scene update checks game-over at its beginning (`0x409939`), still executes
field update/contact traversal, clamps gauge to `0..max`, increments the scene
counter, applies passive drain, and floors gauge at1 (`0x409d14–0x409d73`).
Metsu's passive parameter is zero, so this shared code subtracts zero.
Ordinary actor rot runs afterward and can make gauge nonpositive for the next
scene update. Positive rewards can still exceed the maximum until the scene
clamp. The shared ordinary clear reward hook uses the selected reward120.
The rainbow `clear_color` loop skips all gauge rewards in a nonzero scene mode
(`0x403303–0x40331d`), while retaining its full-pool matching/deletion behavior.

The level cap path is shared and unconditional (`0x40af48–0x40af79`): requested
level>=100 stores100 and finishes; it does not call the nonterminal setter.
Six clears per level means clear594 first requests100, clear600 first requests101.
As with normal, a terminal call does not interrupt the current contact traversal.
Repeated terminal calls and first-persisted versus last-live ending metadata
therefore retain the ordering documented in the normal ledger. Music is
mode-specific; gameplay stopping semantics are shared.

### Ticket builder and original EX sampling bug

`0x412268–0x412295` expands the ordinary sizes/weights using `0x41207c`, writes
its total to `0x4428f0`, sizes to the D array at `0x442900`, and slot indices
to the D array at `0x442910`. In Metsu the total is70. The nonzero-mode path
then expands EX weights into sizes at `0x4428f8`, indices at `0x442908`, and
stores the EX total200 at `0x4428f4` (`0x41229a–0x4122cf`).

However, **the generator reads only the ordinary total**: `0x4059e6`,
`0x4059f3`, and `0x405a09` all read `0x4428f0`, not `0x4428f4`. It draws
`GetRand(69)` for both ordinary and EX actors. The call's EX flag selects
which ticket arrays are indexed afterward (`0x405a19–0x405aac`).
Consequently the shipped EX runtime distribution is:

| Random ticket | Size | Slot |
|---|---:|---:|
| 0..19 | 42 | 0 |
| 20..59 | 58 | 1 |
| 60..69 | 74 | 2 |

The configured EX sizes90/104/120 are unreachable from this generator with the
shipped70-ticket ordinary table. Do not replace this with a normalized200-ticket
draw. The optional `initial` flag uses the same ordinary-total random maximum
and clamps only against that same total (`0x4059ec–0x405a19`).

At the full field-construction checkpoint: level1 consumes no shape-cutoff RNG
call, then special scheduler consumes `GetRand(12)`, then ten initial ordinary
spawns each consume `(69,380,1000,5,100)`. This gives51 draws before replay
word0, assuming every allocation succeeds. The scene's pre-word actor pass,
startup input gate, and special threshold scheduler itself are shared. The first
cadence ordinary spawn consumes the same five calls; the first EX spawn consumes
`(69,304,0,1000,5,100)` in that order. The zero-maximum call consumes RNG.
EX skips orb conversion regardless of threshold; ordinary conversion still
omits the final shape draw on an orb spawn, as documented for normal.

### Actor state and post-contact update

Use separate scene mode and per-actor EX flag. New fields beyond normal state:
`ex = Block+0xf0`, signed `ex_timer = Block+0xf4`, unsigned
`dynamic_updates = Block+0xf8`. Constructor defaults are `(false,-1,0)` at
`0x4033b7–0x4033c3`. `Actor.remaining=-1` is explicitly written at
`0x40764d–0x407664`. EX generator leaves it unchanged, while ordinary actors
receive the configured10000-update lifetime. Existing out-of-bounds cleanup can
still set remaining1 for an EX actor once dynamic and `e5` is set.

After the shared base update, out-of-bounds check, fresh-state transition, and
post-contact cleanup, the actor path at `0x40348e–0x4035b1` performs:

```text
if not ex and lifecycle != rotten and rot_timer != 0:
    ++rot_timer
    if rot_timer > selected_death_delay and role == block:
        lifecycle = rotten
        gauge -= selected_rot_penalty
if dynamic:
    ++dynamic_updates
if ex_timer > 0:
    --ex_timer
    if ex_timer == 0:
        grouped = true
```

Thus EX actors can acquire `rot_timer=1` as a contact property but do not
increment it into the ordinary rotten state or pay a rot penalty. The countdown
runs after post-contact cleanup, so reaching zero sets grouped during the later
actor pass. It does not immediately set clear-pending, increment a group clear
count, reward gauge, or score. An eligible subsequent contact performs those
operations. No separate EX deletion or score routine is introduced: clear-pending
and ordinary cleanup drive shared scoring/destruction. Countdown expiry also
plays sound, which has no simulation state consequence.

### Full mode-aware contact pseudocode

Names map to existing normal ledger fields: `dynamic=c4`, `special=c5`,
`state=c8`, `role=cc`, `rot=d0`, `grouped=d4`, `clear_pending=d5`,
`group=d8`, `hits=dc`, `contacts=e0`, `gate_seen=e4`, `past_gate=e5`.
Roles are block4, projectile5, bottom2, side3, top6. Contact traversal order
is unchanged from normal; only the branch predicates below differ.

`both_scripted(A,B)`, `0x40a958–0x40a9de`:

```text
if A.dynamic or B.dynamic or A.ex != B.ex: return false
for actor in [A,B]:
    if actor.age <= 2 and not actor.special and actor.role == block:
        destroy_immediately(actor)
return true
```

Mixed EX/ordinary scripted contact therefore proceeds to grouping/activation.

`top_gate(target,source)`, `0x40a6f0–0x40a743`:

```text
if not target.dynamic and (
    (source.role == top and not target.ex) or
    (source.role == bottom and target.ex)):
    target.gate_seen = true
    return true
return false
```

The same `past_gate` latch/cleanup semantics apply; EX actors use the bottom
as their startup gate while ordinary actors use the top.

`group_pair(A,B)`, `0x40a2a4–0x40a357`:

```text
if A.color != B.color: return 0
if A.state == rotten and (B.state == rotten or B.ex): return 0
if B.state == rotten and (A.state == rotten or A.ex): return 0
if A.role != block and B.role != block: return 0
if A.grouped and B.grouped: return 0
return max(add_to_group(A,B), add_to_group(B,A))
```

`add_to_group(target,source)`, `0x40a35c–0x40a49e`:

```text
if target.ex and source.ex:
    if target.ex_timer == -1 and 0 < source.ex_timer <= 112:
        assert source.group != null
        target.group = source.group
        ++target.group.chain
        ++target.group.shadow
        target.ex_timer = 120
        target.dynamic = true
    return 0
if target.grouped or target.role != block: return 0
if source.grouped:
    assert source.group != null
    target.group = source.group
else:
    target.group = new zeroed shared chain record
++target.group.chain
++target.group.shadow
if target.ex and source.grouped:
    target.ex_timer = 120
    target.dynamic = true
else:
    target.grouped = true
return target.group.chain
```

The EX-to-EX propagation branch returns0 even when it attaches a chain.
It waits until the source has completed at least8 countdown decrements.
The mixed branch intentionally allows replacing the group and resetting the
timer of a still-ungrouped EX actor; no extra `ex_timer==-1` gate applies there.

`activate(target,source)`, `0x40a7d8–0x40a899`:

```text
if target.dynamic: return
activate = false
if not target.ex:
    if source.dynamic and source.role != side:
        activate = (not source.grouped or target.color == source.color
                    or source.role == bottom)
else:
    activate = (source.role == projectile and source.rot == 0) or (
        source.role == block and not source.ex and not source.dynamic)
if activate:
    target.dynamic = true
    if shared_sound == -1: shared_sound = 0
    if source.role == projectile: source.delete = true
```

`eligible_clear_source(source)`, `0x40a4a4–0x40a4d5`:
`source.ex or source.rot>0 or source.role==bottom`.
`eligible_clear_target(target)`, `0x40a4d8–0x40a50b`:
`target.grouped and target.role==block and not target.clear_pending`.

`burst(target,source)`, `0x40a510–0x40a627`:

```text
if not eligible_clear_target(target) or not eligible_clear_source(source): return
set target.clear_pending = true       # same virtual hook as normal
++target.group.num
if target.role == block:
    reward_hook(target)               # zero if rotten, else group.num * reward120
    ++qualifying_clears
    candidate = floor(qualifying_clears / 6) + 1
    if candidate > current_level: set_level(candidate)
if not target.ex: shared_sound_flag = 0
if target.color == source.color and eligible_clear_target(source):
    set source.clear_pending = true   # no extra num/reward/count increment here
```

The role check is present in the original even though the target predicate
already requires block. The wrapper `0x40a62c` calls this in both directions.
The normal actor cleanup scores pending clears with the selected Metsu score
scale, slot multiplier, and shared group record, preserving late mutations.

`start_rot(target,source)`, `0x40a66c–0x40a6ec`:

```text
if target.role not in {block,projectile}: return
if source.role != bottom and source.rot == 0: return
if target.special or target.rot != 0 or target.age <= 100: return
if target.ex and target.dynamic_updates < 5: return
target.rot = 1
```

`special(special,other)`, `0x40a748–0x40a7d4`, retains the normal predicates:
only an orb with `special && past_gate && dynamic` acts; a projectile is deleted
and the orb survives; an ungrouped ordinary/EX block that is not rotten invokes
full-pool clear-color and consumes the orb; an ungrouped rotten block consumes
the orb without clearing color. Grouped blocks do nothing. The scene-mode check
inside clear-color suppresses Metsu gauge gain entirely, as noted above.

`direct_hit(block,projectile)`, `0x40a89c–0x40a952`:

```text
if projectile.role != projectile or block.role != block: return
if projectile.contacts == 1 and block.grouped:
    ++block.hits
    if block.hits >= 2:
        projectile.delete = true
        block.delete = true
if block.state != rotten and not block.ex:
    projectile.delete = true
```

Parameter labels distinguish actor arguments from role constants. Initial EX
activation may already have deleted a projectile in the preceding handler;
this handler never reverses an earlier deletion. It adds no EX-specific hit
counter beyond the existing `hits`.


### Original-byte execution receipts and clone comparisons

The first parameter-component extraction ran the pinned original setters at
x87 control word `0x027f`. It reproduced all 99 prior normal score-scale and
descent rows, then produced the historical PC53 Metsu table
[`metsu-level-parameters.tsv`](metsu-level-parameters.tsv), SHA-256
`d1786c82eec4583f71cb23be664b879038b2929be262bb2726430025c3f73d1a`.
That component table reports a level-1 EX cadence of 176. It is retained for
calibration; it is **not the currently selected Metsu table**.

Actual original-process captures under Wine observed `0x137f` before observer
loading and throughout gameplay, including after title warmup. They spawn the
next EX actor on update 176, using cadence 175. The earlier PC53 implementation
used cadence 176 and first diverged on that update by six RNG calls. Merely
matching the replay's final score had concealed the incorrect trajectory.

The promoted table is now
[`metsu-level-parameters-pc64.tsv`](metsu-level-parameters-pc64.tsv), SHA-256
`d28a00cb359853665dda5a2faf0ad5c16b5b4fbe6617ebfcb1475ab38faba4f5`.
It runs the same original setter, sine, power, and float-store code under the
observed `0x137f`; its harness still calibrates normal's 99 rows separately at
`0x027f`. Compared with the historical PC53 Metsu table, 25 EX cadence rows and
33 score-scale bit patterns differ, while descent bits do not.
Reproduce with `tools/metsu/dump_level_parameters.py --control-word 0x137f`;
the receipt is `benchmarks/results/metsu-level-parameters-pc64-2026-11-03.json`.
Metsu rules revision 2 selects these stores and has configuration hash
`1958987580749614460`. Normal's tables and the physics host's `0x027f` profile
remain unchanged. Parameter-component evidence alone does not prove that
choice: the actual per-update comparisons below test its observable results.

```sh
python3 tools/metsu/dump_level_parameters.py \
  reference/game/irisu-v2.03-jp/irisu203/irisu.exe \
  --control-word 0x137f --table reference/metsu-level-parameters-pc64.tsv \
  --report benchmarks/results/metsu-level-parameters-pc64-2026-11-03.json
```

The isolated original contact and actor timer routines also execute natively:
`tools/metsu/dump_original_contacts.py` builds `original_contacts_probe.c`,
authenticates the executable hash, and produces
[`metsu-contacts-original-components.json`](metsu-contacts-original-components.json).
It contains21 named scenarios and a complete stub inventory. These exercise
EX countdown120→112, propagation's112/113 boundary, countdown expiry, mixed
group reset, bottom/top gates, activation predicates, rot's4/5-update boundary,
EX rot non-advancement, scripted-pair dispatch, mature EX burst/reward,
direct-hit projectile retention, and Metsu orb no-reward behavior. Original
normal color-clear and fresh ordinary grouping provide positive controls.

`tests/native/test_metsu_components.cpp` compares those same21 named scenarios
against the implemented simulator over67 measured fields. Metsu cases call its
real complete contact dispatcher or actor update; unrelated dispatcher effects
are excluded from each direct-handler measurement projection. The one normal
color-clear control uses the existing normal simulator through public snapshot
and step APIs with reward120, allowing the same measured+240 gain for a live
and a retained dead color slot. No normal implementation/header was changed.

Component-probe exclusions are material: native physics, base actor integration,
full actor post-contact cleanup, rendering, particles, and audio are stubbed or
omitted. The original clear-pending routine and shared reward hook execute.
This establishes direct original-code component evidence and clone agreement
on the listed projections. Complete recorded-run comparisons are a separate
body of evidence below; neither establishes unrestricted trajectory parity.

Dispatcher orientation is verified at `0x40ab3c–0x40ab48`: native contact(A,B)
is pushed as arguments(B,A) to the pair helper. The pair helper itself reverses
again for its first directional grouping call. Therefore the implementation's
first directional target is nativeA, source nativeB; its second target is
nativeB. The burst wrapper has the same net target order. Do not reverse the
implementation after reading only the bare helper argument order.

## Actual original-process replay qualification (2026-11-03)

The later full-game audit found that the original process uses x87 CW `0x137f`
in the recovered runtime. At level1 its genuine EX cadence is175, whereas the
older PC53-only component dump produced176. Sixty ordinary title render passes,
Wine11.13/11.21 and classic/WoW64 checks, and eager authentic DLL loading before
D startup retain the observed CW/cadence. Metsu's parameter table was therefore
recovered under its measured original-process control word, separately from
normal mode. `metsu-level-parameters-pc64.tsv` records that component evidence;
normal's implementation and qualification remain separate.

Actual original-process instrumentation, including owning addresses and
transparent hooks, is documented in `tools/metsu/ORIGINAL_STATE_TRACE.md`.
The observer executes the complete original game with the shipped authentic
physics DLL, reads every post-actor boundary, and captures canonical ending
metadata at entry to the authentic finish routine. It does not substitute the
component harness for the original process.

A deterministic active-input native-policy replay (seed1,4652 input words)
reaches level4,18 qualifying clears, score426, and chain3, with weak and strong
shots held for3 updates and EX-chain interactions. Its complete replay passes
strict original-process comparison:4653 frame boundaries,271122 actor states,
all 28 measured actor fields, scalar/RNG/pool state, the live group-sharing
partition at every boundary, and 6 canonical finish fields.
Both processes naturally finish on the same input; the actual original first
finish tuple426/4/3 matches the saved header. The original receipt and strict
comparison are linked in the instrumentation document. This controlled replay
is distinct from the independently captured browser-export and genuine
original-recorder interoperability routes.

The initial bounded-audit/component-only claim limits earlier in this ledger
are superseded **only** for these explicitly executed, hash-bound full-game
captures. Unbounded and all-path Metsu fidelity is not claimed. In particular,
the active-input completed run does not reach the special-orb scheduler.

The actual original recorder also created the 2,994-input fixture
`tests/fixtures/metsu/original-generated.rpy`. Both its live recording and its
subsequent original-engine self-playback match all 2,995 boundaries and 41,999
actor states with the native engine, including raw float bits and normalized
group sharing. The latter run captures the actual finish entry and verifies
canonical 0 / 1 / 0 against the file header. Reports:
`benchmarks/results/metsu-original-writer-native-2026-11-03.json` and
`benchmarks/results/metsu-original-writer-playback-native-2026-11-03.json`.

This is recorded-input observational qualification, not proof of all-path
equivalence. A replay that merely exhausts its inputs while both games remain
active cannot pass saved-replay interchange: the comparator separately requires
natural termination and matching canonical first-finish headers. The initial
premature browser export is retained as that negative case in
`benchmarks/results/metsu-premature-browser-replay-rejected-2026-11-03.json`.
The browser-action routes and final report are collected in
`benchmarks/results/metsu-replay-interoperability-2026-11-03.json`.

The corrected actual browser Save file
`tests/fixtures/metsu/browser-export.rpy` (seed 1274909209, 3,942 inputs) is the
positive browser-origin case. The actual original process and native engine
match all 3,943 boundaries and 57,778 actor states, including every live group
partition and canonical finish fields. Both naturally finish at input 3,942
with canonical 40 / 1 / 2. The hash-bound report is
`benchmarks/results/metsu-browser-export-original-2026-11-03.json`. These
additional observations do not enlarge the claim to arbitrary inputs or
unobserved levels.
