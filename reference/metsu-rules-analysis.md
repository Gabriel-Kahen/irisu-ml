# Metsu mode: bounded original-binary audit

Audit date: 2026-11-03. This is a context and implementation-planning checkpoint,
not an implemented mode or a runtime-fidelity claim. No simulator source was
changed during this audit. Addresses below refer to the original v2.03 engine.

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
executable, and original assets remain ignored. This checkpoint used static
`objdump -D -Mintel` inspection; it did not run the original game. Local full
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

## Next evidence and claim boundaries

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

No complete original-game mode-1 replay parity, long-run trajectory fidelity,
EX sampling distribution, browser behavior, or Metsu simulator is claimed here.
The pinned original is now available locally for the next audit phase.
