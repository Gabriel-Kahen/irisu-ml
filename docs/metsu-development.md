# Metsu development checkpoint

This checkpoint establishes an isolated development workspace and an audit
plan. **Metsu gameplay is not implemented or qualified here.** Normal mode
remains the default; no website deployment is part of this checkpoint.

Development branch: `codex/metsu-mode`. The working checkout is
`/home/gabe/.codex/worktrees/irisu-metsu/irisu`, created from `13023da`.
The existing `codex/v100-exact-backend` checkout and its uncommitted GPU work
are separate and must not be changed by Metsu development.

## How normal mode was developed

`clone.md` preserves the initial research plan and subsequent implementation
history. Its early unknowns are superseded by `reference/game-rules-analysis.md`
and `docs/mechanics.md`. The development process was:

1. Recover the v2.03 executable's rules, parameter selection, replay format,
   exact random generator, input edges, and update/contact order.
2. Probe the original Box2D DLL and instrument original-game replay playback
   to obtain numerical and gameplay-event oracles.
3. Implement the deterministic C++ simulator and compare its behavior with
   those measurements, including causes of discrepancies rather than tuning
   scores to match replay headers.
4. Address long-horizon numerical drift with the separate 32-bit MSVC9 r58
   exact physics host and isolated worker processes.
5. Run that exact worker in the browser under v86, using a pinned static
   runtime and original-format replays.

`docs/fidelity.md` records both the successful replay comparisons and the
remaining limits of the historical evidence. The current checkout does not
contain the old optional original-process trace bundles. The recovered CPU
runtime and published final replay provide a current normal-mode oracle;
`docs/exact-runtime-recovery.md` describes its provenance and verification.
Passing portable tests does not establish trajectory-exact original-game
behavior, and historical checks must not be reported as newly rerun.

## Separation contract

- Keep `irisu::Simulator`, normal configuration defaults, its configuration
  hash, and mode-0 replay behavior unchanged. Normal continues to work without
  selecting or building Metsu.
- Implement gameplay in a separate `irisu::metsu::Simulator`, with its own
  source/header, mode-specific configuration factory, and validation. Reuse
  the existing physics, RNG, numeric helpers, and suitable common value types.
- Use a distinct `irisu_metsu_core` target, `irisu-metsu-headless` executable,
  and `irisu-metsu-exact-worker`. Link shared low-level functionality from
  `irisu_core`; the normal simulator implementation need not be linked into
  a Metsu executable. Put these targets behind a separate build option that
  defaults to off, so normal's existing build remains sufficient.
- If the existing IPC worker implementation is reused, select its simulator
  and initial configuration at build time. Keep the normal branch's wire
  behavior intact; do not add a runtime mode switch to the normal worker.
- Give Metsu snapshots their own magic/schema and include a mode discriminator
  in their configuration identity. Reject cross-mode restores, even when
  numeric configuration values happen to match.
- Give Metsu a separate `/metsu/` web entry, worker/runtime directory, cache
  namespace, replay mode checks, and runtime manifest. Preserve the normal
  entry and its pinned runtime. Never replace the normal runtime with a Metsu
  worker or relax its identity checks to accept either mode.
- Run the existing normal tests and compare normal exact-worker responses
  against the pinned released worker before integration. Keep new Metsu tests
  and fidelity claims identifiable separately.

This deliberately favors a clear gameplay boundary while sharing the already
qualified low-level engine. Broader refactoring can wait until both modes have
independent behavioral evidence.

## Evidence that must guide implementation

The recovered Metsu-side parameter table is in
`reference/game-rules-analysis.md` under “Mode-selected parameter profile.”
It is a starting point, not proof that changing configuration reproduces Metsu.
Initialization, spawning, level handling, contact rules, rewards, and terminal
behavior still require mode-specific examination and original-game comparison.

The original-executable audit distinguishes two different selectors:
the **scene mode** chooses normal versus Metsu, while **`Block+0xf0` marks an
EX actor**. Ordinary actors can retain `f0 == 0` in Metsu. Do not set `f0 = 1`
on every Metsu body or interpret every `f0` condition as a scene-mode check.
See `reference/metsu-rules-analysis.md` for the evolving binary evidence and
its unresolved points.

The common `Body` type already preserves `rule_guard_f0` and
`physics_update_count` (`f8`). Whether additional state is required depends
on the recovered EX scheduler and lifecycle. Normal snapshot validation
explicitly restricts `f0`, so Metsu must own its own validation rather than
loosening normal's validator.

## Exact browser packaging

The normal `prepare-exact-runtime.sh` intentionally pins the worker, object,
core archive, host library, and staged worker hashes. Keep those checks intact.
A Metsu build needs a separate packaging path:

1. Verify and copy the pinned baseline runtime into a new build directory.
2. Build the Metsu worker object and gameplay archive against the same exact
   physics host, retaining normal's common low-level archive separately.
3. Relink the Metsu worker against the guest toolchain and guest C++ runtime.
   `apps/web/relink-exact-worker.sh` documents the required interpreter,
   dependency set, and `$ORIGIN` search path; a Metsu-specific equivalent can
   accept both gameplay and common archives.
4. Place the result at `guest/irisu-exact-worker` inside the separate Metsu
   runtime. The existing guest init launches that filename, so a new kernel
   is unnecessary solely to select the new executable.
5. Generate a distinct manifest and checksums, then stage that runtime with
   the Metsu client under `/metsu/`.

A newly linked host executable must not simply be copied into the old guest.
For example, the current native recovered build requires `GLIBC_ABI_GNU_TLS`,
which must not be assumed available in the older guest ABI. The guest relink and an actual
browser boot/parity check remain required before calling Metsu web-ready.

## Next work

Finish the original-executable audit, retain reproducible Metsu reference
traces, implement the separate engine, then add the client. A runnable mode
does not by itself establish original-game fidelity. Record measured results
and unresolved differences separately from normal's existing evidence.
