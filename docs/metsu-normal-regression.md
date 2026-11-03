# Normal-mode protection during Metsu development

This is setup and regression evidence, not a Metsu implementation or fidelity
claim. Metsu work starts on `codex/metsu-mode` in a separate worktree; the
existing normal/GPU checkout and its uncommitted work remain untouched.

## Protected baseline

Commit `13023da` was copied with `git archive HEAD` into the ignored
`.tmp/normal-baseline-source` directory before any simulator changes. Its
portable and exact builds are independent of subsequent edits. The checked-in
[baseline receipt](../benchmarks/results/metsu-normal-baseline-2026-11-03.json)
records the full source commit, selected source hashes, runtime identities,
test counts, and byte differential.

Baseline checks passed on 2026-11-03:

- Portable native tests: 8/8.
- Exact native/worker tests: 12/12.
- Browser tests: 70/70.
- Published normal replay: all 97,219 steps and 188,356,329 response bytes
  matched the pinned released worker, including observations, events,
  diagnostics and terminal metadata. Canonical result: 4,393,101 points,
  level 100, chain 70.

The old four original-game trace bundles are absent. Their optional CTests
were skipped at configuration; these results do not claim those oracles were
rerun. The browser suite's ignored seed-41 input fixture was regenerated with
the existing preset and verified against its recorded SHA-256.

## Guard a candidate normal worker

Build the candidate exact worker using the instructions in
[exact-runtime-recovery.md](exact-runtime-recovery.md), then run:

```sh
python3 tools/metsu/check_normal_regression.py \
  --baseline /path/to/published-runtime/guest/irisu-exact-worker \
  --worker build-exact/irisu-exact-worker \
  --output artifacts/metsu-normal-regression.json
ctest --test-dir build --output-on-failure -j 2
ctest --test-dir build-exact --output-on-failure -j 2
node --test apps/web/tests/*.test.mjs
```

The guard pins the released worker hash, exact physics host hash, normal
configuration hash (`0xec0e8463feaf2670`), and published replay hash before
calling the existing full-response comparator. Comparing two modified workers
cannot substitute for the released baseline. This is a bounded normal
regression gate; it does not establish Metsu fidelity or unrestricted normal
behavior.

If the browser fixture is missing, regenerate its original bytes before the
browser tests:

```sh
mkdir -p reference/captures/seed41-score-parity-20260720-001
python3 tools/generate-controlled-rpy.py \
  reference/captures/seed41-score-parity-20260720-001/input.rpy \
  --preset score-seed41-parity --library build/libirisu_clone.so
```

Keep normal as the default mode and keep its existing replay/configuration
identity. Metsu must opt into a separate mode/profile and its own evidence;
sharing low-level physics does not authorize changing normal's rules.
