# Metsu replay regression inputs

These are v2.03 mode-1 replay headers and packed input words, without game
executables, assets, or saved-game data.

| File | Origin | Inputs | Canonical score / level / chain |
|---|---|---:|---|
| `browser-export.rpy` | Downloaded through the corrected browser client's Save button, then loaded in the actual original game | 3,942 | 40 / 1 / 2 |
| `original-generated.rpy` | Written by the actual original game's recorder in a disposable live run | 2,994 | 0 / 1 / 0 |
| `controlled-level4.rpy` | Deterministic native coverage policy, then loaded and verified in the actual original game | 4,652 | 426 / 4 / 3 |

The controlled policy alternates weak and strong shots, holds each for three
updates, and reaches EX chains and three level transitions. Reproduce its
exact bytes with the exact-backend target:

```sh
build-metsu-exact/irisu-metsu-generate-controlled 1 OUTPUT.rpy
```

The original-written file has SHA-256
`c445ae72494ac25599b2aea5629772548b0a6daba8c357ccddf69f75c2ac38cc`;
the controlled file has SHA-256
`c76a26c0f1b5a8526efaf53f41d704e5275c608077d07cb1d6ca4d782d632fa8`.
The browser export has SHA-256
`18d630363610072e123182421353db9ef79a2df2fee4f9473c72d4499a545e9e`.

See [the comparison workflow](../../../tools/metsu/REPLAY_INTERCHANGE.md)
and [the qualification report](../../../benchmarks/results/metsu-replay-interoperability-2026-11-03.json)
for full state coverage, provenance, browser actions, and limits. A finite set
of matching replays is not proof that every possible run matches.

Qualify a newly compiled exact tracer against the recorded original states:

```sh
python3 tools/metsu/check_replay_fixtures.py \
  --tracer build-metsu-exact/irisu-metsu-trace \
  --output .tmp/metsu-original-fixture-check.json
```

The read-only check uses `original-state-digests.json`, derived exclusively from
the three complete original-process traces pinned there by SHA-256, along with
their capture receipts, comparison reports and replay inputs. It checks every
reset/update row: eight scalar fields (including all 200 pool colors), 28 actor
fields with exact float32 bits, six finish fields and normalized live group
sharing. It also requires complete input coverage, a natural finish and the
canonical replay header. Original executables and raw traces are not needed
to run the gate. This guards the recorded coverage when changing compilers;
it does not extend that coverage or repeat an original-game execution.
