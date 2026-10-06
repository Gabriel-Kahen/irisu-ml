# Exact end-to-end training policy

Target-policy training uses the pinned MSVC9/Box2D exact worker for every
state-producing stage. Portable GNU physics is a throughput diagnostic only.

## Promotion contract

A checkpoint or policy result is promotable only when all of the following are
true:

- demonstration, rollout, branch-label, calibration, model-selection, test,
  full-game, and replay-verification environments declare
  `physics_backend = "exact"`;
- the worker and mapped exact-library hashes match
  `configs/rl/runtime/exact-worker-2026-07-21.json`;
- runtime call-target and mapped-library attestations pass before collection;
- the artifact stores the exact runtime provenance for every environment
  family; and
- replay score, level, chain, terminal tick, and action count come from exact
  re-execution rather than replay-header metadata.

Portable runs must opt in through an explicitly diagnostic CLI flag. Their
artifacts are non-promotable even when they are deterministic on the same GNU
build.

Promotion replay checks are also fail-closed:

```bash
PYTHONPATH=python python tools/evaluate-rpy.py candidate.rpy \
  --layout padded \
  --worker /absolute/path/to/the/pinned/irisu-exact-worker \
  --purpose promotion
```

This mode rejects an unpinned worker, header/result disagreement, truncation,
nonterminal or post-terminal traces, invalid actions, cadence disagreement, or
an action encoding that omits a simultaneous shot edge.

`configs/rl/portable-diagnostics-v1.toml` is the exhaustive exception registry.
It classifies frozen portable entrypoints and backend-compatibility APIs as
legacy diagnostics, separately lists display/check utilities, and records the
explicit portable comparison paths retained by exact-primary entrypoints. A
regression test rejects any unclassified portable Python source or experiment
config. Legacy configs repeat the non-promotable policy in their own metadata.

Names are not evidence of backend fidelity. In particular, the historical
`r3h_exact_collect.py` and `r3h_g2_exact_collect.py` entrypoints are portable
development collectors and remain non-promotable until replaced by a fresh
exact lineage.

## Runtime entry points

`irisu_rl.exact_training_runtime.ExactTrainingRuntime` is the production
factory. It requires an explicit absolute, regular, nonsymlink worker path,
checks its bytes against the pinned identity, constructs exact environments,
attests every live lane and mapped legacy library, and withholds the environment
if any check fails.

Shared pointer teachers use `TransactionalBranches`. Exact source-only branches
use fork/COW checkpoints; portable diagnostics use clone/restore. Durable exact
snapshots remain available for restart and nested controller-coupled searches,
but restoring them replays the action history and is not the default branch
mechanism.

## Artifact migration

Existing R3d frozen-v5 and downstream R3e–R3n artifacts remain immutable
portable-development history. They may initialize weights, but they cannot be
called exact-trained or promoted.

`configs/rl/portable-diagnostics-v1.toml` is the exhaustive exception registry
for those legacy runners and intentional comparison tools. The regression test
fails when a new explicit or implicit portable Python environment or portable
RL config appears without classification. Some historical filenames contain
`exact`; that described their branch-label algorithm, not their physics backend.

Regenerate the authoritative lineage in this order:

1. exact demonstrations, archive branches, and R3d steering checkpoint;
2. exact on-policy/DAgger branch labels and proposal models;
3. exact controller and restraint screens;
4. exact full games; and
5. exact replay re-execution and winner selection.

Changing only the final evaluator is insufficient: the policy must observe and
act on exact trajectories during collection and training.

## Fidelity boundary

The exact worker is the highest-fidelity recreation and matches the available
original-game replay/physics corpus, including the 47,019-update trace. This is
strong tested coverage, not proof that every possible original-game behavior
has already been measured. Original-executable transfer remains a separate
gate.
