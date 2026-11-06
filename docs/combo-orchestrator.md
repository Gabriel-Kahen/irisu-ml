# Combo-focused search

The record's dominant group contributes 4,155,740 of 4,393,101 points (94.60%).
It starts at tick 81,803 and reaches chain 70 / num 54 while repeatedly recruiting,
paying out members and preserving a carrier. Group counters survive member
removal: chain counts recruited members; num counts qualifying events, not paid
pieces. The exact per-actor score is truncated after multiplying
`(num + 1) / 2 * chain` by the native level and size factors. Two existing groups
cannot be merged. These facts come from `clone/core/normal_rules.cpp`,
`clone/core/simulator.cpp` and the independently reconciled record genealogy.

This makes carrier continuity and eventual payout the useful search targets.
A sum of small group potentials and early fresh-seed score are inadequate.

## Research basis and scope

[Go-Explore](https://arxiv.org/abs/2004.12919) separates remembering and returning
to promising states from exploring their continuations. Exact input recipes
provide our return mechanism; explicit root and frontier scheduling supplies the
previously missing follow-through.
[Single-player MCTS](https://cris.maastrichtuniversity.nl/en/publications/single-player-monte-carlo-tree-search/)
studies terminal playout evaluation when simple state heuristics are unreliable.
[Rolling Horizon Evolution Enhancements](https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf)
studies persistent populations and rollout tails under bounded budgets.
These results motivate the design; they do not establish Irisu performance or
justify claiming this implementation is a full MCTS/NRPA algorithm.

## Implemented policy

`configs/search/high-score-combo-v2.json` opts into combo search. Legacy configs
retain their prior scheduling semantics. All-in wall-time lane targets are 60%
record-combo development, 20% shortlisted challenger-combo development, 15%
cohort seed discovery and 5% wildcard exploration. The record seed follows the
verified champion rather than a hardcoded seed. A new global winner reconstructs later roots on
its changed recipe, and interrupted reconstruction resumes under the campaign
lock; old input recipes remain immutable. Cohorts finish their configured
attempts before a new cohort receives discovery work.

A replay-bound native genealogy selects up to 64 roots before recruitment,
payout and single-carrier risk events. Selection covers stages of the dominant
scoring combo, rather than merely retaining 24 coarse checkpoints. Every root
receives a first visit before least-used revisits alternate with frontier work.

Record jobs evaluate a 128-wide neighborhood of shot position, strength, timing,
suppression and carrier handoff changes. Original future inputs keep their
absolute ticks. The horizon covers the known remaining replay when possible,
plus available tail up to the configured 32768-tick bound. Natural terminal
scores are compared against the unchanged reference executed from the same
parent. Capped futures stay censored. Novel best terminals are independently
verified even below the global record; unchanged references and baseline ties
are reported separately. Attempted candidate signatures prevent repeatedly
charging identical deterministic futures across jobs, apart from the retained
paired reference.

Live development retains four exact alternative recipes by dominant combo,
realized score, carrier safety and geometry diversity. These alternatives are
persisted, protected while shortlisted, and expanded by least use across and
within jobs. Retained siblings receive a turn before replacement. Full terminal plans are also retained as bounded per-root elites, including
planned inputs after an early failure. Later jobs can compound a second edit
without requiring the first edit to beat the record. Their rollout scores are
exploration estimates until independently verified, and are never records merely
because they entered the plan archive. Root coverage interleaves immediate plan
and frontier follow-up rather than waiting for a complete root sweep. A frontier is
bounded per seed; only the record and shortlisted challengers reserve active
archive space. Other exact recipes stay recoverable in cold storage. This is a
persistent bounded frontier using the existing single-parent GPU worker, not a
claim of simultaneous multi-parent GPU execution.

Engine group deltas credit extension and qualification even when the group
fully pays out and disappears. The heuristic never becomes an exported score.
Only an independently verified first-recorded canonical terminal can be a record.
Learned allocation stays inactive in the sample config; its old per-seed gain
head does not control the new lane allocator.

## Run

Build the resident worker and independent verifier as described in
`high-score-orchestrator.md`. Keep the qualified pristine-oracle initialization
fix. Generate replay-bound events with the exact native trace binary:

```sh
PYTHONPATH=python python tools/orchestrator/trace_combo.py \
  irisu-high-score-4393101.rpy /tmp/record-combo-events.jsonl \
  --trace ./build-project-exact/irisu-search-trace \
  --oracle-library ./.local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so
PYTHONPATH=python python -m irisu_search init artifacts/combo-search-v2 \
  --config configs/search/high-score-combo-v2.json \
  --incumbent irisu-high-score-4393101.rpy \
  --combo-events /tmp/record-combo-events.jsonl.receipt.json
PYTHONPATH=python python -m irisu_search run artifacts/combo-search-v2 --seconds 3600
PYTHONPATH=python python -m irisu_search status artifacts/combo-search-v2
```

The trace and receipt
are integrity-bound to the replay, seed and canonical score. Initialization
rejects a mismatched receipt. Source hashes include the combo and diagnostics
modules. Upgrade by freezing a successor campaign; never rewrite old provenance.

## Search-quality qualification

`tools/orchestrator/qualify_combo.py` constructs damaged references from real
verified replays and compares the old and new candidate generators with bounded
candidates and all-in wall budgets. The common prefix is frozen before the
changed shot. Only the damaged suffix is given to either planner. The healthy
original remains outside both candidate generators as an independently verified
reachable witness. Unchanged damaged-baseline execution cannot count as recovery.
Every claimed novel terminal gain is replayed by the fixed native oracle.

Keep fixture discovery outcomes, censored futures, distinct executed recipes,
verified gains, wall costs and search RNG identities in the receipt. Recovery of
a damaged 4,393,101-point replay is a positive control, not a new high score.
Use multiple damages and RNG seeds before generalizing from one recovery case.
GPU throughput and unrestricted campaign improvement remain separate checks.

The status report exposes root scheduling coverage, frontier states/expansions,
novel terminal results, baseline ties, positive paired gains and attempted
candidate signatures. Simulated ticks alone are not an optimizer-quality metric.

## Initial measured qualification

The frozen three-case corpus contains horizontal, vertical and timing damage
to late combo shots, with independently verified healthy witnesses kept outside
the planners. Across two search RNG seeds, the final symmetric neighborhood
recovered all lost score in 6/6 comparisons. The previous generator recovered
0–0.685% of the lost score. Each arm received 32 candidate attempts and a 120-second
wall cap; actual costs were approximately 3–7 seconds including native verification.
These are bounded regression controls, not an equal-full-wall-time production
comparison or blind generalization result. Early catalog failures on vertical
and timing cases were retained; they motivated symmetric neighborhood coverage.
All successful repairs tied 4,393,101 and did not establish a new record. The
receipts and limitations are retained under
`artifacts/combo-qualification-20261105/summary.json`.
