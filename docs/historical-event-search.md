# October event-search fallback

The fallback restores the actual record-producing October event beam, whose
runner SHA-256 was `a800d330a3e53c37eff711d60bcf406fe644f956ac3022d0258d05344e90b875`.
All seven recovered files match the hashes in the historical run plans.
[Original snapshots and recovery evidence](../tools/historical/README.md) are
retained alongside the portable planner package.

The unchanged planner functions generate carrier protection, retirement,
feeder preparation and shared-payout options; execute paired guards; maintain
event histories; value live assets; and select diverse parents. The winning
configuration uses 24 parents, 48 options, 32-tick segments and a 3,072-tick
horizon. It bypasses the November scheduler and proposal generation.

The new `irisu_search.legacy` driver supplies checkpoint ownership, batched
simulation, persistence and independent record verification. It repeats searches
on the verified champion seed, starting with the historically winning tick
96,000 and then nearby late-game roots. A 20-minute search slice stops between
complete stages and resumes the same frontier in the next slice until that
root finishes. A process restart restores the saved beam and RNG state. This
wrapper replaces the missing historical campaign installation; the October
NRPA, training and MPC finishing paths are not enabled.

The simulator transport now exposes actual lifecycle, geometry and age fields,
plus every ordered game event. `batch_events` retains events from one parent;
`multi_batch_events` evaluates independent parents and durations together.
Both accept horizons up to 512 ticks and fail closed on event-buffer overflow.
Ordinary long replay batches preserve their existing behavior. Options keep
their original decisions while up to 64 independent executors share GPU
launches. Proposal RNG and final tie order remain sequential.

Prefix genealogy comes from independent native replay, cut off at the root
tick. Its replay and trace hashes are checked before reuse. Selected prefixes
are reconstructed from exact inputs and compared against complete observations.
New records require natural termination, canonical scoring and an independent
native replay receipt. The current verified incumbent is retained.

Copy [the configuration](../configs/search/high-score-historical-event.json)
into a frozen runtime containing the newly built CUDA worker and the qualified
native oracle, library and trace executable. Paths in the example are relative
to that runtime's campaign directory. Run from that directory:

```sh
PYTHONPATH=runtime/python python -m irisu_search.legacy \
  --root . --config launch-config.json --incumbent incumbent.rpy
```

`progress-source.json` reports verified score, completed roots (`jobs_done`),
search slices, completed stages, branches, simulated ticks, mixed GPU batches,
rejections and verifier failures. `jobs/*/frontier.json` contains the durable
beam; completed candidates are logged in deterministic planner order. A bounded
qualification run can use `--max-jobs 1`, which executes one search slice.

Historical regressions and AST comparisons check the recovered decisions.
Native/device tests compare ordered events, full observations, mixed seeds,
durations, parent immutability and slot reuse. These are compatibility checks;
they do not establish a new score or a 20-fold end-to-end campaign speedup.
