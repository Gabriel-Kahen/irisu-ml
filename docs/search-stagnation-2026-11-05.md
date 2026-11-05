# High-score search stagnation investigation — 2026-11-05

The running search has allocation and archive problems that make its lack of
improvement unsurprising. The evidence does not establish a new physics defect
or imply that a faster simulator should automatically beat the incumbent.

## Scope and snapshot

Inspected the **Run high score on V100** chat, the frozen planner sources, and a
read-only SQLite transaction at **2026-11-05 10:48:35 EST (15:48:35 UTC)**.
Campaign: `artifacts/high-score-gpu-parity-20261105-082726/campaign`.
Its frozen source commit is `a1694b79a0c5e9e89220c6954c2e26281b50df68`.
The running process, frozen runtime, archive and verifier were left intact;
this investigation launched no competing GPU work.

At that snapshot:

| Observation | Value |
| --- | ---: |
| Completed / running jobs | 259 / 1 |
| Settled simulated ticks | 54,958,287 |
| Settled job wall time | 7,961.670 seconds |
| Control-seed jobs / all jobs, including running | 190 / 260 |
| Seeds with status `discovery` | 129 / 129 |
| Maximum completed bootstrap attempts per seed | 1 |
| Imported live replay roots inactive | 22 / 24 |
| Probe jobs | 49, all with horizon 2,048 |
| Generated control-seed terminal checkpoints | 40,817 |
| Best generated control canonical terminal score | 2,389,932 |
| Highest chain in those generated terminals | 53 |
| Imported verified incumbent | 4,393,101, chain 70 |

The 40,817 checkpoints are stored terminal recipes, not a claim of 40,817
independent successful strategies. The raw terminal score 4,393,699 belongs to
the imported replay after its first recorded terminal snapshot; its canonical
score remains 4,393,101. There is no hidden new record in that number.

## Confirmed allocation and archive defects

The frozen `scheduler.rebalance()` intentionally requires both 30 seconds of
search and three bootstrap attempts before a seed can be promoted. Bootstrap scheduling first
chooses the smallest attempt count across all 129 seeds. Thus every seed must
receive its first attempt before any receives its second. Once a newly explored
seed has spent 30 seconds, it leaves the `under` allocation pool, but remains
ineligible for promotion. The control seed is the only fallback candidate because
it has the `control` flag. This control-only fallback, rather than the deliberate
three-attempt tournament gate itself, is the scheduling defect. Wildcard allocation provides occasional relief, but
does not correct that fallback rule. The observed all-`discovery` state and
190/260 control jobs demonstrate this effect in the actual campaign.

The broad initial tournament takes substantial time: 129 seeds require 387
bootstrap attempts before all can pass its attempt gate. While seeds remain in
discovery, they cannot receive the probe lane's 8,192-tick finalist horizon.
All 49 observed probes used 2,048 ticks. Bootstrap
jobs in an earlier nearby snapshot completed only one or two batches, averaging
1.25, so most fresh seeds received little depth before that allocation gap.

The frozen `store.prune_archive()` ranks all states, including terminals, for
the active archive. It protects the champion and one control live state, but
does not reserve imported replay coverage. Terminal evidence consumes active
slots despite having no continuation, and 22 of the 24 live incumbent roots
became inactive. Their immutable evidence remains on disk, but normal scheduling
does not consider inactive roots. In particular, the earlier roots needed to
change the record's large chain were lost from the search surface.
A separate nearby snapshot had 504 terminal states among its 1,024 active
checkpoints, despite terminal states offering no continuation.

## Search depth and proposal limits

At imported tick 86,016, the record has score 256,584 and a live chain of 17.
It reaches its canonical record around tick 97,219 with chain 70. A promising
intervention there can need roughly 11,000 subsequent ticks to reveal its value.
The current 384-tick expansion horizon and 2,048-tick probes expose only part of
that delayed payoff.

`Coordinator.execute()` advances only the first live survivor within each job.
The other seven configured survivors are archived for later scheduling; this is
not an eight-branch beam expanding all eight branches at every depth. Selection
normally puts current live-group potential first. This is a limitation of the
search strategy, distinct from the confirmed allocation defects.

`proposals.generate()` preserves the exact original replay suffix as one
baseline candidate when the prefix still matches. Intervention candidates make
a short shot/wait prefix and then use the inexpensive reactive controller.
They do not make a small replay mutation and resume the original timed suffix.
Consequently, a local change also discards the remainder of a known successful
strategy. Adding suffix-preserving mutations is a useful next experiment, not a
proven cure for stagnation.

Automatic neural training creates shadow models. It does not activate them;
there was no `active_model` entry at the snapshot. Training more examples in
this run therefore does not mean the live search has learned a better policy.

## Throughput interpretation

The approximately 20× simulator benchmark is not an end-to-end measurement of
this resident orchestrator. That benchmark used 12,288 distinct four-tick futures
at prefix 100, whereas this campaign uses 128 reactive candidates with horizons
384–2,048, including dense late-game states. Its workload includes closed-loop policy evaluation,
checkpoint reconstruction, full input recipes, JSON transport, SQLite publication,
archive pruning and independent verification. The benchmark and the campaign
must be compared using matching workloads before assigning an end-to-end speedup.
At the fixed snapshot, settled work averaged approximately 6,903 simulated ticks
per charged job second. This is a campaign throughput observation, not a matched
CPU/GPU speedup measurement.
A nearby 255-completed-job snapshot similarly measured approximately 6,864
ticks per charged job second.

There was also avoidable terminal-ranking overhead. The frozen coordinator
called `store.get_checkpoint()` on every terminal sibling solely to compare
scores, reading, checksumming and decoding its full input recipe. A nearby
snapshot contained 40,428 generated control terminal recipes with approximately
11.6 GB of full input traces in total. Ranking can use the scores already
returned by the worker and load only the winning recipe for verification.
A CPU-only comparison on 128 existing late terminal recipes selected the same
winner and reduced median ranking time from 0.629 seconds to approximately
7 microseconds. This measures ranking alone, not whole-job acceleration.

The immediate fixes are to keep discovery seeds in the live development
pool while their independent bootstrap attempts remain scheduled, reserve active
replay-root coverage, and keep redundant terminals as cold evidence. None of
these changes alone establishes a higher score. A new frozen campaign or a
bounded comparison is needed to measure their gameplay effect; the existing
campaign's provenance must not be changed in place.

## Validation of working-tree fixes

Independent review found no correctness regression in the discovery allocation,
replay-root retention, cold terminal evidence or terminal-ranking changes.
The focused coordinator/store/scheduler checks passed independently (45 tests).
The full `tests/test_search*.py` suite passed with 161 tests and three skipped;
`git diff --check` was clean. These CPU-only checks validate the changed search
contracts and ranking equivalence. They do not establish higher gameplay scores
or qualify a replacement GPU campaign.

## Local fixes and validation

The working branch reserves imported replay coverage within archive quotas,
reactivates those roots when space returns, leaves only each seed's best terminal
active, includes discovery seeds in live development, and ranks terminal results
without rereading their recipes. All immutable terminal evidence remains stored.
The three-attempt tournament gate and proposal strategy are unchanged.

Regression tests cover discovery allocation, replay coverage under pruning,
tight quotas and recovery, cold terminal evidence, and terminal ranking reads.
The full search suite passed: **161 passed, 3 GPU-specific checks skipped** using
`PYTHONPATH=python /tmp/irisu-audit-tests/bin/python -m pytest -q tests/test_search*.py`.
An independent code review found no correctness regression. No GPU comparison
or higher score has been established for these local changes.

## Reproduce the main observations read-only

Run from the repository root. These queries return aggregates rather than large
checkpoint summaries; totals will advance while the campaign runs.

```python
import sqlite3

path = "artifacts/high-score-gpu-parity-20261105-082726/campaign/search.sqlite3"
db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
db.execute("BEGIN")  # one consistent read snapshot
queries = {
    "jobs": "SELECT kind,status,count(*),sum(simulated_ticks) FROM jobs GROUP BY kind,status",
    "seed_status": "SELECT status,count(*),max(attempts) FROM seeds GROUP BY status",
    "control_share": "SELECT seed=1298144938,count(*) FROM jobs GROUP BY 1",
    "imported_live_roots": """
        SELECT active,count(*) FROM checkpoints
        WHERE operator='import' AND NOT json_extract(summary,'$.terminated')
        GROUP BY active""",
    "probe_horizons": """
        SELECT json_extract(payload,'$.horizon'),count(*)
        FROM jobs WHERE kind='probe' GROUP BY 1""",
    "generated_control_terminals": """
        SELECT count(*),max(json_extract(summary,'$.canonical_score')),
               max(json_extract(summary,'$.highest_chain'))
        FROM checkpoints WHERE seed=1298144938 AND operator!='import'
        AND json_extract(summary,'$.terminated')""",
    "active_guide": "SELECT value FROM meta WHERE key='active_model'",
}
for label, query in queries.items():
    print(label, db.execute(query).fetchall())
db.close()
```

Owning frozen sources:
`runtime/python/irisu_search/{scheduler,store,coordinator,proposals}.py` beneath
the campaign's parent directory. The working-tree versions may contain fixes
that the preserved running campaign does not use.
