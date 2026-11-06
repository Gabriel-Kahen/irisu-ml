# Persistent high-score search

For the restored October planner on the fast resident simulator, see
[October event-search fallback](historical-event-search.md).

For the combo-focused policy, event roots, durable frontier and terminal repair
qualification, see [Combo-focused search](combo-orchestrator.md).

`irisu-search` searches normal-mode game seeds, maintains an archive of exact
input recipes and live combo groups, rebuilds continuations after interventions,
and independently checks prospective records against the native exact oracle.
Its objective is the highest **first-recorded canonical terminal score**. A
natural game-over is eligible; completion does not outrank a higher score.
Capped rollouts remain exploration states and cannot become records.

The Python orchestrator has no third-party runtime dependencies. The MLP and
its trainer also use the standard library. Existing neural policy weights are
not required. Run commands below from the repository root with `PYTHONPATH=python`,
or install the package to use the `irisu-search` entry point.

## Build and start

Build the native bridge and independent verifier in the existing exact build:

```sh
cmake -S . -B build-project-exact
cmake --build build-project-exact --target irisu-search-engine irisu-game-oracle irisu-search-trace -j 2
```

Build the resident CUDA service in the accelerated physics configuration:

```sh
cmake -S . -B build-cuda-native-efficiency3 \
  -DIRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS=4096
cmake --build build-cuda-native-efficiency3 --target irisu-cuda-search-engine -j 2
```

A configuration inherits that build's numeric profile. The native bridge uses
immutable Linux fork/COW keepers. Fresh roots come from a pristine fixed-message
factory so protocol-buffer heap reuse cannot contaminate the legacy constructor. The CUDA bridge retains worlds on the GPU,
clones active state with existing compact snapshot helpers, and runs closed-loop
continuations in its kernel. `--host-prefix` prepares reconstruction prefixes
with the typed host implementation; future simulation remains on the GPU. Adaptive launch sizing distributes small
batches across SMs; `--threads N` overrides it for profiling. The sample campaign
uses 128 candidates and eight live survivors, with a 512-checkpoint device pool.
The independent native verifier remains authoritative for exported records.

The resident worker's contact capacity is independent of short-horizon benchmark
profiles: `IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS` defaults to4096 even when
`IRISU_CUDA_NATIVE_MAX_CONTACTS=512`. Worker identity reports `max_contacts` and
`checkpoint_bytes`. Dense, long continuations can exceed512 contacts; spare
checkpoint slots do not enlarge a simulated world's contact pool.

A per-candidate simulation rejection preserves healthy siblings. The worker
returns original candidate indices, rejected input traces and physics diagnostics,
and resets rejected slots before reuse. The coordinator stores `rollout_rejected`
events and immutable trace blobs, excludes invalid states from checkpoints and
training, and counts all attempted updates in the job's simulated-work ledger.
If every candidate is rejected, that job fails and the campaign continues; a
completed exhausted bootstrap probe counts once toward its seed's attempt budget.
Transport, protocol and CUDA infrastructure errors still stop the run.

See the [failure reproduction and recovery checks](resident-rollout-recovery.md).
Upgrading a frozen campaign starts a new campaign directory with the new
engine/planner identity and imports its previous best verified replay. Existing
campaign archives and receipts retain their original provenance.

The separate [GPU-first acceleration tools](../tools/cuda-native/gpu_first_README.md)
provide wider sparse-input fanouts, GPU-prepared checkpoint banks and optional
cooperative physics. Their batch runner uses binary candidate/result packets;
the orchestrator's `engine` setting requires the resident JSON worker described
above. Keep that worker and its existing hybrid profile for these campaigns.
The acceleration guide's short-horizon timings do not establish end-to-end
throughput for this orchestrator's closed-loop searches.

Create the campaign, including the published record and intermediate checkpoints:

```sh
PYTHONPATH=python python -m irisu_search init artifacts/high-score-v1 \
  --config configs/search/high-score-v1.json \
  --incumbent irisu-high-score-4393101.rpy --stride 4096
PYTHONPATH=python python -m irisu_search run artifacts/high-score-v1 --seconds 3600
PYTHONPATH=python python -m irisu_search status artifacts/high-score-v1
```

The sample config selects the current CUDA build and one GPU. Adjust build paths
or device before initialization. `cuda_device` controls only the engine child;
it does not change the desktop environment or unrelated processes. An empty
selector inherits the environment. For native-only operation, replace `engine`
with `["./build-project-exact/irisu-search-engine"]`.

Initialization expands paths and freezes settings plus engine/verifier hashes.
Resume with `run`; use a new campaign directory when changing the engine,
configuration, or numeric profile. Existing replay files are never modified.
There is no background service: `run` stays in the foreground and stops at its
budget. A running batch or verification completes at a boundary, so the wall
budget can overshoot by an engine request. Ctrl-C preserves completed batches;
rerun the same command to continue. Abruptly abandoned leases can be reclaimed
after their timeout. The ledger distinguishes unknown abandoned work from
completed, charged work.

## Work only on the current record seed

Initialize with `--champion-only` (or `"champion_only": true` in the JSON config):

```sh
PYTHONPATH=python python -m irisu_search init artifacts/high-score-champion \
  --config configs/search/high-score-v1.json --champion-only \
  --incumbent path/to/current-best.rpy --stride 4096
PYTHONPATH=python python -m irisu_search run artifacts/high-score-champion --seconds 3600
```

This frozen option requires an imported, independently verified champion before
search starts. It creates no random seed manifest, schedules every lane on the
verified champion's seed, and transfers the seed-discovery share to the combo
lane. Both ordinary and combo-focused policies support it. Combo event roots,
frontier continuations and terminal edits remain available within that seed.
Other seeds' queued jobs remain untouched and cannot be claimed in this mode.
If there are no usable live checkpoints, bootstrap reconstructs only the same
champion seed. A higher verified score becomes the new continuation reference;
changing the search policy requires a fresh campaign and an incumbent import.
`discover_new_seeds: false` alone still searches other seeds already in the
manifest, so it does not provide this restriction.

## What gets searched

A job starts at an immutable checkpoint and evaluates a batch of variable-length
input prefixes. Each prefix can be followed by a deterministic controller that
responds to the changed board. Modes prioritize recruitment, preservation, or
payout. Explicit interventions cover recruitment, handoff, partial payout,
preparation, timing and continuation repair. These are candidate generators,
not guarantees that a named goal succeeds. Exact rollout consequences decide.
Held-button edges are preserved; injected same-button shots receive a release
update when needed. The original replay suffix is retained as a baseline when
its prefix matches the parent; otherwise the baseline is waiting. A separate
reserved repair allocation can inject or suppress a shot and then resume the
reference inputs at their original absolute ticks. Diverged branches can also
follow those reference inputs without being labeled incumbent; the actual
changed-board outcome still requires independent verification. Reference recipes
are cached by immutable checkpoint identity rather than decoded for every batch.

Search returns true cumulative `chain` and `num`, live members, geometry, rot,
hit counts and pending clear/deletion state. Per-branch group deltas survive
extinction, so extension followed by full payout is not mislabeled as failure.
The search does not use the legacy global-combo tool's squared `Confirmed.value`
proxy (`value` is `num`, not chain length).

Selection separately retains live group potential, current score, and naturally
terminated candidates. Even a width-one live selection cannot be displaced by
an already-completed lower-score run. Every terminal is archived; only the best
terminal in a batch needs a verification attempt. Each new per-seed best is
verified before entering its verified scoreboard. The champion is protected
independently of live exploration.

The seed lane begins with 128 fresh seeds plus the control seed, with three
independent bootstrap attempts each. Retries of an interrupted attempt do not
count as additional attempts. New seeds continue to enter by default. Seed
shortlists combine live-group potential with verified results after a minimum
wall-time investment. Promoted seeds receive deeper work; finalists must first
receive the configured second-stage investment. Probe horizons can also deepen
after that actual investment while independent bootstrap attempts are still
pending; tournament status remains unchanged. The sample worker timeout allows
up to 600 seconds for a dense long probe to reach its batch boundary. Discovery seeds remain eligible for live development while their remaining
bootstrap attempts are scheduled; they need not wait for promotion to receive
deeper work. Dormant seeds remain eligible through the wildcard lane. Early
score alone is not an elimination rule.

Carrier-specific proposals have reserved catalog capacity, so a dense set of
generic pair candidates cannot crowd out the requested intervention.

Initial wall-time lane shares are 30% seed discovery, 45% combo development and
repair, 15% longer probes, and 10% wildcard exploration. Costs include preparation,
inference, publication and verification. The shares are adaptive queue targets,
not rigid resource reservations. Set `discover_new_seeds=false` for fixed-manifest
comparisons. Exploration floors remain in force with an active neural model.

## Archive and restart

SQLite WAL stores seed statistics, fenced job leases, settled costs, examples,
checkpoint lineage and verification receipts. Input recipes are immutable,
SHA-256-addressed little-endian words. Archive cells organize similar progress,
group counters and carrier geometry; they never assert identical physical state.
Several distinct recipes survive per cell. Imported replay roots reserve
coverage across replay progress within the active quotas, and can return from
cold storage when space becomes available. Each seed's best terminal stays active;
other terminal recipes remain cold evidence. Resident handles are only caches;
cold restart replays the recipe and checks the reconstructed summary.

The archive byte and checkpoint limits bound the **active archive**. Cold
recipes and evidence remain on disk; the limit is not a total disk quota.
Protected champions, per-seed terminal bests and useful control states can exceed
an undersized active limit; the report exposes `protected_over_budget`.
Snapshot formats and addresses are not transplanted across engine builds.

## Neural learning

Every evaluated branch contributes causal features of its starting state and
candidate. Features use actual group membership and counters, not numeric IDs
or the seed integer. A small MLP predicts eventual score gain, extension,
carrier survival, and the benefit of further search. Incomplete branches can
supervise observed extension/survival but are not scored as terminal failures.
The compute head uses job-level improvement in verified per-seed score divided
by actual search wall time, not the gross payout of one future.

Each job attempt and training example records the active guide hash. Retries
under a different guide retain both policy identities, and a prior attempt
cannot make an unfinished retry count as an observed terminal outcome.

Training holds out whole game seeds; normalization is fitted on training seeds.
Censored return targets are omitted and untrained heads remain neutral.
Automatic training uses deterministic, seed-stratified streaming samples with
bounded memory; scheduling uses database aggregates rather than loading past
job payloads. Periodic training (`train_every_jobs`) creates **shadow models** without changing
active search. For a manual train-and-use cycle:

```sh
PYTHONPATH=python python -m irisu_search train artifacts/high-score-v1 \
  artifacts/high-score-v1/models/guide.json --epochs 80
PYTHONPATH=python python -m irisu_search activate-model artifacts/high-score-v1 \
  artifacts/high-score-v1/models/guide.json
```

Activation requires matching planner/engine provenance and whole-seed prediction
validation. It does not prove a gameplay improvement. The candidate ranker always
retains the baseline and a random exploration allocation. Compute allocation uses
the learned head only after at least 32 observed training labels, and still
honors minimum seed budgets and wildcard exploration. Models use atomic,
checksummed JSON files; changing an activated file is rejected.

## Records, diagnostics and comparisons

```sh
PYTHONPATH=python python -m irisu_search verify artifacts/high-score-v1
PYTHONPATH=python python -m irisu_search export artifacts/high-score-v1 /tmp/new-record.rpy
PYTHONPATH=python python -m irisu_search diagnose artifacts/high-score-v1 \
  irisu-high-score-4393101.rpy --stride 256
PYTHONPATH=python python -m irisu_search diagnose artifacts/high-score-v1 \
  irisu-high-score-4393101.rpy --events --output /tmp/group-history.jsonl
PYTHONPATH=python python -m irisu_search compare artifacts/baseline artifacts/treatment
```

Export creates a new file exclusively. Verification replays from reset in the
independent native oracle, checking the canonical replay header and actual frame
count. The oracle's last live score/chain can differ from recorded metadata;
only first-recorded canonical score establishes a record.

`diagnose` imports checkpoint samples and reports persistent group identities;
its sample interval does not reconstruct every intervening event. Use `diagnose --events` (or the native `irisu-search-trace` executable) for full
event genealogy and score attribution. The recorded run has 1,187 scored-piece
events; group 399 alone contributes 4,155,740 canonical points, chain 70, with
two members still alive at termination. The tool distinguishes canonical-eligible
payouts from points applied after the first terminal snapshot.
Comparison reports all-in search costs and verified score curves. Use matching
seed manifests and wall-time budgets; compare old orchestration on the fast engine,
then continuation repair/archive, then neural guidance. Prediction loss alone
is not a score-search benchmark.

The new resident service reuses accelerated physics but has a different workload
from the engine's approximately 20x benchmark. Its full-campaign speed must be
measured separately. For 32-candidate resident jobs, measured adaptive launches
were about 4.1x faster at 128 ticks and 5.1x at 384 ticks than the original
one-CTA layout, with identical results. These are launch-layout comparisons,
not an overall search-speed claim. Successful bounded campaigns and parity tests establish
integration and correctness coverage, not a new high score or universal speedup.

## Validation

```sh
PYTHONPATH=python uv run --no-project --with pytest python -m pytest -q tests/test_search_*.py
PYTHONPATH=python IRISU_SEARCH_CUDA_ENGINE=build-cuda-native-efficiency3/clone/cuda_native/irisu-cuda-search-engine \
  uv run --no-project --with pytest python -m pytest -q tests/test_search_engine.py
```

Tests cover record replay, native/CUDA branch parity, parent immutability,
lease recovery, no duplicated accounting, stale-worker fencing, tight resident
capacity, reconstruction, verified record promotion, neural censoring/holdout
boundaries, feature invariance, and corrupt/runtime-mismatched artifacts.

The `High-score search` GitHub workflow runs the Python suite on Python 3.11
and 3.14, builds the portable native worker, and checks its resident checkpoint
behavior. Exact-library and CUDA gates remain separate hardware-dependent checks.
