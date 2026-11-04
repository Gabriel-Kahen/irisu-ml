# Shared futures and local solver state

The third efficiency campaign improves the preceding capacity512 profile on
the same V100 and the same 12,288 distinct four-tick candidates. Three paired,
alternating-order runs per mode measure complete process wall time, including
allocation, prefix preparation, transfers, output, guards, and teardown:

| Mode | Previous seconds | New seconds | Speedup |
| --- | ---: | ---: | ---: |
| Same simulated work, solver changes only | 1.829966 | 1.768809 | 1.035x |
| Shared futures, future physics on GPU | 1.824302 | 1.503289 | 1.214x |
| Shared futures, checked CPU checkpoints | 1.830195 | 1.379781 | 1.326x |

The best result takes 24.61% less time. It is an algorithmic search improvement:
equivalent candidates share exact partial trajectories, reducing simulation
work. It is not a 1.326x improvement in execution of identical physics work.
Each comparison has its own contemporary baseline. The candidate ranges are
1.767668–1.775347, 1.502819–1.506745, and 1.378154–1.389647 seconds, respectively.
The [receipt](../../benchmarks/results/cuda-native-efficiency3-2026-11-04.json)
pins binaries, sources, raw timing samples, experiments, and qualification.

## Selected changes

The three selected solver options preserve native contact order and every
rounded operation. Local velocity/position states commit at every contact exit,
including numeric rejection. Contact-local anchors are reused between normal
and tangent passes without allocating a whole-island cache. Their generated
graphs and qualification commands are in the
[solver contract](solver_efficiency3_contract.md).

`--adaptive-shared-prefixes` builds a bounded cache forest of exact input-word
prefixes. Each candidate resumes from its deepest selected checkpoint. Singleton
groups retain an uninterrupted future; unused checkpoints are removed. Every
checkpoint is prepared independently, so its full preparation cost is included.
This is a cache forest, not a claim that all common trie edges execute once.
The planner bounds split depths, candidate groups, and retained checkpoints.
`--share-future-prefixes` retains the simpler single-split experiment;
`--shared-split N` explicitly chooses that split.

For the campaign, four slots hold the root and prefixes of lengths 1, 2, and 3.
Per batch, the logical 49,152 candidate ticks need 30,734 actual GPU ticks.
Across 20 fresh batches this is 614,680 GPU ticks, versus 983,040 previously.
The original shared 100-tick prefix is checked CPU preparation in all table rows.

`--host-shared-checkpoints` additionally prepares six checked CPU ticks once
alongside that original prefix, uploads and rebinds the four checkpoints, then
retains them across prefix reuse. The 20 batches execute 614,560 GPU ticks;
the six CPU ticks are reported separately. Every distinct candidate still
executes its remaining future. Complete Result/rank packets match native.
Full raw World/Scene/carry-state bytes of each retained source are checked
after every batch, in addition to the original prefix guard. Allocation is
13,405,208,695 bytes, compared with 13,396,413,647 bytes previously.

## Build and reproduce

Start with the [capacity512 build](efficiency2_README.md), use a separate build
directory, and additionally select:

```sh
-DIRISU_CUDA_NATIVE_LOCAL_VELOCITY=ON \
-DIRISU_CUDA_NATIVE_LOCAL_POSITION=ON \
-DIRISU_CUDA_NATIVE_CONTACT_ANCHORS=ON \
-DIRISU_CUDA_NATIVE_CACHED_VELOCITY=OFF \
-DIRISU_CUDA_NATIVE_SPLIT_CONSTRUCT=OFF
```

The selected build is `build-cuda-native-efficiency3`. Keep all other preceding
selected options, strict arithmetic, contact capacity512, and frontend `min`.
The new options default off. Approximate trig, global anchor caching, entry
indexing, sparse reset, and register caps remain off.

Use the preceding [distinct input and native packet commands](search_README.md).
Add the following candidate flags to `benchmark_gpu_efficiency.py`; use the
preceding optimized runtime flags on both baseline and candidate:

```sh
--candidate-flag=--adaptive-shared-prefixes \
--candidate-flag=--host-shared-checkpoints
```

Omit the second flag to prepare all partial futures on GPU. Both table sharing
modes use `--host-prefix`; removing that flag is a separate experiment and is
incompatible with host shared checkpoints. Report logical candidate ticks,
actual GPU ticks, and CPU checkpoint preparation separately. Timing disables
only extra selected-state verification, which is qualified separately.

## Verification and limits

With sharing and selected-state checks enabled, selected candidates are replayed
independently from the original prefix. Their complete Results, normalized live
Game values, occupied physics state, full SAP history, and poisoned-scratch
continuations are compared with the shared path. Inactive payloads and listed
live ABI padding retain the existing compact-snapshot contract. Small gates
select every candidate; the large gate selects the top two.

Run the native search matrix with `qualify_search_efficiency.py`, adding
`--all-small-selected --late-prefixes --spread-small-candidates` and
`--runner-flag=--adaptive-shared-prefixes`. For the checked CPU checkpoint mode,
also add `--force-host-prefix --runner-flag=--host-shared-checkpoints`.
The spread option samples eight inputs across the 512-candidate mutation menu;
the first eight ordinary inputs have no reusable prefix and test the fallback.
The strengthened gates require sharing to activate and independently replay
all small leaves. Memory checks must likewise use the spread fixture.

The selected profile passes the complete 97,220-frame native replay, all seven
256-case GPU solver modes, 3,072 dynamic native physical/SAP frames, 12,288
physical snapshots, and game snapshot continuations at ticks100,96000,97150.
The strengthened search matrices pass11 GPU-prepared and9 CPU-checkpoint cases;
every case actually activates sharing. Memcheck/initcheck pass for both modes,
and virgin SceneScratch initcheck passes for both snapshot qualifiers. All ten
memory-check logs report zero errors. The12 original CPU regression suites pass.

The campaign is a large cached-prefix workload, not a universal speedup or a
demonstrated score improvement. Contact capacity512 retains its explicit
[restricted admission and overflow behavior](contact_capacity_contract.md).
CPU authority and existing CPU/translated implementations remain unchanged.

Rejected or weaker trials include whole-island anchor caching, split factory
construction, compiler vectorization, register caps128–224, soft register
optimization, L1 preference, and bypassing L1. Full frontend optimization did
not finish a bounded 900-second compile; it has no executable or speed claim.
The first shared-prefix implementation allocated thousands of redundant
singleton checkpoints; selective and adaptive caching supersede it. Further
large kernel gains would require a broader state-layout/execution redesign;
these measurements do not establish a hardware-stall diagnosis.
