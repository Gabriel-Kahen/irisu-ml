# Native CPU cached-prefix campaign control

`native_search_campaign.cpp` links only the existing authoritative native exact
`irisu::Simulator` and reuses the read-only `search_io.hpp` / search-v2 packet
definitions, sparse candidates, frame folding and canonical ranking. It does
not include typed GPU physics or its optional arithmetic mode.

```
irisu-native-search-campaign replay.rpy specs.bin prefix horizon workers batches target-score [--final-only]
```

The exact-only benchmark target must use the same ELF32 native core/linkage and
floating-point settings as `irisu-search-oracle`. Workers are 1..64; batches are
1..10000. The existing loader enforces candidate count, edits and replay bounds.
The process installs x87 PC53 round-nearest before constructing/resetting the
simulator. A prefix ending at or after termination/truncation is rejected.

The single-threaded parent constructs/resets and replays the prefix exactly
once. It never steps that checkpoint again. Every batch forks direct candidate
children from that same parent, preserving the native heap and hidden physics
state through OS copy-on-write. A shared mapping contains only result packets
and child timing/completion slots. Each child runs the same future/fold/objective
code as the original native oracle, stopping after terminal/truncated state.
Wait-any scheduling keeps up to the requested number of workers active. The
parent ranks the complete array with the same candidate-index tie breaker.

Every batch's complete 256-byte Results and rank array must equal the first
batch byte for byte. Completion slots prevent accepting stale shared output.
The parent checks its unchanged prefix frame and loaded-input fingerprint after
each batch. This frame check is not a raw hidden-state digest; isolation of the
native private heap follows from fork/COW and the absence of parent stepping.
Shared output is deliberately outside that checkpoint. Any child error aborts
the campaign, kills/reaps remaining direct children, and emits no successful
packet. Parent-death signals plus the post-fork owner check prevent orphan
candidate jobs when the launcher dies.

Stdout contains only the last complete standard search-v2 packet, so an external
qualifier can compare the entire packet against the original native oracle and
the GPU `--reuse-prefix` path. Stderr emits one JSON receipt, including each
batch's actual simulated future ticks, rank/verification costs and timings.
`packet_fnv1a64` is a regression fingerprint, not a cryptographic attestation;
the external receipt must hash the replay, specs, binaries, mapped native DSO
and full output packet with SHA-256.

The timing fields have explicit scopes:

* `prefix_ms`: native simulator construction/reset, the one replay prefix and
  its initial frame digest.
* `fork_and_rollout_ms`: dispatch, bounded concurrency, candidate futures and
  all child reaping. This is the elapsed native branch-work phase.
* `fork_syscall_ms` and `wait_ms`: parent-observed sums within that phase.
  `child_rollout_sum_ms` / `child_rollout_max_ms` measure child future/fold/key
  work. These overlap and must not be added to the elapsed phase.
* `rank_ms`: result materialization and canonical rank sorting.
* `batch_wall_ms`: output-slot reset through packet/input/checkpoint verification.
* `output_ms`: one final binary packet emission, not one emission per batch.
* `cold_process_wall_ms`: main entry through that final packet flush, including
  loading, one prefix and all batches. Dynamic-loader startup, final stderr
  reporting and process exit require an external launcher wall timer.

This controls the same repeated candidate workload as a GPU campaign retaining
one prefix. Both sides must retain their prefix once, use identical specs,
horizon, trace policy and actual terminal stopping, and return the same packet.
Report cold wall and prefix separately from repeated future work. Keep original
one-shot end-to-end anchors; this control neither replaces them nor establishes
a speedup by itself. An independent native packet gate and cleanup check are
required before uncontended timing. No performance result is claimed here.
