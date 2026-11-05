# Resident CUDA rollout rejection and recovery

The November 4 campaign encountered two valid native continuations that exceeded
its short-horizon CUDA build's512-contact pool. One failed branch caused the
worker to reject128 candidates, and the resulting coordinator exception stopped
the campaign. Neither failure was a CUDA allocation or numerical failure.

| Game seed | Candidate index | Failed update | Occupied contacts | Healthy siblings |
| --- | ---: | ---: | ---: | ---: |
| 391769539 | 39 | 1822 | 512 | 127 |
| 400599612 | 27 | 1261 | 512 | 127 |

Indices are zero-based. Each original batch requested128 continuations of2048
updates. All254 healthy CUDA results matched native summaries, traces, interval
metrics and group deltas. Both rejected traces matched their native prefixes.
With4096 contacts, all256 original candidates completed and matched native.
The [receipt](../benchmarks/results/resident-rollout-recovery-2026-11-04.json)
records the input, binary and source pins and the executed regression checks.

## Capacity and partial results

`IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS=4096` is the resident worker default.
The global benchmark capacity setting remains independent. Explicit512 and1024
worker profiles remain available for controlled experiments. Larger contact
capacity preserves the original arithmetic, contact ordering and native verifier;
it does not remove the other typed-world admission limits.

The worker advertises `partial_batches:true`. Successful results carry their
original `candidate_index`. Rejected entries carry that index, the executed
`words` and a `diagnostic` containing physics status, tick, occupancy counts,
capacity and overflow flags. Rejected entries receive no handles. The wrapper
checks complete, unique index coverage and includes rejected traces in work
accounting. Legacy native full-batch replies remain supported.

The coordinator archives each rejected trace as an immutable blob with its
parent checkpoint, original proposal and diagnostic. Valid results retain the
correct proposal pairing. Exhausted batches settle as failed jobs and continue
scheduling; their completed bootstrap probes count exactly once. Interrupted
work, malformed protocol replies and infrastructure failures keep their existing
failure semantics.

Failed slots are reset to a cold Game and zero Scene before becoming reusable.
This matters because compact `clear_reachable` deliberately rejects a Scene
whose physics status is already failed.

## Regression gates

The committed fixture is an exact-native input prefix from the first failure.
Its last12 updates reproduce contact exhaustion quickly, avoiding a full2048
update rollout in each sanitizer run.

Build two separate CUDA worker profiles using the existing strict numeric flags,
one with `IRISU_CUDA_NATIVE_ORCHESTRATOR_MAX_CONTACTS=512` and one with4096. Then:

```sh
env -u CUDA_VISIBLE_DEVICES \
  IRISU_SEARCH_CUDA_LIMITED_ENGINE=/absolute/path/to/512/irisu-cuda-search-engine \
  IRISU_SEARCH_CUDA_ENGINE=/absolute/path/to/4096/irisu-cuda-search-engine \
  uv run --extra test python -m pytest -v tests/test_search_cuda_rejections.py
```

The small pool test retains a healthy sibling while immediately reusing the only
free rejected slot, then rejects an entire batch and successfully runs another.
Three checkpoint slots make leaks or failure-state reuse observable. Native
summary hashes check reconstruction and successful continuations. Portable tests
cover malformed partial replies, proposal lineage, training exclusion, durable
rejection traces, interruption and exactly-once tick/bootstrap accounting.

Frozen campaign identity includes the executable and planner source hashes.
Use a new campaign with the corrected frozen runtime and import the previous
verified champion when upgrading; retain the old campaign as evidence.
