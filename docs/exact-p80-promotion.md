# Exact 200k-p80 promotion contract

Version: `irisu-exact-p80-200k-contract-v1`

The final report passes only when all of the following hold over exactly 20
distinct, manifest-bound episodes:

- at least 16 scores are at least 100,000;
- the median score is at least 100,000;
- the linear p80 score is at least 200,000;
- ascending order statistic 16 is at least 200,000, equivalently at least five
  scores are at least 200,000; and
- the live invalid-action count is zero.

The headline p80 uses Hyndman-Fan type 7, the default `linear` definition used
by NumPy. For ascending scores `x[1] ... x[20]`, its zero-based position is
`(20 - 1) * 0.8 = 15.2`, so the reported value is
`0.8 * x[16] + 0.2 * x[17]`. The report also records this value as an exact
rational number. The order-statistic companion prevents a very large `x[17]`
from interpolating a sub-200k `x[16]` into a pass.

This is an upper-tail objective: it guarantees five of 20 scores at or above
200,000. A claim that 80% of episodes score at least 200,000 would instead
need a lower-tail p20 requirement.

The development manifest uses a fresh `validation` allocator namespace. The
one-shot final manifest uses a fresh `test` namespace, is disjoint from the
development and earlier checked-in evaluation manifests, and may be opened at
most once. Exact evaluation still rejects any overlap with checkpoint-declared
training seeds at runtime.

The p80 evaluator and merger are new versioned entry points. Earlier 100k
shards and reports retain their original format and pass semantics. P80 shards
bind the full contract, checkpoint/model/inference configuration, planner,
runtime, exact worker/library, runner, and evaluator-engine hashes. The merger
requires identical identities, disjoint shards, exact requested-seed coverage,
and recomputes every gate statistic from episode scores.
