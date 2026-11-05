"""Rejected candidates keep their original indices without creating checkpoints."""
from copy import deepcopy

import pytest

from irisu_search.engine import Engine, EngineError


def success(index=0, handle=2):
    return {"candidate_index": index, "handle": handle, "words": [0, 0], "summary": {
        "seed": 7, "tick": 2, "score": 0, "gauge": 0, "level": 0, "qualifying": 0,
        "highest_chain": 0, "canonical_score": 0, "canonical_level": 0,
        "canonical_chain": 0, "terminated": True, "truncated": False,
        "groups": [], "bodies": [],
    }}


def rejection(index=1):
    return {"candidate_index": index, "words": [0], "diagnostic": {
        "physics_status": 1, "status": "capacity", "tick": 1, "max_contacts": 512,
    }}


def batch(reply):
    engine = Engine.__new__(Engine)
    engine._request = lambda _: deepcopy(reply)
    return engine.batch(1, 4, [{"mode": 0}, {"mode": 0}])


def test_partial_batch_preserves_indices_and_counts_rejected_work():
    reply = batch({"results": [success(1)], "rejected": [rejection(0)], "simulated_ticks": 3})
    assert reply["results"][0]["candidate_index"] == 1
    assert reply["rejected"][0]["candidate_index"] == 0
    assert "handle" not in reply["rejected"][0]
    assert reply["simulated_ticks"] == 3


def test_all_rejected_reply_is_valid_and_keeps_its_diagnostics():
    reply = batch({"results": [], "rejected": [rejection(1), rejection(0)], "simulated_ticks": 2})
    assert reply["results"] == []
    assert [row["candidate_index"] for row in reply["rejected"]] == [0, 1]
    assert reply["simulated_ticks"] == 2


def test_legacy_full_native_batch_without_tick_accounting_still_works():
    rows = [success(handle=2), success(handle=3)]
    for row in rows:
        del row["candidate_index"]
    reply = batch({"results": rows, "requested_ticks": 8})
    assert [row["candidate_index"] for row in reply["results"]] == [0, 1]
    assert reply["rejected"] == []
    assert reply["simulated_ticks"] == 4


@pytest.mark.parametrize("index", [-1, 2, True, None])
def test_partial_batch_rejects_invalid_original_indices(index):
    with pytest.raises(EngineError, match="candidate index"):
        batch({"results": [success()], "rejected": [rejection(index)], "simulated_ticks": 3})


@pytest.mark.parametrize("change", [
    lambda r: r["rejected"][0].update(candidate_index=0),
    lambda r: r["results"][0].pop("candidate_index"),
    lambda r: r["rejected"][0].update(handle=9),
    lambda r: r["rejected"][0].update(diagnostic={}),
    lambda r: r["rejected"][0].update(words=[1 << 21]),
    lambda r: r["rejected"][0].update(words=[0] * 5),
    lambda r: r["results"][0]["summary"].update(failed=True),
    lambda r: r["results"][0].update(handle=1),
    lambda r: r.update(simulated_ticks=2),
    lambda r: r.update(simulated_ticks=True),
    lambda r: r.pop("simulated_ticks"),
    lambda r: r.update(rejected=[]),
])
def test_partial_batch_rejects_malformed_evidence_or_accounting(change):
    reply = {"results": [success()], "rejected": [rejection()], "simulated_ticks": 3}
    change(reply)
    with pytest.raises(EngineError):
        batch(reply)
