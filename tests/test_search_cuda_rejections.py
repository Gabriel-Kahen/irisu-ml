"""Optional device gates using the campaign's exact contact-exhaustion recipe."""
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from irisu_search.engine import Engine


def recipe():
    data = json.loads((Path(__file__).parent / "fixtures/search/contact-capacity-512.json").read_text())
    words = [0] * data["prefix_ticks"]
    for tick, word in data["nonzero_words"]:
        words[tick] = word
    assert hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest() == data["prefix_sha256"]
    return data, words


def summary_hash(summary):
    return hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def exercise_limited(engine):
    data, words = recipe()
    assert engine.identity["max_contacts"] == 512
    parent = engine.load(data["seed"], words)
    assert summary_hash(parent["summary"]) == data["prefix_summary_sha256"]
    bad = {"words": data["failing_tail"]}
    mixed = engine.batch(parent["handle"], 12, [bad, {"words": [0] * 12}])
    assert [row["candidate_index"] for row in mixed["results"]] == [1]
    assert [row["candidate_index"] for row in mixed["rejected"]] == [0]
    assert summary_hash(mixed["results"][0]["summary"]) == data["waiting_summary_sha256"]
    assert mixed["simulated_ticks"] == 24
    failure = mixed["rejected"][0]
    assert failure["words"] == data["failing_tail"]
    assert failure["diagnostic"]["status"] == "capacity"
    assert failure["diagnostic"]["contact_count"] == failure["diagnostic"]["max_contacts"] == 512
    # With three slots, the rejected slot is the only free one. Its poisoned
    # failed status must not reach compact clear_reachable or a new checkpoint.
    follow = engine.batch(parent["handle"], 1, [{"words": data["failing_tail"][:1]}])
    assert not follow["rejected"]
    assert summary_hash(follow["results"][0]["summary"]) == data["one_tick_summary_sha256"]
    engine.release([mixed["results"][0]["handle"], follow["results"][0]["handle"]])
    exhausted = engine.batch(parent["handle"], 12, [bad, bad])
    assert not exhausted["results"] and len(exhausted["rejected"]) == 2
    assert exhausted["simulated_ticks"] == 24
    healthy = engine.batch(parent["handle"], 1, [{"words": data["failing_tail"][:1]}])
    assert summary_hash(healthy["results"][0]["summary"]) == data["one_tick_summary_sha256"]
    engine.release([healthy["results"][0]["handle"], parent["handle"]])
    reset = engine.load(42, [])
    assert reset["summary"]["seed"] == 42 and reset["summary"]["tick"] == 0
    engine.release([reset["handle"]])


@pytest.mark.skipif(not os.environ.get("IRISU_SEARCH_CUDA_LIMITED_ENGINE"),
                    reason="set explicit 512-contact CUDA worker for rejection gate")
def test_cuda_mixed_and_exhausted_batches_recycle_failed_slots():
    with Engine([os.environ["IRISU_SEARCH_CUDA_LIMITED_ENGINE"], "--host-prefix",
                 "--max-checkpoints", "3"], timeout_seconds=240) as engine:
        exercise_limited(engine)


@pytest.mark.skipif(not os.environ.get("IRISU_SEARCH_CUDA_ENGINE"),
                    reason="set explicit default-capacity CUDA worker for native gate")
def test_cuda_default_worker_admits_contact_dense_native_recipe():
    data, words = recipe()
    with Engine([os.environ["IRISU_SEARCH_CUDA_ENGINE"], "--host-prefix",
                 "--max-checkpoints", "3"], timeout_seconds=240) as engine:
        assert engine.identity["max_contacts"] == 4096
        parent = engine.load(data["seed"], words)
        result = engine.batch(parent["handle"], 12, [{"words": data["failing_tail"]}])
        assert not result["rejected"] and len(result["results"]) == 1
        assert summary_hash(result["results"][0]["summary"]) == data["bad_summary_sha256"]
