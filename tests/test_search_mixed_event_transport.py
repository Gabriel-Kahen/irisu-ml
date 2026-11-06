"""Mixed-parent event futures must equal independent immutable-parent runs."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from irisu_search.engine import Engine
from irisu_search.verification import read_replay

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("IRISU_SEARCH_ENGINE", ROOT / "build-project-exact/irisu-search-engine"))
CUDA = os.environ.get("IRISU_SEARCH_CUDA_ENGINE")


def exercise(engine):
    seed, words, _ = read_replay(ROOT / "irisu-high-score-4393101.rpy")
    parents = [engine.load(seed, words[:270]), engine.load(seed, words[:953]),
               engine.load(42, [0] * 100)]
    requests = [
        {"handle": parents[0]["handle"], "horizon": 130, "words": words[270:400]},
        {"handle": parents[1]["handle"], "horizon": 1, "words": words[953:954]},
        {"handle": parents[0]["handle"], "horizon": 7, "words": []},
        {"handle": parents[2]["handle"], "horizon": 12, "words": [0, (200 << 12) | (300 << 2) | 2]},
    ]
    mixed = engine.multi_batch_events(requests)
    assert mixed["rejected"] == []
    assert mixed["simulated_ticks"] == 150
    outputs = []
    for index, row in enumerate(mixed["results"]):
        assert row["candidate_index"] == index
        request = requests[index]
        independent = engine.batch_events(request["handle"], request["horizon"],
                                          [{"mode": 0, "words": request["words"]}])["results"][0]
        for key in ("summary", "words", "events", "metrics", "group_deltas"):
            assert row[key] == independent[key]
        outputs.append({key: row[key] for key in ("summary", "words", "events", "metrics", "group_deltas")})
        engine.release([independent["handle"], row["handle"]])
    again = engine.multi_batch_events(requests)
    assert [{key: row[key] for key in outputs[0]} for row in again["results"]] == outputs
    return outputs


@pytest.mark.skipif(not NATIVE.is_file(), reason="build native search engine")
def test_native_mixed_parents_durations_seeds_and_reuse():
    with Engine([str(NATIVE), "--max-checkpoints", "16"]) as engine:
        exercise(engine)


@pytest.mark.skipif(not CUDA or not NATIVE.is_file(), reason="set IRISU_SEARCH_CUDA_ENGINE for device integration")
@pytest.mark.parametrize("threads", [None, 4])
def test_device_mixed_compact_parents_match_independent_native(threads):
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    command = [str(CUDA), "--host-prefix", "--max-checkpoints", "16"]
    if threads:
        command.extend(["--threads", str(threads)])
    with Engine([str(NATIVE), "--max-checkpoints", "16"]) as native:
        expected = exercise(native)
    with Engine(command, env=environment) as device:
        assert exercise(device) == expected
