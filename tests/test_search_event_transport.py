"""Raw event capture preserves replay semantics and event ordering."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from irisu_search.engine import Engine
from irisu_search.verification import read_replay

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("IRISU_SEARCH_ENGINE", ROOT / "build-project-exact/irisu-search-engine"))
CUDA = os.environ.get("IRISU_SEARCH_CUDA_ENGINE")


@pytest.mark.skipif(not NATIVE.is_file(), reason="build native search engine")
def test_event_capture_is_exact_chunkable_and_parent_is_immutable():
    seed, words, _ = read_replay(ROOT / "irisu-high-score-4393101.rpy")
    start, horizon = 270, 130
    with Engine([str(NATIVE), "--max-checkpoints", "8"]) as engine:
        parent = engine.load(seed, words[:start])
        candidate = {"mode": 0, "words": words[start:start + horizon]}
        captured = engine.batch_events(parent["handle"], horizon, [candidate])["results"][0]
        ordinary = engine.batch(parent["handle"], horizon, [candidate])["results"][0]
        assert captured["summary"] == ordinary["summary"]
        assert captured["words"] == ordinary["words"]
        assert "events" not in ordinary
        events = captured["events"]
        assert {"chain_joined", "confirmed", "score_changed", "contact"} <= {e["kind"] for e in events}
        assert all(start <= event["tick"] <= start + horizon for event in events)
        assert [e["sequence"] for e in events] == list(range(events[0]["sequence"], events[0]["sequence"] + len(events)))
        first = engine.batch_events(parent["handle"], 64, [{"mode": 0, "words": candidate["words"][:64]}])["results"][0]
        second = engine.batch_events(first["handle"], 66, [{"mode": 0, "words": candidate["words"][64:]}])["results"][0]
        assert first["events"] + second["events"] == events
        assert second["summary"] == captured["summary"]
        bodies = captured["summary"]["bodies"]
        assert bodies
        for body in bodies:
            assert body["projectile_hits"] == body["hits"]
            assert body["remaining_lifetime"] == body["lifetime"]
            assert body["shape"] in (0, 1, 2)
            assert body["lifecycle"] in (0, 1, 2, 3)
            assert body["age_ticks"] >= 0
        with pytest.raises(ValueError, match="1..512"):
            engine.batch_events(parent["handle"], 513, [{"mode": 0}])


@pytest.mark.skipif(not CUDA or not NATIVE.is_file(), reason="set IRISU_SEARCH_CUDA_ENGINE for device integration")
@pytest.mark.parametrize("start,horizon", [(270, 130), (96_000, 32)])
def test_device_events_and_observations_match_independent_native(start, horizon):
    seed, words, _ = read_replay(ROOT / "irisu-high-score-4393101.rpy")
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    with Engine([str(NATIVE)]) as native, Engine([str(CUDA), "--host-prefix", "--max-checkpoints", "8"], env=environment) as device:
        a = native.load(seed, words[:start])
        b = device.load(seed, words[:start])
        assert a["summary"] == b["summary"]
        candidates = [{"mode": 0, "words": words[start:start + horizon]}, {"mode": 0}]
        left = native.batch_events(a["handle"], horizon, candidates)
        right = device.batch_events(b["handle"], horizon, candidates)
        assert right["rejected"] == []
        for x, y in zip(left["results"], right["results"]):
            assert x["summary"] == y["summary"]
            assert x["words"] == y["words"]
            assert x["events"] == y["events"]


@pytest.mark.skipif(not NATIVE.is_file(), reason="build native search engine")
def test_short_event_descendants_keep_a_bounded_keeper_stack():
    with Engine([str(NATIVE), "--max-checkpoints", "3"]) as engine:
        state = engine.load(42)
        for tick in range(1, 129):
            child = engine.batch_events(state["handle"], 1, [{"mode": 0}])["results"][0]
            assert child["summary"]["tick"] == tick
            engine.release([state["handle"]])
            state = child
        independent = engine.load(42, [0] * 128)
        assert state["summary"] == independent["summary"]
