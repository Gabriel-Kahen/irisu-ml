"""Reference repairs execute exact input timing in the real native worker."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from irisu_search.engine import Engine
from irisu_search.proposals import generate
from irisu_search.verification import read_replay

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("IRISU_SEARCH_ENGINE", ROOT / "build-project-exact/irisu-search-engine"))


@pytest.mark.skipif(not NATIVE.is_file(), reason="build native search engine")
def test_replay_repair_keeps_absolute_input_timing_and_immutable_parent():
    seed, words, _ = read_replay(ROOT / "irisu-high-score-4393101.rpy")
    start = next(index + 1 for index, word in enumerate(words) if word & 3)
    horizon = 96
    held = words[start - 1] & 3
    reference = words[start:start + horizon]
    with Engine([str(NATIVE)], timeout_seconds=120) as engine:
        parent = engine.load(seed, words[:start])
        proposals = generate(parent["summary"], 32, 711, "repair", reference,
                             horizon, reference, held)
        repaired = next(candidate for candidate in proposals
                        if candidate.get("replay_repair")
                        and candidate["words"][0] == 0
                        and candidate["words"][1] & held)
        assert repaired["mode"] == 0
        assert repaired["words"][2] == 0
        assert repaired["words"][3:] == reference[3:]
        candidates = [proposals[0], repaired]
        baseline, child = engine.batch(parent["handle"], horizon, candidates)["results"]
        assert baseline["words"] == reference
        assert child["words"] == repaired["words"]
        assert child["summary"]["tick"] == start + horizon
        assert child["words"] != reference
        replayed = engine.load(seed, words[:start] + child["words"])
        assert replayed["summary"] == child["summary"]

        # Continue the changed board with the original inputs at its new tick.
        tick = child["summary"]["tick"]
        suffix = words[tick:tick + 64]
        followups = generate(child["summary"], 16, 712, "repair", (), 64, suffix,
                             child["words"][-1] & 3)
        followup = next(candidate for candidate in followups
                        if candidate["operator"] == "replay-reference")
        assert followups[0]["operator"] == "wait"
        continued = engine.batch(child["handle"], 64, [followup])["results"][0]
        assert continued["words"] == suffix
        reconstructed = engine.load(seed, words[:start] + child["words"] + suffix)
        assert reconstructed["summary"] == continued["summary"]

        # Both repairs forked the keeper; its original future is unchanged.
        again = engine.batch(parent["handle"], horizon, candidates)["results"]
        assert [(row["words"], row["summary"]) for row in again] == [
            (row["words"], row["summary"]) for row in (baseline, child)]
