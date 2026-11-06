"""Repair qualification must distinguish novel verified gains from replay ties."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualify_combo", ROOT / "tools/orchestrator/qualify_combo.py")
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def case():
    return {"id": "damage", "prefix_tick": 0, "prefix_sha256": qualification.words_hash([]),
            "damage": [{"tick": 0, "original_word": 2, "damaged_word": 1}],
            "damaged_suffix_sha256": qualification.words_hash([1, 0, 0]),
            "horizon": 3, "damaged_score": 100, "reachable_score": 200}


def result(words, score, ended=True):
    return {"handle": 2, "words": words, "summary": {
        "tick": len(words), "score": score, "canonical_score": score if ended else 0,
        "level": 1, "canonical_level": 1, "highest_chain": 0, "canonical_chain": 0,
        "terminated": ended, "truncated": False}}


class Verifier:
    def __init__(self):
        self.calls = []

    def verify(self, checkpoint):
        self.calls.append(checkpoint)
        return {"verified": True, "canonical_score": checkpoint["summary"]["canonical_score"]}


def fake_worker(monkeypatch, rows):
    class Worker:
        identity = {"max_checkpoints": 64}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load(self, seed, words):
            return {"handle": 1, "summary": {"seed": seed, "tick": len(words)}}

        def batch(self, parent, horizon, candidates):
            assert parent == 1 and horizon == 3
            return {"results": rows, "simulated_ticks": sum(len(row["words"]) for row in rows)}

        def release(self, handles):
            assert handles == [row["handle"] for row in rows]

    monkeypatch.setattr(qualification, "Engine", Worker)
    monkeypatch.setattr(qualification, "candidates_for", lambda *args: [{"mode": 0}] * len(rows))


def test_fixture_damage_is_reconstructed_and_hash_checked():
    assert qualification.damaged_reference([2, 0, 0], case()) == [1, 0, 0]
    with pytest.raises(ValueError, match="does not match"):
        qualification.damaged_reference([3, 0, 0], case())
    altered = {**case(), "damaged_suffix_sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash mismatch"):
        qualification.damaged_reference([2, 0, 0], altered)


def test_damage_classes_preserve_legal_coordinates_and_button_edges():
    word = 1 | (640 << 2) | (480 << 12)
    reference = [word, 0, 0, 0]
    moved_x = qualification.damage_shot(reference, "x")
    moved_y = qualification.damage_shot(reference, "y")
    assert ((moved_x[0] >> 2) & 1023) == 638
    assert ((moved_y[0] >> 12) & 511) == 478
    assert qualification.damage_shot(reference, "timing") == [0, 0, word, 0]
    assert qualification.damage_shot([word, word, 0, 0], "timing") is None
    assert reference == [word, 0, 0, 0]


def test_unchanged_baseline_cannot_count_as_recovery(tmp_path, monkeypatch):
    fake_worker(monkeypatch, [result([1, 0, 0], 100)])
    verifier = Verifier()
    report = qualification.run_arm("worker", verifier, 7, [2, 0, 0], case(),
                                   "current", 1, 10, 13, tmp_path)
    assert report["baseline_terminal_ties"] == 1
    assert report["verified_novel_gain"] == report["recovery_fraction"] == 0
    assert report["verification"] is None and verifier.calls == []
    assert not list(tmp_path.iterdir())


def test_changed_baseline_result_invalidates_the_comparison(tmp_path, monkeypatch):
    fake_worker(monkeypatch, [result([1, 0, 0], 200)])
    with pytest.raises(RuntimeError, match="independently verified fixture"):
        qualification.run_arm("worker", Verifier(), 7, [2, 0, 0], case(),
                              "combo", 1, 10, 13, tmp_path)


def test_capped_live_score_cannot_count_as_terminal_gain(tmp_path, monkeypatch):
    fake_worker(monkeypatch, [result([3, 0, 0], 10000, ended=False)])
    verifier = Verifier()
    report = qualification.run_arm("worker", verifier, 7, [2, 0, 0], case(),
                                   "combo", 1, 10, 13, tmp_path)
    assert report["censored_results"] == 1 and report["novel_terminal_results"] == 0
    assert report["verified_novel_gain"] == 0 and verifier.calls == []


def test_novel_gain_requires_verification_and_exports_exact_recipe(tmp_path, monkeypatch):
    fake_worker(monkeypatch, [result([3, 0, 0], 150)])
    verifier = Verifier()
    report = qualification.run_arm("worker", verifier, 7, [2, 0, 0], case(),
                                   "combo", 1, 10, 13, tmp_path)
    assert verifier.calls[0]["words"] == [3, 0, 0]
    assert report["verified_novel_gain"] == 50 and report["recovery_fraction"] == .5
    assert report["attempted_candidates"] == 1 and report["simulated_ticks"] == 3
    assert Path(report["verified_replay"]).is_file()


def test_frozen_native_fixture_has_a_nontrivial_hidden_recovery_witness():
    fixture = ROOT / "tests/fixtures/search/combo-repair.json"
    if not fixture.exists():
        pytest.skip("generate the independently verified combo repair fixture")
    data = json.loads(fixture.read_text())
    replay = ROOT / "irisu-high-score-4393101.rpy"
    seed, words, _ = qualification.read_replay(replay)
    assert qualification.sha256_file(replay) == data["replay_sha256"]
    for entry in data["cases"]:
        broken = qualification.damaged_reference(words, entry)
        assert entry["seed"] == seed and broken != words[entry["prefix_tick"]:]
        assert entry["reachable_score"] > entry["damaged_score"]
        assert entry["damaged_verification"]["verified"]
        assert entry["hidden_witness_verification"]["verified"]


def test_partial_receipt_survives_a_later_arm_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "NativeVerifier", lambda *args, **kwargs: object())
    calls = []

    def run_arm(*args):
        calls.append(args[5])
        if len(calls) == 2:
            raise RuntimeError("independent verification rejected the candidate")
        return {"algorithm": args[5], "verification": None, "verified_novel_gain": 0}

    monkeypatch.setattr(qualification, "run_arm", run_arm)
    output = tmp_path / "comparison"
    with pytest.raises(RuntimeError, match="verification rejected"):
        qualification.main(["eval", "--output", str(output), "--cases", "1"])
    report = json.loads((output / "receipt.json").read_text())
    assert len(report["results"]) == 1 and report["results"][0]["algorithm"] == "current"
    assert report["failure"]["algorithm"] == "combo"


@pytest.mark.skipif(not os.environ.get("IRISU_SEARCH_ORACLE"),
                    reason="set the fixed independent native oracle for real repair qualification")
def test_real_combo_repair_recovers_hidden_record_suffix(tmp_path):
    fixture = json.loads((ROOT / "tests/fixtures/search/combo-repair.json").read_text())
    seed, words, _ = qualification.read_replay(ROOT / "irisu-high-score-4393101.rpy")
    verifier = qualification.NativeVerifier(os.environ["IRISU_SEARCH_ORACLE"],
                                            os.environ.get("IRISU_SEARCH_ORACLE_LIBRARY", ""))
    engine = Path(os.environ.get("IRISU_SEARCH_ENGINE", ROOT / "build-project-exact/irisu-search-engine"))
    if not engine.is_file():
        pytest.skip("build the exact native worker")
    report = qualification.run_arm(engine, verifier, seed, words, fixture["cases"][0],
                                   "combo", 32, 120, 20261105, tmp_path)
    assert report["verification"]["verified"]
    assert report["verified_novel_gain"] > 0
    assert report["recovery_fraction"] >= 1
