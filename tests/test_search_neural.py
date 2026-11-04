from __future__ import annotations

import copy
import json
import math

import pytest

from irisu_search.neural import FEATURE_NAMES, NeuralGuide, features, train_guide


def board(chain=20, num=10):
    return {
        "seed": 101, "tick": 80_000, "score": 123_456, "level": 80,
        "qualifying": 790, "gauge": 12_000,
        "groups": [{"id": 17, "chain": chain, "num": num, "members": [4, 9]}],
        "bodies": [
            {"id": 4, "color": 1, "chain_id": 17, "x": 250, "y": 180, "hits": 0},
            {"id": 9, "color": 1, "chain_id": 17, "x": 300, "y": 260, "hits": 1},
            {"id": 2, "color": 1, "chain_id": 0, "x": 290, "y": 120},
        ],
    }


def examples():
    result = []
    for seed in range(8):
        for i in range(12):
            useful = i % 2
            summary = board(chain=2 + useful * 30, num=useful * 20)
            candidate = {"mode": useful, "operator": "recruitment", "budget_seconds": 2}
            result.append({
                "seed": seed, "features": features(summary, candidate),
                "targets": {"gain": useful * 30_000, "extension": useful,
                            "survival": 1, "compute_gain": useful * 5_000},
                "budget_seconds": 2, "planner_version": "test-planner-v1", "censored": False,
            })
    return result


def test_features_use_group_counters_and_actual_available_carriers():
    original = features(board())
    assert len(original) == len(FEATURE_NAMES)
    # Highest historical chain must not substitute for live C/N counters.
    historic = board()
    historic["highest_chain"] = 999
    assert features(historic) == original
    assert features(board(chain=40)) != original
    assert features(board(num=20)) != original
    dying = board()
    dying["bodies"][0]["pending_delete"] = True
    assert features(dying)[FEATURE_NAMES.index("best_safe_carriers")] < original[FEATURE_NAMES.index("best_safe_carriers")]


def test_feature_permutation_and_numeric_identifier_invariance():
    original = board()
    original["groups"].append({"id": 21, "chain": 20, "num": 10, "members": [31]})
    original["bodies"].append({"id": 31, "color": 2, "chain_id": 21, "x": 280, "y": 220, "hits": 1})
    renamed = copy.deepcopy(original)
    renamed["seed"] = 0xFFFFFFFF
    renamed["tick"] = 123
    for body in renamed["bodies"]:
        body["id"] += 10_000
        if body.get("chain_id"):
            body["chain_id"] += 2000
    for group in renamed["groups"]:
        group["id"] += 2000
        group["members"] = list(reversed([value + 10_000 for value in group["members"]]))
    renamed["groups"].reverse()
    renamed["bodies"].reverse()
    assert features(original) == features(renamed)


def test_training_reduces_loss_with_whole_seed_holdout_and_roundtrips(tmp_path):
    model, report = train_guide(examples(), hidden=8, seed=42, epochs=120, learning_rate=.15)
    assert report["train_loss"] < report["initial_train_loss"] * .25
    assert report["validation_loss"] < report["initial_validation_loss"] * .25
    assert set(report["train_seeds"]).isdisjoint(report["validation_seeds"])
    assert len(report["train_seeds"]) + len(report["validation_seeds"]) == 8
    before = model.predict(board(32, 20), {"mode": 1, "budget_seconds": 2})
    assert before["extension_probability"] > .8
    path = tmp_path / "guide.json"
    model.save(path, {"engine_identity": "test"})
    loaded = NeuralGuide.load(path)
    assert loaded.predict(board(32, 20), {"mode": 1, "budget_seconds": 2}) == before
    assert loaded.metadata["engine_identity"] == "test"
    assert all(math.isfinite(value) for value in before.values())


def test_censored_return_is_never_trained_as_failure():
    rows = examples()
    for row in rows:
        row["censored"] = True
        row["targets"] = {"gain": -1_000_000, "compute_gain": -1_000_000}
    with pytest.raises(ValueError, match="no observed supervision"):
        NeuralGuide.create().train(rows)
    rows[0]["targets"]["extension"] = 1
    guide, report = train_guide(rows, epochs=1)
    assert report["train_examples"] == 1
    assert report["skipped_unsupervised_examples"] == len(rows) - 1
    assert report["validation_seeds"] == []
    prediction = guide.predict(board())
    assert prediction["predicted_gain"] == prediction["compute_gain"] == 0
    assert prediction["survival_probability"] == .5


def test_normalization_never_reads_holdout_and_training_is_reproducible():
    rows = examples()
    initial = NeuralGuide.create(seed=3, hidden=3)
    report = initial.train(rows, seed=9, epochs=2)
    heldout = set(report["validation_seeds"])
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["seed"] in heldout:
            row["features"] = [1e6] * len(FEATURE_NAMES)
            row["targets"]["gain"] = 1e10
    other = NeuralGuide.create(seed=3, hidden=3)
    other.train(changed, seed=9, epochs=2)
    assert initial.mean == other.mean
    assert initial.scale == other.scale
    assert initial.w1 == other.w1
    assert initial.w2 == other.w2
    reordered = NeuralGuide.create(seed=3, hidden=3)
    reordered.train(list(reversed(rows)), seed=9, epochs=2)
    assert initial.w1 == reordered.w1


def test_ranking_retains_every_candidate_and_unknown_planner_versions_reject():
    guide = NeuralGuide.create(hidden=3)
    candidates = [{"mode": i, "words": [0], "operator": "repair"} for i in range(4)]
    ranked = guide.rank(board(), candidates)
    assert sorted(row["mode"] for row in ranked) == [0, 1, 2, 3]
    assert all("neural" in row for row in ranked)
    rows = examples()
    rows[-1]["planner_version"] = "other"
    with pytest.raises(ValueError, match="different planner"):
        guide.train(rows)


def test_corrupt_or_relabelled_model_rejected(tmp_path):
    guide = NeuralGuide.create(hidden=3)
    path = tmp_path / "guide.json"
    guide.save(path)
    payload = json.loads(path.read_text())
    payload["payload"]["w1"][0][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        NeuralGuide.load(path)
    guide.metadata["planner_version"] = "original"
    with pytest.raises(ValueError, match="provenance"):
        guide.save(path, {"planner_version": "relabelled"})


def test_nonfinite_features_and_illegal_words_reject():
    broken = board()
    broken["gauge"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        features(broken)
    with pytest.raises(ValueError, match="legal"):
        features(board(), {"words": [1 << 21]})
