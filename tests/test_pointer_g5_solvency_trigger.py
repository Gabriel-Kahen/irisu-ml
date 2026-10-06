from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from irisu_pointer.g5_solvency_trigger import (
    TARGET_NAMES,
    calibration_report,
    checkpoint_envelope,
    load_checkpoint_g5,
    phase0_public_entry,
    phase0_targets,
    query_features,
    seed_partition,
    sha256,
    train_g5,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3/development/exact-phase0-oracle-pilot-train4-20260813-001/oracle-pilot.json"


def first_query() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text())["episodes"][0]["queries"][0]


def test_phase0_reconstruction_uses_strict_identity_free_g4_features() -> None:
    entry = phase0_public_entry(first_query())
    features, board = query_features(entry)
    changed = copy.deepcopy(entry)
    changed["seed"] = 123
    changed["query_id"] = "different-public-identity"
    rebound, _ = query_features(changed)
    assert features.shape == (712,)
    assert np.array_equal(features, rebound)
    assert len(board.identities) >= 1
    assert all(identity.ordinal >= 1 for identity in board.identities)


def test_phase0_features_ignore_consistently_remapped_body_ids() -> None:
    query = first_query()
    expected, _ = query_features(phase0_public_entry(query))
    changed = copy.deepcopy(query)
    for body in changed["source_public_observation"]["bodies"]:
        body["id"] += 10_000
    for row in changed["horizons"]["2048"]:
        if row["category"] != "wait":
            row["decision"]["source_body_id"] += 10_000
            row["decision"]["destination_body_id"] += 10_000
    changed["source_public_observation_sha256"] = sha256(
        changed["source_public_observation"]
    )
    actual, _ = query_features(phase0_public_entry(changed))
    assert np.array_equal(actual, expected)


def test_targets_use_all_branch_horizons_and_query_level_label() -> None:
    query = first_query()
    targets = phase0_targets(query)
    assert tuple(targets) == TARGET_NAMES
    assert targets["delayed_disagreement"] == float(
        bool(query["safe_delayed_disagreement_ordinals"])
    )
    assert all(0.0 <= value <= 1.0 for value in targets.values())


def test_terminal_target_counts_failure_exactly_at_horizon() -> None:
    query = copy.deepcopy(first_query())
    for row in query["horizons"]["2048"]:
        row["probe"]["survival_ticks"] = 2_048
        row["probe"]["terminated"] = False
        row["probe"]["truncated"] = False
    query["horizons"]["2048"][0]["probe"]["terminated"] = True
    targets = phase0_targets(query)
    assert targets["terminal_2048"] == pytest.approx(
        1 / len(query["horizons"]["2048"])
    )


def test_bonus_normalization_is_deterministic_and_source_bound() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    query = artifact["episodes"][0]["queries"][12]
    assert any(body["kind"] == "bonus" for body in query["source_public_observation"]["bodies"])
    left = phase0_public_entry(query)
    right = phase0_public_entry(copy.deepcopy(query))
    assert left == right
    normalized_bonus = [
        body for body in left["pre_query_public_observation"]["bodies"]
        if body["kind"] == "bonus"
    ]
    assert normalized_bonus
    assert all(body["color"] == -1 for body in normalized_bonus)
    tampered = copy.deepcopy(query)
    tampered["source_public_observation"]["gauge"] += 1
    with pytest.raises(ValueError, match="source observation SHA"):
        phase0_public_entry(tampered)


def test_bonus_pair_endpoint_is_retained_and_strictly_featurized() -> None:
    artifact = json.loads(
        (ROOT / "artifacts/r3/development/exact-phase0-g5-expansion-train2-20260813-001/oracle-pilot.json").read_text()
    )
    query = artifact["episodes"][0]["queries"][8]
    bonus_ids = {
        body["id"] for body in query["source_public_observation"]["bodies"]
        if body["kind"] == "bonus"
    }
    assert any(
        row["decision"].get("source_body_id") in bonus_ids
        for row in query["horizons"]["2048"]
    )
    features, board = query_features(phase0_public_entry(query))
    assert features.shape == (712,)
    assert len(board.identities) >= 1


def test_five_fold_partition_refuses_four_seed_pseudofolds() -> None:
    with pytest.raises(ValueError, match="at least six seeds"):
        seed_partition([1, 2, 3, 4])
    folds = seed_partition([6, 5, 4, 3, 2, 1])
    assert len(folds) == 5
    assert sorted(seed for fold in folds for seed in fold) == [1, 2, 3, 4, 5, 6]


def test_calibration_is_oof_high_recall_and_compute_bounded() -> None:
    report = calibration_report(
        [0.9, 0.8, 0.7, 0.6, 0.2, 0.1],
        [1, 1, 1, 0, 0, 0],
        [1, 2, 3, 4, 5, 6],
        strata=["early", "middle", "late", "early", "middle", "late"],
        minimum_recall=1.0,
        maximum_trigger_fraction=0.5,
    )
    selected = report["selected"]
    assert selected["recall"] == 1.0
    assert selected["trigger_fraction"] == 0.5
    assert selected["threshold"] == 0.7
    assert {row["stratum"] for row in selected["by_stratum"]} == {
        "early", "middle", "late"
    }


def test_expansion_plan_is_fresh_train_split() -> None:
    plan = json.loads(
        (ROOT / "configs/rl/experiments/exact-phase0-g5-expansion-train-v1.json").read_text()
    )
    assert plan["seeds"] == [406796264, 479242827]
    assert all(0 <= seed < 1 << 30 for seed in plan["seeds"])


def _tiny_model():
    generator = np.random.default_rng(1)
    seeds = np.repeat(np.arange(1, 7), 4)
    binary = np.tile([0.0, 0.0, 1.0, 1.0], 6)
    features = generator.normal(size=(len(seeds), 712))
    features[:, 0] = 10.0 * binary
    targets = {
        name: binary if name == "delayed_disagreement" else 0.2 + 0.6 * binary
        for name in TARGET_NAMES
    }
    return train_g5(
        features,
        targets,
        seeds,
        dataset_sha256="a" * 64,
        feature_inventory_sha256="b" * 64,
        provenance={"synthetic-test": "c" * 64},
        rounds=1,
    )


def test_checkpoint_loader_requires_complete_bound_expectations(tmp_path: Path) -> None:
    model, calibration = _tiny_model()
    assert set(calibration["oof_predictions"]) == set(TARGET_NAMES)
    assert set(calibration["oof_metrics"]) == set(TARGET_NAMES)
    envelope = checkpoint_envelope(model, {"limited_sample_warning": "test"})
    path = tmp_path / "g5.json"
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    loaded, report = load_checkpoint_g5(
        path,
        expected_checkpoint_sha256=envelope["checkpoint_sha256"],
        expected_model_sha256=model.sha256,
        expected_dataset_sha256=model.training_dataset_sha256,
        expected_feature_inventory_sha256=model.training_feature_inventory_sha256,
        expected_calibration_sha256=model.calibration_sha256,
        expected_provenance=dict(model.provenance),
    )
    assert loaded.sha256 == model.sha256
    assert report == {"limited_sample_warning": "test"}
    with pytest.raises(RuntimeError, match="expectation or identity"):
        load_checkpoint_g5(
            path,
            expected_checkpoint_sha256="d" * 64,
            expected_model_sha256=model.sha256,
            expected_dataset_sha256=model.training_dataset_sha256,
            expected_feature_inventory_sha256=model.training_feature_inventory_sha256,
            expected_calibration_sha256=model.calibration_sha256,
            expected_provenance=dict(model.provenance),
        )


def test_checkpoint_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(RuntimeError, match="duplicate keys"):
        load_checkpoint_g5(
            path,
            expected_checkpoint_sha256="a" * 64,
            expected_model_sha256="b" * 64,
            expected_dataset_sha256="c" * 64,
            expected_feature_inventory_sha256="d" * 64,
            expected_calibration_sha256="e" * 64,
            expected_provenance={},
        )
