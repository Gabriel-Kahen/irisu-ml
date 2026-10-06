from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_reactive_distill.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_reactive_distill", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def observation() -> dict[str, object]:
    return {
        "tick": 10,
        "score": 50,
        "gauge": 2_000,
        "gauge_max": 40_000,
        "level": 2,
        "highest_chain": 3,
        "qualifying_clear_count": 4,
        "terminated": 0,
        "truncated": 0,
        "left_held": 0,
        "right_held": 0,
        "difficulty": {"active_colors": 3, "spawn_interval_ticks": 80},
        "field": {"x": 130.0, "y": 120.0, "width": 320.0, "height": 250.0},
        "bodies": [
            {
                "id": 7,
                "age_ticks": 20,
                "kind": "piece",
                "shape": "circle",
                "lifecycle": "fresh",
                "color": 1,
                "x": 200.0,
                "y": 300.0,
                "vx": 1.0,
                "vy": 2.0,
                "size": 32.0,
                "chain_id": 0,
            },
            {
                "id": 8,
                "age_ticks": 11,
                "kind": "piece",
                "shape": "box",
                "lifecycle": "confirmed",
                "color": 1,
                "x": 250.0,
                "y": 340.0,
                "vx": 0.0,
                "vy": 0.0,
                "size": 46.0,
                "chain_id": 0,
            },
        ],
    }


def test_digest_excludes_episode_clock_and_identity() -> None:
    first = observation()
    second = copy.deepcopy(first)
    second.update(
        tick=99_999,
        score=123_456,
        highest_chain=20,
        qualifying_clear_count=999,
    )
    second["bodies"].reverse()  # type: ignore[union-attr]
    for index, body in enumerate(second["bodies"]):  # type: ignore[union-attr]
        body["id"] = 100 + index
        body["age_ticks"] = 500 + index
    assert MODULE.state_digest(first) == MODULE.state_digest(second)


def test_digest_changes_with_physical_state() -> None:
    first = observation()
    second = copy.deepcopy(first)
    second["bodies"][0]["x"] += 0.25  # type: ignore[index]
    assert MODULE.state_digest(first) != MODULE.state_digest(second)


def test_features_exclude_tick_and_seed_proxies() -> None:
    first = observation()
    second = copy.deepcopy(first)
    second.update(tick=1_000, score=90_000, highest_chain=99)
    assert (MODULE.features(first) == MODULE.features(second)).all()


def test_features_encode_combo_group_topology_without_chain_identity() -> None:
    joined = observation()
    joined["bodies"][0]["chain_id"] = 7  # type: ignore[index]
    joined["bodies"][1]["chain_id"] = 7  # type: ignore[index]

    relabeled = copy.deepcopy(joined)
    relabeled["bodies"].reverse()  # type: ignore[union-attr]
    for body in relabeled["bodies"]:  # type: ignore[union-attr]
        body["chain_id"] = 91
    assert (MODULE.features(joined) == MODULE.features(relabeled)).all()

    split = copy.deepcopy(joined)
    split["bodies"][1]["chain_id"] = 8  # type: ignore[index]
    assert not (MODULE.features(joined) == MODULE.features(split)).all()


def test_features_encode_projectile_clear_readiness() -> None:
    untouched = observation()
    touched = copy.deepcopy(untouched)
    touched["bodies"][0]["projectile_hits"] = 2  # type: ignore[index]
    assert not (MODULE.features(untouched) == MODULE.features(touched)).all()


def test_legacy_feature_width_remains_loadable() -> None:
    assert len(MODULE.features(observation(), combo_aware=False)) == 218
    assert len(MODULE.features(observation())) == 226


def test_reactive_vote_threshold_is_bounded_and_configurable() -> None:
    value = {
        "state_actions": {},
        "mean": np.zeros(226, dtype=np.float32),
        "scale": np.ones(226, dtype=np.float32),
        "exemplar_features": np.zeros((7, 226), dtype=np.float32),
        "exemplar_words": np.zeros(7, dtype=np.uint32),
    }
    assert MODULE.ReactiveExactPolicy(
        value, minimum_shot_neighbors=2
    ).minimum_shot_neighbors == 2
    with pytest.raises(ValueError, match="minimum_shot_neighbors"):
        MODULE.ReactiveExactPolicy(value, minimum_shot_neighbors=0)


def test_canonical_checkpoint_prefers_recorded_finish_metadata() -> None:
    value = observation()
    value.update(terminated=1, score=252_538, level=100, highest_chain=7)
    result = MODULE.canonical_checkpoint(
        value,
        {
            "diagnostics": {
                "terminal_metadata_recorded": True,
                "recorded_final_score": 244_158,
                "recorded_final_level": 100,
                "recorded_final_highest_chain": 7,
            }
        },
    )
    assert result["score"] == 244_158
    assert result["terminal_metadata_recorded"] is True
