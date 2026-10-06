from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from irisu_pointer.g5_solvency_trigger import sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rl_g5_hazard_shadow_audit",
    ROOT / "benchmarks/rl_g5_hazard_shadow_audit.py",
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
CHECKPOINT = (
    ROOT / "artifacts/r3/development/"
    "g5-hazard-scheduler-motion-size-train6-20260814-001/checkpoint.json"
)


def _oracle(seed: int) -> dict[str, object]:
    trigger = {"bodies": []}
    skip = {
        "bodies": [
            {"vx": 20.0, "vy": 0.0, "size": 60.0},
            {"vx": -20.0, "vy": 0.0, "size": 60.0},
        ]
    }
    queries = [
        {
            "seed": seed, "query_index": 0, "stratum": "late",
            "safe_delayed_disagreement_ordinals": [1],
            "source_public_observation": trigger, "candidate_count": 3,
        },
        {
            "seed": seed, "query_index": 1, "stratum": "early",
            "safe_delayed_disagreement_ordinals": [],
            "source_public_observation": skip, "candidate_count": 3,
        },
    ]
    value = {
        "seed_plan": {"split": "train"},
        "pilot_seeds": [seed],
        "episodes": [{"queries": queries}],
    }
    value["content_sha256"] = sha256(value)
    return value


def test_shadow_audit_requires_fresh_train_seeds_and_passes_recall_gate() -> None:
    scheduler, _checkpoint = AUDIT.load_scheduler(CHECKPOINT)
    report = AUDIT.audit(scheduler, _oracle(123_456_789))
    assert report["pass"]
    assert report["recall"] == 1.0
    assert report["compute_fraction"] == 0.5
    with pytest.raises(ValueError, match="overlap"):
        AUDIT.audit(scheduler, _oracle(scheduler.training_seeds[0]))


def test_shadow_audit_rejects_content_tamper() -> None:
    scheduler, _checkpoint = AUDIT.load_scheduler(CHECKPOINT)
    oracle = _oracle(123_456_789)
    oracle["pilot_seeds"] = [987_654_321]
    with pytest.raises(ValueError, match="content hash"):
        AUDIT.audit(scheduler, oracle)
