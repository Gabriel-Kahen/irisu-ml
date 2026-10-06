from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_100k_promotion_audit.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_100k_promotion_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _episode(seed, score, tick, level, gauge, reason):
    return {
        "seed": seed,
        "score": score,
        "tick": tick,
        "level": level,
        "gauge": gauge,
        "terminated": True,
        "attempted_shots": 10,
        "gate_reasons": {reason: 10},
        "trace": [
            {"tick": 0, "score": 0, "gauge": 3000},
            {"tick": 10_000, "score": min(score, 10_000), "gauge": 19_000},
        ],
    }


def test_audit_counts_threshold_survival_gauge_and_rot_signature() -> None:
    report = {
        "format": "promotion-v2",
        "physics_backend": "exact",
        "promotion_report_content_sha256": "a" * 64,
        "planner_config": {"short_horizon": 256, "long_horizon": 256},
        "episodes": [
            _episode(1, 60_000, 50_000, 35, 1 - (1800 + 20 * 35), "wait-reserve"),
            _episode(2, 110_000, 75_000, 65, 1, "shot-clear"),
        ],
    }
    result = MODULE.analyze(report)
    assert result["reached_target_count"] == 1
    assert result["below_target_count"] == 1
    assert result["minimum_survival_tick_among_target_reachers"] == 75_000
    assert result["maximum_survival_tick_below_target"] == 50_000
    assert result["terminated_at_nonpositive_or_floor_gauge_count"] == 2
    assert result["terminal_rot_from_floor_seeds"] == [1]
    assert result["planner_horizons_are_identical"] is True
    assert result["survival_curve"][0]["gauge_at_or_below_half_count"] == 2


def test_audit_rejects_nonexact_empty_and_invalid_target() -> None:
    with pytest.raises(ValueError, match="exact-runtime"):
        MODULE.analyze({"physics_backend": "portable", "episodes": [{}]})
    with pytest.raises(ValueError, match="episodes"):
        MODULE.analyze({"physics_backend": "exact", "episodes": []})
    with pytest.raises(ValueError, match="positive"):
        MODULE.analyze(
            {"physics_backend": "exact", "episodes": [{}]}, target_score=0
        )
