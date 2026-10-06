"""Deterministic trajectory gates for expensive exact seed searches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrajectoryGate:
    tick: int
    minimum_level: int
    minimum_projected_score: int
    minimum_delta_20k: int = 0
    minimum_gauge: int = 0

    def manifest(self) -> dict[str, int]:
        return {
            "tick": self.tick,
            "minimum_level": self.minimum_level,
            "minimum_projected_score": self.minimum_projected_score,
            "minimum_delta_20k": self.minimum_delta_20k,
            "minimum_gauge": self.minimum_gauge,
        }


@dataclass(frozen=True)
class CollapseRiskRule:
    """Conservative secondary gate for volatile, low-yield reserve conversion."""

    ticks: tuple[int, ...] = (50_000, 60_000, 70_000, 80_000)
    minimum_interval_gauge: int = 5_000
    minimum_endpoint_gauge: int = 15_000
    maximum_score_gain_10k_exclusive: int = 35_000
    gauge_drop_threshold: int = -3_000
    gauge_rise_threshold: int = 5_000

    def manifest(self) -> dict[str, object]:
        return {
            "ticks": list(self.ticks),
            "minimum_interval_gauge": self.minimum_interval_gauge,
            "minimum_endpoint_gauge": self.minimum_endpoint_gauge,
            "maximum_score_gain_10k_exclusive": (
                self.maximum_score_gain_10k_exclusive
            ),
            "gauge_drop_threshold": self.gauge_drop_threshold,
            "gauge_rise_threshold": self.gauge_rise_threshold,
        }


DEFAULT_GATES = (
    TrajectoryGate(50_000, 40, 140_000),
    TrajectoryGate(60_000, 52, 180_000, minimum_delta_20k=48_000),
    TrajectoryGate(70_000, 60, 190_000, minimum_delta_20k=50_000),
    TrajectoryGate(80_000, 70, 205_000, minimum_gauge=2_000),
)

# Target-300k screening profile.  It deliberately prunes some 200k-class
# trajectories and is an explicit heuristic until the campaign produces its
# first >=300k positive.
TARGET_300_GATES = (
    TrajectoryGate(50_000, 44, 150_000),
    TrajectoryGate(60_000, 56, 200_000, minimum_delta_20k=50_000),
    TrajectoryGate(
        70_000,
        64,
        205_000,
        minimum_delta_20k=55_000,
        minimum_gauge=3_000,
    ),
    TrajectoryGate(
        80_000,
        73,
        220_000,
        minimum_delta_20k=57_000,
        minimum_gauge=5_000,
    ),
)

TARGET_300_COLLAPSE_RISK = CollapseRiskRule()

GATE_PROFILES = {
    "conservative-200": DEFAULT_GATES,
    "target-300": TARGET_300_GATES,
}

# Frozen 2026-08-13 replay-native campaign calibration.  All three known
# >=200k seeds pass every gate; gauge is intentionally absent at 70k because
# seed 4000000005 recovered from gauge=1 to finish at 229023.
CALIBRATION = {
    "replay_native_seed_count": 45,
    "positive_threshold": 200_000,
    "positive_seeds": [
        4_000_000_005,
        4_000_000_026,
        4_000_000_027,
        4_000_000_033,
    ],
    "positive_scores": [229_023, 201_873, 215_359, 206_633],
    "observed_positive_false_negatives": 0,
}

TARGET_300_CALIBRATION = {
    "replay_native_seed_count": 45,
    "positive_threshold": 300_000,
    "observed_positive_count": 0,
    "observed_positive_false_negatives": None,
    "heuristic": True,
}

GATE_CALIBRATIONS = {
    "conservative-200": CALIBRATION,
    "target-300": TARGET_300_CALIBRATION,
}


def gate_manifest(gates: Sequence[TrajectoryGate]) -> dict[str, object]:
    rows = [gate.manifest() for gate in gates]
    return {"gates": rows, "gates_sha256": canonical_sha256(rows)}


def collapse_risk_manifest(
    rule: CollapseRiskRule | None,
) -> dict[str, object]:
    row = None if rule is None else rule.manifest()
    return {
        "collapse_risk_rule": row,
        "collapse_risk_rule_sha256": canonical_sha256(row),
    }


def evaluate_collapse_risk(
    rule: CollapseRiskRule,
    observation: Mapping[str, object],
    checkpoint_scores: Mapping[int, int],
    checkpoint_gauges: Mapping[int, int],
    interval_minimum_gauge: int,
) -> dict[str, object]:
    """Return a fail-closed verdict for the calibrated volatility pattern."""

    tick = int(observation["tick"])
    if tick not in rule.ticks:
        raise ValueError(f"collapse-risk rule is not configured at tick {tick}")
    prior_tick = tick - 10_000
    if prior_tick not in checkpoint_scores or prior_tick not in checkpoint_gauges:
        raise ValueError(f"missing 10k checkpoint before tick {tick}")
    score_gain_10k = int(observation["score"]) - checkpoint_scores[prior_tick]
    gauge = int(observation.get("gauge", 0))
    gauge_delta_10k = gauge - checkpoint_gauges[prior_tick]
    volatile = (
        gauge_delta_10k <= rule.gauge_drop_threshold
        or gauge_delta_10k >= rule.gauge_rise_threshold
    )
    risky = (
        interval_minimum_gauge >= rule.minimum_interval_gauge
        and gauge >= rule.minimum_endpoint_gauge
        and score_gain_10k < rule.maximum_score_gain_10k_exclusive
        and volatile
    )
    return {
        "rule": rule.manifest(),
        "values": {
            "interval_minimum_gauge": interval_minimum_gauge,
            "endpoint_gauge": gauge,
            "score_gain_10k": score_gain_10k,
            "gauge_delta_10k": gauge_delta_10k,
        },
        "risk_detected": risky,
        "passed": not risky,
    }


def evaluate_gate(
    gate: TrajectoryGate,
    observation: Mapping[str, object],
    checkpoint_scores: Mapping[int, int],
    *,
    terminal_tick: int = 100_000,
) -> dict[str, object]:
    tick = int(observation["tick"])
    if tick != gate.tick:
        raise ValueError(f"gate {gate.tick} evaluated at tick {tick}")
    prior_tick = tick - 20_000
    if prior_tick not in checkpoint_scores:
        raise ValueError(f"missing score checkpoint at tick {prior_tick}")
    score = int(observation["score"])
    delta_20k = score - int(checkpoint_scores[prior_tick])
    remaining = max(0, terminal_tick - tick)
    projected_score = score + (remaining * delta_20k) // 20_000
    values = {
        "level": int(observation.get("level", 0)),
        "projected_score": projected_score,
        "delta_20k": delta_20k,
        "gauge": int(observation.get("gauge", 0)),
    }
    checks = {
        "level": values["level"] >= gate.minimum_level,
        "projected_score": projected_score >= gate.minimum_projected_score,
        "delta_20k": delta_20k >= gate.minimum_delta_20k,
        "gauge": values["gauge"] >= gate.minimum_gauge,
    }
    return {
        "gate": gate.manifest(),
        "values": values,
        "checks": checks,
        "passed": all(checks.values()),
    }
