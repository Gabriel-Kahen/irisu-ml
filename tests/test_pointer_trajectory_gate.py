from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from irisu_pointer.trajectory_gate import (
    CollapseRiskRule,
    TARGET_300_COLLAPSE_RISK,
    DEFAULT_GATES,
    GATE_CALIBRATIONS,
    TARGET_300_GATES,
    TrajectoryGate,
    evaluate_collapse_risk,
    evaluate_gate,
    gate_manifest,
)


class _OneStepEnvironment:
    def step(self, _action):
        return _observation(20_000, 1, 1, 1), 0, False, False, {}


def _observation(tick: int, score: int, level: int, gauge: int):
    return {"tick": tick, "score": score, "level": level, "gauge": gauge}


def test_manifest_is_hash_bound_and_stable() -> None:
    first = gate_manifest(DEFAULT_GATES)
    second = gate_manifest(DEFAULT_GATES)
    assert first == second
    assert len(first["gates_sha256"]) == 64


def test_seed_five_recovery_trajectory_passes_without_70k_gauge_gate() -> None:
    scores = {50_000: 66_462}
    verdict = evaluate_gate(
        DEFAULT_GATES[2],
        _observation(70_000, 120_517, 63, 1),
        scores,
    )
    assert verdict["passed"] is True
    assert verdict["values"]["projected_score"] == 201_599


def test_weak_60k_trajectory_is_rejected() -> None:
    verdict = evaluate_gate(
        DEFAULT_GATES[1],
        _observation(60_000, 80_000, 50, 12_000),
        {40_000: 42_000},
    )
    assert verdict["passed"] is False
    assert verdict["checks"]["level"] is False
    assert verdict["checks"]["delta_20k"] is False


def test_seed_26_low_reserve_80k_checkpoint_is_retained() -> None:
    verdict = evaluate_gate(
        DEFAULT_GATES[3],
        _observation(80_000, 168_336, 77, 2_673),
        {60_000: 105_137},
    )
    assert verdict["passed"] is True


def test_target_300_profile_is_manifest_distinct_and_stricter() -> None:
    assert gate_manifest(TARGET_300_GATES) != gate_manifest(DEFAULT_GATES)
    assert TARGET_300_GATES[0].minimum_level == 44
    assert TARGET_300_GATES[1].minimum_projected_score == 200_000
    assert TARGET_300_GATES[3].minimum_delta_20k == 57_000
    assert TARGET_300_GATES[3].minimum_gauge == 5_000
    assert GATE_CALIBRATIONS["target-300"]["heuristic"] is True
    assert GATE_CALIBRATIONS["target-300"]["observed_positive_count"] == 0


def test_target_300_80k_gate_rejects_seed_151_collapse_state() -> None:
    verdict = evaluate_gate(
        TARGET_300_GATES[3],
        _observation(80_000, 198_439, 85, 4_430),
        {60_000: 126_093},
    )
    assert verdict["checks"]["projected_score"] is True
    assert verdict["checks"]["gauge"] is False
    assert verdict["passed"] is False


def test_collapse_risk_flags_volatile_low_yield_conversion() -> None:
    verdict = evaluate_collapse_risk(
        TARGET_300_COLLAPSE_RISK,
        _observation(60_000, 104_964, 60, 22_578),
        {50_000: 72_297},
        {50_000: 14_018},
        8_376,
    )
    assert verdict["risk_detected"] is True
    assert verdict["passed"] is False


def test_collapse_risk_retains_high_yield_checkpoint() -> None:
    verdict = evaluate_collapse_risk(
        TARGET_300_COLLAPSE_RISK,
        _observation(60_000, 125_480, 62, 10_855),
        {50_000: 82_190},
        {50_000: 22_362},
        7_000,
    )
    assert verdict["risk_detected"] is False
    assert verdict["passed"] is True


def test_gate_requires_exact_tick_and_prior_checkpoint() -> None:
    try:
        evaluate_gate(DEFAULT_GATES[0], _observation(50_001, 1, 1, 1), {})
    except ValueError as exc:
        assert "evaluated at tick" in str(exc)
    else:
        raise AssertionError("off-tick evaluation did not fail closed")


def test_gated_environment_persists_rejected_prefix(tmp_path, monkeypatch) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/rl_exact_trajectory_gated_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rl_exact_trajectory_gated_capture_test", source
    )
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = capture
    spec.loader.exec_module(capture)

    gate = TrajectoryGate(20_000, 2, 10)
    words = []
    environment = capture.GatedEnvironment(
        _OneStepEnvironment(),
        words,
        0,
        tmp_path / "progress.json",
        {"policy_sha256": "a" * 64},
        "enforce",
        (gate,),
    )
    environment.checkpoint_scores[0] = 0
    try:
        environment.step(capture.Action.wait(1))
    except capture.WeakSeedStop as stop:
        assert stop.verdict["passed"] is False
    else:
        raise AssertionError("weak trajectory was not stopped")
    assert words == [0]
    assert (tmp_path / "progress.json").is_file()


def test_gated_environment_can_shadow_collapse_risk(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/rl_exact_trajectory_gated_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rl_exact_trajectory_gated_capture_shadow_test", source
    )
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = capture
    spec.loader.exec_module(capture)

    gate = TrajectoryGate(20_000, 0, 0)
    rule = CollapseRiskRule(ticks=(20_000,))
    environment = capture.GatedEnvironment(
        _OneStepEnvironment(),
        [],
        0,
        tmp_path / "shadow-progress.json",
        {"policy_sha256": "b" * 64},
        "enforce",
        (gate,),
        rule,
        False,
    )
    environment.checkpoint_scores.update({0: 0, 10_000: 0})
    environment.checkpoint_gauges[10_000] = 10_000
    environment.interval_minimum_gauge = 6_000
    environment.step(capture.Action.wait(1))
    progress = __import__("json").loads(
        (tmp_path / "shadow-progress.json").read_text()
    )
    assert progress["gate_verdict"]["passed"] is True
    assert progress["gate_verdict"]["collapse_risk_enforced"] is False
