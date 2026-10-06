from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from rl_portable_checkpoint_eval import (  # noqa: E402
    MODES,
    parse_seeds,
    reserve_choice,
    summarize,
)
from irisu_pointer.shot_necessity import ProbeOutcome  # noqa: E402


def outcome(*, score: int, gauge: int, ticks: int = 128) -> ProbeOutcome:
    return ProbeOutcome(ticks, score, 1, gauge, gauge, False, False)


def test_seed_parser_requires_unique_caller_supplied_uint32_values() -> None:
    assert parse_seeds("1, 0x2, 4294967295") == (1, 2, 0xFFFF_FFFF)
    with pytest.raises(Exception):
        parse_seeds("")
    with pytest.raises(Exception):
        parse_seeds("1,1")
    with pytest.raises(Exception):
        parse_seeds("4294967296")


def test_modes_cover_raw_model_and_adaptive_wait_gate() -> None:
    assert MODES == ("model-only", "adaptive-wait-gate")


def test_reserve_choice_rejects_small_gain_bought_with_gauge() -> None:
    assert reserve_choice(
        outcome(score=100, gauge=10_000),
        outcome(score=0, gauge=12_000),
        maximum_gauge_debt=1_000,
        rescue_score_margin=500,
    ) == (False, "wait-reserve")


def test_summary_reports_target_rate_and_invalid_actions() -> None:
    report = summarize(
        (
            {"score": 40_000, "tick": 50, "invalid_actions": 1},
            {"score": 60_000, "tick": 70, "invalid_actions": 0},
        )
    )
    assert report["median_score"] == 50_000
    assert report["success_fraction"] == 0.5
    assert report["invalid_actions"] == 1


def test_evaluator_does_not_embed_acceptance_seeds() -> None:
    source = (ROOT / "benchmarks/rl_portable_checkpoint_eval.py").read_text()
    assert "CALIBRATION_SEEDS" not in source
    assert "1439993096" not in source
