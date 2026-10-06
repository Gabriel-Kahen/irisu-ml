from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from rl_portable_multiseed_teacher_probe import (  # noqa: E402
    ReserveObjective,
    parse_seeds,
    summarize,
)
from irisu_pointer.shot_necessity import ProbeOutcome  # noqa: E402


def outcome(*, score: int, gauge: int, ticks: int = 256) -> ProbeOutcome:
    return ProbeOutcome(ticks, score, 10, gauge, gauge, False, False)


def test_reserve_objective_rejects_small_gain_with_large_debt() -> None:
    objective = ReserveObjective(maximum_gauge_debt=1_000, rescue_score_margin=500)
    assert objective.choose(
        outcome(score=100, gauge=10_000), outcome(score=0, gauge=12_000)
    ) == (False, "wait-reserve")


def test_reserve_objective_keeps_large_score_gain() -> None:
    objective = ReserveObjective(maximum_gauge_debt=1_000, rescue_score_margin=500)
    assert objective.choose(
        outcome(score=600, gauge=10_000), outcome(score=0, gauge=12_000)
    )[0]


def test_parse_and_summary() -> None:
    assert parse_seeds("1, 0x2") == (1, 2)
    report = summarize(({"score": 40_000, "tick": 50}, {"score": 60_000, "tick": 70}))
    assert report["median_score"] == 50_000
    assert report["scores_at_least_50000"] == 1


def test_parse_rejects_empty_seed_list() -> None:
    with pytest.raises(Exception):
        parse_seeds("")
