from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_reserve_band_comparator.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_exact_reserve_band_comparator", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


CONFIG = MODULE.ReserveBandConfig(
    horizon_ticks=256,
    gauge_max=40_000,
    contingency_gauge=3_000,
)


def candidate(index: int, **changes):
    values = {
        "candidate_index": index,
        "survival_ticks": 256,
        "terminated": False,
        "truncated": False,
        "minimum_gauge": 8_000,
        "final_gauge": 10_000,
        "imminent_visible_rot_liability": 2_000,
        "renewal_clears": 1,
        "score": 100,
        "invalid_actions": 0,
    }
    values.update(changes)
    return MODULE.ProbeCandidate(**values)


def winner(*values):
    return MODULE.choose_candidate(values, CONFIG)[0].candidate.candidate_index


def test_survival_and_validity_dominate_reserve_and_score() -> None:
    safe = candidate(0, minimum_gauge=100, final_gauge=100, score=0)
    failed = candidate(
        1,
        survival_ticks=255,
        terminated=True,
        minimum_gauge=20_000,
        final_gauge=20_000,
        score=1_000_000,
    )
    assert winner(safe, failed) == 0
    invalid = candidate(2, invalid_actions=1, score=2_000_000)
    assert winner(safe, invalid) == 0


def test_insolvent_candidates_rank_minimum_then_final_net_gauge() -> None:
    low_minimum = candidate(
        0, minimum_gauge=2_500, final_gauge=10_000, imminent_visible_rot_liability=0
    )
    high_minimum = candidate(
        1, minimum_gauge=2_800, final_gauge=3_500, imminent_visible_rot_liability=0
    )
    assert winner(low_minimum, high_minimum) == 1

    low_final = candidate(
        2, minimum_gauge=2_800, final_gauge=2_900, imminent_visible_rot_liability=0
    )
    assert winner(high_minimum, low_final) == 1


def test_visible_rot_liability_is_subtracted_and_breaks_band_ties() -> None:
    debt = candidate(
        0,
        minimum_gauge=8_000,
        final_gauge=8_000,
        imminent_visible_rot_liability=4_000,
    )
    clear = candidate(
        1,
        minimum_gauge=8_000,
        final_gauge=8_000,
        imminent_visible_rot_liability=1_000,
    )
    assert winner(debt, clear) == 1


def test_funded_reserve_is_capped_then_renewal_clears_precede_score() -> None:
    hoard = candidate(
        0,
        minimum_gauge=40_000,
        final_gauge=40_000,
        imminent_visible_rot_liability=0,
        renewal_clears=1,
        score=1_000_000,
    )
    renewable = candidate(
        1,
        minimum_gauge=5_000,
        final_gauge=5_000,
        imminent_visible_rot_liability=0,
        renewal_clears=2,
        score=0,
    )
    assert winner(hoard, renewable) == 1
    ranked = MODULE.rank_candidate(hoard, CONFIG)
    assert ranked.net_minimum_gauge == CONFIG.useful_gauge_ceiling
    assert ranked.rank[4] == CONFIG.contingency_gauge


def test_score_is_last_utility_term_and_index_is_deterministic_tie_break() -> None:
    lower_score = candidate(0, score=100)
    higher_score = candidate(1, score=101)
    assert winner(lower_score, higher_score) == 1
    identical_later = candidate(2)
    assert winner(lower_score, identical_later) == 0


def test_manifest_is_development_only_and_supports_many_candidates() -> None:
    value = {
        "config": {
            "horizon_ticks": 256,
            "gauge_max": 40_000,
            "contingency_gauge": 3_000,
        },
        "candidates": [
            candidate(0).manifest(),
            candidate(1, renewal_clears=2).manifest(),
            candidate(2, score=999).manifest(),
        ],
    }
    report = MODULE.evaluate_manifest(value)
    assert report["development_only"] is True
    assert report["promotion_eligible"] is False
    assert report["winner_candidate_index"] == 1
    assert report["rank_order"] == [1, 2, 0]
    assert report["config"]["useful_gauge_ceiling"] == 20_000


def test_rejects_empty_duplicate_and_out_of_horizon_candidates() -> None:
    with pytest.raises(ValueError, match="requires candidates"):
        MODULE.choose_candidate([], CONFIG)
    with pytest.raises(ValueError, match="indices"):
        MODULE.choose_candidate([candidate(0), candidate(0)], CONFIG)
    with pytest.raises(ValueError, match="exceeds"):
        MODULE.rank_candidate(candidate(0, survival_ticks=257), CONFIG)

