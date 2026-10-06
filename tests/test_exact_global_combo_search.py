from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "benchmarks/rl_exact_global_combo_search.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_global_combo_search", MODULE)
assert SPEC and SPEC.loader
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


def test_materialize_applies_delayed_shot() -> None:
    assert search.materialize([0, 55, 0, 0], {1: 0, 3: 55}, 5) == [0, 0, 0, 55, 0]


def test_combo_power_rewards_larger_chains() -> None:
    assert search.combo_power([(10, 1), (12, 3), (20, 5)], 9, 12) == 10


def test_select_opportunities_excludes_endgame_and_finds_preclear_waits() -> None:
    values = [0] * 100
    values[10] = 99
    values[95] = 88
    rows = search.select_opportunities(
        values,
        [(20, 1), (98, 1)],
        lookahead=20,
        cap=100,
        excluded_suffix=10,
    )
    identities = {(row["kind"], row["tick"]) for row in rows}
    assert ("shot", 10) in identities
    assert ("shot", 95) not in identities
    assert any(kind == "inject" for kind, _tick in identities)


def test_select_opportunities_respects_replay_native_startup_floor() -> None:
    values = [0, 0, 99, 0, 0]
    rows = search.select_opportunities(
        values,
        [(4, 1)],
        lookahead=10,
        cap=20,
        excluded_suffix=0,
        minimum_tick=2,
    )
    assert rows
    assert all(int(row["tick"]) >= 2 for row in rows)


def test_local_key_prefers_combo_gain_before_raw_score() -> None:
    combo = {
        "terminated": False,
        "combo_power_gain": 8,
        "max_chain": 4,
        "score_gain": 1,
        "clear_gain": 0,
        "gauge_gain": 0,
    }
    score = {
        "terminated": False,
        "combo_power_gain": 0,
        "max_chain": 2,
        "score_gain": 10_000,
        "clear_gain": 5,
        "gauge_gain": 5,
    }
    assert search.local_key(combo) > search.local_key(score)


def test_diverse_ticks_preserves_multiple_game_bands() -> None:
    ranked = [
        {"tick": 10, "rank": 1},
        {"tick": 20, "rank": 2},
        {"tick": 10_010, "rank": 3},
        {"tick": 20_010, "rank": 4},
    ]
    assert {row["tick"] // 10_000 for row in search.diverse_ticks(ranked, 3)} == {
        0,
        1,
        2,
    }


def test_final_key_rejects_late_death_over_completed_record() -> None:
    completed = {
        "canonical_score": 251_492,
        "canonical_level": 100,
        "canonical_highest_chain": 8,
        "canonical_clears": 990,
        "score": 253_000,
        "level": 100,
        "highest_chain": 8,
        "qualifying_clear_count": 990,
        "gauge": 8_000,
        "tick": 112_000,
    }
    death = {
        "score": 280_000,
        "level": 99,
        "highest_chain": 12,
        "qualifying_clear_count": 989,
        "gauge": 1,
        "tick": 120_000,
    }
    assert search.final_key(completed, True) > search.final_key(death, True)
