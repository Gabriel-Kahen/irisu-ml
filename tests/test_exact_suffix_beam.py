from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "benchmarks/rl_exact_suffix_beam.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_suffix_beam", MODULE)
assert SPEC and SPEC.loader
beam = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(beam)


def row(tick: int, action: int, final_tick: int, score: int, clears: int) -> dict:
    return {
        "tick": tick,
        "action": action,
        "final": {
            "tick": final_tick,
            "score": score,
            "qualifying_clear_count": clears,
            "gauge": 1,
        },
    }


def test_reconstructed_candidate_replaces_and_pads() -> None:
    candidate = row(1, 99, 6, 10, 1)
    assert beam.reconstructed_candidate([1, 2, 3], candidate) == [1, 99, 3, 0, 0, 0]


def test_reconstructed_candidate_materializes_timing_move() -> None:
    candidate = row(1, 0, 5, 10, 1)
    candidate["replacements"] = {"1": 0, "3": 77}
    assert beam.reconstructed_candidate([1, 22, 3, 0, 5], candidate) == [1, 0, 3, 77, 5]


def test_pareto_rows_keeps_score_and_survival_extremes() -> None:
    survival = row(1, 11, 100, 40, 2)
    score = row(2, 22, 90, 50, 3)
    dominated = row(3, 33, 80, 20, 1)
    selected = beam.pareto_rows([survival, score, dominated], 2)
    assert survival in selected
    assert score in selected
    assert dominated not in selected


def test_candidates_include_local_variants_of_incumbent_shot() -> None:
    incumbent = beam.encode(2, 300, 200)
    assert incumbent is not None
    selected = beam.candidates({"bodies": []}, 4, incumbent)
    assert selected == [
        beam.encode(2, 296, 200),
        beam.encode(1, 296, 200),
        beam.encode(2, 304, 200),
        beam.encode(1, 304, 200),
    ]


def test_candidates_prioritize_extending_an_active_chain() -> None:
    observation = {
        "bodies": [
            {
                "id": 1,
                "kind": "piece",
                "lifecycle": "scripted_falling",
                "chain_id": 0,
                "projectile_hits": 0,
                "color": 2,
                "x": 300.0,
                "y": 200.0,
                "size": 20.0,
            },
            {
                "id": 2,
                "kind": "piece",
                "lifecycle": "confirmed",
                "chain_id": 0,
                "color": 2,
                "x": 310.0,
                "y": 200.0,
            },
            {
                "id": 3,
                "kind": "piece",
                "lifecycle": "confirmed",
                "chain_id": 9,
                "color": 2,
                "x": 200.0,
                "y": 200.0,
            },
        ]
    }
    assert beam.candidates(observation, 1) == [beam.encode(2, 310, 215)]


def test_search_grid_always_includes_incumbent_shots() -> None:
    assert beam.should_search_tick(100, 98, 8, 12345)
    assert not beam.should_search_tick(100, 98, 8, 0)
    assert beam.should_search_tick(106, 98, 8, 0)
    assert beam.should_search_tick(101, 98, 8, 0, has_future_shot=True)


def test_timing_jitter_moves_shot_and_clears_original_tick() -> None:
    incumbent = [0] * 20
    incumbent[10] = 12345
    assert (12345, {8: 12345, 10: 0}) in beam.timing_jitter_specs(incumbent, 8)
    assert (0, {10: 0, 12: 12345}) in beam.timing_jitter_specs(incumbent, 10)


def test_terminal_snapshot_and_objective_use_recorded_score() -> None:
    observation = {
        "tick": 100,
        "score": 252_538,
        "gauge": 1,
        "level": 100,
        "highest_chain": 7,
        "qualifying_clear_count": 993,
        "terminated": 1,
        "truncated": 0,
    }
    diagnostics = {
        "terminal_metadata_recorded": True,
        "recorded_final_score": 244_158,
        "recorded_final_level": 100,
        "recorded_final_highest_chain": 7,
        "recorded_final_clears": 993,
    }
    final = beam.snapshot(observation, diagnostics)
    assert final["canonical_score"] == 244_158
    assert not beam.objective(final, True, 300_000, 250_000)[0]


def test_objective_keeps_completed_game_over_later_death() -> None:
    completed = {
        "tick": 112_633,
        "score": 252_337,
        "canonical_score": 250_242,
        "level": 100,
        "canonical_level": 100,
        "qualifying_clear_count": 990,
        "gauge": 10_601,
    }
    later_death = {
        "tick": 113_692,
        "score": 246_603,
        "canonical_score": 246_603,
        "level": 99,
        "canonical_level": 99,
        "qualifying_clear_count": 989,
        "gauge": 1,
    }
    assert beam.objective(completed, True, 300_000, 250_243) > beam.objective(
        later_death, True, 300_000, 250_243
    )


def test_objective_maximizes_score_between_completed_games() -> None:
    lower = {
        "tick": 113_000,
        "score": 250_242,
        "level": 100,
        "qualifying_clear_count": 990,
        "gauge": 12_000,
    }
    higher = {
        "tick": 112_500,
        "score": 251_000,
        "level": 100,
        "qualifying_clear_count": 990,
        "gauge": 8_000,
    }
    assert beam.objective(higher, True, 300_000, 260_000) > beam.objective(
        lower, True, 300_000, 260_000
    )


def test_objective_uses_reserve_before_timing_for_equal_score_completions() -> None:
    lower_reserve = {
        "tick": 112_746,
        "score": 250_242,
        "level": 100,
        "qualifying_clear_count": 990,
        "gauge": 691,
    }
    higher_reserve = {
        "tick": 112_633,
        "score": 250_242,
        "level": 100,
        "qualifying_clear_count": 990,
        "gauge": 10_601,
    }
    assert beam.objective(higher_reserve, True, 300_000, 260_000) > beam.objective(
        lower_reserve, True, 300_000, 260_000
    )
