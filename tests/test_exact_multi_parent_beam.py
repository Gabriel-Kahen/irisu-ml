from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "benchmarks/rl_exact_multi_parent_beam.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_multi_parent_beam", MODULE)
assert SPEC and SPEC.loader
beam = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(beam)


def parent(score: int, gauge: int, trace: tuple[int, ...], chain: int = 7):
    return beam.Parent(
        trace,
        {
            "tick": len(trace),
            "score": score,
            "canonical_score": score,
            "level": 100,
            "canonical_level": 100,
            "highest_chain": chain,
            "canonical_highest_chain": chain,
            "qualifying_clear_count": 990,
            "gauge": gauge,
        },
        True,
    )


def test_beam_select_keeps_distinct_score_chain_reserve_buckets() -> None:
    top = parent(251_492, 8_061, (1, 2), 8)
    duplicate_bucket = parent(251_492, 8_050, (1, 3), 8)
    alternate = parent(250_839, 691, (1, 4), 7)
    selected = beam.beam_select([duplicate_bucket, alternate, top], 2, 120_000, 300_000)
    assert selected == [top, alternate]


def test_branch_specs_omits_noop_and_deduplicates(monkeypatch) -> None:
    monkeypatch.setattr(beam.suffix, "candidates", lambda *_args: [11, 12, 12])
    monkeypatch.setattr(beam.suffix, "timing_jitter_specs", lambda *_args: [])
    assert beam.branch_specs({}, (10, 11, 0), 1, 8, 4) == [
        (12, {1: 12}),
        (0, {1: 0}),
    ]
