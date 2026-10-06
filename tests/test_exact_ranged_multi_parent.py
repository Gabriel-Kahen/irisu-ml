from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/rl_exact_ranged_multi_parent.py"
)
SPEC = importlib.util.spec_from_file_location("rl_exact_ranged_multi_parent", MODULE)
assert SPEC and SPEC.loader
ranged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranged)


def test_validate_range_rejects_unbounded_intervals() -> None:
    ranged.validate_range(100, 200, 200)
    for values in ((-1, 10, 100), (10, 10, 100), (10, 101, 100)):
        with pytest.raises(ValueError, match="search range"):
            ranged.validate_range(*values)


def test_should_search_tick_honors_half_open_interval_and_stride() -> None:
    assert ranged.should_search_tick(
        100_352, 100_352, 102_400, 128, 0,
        include_incumbent_shots=False,
    )
    assert ranged.should_search_tick(
        100_480, 100_352, 102_400, 128, 0,
        include_incumbent_shots=False,
    )
    assert not ranged.should_search_tick(
        102_400, 100_352, 102_400, 128, 1,
        include_incumbent_shots=True,
    )


def test_incumbent_shot_search_is_explicitly_opt_in() -> None:
    assert not ranged.should_search_tick(
        100_353, 100_352, 102_400, 128, 99,
        include_incumbent_shots=False,
    )
    assert ranged.should_search_tick(
        100_353, 100_352, 102_400, 128, 99,
        include_incumbent_shots=True,
    )
