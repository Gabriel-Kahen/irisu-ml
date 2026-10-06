from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "benchmarks/rl_exact_combo_survival_repair.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_combo_survival_repair", MODULE)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def local(tick: int, combo: int, score: int, terminated: bool = False) -> dict:
    return {
        "kind": "delay-2",
        "tick": tick,
        "terminated": terminated,
        "combo_power_gain": combo,
        "score_gain": score,
        "max_chain": combo,
        "clear_gain": 0,
        "gauge_gain": 0,
        "replacements": {str(tick): 0},
    }


def test_select_global_mutation_uses_combo_objective() -> None:
    assert repair.select_global_mutation([local(10, 3, 900), local(20, 7, 20)])["tick"] == 20


def test_select_global_mutation_rejects_local_death() -> None:
    assert repair.select_global_mutation([local(10, 9, 900, True), local(20, 2, 20)])["tick"] == 20


def test_select_global_mutation_respects_tick_range() -> None:
    rows = [local(10, 9, 900), local(20, 7, 20), local(30, 2, 40)]
    assert repair.select_global_mutation(rows, minimum_tick=15, maximum_tick=25)["tick"] == 20


def test_repair_continues_only_on_survival_advance() -> None:
    before = {"tick": 100, "level": 20}
    assert repair.repair_should_continue(before, {"tick": 101, "level": 20}, True)
    assert not repair.repair_should_continue(before, {"tick": 100, "level": 20}, True)
    assert not repair.repair_should_continue(before, {"tick": 110, "level": 100}, True)
