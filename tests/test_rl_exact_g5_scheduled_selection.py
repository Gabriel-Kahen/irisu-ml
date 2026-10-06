from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rl_exact_g5_scheduled_selection",
    ROOT / "benchmarks/rl_exact_g5_scheduled_selection.py",
)
assert SPEC and SPEC.loader
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)
CONFIG = ROOT / "configs/rl/experiments/exact-g5-scheduled-paired-selection-v1.json"


def _episode(seed: int, *, score: int, tick: int = 80_000, gauge: int = 20_000,
             invalid: int = 0, failure: int = 0):
    return {
        "seed": seed,
        "score": score,
        "tick": tick,
        "gauge": gauge,
        "invalid_actions": invalid,
        "exact_stage_failures": failure,
        "exact_branch_errors": 0,
    }


def test_frozen_selection_config_loads_and_binds_scheduler() -> None:
    config = SELECTION.load_config(CONFIG)
    assert config["maximum_ticks"] == 80_000
    assert config["base_planner"]["top_k_pairs"] == 0
    assert config["scheduled_planner"]["long_horizons"] == [8_192, 12_288]
    assert config["hard_reject"]["maximum_exact_stage_failures"] == 0


def test_hard_gate_accepts_only_per_seed_nondominated_strict_improvement() -> None:
    contract = SELECTION.load_config(CONFIG)["hard_reject"]
    base = [_episode(1, score=100_000), _episode(2, score=90_000)]
    candidate = [
        _episode(1, score=101_000, gauge=19_000),
        _episode(2, score=90_000, gauge=20_000),
    ]
    report = SELECTION.hard_reject_report(
        base, candidate, contract, target_score=100_000
    )
    assert report["accepted"]
    assert not report["hard_reject_reasons"]


def test_hard_gate_rejects_score_survival_reserve_and_exact_failures() -> None:
    contract = SELECTION.load_config(CONFIG)["hard_reject"]
    base = [
        _episode(1, score=100_000),
        _episode(2, score=90_000),
        _episode(3, score=80_000),
    ]
    candidate = [
        _episode(1, score=99_999),
        _episode(2, score=90_000, tick=79_999),
        _episode(3, score=80_000, gauge=18_999, failure=1),
    ]
    report = SELECTION.hard_reject_report(
        base, candidate, contract, target_score=100_000
    )
    assert not report["accepted"]
    reasons = report["hard_reject_reasons"]
    assert "seed-1:score-regression" in reasons
    assert "seed-1:target-success-regression" in reasons
    assert "seed-2:survival-regression" in reasons
    assert "seed-3:final-gauge-regression" in reasons
    assert "seed-3:exact-stage-failure" in reasons


def test_hard_gate_rejects_noop_candidate() -> None:
    contract = SELECTION.load_config(CONFIG)["hard_reject"]
    base = [_episode(1, score=100_000)]
    report = SELECTION.hard_reject_report(
        base, list(base), contract, target_score=100_000
    )
    assert report["hard_reject_reasons"] == ["aggregate:no-strict-improvement"]


def test_runtime_core_excludes_horizon_specific_runner_config() -> None:
    common = {
        "version": "v1", "physics_backend": "exact", "identity": {"worker": "a"},
        "runtime_attestation_sha256": "b", "runtime_attestation": {"library": "c"},
    }
    left = {**common, "runner_identity": {"config_hash": 1}}
    right = {**common, "runner_identity": {"config_hash": 2}}
    assert SELECTION.runtime_core(left) == SELECTION.runtime_core(right)


def test_selection_seeds_must_be_two_unique_fresh_train_split_values() -> None:
    assert SELECTION.validate_selection_seeds(
        {"split": "train", "seeds": [10, 11]}, [1], [2]
    ) == (10, 11)
    for plan in (
        {"split": "dev", "seeds": [10, 11]},
        {"split": "train", "seeds": [10, 10]},
        {"split": "train", "seeds": [10, 2**30]},
    ):
        with pytest.raises(ValueError):
            SELECTION.validate_selection_seeds(plan, [1], [2])
    with pytest.raises(ValueError, match="overlap"):
        SELECTION.validate_selection_seeds(
            {"split": "train", "seeds": [10, 11]}, [10], [2]
        )
