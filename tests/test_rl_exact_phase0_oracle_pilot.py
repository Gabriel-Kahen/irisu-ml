from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from irisu_env import ExactWorkerError
from irisu_pointer.shot_necessity import ProbeOutcome


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rl_exact_phase0_oracle_pilot as pilot  # noqa: E402


def probe(survival: int, minimum: int, final: int, clears: int = 1, score: int = 10) -> ProbeOutcome:
    return ProbeOutcome(survival, score, clears, final, minimum, survival < 12_288, False)


def test_seed_plan_is_four_fresh_canonical_train_seeds() -> None:
    value = pilot.prior.load_seed_plan(
        ROOT / "configs/rl/experiments/exact-phase0-oracle-pilot-train-v1.json"
    )
    assert value["seeds"] == [832116684, 663099469, 494082254, 325065039]
    assert all(0 <= seed < 1 << 30 for seed in value["seeds"])


def test_strata_are_bounded_and_prioritize_low_gauge() -> None:
    assert pilot.state_stratum({"tick": 40_000, "gauge": 10_000}, {}, cap=4) == "low-gauge"
    assert pilot.state_stratum({"tick": 40_000, "gauge": 10_000}, {"low-gauge": 4}, cap=4) == "late"
    assert pilot.state_stratum({"tick": 5_000, "gauge": 20_000}, {"early": 4}, cap=4) is None


def test_delayed_robust_rule_requires_survival_or_large_reserve() -> None:
    base = probe(12_000, 100, 200)
    assert pilot.robust_better(probe(12_001, 100, 100), base, horizon=12_288)
    assert not pilot.robust_better(probe(12_001, 99, 10_000), base, horizon=12_288)
    capped = probe(12_288, 100, 200)
    assert pilot.robust_better(probe(12_288, 1_100, 1_200), capped, horizon=12_288)
    assert not pilot.robust_better(probe(12_288, 1_099, 10_000), capped, horizon=12_288)
    assert not pilot.robust_better(probe(12_288, 1_100, 1_199), capped, horizon=12_288)


def test_disagreement_must_emerge_after_short_horizon_and_persist() -> None:
    short_base = probe(2_048, 100, 200)
    short_same = probe(2_048, 100, 200)
    long8_base = probe(8_000, 100, 200)
    long8_better = probe(8_001, 100, 200)
    long12_base = probe(12_000, 100, 200)
    long12_better = probe(12_001, 100, 200)
    assert pilot.safe_delayed_disagreement(
        short_same, short_base, long8_better, long8_base, long12_better, long12_base
    )
    assert not pilot.safe_delayed_disagreement(
        short_same, short_base, long8_base, long8_base, long12_better, long12_base
    )
    assert not pilot.safe_delayed_disagreement(
        short_same, short_base, long8_better, long8_base, long12_base, long12_base
    )


def test_defaults_match_bounded_oracle_contract() -> None:
    parser = pilot.parser()
    assert pilot.HORIZONS == (2048, 8192, 12288)
    assert parser.get_default("states_per_stratum") == 4
    assert parser.get_default("maximum_ticks") == 60_000
    assert parser.get_default("expected_seed_count") == 4
    assert parser.get_default("resume") is False
    assert pilot.SCHEMA.endswith("-v2")


def test_go_gate_fails_closed_on_incomplete_seed_quota() -> None:
    episodes = [
        {
            "query_count": 16,
            "safe_delayed_disagreements": 2,
            "late_safe_delayed_disagreements": 1 if index == 0 else 0,
        }
        for index in range(4)
    ]
    assert pilot.go_decision(episodes, minimum_queries_per_seed=16)[0]
    episodes[3]["query_count"] = 15
    assert not pilot.go_decision(episodes, minimum_queries_per_seed=16)[0]


def test_source_has_no_pilot_seed_constants() -> None:
    source = (ROOT / "benchmarks/rl_exact_phase0_oracle_pilot.py").read_text()
    for seed in (832116684, 663099469, 494082254, 325065039):
        assert str(seed) not in source


class _HashEnv:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_hash(self) -> int:
        return self.value


def test_exact_worker_recovery_requires_unchanged_parent_and_is_stage_specific() -> None:
    error = ExactWorkerError("forked child failed")
    live = pilot.unchanged_parent_error_event(
        _HashEnv(7), {"tick": 42}, 7, stage="live-planner", error=error
    )
    assert live["parent_state_unchanged"]
    assert live["recovery"] == "restore-preprediction-policy-and-execute-wait"
    query = pilot.unchanged_parent_error_event(
        _HashEnv(7), {"tick": 42}, 7, stage="oracle-query", error=error
    )
    assert query["recovery"] == "no-label-then-continue-live-planner"
    with pytest.raises(RuntimeError, match="altered live parent"):
        pilot.unchanged_parent_error_event(
            _HashEnv(8), {"tick": 42}, 7, stage="live-planner", error=error
        )


def test_durable_seed_roundtrip_binds_content_and_rejects_tamper(tmp_path: Path) -> None:
    binding = {"seed": 123, "source": "a" * 64}
    episode = {"seed": 123, "queries": [], "wall_seconds": 99.0}
    envelope = pilot.durable_seed_envelope(binding, episode)
    path = tmp_path / "seed.json"
    pilot.gate._write_json_new(path, envelope)
    assert pilot.load_durable_seed(path, binding) == {"seed": 123, "queries": []}

    tampered = json.loads(path.read_text())
    tampered["episode"]["seed"] = 124
    tampered_path = tmp_path / "tampered.json"
    pilot.gate._write_json_new(tampered_path, tampered)
    with pytest.raises(ValueError, match="content hash"):
        pilot.load_durable_seed(tampered_path, binding)


def test_collector_catches_only_exact_worker_errors_for_recovery() -> None:
    source = (ROOT / "benchmarks/rl_exact_phase0_oracle_pilot.py").read_text()
    assert source.count("except ExactWorkerError as error:") == 2
    assert "except Exception" not in source
