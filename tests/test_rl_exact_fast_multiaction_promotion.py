from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEV_TEST = _load(
    "test_rl_exact_fast_multiaction_eval_support",
    "../tests/test_rl_exact_fast_multiaction_eval.py",
)
DEV = sys.modules["rl_exact_fast_multiaction_eval"]
PROMOTION = _load(
    "rl_exact_fast_multiaction_promotion_eval",
    "rl_exact_fast_multiaction_promotion_eval.py",
)
MERGE = _load(
    "rl_exact_fast_multiaction_promotion_merge",
    "rl_exact_fast_multiaction_promotion_merge.py",
)


def _config():
    return DEV.FastMultiActionConfig(
        probe_ticks=2,
        long_probe_ticks=2,
        wait_ticks=1,
        low_gauge_threshold=0,
        top_k_pairs=0,
        maximum_gauge_debt=100,
        rescue_score_margin=5,
        gauge_advantage=1,
    )


def _report(seeds):
    return PROMOTION.evaluate_promotion_shard(
        DEV_TEST._Runtime(),
        DEV_TEST._bundle(),
        declared_training_seeds=[1],
        evaluation_seeds=seeds,
        maximum_ticks=2,
        planner_config=_config(),
        trace_interval_ticks=1,
        maximum_logged_queries=1,
    )


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_promotion_shard_has_fixed_100k_contract_and_bound_engine() -> None:
    report = _report([2])

    assert report["promotion_contract"] == {
        "required_episode_count": 20,
        "required_success_count": 16,
        "target_score": 100_000,
        "require_median_at_or_above_target": True,
        "require_zero_live_invalid_actions": True,
    }
    assert report["promotion_eligible"] is False
    assert report["passed"] is False
    assert report["branch_error_recovery_enabled"] is True
    assert len(report["planner_source_sha256"]) == 64
    assert len(report["evaluator_engine_sha256"]) == 64
    assert len(report["runner_sha256"]) == 64
    assert report["planner_probe_mode_counts"] == {"normal": 1}
    assert report["planner_probe_horizon_counts"] == {"2": 1}
    assert report["planner_probe_mode_switch_count"] == 0
    assert report["episodes"][0]["planner_probe_schedule"] == [
        {
            "tick": 0,
            "gauge": 40_000,
            "probe_mode": "normal",
            "probe_ticks": 2,
        }
    ]
    assert PROMOTION.parser().get_default("probe_ticks") == 256
    assert PROMOTION.parser().get_default("long_probe_ticks") == 512
    assert PROMOTION.parser().get_default("low_gauge_threshold") == 20_000
    assert PROMOTION.parser().get_default("low_gauge_exit_threshold") == 30_000


def test_recovery_catches_only_exact_worker_error_and_restores_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise DEV.ExactWorkerError("forked counterfactual worker exited on signal 11")

    monkeypatch.setattr(DEV.FastMultiActionPlanner, "evaluate", fail)
    episode, _provenance = DEV.run_episode(
        DEV_TEST._Runtime(),
        DEV_TEST._bundle(),
        2,
        maximum_ticks=2,
        planner_config=_config(),
        target_score=100_000,
        trace_interval_ticks=1,
        maximum_logged_queries=2,
        recover_exact_branch_errors=True,
    )

    assert episode["exact_branch_errors"] == 1
    assert episode["selected_categories"] == {"wait-exact-branch-error": 1}
    assert episode["invalid_actions"] == 0
    assert all(
        event["state_hash_before"] == event["state_hash_after"]
        and event["parent_state_unchanged"] is True
        for event in episode["exact_branch_error_events"]
    )

    for exception in (RuntimeError("not recoverable"), ValueError("also fatal")):
        def wrong_failure(*_args, **_kwargs):
            raise exception

        monkeypatch.setattr(DEV.FastMultiActionPlanner, "evaluate", wrong_failure)
        with pytest.raises(type(exception), match=str(exception)):
            DEV.run_episode(
                DEV_TEST._Runtime(),
                DEV_TEST._bundle(),
                2,
                maximum_ticks=2,
                planner_config=_config(),
                target_score=100_000,
                trace_interval_ticks=1,
                maximum_logged_queries=2,
                recover_exact_branch_errors=True,
            )


def test_recovery_refuses_changed_live_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def corrupt(_planner, env, *_args, **_kwargs):
        env.state["tick"] += 1
        raise DEV.ExactWorkerError("branch failed after parent corruption")

    monkeypatch.setattr(DEV.FastMultiActionPlanner, "evaluate", corrupt)
    with pytest.raises(RuntimeError, match="altered the live parent state"):
        DEV.run_episode(
            DEV_TEST._Runtime(),
            DEV_TEST._bundle(),
            2,
            maximum_ticks=2,
            planner_config=_config(),
            target_score=100_000,
            trace_interval_ticks=1,
            maximum_logged_queries=2,
            recover_exact_branch_errors=True,
        )


def test_merger_verifies_exact_coverage_and_recomputes_contract(tmp_path: Path) -> None:
    seeds = tuple(range(2, 22))
    shard = _report(seeds)
    path = tmp_path / "shard.json"
    _write(path, shard)

    verified, evidence = MERGE.verify_shard(path)
    merged = MERGE.merge_shards([path], seeds)

    assert verified["episode_count"] == 20
    assert evidence["evaluation_seeds"] == list(seeds)
    assert merged["promotion_eligible"] is True
    assert merged["passed"] is False
    assert merged["success_count"] == 0
    assert merged["invalid_actions"] == 0
    assert len(merged["promotion_report_content_sha256"]) == 64

    with pytest.raises(ValueError, match="exactly cover"):
        MERGE.merge_shards([path], tuple(range(3, 23)))


def test_merger_rejects_logged_selection_of_invalid_candidate(tmp_path: Path) -> None:
    report = _report([2])
    tampered = copy.deepcopy(report)
    query = tampered["episodes"][0]["query_log"][0]
    selected = query["selected_ordinal"]
    outcome = next(value for value in query["outcomes"] if value["ordinal"] == selected)
    outcome["invalid_actions"] = 1
    tampered["episodes"][0]["query_log_sha256"] = DEV.canonical_sha256(
        tampered["episodes"][0]["query_log"]
    )
    tampered.pop("report_content_sha256")
    tampered["report_content_sha256"] = DEV.canonical_sha256(tampered)
    path = tmp_path / "tampered.json"
    _write(path, tampered)

    with pytest.raises(ValueError, match="selected an invalid candidate"):
        MERGE.verify_shard(path)


def test_merger_rejects_tampered_per_query_probe_horizon(tmp_path: Path) -> None:
    report = _report([2])
    tampered = copy.deepcopy(report)
    query = tampered["episodes"][0]["query_log"][0]
    query["probe_ticks"] = 3
    tampered["episodes"][0]["query_log_sha256"] = DEV.canonical_sha256(
        tampered["episodes"][0]["query_log"]
    )
    tampered.pop("report_content_sha256")
    tampered["report_content_sha256"] = DEV.canonical_sha256(tampered)
    path = tmp_path / "tampered-horizon.json"
    _write(path, tampered)

    with pytest.raises(ValueError, match="probe horizon differs|logged query differs"):
        MERGE.verify_shard(path)


def test_merger_accepts_signed_terminal_gauge_in_probe_schedule() -> None:
    report = _report([2])
    episode = report["episodes"][0]
    episode["planner_probe_schedule"][0]["gauge"] = -1
    episode["query_log"][0]["gauge"] = -1
    episode["planner_probe_schedule_sha256"] = DEV.canonical_sha256(
        episode["planner_probe_schedule"]
    )
    episode["query_log_sha256"] = DEV.canonical_sha256(episode["query_log"])

    MERGE._verify_probe_aggregates(episode, 2, report["planner_config"])
