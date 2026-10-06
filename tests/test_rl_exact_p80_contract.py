from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

from irisu_rl.seeds import SeedAllocator


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _load(name: str, filename: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = _load(
    "test_rl_exact_fast_multiaction_eval_support",
    "../tests/test_rl_exact_fast_multiaction_eval.py",
)
DEV = sys.modules["rl_exact_fast_multiaction_eval"]
P80 = _load("rl_exact_p80_contract", "rl_exact_p80_contract.py")
EVAL = _load(
    "rl_exact_fast_multiaction_p80_eval",
    "rl_exact_fast_multiaction_p80_eval.py",
)
MERGE = _load(
    "rl_exact_fast_multiaction_p80_merge",
    "rl_exact_fast_multiaction_p80_merge.py",
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
    return EVAL.evaluate_p80_shard(
        SUPPORT._Runtime(),
        SUPPORT._bundle(),
        declared_training_seeds=[1],
        evaluation_seeds=seeds,
        maximum_ticks=2,
        planner_config=_config(),
        trace_interval_ticks=1,
        maximum_logged_queries=1,
    )


def _set_scores(report, scores) -> None:
    for row, score in zip(report["episodes"], scores, strict=True):
        row["score"] = score
        row["success"] = score >= 100_000
    gate = P80.evaluate_contract(scores, invalid_actions=0)
    report.update(
        {
            "scores": list(scores),
            "median_score": float(statistics.median(scores)),
            "mean_score": statistics.fmean(scores),
            "minimum_score": min(scores),
            "maximum_score": max(scores),
            "success_count": gate["baseline_success_count"],
            "success_fraction_at_or_above_target": gate[
                "baseline_success_fraction_at_or_above_target"
            ],
            **{field: gate[field] for field in MERGE.P80_SUMMARY_FIELDS},
        }
    )
    DEV.bind_report_content_sha256(report)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_p80_is_linear_type7_with_nearest_rank_companion() -> None:
    scores = [
        124_733,
        199_325,
        184_096,
        205_487,
        126_850,
        86_953,
        126_930,
        121_769,
        113_005,
        184_899,
        141_759,
        198_180,
        183_759,
        148_102,
        125_366,
        121_926,
        188_539,
        127_596,
        251_840,
        80_284,
    ]
    result = P80.evaluate_contract(scores, invalid_actions=0)
    assert result["p80_score_linear_type7"] == pytest.approx(190_467.2)
    assert result["p80_score_linear_type7_exact"] == {
        "numerator": 952_336,
        "denominator": 5,
    }
    assert result["p80_nearest_rank_score"] == 188_539
    assert result["passed"] is False


def test_interpolation_cannot_manufacture_a_p80_pass() -> None:
    scores = [0] * 4 + [100_000] * 11 + [199_999] + [300_000] * 4
    result = P80.evaluate_contract(scores, invalid_actions=0)
    assert result["baseline_success_count"] == 16
    assert result["median_score"] == 100_000
    assert result["p80_score_linear_type7"] > 200_000
    assert result["p80_nearest_rank_score"] == 199_999
    assert result["scores_at_or_above_200k"] == 4
    assert result["passed"] is False
    scores[15] = 200_000
    assert P80.evaluate_contract(scores, invalid_actions=0)["passed"] is True
    assert P80.evaluate_contract(scores, invalid_actions=1)["passed"] is False


def test_p80_merger_recomputes_gate_and_rejects_tampering(tmp_path: Path) -> None:
    seeds = tuple(range(2, 22))
    scores = [0] * 4 + [100_000] * 11 + [200_000] + [300_000] * 4
    left, right = _report(seeds[:10]), _report(seeds[10:])
    _set_scores(left, scores[:10])
    _set_scores(right, scores[10:])
    left_path, right_path = tmp_path / "left.json", tmp_path / "right.json"
    _write(left_path, left)
    _write(right_path, right)
    merged = MERGE.merge_p80_shards([left_path, right_path], seeds)
    assert merged["passed"] is True
    assert merged["success_count"] == 16
    assert merged["p80_nearest_rank_score"] == 200_000
    assert merged["scores_at_or_above_200k"] == 5
    interpolated_only = scores.copy()
    interpolated_only[15] = 199_999
    _set_scores(right, interpolated_only[10:])
    _write(right_path, right)
    rejected = MERGE.merge_p80_shards([left_path, right_path], seeds)
    assert rejected["p80_score_linear_type7"] > 200_000
    assert rejected["p80_nearest_rank_score"] == 199_999
    assert rejected["passed"] is False
    left["p80_score_linear_type7"] += 1
    DEV.bind_report_content_sha256(left)
    _write(left_path, left)
    with pytest.raises(ValueError, match="p80_score_linear_type7"):
        MERGE.verify_p80_shard(left_path)


def test_p80_defaults_are_frozen_fixed_512() -> None:
    parser = EVAL.parser()
    assert parser.get_default("probe_ticks") == 512
    assert parser.get_default("long_probe_ticks") == 512
    assert parser.get_default("top_k_pairs") == 0
    assert P80.CONTRACT["robust_order_statistic_companion"] == {
        "name": "p80_nearest_rank_score",
        "ascending_rank": 16,
        "target_score": 200_000,
        "minimum_scores_at_or_above_target": 5,
    }


def test_fresh_development_and_final_manifests_are_reproducible_and_disjoint() -> None:
    paths = (
        ROOT / "configs/rl/eval/development-p80-200k-v1.json",
        ROOT / "configs/rl/eval/locked-final-p80-200k-v1.json",
    )
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    known_prior: set[int] = set()
    for path in (ROOT / "configs/rl/eval").glob("*.json"):
        if path in paths:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            known_prior.update(value)
        elif isinstance(value, dict):
            known_prior.update(value.get("evaluation_seeds", ()))
    observed: list[set[int]] = []
    for manifest in manifests:
        allocator = manifest["allocator"]
        seeds = tuple(manifest["evaluation_seeds"])
        assert seeds == SeedAllocator(
            allocator["split"], key=allocator["key"], cursor=allocator["cursor"]
        ).take(allocator["count"])
        assert DEV.canonical_sha256(list(seeds)) == manifest[
            "evaluation_seeds_sha256"
        ]
        assert manifest["promotion_contract_sha256"] == DEV.canonical_sha256(
            P80.CONTRACT
        )
        assert not set(seeds) & known_prior
        observed.append(set(seeds))
    assert not observed[0] & observed[1]
    assert manifests[0]["allocator"]["split"] == "validation"
    assert manifests[1]["allocator"]["split"] == "test"
    assert manifests[1]["allocator"]["maximum_promotion_attempts"] == 1


def test_p80_confirmation_manifest_is_cursor20_and_disjoint() -> None:
    directory = ROOT / "configs/rl/eval"
    selection = json.loads(
        (directory / "development-p80-200k-v1.json").read_text(encoding="utf-8")
    )
    confirmation = json.loads(
        (directory / "development-p80-200k-confirmation-v2.json").read_text(
            encoding="utf-8"
        )
    )
    locked_final = json.loads(
        (directory / "locked-final-p80-200k-v1.json").read_text(encoding="utf-8")
    )

    allocator = confirmation["allocator"]
    assert allocator == {
        "version": "seed-allocator-v1",
        "split": "validation",
        "key": 6342426784305351267,
        "cursor": 20,
        "count": 20,
        "selected_before_evaluation": True,
    }
    seeds = tuple(confirmation["evaluation_seeds"])
    assert seeds == SeedAllocator(
        "validation", key=allocator["key"], cursor=20
    ).take(20)
    assert confirmation["evaluation_seeds_sha256"] == (
        "d791706d98a5eea5f06bf7e74196bc6c03bfc5485e70752c0e766599af10eed9"
    )
    assert DEV.canonical_sha256(list(seeds)) == confirmation[
        "evaluation_seeds_sha256"
    ]
    assert confirmation["promotion_contract_sha256"] == DEV.canonical_sha256(
        P80.CONTRACT
    )

    confirmation_set = set(seeds)
    assert confirmation_set.isdisjoint(selection["evaluation_seeds"])
    assert confirmation_set.isdisjoint(locked_final["evaluation_seeds"])
    assert set(selection["evaluation_seeds"]).isdisjoint(
        locked_final["evaluation_seeds"]
    )
