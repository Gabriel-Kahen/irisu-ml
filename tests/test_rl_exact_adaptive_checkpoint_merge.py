from __future__ import annotations

import importlib.util
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
SOURCE = BENCHMARKS / "rl_exact_adaptive_checkpoint_merge.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_exact_adaptive_checkpoint_merge", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REQUESTED = tuple(range(101, 121))
CONTRACT = {
    "required_episode_count": 20,
    "required_success_count": 16,
    "target_score": 50_000,
    "require_median_at_or_above_target": True,
    "require_zero_invalid_actions": True,
}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _make_shard(
    tmp_path: Path,
    name: str,
    seeds,
    *,
    episode_artifacts: bool = False,
    identity_suffix: str = "",
) -> Path:
    rows = []
    artifacts = []
    for seed in seeds:
        score = 60_000 if seed <= 116 else 40_000
        trace = [{"tick": 0, "score": 0}, {"tick": 100, "score": score}]
        row = {
            "seed": seed,
            "score": score,
            "success": score >= 50_000,
            "invalid_actions": 0,
            "trace": trace,
            "trace_sha256": MODULE.canonical_sha256(trace),
        }
        rows.append(row)
        if episode_artifacts:
            episode_path = (tmp_path / f"episodes-{name}" / f"seed-{seed}.json").resolve()
            _write(episode_path, row)
            artifacts.append(
                {
                    "seed": seed,
                    "path": str(episode_path),
                    "file_sha256": MODULE.file_sha256(episode_path),
                    "content_sha256": MODULE.canonical_sha256(row),
                }
            )
    scores = [row["score"] for row in rows]
    successes = sum(row["success"] for row in rows)
    checkpoint_metadata = {"training_seeds": [1, 2, 3]}
    exact_runtime = {
        "physics_backend": "exact",
        "runtime_attestation_sha256": "9" * 64,
    }
    report = {
        "format": MODULE.SHARD_FORMAT,
        "physics_backend": "exact",
        "deterministic_policy": True,
        "promotion_contract": CONTRACT,
        "promotion_eligible": len(rows) == 20,
        "training_seeds": [1, 2, 3],
        "training_seeds_sha256": MODULE.canonical_sha256([1, 2, 3]),
        "evaluation_seeds": list(seeds),
        "evaluation_seeds_sha256": MODULE.canonical_sha256(list(seeds)),
        "training_evaluation_overlap": [],
        "checkpoint_sha256": "a" * 63 + (identity_suffix or "a"),
        "model_sha256": "b" * 64,
        "checkpoint_metadata_sha256": MODULE.canonical_sha256(checkpoint_metadata),
        "checkpoint_metadata": checkpoint_metadata,
        "inference_config": {"act_logit_bias": 1.0},
        "planner_config": {
            "short_horizon": 128,
            "long_horizon": 256,
            "gauge_threshold": 30_000,
        },
        "maximum_ticks": 120_000,
        "runtime_hashes": {
            "worker_sha256": "d" * 64,
            "exact_library_sha256": "e" * 64,
            "identity_config_sha256": "7" * 64,
            "runtime_provenance_sha256": MODULE.canonical_sha256(exact_runtime),
        },
        "exact_runtime": exact_runtime,
        "runner_sha256": "f" * 64,
        "episodes": rows,
        "episode_count": len(rows),
        "scores": scores,
        "median_score": float(statistics.median(scores)),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(rows),
        "invalid_actions": 0,
        "passed": len(rows) == 20
        and statistics.median(scores) >= 50_000
        and successes >= 16,
    }
    if artifacts:
        report["episode_artifacts"] = artifacts
    report["report_content_sha256"] = MODULE.canonical_sha256(report)
    path = tmp_path / f"{name}.json"
    _write(path, report)
    return path


def _upgrade_v2(path: Path, *, error_seed: int | None = None) -> Path:
    report = json.loads(path.read_text())
    report["format"] = MODULE.SHARD_FORMAT_V2
    report["planner_config"]["exact_branch_error_policy"] = {
        "version": "exact-branch-error-conservative-wait-v1",
        "caught_exception": "irisu_env.exact_ipc.ExactWorkerError",
        "action": "restore-policy-and-execute-wait",
        "require_live_parent_state_hash_unchanged": True,
    }
    total = 0
    for row in report["episodes"]:
        events = []
        if row["seed"] == error_seed:
            message = "forked counterfactual worker exited on signal 11"
            events = [
                {
                    "tick": 123,
                    "state_hash_before": 456,
                    "state_hash_after": 456,
                    "exception_type": "irisu_env.exact_ipc.ExactWorkerError",
                    "message": message,
                    "message_sha256": hashlib.sha256(
                        message.encode()
                    ).hexdigest(),
                    "recovery": "restore-policy-and-execute-wait",
                    "parent_state_unchanged": True,
                }
            ]
        row["exact_branch_errors"] = len(events)
        row["exact_branch_error_events"] = events
        row["exact_branch_error_events_sha256"] = MODULE.canonical_sha256(events)
        row["gate_reasons"] = (
            {"wait-exact-branch-error": 1} if events else {}
        )
        total += len(events)
    report["exact_branch_errors"] = total
    report["branch_error_recovery_enabled"] = True
    report["evaluator_engine_sha256"] = "8" * 64
    report.pop("report_content_sha256")
    report["report_content_sha256"] = MODULE.canonical_sha256(report)
    _write(path, report)
    return path


def test_merges_disjoint_shards_and_recomputes_promotion(tmp_path) -> None:
    left = _make_shard(
        tmp_path, "left", REQUESTED[:10], episode_artifacts=True
    )
    right = _make_shard(tmp_path, "right", REQUESTED[10:])
    report = MODULE.merge_shards([left, right], REQUESTED)
    assert report["episode_count"] == 20
    assert report["success_count"] == 16
    assert report["success_fraction_at_or_above_target"] == 0.8
    assert report["median_score"] == 60_000
    assert report["invalid_actions"] == 0
    assert report["passed"] is True
    assert [row["seed"] for row in report["episodes"]] == list(REQUESTED)
    assert report["shards"][0]["episode_artifacts_verified"] == 10
    content = dict(report)
    digest = content.pop("promotion_report_content_sha256")
    assert digest == MODULE.canonical_sha256(content)


def test_rejects_identity_drift_between_shards(tmp_path) -> None:
    left = _make_shard(tmp_path, "left", REQUESTED[:10])
    right = _make_shard(
        tmp_path, "right", REQUESTED[10:], identity_suffix="9"
    )
    with pytest.raises(ValueError, match="identities differ"):
        MODULE.merge_shards([left, right], REQUESTED)


def test_rejects_duplicate_or_missing_requested_seed(tmp_path) -> None:
    left = _make_shard(tmp_path, "left", REQUESTED[:10])
    duplicate = _make_shard(tmp_path, "duplicate", REQUESTED[9:19])
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.merge_shards([left, duplicate], REQUESTED)

    incomplete = _make_shard(tmp_path, "incomplete", REQUESTED[10:19])
    with pytest.raises(ValueError, match="missing"):
        MODULE.merge_shards([left, incomplete], REQUESTED)


def test_rejects_tampered_shard_self_hash_and_summary(tmp_path) -> None:
    path = _make_shard(tmp_path, "shard", REQUESTED[:10])
    value = json.loads(path.read_text())
    value["scores"][0] = 123
    _write(path, value)
    with pytest.raises(ValueError, match="content SHA-256"):
        MODULE.verify_shard(path)

    path = _make_shard(tmp_path, "summary", REQUESTED[:10])
    value = json.loads(path.read_text())
    value["success_count"] = 0
    value["report_content_sha256"] = MODULE.canonical_sha256(
        {key: child for key, child in value.items() if key != "report_content_sha256"}
    )
    _write(path, value)
    with pytest.raises(ValueError, match="success_count"):
        MODULE.verify_shard(path)


def test_rejects_tampered_episode_artifact(tmp_path) -> None:
    path = _make_shard(
        tmp_path, "shard", REQUESTED[:10], episode_artifacts=True
    )
    report = json.loads(path.read_text())
    episode_path = Path(report["episode_artifacts"][0]["path"])
    episode = json.loads(episode_path.read_text())
    episode["score"] = 999_999
    _write(episode_path, episode)
    with pytest.raises(ValueError, match="content differs"):
        MODULE.verify_shard(path)


def test_requested_manifest_must_be_hash_bound_and_exactly_20(tmp_path) -> None:
    manifest = tmp_path / "seeds.json"
    _write(
        manifest,
        {
            "evaluation_seeds": list(REQUESTED),
            "evaluation_seeds_sha256": MODULE.canonical_sha256(list(REQUESTED)),
        },
    )
    assert MODULE.load_requested_seeds(manifest) == REQUESTED
    _write(
        manifest,
        {"evaluation_seeds": list(REQUESTED), "evaluation_seeds_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE.load_requested_seeds(manifest)


def test_v2_merger_verifies_and_aggregates_recovered_branch_errors(tmp_path) -> None:
    left = _upgrade_v2(
        _make_shard(tmp_path, "left", REQUESTED[:10]),
        error_seed=REQUESTED[3],
    )
    right = _upgrade_v2(_make_shard(tmp_path, "right", REQUESTED[10:]))
    report = MODULE.merge_shards(
        [left, right],
        REQUESTED,
        expected_shard_format=MODULE.SHARD_FORMAT_V2,
        output_format=MODULE.FORMAT_V2,
        merger_path=SOURCE,
    )
    assert report["format"] == MODULE.FORMAT_V2
    assert report["evaluator_format"] == MODULE.SHARD_FORMAT_V2
    assert report["branch_error_recovery_enabled"] is True
    assert report["exact_branch_errors"] == 1
    assert report["passed"] is True


def test_v2_rejects_recovery_event_without_parent_state_equality(tmp_path) -> None:
    path = _upgrade_v2(
        _make_shard(tmp_path, "shard", REQUESTED[:10]),
        error_seed=REQUESTED[0],
    )
    report = json.loads(path.read_text())
    row = report["episodes"][0]
    row["exact_branch_error_events"][0]["state_hash_after"] = 999
    row["exact_branch_error_events_sha256"] = MODULE.canonical_sha256(
        row["exact_branch_error_events"]
    )
    report.pop("report_content_sha256")
    report["report_content_sha256"] = MODULE.canonical_sha256(report)
    _write(path, report)
    with pytest.raises(ValueError, match="event is malformed"):
        MODULE.verify_shard(path, expected_format=MODULE.SHARD_FORMAT_V2)
