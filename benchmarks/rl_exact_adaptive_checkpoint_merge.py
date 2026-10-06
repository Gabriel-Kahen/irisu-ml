#!/usr/bin/env python3
"""Merge independently verified exact adaptive-evaluator shards for promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    DEFAULT_EPISODES,
    DEFAULT_SUCCESSES,
    DEFAULT_TARGET_SCORE,
    FORMAT as SHARD_FORMAT,
    canonical_sha256,
    file_sha256,
    validate_seed_sequence,
)


FORMAT = "irisu-exact-adaptive-learned-planner-promotion-v1"
SHARD_FORMAT_V2 = "irisu-exact-adaptive-learned-planner-eval-v2"
FORMAT_V2 = "irisu-exact-adaptive-learned-planner-promotion-v2"
IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_sha256",
    "checkpoint_metadata_sha256",
    "training_seeds",
    "training_seeds_sha256",
    "inference_config",
    "planner_config",
    "maximum_ticks",
    "runtime_hashes",
    "runner_sha256",
    "promotion_contract",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return value


def load_requested_seeds(path: Path) -> tuple[int, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        raw = value
        recorded_sha = None
    elif isinstance(value, dict) and isinstance(value.get("evaluation_seeds"), list):
        raw = value["evaluation_seeds"]
        recorded_sha = value.get("evaluation_seeds_sha256")
    else:
        raise ValueError(
            "evaluation seed manifest must be a JSON list or contain evaluation_seeds"
        )
    seeds = validate_seed_sequence(raw, "requested evaluation")
    if len(seeds) != DEFAULT_EPISODES:
        raise ValueError(
            f"promotion seed manifest must contain exactly {DEFAULT_EPISODES} seeds"
        )
    digest = canonical_sha256(list(seeds))
    if recorded_sha is not None and recorded_sha != digest:
        raise ValueError("evaluation seed manifest SHA-256 mismatch")
    return seeds


def _expected_pass(report: Mapping[str, Any]) -> bool:
    contract = report["promotion_contract"]
    episodes = report["episodes"]
    target = int(contract["target_score"])
    scores = [int(row["score"]) for row in episodes]
    successes = sum(score >= target for score in scores)
    invalid = sum(int(row["invalid_actions"]) for row in episodes)
    return (
        len(episodes) == int(contract["required_episode_count"])
        and statistics.median(scores) >= target
        and successes >= int(contract["required_success_count"])
        and invalid == 0
    )


def _verify_episode_artifacts(
    report: Mapping[str, Any], episodes_by_seed: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, object]]:
    raw = report.get("episode_artifacts")
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) != len(episodes_by_seed):
        raise ValueError("episode_artifacts must cover every shard episode")
    verified: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("episode artifact entries must be objects")
        try:
            seed = int(item["seed"])
            path = Path(item["path"])
            recorded_file_sha = item["file_sha256"]
            recorded_content_sha = item["content_sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode artifact entry is malformed") from exc
        if seed in seen or seed not in episodes_by_seed:
            raise ValueError("episode artifact seeds differ from shard episodes")
        if not path.is_absolute():
            raise ValueError("episode artifact paths must be absolute")
        episode = _read_json_object(path, "episode artifact")
        if episode != episodes_by_seed[seed]:
            raise ValueError(f"episode artifact content differs for seed {seed}")
        actual_file_sha = file_sha256(path)
        actual_content_sha = canonical_sha256(episode)
        if recorded_file_sha != actual_file_sha:
            raise ValueError(f"episode artifact file SHA-256 mismatch for seed {seed}")
        if recorded_content_sha != actual_content_sha:
            raise ValueError(f"episode artifact content SHA-256 mismatch for seed {seed}")
        verified.append(
            {
                "seed": seed,
                "path": str(path),
                "file_sha256": actual_file_sha,
                "content_sha256": actual_content_sha,
            }
        )
        seen.add(seed)
    if seen != set(episodes_by_seed):
        raise ValueError("episode artifacts omit shard episodes")
    return verified


def verify_shard(
    path: Path, *, expected_format: str = SHARD_FORMAT
) -> tuple[dict[str, Any], dict[str, object]]:
    report = _read_json_object(path, "shard report")
    if report.get("format") != expected_format or report.get("physics_backend") != "exact":
        raise ValueError(f"not an exact adaptive evaluator shard: {path}")
    recorded_report_sha = report.get("report_content_sha256")
    content = dict(report)
    content.pop("report_content_sha256", None)
    actual_report_sha = canonical_sha256(content)
    if recorded_report_sha != actual_report_sha:
        raise ValueError(f"shard report content SHA-256 mismatch: {path}")

    contract = report.get("promotion_contract")
    expected_contract = {
        "required_episode_count": DEFAULT_EPISODES,
        "required_success_count": DEFAULT_SUCCESSES,
        "target_score": DEFAULT_TARGET_SCORE,
        "require_median_at_or_above_target": True,
        "require_zero_invalid_actions": True,
    }
    if contract != expected_contract:
        raise ValueError("shard promotion contract is not the canonical 16/20 at 50k gate")
    for field in IDENTITY_FIELDS:
        if field not in report:
            raise ValueError(f"shard is missing identity field {field}")
    if report.get("deterministic_policy") is not True:
        raise ValueError("shard policy must be deterministic")
    for field in (
        "checkpoint_sha256",
        "model_sha256",
        "checkpoint_metadata_sha256",
        "runner_sha256",
    ):
        if not _is_sha256(report[field]):
            raise ValueError(f"shard {field} is not a SHA-256")
    metadata = report.get("checkpoint_metadata")
    if not isinstance(metadata, dict) or report["checkpoint_metadata_sha256"] != canonical_sha256(metadata):
        raise ValueError("shard checkpoint metadata SHA-256 mismatch")
    runtime_hashes = report["runtime_hashes"]
    required_runtime_hashes = {
        "worker_sha256",
        "exact_library_sha256",
        "identity_config_sha256",
        "runtime_provenance_sha256",
    }
    if (
        not isinstance(runtime_hashes, dict)
        or not required_runtime_hashes.issubset(runtime_hashes)
        or any(not _is_sha256(runtime_hashes[name]) for name in required_runtime_hashes)
    ):
        raise ValueError("shard exact runtime hashes are malformed")
    exact_runtime = report.get("exact_runtime")
    if (
        not isinstance(exact_runtime, dict)
        or exact_runtime.get("physics_backend") != "exact"
        or runtime_hashes["runtime_provenance_sha256"]
        != canonical_sha256(exact_runtime)
    ):
        raise ValueError("shard exact runtime provenance SHA-256 mismatch")

    training = validate_seed_sequence(report["training_seeds"], "shard training")
    evaluation = validate_seed_sequence(report["evaluation_seeds"], "shard evaluation")
    if report.get("training_seeds_sha256") != canonical_sha256(list(training)):
        raise ValueError("shard training-seed SHA-256 mismatch")
    if report.get("evaluation_seeds_sha256") != canonical_sha256(list(evaluation)):
        raise ValueError("shard evaluation-seed SHA-256 mismatch")
    if set(training) & set(evaluation) or report.get("training_evaluation_overlap") != []:
        raise ValueError("shard has training/evaluation seed leakage")

    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("shard episodes must be a nonempty list")
    try:
        episode_seeds = tuple(int(row["seed"]) for row in episodes)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("shard episode rows are malformed") from exc
    if episode_seeds != evaluation or len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("shard episodes do not exactly match evaluation seeds")

    target = DEFAULT_TARGET_SCORE
    scores: list[int] = []
    invalids: list[int] = []
    episodes_by_seed: dict[int, Mapping[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError("shard episode rows must be objects")
        seed = int(row["seed"])
        score = int(row["score"])
        invalid = int(row["invalid_actions"])
        if score < 0 or invalid < 0:
            raise ValueError("shard episode score/invalid count is malformed")
        if row.get("success") is not (score >= target):
            raise ValueError(f"shard episode success flag differs for seed {seed}")
        trace = row.get("trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError(f"shard episode trace is missing for seed {seed}")
        if row.get("trace_sha256") != canonical_sha256(trace):
            raise ValueError(f"shard episode trace SHA-256 mismatch for seed {seed}")
        if expected_format == SHARD_FORMAT_V2:
            error_count = row.get("exact_branch_errors")
            error_events = row.get("exact_branch_error_events")
            if (
                isinstance(error_count, bool)
                or not isinstance(error_count, int)
                or error_count < 0
                or not isinstance(error_events, list)
                or len(error_events) != error_count
                or row.get("exact_branch_error_events_sha256")
                != canonical_sha256(error_events)
            ):
                raise ValueError(
                    f"shard branch-error evidence is inconsistent for seed {seed}"
                )
            if int(row.get("gate_reasons", {}).get("wait-exact-branch-error", 0)) != error_count:
                raise ValueError(
                    f"shard branch-error reason count differs for seed {seed}"
                )
            for event in error_events:
                if (
                    not isinstance(event, dict)
                    or event.get("exception_type")
                    != "irisu_env.exact_ipc.ExactWorkerError"
                    or event.get("recovery")
                    != "restore-policy-and-execute-wait"
                    or event.get("parent_state_unchanged") is not True
                    or event.get("state_hash_before") != event.get("state_hash_after")
                    or not isinstance(event.get("message"), str)
                    or event.get("message_sha256")
                    != hashlib.sha256(
                        event["message"].encode("utf-8")
                    ).hexdigest()
                ):
                    raise ValueError(
                        f"shard branch-error event is malformed for seed {seed}"
                    )
        scores.append(score)
        invalids.append(invalid)
        episodes_by_seed[seed] = row

    successes = sum(score >= target for score in scores)
    expected = {
        "episode_count": len(episodes),
        "scores": scores,
        "median_score": float(statistics.median(scores)),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(scores),
        "invalid_actions": sum(invalids),
        "promotion_eligible": len(episodes) == DEFAULT_EPISODES,
        "passed": _expected_pass(report),
    }
    for field, expected_value in expected.items():
        actual = report.get(field)
        if isinstance(expected_value, float):
            valid = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=0.0
            )
        else:
            valid = actual == expected_value
        if not valid:
            raise ValueError(f"shard summary field {field} is inconsistent")

    if expected_format == SHARD_FORMAT_V2:
        expected_policy = {
            "version": "exact-branch-error-conservative-wait-v1",
            "caught_exception": "irisu_env.exact_ipc.ExactWorkerError",
            "action": "restore-policy-and-execute-wait",
            "require_live_parent_state_hash_unchanged": True,
        }
        branch_errors = sum(int(row["exact_branch_errors"]) for row in episodes)
        if (
            report.get("branch_error_recovery_enabled") is not True
            or report.get("exact_branch_errors") != branch_errors
            or not _is_sha256(report.get("evaluator_engine_sha256"))
            or report["planner_config"].get("exact_branch_error_policy")
            != expected_policy
        ):
            raise ValueError("shard v2 branch-error recovery identity is inconsistent")

    artifacts = _verify_episode_artifacts(report, episodes_by_seed)
    evidence: dict[str, object] = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": actual_report_sha,
        "evaluation_seeds": list(evaluation),
        "episode_artifacts_verified": len(artifacts),
    }
    return report, evidence


def merge_shards(
    shard_paths: Sequence[Path],
    requested_seeds: Sequence[int],
    *,
    expected_shard_format: str = SHARD_FORMAT,
    output_format: str = FORMAT,
    merger_path: Path | None = None,
) -> dict[str, object]:
    requested = validate_seed_sequence(requested_seeds, "requested evaluation")
    if len(requested) != DEFAULT_EPISODES:
        raise ValueError(f"promotion requires exactly {DEFAULT_EPISODES} requested seeds")
    if not shard_paths:
        raise ValueError("at least one shard report is required")

    started = time.monotonic()
    shards: list[dict[str, Any]] = []
    evidence: list[dict[str, object]] = []
    identity: dict[str, object] | None = None
    identity_fields = IDENTITY_FIELDS + (
        ("evaluator_engine_sha256",)
        if expected_shard_format == SHARD_FORMAT_V2
        else ()
    )
    by_seed: dict[int, dict[str, Any]] = {}
    artifact_evidence: list[dict[str, object]] = []
    for raw_path in shard_paths:
        path = raw_path.resolve(strict=True)
        shard, shard_evidence = verify_shard(
            path, expected_format=expected_shard_format
        )
        current_identity = {field: shard[field] for field in identity_fields}
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            differing = [
                field
                for field in identity_fields
                if current_identity[field] != identity[field]
            ]
            raise ValueError(f"shard identities differ: {differing}")
        for row in shard["episodes"]:
            seed = int(row["seed"])
            if seed in by_seed:
                raise ValueError(f"shards contain duplicate evaluation seed {seed}")
            by_seed[seed] = dict(row)
        for item in shard.get("episode_artifacts", []):
            artifact_evidence.append(dict(item))
        shards.append(shard)
        evidence.append(shard_evidence)

    requested_set = set(requested)
    observed_set = set(by_seed)
    if observed_set != requested_set:
        missing = sorted(requested_set - observed_set)
        unexpected = sorted(observed_set - requested_set)
        raise ValueError(
            f"shards do not exactly cover requested seed manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    episodes = [by_seed[seed] for seed in requested]
    scores = [int(row["score"]) for row in episodes]
    successes = sum(score >= DEFAULT_TARGET_SCORE for score in scores)
    invalid = sum(int(row["invalid_actions"]) for row in episodes)
    median_score = float(statistics.median(scores))
    assert identity is not None
    report: dict[str, object] = {
        "format": output_format,
        "evaluator_format": expected_shard_format,
        "physics_backend": "exact",
        "deterministic_policy": True,
        **identity,
        "checkpoint_metadata": dict(shards[0]["checkpoint_metadata"]),
        "exact_runtime": dict(shards[0]["exact_runtime"]),
        "requested_evaluation_seeds": list(requested),
        "requested_evaluation_seeds_sha256": canonical_sha256(list(requested)),
        "training_evaluation_overlap": [],
        "shards": evidence,
        "shard_count": len(shards),
        "episode_artifacts": artifact_evidence,
        "episodes": episodes,
        "episode_count": len(episodes),
        "scores": scores,
        "median_score": median_score,
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(scores),
        "invalid_actions": invalid,
        "promotion_eligible": True,
        "passed": median_score >= DEFAULT_TARGET_SCORE
        and successes >= DEFAULT_SUCCESSES
        and invalid == 0,
        "merger_sha256": file_sha256(
            Path(__file__).resolve()
            if merger_path is None
            else merger_path.resolve(strict=True)
        ),
        "wall_seconds": time.monotonic() - started,
    }
    if expected_shard_format == SHARD_FORMAT_V2:
        report["branch_error_recovery_enabled"] = True
        report["exact_branch_errors"] = sum(
            int(row["exact_branch_errors"]) for row in episodes
        )
    report["promotion_report_content_sha256"] = canonical_sha256(report)
    return report


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shard", type=Path, action="append", required=True)
    result.add_argument("--evaluation-seed-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    expected_shard_format: str = SHARD_FORMAT,
    output_format: str = FORMAT,
    merger_path: Path | None = None,
) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        parser().error("--output must not already exist")
    requested = load_requested_seeds(
        args.evaluation_seed_manifest.resolve(strict=True)
    )
    report = merge_shards(
        args.shard,
        requested,
        expected_shard_format=expected_shard_format,
        output_format=output_format,
        merger_path=merger_path,
    )
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
