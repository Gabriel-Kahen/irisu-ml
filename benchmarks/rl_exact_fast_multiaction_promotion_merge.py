#!/usr/bin/env python3
"""Verify and merge FastMultiActionPlanner exact promotion shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    validate_seed_sequence,
)
from rl_exact_fast_multiaction_promotion_eval import (  # noqa: E402
    BRANCH_ERROR_POLICY,
    CONTRACT,
    FORMAT as SHARD_FORMAT,
    REQUIRED_EPISODES,
    REQUIRED_SUCCESSES,
    TARGET_SCORE,
)


FORMAT = "irisu-exact-fast-multiaction-promotion-v1"
IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_sha256",
    "checkpoint_metadata_sha256",
    "training_seeds",
    "training_seeds_sha256",
    "inference_config",
    "planner_config",
    "planner_objective",
    "branch_error_policy",
    "maximum_ticks",
    "runtime_hashes",
    "planner_source_sha256",
    "evaluator_engine_sha256",
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation seed manifest JSON: {path}") from exc
    if isinstance(value, list):
        raw = value
        recorded = None
    elif isinstance(value, dict) and isinstance(value.get("evaluation_seeds"), list):
        raw = value["evaluation_seeds"]
        recorded = value.get("evaluation_seeds_sha256")
    else:
        raise ValueError(
            "evaluation seed manifest must be a list or contain evaluation_seeds"
        )
    seeds = validate_seed_sequence(raw, "requested evaluation")
    if len(seeds) != REQUIRED_EPISODES:
        raise ValueError(
            f"promotion seed manifest must contain exactly {REQUIRED_EPISODES} seeds"
        )
    digest = canonical_sha256(list(seeds))
    if recorded is not None and recorded != digest:
        raise ValueError("evaluation seed manifest SHA-256 mismatch")
    return seeds


def _verify_artifacts(
    report: Mapping[str, Any], episodes: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, object]]:
    raw = report.get("episode_artifacts")
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) != len(episodes):
        raise ValueError("episode_artifacts must cover every shard episode")
    verified: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("episode artifact entries must be objects")
        try:
            seed = int(item["seed"])
            path = Path(item["path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode artifact entry is malformed") from exc
        if seed in seen or seed not in episodes or not path.is_absolute():
            raise ValueError("episode artifact seed/path is invalid")
        artifact = _read_json_object(path, "episode artifact")
        if artifact != episodes[seed]:
            raise ValueError(f"episode artifact differs for seed {seed}")
        file_digest = file_sha256(path)
        content_digest = canonical_sha256(artifact)
        if item.get("file_sha256") != file_digest:
            raise ValueError(f"episode artifact file SHA-256 mismatch for seed {seed}")
        if item.get("content_sha256") != content_digest:
            raise ValueError(
                f"episode artifact content SHA-256 mismatch for seed {seed}"
            )
        verified.append(
            {
                "seed": seed,
                "path": str(path),
                "file_sha256": file_digest,
                "content_sha256": content_digest,
            }
        )
        seen.add(seed)
    if seen != set(episodes):
        raise ValueError("episode artifacts omit shard episodes")
    return verified


def _probe_mode_horizon(
    planner_config: Mapping[str, Any], mode: object
) -> int:
    if mode == "normal":
        return int(planner_config["probe_ticks"])
    if mode == "low-gauge-long":
        return int(planner_config["long_probe_ticks"])
    raise ValueError(f"unknown planner probe mode {mode!r}")


def _verify_logged_queries(
    row: Mapping[str, Any], seed: int, planner_config: Mapping[str, Any]
) -> None:
    log = row.get("query_log")
    if not isinstance(log, list):
        raise ValueError(f"query log is missing for seed {seed}")
    if row.get("query_log_sha256") != canonical_sha256(log):
        raise ValueError(f"query log SHA-256 mismatch for seed {seed}")
    if row.get("logged_query_count") != len(log):
        raise ValueError(f"logged query count differs for seed {seed}")
    for query in log:
        if not isinstance(query, dict):
            raise ValueError(f"query log entry is malformed for seed {seed}")
        expected_horizon = _probe_mode_horizon(
            planner_config, query.get("probe_mode")
        )
        if query.get("probe_ticks") != expected_horizon:
            raise ValueError(f"query probe horizon differs for seed {seed}")
        if "selected_ordinal" not in query:
            continue
        outcomes = query.get("outcomes")
        selected_ordinal = query["selected_ordinal"]
        if not isinstance(outcomes, list):
            raise ValueError(f"query outcomes are malformed for seed {seed}")
        selected = [
            value
            for value in outcomes
            if isinstance(value, dict) and value.get("ordinal") == selected_ordinal
        ]
        if len(selected) != 1 or int(selected[0].get("invalid_actions", -1)) != 0:
            raise ValueError(
                f"logged planner query selected an invalid candidate for seed {seed}"
            )
        if any(
            int(value.get("probe", {}).get("survival_ticks", -1))
            > expected_horizon
            for value in outcomes
            if isinstance(value, dict)
        ):
            raise ValueError(f"query outcome exceeds probe horizon for seed {seed}")


def _verify_probe_aggregates(
    row: Mapping[str, Any], seed: int, planner_config: Mapping[str, Any]
) -> None:
    query_count = int(row["planner_queries"])
    mode_counts = row.get("planner_probe_mode_counts")
    horizon_counts = row.get("planner_probe_horizon_counts")
    switches = row.get("planner_probe_mode_switches")
    schedule = row.get("planner_probe_schedule")
    if (
        not isinstance(mode_counts, dict)
        or not isinstance(horizon_counts, dict)
        or not isinstance(switches, list)
        or not isinstance(schedule, list)
        or len(schedule) != query_count
        or row.get("planner_probe_schedule_sha256")
        != canonical_sha256(schedule)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*mode_counts.values(), *horizon_counts.values())
        )
        or sum(mode_counts.values()) != query_count
        or sum(horizon_counts.values()) != query_count
    ):
        raise ValueError(f"planner probe aggregates differ for seed {seed}")
    short = int(planner_config["probe_ticks"])
    long = int(planner_config["long_probe_ticks"])
    expected_horizons: Counter[str] = Counter()
    for mode, count in mode_counts.items():
        expected_horizons[str(_probe_mode_horizon(planner_config, mode))] += count
    if dict(sorted(expected_horizons.items())) != horizon_counts:
        raise ValueError(f"planner mode/horizon counts differ for seed {seed}")
    if long == short and mode_counts.get("low-gauge-long", 0) != 0:
        raise ValueError(f"equal horizons entered long mode for seed {seed}")
    current = "normal"
    enter = int(planner_config["low_gauge_threshold"])
    exit_threshold = int(planner_config["low_gauge_exit_threshold"])
    derived_modes: Counter[str] = Counter()
    derived_horizons: Counter[str] = Counter()
    derived_switches: list[dict[str, object]] = []
    for item in schedule:
        if not isinstance(item, dict):
            raise ValueError(f"probe schedule is malformed for seed {seed}")
        raw_gauge = item.get("gauge")
        if isinstance(raw_gauge, bool) or not isinstance(raw_gauge, int):
            raise ValueError(f"probe schedule gauge is malformed for seed {seed}")
        gauge = raw_gauge
        expected = current
        if long == short:
            expected = "normal"
        elif current == "low-gauge-long" and gauge > exit_threshold:
            expected = "normal"
        elif current == "normal" and gauge < enter:
            expected = "low-gauge-long"
        horizon = _probe_mode_horizon(planner_config, expected)
        if (
            item.get("probe_mode") != expected
            or item.get("probe_ticks") != horizon
            or isinstance(item.get("tick"), bool)
            or not isinstance(item.get("tick"), int)
            or int(item["tick"]) < 0
        ):
            raise ValueError(f"probe schedule hysteresis differs for seed {seed}")
        derived_modes[expected] += 1
        derived_horizons[str(horizon)] += 1
        if expected != current:
            derived_switches.append(
                {
                    "tick": int(item["tick"]),
                    "gauge": gauge,
                    "from": current,
                    "to": expected,
                    "probe_ticks": horizon,
                }
            )
        current = expected
    if (
        dict(sorted(derived_modes.items())) != mode_counts
        or dict(sorted(derived_horizons.items())) != horizon_counts
        or derived_switches != switches
    ):
        raise ValueError(f"probe schedule aggregates differ for seed {seed}")
    log = row.get("query_log", [])
    if not isinstance(log, list):
        raise ValueError(f"query log is missing for seed {seed}")
    for logged, scheduled in zip(log, schedule):
        if not isinstance(logged, dict):
            raise ValueError(f"query log entry is malformed for seed {seed}")
        if any(
            logged.get(name) != scheduled.get(name)
            for name in ("tick", "gauge", "probe_mode", "probe_ticks")
        ):
            raise ValueError(f"logged query differs from probe schedule for seed {seed}")


def _verify_branch_errors(
    row: Mapping[str, Any], seed: int, planner_config: Mapping[str, Any]
) -> None:
    count = row.get("exact_branch_errors")
    events = row.get("exact_branch_error_events")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(events, list)
        or len(events) != count
        or row.get("exact_branch_error_events_sha256")
        != canonical_sha256(events)
    ):
        raise ValueError(f"branch-error evidence is inconsistent for seed {seed}")
    if int(row.get("selected_categories", {}).get("wait-exact-branch-error", 0)) != count:
        raise ValueError(f"branch-error selection count differs for seed {seed}")
    for event in events:
        try:
            event_horizon = _probe_mode_horizon(
                planner_config, event.get("probe_mode")
            )
        except ValueError as exc:
            raise ValueError(
                f"branch-error probe mode is malformed for seed {seed}"
            ) from exc
        if (
            not isinstance(event, dict)
            or event.get("exception_type")
            != "irisu_env.exact_ipc.ExactWorkerError"
            or event.get("recovery") != "restore-policy-and-execute-wait"
            or event.get("parent_state_unchanged") is not True
            or event.get("state_hash_before") != event.get("state_hash_after")
            or not isinstance(event.get("message"), str)
            or event.get("message_sha256")
            != hashlib.sha256(event["message"].encode("utf-8")).hexdigest()
            or event.get("probe_ticks") != event_horizon
        ):
            raise ValueError(f"branch-error event is malformed for seed {seed}")


def _expected_pass(report: Mapping[str, Any]) -> bool:
    episodes = report["episodes"]
    scores = [int(row["score"]) for row in episodes]
    return (
        len(episodes) == REQUIRED_EPISODES
        and sum(score >= TARGET_SCORE for score in scores) >= REQUIRED_SUCCESSES
        and statistics.median(scores) >= TARGET_SCORE
        and sum(int(row["invalid_actions"]) for row in episodes) == 0
    )


def verify_shard(
    path: Path,
    *,
    expected_format: str = SHARD_FORMAT,
    expected_contract: Mapping[str, object] = CONTRACT,
    expected_pass: bool | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    report = _read_json_object(path, "shard report")
    if report.get("format") != expected_format or report.get("physics_backend") != "exact":
        raise ValueError(f"not a FastMultiAction exact promotion shard: {path}")
    recorded_sha = report.get("report_content_sha256")
    content = dict(report)
    content.pop("report_content_sha256", None)
    content_sha = canonical_sha256(content)
    if recorded_sha != content_sha:
        raise ValueError(f"shard report content SHA-256 mismatch: {path}")
    if report.get("promotion_contract") != expected_contract:
        raise ValueError("shard promotion contract is not the expected canonical gate")
    if (
        report.get("deterministic_policy") is not True
        or report.get("branch_error_recovery_enabled") is not True
        or report.get("branch_error_policy") != BRANCH_ERROR_POLICY
    ):
        raise ValueError("shard policy/recovery identity is inconsistent")
    for field in IDENTITY_FIELDS:
        if field not in report:
            raise ValueError(f"shard is missing identity field {field}")
    planner_config = report.get("planner_config")
    horizon_fields = (
        "probe_ticks",
        "long_probe_ticks",
        "low_gauge_threshold",
        "low_gauge_exit_threshold",
    )
    if (
        not isinstance(planner_config, dict)
        or any(
            isinstance(planner_config.get(name), bool)
            or not isinstance(planner_config.get(name), int)
            or int(planner_config[name]) < (1 if "probe_ticks" in name else 0)
            for name in horizon_fields
        )
        or int(planner_config["long_probe_ticks"])
        < int(planner_config["probe_ticks"])
        or (
            int(planner_config["long_probe_ticks"])
            != int(planner_config["probe_ticks"])
            and int(planner_config["low_gauge_exit_threshold"])
            <= int(planner_config["low_gauge_threshold"])
        )
    ):
        raise ValueError("shard planner horizon configuration is malformed")
    for field in (
        "checkpoint_sha256",
        "model_sha256",
        "checkpoint_metadata_sha256",
        "planner_source_sha256",
        "evaluator_engine_sha256",
        "runner_sha256",
    ):
        if not _is_sha256(report[field]):
            raise ValueError(f"shard {field} is not a SHA-256")
    metadata = report.get("checkpoint_metadata")
    if (
        not isinstance(metadata, dict)
        or report["checkpoint_metadata_sha256"] != canonical_sha256(metadata)
    ):
        raise ValueError("shard checkpoint metadata SHA-256 mismatch")
    runtime_hashes = report.get("runtime_hashes")
    required_hashes = {
        "worker_sha256",
        "exact_library_sha256",
        "identity_config_sha256",
        "runtime_provenance_sha256",
    }
    if (
        not isinstance(runtime_hashes, dict)
        or not required_hashes.issubset(runtime_hashes)
        or any(not _is_sha256(runtime_hashes[name]) for name in required_hashes)
    ):
        raise ValueError("shard runtime hashes are malformed")
    exact_runtime = report.get("exact_runtime")
    if (
        not isinstance(exact_runtime, dict)
        or exact_runtime.get("physics_backend") != "exact"
        or runtime_hashes["runtime_provenance_sha256"]
        != canonical_sha256(exact_runtime)
    ):
        raise ValueError("shard exact runtime provenance SHA-256 mismatch")

    training = validate_seed_sequence(report.get("training_seeds", ()), "shard training")
    evaluation = validate_seed_sequence(
        report.get("evaluation_seeds", ()), "shard evaluation"
    )
    if report.get("training_seeds_sha256") != canonical_sha256(list(training)):
        raise ValueError("shard training seed SHA-256 mismatch")
    if report.get("evaluation_seeds_sha256") != canonical_sha256(list(evaluation)):
        raise ValueError("shard evaluation seed SHA-256 mismatch")
    if set(training) & set(evaluation) or report.get("training_evaluation_overlap") != []:
        raise ValueError("shard has training/evaluation seed leakage")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("shard episodes must be a nonempty list")
    if tuple(int(row.get("seed", -1)) for row in episodes) != evaluation:
        raise ValueError("shard episodes do not exactly match evaluation seeds")

    scores: list[int] = []
    invalids: list[int] = []
    counterfactual_invalids: list[int] = []
    episodes_by_seed: dict[int, Mapping[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError("shard episode row must be an object")
        seed = int(row["seed"])
        score = int(row["score"])
        policy_invalid = int(row["policy_invalid_actions"])
        simulator_invalid = int(row["simulator_invalid_actions"])
        invalid = int(row["invalid_actions"])
        counterfactual_invalid = int(row["counterfactual_invalid_actions"])
        if min(score, policy_invalid, simulator_invalid, invalid, counterfactual_invalid) < 0:
            raise ValueError(f"episode counters are malformed for seed {seed}")
        if invalid != policy_invalid + simulator_invalid:
            raise ValueError(f"live invalid count differs for seed {seed}")
        if row.get("success") is not (score >= TARGET_SCORE):
            raise ValueError(f"episode success flag differs for seed {seed}")
        trace = row.get("trace")
        if (
            not isinstance(trace, list)
            or not trace
            or row.get("trace_sha256") != canonical_sha256(trace)
        ):
            raise ValueError(f"episode trace identity differs for seed {seed}")
        if int(trace[-1]["tick"]) != int(row["tick"]):
            raise ValueError(f"episode trace final tick differs for seed {seed}")
        selected_categories = row.get("selected_categories")
        if (
            not isinstance(selected_categories, dict)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in selected_categories.values()
            )
            or sum(selected_categories.values()) != int(row["planner_queries"])
        ):
            raise ValueError(f"planner selection counts differ for seed {seed}")
        _verify_probe_aggregates(row, seed, planner_config)
        _verify_logged_queries(row, seed, planner_config)
        _verify_branch_errors(row, seed, planner_config)
        scores.append(score)
        invalids.append(invalid)
        counterfactual_invalids.append(counterfactual_invalid)
        episodes_by_seed[seed] = row

    successes = sum(score >= TARGET_SCORE for score in scores)
    aggregate_probe_modes: Counter[str] = Counter()
    aggregate_probe_horizons: Counter[str] = Counter()
    for row in episodes:
        aggregate_probe_modes.update(row["planner_probe_mode_counts"])
        aggregate_probe_horizons.update(row["planner_probe_horizon_counts"])
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
        "counterfactual_invalid_actions": sum(counterfactual_invalids),
        "planner_probe_mode_counts": dict(sorted(aggregate_probe_modes.items())),
        "planner_probe_horizon_counts": dict(
            sorted(aggregate_probe_horizons.items())
        ),
        "planner_probe_mode_switch_count": sum(
            len(row["planner_probe_mode_switches"]) for row in episodes
        ),
        "exact_branch_errors": sum(
            int(row["exact_branch_errors"]) for row in episodes
        ),
        "promotion_eligible": len(episodes) == REQUIRED_EPISODES,
        "passed": _expected_pass(report) if expected_pass is None else expected_pass,
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
    artifacts = _verify_artifacts(report, episodes_by_seed)
    return report, {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": content_sha,
        "evaluation_seeds": list(evaluation),
        "episode_artifacts_verified": len(artifacts),
    }


def merge_shards(
    shard_paths: Sequence[Path],
    requested_seeds: Sequence[int],
    *,
    merger_path: Path | None = None,
) -> dict[str, object]:
    requested = validate_seed_sequence(requested_seeds, "requested evaluation")
    if len(requested) != REQUIRED_EPISODES:
        raise ValueError(f"promotion requires exactly {REQUIRED_EPISODES} seeds")
    if not shard_paths:
        raise ValueError("at least one shard report is required")
    started = time.monotonic()
    shards: list[dict[str, Any]] = []
    evidence: list[dict[str, object]] = []
    identity: dict[str, object] | None = None
    by_seed: dict[int, dict[str, Any]] = {}
    artifact_evidence: list[dict[str, object]] = []
    for raw_path in shard_paths:
        path = raw_path.resolve(strict=True)
        shard, shard_evidence = verify_shard(path)
        current_identity = {field: shard[field] for field in IDENTITY_FIELDS}
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            differing = [
                field
                for field in IDENTITY_FIELDS
                if current_identity[field] != identity[field]
            ]
            raise ValueError(f"shard identities differ: {differing}")
        for row in shard["episodes"]:
            seed = int(row["seed"])
            if seed in by_seed:
                raise ValueError(f"shards contain duplicate evaluation seed {seed}")
            by_seed[seed] = dict(row)
        artifact_evidence.extend(
            dict(value) for value in shard.get("episode_artifacts", [])
        )
        shards.append(shard)
        evidence.append(shard_evidence)
    requested_set = set(requested)
    observed_set = set(by_seed)
    if observed_set != requested_set:
        raise ValueError(
            "shards do not exactly cover requested seeds; "
            f"missing={sorted(requested_set - observed_set)}, "
            f"unexpected={sorted(observed_set - requested_set)}"
        )
    episodes = [by_seed[seed] for seed in requested]
    scores = [int(row["score"]) for row in episodes]
    successes = sum(score >= TARGET_SCORE for score in scores)
    invalid = sum(int(row["invalid_actions"]) for row in episodes)
    median_score = float(statistics.median(scores))
    aggregate_probe_modes: Counter[str] = Counter()
    aggregate_probe_horizons: Counter[str] = Counter()
    for row in episodes:
        aggregate_probe_modes.update(row["planner_probe_mode_counts"])
        aggregate_probe_horizons.update(row["planner_probe_horizon_counts"])
    assert identity is not None
    report: dict[str, object] = {
        "format": FORMAT,
        "evaluator_format": SHARD_FORMAT,
        "physics_backend": "exact",
        "deterministic_policy": True,
        **identity,
        "branch_error_recovery_enabled": True,
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
        "counterfactual_invalid_actions": sum(
            int(row["counterfactual_invalid_actions"]) for row in episodes
        ),
        "planner_probe_mode_counts": dict(sorted(aggregate_probe_modes.items())),
        "planner_probe_horizon_counts": dict(
            sorted(aggregate_probe_horizons.items())
        ),
        "planner_probe_mode_switch_count": sum(
            len(row["planner_probe_mode_switches"]) for row in episodes
        ),
        "exact_branch_errors": sum(
            int(row["exact_branch_errors"]) for row in episodes
        ),
        "promotion_eligible": True,
        "passed": successes >= REQUIRED_SUCCESSES
        and median_score >= TARGET_SCORE
        and invalid == 0,
        "merger_sha256": file_sha256(
            Path(__file__).resolve()
            if merger_path is None
            else merger_path.resolve(strict=True)
        ),
        "wall_seconds": time.monotonic() - started,
    }
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        parser().error("--output must not already exist")
    requested = load_requested_seeds(
        args.evaluation_seed_manifest.resolve(strict=True)
    )
    report = merge_shards(args.shard, requested)
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
