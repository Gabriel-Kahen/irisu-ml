#!/usr/bin/env python3
"""Verify and merge exact shards for the 200k linear-p80 promotion gate."""

from __future__ import annotations

import argparse
import json
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

import rl_exact_fast_multiaction_promotion_merge as base  # noqa: E402
from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    validate_seed_sequence,
)
from rl_exact_fast_multiaction_p80_eval import FORMAT as SHARD_FORMAT  # noqa: E402
from rl_exact_p80_contract import (  # noqa: E402
    CONTRACT,
    REQUIRED_EPISODES,
    evaluate_contract,
)


FORMAT = "irisu-exact-fast-multiaction-p80-promotion-v1"
P80_SUMMARY_FIELDS = (
    "baseline_success_count",
    "baseline_success_fraction_at_or_above_target",
    "p80_score_linear_type7",
    "p80_score_linear_type7_exact",
    "p80_nearest_rank_score",
    "scores_at_or_above_200k",
    "promotion_eligible",
    "passed",
)


def _gate_for_report(report: Mapping[str, Any]) -> dict[str, object]:
    scores = report.get("scores")
    if not isinstance(scores, list):
        raise ValueError("shard scores are missing")
    return evaluate_contract(
        scores, invalid_actions=int(report.get("invalid_actions", -1))
    )


def verify_p80_shard(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    raw = base._read_json_object(path, "p80 shard report")
    gate = _gate_for_report(raw)
    report, evidence = base.verify_shard(
        path,
        expected_format=SHARD_FORMAT,
        expected_contract=CONTRACT,
        expected_pass=gate["passed"],
    )
    for field in P80_SUMMARY_FIELDS:
        if report.get(field) != gate[field]:
            raise ValueError(f"p80 shard summary field {field} is inconsistent")
    if report.get("success_count") != gate["baseline_success_count"]:
        raise ValueError("p80 shard baseline success count is inconsistent")
    return report, evidence


def merge_p80_shards(
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
        shard, shard_evidence = verify_p80_shard(path)
        current_identity = {
            field: shard[field] for field in base.IDENTITY_FIELDS
        }
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            differing = [
                field
                for field in base.IDENTITY_FIELDS
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
    requested_set, observed_set = set(requested), set(by_seed)
    if observed_set != requested_set:
        raise ValueError(
            "shards do not exactly cover requested seeds; "
            f"missing={sorted(requested_set - observed_set)}, "
            f"unexpected={sorted(observed_set - requested_set)}"
        )
    episodes = [by_seed[seed] for seed in requested]
    scores = [int(row["score"]) for row in episodes]
    invalid = sum(int(row["invalid_actions"]) for row in episodes)
    gate = evaluate_contract(scores, invalid_actions=invalid)
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
        "median_score": float(statistics.median(scores)),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": gate["baseline_success_count"],
        "success_fraction_at_or_above_target": gate[
            "baseline_success_fraction_at_or_above_target"
        ],
        **{field: gate[field] for field in P80_SUMMARY_FIELDS},
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
        "merger_sha256": file_sha256(
            Path(__file__).resolve()
            if merger_path is None
            else merger_path.resolve(strict=True)
        ),
        "wall_seconds": time.monotonic() - started,
    }
    report["promotion_report_content_sha256"] = canonical_sha256(report)
    return report


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
    requested = base.load_requested_seeds(
        args.evaluation_seed_manifest.resolve(strict=True)
    )
    report = merge_p80_shards(args.shard, requested)
    base._write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
