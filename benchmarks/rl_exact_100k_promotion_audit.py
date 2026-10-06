#!/usr/bin/env python3
"""Offline diagnostics for a completed exact adaptive promotion report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _group_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not episodes:
        return {"episodes": 0}
    attempts = sum(int(row["attempted_shots"]) for row in episodes)
    reasons: Counter[str] = Counter()
    for row in episodes:
        reasons.update({key: int(value) for key, value in row["gate_reasons"].items()})
    return {
        "episodes": len(episodes),
        "mean_score": statistics.fmean(int(row["score"]) for row in episodes),
        "mean_survival_ticks": statistics.fmean(int(row["tick"]) for row in episodes),
        "mean_level": statistics.fmean(int(row["level"]) for row in episodes),
        "mean_score_per_tick": statistics.fmean(
            int(row["score"]) / max(int(row["tick"]), 1) for row in episodes
        ),
        "mean_attempted_shots_per_1000_ticks": statistics.fmean(
            int(row["attempted_shots"]) * 1000 / max(int(row["tick"]), 1)
            for row in episodes
        ),
        "gate_reason_fractions": {
            key: value / attempts for key, value in sorted(reasons.items())
        },
    }


def analyze(report: Mapping[str, Any], *, target_score: int = 100_000) -> dict[str, object]:
    if report.get("physics_backend") != "exact":
        raise ValueError("audit input must be exact-runtime evidence")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("audit input must contain episodes")
    if target_score < 1:
        raise ValueError("target_score must be positive")

    below = [row for row in episodes if int(row["score"]) < target_score]
    reached = [row for row in episodes if int(row["score"]) >= target_score]
    scores = [float(row["score"]) for row in episodes]
    ticks = [float(row["tick"]) for row in episodes]
    levels = [float(row["level"]) for row in episodes]
    final_gauge = [int(row["gauge"]) for row in episodes]

    regular_trace_ticks = sorted(
        {
            int(point["tick"])
            for row in episodes
            for point in row["trace"]
            if int(point["tick"]) > 0 and int(point["tick"]) % 10_000 == 0
        }
    )
    survival_curve = []
    for tick in regular_trace_ticks:
        points = [
            point
            for row in episodes
            for point in row["trace"]
            if int(point["tick"]) == tick
        ]
        survival_curve.append(
            {
                "tick": tick,
                "episodes_alive": sum(int(row["tick"]) >= tick for row in episodes),
                "observed_checkpoints": len(points),
                "median_score": float(
                    statistics.median(int(point["score"]) for point in points)
                ),
                "median_gauge": float(
                    statistics.median(int(point["gauge"]) for point in points)
                ),
                "gauge_at_or_below_half_count": sum(
                    int(point["gauge"]) <= 20_000 for point in points
                ),
            }
        )

    # At normal level L, a rot costs 1800+20L. A final value of
    # 1-(1800+20L) therefore proves a rot struck from the one-unit floor.
    rot_at_floor = [
        int(row["seed"])
        for row in episodes
        if int(row["gauge"]) == 1 - (1800 + 20 * int(row["level"]))
    ]
    exhausted = [
        int(row["seed"])
        for row in episodes
        if bool(row["terminated"]) and int(row["gauge"]) <= 1
    ]
    planner = report.get("planner_config", {})
    return {
        "schema": "irisu-exact-100k-promotion-audit-v1",
        "source_format": report.get("format"),
        "source_report_sha256": report.get("promotion_report_content_sha256"),
        "target_score": target_score,
        "episode_count": len(episodes),
        "reached_target_count": len(reached),
        "below_target_count": len(below),
        "score_tick_pearson": _pearson(scores, ticks),
        "score_level_pearson": _pearson(scores, levels),
        "minimum_survival_tick_among_target_reachers": (
            min(int(row["tick"]) for row in reached) if reached else None
        ),
        "maximum_survival_tick_below_target": (
            max(int(row["tick"]) for row in below) if below else None
        ),
        "terminated_at_nonpositive_or_floor_gauge_count": len(exhausted),
        "terminal_rot_from_floor_seeds": rot_at_floor,
        "final_gauge_counts": dict(sorted(Counter(final_gauge).items())),
        "planner_horizons_are_identical": (
            planner.get("short_horizon") == planner.get("long_horizon")
        ),
        "below_target": _group_metrics(below),
        "reached_target": _group_metrics(reached),
        "survival_curve": survival_curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-score", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze(report, target_score=args.target_score)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
