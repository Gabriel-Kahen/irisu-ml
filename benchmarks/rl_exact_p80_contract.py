#!/usr/bin/env python3
"""Pure, exact promotion gate for the 200k p80 objective."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction


REQUIRED_EPISODES = 20
BASELINE_TARGET_SCORE = 100_000
REQUIRED_BASELINE_SUCCESSES = 16
P80_TARGET_SCORE = 200_000
P80_NUMERATOR = 4
P80_DENOMINATOR = 5

# With 20 samples, nearest-rank p80 is ascending rank 16.  Requiring that
# observation to reach 200k guarantees at least five observations at 200k;
# a large 17th score cannot manufacture a pass by interpolation.
P80_ORDER_STATISTIC_ASCENDING_RANK = math.ceil(
    P80_NUMERATOR * REQUIRED_EPISODES / P80_DENOMINATOR
)
REQUIRED_P80_TARGET_COUNT = (
    REQUIRED_EPISODES - P80_ORDER_STATISTIC_ASCENDING_RANK + 1
)

CONTRACT: dict[str, object] = {
    "version": "irisu-exact-p80-200k-contract-v1",
    "required_episode_count": REQUIRED_EPISODES,
    "baseline_target_score": BASELINE_TARGET_SCORE,
    "required_baseline_success_count": REQUIRED_BASELINE_SUCCESSES,
    "require_median_at_or_above_baseline_target": True,
    "headline_percentile": {
        "name": "p80_score_linear_type7",
        "probability_numerator": P80_NUMERATOR,
        "probability_denominator": P80_DENOMINATOR,
        "method": "Hyndman-Fan-type-7-linear",
        "target_score": P80_TARGET_SCORE,
    },
    "robust_order_statistic_companion": {
        "name": "p80_nearest_rank_score",
        "ascending_rank": P80_ORDER_STATISTIC_ASCENDING_RANK,
        "target_score": P80_TARGET_SCORE,
        "minimum_scores_at_or_above_target": REQUIRED_P80_TARGET_COUNT,
    },
    "require_zero_live_invalid_actions": True,
}


def _validated_scores(scores: Sequence[int]) -> tuple[int, ...]:
    values = tuple(scores)
    if not values or any(
        isinstance(score, bool) or not isinstance(score, int) or score < 0
        for score in values
    ):
        raise ValueError("scores must be a nonempty sequence of nonnegative integers")
    return values


def linear_percentile_type7(
    scores: Sequence[int], *, numerator: int, denominator: int
) -> Fraction:
    """Return a Hyndman-Fan type-7 quantile without floating-point drift."""

    values = sorted(_validated_scores(scores))
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator < 1
        or not 0 <= numerator <= denominator
    ):
        raise ValueError("percentile probability must be an integer fraction in [0, 1]")
    position = Fraction((len(values) - 1) * numerator, denominator)
    lower = position.numerator // position.denominator
    remainder = position - lower
    if lower == len(values) - 1:
        return Fraction(values[lower])
    return Fraction(values[lower]) + remainder * (values[lower + 1] - values[lower])


def p80_linear(scores: Sequence[int]) -> Fraction:
    return linear_percentile_type7(
        scores, numerator=P80_NUMERATOR, denominator=P80_DENOMINATOR
    )


def p80_nearest_rank(scores: Sequence[int]) -> int:
    values = sorted(_validated_scores(scores))
    rank = math.ceil(P80_NUMERATOR * len(values) / P80_DENOMINATOR)
    return values[max(1, rank) - 1]


def evaluate_contract(
    scores: Sequence[int], *, invalid_actions: int
) -> dict[str, object]:
    values = _validated_scores(scores)
    if (
        isinstance(invalid_actions, bool)
        or not isinstance(invalid_actions, int)
        or invalid_actions < 0
    ):
        raise ValueError("invalid_actions must be a nonnegative integer")
    linear = p80_linear(values)
    order_statistic = p80_nearest_rank(values)
    baseline_successes = sum(score >= BASELINE_TARGET_SCORE for score in values)
    p80_target_count = sum(score >= P80_TARGET_SCORE for score in values)
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        Fraction(ordered[midpoint])
        if len(ordered) % 2
        else Fraction(ordered[midpoint - 1] + ordered[midpoint], 2)
    )
    eligible = len(values) == REQUIRED_EPISODES
    passed = (
        eligible
        and baseline_successes >= REQUIRED_BASELINE_SUCCESSES
        and median >= BASELINE_TARGET_SCORE
        and linear >= P80_TARGET_SCORE
        and order_statistic >= P80_TARGET_SCORE
        and p80_target_count >= REQUIRED_P80_TARGET_COUNT
        and invalid_actions == 0
    )
    return {
        "episode_count": len(values),
        "scores": list(values),
        "median_score": float(median),
        "baseline_success_count": baseline_successes,
        "baseline_success_fraction_at_or_above_target": baseline_successes
        / len(values),
        "p80_score_linear_type7": float(linear),
        "p80_score_linear_type7_exact": {
            "numerator": linear.numerator,
            "denominator": linear.denominator,
        },
        "p80_nearest_rank_score": order_statistic,
        "scores_at_or_above_200k": p80_target_count,
        "invalid_actions": invalid_actions,
        "promotion_eligible": eligible,
        "passed": passed,
    }


__all__ = [
    "BASELINE_TARGET_SCORE",
    "CONTRACT",
    "P80_ORDER_STATISTIC_ASCENDING_RANK",
    "P80_TARGET_SCORE",
    "REQUIRED_BASELINE_SUCCESSES",
    "REQUIRED_EPISODES",
    "REQUIRED_P80_TARGET_COUNT",
    "evaluate_contract",
    "linear_percentile_type7",
    "p80_linear",
    "p80_nearest_rank",
]
