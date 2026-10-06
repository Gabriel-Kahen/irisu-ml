"""Development-only deterministic scheduler for staged exact lookahead."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .fast_multiaction_planner import (
    CandidateOutcome,
    FastMultiActionPlanner,
    PlannerCandidate,
    visible_rot_liability,
)
from .branching import TransactionalBranches
from .g5_solvency_trigger import sha256


SCHEMA = "irisu-g5-motion-size-hazard-scheduler-v2"


def hazard_counts(observation: Mapping[str, Any]) -> tuple[int, int]:
    bodies = observation.get("bodies")
    if type(bodies) is not list:
        raise ValueError("G5 scheduler observation bodies are malformed")
    if any(type(body) is not dict for body in bodies):
        raise ValueError("G5 scheduler observation body is malformed")
    moving = sum(abs(float(body["vx"])) + abs(float(body["vy"])) >= 20.0 for body in bodies)
    large = sum(float(body["size"]) >= 60.0 for body in bodies)
    return moving, large


@dataclass(frozen=True, slots=True)
class HazardScheduler:
    moving_threshold: int
    large_threshold: int
    training_seeds: tuple[int, ...]
    dataset_sha256: str
    provenance: tuple[tuple[str, str], ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "role": "compute-trigger-only-never-selects-or-prunes",
            "features": [
                "count(abs(vx)+abs(vy)>=20)", "count(size>=60)"
            ],
            "fit_rule": "minimum-compute-positive-cover-grid:M<=m OR L<=l",
            "moving_threshold": self.moving_threshold,
            "large_threshold": self.large_threshold,
            "training_seeds": list(self.training_seeds),
            "dataset_sha256": self.dataset_sha256,
            "provenance": [list(row) for row in self.provenance],
        }

    @property
    def sha256(self) -> str:
        return sha256(self.manifest())

    def should_compute(self, observation: Mapping[str, Any]) -> bool:
        moving, large = hazard_counts(observation)
        return moving <= self.moving_threshold or large <= self.large_threshold


def fit_scheduler(
    observations: Sequence[Mapping[str, Any]], labels: Sequence[bool],
    seeds: Sequence[int], *, dataset_sha256: str,
    provenance: Mapping[str, str], maximum_compute_fraction: float = 0.90,
) -> tuple[HazardScheduler, dict[str, object]]:
    counts = np.asarray([hazard_counts(row) for row in observations], dtype=np.int64)
    target = np.asarray(labels, dtype=np.bool_)
    seed = np.asarray(seeds, dtype=np.int64)
    if counts.shape != (len(target), 2) or seed.shape != target.shape or not target.any():
        raise ValueError("G5 scheduler supervision is malformed")
    def choose(train: np.ndarray) -> tuple[int, int]:
        candidates = []
        for moving in range(int(np.max(counts[:, 0])) + 1):
            for large in range(int(np.max(counts[:, 1])) + 1):
                selected = (counts[:, 0] <= moving) | (counts[:, 1] <= large)
                if selected[train & target].all():
                    candidates.append((int(np.sum(selected[train])), moving, large))
        if not candidates:
            raise RuntimeError("G5 scheduler grid has no positive-cover rule")
        _selected, moving, large = min(candidates)
        return moving, large
    folds = []
    oof = np.zeros(len(seed), dtype=np.bool_)
    for heldout in sorted(set(int(value) for value in seed)):
        train = seed != heldout
        test = ~train
        moving, large = choose(train)
        oof[test] = (counts[test, 0] <= moving) | (counts[test, 1] <= large)
        folds.append({
            "heldout_seed": heldout,
            "moving_threshold": moving,
            "large_threshold": large,
            "positives": int(np.sum(target[test])),
            "recalled": int(np.sum(oof[test] & target[test])),
            "queries": int(np.sum(test)),
            "triggers": int(np.sum(oof[test])),
        })
    recall = float(np.sum(oof & target) / np.sum(target))
    compute = float(np.mean(oof))
    if recall != 1.0 or compute > maximum_compute_fraction:
        raise RuntimeError("G5 deterministic scheduler failed OOF gate")
    moving, large = choose(np.ones(len(seed), dtype=np.bool_))
    scheduler = HazardScheduler(
        moving, large,
        tuple(sorted(set(int(value) for value in seed))),
        dataset_sha256,
        tuple(sorted((str(key), str(value)) for key, value in provenance.items())),
    )
    return scheduler, {
        "schema": "irisu-g5-hazard-scheduler-oof-v1",
        "recall": recall,
        "compute_fraction": compute,
        "positives": int(np.sum(target)),
        "queries": len(target),
        "folds": folds,
    }


def robust_better(candidate: CandidateOutcome, base: CandidateOutcome, horizon: int) -> bool:
    a, b = candidate.probe, base.probe
    if a.survival_ticks > b.survival_ticks:
        return a.minimum_gauge >= b.minimum_gauge
    return bool(
        a.survival_ticks == b.survival_ticks == horizon
        and a.minimum_gauge >= b.minimum_gauge + 1_000
        and a.final_gauge >= b.final_gauge + 1_000
        and a.clears >= b.clears
        and a.score >= b.score
    )


@dataclass(frozen=True, slots=True)
class StagedExactVerdict:
    selected: PlannerCandidate
    base: PlannerCandidate
    compute_long: bool
    evaluated_ordinals: tuple[tuple[int, tuple[int, ...]], ...]
    scheduler_sha256: str
    candidate_inventory_sha256: str
    reason: str
    branch_checks: int = 0
    used_fast_checkpoint: bool = False
    source_state_hash: int | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-g5-staged-exact-verdict-v1",
            "selected_ordinal": self.selected.ordinal,
            "base_ordinal": self.base.ordinal,
            "compute_long": self.compute_long,
            "evaluated_ordinals": [
                [horizon, list(ordinals)] for horizon, ordinals in self.evaluated_ordinals
            ],
            "scheduler_sha256": self.scheduler_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "reason": self.reason,
            "branch_checks": self.branch_checks,
            "used_fast_checkpoint": self.used_fast_checkpoint,
            "source_state_hash": self.source_state_hash,
        }


def staged_exact_select(
    planner: FastMultiActionPlanner,
    observation: Mapping[str, Any],
    candidates: Sequence[PlannerCandidate],
    evaluator: Callable[[tuple[PlannerCandidate, ...], int], tuple[CandidateOutcome, ...]],
    scheduler: HazardScheduler,
) -> StagedExactVerdict:
    frozen = tuple(candidates)
    if tuple(value.ordinal for value in frozen) != tuple(range(len(frozen))):
        raise ValueError("G5 frozen candidate ordinals are not closed")
    inventory = sha256([
        {
            "ordinal": value.ordinal,
            "category": value.category,
            "decision": {
                "kind": int(value.decision.action.kind),
                "x_norm": float(value.decision.action.x_norm),
                "y_norm": float(value.decision.action.y_norm),
                "source_body_id": value.decision.source_body_id,
                "destination_body_id": value.decision.destination_body_id,
            },
        }
        for value in frozen
    ])

    def evaluate(subset: tuple[PlannerCandidate, ...], horizon: int):
        outcomes = evaluator(subset, horizon)
        if len(outcomes) != len(subset) or any(
            outcome.candidate is not candidate
            for outcome, candidate in zip(outcomes, subset, strict=True)
        ):
            raise RuntimeError("G5 exact evaluator changed frozen candidate identity")
        return outcomes

    short = evaluate(frozen, 2_048)
    wait = next(value for value in short if value.candidate.category == "wait")
    eligible = [
        value for value in short
        if value is not wait
        and not value.candidate.category.startswith("top-")
        and planner._eligible(value, wait)[0]
    ]
    base = max(eligible, key=planner._objective) if eligible else wait
    try:
        compute_long = scheduler.should_compute(observation)
    except (KeyError, TypeError, ValueError, OverflowError):
        # A malformed feature projection must never suppress exact computation.
        compute_long = True
    if not compute_long:
        return StagedExactVerdict(
            base.candidate, base.candidate, False,
            ((2_048, tuple(value.candidate.ordinal for value in short)),),
            scheduler.sha256, inventory, "short-exact-base-fallback",
        )
    contenders = tuple(sorted(
        (value for value in short if value is not base),
        key=planner._objective, reverse=True,
    )[:2])
    selected = (base.candidate, *(value.candidate for value in contenders))
    try:
        long8 = evaluate(selected, 8_192)
        long12 = evaluate(selected, 12_288)
    except Exception:
        # The 2,048-tick exact base is immutable and remains the only safe
        # verdict if an optional long stage fails.  Runtime callers separately
        # verify that the live parent state is unchanged before accepting it.
        return StagedExactVerdict(
            base.candidate, base.candidate, True,
            ((2_048, tuple(value.candidate.ordinal for value in short)),),
            scheduler.sha256, inventory, "long-exact-failure-retains-base",
        )
    by8 = {value.candidate.ordinal: value for value in long8}
    by12 = {value.candidate.ordinal: value for value in long12}
    safe = [
        contender for contender in contenders
        if not robust_better(contender, base, 2_048)
        and robust_better(by8[contender.candidate.ordinal], by8[base.candidate.ordinal], 8_192)
        and robust_better(by12[contender.candidate.ordinal], by12[base.candidate.ordinal], 12_288)
    ]
    winner = max(safe, key=lambda value: planner._objective(by12[value.candidate.ordinal])) if safe else base
    return StagedExactVerdict(
        winner.candidate, base.candidate, True,
        (
            (2_048, tuple(value.candidate.ordinal for value in short)),
            (8_192, tuple(value.candidate.ordinal for value in long8)),
            (12_288, tuple(value.candidate.ordinal for value in long12)),
        ),
        scheduler.sha256, inventory,
        "long-exact-safe-disagreement" if safe else "long-exact-retains-base",
    )


def evaluate_scheduled_exact(
    planner: FastMultiActionPlanner,
    env: object,
    observation: Mapping[str, Any],
    policy_before: object,
    policy_after_prediction: object,
    prediction: object,
    scheduler: HazardScheduler,
) -> StagedExactVerdict:
    """Run the staged exact scheduler against one immutable COW parent.

    Candidate generation occurs exactly once.  The scheduler only controls the
    optional 8,192/12,288 stages; all candidate choices are exact-rule choices.
    """

    state_hash = getattr(env, "state_hash", None)
    if not callable(state_hash):
        raise TypeError("G5 staged exact runtime requires state_hash")
    source_hash = int(state_hash())
    frozen = planner.candidates(
        observation, policy_before, policy_after_prediction, prediction
    )
    source_clears = int(observation.get("qualifying_clear_count", 0))
    checks = 0

    with TransactionalBranches(env, observation) as branches:
        if not branches.uses_fast_checkpoint:
            raise RuntimeError("G5 staged exact runtime requires fast checkpoints")

        def evaluate(
            subset: tuple[PlannerCandidate, ...], horizon: int
        ) -> tuple[CandidateOutcome, ...]:
            nonlocal checks
            outcomes = []
            for candidate in subset:
                if not any(candidate is known for known in frozen):
                    raise RuntimeError("G5 staged exact evaluator received foreign candidate")
                with branches.branch() as (branch, branch_observation):
                    if int(branch.state_hash()) != source_hash:
                        raise RuntimeError("G5 staged exact branch state mismatch")
                    checks += 1
                    probe, invalid, liability = planner._advance(
                        branch,
                        branch_observation,
                        copy.deepcopy(candidate.continuation_policy),
                        candidate.decision,
                        horizon,
                    )
                    outcomes.append(CandidateOutcome(
                        candidate,
                        probe,
                        invalid,
                        imminent_visible_rot_liability=liability,
                        renewal_clears=max(0, probe.clears - source_clears),
                    ))
            return tuple(outcomes)

        verdict = staged_exact_select(
            planner, observation, frozen, evaluate, scheduler
        )

    if int(state_hash()) != source_hash:
        raise RuntimeError("G5 staged exact runtime altered the live parent")
    checks += 1
    return replace(
        verdict,
        branch_checks=checks,
        used_fast_checkpoint=True,
        source_state_hash=source_hash,
    )


__all__ = [
    "HazardScheduler", "StagedExactVerdict", "hazard_counts",
    "fit_scheduler", "robust_better", "staged_exact_select",
    "evaluate_scheduled_exact",
]
