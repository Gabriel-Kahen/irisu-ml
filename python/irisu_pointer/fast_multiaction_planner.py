"""Bounded counterfactual planner for learned steering decisions.

The planner keeps the learned policy's cadence and geometry, but tests both
shot strengths, restraint, and (at low gauge) a few model-ranked alternative
directed pairs.  Branches are isolated through :class:`TransactionalBranches`;
exact workers therefore use fork/COW instead of replaying the live action log.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np
import torch

from irisu_pointer.branching import TransactionalBranches
from irisu_pointer.development_reserve_band import (
    ProbeCandidate as ReserveProbeCandidate,
    ReserveBandConfig,
    choose_candidate as choose_reserve_candidate,
)
from irisu_pointer.policy import encoded_body_ids
from irisu_pointer.shot_necessity import ProbeOutcome, choose_shot
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind


@dataclass(frozen=True, slots=True)
class FastMultiActionConfig:
    probe_ticks: int = 256
    long_probe_ticks: int = 512
    wait_ticks: int = 16
    low_gauge_threshold: int = 20_000
    low_gauge_exit_threshold: int = 30_000
    long_probe_min_tick: int = 0
    top_k_pairs: int = 2
    maximum_gauge_debt: int = 1_000
    rescue_score_margin: int = 500
    gauge_advantage: int = 1
    objective_mode: str = "wait-relative"
    reserve_contingency_gauge: int = 0
    robust_reserve_margin: int = 1_000
    robust_score_trade: int = 500
    rot_delay_ticks: int = 40

    def __post_init__(self) -> None:
        positive = (
            "probe_ticks",
            "long_probe_ticks",
            "wait_ticks",
            "maximum_gauge_debt",
            "rescue_score_margin",
            "gauge_advantage",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "low_gauge_threshold",
            "low_gauge_exit_threshold",
            "top_k_pairs",
            "reserve_contingency_gauge",
            "long_probe_min_tick",
            "robust_reserve_margin",
            "robust_score_trade",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.long_probe_ticks < self.probe_ticks:
            raise ValueError("long_probe_ticks must be at least probe_ticks")
        if (
            self.long_probe_ticks != self.probe_ticks
            and self.low_gauge_exit_threshold <= self.low_gauge_threshold
        ):
            raise ValueError(
                "low_gauge_exit_threshold must exceed low_gauge_threshold "
                "when probe horizons differ"
            )
        if self.objective_mode not in {
            "wait-relative",
            "reserve-band",
            "robust-reserve-tie",
            "robust-reserve-bounded",
            "chain-first",
        }:
            raise ValueError(
                "objective_mode must be wait-relative, reserve-band, "
                "robust-reserve-tie, robust-reserve-bounded, or chain-first"
            )
        if (
            isinstance(self.rot_delay_ticks, bool)
            or not isinstance(self.rot_delay_ticks, int)
            or self.rot_delay_ticks < 1
        ):
            raise ValueError("rot_delay_ticks must be a positive integer")

    def manifest(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    ordinal: int
    category: str
    decision: SteeringDecision
    continuation_policy: object


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    candidate: PlannerCandidate
    probe: ProbeOutcome
    invalid_actions: int
    imminent_visible_rot_liability: int = 0
    renewal_clears: int = 0

    @property
    def valid(self) -> bool:
        return self.invalid_actions == 0

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.candidate.ordinal,
            "category": self.candidate.category,
            "decision": _decision_manifest(self.candidate.decision),
            "probe": self.probe.manifest(),
            "invalid_actions": self.invalid_actions,
            "imminent_visible_rot_liability": self.imminent_visible_rot_liability,
            "renewal_clears": self.renewal_clears,
        }


@dataclass(frozen=True, slots=True)
class MultiActionVerdict:
    selected: PlannerCandidate
    reason: str
    outcomes: tuple[CandidateOutcome, ...]
    branch_checks: int
    used_fast_checkpoint: bool
    objective_mode: str = "wait-relative"
    objective_evidence: Mapping[str, object] | None = None
    probe_mode: str = "normal"
    probe_ticks: int = 256

    def manifest(self) -> dict[str, object]:
        return {
            "selected_ordinal": self.selected.ordinal,
            "selected_category": self.selected.category,
            "reason": self.reason,
            "outcomes": [value.manifest() for value in self.outcomes],
            "branch_checks": self.branch_checks,
            "used_fast_checkpoint": self.used_fast_checkpoint,
            "objective_mode": self.objective_mode,
            "objective_evidence": dict(self.objective_evidence or {}),
            "probe_mode": self.probe_mode,
            "probe_ticks": self.probe_ticks,
        }


def _decision_manifest(decision: SteeringDecision) -> dict[str, object]:
    action = decision.action
    return {
        "kind": int(action.kind),
        "wait_ticks": int(action.wait_ticks),
        "x_norm": float(action.x_norm),
        "y_norm": float(action.y_norm),
        "source_body_id": decision.source_body_id,
        "destination_body_id": decision.destination_body_id,
        "intent": decision.intent.value,
    }


def _inner_policy(policy: object) -> object:
    """Return the mutable controller beneath a shared-model wrapper."""

    current = policy
    seen: set[int] = set()
    while hasattr(current, "inner") and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "inner")
    return current


def _copy_policy(policy: object) -> object:
    return copy.deepcopy(policy)


def _set_last_decision(policy: object, decision: SteeringDecision) -> bool:
    inner = _inner_policy(policy)
    if not hasattr(inner, "_last_decision"):
        return False
    setattr(inner, "_last_decision", decision)
    return True


def _rebind_policy(
    policy: object,
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
    selected: SteeringDecision,
) -> bool:
    """Rebind a private policy copy to a selected pair without touching weights."""

    inner = _inner_policy(policy)
    if selected.source_body_id is None or selected.destination_body_id is None:
        return False
    pair_changed = (
        selected.source_body_id != incumbent.source_body_id
        or selected.destination_body_id != incumbent.destination_body_id
    )
    if pair_changed:
        tracker = getattr(inner, "_progress", None)
        begin = getattr(tracker, "begin", None)
        pending = getattr(tracker, "pending_pair", None)
        if not callable(begin):
            return False
        pending_pair = (
            None if pending is None else (pending.source_id, pending.destination_id)
        )
        incumbent_pair = (
            incumbent.source_body_id,
            incumbent.destination_body_id,
        )
        selected_pair = (
            selected.source_body_id,
            selected.destination_body_id,
        )
        if pending_pair != selected_pair:
            if pending_pair != incumbent_pair or not hasattr(tracker, "_attempt"):
                return False
            previous_attempt = tracker._attempt
            tracker._attempt = None
            try:
                begin(observation, *selected_pair)
            except (RuntimeError, TypeError, ValueError):
                tracker._attempt = previous_attempt
                return False
    return _set_last_decision(policy, selected)


def _with_strength(
    decision: SteeringDecision, kind: SemanticActionKind, reason: str
) -> SteeringDecision:
    action = decision.action
    if kind is SemanticActionKind.FIRE_WEAK:
        semantic = SemanticAction.weak(action.x_norm, action.y_norm)
    elif kind is SemanticActionKind.FIRE_STRONG:
        semantic = SemanticAction.strong(action.x_norm, action.y_norm)
    else:
        raise ValueError("shot strength must be weak or strong")
    return replace(decision, action=semantic, reason=reason)


def _bodies(observation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(body["id"]): body
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping)
        and isinstance(body.get("id"), Integral)
        and not isinstance(body.get("id"), bool)
    }


def _safe_pair(
    policy: object,
    observation: Mapping[str, Any],
    source_id: int,
    destination_id: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    bodies = _bodies(observation)
    source = bodies.get(source_id)
    destination = bodies.get(destination_id)
    if source is None or destination is None:
        return None
    source_lifecycle = str(source.get("lifecycle", ""))
    source_safe = (
        source.get("kind") in {"piece", "bonus"}
        and int(source.get("chain_id", 0)) == 0
        and (
            source.get("kind") == "bonus"
            or source_lifecycle
            in {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
        )
    )
    destination_safe = (
        destination.get("kind") == "piece"
        and str(destination.get("lifecycle", "")) != "deleted"
    )
    same_color = (
        source.get("kind") == "bonus"
        or source.get("color") == destination.get("color")
    )
    tracker = getattr(_inner_policy(policy), "_progress", None)
    is_stalled = getattr(tracker, "is_stalled", None)
    stalled = bool(
        callable(is_stalled) and is_stalled(observation, source_id, destination_id)
    )
    if not (source_safe and destination_safe and same_color and not stalled):
        return None
    return source, destination


@torch.no_grad()
def _ranked_alternatives(
    policy: object,
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
    maximum_pairs: int,
) -> tuple[SteeringDecision, ...]:
    if maximum_pairs <= 0:
        return ()
    inner = _inner_policy(policy)
    encoder = getattr(inner, "encoder")
    model = getattr(inner, "model")
    encoded = encoder.encode([observation])
    active = np.flatnonzero(encoded.body_mask[0])
    width = int(active[-1]) + 1 if active.size else 1
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    output = model(
        torch.from_numpy(encoded.global_features).to(device),
        torch.from_numpy(encoded.body_features[:, :width]).to(device),
        torch.from_numpy(encoded.body_mask[:, :width]).to(device),
    )
    identifiers = list(encoded_body_ids(encoded, observation)[:width])
    flat = output.legal_pair_mask[0].flatten().nonzero(as_tuple=False).reshape(-1)
    scores = output.pair_logits[0].flatten()[flat]
    ranked = flat[scores.argsort(descending=True, stable=True)]
    incumbent_pair = (incumbent.source_body_id, incumbent.destination_body_id)
    results: list[SteeringDecision] = []
    for raw in ranked.tolist():
        source_index, destination_index = divmod(int(raw), width)
        source_id = identifiers[source_index]
        destination_id = identifiers[destination_index]
        if source_id is None or destination_id is None:
            continue
        if (source_id, destination_id) == incumbent_pair:
            continue
        pair = _safe_pair(inner, observation, source_id, destination_id)
        if pair is None:
            continue
        analytic = inner._analytic_action(*pair)
        if analytic is None:
            continue
        action, impact_x, impact_y = analytic
        destination = pair[1]
        results.append(
            SteeringDecision(
                action,
                SteeringIntent.MATCH_ROTTEN
                if str(destination.get("lifecycle", "")) == "rotten"
                else SteeringIntent.STEER_MATCH,
                source_body_id=source_id,
                destination_body_id=destination_id,
                destination_chain_id=int(destination.get("chain_id", 0)),
                impact_x_sizes=impact_x,
                impact_y_sizes=impact_y,
                reason="low-gauge model-ranked alternate directed pair",
            )
        )
        if len(results) >= maximum_pairs:
            break
    return tuple(results)


def _candidate_key(decision: SteeringDecision) -> tuple[object, ...]:
    action = decision.action
    return (
        int(action.kind),
        int(action.wait_ticks),
        float(action.x_norm),
        float(action.y_norm),
        decision.source_body_id,
        decision.destination_body_id,
    )


def visible_rot_liability(
    observation: Mapping[str, Any],
    *,
    horizon_ticks: int,
    rot_delay_ticks: int = 40,
) -> tuple[int, int]:
    """Conservatively value publicly visible nonrotten rot timers.

    Hidden rule guards are unavailable to the policy, so every visible timer
    due inside the horizon is counted.  The normal exact penalty at level L is
    ``1800 + 20*min(L, 99)`` gauge units.
    """

    if horizon_ticks < 1 or rot_delay_ticks < 1:
        raise ValueError("rot liability horizons must be positive")
    level = int(observation.get("level", 1))
    penalty = 1_800 + 20 * min(max(level, 0), 99)
    count = 0
    for body in observation.get("bodies", ()):
        if not isinstance(body, Mapping):
            continue
        if str(body.get("kind", "")) == "projectile":
            continue
        if str(body.get("lifecycle", "")) in {"deleted", "rotten"}:
            continue
        timer = body.get("rot_timer", 0)
        if isinstance(timer, bool) or not isinstance(timer, int) or timer <= 0:
            continue
        ticks_until_rot = max(0, rot_delay_ticks + 1 - timer)
        if ticks_until_rot <= horizon_ticks:
            count += 1
    return count * penalty, count


class FastMultiActionPlanner:
    """Evaluate a small deterministic action set from one simulator state."""

    def __init__(
        self,
        primitive_actions: Callable[[object], tuple[object, ...]],
        *,
        config: FastMultiActionConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.primitive_actions = primitive_actions
        self.config = FastMultiActionConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        self._long_probe_active = False
        self._current_probe_ticks = self.config.probe_ticks

    @property
    def probe_mode(self) -> str:
        return "low-gauge-long" if self._long_probe_active else "normal"

    @property
    def current_probe_ticks(self) -> int:
        return self._current_probe_ticks

    def _select_probe_horizon(self, gauge: int, tick: int) -> tuple[str, int]:
        if (
            self.config.long_probe_ticks == self.config.probe_ticks
            or tick < self.config.long_probe_min_tick
        ):
            self._long_probe_active = False
        elif self._long_probe_active:
            if gauge > self.config.low_gauge_exit_threshold:
                self._long_probe_active = False
        elif gauge < self.config.low_gauge_threshold:
            self._long_probe_active = True
        self._current_probe_ticks = (
            self.config.long_probe_ticks
            if self._long_probe_active
            else self.config.probe_ticks
        )
        return self.probe_mode, self._current_probe_ticks

    def wait_decision(self, reason: str) -> SteeringDecision:
        return SteeringDecision(
            self.action_spec.validate(SemanticAction.wait(self.config.wait_ticks)),
            SteeringIntent.WAIT,
            reason=reason,
        )

    def candidates(
        self,
        observation: Mapping[str, Any],
        policy_before: object,
        policy_after_prediction: object,
        prediction: SteeringDecision,
    ) -> tuple[PlannerCandidate, ...]:
        if not prediction.is_shot:
            raise ValueError("multi-action planner requires a shot prediction")
        raw: list[tuple[str, SteeringDecision, object]] = []
        for category, kind in (
            ("predicted-pair-strong", SemanticActionKind.FIRE_STRONG),
            ("predicted-pair-weak", SemanticActionKind.FIRE_WEAK),
        ):
            decision = _with_strength(prediction, kind, category)
            state = _copy_policy(policy_after_prediction)
            if _rebind_policy(state, observation, prediction, decision):
                raw.append((category, decision, state))
        wait = self.wait_decision("multi-action restraint candidate")
        raw.append(("wait", wait, _copy_policy(policy_before)))
        if int(observation.get("gauge", 0)) <= self.config.low_gauge_threshold:
            alternatives = _ranked_alternatives(
                policy_after_prediction,
                observation,
                prediction,
                self.config.top_k_pairs,
            )
            for rank, base in enumerate(alternatives, start=1):
                for strength, kind in (
                    ("weak", SemanticActionKind.FIRE_WEAK),
                    ("strong", SemanticActionKind.FIRE_STRONG),
                ):
                    category = f"top-{rank}-pair-{strength}"
                    decision = _with_strength(base, kind, category)
                    state = _copy_policy(policy_after_prediction)
                    if _rebind_policy(state, observation, prediction, decision):
                        raw.append((category, decision, state))
        seen: set[tuple[object, ...]] = set()
        candidates: list[PlannerCandidate] = []
        for category, decision, policy in raw:
            key = _candidate_key(decision)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                PlannerCandidate(len(candidates), category, decision, policy)
            )
        return tuple(candidates)

    def _advance(
        self,
        env: object,
        observation: Mapping[str, Any],
        policy: object,
        first: SteeringDecision,
        horizon_ticks: int,
    ) -> tuple[ProbeOutcome, int, int]:
        start = int(observation["tick"])
        current = observation
        minimum_gauge = int(current["gauge"])
        invalid_actions = 0
        terminated = truncated = False
        decision: object | None = first
        while int(current["tick"]) - start < horizon_ticks and not (
            terminated or truncated
        ):
            if decision is None:
                decision = policy.predict(current)
            for action in self.primitive_actions(decision):
                kind = SemanticActionKind(int(action.kind))
                duration = int(action.wait_ticks) if kind is SemanticActionKind.WAIT else 1
                remaining = horizon_ticks - (int(current["tick"]) - start)
                if remaining <= 0:
                    break
                if duration > remaining:
                    if kind is not SemanticActionKind.WAIT:
                        break
                    action = self.action_spec.press(SemanticAction.wait(remaining))
                    duration = remaining
                for _ in range(duration):
                    primitive = (
                        self.action_spec.press(SemanticAction.wait(1))
                        if kind is SemanticActionKind.WAIT
                        else action
                    )
                    current, _reward, terminated, truncated, info = env.step(primitive)
                    invalid_actions += int(bool(info.get("invalid_action", False)))
                    minimum_gauge = min(minimum_gauge, int(current["gauge"]))
                    if terminated or truncated:
                        break
                if (
                    terminated
                    or truncated
                    or int(current["tick"]) - start >= horizon_ticks
                ):
                    break
            decision = None
        liability, _count = visible_rot_liability(
            current,
            horizon_ticks=horizon_ticks,
            rot_delay_ticks=self.config.rot_delay_ticks,
        )
        return (
            ProbeOutcome(
                int(current["tick"]) - start,
                int(current["score"]),
                int(current.get("qualifying_clear_count", 0)),
                int(current["gauge"]),
                minimum_gauge,
                bool(terminated or current.get("terminated", False)),
                bool(truncated or current.get("truncated", False)),
                int(current.get("highest_chain", 0)),
            ),
            invalid_actions,
            liability,
        )

    def _eligible(
        self, shot: CandidateOutcome, wait: CandidateOutcome
    ) -> tuple[bool, str]:
        if not shot.valid:
            return False, "wait-invalid-shot"
        a, b = shot.probe, wait.probe
        if a.survival_ticks != b.survival_ticks:
            return (
                a.survival_ticks > b.survival_ticks,
                "shot-survival" if a.survival_ticks > b.survival_ticks else "wait-survival",
            )
        a_failed = a.terminated or a.truncated
        b_failed = b.terminated or b.truncated
        if a_failed != b_failed:
            return not a_failed, "shot-rescue" if b_failed else "wait-safer"
        gauge_debt = b.final_gauge - a.final_gauge
        score_gain = a.score - b.score
        if (
            gauge_debt > self.config.maximum_gauge_debt
            and score_gain < self.config.rescue_score_margin
        ):
            return False, "wait-reserve"
        return choose_shot(a, b, gauge_advantage=self.config.gauge_advantage)

    @staticmethod
    def _objective(value: CandidateOutcome) -> tuple[int, int, int, int, int, int]:
        probe = value.probe
        return (
            probe.survival_ticks,
            int(not (probe.terminated or probe.truncated)),
            probe.clears,
            probe.score,
            probe.final_gauge,
            -value.candidate.ordinal,
        )

    def _selection_objective(
        self, value: CandidateOutcome
    ) -> tuple[int, ...]:
        if self.config.objective_mode == "chain-first":
            probe = value.probe
            return (
                probe.survival_ticks,
                int(not (probe.terminated or probe.truncated)),
                probe.highest_chain,
                probe.clears,
                probe.score,
                probe.final_gauge,
                -value.candidate.ordinal,
            )
        return self._objective(value)

    def evaluate(
        self,
        env: object,
        observation: Mapping[str, Any],
        policy_before: object,
        policy_after_prediction: object,
        prediction: SteeringDecision,
    ) -> MultiActionVerdict:
        probe_mode, horizon_ticks = self._select_probe_horizon(
            int(observation.get("gauge", 0)), int(observation.get("tick", 0))
        )
        candidates = self.candidates(
            observation, policy_before, policy_after_prediction, prediction
        )
        source_hash = env.state_hash()
        source_clears = int(observation.get("qualifying_clear_count", 0))
        source_liability, source_rot_count = visible_rot_liability(
            observation,
            horizon_ticks=horizon_ticks,
            rot_delay_ticks=self.config.rot_delay_ticks,
        )
        outcomes: list[CandidateOutcome] = []
        checks = 0
        with TransactionalBranches(env, observation) as branches:
            fast = branches.uses_fast_checkpoint
            for candidate in candidates:
                with branches.branch() as (branch_env, branch_observation):
                    if branch_env.state_hash() != source_hash:
                        raise RuntimeError("multi-action branch state mismatch")
                    checks += 1
                    probe, invalid, liability = self._advance(
                        branch_env,
                        branch_observation,
                        _copy_policy(candidate.continuation_policy),
                        candidate.decision,
                        horizon_ticks,
                    )
                    outcomes.append(
                        CandidateOutcome(
                            candidate,
                            probe,
                            invalid,
                            imminent_visible_rot_liability=liability,
                            renewal_clears=max(0, probe.clears - source_clears),
                        )
                    )
        if env.state_hash() != source_hash:
            raise RuntimeError("multi-action planner altered live state")
        checks += 1
        if self.config.objective_mode == "reserve-band":
            gauge_max = max(
                1,
                int(
                    observation.get(
                        "gauge_max", max(int(observation.get("gauge", 1)), 1)
                    )
                ),
            )
            level = int(observation.get("level", 1))
            contingency = self.config.reserve_contingency_gauge or (
                1_800 + 20 * min(max(level, 0), 99)
            )
            reserve_config = ReserveBandConfig(
                horizon_ticks, gauge_max, contingency
            )
            reserve_candidates = tuple(
                ReserveProbeCandidate(
                    candidate_index=value.candidate.ordinal,
                    survival_ticks=value.probe.survival_ticks,
                    terminated=value.probe.terminated,
                    truncated=value.probe.truncated,
                    minimum_gauge=value.probe.minimum_gauge,
                    final_gauge=value.probe.final_gauge,
                    imminent_visible_rot_liability=(
                        value.imminent_visible_rot_liability
                    ),
                    renewal_clears=value.renewal_clears,
                    score=value.probe.score,
                    invalid_actions=value.invalid_actions,
                )
                for value in outcomes
            )
            reserve_winner, ranked = choose_reserve_candidate(
                reserve_candidates, reserve_config
            )
            selected_outcome = next(
                value
                for value in outcomes
                if value.candidate.ordinal
                == reserve_winner.candidate.candidate_index
            )
            evidence = {
                "config": reserve_config.manifest(),
                "source_visible_rot_liability": source_liability,
                "source_visible_rot_count": source_rot_count,
                "rank_order": [
                    item.candidate.candidate_index
                    for item in sorted(
                        ranked, key=lambda item: item.rank, reverse=True
                    )
                ],
                "winner": reserve_winner.manifest(),
            }
            return MultiActionVerdict(
                selected_outcome.candidate,
                f"{selected_outcome.candidate.category}:reserve-band",
                tuple(outcomes),
                checks,
                fast,
                "reserve-band",
                evidence,
                probe_mode,
                horizon_ticks,
            )

        wait = next(value for value in outcomes if value.candidate.category == "wait")
        eligible: list[tuple[CandidateOutcome, str]] = []
        for value in outcomes:
            if value is wait:
                continue
            allowed, reason = self._eligible(value, wait)
            if allowed:
                eligible.append((value, reason))
        if not eligible:
            selected, reason = wait.candidate, "wait-no-eligible-shot"
            evidence: Mapping[str, object] = {}
        else:
            winner, basis = max(
                eligible, key=lambda item: self._selection_objective(item[0])
            )
            evidence = {}
            if self.config.objective_mode == "robust-reserve-bounded":
                margin = self.config.robust_reserve_margin
                alternatives = [
                    value
                    for value, _reason in eligible
                    if value is not winner
                    and value.probe.survival_ticks == winner.probe.survival_ticks
                    and not (value.probe.terminated or value.probe.truncated)
                    and value.probe.clears == winner.probe.clears
                    and value.probe.score
                    >= winner.probe.score - self.config.robust_score_trade
                    and value.probe.minimum_gauge
                    >= winner.probe.minimum_gauge + margin
                    and value.probe.final_gauge >= winner.probe.final_gauge + margin
                    and value.imminent_visible_rot_liability
                    <= winner.imminent_visible_rot_liability
                ]
                if alternatives:
                    original = winner
                    winner = max(
                        alternatives,
                        key=lambda value: (
                            value.probe.minimum_gauge,
                            value.probe.final_gauge,
                            self._selection_objective(value),
                        ),
                    )
                    evidence = {
                        "margin": margin,
                        "maximum_score_trade": self.config.robust_score_trade,
                        "original_candidate": original.candidate.ordinal,
                        "selected_candidate": winner.candidate.ordinal,
                        "score_trade": original.probe.score - winner.probe.score,
                        "minimum_gauge_gain": (
                            winner.probe.minimum_gauge
                            - original.probe.minimum_gauge
                        ),
                        "final_gauge_gain": (
                            winner.probe.final_gauge - original.probe.final_gauge
                        ),
                    }
                    basis = "robust-reserve-bounded-trade"
            selected = winner.candidate
            reason = f"{selected.category}:{basis}"
        return MultiActionVerdict(
            selected,
            reason,
            tuple(outcomes),
            checks,
            fast,
            self.config.objective_mode,
            evidence,
            probe_mode,
            horizon_ticks,
        )


__all__ = [
    "CandidateOutcome",
    "FastMultiActionConfig",
    "FastMultiActionPlanner",
    "MultiActionVerdict",
    "PlannerCandidate",
    "visible_rot_liability",
]
