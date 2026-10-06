"""Development-only fail-closed blending of two directed-pair models.

The base policy remains the sole live controller.  A residual model is a
stateless pair ranker: late in an episode or at low reserve it may append one
novel pair to the base planner's exact candidate inventory.  Every continuation
uses a copy of the base controller, and a residual branch may replace the base
winner only when its bounded exact outcome is reserve-noninferior.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch

from irisu_env import ExactWorkerError
from irisu_pointer.branching import TransactionalBranches
from irisu_pointer.fast_multiaction_planner import (
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
    MultiActionVerdict,
    PlannerCandidate,
    _copy_policy,
    _inner_policy,
    _rebind_policy,
    _safe_pair,
    _with_strength,
)
from irisu_pointer.policy import encoded_body_ids
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticActionKind


BLEND_VERSION = "irisu-development-fail-closed-proposal-blend-v1"


class PairProposalRanker(Protocol):
    """Stateless proposal-only interface shared by portable and exact tools."""

    def ranked_pairs(
        self,
        observation: Mapping[str, Any],
        *,
        excluded_pairs: frozenset[tuple[int, int]],
        maximum_pairs: int,
    ) -> Sequence[tuple[int, int]]: ...


class ModelPairProposalRanker:
    """Read only pair logits without advancing any controller bookkeeping."""

    def __init__(self, encoder: object, model: torch.nn.Module) -> None:
        self.encoder = encoder
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def ranked_pairs(
        self,
        observation: Mapping[str, Any],
        *,
        excluded_pairs: frozenset[tuple[int, int]],
        maximum_pairs: int,
    ) -> tuple[tuple[int, int], ...]:
        if maximum_pairs < 0:
            raise ValueError("maximum_pairs must be nonnegative")
        if maximum_pairs == 0:
            return ()
        encoded = self.encoder.encode([observation])
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        output = self.model(
            torch.from_numpy(encoded.global_features).to(device),
            torch.from_numpy(encoded.body_features[:, :width]).to(device),
            torch.from_numpy(encoded.body_mask[:, :width]).to(device),
        )
        identifiers = list(encoded_body_ids(encoded, observation)[:width])
        legal = output.legal_pair_mask[0].flatten().nonzero(as_tuple=False).reshape(-1)
        scores = output.pair_logits[0].flatten()[legal]
        ranked = legal[scores.argsort(descending=True, stable=True)]
        result: list[tuple[int, int]] = []
        for raw in ranked.tolist():
            source_index, destination_index = divmod(int(raw), width)
            source_id = identifiers[source_index]
            destination_id = identifiers[destination_index]
            if source_id is None or destination_id is None:
                continue
            pair = (int(source_id), int(destination_id))
            if pair in excluded_pairs:
                continue
            result.append(pair)
            if len(result) >= maximum_pairs:
                break
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ProposalBlendConfig:
    activation_tick_exclusive: int = 50_000
    activation_gauge_exclusive: int = 20_000
    maximum_residual_pairs: int = 1

    def __post_init__(self) -> None:
        for name in ("activation_tick_exclusive", "activation_gauge_exclusive"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.maximum_residual_pairs != 1:
            raise ValueError("development blend permits exactly one residual pair")

    def manifest(self) -> dict[str, int | str]:
        return {"version": BLEND_VERSION, **asdict(self)}


def _pair(decision: SteeringDecision) -> tuple[int, int] | None:
    if decision.source_body_id is None or decision.destination_body_id is None:
        return None
    return int(decision.source_body_id), int(decision.destination_body_id)


class FailClosedProposalBlendPlanner(FastMultiActionPlanner):
    """Append at most one residual pair while preserving every base candidate."""

    def __init__(
        self,
        primitive_actions,
        residual_ranker: PairProposalRanker,
        *,
        config: FastMultiActionConfig | None = None,
        blend_config: ProposalBlendConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        super().__init__(primitive_actions, config=config, action_spec=action_spec)
        if self.config.objective_mode != "wait-relative":
            raise ValueError("proposal blend currently requires wait-relative objective")
        self.residual_ranker = residual_ranker
        self.blend_config = ProposalBlendConfig() if blend_config is None else blend_config
        self.proposal_counts: Counter[str] = Counter()

    def _active(self, observation: Mapping[str, Any]) -> bool:
        return (
            int(observation.get("tick", 0))
            > self.blend_config.activation_tick_exclusive
            or int(observation.get("gauge", 0))
            < self.blend_config.activation_gauge_exclusive
        )

    def candidates(
        self,
        observation: Mapping[str, Any],
        policy_before: object,
        policy_after_prediction: object,
        prediction: SteeringDecision,
    ) -> tuple[PlannerCandidate, ...]:
        base = super().candidates(
            observation, policy_before, policy_after_prediction, prediction
        )
        if not self._active(observation):
            self.proposal_counts["inactive"] += 1
            return base
        excluded = frozenset(
            value
            for value in (_pair(candidate.decision) for candidate in base)
            if value is not None
        )
        try:
            proposed = tuple(
                self.residual_ranker.ranked_pairs(
                    observation,
                    excluded_pairs=excluded,
                    maximum_pairs=self.blend_config.maximum_residual_pairs,
                )
            )
            if len(proposed) > self.blend_config.maximum_residual_pairs:
                raise ValueError("residual ranker exceeded its pair budget")
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            self.proposal_counts["ranker-error-base-fallback"] += 1
            return base
        if not proposed:
            self.proposal_counts["no-novel-pair"] += 1
            return base
        source_id, destination_id = proposed[0]
        pair = (int(source_id), int(destination_id))
        if pair in excluded:
            self.proposal_counts["duplicate-base-fallback"] += 1
            return base
        safe = _safe_pair(
            policy_after_prediction, observation, pair[0], pair[1]
        )
        if safe is None:
            self.proposal_counts["unsafe-pair-base-fallback"] += 1
            return base
        analytic = _inner_policy(policy_after_prediction)._analytic_action(*safe)
        if analytic is None:
            self.proposal_counts["unreachable-pair-base-fallback"] += 1
            return base
        action, impact_x, impact_y = analytic
        destination = safe[1]
        proposal = SteeringDecision(
            action,
            SteeringIntent.MATCH_ROTTEN
            if str(destination.get("lifecycle", "")) == "rotten"
            else SteeringIntent.STEER_MATCH,
            source_body_id=pair[0],
            destination_body_id=pair[1],
            destination_chain_id=int(destination.get("chain_id", 0)),
            impact_x_sizes=impact_x,
            impact_y_sizes=impact_y,
            reason="residual checkpoint proposal-only directed pair",
        )
        appended: list[PlannerCandidate] = []
        for strength, kind in (
            ("weak", SemanticActionKind.FIRE_WEAK),
            ("strong", SemanticActionKind.FIRE_STRONG),
        ):
            category = f"residual-pair-{strength}"
            decision = _with_strength(proposal, kind, category)
            continuation = _copy_policy(policy_after_prediction)
            if _rebind_policy(continuation, observation, prediction, decision):
                appended.append(
                    PlannerCandidate(len(base) + len(appended), category, decision, continuation)
                )
        if not appended:
            self.proposal_counts["rebind-failed-base-fallback"] += 1
            return base
        self.proposal_counts["pair-appended"] += 1
        result = (*base, *appended)
        if result[: len(base)] != base:
            raise RuntimeError("proposal blend changed the base candidate prefix")
        return result

    def _base_choice(
        self, outcomes: Sequence[CandidateOutcome]
    ) -> tuple[CandidateOutcome, str]:
        base = [
            value
            for value in outcomes
            if not value.candidate.category.startswith("residual-pair-")
        ]
        wait = next(value for value in base if value.candidate.category == "wait")
        eligible: list[tuple[CandidateOutcome, str]] = []
        for value in base:
            if value is wait:
                continue
            allowed, reason = self._eligible(value, wait)
            if allowed:
                eligible.append((value, reason))
        if not eligible:
            return wait, "wait-no-eligible-shot"
        winner, basis = max(eligible, key=lambda item: self._objective(item[0]))
        return winner, f"{winner.candidate.category}:{basis}"

    @staticmethod
    def _reserve_noninferior(
        residual: CandidateOutcome, base: CandidateOutcome
    ) -> bool:
        a, b = residual.probe, base.probe
        return (
            residual.valid
            and a.survival_ticks >= b.survival_ticks
            and int(not (a.terminated or a.truncated))
            >= int(not (b.terminated or b.truncated))
            and a.minimum_gauge >= b.minimum_gauge
            and a.final_gauge >= b.final_gauge
        )

    def _apply_residual_safety(
        self, verdict: MultiActionVerdict
    ) -> MultiActionVerdict:
        if not verdict.selected.category.startswith("residual-pair-"):
            return verdict
        selected = next(
            value
            for value in verdict.outcomes
            if value.candidate.ordinal == verdict.selected.ordinal
        )
        base, base_reason = self._base_choice(verdict.outcomes)
        if not (
            self._reserve_noninferior(selected, base)
            and self._objective(selected) > self._objective(base)
        ):
            self.proposal_counts["residual-safety-veto"] += 1
            return MultiActionVerdict(
                base.candidate,
                f"{base_reason}:residual-safety-veto",
                verdict.outcomes,
                verdict.branch_checks,
                verdict.used_fast_checkpoint,
                verdict.objective_mode,
                {"residual_safety_veto": True},
                verdict.probe_mode,
                verdict.probe_ticks,
            )
        self.proposal_counts["residual-safe-override"] += 1
        return MultiActionVerdict(
            verdict.selected,
            f"{verdict.reason}:residual-safe-override",
            verdict.outcomes,
            verdict.branch_checks,
            verdict.used_fast_checkpoint,
            verdict.objective_mode,
            {"residual_safety_veto": False},
            verdict.probe_mode,
            verdict.probe_ticks,
        )

    def evaluate(
        self,
        env: object,
        observation: Mapping[str, Any],
        policy_before: object,
        policy_after_prediction: object,
        prediction: SteeringDecision,
    ) -> MultiActionVerdict:
        """Evaluate base first; a residual worker failure discards only residuals."""

        probe_mode, horizon_ticks = self._select_probe_horizon(
            int(observation.get("gauge", 0)), int(observation.get("tick", 0))
        )
        candidates = self.candidates(
            observation, policy_before, policy_after_prediction, prediction
        )
        source_hash = env.state_hash()
        outcomes: list[CandidateOutcome] = []
        checks = 0
        residual_branch_error = False
        with TransactionalBranches(env, observation) as branches:
            fast = branches.uses_fast_checkpoint
            for candidate in candidates:
                try:
                    with branches.branch() as (branch_env, branch_observation):
                        if branch_env.state_hash() != source_hash:
                            raise RuntimeError(
                                "multi-action branch state mismatch"
                            )
                        checks += 1
                        probe, invalid, liability = self._advance(
                            branch_env,
                            branch_observation,
                            _copy_policy(candidate.continuation_policy),
                            candidate.decision,
                            horizon_ticks,
                        )
                except ExactWorkerError:
                    if not candidate.category.startswith("residual-pair-"):
                        raise
                    residual_branch_error = True
                    self.proposal_counts[
                        "residual-branch-error-base-fallback"
                    ] += 1
                    break
                outcomes.append(
                    CandidateOutcome(
                        candidate,
                        probe,
                        invalid,
                        imminent_visible_rot_liability=liability,
                        renewal_clears=max(
                            0,
                            probe.clears
                            - int(
                                observation.get(
                                    "qualifying_clear_count", 0
                                )
                            ),
                        ),
                    )
                )
        if env.state_hash() != source_hash:
            raise RuntimeError("multi-action planner altered live state")
        checks += 1
        selectable = [
            value
            for value in outcomes
            if not residual_branch_error
            or not value.candidate.category.startswith("residual-pair-")
        ]
        wait = next(
            value
            for value in selectable
            if value.candidate.category == "wait"
        )
        eligible: list[tuple[CandidateOutcome, str]] = []
        for value in selectable:
            if value is wait:
                continue
            allowed, reason = self._eligible(value, wait)
            if allowed:
                eligible.append((value, reason))
        if not eligible:
            selected, reason = wait.candidate, "wait-no-eligible-shot"
        else:
            winner, basis = max(
                eligible, key=lambda item: self._objective(item[0])
            )
            selected = winner.candidate
            reason = f"{selected.category}:{basis}"
        evidence: dict[str, object] = {}
        if residual_branch_error:
            reason += ":residual-branch-error-base-fallback"
            evidence["residual_branch_error_base_fallback"] = True
        verdict = MultiActionVerdict(
            selected,
            reason,
            tuple(outcomes),
            checks,
            fast,
            "wait-relative",
            evidence,
            probe_mode,
            horizon_ticks,
        )
        return self._apply_residual_safety(verdict)

    def manifest(self) -> dict[str, object]:
        return {
            "version": BLEND_VERSION,
            "blend_config": self.blend_config.manifest(),
            "base_planner_config": self.config.manifest(),
            "continuation_policy": "base-only",
            "base_candidate_invariant": "exact-prefix-subset",
            "residual_pair_budget": 1,
            "residual_selection_gate": (
                "exact-base-winner-reserve-noninferiority-then-objective"
            ),
        }


__all__ = [
    "BLEND_VERSION",
    "FailClosedProposalBlendPlanner",
    "ModelPairProposalRanker",
    "PairProposalRanker",
    "ProposalBlendConfig",
]
