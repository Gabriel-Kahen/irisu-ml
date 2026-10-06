"""Proposal-only directed-pair residual for the exact multi-action planner.

The approved base policy owns every live decision and every branch
continuation.  A separate learned model may only append one strong-shot pair
for exact evaluation; it can never replace the base policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from irisu_pointer.fast_multiaction_planner import (
    FastMultiActionConfig,
    FastMultiActionPlanner,
    PlannerCandidate,
    _copy_policy,
    _inner_policy,
    _rebind_policy,
    _safe_pair,
)
from irisu_pointer.policy import encoded_body_ids
from irisu_pointer.steering import SteeringDecision, SteeringIntent


@dataclass(frozen=True, slots=True)
class ProposalOnlyResidualConfig:
    minimum_tick: int = 40_000
    low_reserve_minimum_tick: int = 0
    low_reserve_gauge: int = 12_000

    def __post_init__(self) -> None:
        for name in (
            "minimum_tick",
            "low_reserve_minimum_tick",
            "low_reserve_gauge",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def active(self, observation: Mapping[str, Any]) -> bool:
        tick = int(observation.get("tick", 0))
        gauge = int(observation.get("gauge", 0))
        return tick >= self.minimum_tick or (
            tick >= self.low_reserve_minimum_tick
            and gauge <= self.low_reserve_gauge
        )

    def manifest(self) -> dict[str, int]:
        return {
            "minimum_tick": self.minimum_tick,
            "low_reserve_minimum_tick": self.low_reserve_minimum_tick,
            "low_reserve_gauge": self.low_reserve_gauge,
        }


@torch.no_grad()
def residual_pair_proposal(
    residual_model: torch.nn.Module,
    base_policy: object,
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
) -> SteeringDecision | None:
    """Return the highest residual-ranked safe non-incumbent pair."""

    inner = _inner_policy(base_policy)
    encoded = inner.encoder.encode([observation])
    active = np.flatnonzero(encoded.body_mask[0])
    width = int(active[-1]) + 1 if active.size else 1
    try:
        device = next(residual_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    output = residual_model(
        torch.from_numpy(encoded.global_features).to(device),
        torch.from_numpy(encoded.body_features[:, :width]).to(device),
        torch.from_numpy(encoded.body_mask[:, :width]).to(device),
    )
    identifiers = list(encoded_body_ids(encoded, observation)[:width])
    legal = output.legal_pair_mask[0].flatten().nonzero(as_tuple=False).reshape(-1)
    scores = output.pair_logits[0].flatten()[legal]
    ranked = legal[scores.argsort(descending=True, stable=True)]
    incumbent_pair = (incumbent.source_body_id, incumbent.destination_body_id)
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
        analytic = inner._analytic_action(*pair, use_strong=True)
        if analytic is None:
            continue
        action, impact_x, impact_y = analytic
        destination = pair[1]
        return SteeringDecision(
            action,
            SteeringIntent.MATCH_ROTTEN
            if str(destination.get("lifecycle", "")) == "rotten"
            else SteeringIntent.STEER_MATCH,
            source_body_id=source_id,
            destination_body_id=destination_id,
            destination_chain_id=int(destination.get("chain_id", 0)),
            impact_x_sizes=impact_x,
            impact_y_sizes=impact_y,
            reason="proposal-only learned directed-pair residual",
        )
    return None


class ProposalOnlyPairResidualPlanner(FastMultiActionPlanner):
    """Base planner plus at most one proposal from a separate residual model."""

    def __init__(
        self,
        primitive_actions: object,
        *,
        residual_model: torch.nn.Module,
        residual_config: ProposalOnlyResidualConfig | None = None,
        config: FastMultiActionConfig | None = None,
        action_spec: object | None = None,
    ) -> None:
        planner_config = FastMultiActionConfig() if config is None else config
        if planner_config.top_k_pairs != 0:
            raise ValueError("proposal-only base planner requires top_k_pairs=0")
        super().__init__(
            primitive_actions, config=planner_config, action_spec=action_spec
        )
        self.residual_model = residual_model.eval()
        self.residual_config = (
            ProposalOnlyResidualConfig()
            if residual_config is None
            else residual_config
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
        if not self.residual_config.active(observation):
            return base
        proposal = residual_pair_proposal(
            self.residual_model,
            policy_after_prediction,
            observation,
            prediction,
        )
        if proposal is None:
            return base
        continuation = _copy_policy(policy_after_prediction)
        if not _rebind_policy(continuation, observation, prediction, proposal):
            return base
        return (*base, PlannerCandidate(len(base), "residual-pair-strong", proposal, continuation))


__all__ = [
    "ProposalOnlyPairResidualPlanner",
    "ProposalOnlyResidualConfig",
    "residual_pair_proposal",
]
