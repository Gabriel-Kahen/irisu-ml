#!/usr/bin/env python3
"""Development-only top-K counterfactual ranking distillation.

At each learner-proposed shot, WAIT, the proposal, and a bounded set of
model-ranked alternative directed pairs are continued from one portable
snapshot.  Deployment-matched reserve comparisons provide an act label and
only explicit winner-over-loser pair preferences; unevaluated pairs are never
treated as negatives.  Rollouts still execute the current adaptive-gated
learner, keeping collection on its deployment-state distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_expert_iteration_dagger as search  # noqa: E402
import rl_portable_gate_distill as gate  # noqa: E402
from irisu_env import IrisuEnv  # noqa: E402
from irisu_pointer.shot_necessity import ProbeOutcome, WaitDominanceConfig  # noqa: E402
from irisu_pointer.steering import SteeringDecision, SteeringIntent  # noqa: E402
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (  # noqa: E402
    GoalConditionedSteeringModel,
    SteeringDataset,
    SteeringExample,
    steering_example_from_decision,
)
from irisu_rl.actions import SemanticAction  # noqa: E402
from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder  # noqa: E402


@dataclass(frozen=True, slots=True)
class PairPreference:
    observation: EncodedBatch
    positive_source: int
    positive_destination: int
    negative_source: int
    negative_destination: int
    provenance: Mapping[str, object]

    @property
    def sha256(self) -> str:
        return gate._sha(self.manifest())

    def manifest(self) -> dict[str, object]:
        encoded = self.observation
        return {
            "schema": "irisu-portable-pair-preference-v1",
            "observation": {
                "global_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.global_features).tobytes()
                ).hexdigest(),
                "body_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.body_features).tobytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.body_mask).tobytes()
                ).hexdigest(),
                "schema_sha256": encoded.schema.sha256,
            },
            "positive": [self.positive_source, self.positive_destination],
            "negative": [self.negative_source, self.negative_destination],
            "provenance": dict(self.provenance),
            "provenance_sha256": gate._sha(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class ActPreference:
    observation: EncodedBatch
    execute_shot: bool
    confidence: float
    provenance: Mapping[str, object]

    @property
    def sha256(self) -> str:
        return gate._sha(self.manifest())

    def manifest(self) -> dict[str, object]:
        encoded = self.observation
        return {
            "schema": "irisu-portable-causal-act-preference-v1",
            "observation": {
                "global_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.global_features).tobytes()
                ).hexdigest(),
                "body_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.body_features).tobytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(encoded.body_mask).tobytes()
                ).hexdigest(),
                "schema_sha256": encoded.schema.sha256,
            },
            "execute_shot": self.execute_shot,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "provenance_sha256": gate._sha(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RankingTrainingReport:
    steps: int
    act_examples: int
    causal_act_preferences: int
    pair_preferences: int
    initial_act_loss: float
    final_act_loss: float
    initial_causal_loss: float
    final_causal_loss: float
    initial_pair_loss: float
    final_pair_loss: float
    act_accuracy: float
    causal_accuracy: float
    pair_accuracy: float


def causal_confidence(
    shot: ProbeOutcome,
    wait: ProbeOutcome,
    *,
    horizon: int,
    maximum_gauge_debt: int,
) -> float:
    """Bound confidence by causal survival, score, and reserve separation."""

    survival = 4.0 * abs(shot.survival_ticks - wait.survival_ticks) / horizon
    failure = 2.0 if (shot.terminated or shot.truncated) != (wait.terminated or wait.truncated) else 0.0
    score = min(2.0, abs(shot.score - wait.score) / 500.0)
    gauge_excess = max(0, abs(shot.final_gauge - wait.final_gauge) - maximum_gauge_debt)
    reserve = min(2.0, gauge_excess / 5_000.0)
    return min(8.0, 1.0 + survival + failure + score + reserve)


def configure_trainable_scope(
    model: GoalConditionedSteeringModel, scope: str
) -> tuple[str, ...]:
    prefixes = {
        "pair-head": ("pair_head.",),
        "act-pair": ("act_head.", "pair_head."),
        "heads": (
            "act_head.",
            "wait_head.",
            "pair_head.",
            "kind_head.",
            "template_head.",
            "intent_head.",
        ),
        "all": ("",),
    }
    if scope not in prefixes:
        raise ValueError(f"unknown trainable scope: {scope}")
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes[scope])
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise RuntimeError("trainable scope selected no parameters")
    return tuple(selected)


def _decision_key(decision: SteeringDecision) -> tuple[int | None, int | None]:
    return decision.source_body_id, decision.destination_body_id


def ranked_candidates(
    policy: Any,
    observation: Mapping[str, Any],
    proposal: SteeringDecision,
    *,
    wait_ticks: int,
    maximum_pairs: int,
) -> tuple[SteeringDecision, ...]:
    """WAIT, then both strengths for each unique model-ranked pair.

    Deployment marginalizes shot strength with exact search, so pair
    supervision must not accidentally assign a pair the value of only its
    incumbent strength.
    """

    if not proposal.is_shot:
        raise ValueError("pair ranking requires a learner shot proposal")
    raw = search.legal_candidates(
        policy,
        observation,
        wait_ticks=wait_ticks,
        maximum_pairs=None,
    )
    wait = raw[0]
    output = [wait]
    seen: set[tuple[int | None, int | None]] = set()
    ordered = (proposal, *raw[1:])
    for candidate in ordered:
        key = _decision_key(candidate)
        if key in seen:
            continue
        if policy._progress.is_stalled(observation, *key):
            continue
        for kind, action in (
            ("strong", SemanticAction.strong),
            ("weak", SemanticAction.weak),
        ):
            output.append(
                SteeringDecision(
                    action(candidate.action.x_norm, candidate.action.y_norm),
                    candidate.intent,
                    source_body_id=candidate.source_body_id,
                    destination_body_id=candidate.destination_body_id,
                    destination_chain_id=candidate.destination_chain_id,
                    impact_x_sizes=candidate.impact_x_sizes,
                    impact_y_sizes=candidate.impact_y_sizes,
                    reason=f"pair-ranking {kind} counterfactual",
                )
            )
        seen.add(key)
        if len(seen) >= maximum_pairs:
            break
    return tuple(output)


def candidate_continuation(
    policy_before: Any,
    observation: Mapping[str, Any],
    decision: SteeringDecision,
) -> Any:
    """Advance controller bookkeeping exactly as predict would for a candidate."""

    continuation = copy.deepcopy(policy_before)
    inner = continuation.inner
    tick = int(observation["tick"])
    inner._progress.prune(observation)
    inner._progress.assess(observation)
    inner._last_tick = tick
    inner._last_decision = decision
    if decision.is_shot:
        assert decision.source_body_id is not None
        assert decision.destination_body_id is not None
        inner._cooldown_until = tick + inner.cooldown_ticks
        inner._progress.begin(
            observation,
            decision.source_body_id,
            decision.destination_body_id,
        )
    return continuation


def evaluate_candidates(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    policy_before: Any,
    candidates: Sequence[SteeringDecision],
    *,
    horizon: int,
    wait_ticks: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[ProbeOutcome, ...]:
    if not candidates or candidates[0].is_shot:
        raise ValueError("candidate inventory must begin with WAIT")
    evaluator = gate.ReserveWaitGate(
        lambda decision: decision.primitive_actions(),
        config=WaitDominanceConfig(horizon, wait_ticks, 16),
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    snapshot = env.clone_state()
    expected_hash = env.state_hash()
    outcomes: list[ProbeOutcome] = []
    try:
        for candidate in candidates:
            restored = env.restore_state(snapshot)
            if env.clone_state() != snapshot or env.state_hash() != expected_hash:
                raise RuntimeError("pair teacher transactional restore mismatch")
            continuation = candidate_continuation(
                policy_before, observation, candidate
            )
            outcomes.append(
                evaluator._advance(
                    env,
                    restored,
                    continuation,
                    candidate,
                )
            )
    finally:
        env.restore_state(snapshot)
    if env.clone_state() != snapshot or env.state_hash() != expected_hash:
        raise RuntimeError("pair teacher changed live rollout state")
    return tuple(outcomes)


def _probe_objective(outcome: ProbeOutcome) -> tuple[int, int, int, int, int]:
    return (
        outcome.survival_ticks,
        int(not (outcome.terminated or outcome.truncated)),
        outcome.clears,
        outcome.score,
        outcome.final_gauge,
    )


def tournament_preferences(
    candidates: Sequence[SteeringDecision],
    outcomes: Sequence[ProbeOutcome],
    *,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[int, tuple[tuple[int, int, str], ...]]:
    """Mirror deployment: gate against WAIT, then rank strength-free pairs."""

    if len(outcomes) < 2 or len(candidates) != len(outcomes):
        raise ValueError("ranking tournament requires WAIT and one shot")
    if candidates[0].is_shot:
        raise ValueError("ranking tournament candidate zero must be WAIT")
    eligible: dict[int, bool] = {}
    groups: dict[tuple[int | None, int | None], list[int]] = {}
    for index in range(1, len(outcomes)):
        eligible[index], _reason = gate.reserve_choice(
            outcomes[index],
            outcomes[0],
            maximum_gauge_debt=maximum_gauge_debt,
            rescue_score_margin=rescue_score_margin,
        )
        groups.setdefault(_decision_key(candidates[index]), []).append(index)

    representatives: list[int] = []
    for indices in groups.values():
        viable = [index for index in indices if eligible[index]]
        pool = viable or indices
        representatives.append(
            max(pool, key=lambda index: (_probe_objective(outcomes[index]), -index))
        )
    viable_representatives = [index for index in representatives if eligible[index]]
    winner = (
        max(
            viable_representatives,
            key=lambda index: (_probe_objective(outcomes[index]), -index),
        )
        if viable_representatives
        else 0
    )
    preferences: list[tuple[int, int, str]] = []
    for offset, first in enumerate(representatives):
        for second in representatives[offset + 1 :]:
            first_rank = (eligible[first], _probe_objective(outcomes[first]))
            second_rank = (eligible[second], _probe_objective(outcomes[second]))
            if first_rank == second_rank:
                continue
            positive, negative = (
                (first, second) if first_rank > second_rank else (second, first)
            )
            preferences.append(
                (positive, negative, "strength-marginalized-deployment-objective")
            )
    return winner, tuple(preferences)


def _preference(
    observation: Mapping[str, Any],
    positive: SteeringDecision,
    negative: SteeringDecision,
    provenance: Mapping[str, object],
    *,
    encoder: TeacherStateEncoder,
    pointer_spec: Any,
) -> PairPreference:
    positive_example = steering_example_from_decision(
        observation,
        positive,
        episode_identity="pair-preference-positive",
        provenance_sha256=gate._sha(provenance),
        encoder=encoder,
        pointer_spec=pointer_spec,
        require_representable_template=False,
    )
    negative_example = steering_example_from_decision(
        observation,
        negative,
        episode_identity="pair-preference-negative",
        provenance_sha256=gate._sha(provenance),
        encoder=encoder,
        pointer_spec=pointer_spec,
        require_representable_template=False,
    )
    if positive_example is None or negative_example is None:
        raise RuntimeError("teacher pair preference is not representable")
    return PairPreference(
        positive_example.observation,
        positive_example.source_index,
        positive_example.destination_index,
        negative_example.source_index,
        negative_example.destination_index,
        dict(provenance),
    )


def collect_episode(
    *,
    runtime: Path,
    model: GoalConditionedSteeringModel,
    model_sha256: str,
    seed: int,
    maximum_ticks: int,
    short_horizon: int,
    long_horizon: int,
    gauge_threshold: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
    wait_ticks: int,
    maximum_pairs: int,
    wait_anchor_stride: int,
    maximum_wait_anchors: int,
    act_logit_bias: float,
    iteration: int,
) -> tuple[
    list[gate.Label],
    list[ActPreference],
    list[PairPreference],
    dict[str, object],
    Mapping[str, Any],
]:
    policy = gate._make_policy(model, model_sha256, act_logit_bias)
    policy.reset(seed)
    encoder = TeacherStateEncoder()
    labels: list[gate.Label] = []
    act_preferences: list[ActPreference] = []
    preferences: list[PairPreference] = []
    reasons: Counter[str] = Counter()
    proposals = deployment_kept = deployment_suppressed = teacher_changes = 0
    long_queries = candidates_evaluated = wait_seen = wait_anchors = decisions = 0
    started = time.monotonic()
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": maximum_ticks + long_horizon + 64},
    ) as env:
        runtime_manifest = env.runner_identity_manifest()
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("portable reset seed differs")
        terminated = truncated = False
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            policy_before = copy.deepcopy(policy)
            proposal = policy.predict(observation)
            executed = proposal
            if proposal.is_shot:
                proposals += 1
                horizon = gate.adaptive_horizon(
                    int(observation["gauge"]),
                    short_horizon=short_horizon,
                    long_horizon=long_horizon,
                    gauge_threshold=gauge_threshold,
                )
                long_queries += horizon == long_horizon and long_horizon != short_horizon
                candidates = ranked_candidates(
                    policy,
                    observation,
                    proposal,
                    wait_ticks=wait_ticks,
                    maximum_pairs=maximum_pairs,
                )
                outcomes = evaluate_candidates(
                    env,
                    observation,
                    policy_before,
                    candidates,
                    horizon=horizon,
                    wait_ticks=wait_ticks,
                    maximum_gauge_debt=maximum_gauge_debt,
                    rescue_score_margin=rescue_score_margin,
                )
                candidates_evaluated += len(candidates)
                winner, comparisons = tournament_preferences(
                    candidates,
                    outcomes,
                    maximum_gauge_debt=maximum_gauge_debt,
                    rescue_score_margin=rescue_score_margin,
                )
                proposal_pair = _decision_key(proposal)
                proposal_indices = [
                    index
                    for index in range(1, len(candidates))
                    if _decision_key(candidates[index]) == proposal_pair
                ]
                eligible_proposal_indices = [
                    index
                    for index in proposal_indices
                    if gate.reserve_choice(
                        outcomes[index],
                        outcomes[0],
                        maximum_gauge_debt=maximum_gauge_debt,
                        rescue_score_margin=rescue_score_margin,
                    )[0]
                ]
                deployment_index = (
                    max(
                        eligible_proposal_indices,
                        key=lambda index: (_probe_objective(outcomes[index]), -index),
                    )
                    if eligible_proposal_indices
                    else 0
                )
                teacher_changes += winner == 0 or _decision_key(candidates[winner]) != proposal_pair
                teacher_decision = candidates[winner]
                query = {
                    "schema": "irisu-portable-pair-ranking-query-v1",
                    "seed": seed,
                    "tick": int(observation["tick"]),
                    "iteration": iteration,
                    "horizon": horizon,
                    "gauge": int(observation["gauge"]),
                    "winner": winner,
                    "candidates": [gate._decision_manifest(value) for value in candidates]
                    if hasattr(gate, "_decision_manifest")
                    else [
                        {
                            "source_body_id": value.source_body_id,
                            "destination_body_id": value.destination_body_id,
                            "intent": value.intent.value,
                        }
                        for value in candidates
                    ],
                    "outcomes": [value.manifest() for value in outcomes],
                    "comparisons": [list(value) for value in comparisons],
                }
                query_sha256 = gate._sha(query)
                kind = "teacher-ranked-shot" if teacher_decision.is_shot else "teacher-ranked-wait"
                labels.append(
                    gate._label(
                        observation,
                        teacher_decision,
                        kind=kind,
                        provenance={
                            "schema": "irisu-portable-pair-ranking-label-provenance-v1",
                            "seed": seed,
                            "tick": int(observation["tick"]),
                            "iteration": iteration,
                            "kind": kind,
                            "query_sha256": query_sha256,
                            "query": query,
                        },
                        encoder=encoder,
                        pointer_spec=policy.pointer_spec,
                    )
                )
                best_shot_index = winner if winner > 0 else proposal_indices[0]
                act_preferences.append(
                    ActPreference(
                        encoder.encode([observation]),
                        winner > 0,
                        causal_confidence(
                            outcomes[best_shot_index],
                            outcomes[0],
                            horizon=horizon,
                            maximum_gauge_debt=maximum_gauge_debt,
                        ),
                        {
                            "schema": "irisu-portable-causal-act-preference-provenance-v1",
                            "seed": seed,
                            "tick": int(observation["tick"]),
                            "iteration": iteration,
                            "query_sha256": query_sha256,
                            "shot_candidate": best_shot_index,
                            "execute_shot": winner > 0,
                            "shot_outcome": outcomes[best_shot_index].manifest(),
                            "wait_outcome": outcomes[0].manifest(),
                        },
                    )
                )
                for ordinal, (positive, negative, reason) in enumerate(comparisons):
                    provenance = {
                        "schema": "irisu-portable-pair-preference-provenance-v1",
                        "seed": seed,
                        "tick": int(observation["tick"]),
                        "iteration": iteration,
                        "query_sha256": query_sha256,
                        "ordinal": ordinal,
                        "positive_candidate": positive,
                        "negative_candidate": negative,
                        "reason": reason,
                    }
                    preferences.append(
                        _preference(
                            observation,
                            candidates[positive],
                            candidates[negative],
                            provenance,
                            encoder=encoder,
                            pointer_spec=policy.pointer_spec,
                        )
                    )
                keep_proposal = deployment_index > 0
                deployment_reason = (
                    "shot-strength-marginalized"
                    if keep_proposal
                    else "wait-no-eligible-proposal-strength"
                )
                reasons[deployment_reason] += 1
                if keep_proposal:
                    deployment_kept += 1
                    executed = candidates[deployment_index]
                    policy = candidate_continuation(
                        policy_before, observation, executed
                    )
                else:
                    deployment_suppressed += 1
                    policy = policy_before
                    executed = SteeringDecision(
                        SemanticAction.wait(wait_ticks),
                        SteeringIntent.WAIT,
                        reason=deployment_reason,
                    )
            else:
                wait_seen += 1
                if wait_seen % wait_anchor_stride == 0 and wait_anchors < maximum_wait_anchors:
                    wait_anchors += 1
                    labels.append(
                        gate._label(
                            observation,
                            proposal,
                            kind="learner-wait-anchor",
                            provenance={
                                "schema": "irisu-portable-pair-ranking-label-provenance-v1",
                                "seed": seed,
                                "tick": int(observation["tick"]),
                                "iteration": iteration,
                                "kind": "learner-wait-anchor",
                                "wait_ordinal": wait_seen,
                            },
                            encoder=encoder,
                            pointer_spec=policy.pointer_spec,
                        )
                    )
            observation, terminated, truncated = gate._step_decision(
                env, observation, executed, maximum_tick=maximum_ticks
            )
            decisions += 1
        episode = {
            "seed": seed,
            "iteration": iteration,
            "final_tick": int(observation["tick"]),
            "final_score": int(observation.get("score", 0)),
            "final_gauge": int(observation.get("gauge", 0)),
            "terminated": bool(terminated or observation.get("terminated", False)),
            "truncated": bool(truncated or observation.get("truncated", False)),
            "decisions": decisions,
            "proposed_shots": proposals,
            "deployment_kept": deployment_kept,
            "deployment_suppressed": deployment_suppressed,
            "teacher_changes": teacher_changes,
            "candidate_branches": candidates_evaluated,
            "pair_preferences": len(preferences),
            "causal_act_preferences": len(act_preferences),
            "act_labels": len(labels),
            "long_horizon_queries": long_queries,
            "wait_anchors": wait_anchors,
            "gate_reasons": dict(sorted(reasons.items())),
            "wall_seconds": time.monotonic() - started,
        }
    return labels, act_preferences, preferences, episode, runtime_manifest


def _preference_tensors(
    preferences: Sequence[PairPreference], indices: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    chosen = [preferences[index] for index in indices]
    width = max(
        int(np.flatnonzero(value.observation.body_mask[0])[-1]) + 1
        for value in chosen
    )
    return (
        torch.from_numpy(np.concatenate([value.observation.global_features for value in chosen])),
        torch.from_numpy(np.concatenate([value.observation.body_features[:, :width] for value in chosen])),
        torch.from_numpy(np.concatenate([value.observation.body_mask[:, :width] for value in chosen])),
        torch.tensor([value.positive_source for value in chosen], dtype=torch.long),
        torch.tensor([value.positive_destination for value in chosen], dtype=torch.long),
        torch.tensor([value.negative_source for value in chosen], dtype=torch.long),
        torch.tensor([value.negative_destination for value in chosen], dtype=torch.long),
    )


def _act_preference_tensors(
    preferences: Sequence[ActPreference], indices: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    chosen = [preferences[index] for index in indices]
    width = max(
        int(np.flatnonzero(value.observation.body_mask[0])[-1]) + 1
        for value in chosen
    )
    return (
        torch.from_numpy(np.concatenate([value.observation.global_features for value in chosen])),
        torch.from_numpy(np.concatenate([value.observation.body_features[:, :width] for value in chosen])),
        torch.from_numpy(np.concatenate([value.observation.body_mask[:, :width] for value in chosen])),
        torch.tensor([value.execute_shot for value in chosen], dtype=torch.bool),
        torch.tensor([value.confidence for value in chosen], dtype=torch.float32),
    )


def train_ranked_heads(
    model: GoalConditionedSteeringModel,
    act_examples: Sequence[SteeringExample],
    act_preferences: Sequence[ActPreference],
    preferences: Sequence[PairPreference],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    causal_weight: float,
    ranking_weight: float,
    seed: int,
) -> RankingTrainingReport:
    if not act_examples or steps < 1 or batch_size < 1:
        raise ValueError("ranking training requires act examples and positive counts")
    train_act = any(parameter.requires_grad for parameter in model.act_head.parameters())
    train_pair = any(parameter.requires_grad for parameter in model.pair_head.parameters())
    if train_pair and not preferences:
        raise ValueError(
            "pair-head training requires at least one non-tied, cross-pair preference"
        )
    dataset = SteeringDataset(gate.balanced_examples(act_examples))
    device = next(model.parameters()).device

    def losses() -> tuple[torch.Tensor, ...]:
        model.eval()
        act_loss_sum = act_correct = 0.0
        for start in range(0, len(dataset), batch_size):
            indices = range(start, min(start + batch_size, len(dataset)))
            act_batch = dataset.as_tensors(indices).to(device)
            act_output = model(
                act_batch.global_features, act_batch.body_features, act_batch.body_mask
            )
            act_loss_sum += float(
                F.cross_entropy(
                    act_output.act_logits, act_batch.act_index, reduction="sum"
                )
            )
            act_correct += float(
                (act_output.act_logits.argmax(-1) == act_batch.act_index).sum()
            )
        act_loss = torch.tensor(act_loss_sum / len(dataset), device=device)
        act_accuracy = torch.tensor(act_correct / len(dataset), device=device)
        if act_preferences:
            causal_numerator = causal_weight_sum = causal_correct = 0.0
            for start in range(0, len(act_preferences), batch_size):
                indices = range(start, min(start + batch_size, len(act_preferences)))
                tensors = tuple(
                    value.to(device)
                    for value in _act_preference_tensors(act_preferences, indices)
                )
                global_features, body_features, body_mask, target, confidence = tensors
                causal_output = model(global_features, body_features, body_mask)
                signed = causal_output.act_logits[:, 1] - causal_output.act_logits[:, 0]
                signed = torch.where(target, signed, -signed)
                causal_numerator += float((confidence * F.softplus(-signed)).sum())
                causal_weight_sum += float(confidence.sum())
                causal_correct += float((signed > 0).sum())
            causal_loss = torch.tensor(causal_numerator / causal_weight_sum, device=device)
            causal_accuracy = torch.tensor(
                causal_correct / len(act_preferences), device=device
            )
        else:
            causal_loss = act_loss.detach() * 0.0
            causal_accuracy = causal_loss
        if not preferences:
            zero = act_loss.detach() * 0.0
            return act_loss, causal_loss, zero, act_accuracy, causal_accuracy, zero
        pair_loss_sum = pair_correct = 0.0
        for start in range(0, len(preferences), batch_size):
            indices = range(start, min(start + batch_size, len(preferences)))
            tensors = tuple(
                value.to(device) for value in _preference_tensors(preferences, indices)
            )
            global_features, body_features, body_mask, ps, pd, ns, nd = tensors
            output = model(global_features, body_features, body_mask)
            rows = torch.arange(len(ps), device=device)
            difference = output.pair_logits[rows, ps, pd] - output.pair_logits[rows, ns, nd]
            pair_loss_sum += float(F.softplus(-difference).sum())
            pair_correct += float((difference > 0).sum())
        return (
            act_loss,
            causal_loss,
            torch.tensor(pair_loss_sum / len(preferences), device=device),
            act_accuracy,
            causal_accuracy,
            torch.tensor(pair_correct / len(preferences), device=device),
        )

    with torch.no_grad():
        initial_act, initial_causal, initial_pair, *_accuracies = losses()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=learning_rate,
    )
    model.eval()
    for _ in range(steps):
        act_indices = torch.randint(
            len(dataset), (min(batch_size, len(dataset)),), generator=generator
        ).tolist()
        act_batch = dataset.as_tensors(act_indices).to(device)
        act_output = model(
            act_batch.global_features, act_batch.body_features, act_batch.body_mask
        )
        loss = (
            F.cross_entropy(act_output.act_logits, act_batch.act_index)
            if train_act
            else act_output.act_logits.sum() * 0.0
        )
        if train_act and act_preferences:
            causal_indices = torch.randint(
                len(act_preferences),
                (min(batch_size, len(act_preferences)),),
                generator=generator,
            ).tolist()
            tensors = tuple(
                value.to(device)
                for value in _act_preference_tensors(act_preferences, causal_indices)
            )
            global_features, body_features, body_mask, target, confidence = tensors
            causal_output = model(global_features, body_features, body_mask)
            signed = causal_output.act_logits[:, 1] - causal_output.act_logits[:, 0]
            signed = torch.where(target, signed, -signed)
            loss = loss + causal_weight * (
                confidence * F.softplus(-signed)
            ).sum() / confidence.sum()
        if preferences:
            pref_indices = torch.randint(
                len(preferences),
                (min(batch_size, len(preferences)),),
                generator=generator,
            ).tolist()
            tensors = tuple(value.to(device) for value in _preference_tensors(preferences, pref_indices))
            global_features, body_features, body_mask, ps, pd, ns, nd = tensors
            output = model(global_features, body_features, body_mask)
            rows = torch.arange(len(pref_indices), device=device)
            difference = output.pair_logits[rows, ps, pd] - output.pair_logits[rows, ns, nd]
            loss = loss + ranking_weight * F.softplus(-difference).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad], 5.0
        )
        optimizer.step()
    with torch.no_grad():
        (
            final_act,
            final_causal,
            final_pair,
            act_accuracy,
            causal_accuracy,
            pair_accuracy,
        ) = losses()
    return RankingTrainingReport(
        steps,
        len(dataset),
        len(act_preferences),
        len(preferences),
        float(initial_act),
        float(final_act),
        float(initial_causal),
        float(final_causal),
        float(initial_pair),
        float(final_pair),
        float(act_accuracy),
        float(causal_accuracy),
        float(pair_accuracy),
    )


def _source_identity() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        Path(gate.__file__).resolve(),
        Path(search.__file__).resolve(),
        ROOT / "python/irisu_pointer/steering_learning.py",
    )
    value = {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {str(path.relative_to(ROOT)): gate._file_sha(path) for path in paths},
    }
    return {**value, "sha256": gate._sha(value)}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    if (
        not args.train_seeds
        or len(set(args.train_seeds)) != len(args.train_seeds)
        or any(type(seed) is not int or not 0 <= seed <= 0xFFFF_FFFF for seed in args.train_seeds)
    ):
        raise ValueError("caller-supplied train seeds must be unique uint32 values")
    artifact = load_steering_checkpoint(
        args.base_checkpoint, expected_sha256=args.base_sha256
    )
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    ranking_seeds = tuple(sorted(args.train_seeds))
    training_seeds = tuple(sorted(set(upstream) | set(ranking_seeds)))
    runtime = args.runtime.resolve(strict=True)
    inference = gate.inference_config(args.act_logit_bias)
    planner = {
        "probe_ticks": args.short_horizon,
        "rescue_probe_ticks": args.long_horizon,
        "rescue_gauge_threshold": args.gauge_threshold,
        "wait_ticks": args.wait_ticks,
        "maximum_gauge_debt": args.maximum_gauge_debt,
        "rescue_score_margin": args.rescue_score_margin,
        "maximum_ranked_pairs": args.maximum_pairs,
    }
    warm_start = {
        "path": str(artifact.path),
        "checkpoint_sha256": artifact.sha256,
        "metadata_sha256": gate._sha(dict(artifact.metadata)),
        "model_state_sha256": gate._model_state_sha(artifact.model),
        "architecture_sha256": artifact.model.architecture_sha256,
    }
    args.output.mkdir(parents=True)
    model = artifact.model
    trainable = configure_trainable_scope(model, args.trainable_scope)
    all_labels: list[gate.Label] = []
    all_act_preferences: list[ActPreference] = []
    all_preferences: list[PairPreference] = []
    episodes: list[dict[str, object]] = []
    runtimes: list[Mapping[str, Any]] = []
    training_reports: list[dict[str, object]] = []
    started = time.monotonic()
    for iteration in range(args.iterations):
        rollout_model_sha256 = gate._model_state_sha(model)
        for seed in args.train_seeds:
            labels, act_preferences, preferences, episode, runtime_manifest = collect_episode(
                runtime=runtime,
                model=model,
                model_sha256=rollout_model_sha256,
                seed=seed,
                maximum_ticks=args.maximum_ticks,
                short_horizon=args.short_horizon,
                long_horizon=args.long_horizon,
                gauge_threshold=args.gauge_threshold,
                maximum_gauge_debt=args.maximum_gauge_debt,
                rescue_score_margin=args.rescue_score_margin,
                wait_ticks=args.wait_ticks,
                maximum_pairs=args.maximum_pairs,
                wait_anchor_stride=args.wait_anchor_stride,
                maximum_wait_anchors=args.maximum_wait_anchors,
                act_logit_bias=args.act_logit_bias,
                iteration=iteration,
            )
            all_labels.extend(labels)
            all_act_preferences.extend(act_preferences)
            all_preferences.extend(preferences)
            episode["rollout_model_state_sha256"] = rollout_model_sha256
            episodes.append(episode)
            runtimes.append(runtime_manifest)
            print(json.dumps(episode, sort_keys=True), flush=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.training_seed + iteration)
            training_reports.append(
                asdict(
                    train_ranked_heads(
                        model,
                        [value.example for value in all_labels],
                        all_act_preferences,
                        all_preferences,
                        steps=args.training_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        causal_weight=args.causal_weight,
                        ranking_weight=args.ranking_weight,
                        seed=args.training_seed + iteration,
                    )
                )
            )
    act_dataset = SteeringDataset([value.example for value in all_labels])
    balanced_act = SteeringDataset(gate.balanced_examples([value.example for value in all_labels]))
    preference_manifest = [value.manifest() for value in all_preferences]
    act_preference_manifest = [value.manifest() for value in all_act_preferences]
    payload = {
        "schema": "irisu-portable-pair-ranking-labels-v1",
        "base_checkpoint_sha256": artifact.sha256,
        "upstream_training_seeds": list(upstream),
        "ranking_training_seeds": list(ranking_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": inference,
        "adaptive_wait_gate": planner,
        "act_dataset_manifest": act_dataset.manifest(),
        "balanced_act_dataset_manifest": balanced_act.manifest(),
        "causal_act_dataset_sha256": gate._sha(act_preference_manifest),
        "preference_dataset_sha256": gate._sha(preference_manifest),
        "act_labels": [
            {
                "manifest": value.manifest(),
                "example": value.example.manifest(),
                "global_features": torch.from_numpy(value.example.observation.global_features),
                "body_features": torch.from_numpy(value.example.observation.body_features),
                "body_mask": torch.from_numpy(value.example.observation.body_mask),
            }
            for value in all_labels
        ],
        "pair_preferences": [
            {
                "manifest": value.manifest(),
                "global_features": torch.from_numpy(value.observation.global_features),
                "body_features": torch.from_numpy(value.observation.body_features),
                "body_mask": torch.from_numpy(value.observation.body_mask),
            }
            for value in all_preferences
        ],
        "causal_act_preferences": [
            {
                "manifest": value.manifest(),
                "global_features": torch.from_numpy(value.observation.global_features),
                "body_features": torch.from_numpy(value.observation.body_features),
                "body_mask": torch.from_numpy(value.observation.body_mask),
            }
            for value in all_act_preferences
        ],
    }
    labels_path = args.output / "ranking-labels.pt"
    gate._save_torch_new(labels_path, payload)
    labels_sha256 = gate._file_sha(labels_path)
    source = _source_identity()
    deterministic_episodes = [
        {key: value for key, value in row.items() if key != "wall_seconds"}
        for row in episodes
    ]
    metadata = {
        "schema": "irisu-portable-pair-ranking-distillation-v1",
        "development_only": True,
        "promotion_eligible": False,
        "held_out_seeds_used": False,
        "base_checkpoint_sha256": artifact.sha256,
        "warm_start": warm_start,
        "upstream_training_seeds": list(upstream),
        "ranking_training_seeds": list(ranking_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": inference,
        "adaptive_wait_gate": planner,
        "runtime": {"path": str(runtime), "sha256": gate._file_sha(runtime)},
        "source": source,
        "act_dataset_sha256": act_dataset.sha256,
        "balanced_act_dataset_sha256": balanced_act.sha256,
        "causal_act_dataset_sha256": gate._sha(act_preference_manifest),
        "preference_dataset_sha256": gate._sha(preference_manifest),
        "labels_artifact_sha256": labels_sha256,
        "act_label_count": len(all_labels),
        "causal_act_preference_count": len(all_act_preferences),
        "pair_preference_count": len(all_preferences),
        "label_counts": dict(sorted(Counter(value.kind for value in all_labels).items())),
        "episodes": deterministic_episodes,
        "scores": [int(row["final_score"]) for row in episodes],
        "median_score": statistics.median(int(row["final_score"]) for row in episodes),
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": list(trainable),
        "training_reports": training_reports,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    }
    checkpoint_path = args.output / "pair-ranked.pt"
    checkpoint_sha256 = save_steering_checkpoint(
        checkpoint_path, model, metadata=metadata
    )
    report = {
        **metadata,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "labels_artifact": labels_path.name,
        "runtime_manifests": runtimes,
        "operational": {
            "output": str(args.output.resolve()),
            "episode_wall_seconds": [float(row["wall_seconds"]) for row in episodes],
        },
        "wall_seconds": time.monotonic() - started,
    }
    report["sha256"] = gate._sha(report)
    gate._write_json_new(args.output / "provenance.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--train-seeds", type=gate.parse_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=gate.PORTABLE)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--maximum-ticks", type=int, default=20_000)
    parser.add_argument("--short-horizon", type=int, default=128)
    parser.add_argument("--long-horizon", type=int, default=256)
    parser.add_argument("--gauge-threshold", type=int, default=30_000)
    parser.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    parser.add_argument("--rescue-score-margin", type=int, default=500)
    parser.add_argument("--wait-ticks", type=int, default=16)
    parser.add_argument("--maximum-pairs", type=int, default=4)
    parser.add_argument("--wait-anchor-stride", type=int, default=16)
    parser.add_argument("--maximum-wait-anchors", type=int, default=64)
    parser.add_argument("--act-logit-bias", type=float, default=1.0)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--causal-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--training-seed", type=int, default=2026081101)
    parser.add_argument(
        "--trainable-scope",
        choices=("pair-head", "act-pair", "heads", "all"),
        default="pair-head",
    )
    args = parser.parse_args(argv)
    positive = (
        "iterations",
        "maximum_ticks",
        "short_horizon",
        "long_horizon",
        "maximum_gauge_debt",
        "rescue_score_margin",
        "wait_ticks",
        "maximum_pairs",
        "wait_anchor_stride",
        "training_steps",
        "batch_size",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("counts and horizons must be positive")
    if args.long_horizon < args.short_horizon:
        parser.error("long horizon must be at least short horizon")
    if args.gauge_threshold < 0 or args.maximum_wait_anchors < 0:
        parser.error("gauge threshold and maximum wait anchors must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(args.ranking_weight) or args.ranking_weight <= 0:
        parser.error("ranking weight must be finite and positive")
    if not math.isfinite(args.causal_weight) or args.causal_weight <= 0:
        parser.error("causal weight must be finite and positive")
    if len(args.base_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in args.base_sha256
    ):
        parser.error("base SHA-256 must be lowercase hexadecimal")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "act_labels": report["act_label_count"],
                "causal_act_preferences": report["causal_act_preference_count"],
                "pair_preferences": report["pair_preference_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
