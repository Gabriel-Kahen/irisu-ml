#!/usr/bin/env python3
"""Development-only exact evaluation of a fail-closed pair-proposal blend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_env import Action, ActionKind, ExactWorkerError  # noqa: E402
from irisu_pointer.development_proposal_blend import (  # noqa: E402
    BLEND_VERSION,
    FailClosedProposalBlendPlanner,
    ModelPairProposalRanker,
    ProposalBlendConfig,
)
from irisu_pointer.fast_multiaction_planner import FastMultiActionConfig  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402
from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    FULL_GAME_TICKS,
    PolicyBundle,
    bind_report_content_sha256,
    canonical_sha256,
    file_sha256,
    load_declared_training_seeds,
    load_policy_bundle,
    parse_seeds,
    primitive_actions,
    validate_contract,
)
from rl_exact_fast_multiaction_eval import (  # noqa: E402
    _effective_score,
    _trace_point,
    _validated_action,
    _write_new,
)


FORMAT = "irisu-exact-fail-closed-proposal-blend-development-v1"
BRANCH_ERROR_POLICY = {
    "version": "exact-branch-error-conservative-wait-v1",
    "caught_exception": "irisu_env.exact_ipc.ExactWorkerError",
    "action": "restore-base-policy-and-execute-wait",
    "require_live_parent_state_hash_unchanged": True,
}


def _model_schema_sha(bundle: PolicyBundle) -> str:
    policy = bundle.factory()
    schema = getattr(getattr(policy, "model"), "schema")
    digest = getattr(schema, "sha256", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint model schema lacks a SHA-256 identity")
    return digest


def validate_blend_contract(
    base: PolicyBundle,
    residual: PolicyBundle,
    base_training_seeds: Sequence[int],
    residual_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base_declared, evaluation = validate_contract(
        base, base_training_seeds, evaluation_seeds
    )
    residual_declared, residual_evaluation = validate_contract(
        residual, residual_training_seeds, evaluation_seeds
    )
    if evaluation != residual_evaluation:
        raise RuntimeError("base and residual evaluation seeds differ")
    if _model_schema_sha(base) != _model_schema_sha(residual):
        raise ValueError("base and residual model schemas differ")
    combined = tuple(sorted(set(base_declared) | set(residual_declared)))
    overlap = sorted(set(combined) & set(evaluation))
    if overlap:
        raise ValueError(f"training/evaluation seed overlap: {overlap}")
    return combined, evaluation


def _ranker(bundle: PolicyBundle) -> ModelPairProposalRanker:
    policy = bundle.factory()
    return ModelPairProposalRanker(policy.encoder, policy.model)


def run_blend_episode(
    runtime: ExactTrainingRuntime,
    base_bundle: PolicyBundle,
    residual_ranker: ModelPairProposalRanker,
    seed: int,
    *,
    maximum_ticks: int,
    planner_config: FastMultiActionConfig,
    blend_config: ProposalBlendConfig,
    target_score: int,
    trace_interval_ticks: int,
    maximum_logged_queries: int,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    policy = base_bundle.factory()
    policy.reset(seed)
    planner = FailClosedProposalBlendPlanner(
        primitive_actions,
        residual_ranker,
        config=planner_config,
        blend_config=blend_config,
        action_spec=policy.action_spec,
    )
    queries = branch_checks = branch_invalid = 0
    policy_invalid = simulator_invalid = action_count = 0
    selected: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    probe_modes: Counter[str] = Counter()
    probe_horizons: Counter[str] = Counter()
    query_log: list[dict[str, object]] = []
    branch_error_events: list[dict[str, object]] = []
    terminated = truncated = False
    final_info: Mapping[str, Any] = {}
    started = time.monotonic()
    simulation_config = {
        "max_episode_ticks": maximum_ticks
        + max(planner_config.probe_ticks, planner_config.long_probe_ticks)
    }
    with runtime.open_env(simulation_config=simulation_config) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("exact environment reset returned a different seed")
        trace = [_trace_point(env, observation)]
        next_trace = trace_interval_ticks
        while int(observation["tick"]) < maximum_ticks and not (
            terminated or truncated
        ):
            policy_before = copy.deepcopy(policy)
            try:
                prediction = policy.predict(observation)
            except (AttributeError, TypeError, ValueError, OverflowError):
                policy_invalid += 1
                policy = policy_before
                actions = (Action.wait(1),)
            else:
                decision = prediction
                if bool(getattr(prediction, "is_shot", False)):
                    queries += 1
                    parent_hash_before = int(env.state_hash())
                    try:
                        verdict = planner.evaluate(
                            env,
                            observation,
                            policy_before,
                            policy,
                            prediction,
                        )
                    except ExactWorkerError as exc:
                        parent_hash_after = int(env.state_hash())
                        if parent_hash_after != parent_hash_before:
                            raise RuntimeError(
                                "exact branch failure altered the live parent state"
                            ) from exc
                        message = str(exc)
                        event = {
                            "tick": int(observation["tick"]),
                            "state_hash_before": parent_hash_before,
                            "state_hash_after": parent_hash_after,
                            "exception_type": "irisu_env.exact_ipc.ExactWorkerError",
                            "message": message,
                            "message_sha256": hashlib.sha256(
                                message.encode("utf-8")
                            ).hexdigest(),
                            "recovery": "restore-base-policy-and-execute-wait",
                            "parent_state_unchanged": True,
                            "probe_mode": planner.probe_mode,
                            "probe_ticks": planner.current_probe_ticks,
                        }
                        branch_error_events.append(event)
                        probe_modes[planner.probe_mode] += 1
                        probe_horizons[str(planner.current_probe_ticks)] += 1
                        category = "wait-exact-branch-error"
                        selected[category] += 1
                        reasons[category] += 1
                        policy = policy_before
                        decision = planner.wait_decision(category)
                        if len(query_log) < maximum_logged_queries:
                            query_log.append(
                                {
                                    "tick": int(observation["tick"]),
                                    "gauge": int(observation["gauge"]),
                                    "selected_category": category,
                                    "reason": category,
                                    "exact_branch_error": event,
                                }
                            )
                    else:
                        probe_modes[verdict.probe_mode] += 1
                        probe_horizons[str(verdict.probe_ticks)] += 1
                        branch_checks += verdict.branch_checks
                        branch_invalid += sum(
                            value.invalid_actions for value in verdict.outcomes
                        )
                        selected[verdict.selected.category] += 1
                        reasons[verdict.reason] += 1
                        if len(query_log) < maximum_logged_queries:
                            query_log.append(
                                {
                                    "tick": int(observation["tick"]),
                                    "gauge": int(observation["gauge"]),
                                    **verdict.manifest(),
                                }
                            )
                        policy = verdict.selected.continuation_policy
                        decision = verdict.selected.decision
                try:
                    actions = primitive_actions(decision)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    policy_invalid += 1
                    policy = policy_before
                    actions = (Action.wait(1),)
            for raw_action in actions:
                remaining = maximum_ticks - int(observation["tick"])
                if remaining <= 0 or terminated or truncated:
                    break
                try:
                    action = _validated_action(raw_action, remaining)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    policy_invalid += 1
                    action = Action.wait(1)
                kind = ActionKind.parse(action.kind)
                duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
                for _ in range(duration):
                    primitive = Action.wait(1) if kind is ActionKind.WAIT else action
                    observation, _reward, terminated, truncated, final_info = env.step(
                        primitive
                    )
                    action_count += 1
                    simulator_invalid += int(
                        bool(final_info.get("invalid_action", False))
                    )
                    if int(observation["tick"]) >= next_trace:
                        trace.append(_trace_point(env, observation))
                        while next_trace <= int(observation["tick"]):
                            next_trace += trace_interval_ticks
                    if terminated or truncated:
                        break
        final_point = _trace_point(env, observation)
        if trace[-1] != final_point:
            trace.append(final_point)
        provenance = session.provenance_manifest
    score = _effective_score(observation, final_info)
    result: dict[str, object] = {
        "seed": seed,
        "score": score,
        "success": score >= target_score,
        "tick": int(observation["tick"]),
        "level": int(observation.get("level", 0)),
        "gauge": int(observation.get("gauge", 0)),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "terminated": bool(terminated or observation.get("terminated", False)),
        "truncated": bool(truncated or observation.get("truncated", False)),
        "policy_invalid_actions": policy_invalid,
        "simulator_invalid_actions": simulator_invalid,
        "invalid_actions": policy_invalid + simulator_invalid,
        "counterfactual_invalid_actions": branch_invalid,
        "action_count": action_count,
        "planner_queries": queries,
        "planner_branch_checks": branch_checks,
        "planner_probe_mode_counts": dict(sorted(probe_modes.items())),
        "planner_probe_horizon_counts": dict(sorted(probe_horizons.items())),
        "selected_categories": dict(sorted(selected.items())),
        "selection_reasons": dict(sorted(reasons.items())),
        "proposal_counts": dict(sorted(planner.proposal_counts.items())),
        "logged_query_count": len(query_log),
        "query_log": query_log,
        "query_log_sha256": canonical_sha256(query_log),
        "trace": trace,
        "trace_sha256": canonical_sha256(trace),
        "exact_branch_errors": len(branch_error_events),
        "residual_branch_errors": planner.proposal_counts[
            "residual-branch-error-base-fallback"
        ],
        "exact_branch_error_events": branch_error_events,
        "exact_branch_error_events_sha256": canonical_sha256(branch_error_events),
        "wall_seconds": time.monotonic() - started,
    }
    return result, provenance


def evaluate_blend(
    runtime: ExactTrainingRuntime,
    base_bundle: PolicyBundle,
    residual_bundle: PolicyBundle,
    *,
    base_training_seeds: Sequence[int],
    residual_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    maximum_ticks: int = FULL_GAME_TICKS,
    planner_config: FastMultiActionConfig | None = None,
    blend_config: ProposalBlendConfig | None = None,
    target_score: int = 100_000,
    trace_interval_ticks: int = 10_000,
    maximum_logged_queries: int = 128,
    episode_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    training, evaluation = validate_blend_contract(
        base_bundle,
        residual_bundle,
        base_training_seeds,
        residual_training_seeds,
        evaluation_seeds,
    )
    config = (
        FastMultiActionConfig(probe_ticks=512, long_probe_ticks=512, top_k_pairs=2)
        if planner_config is None
        else planner_config
    )
    blend = ProposalBlendConfig() if blend_config is None else blend_config
    if target_score < 0:
        raise ValueError("target_score must be nonnegative")
    for name, value in (
        ("maximum_ticks", maximum_ticks),
        ("trace_interval_ticks", trace_interval_ticks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_logged_queries < 0:
        raise ValueError("maximum_logged_queries must be nonnegative")
    ranker = _ranker(residual_bundle)
    episodes: list[dict[str, object]] = []
    runtime_provenance: Mapping[str, Any] | None = None
    runtime_provenance_sha256: str | None = None
    started = time.monotonic()
    for seed in evaluation:
        episode, current_provenance = run_blend_episode(
            runtime,
            base_bundle,
            ranker,
            seed,
            maximum_ticks=maximum_ticks,
            planner_config=config,
            blend_config=blend,
            target_score=target_score,
            trace_interval_ticks=trace_interval_ticks,
            maximum_logged_queries=maximum_logged_queries,
        )
        current_sha = canonical_sha256(current_provenance)
        if runtime_provenance_sha256 not in (None, current_sha):
            raise RuntimeError("exact runtime provenance changed between episodes")
        runtime_provenance = current_provenance
        runtime_provenance_sha256 = current_sha
        episodes.append(episode)
        if episode_callback is not None:
            episode_callback(episode)
    scores = [int(value["score"]) for value in episodes]
    successes = sum(score >= target_score for score in scores)
    invalid = sum(int(value["invalid_actions"]) for value in episodes)
    proposal_counts: Counter[str] = Counter()
    for episode in episodes:
        proposal_counts.update(episode["proposal_counts"])
    blend_source = ROOT / "python/irisu_pointer/development_proposal_blend.py"
    base_source = ROOT / "python/irisu_pointer/fast_multiaction_planner.py"
    helper_source = ROOT / "benchmarks/rl_exact_fast_multiaction_eval.py"
    report: dict[str, object] = {
        "format": FORMAT,
        "physics_backend": "exact",
        "evidence_class": "development-only",
        "promotion_eligible": False,
        "deterministic_policy": True,
        "blend_version": BLEND_VERSION,
        "blend_config": blend.manifest(),
        "base_planner_config": config.manifest(),
        "blend_objective": (
            "base-winner-reserve-noninferiority-then-score-objective-v1"
        ),
        "branch_error_policy": dict(BRANCH_ERROR_POLICY),
        "training_seeds": list(training),
        "training_seeds_sha256": canonical_sha256(list(training)),
        "base_training_seeds": list(base_training_seeds),
        "residual_training_seeds": list(residual_training_seeds),
        "evaluation_seeds": list(evaluation),
        "evaluation_seeds_sha256": canonical_sha256(list(evaluation)),
        "training_evaluation_overlap": [],
        "maximum_ticks": maximum_ticks,
        "target_score": target_score,
        "base_checkpoint": {
            "path": base_bundle.checkpoint_path,
            "checkpoint_sha256": base_bundle.checkpoint_sha256,
            "model_sha256": base_bundle.model_sha256,
            "metadata": dict(base_bundle.metadata),
            "metadata_sha256": canonical_sha256(base_bundle.metadata),
            "inference_config": dict(base_bundle.inference_config),
        },
        "residual_checkpoint": {
            "path": residual_bundle.checkpoint_path,
            "checkpoint_sha256": residual_bundle.checkpoint_sha256,
            "model_sha256": residual_bundle.model_sha256,
            "metadata": dict(residual_bundle.metadata),
            "metadata_sha256": canonical_sha256(residual_bundle.metadata),
            "inference_config": dict(residual_bundle.inference_config),
            "role": "stateless-pair-ranker-only",
        },
        "model_schema_sha256": _model_schema_sha(base_bundle),
        "exact_runtime": dict(runtime_provenance or {}),
        "runtime_hashes": {
            "worker_sha256": runtime.identity.worker_sha256,
            "exact_library_sha256": runtime.identity.exact_library_sha256,
            "identity_config_sha256": runtime.identity.config_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
        },
        "base_planner_source_sha256": file_sha256(base_source),
        "blend_planner_source_sha256": file_sha256(blend_source),
        "runner_helpers_source_sha256": file_sha256(helper_source),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "episodes": episodes,
        "episode_count": len(episodes),
        "scores": scores,
        "median_score": float(statistics.median(scores)),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(scores),
        "invalid_actions": invalid,
        "counterfactual_invalid_actions": sum(
            int(value["counterfactual_invalid_actions"]) for value in episodes
        ),
        "exact_branch_errors": sum(
            int(value["exact_branch_errors"]) for value in episodes
        ),
        "residual_branch_errors": sum(
            int(value["residual_branch_errors"]) for value in episodes
        ),
        "proposal_counts": dict(sorted(proposal_counts.items())),
        "passed": False,
        "wall_seconds": time.monotonic() - started,
    }
    bind_report_content_sha256(report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--base-checkpoint", type=Path, required=True)
    result.add_argument("--base-checkpoint-sha256", required=True)
    result.add_argument("--base-training-seed-manifest", type=Path, required=True)
    result.add_argument("--residual-checkpoint", type=Path, required=True)
    result.add_argument("--residual-checkpoint-sha256", required=True)
    result.add_argument("--residual-training-seed-manifest", type=Path, required=True)
    result.add_argument("--seeds", type=parse_seeds, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--maximum-ticks", type=int, default=FULL_GAME_TICKS)
    result.add_argument("--probe-ticks", type=int, default=512)
    result.add_argument("--wait-ticks", type=int, default=16)
    result.add_argument("--low-gauge-threshold", type=int, default=20_000)
    result.add_argument("--top-k-pairs", type=int, default=2)
    result.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    result.add_argument("--rescue-score-margin", type=int, default=500)
    result.add_argument("--activation-tick-exclusive", type=int, default=50_000)
    result.add_argument("--activation-gauge-exclusive", type=int, default=20_000)
    result.add_argument("--target-score", type=int, default=100_000)
    result.add_argument("--trace-interval-ticks", type=int, default=10_000)
    result.add_argument("--maximum-logged-queries", type=int, default=128)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.worker.is_absolute():
        parser().error("--worker must be absolute")
    if args.output.exists():
        parser().error("--output must not already exist")
    base = load_policy_bundle(
        args.base_checkpoint, expected_sha256=args.base_checkpoint_sha256
    )
    residual = load_policy_bundle(
        args.residual_checkpoint, expected_sha256=args.residual_checkpoint_sha256
    )
    report = evaluate_blend(
        ExactTrainingRuntime(args.worker),
        base,
        residual,
        base_training_seeds=load_declared_training_seeds(
            args.base_training_seed_manifest.resolve(strict=True)
        ),
        residual_training_seeds=load_declared_training_seeds(
            args.residual_training_seed_manifest.resolve(strict=True)
        ),
        evaluation_seeds=args.seeds,
        maximum_ticks=args.maximum_ticks,
        planner_config=FastMultiActionConfig(
            probe_ticks=args.probe_ticks,
            long_probe_ticks=args.probe_ticks,
            wait_ticks=args.wait_ticks,
            low_gauge_threshold=args.low_gauge_threshold,
            low_gauge_exit_threshold=args.low_gauge_threshold,
            top_k_pairs=args.top_k_pairs,
            maximum_gauge_debt=args.maximum_gauge_debt,
            rescue_score_margin=args.rescue_score_margin,
            gauge_advantage=1,
        ),
        blend_config=ProposalBlendConfig(
            activation_tick_exclusive=args.activation_tick_exclusive,
            activation_gauge_exclusive=args.activation_gauge_exclusive,
        ),
        target_score=args.target_score,
        trace_interval_ticks=args.trace_interval_ticks,
        maximum_logged_queries=args.maximum_logged_queries,
        episode_callback=lambda row: print(
            json.dumps(
                {
                    "seed": row["seed"],
                    "score": row["score"],
                    "tick": row["tick"],
                    "proposal_counts": row["proposal_counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        ),
    )
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
