#!/usr/bin/env python3
"""Development-only exact evaluation of the fast multi-action planner.

The caller supplies every seed.  This evaluator intentionally cannot claim
promotion eligibility; it is a tuning instrument for the frozen checkpoint
and a bounded STRONG/WEAK/WAIT plus low-gauge top-k counterfactual planner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_env import Action, ActionKind, ExactWorkerError  # noqa: E402
from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    FastMultiActionConfig,
    FastMultiActionPlanner,
)
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


FORMAT = "irisu-exact-fast-multiaction-development-eval-v1"
DEFAULT_TARGET_SCORE = 100_000


def _validated_action(value: object, remaining_ticks: int) -> Action:
    kind = ActionKind.parse(getattr(value, "kind"))
    x = float(getattr(value, "cursor_x"))
    y = float(getattr(value, "cursor_y"))
    wait_ticks = int(getattr(value, "wait_ticks"))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("policy cursor coordinates must be finite")
    if kind is ActionKind.WAIT:
        if wait_ticks < 1:
            raise ValueError("policy wait duration must be positive")
        return Action.wait(min(wait_ticks, remaining_ticks))
    if wait_ticks != 1:
        raise ValueError("shot actions must span exactly one tick")
    return Action(kind, x, y, 1)


def _effective_score(observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
    diagnostics = info.get("diagnostics")
    if diagnostics is not None and bool(
        getattr(diagnostics, "terminal_metadata_recorded", False)
    ):
        return int(getattr(diagnostics, "recorded_final_score"))
    return int(observation["score"])


def _trace_point(env: object, observation: Mapping[str, Any]) -> dict[str, int]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "level": int(observation.get("level", 0)),
        "gauge": int(observation.get("gauge", 0)),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "state_hash": int(env.state_hash()),
    }


def run_episode(
    runtime: ExactTrainingRuntime,
    bundle: PolicyBundle,
    seed: int,
    *,
    maximum_ticks: int,
    planner_config: FastMultiActionConfig,
    target_score: int,
    trace_interval_ticks: int,
    maximum_logged_queries: int,
    recover_exact_branch_errors: bool = False,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    policy = bundle.factory()
    policy.reset(seed)
    planner = FastMultiActionPlanner(
        primitive_actions,
        config=planner_config,
        action_spec=policy.action_spec,
    )
    queries = branch_checks = branch_invalid = 0
    policy_invalid = simulator_invalid = action_count = 0
    selected: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    probe_modes: Counter[str] = Counter()
    probe_horizons: Counter[str] = Counter()
    probe_schedule: list[dict[str, object]] = []
    probe_mode_switches: list[dict[str, object]] = []
    previous_probe_mode = "normal"
    query_log: list[dict[str, object]] = []
    branch_error_events: list[dict[str, object]] = []
    terminated = truncated = False
    final_info: Mapping[str, Any] = {}
    started = time.monotonic()
    simulation_config = {
        "max_episode_ticks": maximum_ticks
        + max(planner_config.probe_ticks, planner_config.long_probe_ticks)
    }

    def record_probe(mode: str, horizon: int) -> None:
        nonlocal previous_probe_mode
        probe_modes[mode] += 1
        probe_horizons[str(horizon)] += 1
        probe_schedule.append(
            {
                "tick": int(observation["tick"]),
                "gauge": int(observation["gauge"]),
                "probe_mode": mode,
                "probe_ticks": horizon,
            }
        )
        if mode != previous_probe_mode:
            probe_mode_switches.append(
                {
                    "tick": int(observation["tick"]),
                    "gauge": int(observation["gauge"]),
                    "from": previous_probe_mode,
                    "to": mode,
                    "probe_ticks": horizon,
                }
            )
            previous_probe_mode = mode

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
                        if not recover_exact_branch_errors:
                            raise
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
                            "recovery": "restore-policy-and-execute-wait",
                            "parent_state_unchanged": True,
                            "probe_mode": planner.probe_mode,
                            "probe_ticks": planner.current_probe_ticks,
                        }
                        record_probe(
                            planner.probe_mode, planner.current_probe_ticks
                        )
                        branch_error_events.append(event)
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
                                    "probe_mode": planner.probe_mode,
                                    "probe_ticks": planner.current_probe_ticks,
                                    "exact_branch_error": event,
                                }
                            )
                    else:
                        record_probe(verdict.probe_mode, verdict.probe_ticks)
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
        "planner_probe_mode_switches": probe_mode_switches,
        "planner_probe_schedule": probe_schedule,
        "planner_probe_schedule_sha256": canonical_sha256(probe_schedule),
        "selected_categories": dict(sorted(selected.items())),
        "selection_reasons": dict(sorted(reasons.items())),
        "logged_query_count": len(query_log),
        "query_log": query_log,
        "query_log_sha256": canonical_sha256(query_log),
        "trace": trace,
        "trace_sha256": canonical_sha256(trace),
        "wall_seconds": time.monotonic() - started,
    }
    if recover_exact_branch_errors:
        result["exact_branch_errors"] = len(branch_error_events)
        result["exact_branch_error_events"] = branch_error_events
        result["exact_branch_error_events_sha256"] = canonical_sha256(
            branch_error_events
        )
    return result, provenance


def evaluate_bundle(
    runtime: ExactTrainingRuntime,
    bundle: PolicyBundle,
    *,
    declared_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    maximum_ticks: int = FULL_GAME_TICKS,
    planner_config: FastMultiActionConfig | None = None,
    target_score: int = DEFAULT_TARGET_SCORE,
    trace_interval_ticks: int = 10_000,
    maximum_logged_queries: int = 128,
    episode_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    declared, evaluation = validate_contract(
        bundle, declared_training_seeds, evaluation_seeds
    )
    config = FastMultiActionConfig() if planner_config is None else planner_config
    for name, value in (
        ("maximum_ticks", maximum_ticks),
        ("trace_interval_ticks", trace_interval_ticks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(maximum_logged_queries, bool)
        or not isinstance(maximum_logged_queries, int)
        or maximum_logged_queries < 0
    ):
        raise ValueError("maximum_logged_queries must be nonnegative")
    if target_score < 0:
        raise ValueError("target_score must be nonnegative")
    started = time.monotonic()
    episodes: list[dict[str, object]] = []
    runtime_provenance: Mapping[str, Any] | None = None
    runtime_provenance_sha256: str | None = None
    for seed in evaluation:
        episode, current_provenance = run_episode(
            runtime,
            bundle,
            seed,
            maximum_ticks=maximum_ticks,
            planner_config=config,
            target_score=target_score,
            trace_interval_ticks=trace_interval_ticks,
            maximum_logged_queries=maximum_logged_queries,
        )
        current_sha = canonical_sha256(current_provenance)
        if (
            runtime_provenance_sha256 is not None
            and runtime_provenance_sha256 != current_sha
        ):
            raise RuntimeError("exact runtime provenance changed between episodes")
        runtime_provenance = current_provenance
        runtime_provenance_sha256 = current_sha
        episodes.append(episode)
        if episode_callback is not None:
            episode_callback(episode)
    scores = [int(value["score"]) for value in episodes]
    source = Path(__file__).resolve()
    planner_source = ROOT / "python/irisu_pointer/fast_multiaction_planner.py"
    reserve_source = ROOT / "python/irisu_pointer/development_reserve_band.py"
    successes = sum(bool(value["success"]) for value in episodes)
    aggregate_probe_modes: Counter[str] = Counter()
    aggregate_probe_horizons: Counter[str] = Counter()
    for episode in episodes:
        aggregate_probe_modes.update(episode["planner_probe_mode_counts"])
        aggregate_probe_horizons.update(episode["planner_probe_horizon_counts"])
    report: dict[str, object] = {
        "format": FORMAT,
        "physics_backend": "exact",
        "evidence_class": "development-only",
        "promotion_eligible": False,
        "deterministic_policy": True,
        "training_seeds": list(declared),
        "training_seeds_sha256": canonical_sha256(list(declared)),
        "evaluation_seeds": list(evaluation),
        "evaluation_seeds_sha256": canonical_sha256(list(evaluation)),
        "training_evaluation_overlap": [],
        "maximum_ticks": maximum_ticks,
        "target_score": target_score,
        "planner_config": config.manifest(),
        "planner_objective": (
            "liability-adjusted-reserve-band-survival-renewal-score-v1"
            if config.objective_mode == "reserve-band"
            else "wait-relative-reserve-filter-then-survival-clears-score-gauge-v1"
        ),
        "planner_source_sha256": file_sha256(planner_source),
        "reserve_comparator_source_sha256": (
            file_sha256(reserve_source)
            if config.objective_mode == "reserve-band"
            else None
        ),
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_sha256": bundle.model_sha256,
        "checkpoint_metadata": dict(bundle.metadata),
        "checkpoint_metadata_sha256": canonical_sha256(bundle.metadata),
        "inference_config": dict(bundle.inference_config),
        "exact_runtime": dict(runtime_provenance or {}),
        "runtime_hashes": {
            "worker_sha256": runtime.identity.worker_sha256,
            "exact_library_sha256": runtime.identity.exact_library_sha256,
            "identity_config_sha256": runtime.identity.config_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
        },
        "runner_sha256": file_sha256(source),
        "episodes": episodes,
        "episode_count": len(episodes),
        "scores": scores,
        "median_score": float(statistics.median(scores)),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(episodes),
        "invalid_actions": sum(int(value["invalid_actions"]) for value in episodes),
        "counterfactual_invalid_actions": sum(
            int(value["counterfactual_invalid_actions"]) for value in episodes
        ),
        "planner_probe_mode_counts": dict(sorted(aggregate_probe_modes.items())),
        "planner_probe_horizon_counts": dict(
            sorted(aggregate_probe_horizons.items())
        ),
        "planner_probe_mode_switch_count": sum(
            len(value["planner_probe_mode_switches"]) for value in episodes
        ),
        "wall_seconds": time.monotonic() - started,
    }
    bind_report_content_sha256(report)
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
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--training-seed-manifest", type=Path, required=True)
    result.add_argument("--seeds", type=parse_seeds, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--maximum-ticks", type=int, default=FULL_GAME_TICKS)
    result.add_argument("--probe-ticks", type=int, default=256)
    result.add_argument("--long-probe-ticks", type=int, default=512)
    result.add_argument("--wait-ticks", type=int, default=16)
    result.add_argument("--low-gauge-threshold", type=int, default=20_000)
    result.add_argument("--low-gauge-exit-threshold", type=int, default=30_000)
    result.add_argument("--top-k-pairs", type=int, default=2)
    result.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    result.add_argument("--rescue-score-margin", type=int, default=500)
    result.add_argument("--gauge-advantage", type=int, default=1)
    result.add_argument(
        "--objective-mode",
        choices=("wait-relative", "reserve-band"),
        default="wait-relative",
    )
    result.add_argument("--reserve-contingency-gauge", type=int, default=0)
    result.add_argument("--rot-delay-ticks", type=int, default=40)
    result.add_argument("--target-score", type=int, default=DEFAULT_TARGET_SCORE)
    result.add_argument("--trace-interval-ticks", type=int, default=10_000)
    result.add_argument("--maximum-logged-queries", type=int, default=128)
    result.add_argument("--act-logit-bias", type=float)
    result.add_argument("--cooldown-ticks", type=int)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.worker.is_absolute():
        parser().error("--worker must be absolute")
    if args.output.exists():
        parser().error("--output must not already exist")
    inference_options = {
        key: value
        for key, value in {
            "act_logit_bias": args.act_logit_bias,
            "cooldown_ticks": args.cooldown_ticks,
        }.items()
        if value is not None
    }
    bundle = load_policy_bundle(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        inference_options=inference_options,
    )
    training = load_declared_training_seeds(args.training_seed_manifest)
    runtime = ExactTrainingRuntime(args.worker)
    report = evaluate_bundle(
        runtime,
        bundle,
        declared_training_seeds=training,
        evaluation_seeds=args.seeds,
        maximum_ticks=args.maximum_ticks,
        planner_config=FastMultiActionConfig(
            probe_ticks=args.probe_ticks,
            long_probe_ticks=args.long_probe_ticks,
            wait_ticks=args.wait_ticks,
            low_gauge_threshold=args.low_gauge_threshold,
            low_gauge_exit_threshold=args.low_gauge_exit_threshold,
            top_k_pairs=args.top_k_pairs,
            maximum_gauge_debt=args.maximum_gauge_debt,
            rescue_score_margin=args.rescue_score_margin,
            gauge_advantage=args.gauge_advantage,
            objective_mode=args.objective_mode,
            reserve_contingency_gauge=args.reserve_contingency_gauge,
            rot_delay_ticks=args.rot_delay_ticks,
        ),
        target_score=args.target_score,
        trace_interval_ticks=args.trace_interval_ticks,
        maximum_logged_queries=args.maximum_logged_queries,
        episode_callback=lambda value: print(
            json.dumps(
                {
                    "seed": value["seed"],
                    "score": value["score"],
                    "tick": value["tick"],
                    "gauge": value["gauge"],
                    "selected_categories": value["selected_categories"],
                },
                sort_keys=True,
            ),
            flush=True,
        ),
    )
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
