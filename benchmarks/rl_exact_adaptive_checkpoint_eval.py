#!/usr/bin/env python3
"""Exact multi-seed evaluation of a learned steering policy plus wait planner.

Unlike the locked policy-only gate, this runner contains no seed suite.  Every
evaluation seed is supplied by the caller, which also makes one-seed and
sharded development runs possible without exposing promotion seeds to code.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from irisu_env import Action, ActionKind, ExactWorkerError  # noqa: E402
from irisu_pointer.shot_necessity import (  # noqa: E402
    ExactWaitDominanceGate,
    GateVerdict,
    ProbeOutcome,
    WaitDominanceConfig,
    choose_shot,
)
from irisu_pointer.steering_checkpoint import load_steering_checkpoint  # noqa: E402
from irisu_pointer.steering_learning import (  # noqa: E402
    GoalConditionedSteeringPolicy,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


FORMAT = "irisu-exact-adaptive-learned-planner-eval-v1"
DEFAULT_EPISODES = 20
DEFAULT_SUCCESSES = 16
DEFAULT_TARGET_SCORE = 50_000
FULL_GAME_TICKS = 120_000


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_report_content_sha256(report: dict[str, object]) -> None:
    """Bind a report after removing any stale self-hash."""

    report.pop("report_content_sha256", None)
    report["report_content_sha256"] = canonical_sha256(report)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip(), 0) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated uint32 values"
        ) from exc
    try:
        return validate_seed_sequence(seeds, "evaluation")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def validate_seed_sequence(values: Sequence[int], label: str) -> tuple[int, ...]:
    seeds = tuple(values)
    if (
        not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(not 0 <= seed <= 0xFFFF_FFFF for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{label} seeds must be unique uint32 values")
    return seeds


def load_declared_training_seeds(path: Path) -> tuple[int, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        raw = value
    elif isinstance(value, dict) and isinstance(value.get("training_seeds"), list):
        raw = value["training_seeds"]
    else:
        raise ValueError(
            "training seed manifest must be a JSON list or contain training_seeds"
        )
    return validate_seed_sequence(raw, "declared training")


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0" + value.numpy().tobytes())
    return digest.hexdigest()


class Policy(Protocol):
    def reset(self, seed: int = 0) -> None: ...

    def predict(self, observation: Mapping[str, Any]) -> Any: ...


def _copy_goal_policy(
    policy: GoalConditionedSteeringPolicy,
) -> GoalConditionedSteeringPolicy:
    return copy.deepcopy(policy, {id(policy.model): policy.model})


class SharedModelPolicy:
    """Copy controller state while sharing immutable inference weights."""

    def __init__(self, inner: GoalConditionedSteeringPolicy) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __deepcopy__(self, memo: dict[int, object]) -> "SharedModelPolicy":
        copied = SharedModelPolicy(_copy_goal_policy(self.inner))
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    checkpoint_path: str
    checkpoint_sha256: str
    model_sha256: str
    metadata: Mapping[str, Any]
    inference_config: Mapping[str, Any]
    factory: Callable[[], Policy]


def checkpoint_training_seeds(bundle: PolicyBundle) -> tuple[int, ...]:
    raw = bundle.metadata.get("training_seeds")
    if not isinstance(raw, list):
        raise ValueError("checkpoint must declare its complete training_seeds")
    return tuple(sorted(validate_seed_sequence(raw, "checkpoint training")))


def load_policy_bundle(
    path: Path,
    *,
    expected_sha256: str,
    inference_options: Mapping[str, Any] | None = None,
) -> PolicyBundle:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected checkpoint SHA-256 must be lowercase hexadecimal")
    checkpoint = load_steering_checkpoint(
        path.resolve(strict=True), expected_sha256=expected_sha256, device="cpu"
    )
    checkpoint.model.eval()
    recorded = checkpoint.metadata.get("inference_config", {})
    if not isinstance(recorded, dict):
        raise ValueError("checkpoint inference_config must be a mapping")
    supplied = {} if inference_options is None else dict(inference_options)
    for name, value in supplied.items():
        if name in recorded and value != recorded[name]:
            raise ValueError(f"inference option {name} differs from checkpoint metadata")
    defaults: dict[str, int | float] = {
        "cooldown_ticks": 16,
        "minimum_pair_closure_sizes": 0.05,
        "impact_side_sizes": 0.5,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 50.0,
        "act_logit_bias": 0.0,
    }
    resolved = defaults | recorded | supplied
    config: dict[str, int | float] = {
        "cooldown_ticks": int(resolved["cooldown_ticks"]),
        "minimum_pair_closure_sizes": float(resolved["minimum_pair_closure_sizes"]),
        "impact_side_sizes": float(resolved["impact_side_sizes"]),
        "impact_below_sizes": float(resolved["impact_below_sizes"]),
        "source_velocity_lead_ticks": float(resolved["source_velocity_lead_ticks"]),
        "ticks_per_second": float(resolved["ticks_per_second"]),
        "act_logit_bias": float(resolved["act_logit_bias"]),
    }

    def factory() -> SharedModelPolicy:
        return SharedModelPolicy(
            GoalConditionedSteeringPolicy(
                checkpoint.model,
                **config,
                artifact_sha256=checkpoint.sha256,
            )
        )

    return PolicyBundle(
        checkpoint_path=str(checkpoint.path),
        checkpoint_sha256=checkpoint.sha256,
        model_sha256=state_dict_sha256(checkpoint.model),
        metadata=dict(checkpoint.metadata),
        inference_config=config,
        factory=factory,
    )


def reserve_choice(
    shot: ProbeOutcome,
    wait: ProbeOutcome,
    *,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[bool, str]:
    """The reserve-aware objective used by rl_portable_checkpoint_eval."""

    if shot.survival_ticks != wait.survival_ticks:
        return (
            shot.survival_ticks > wait.survival_ticks,
            "shot-survival" if shot.survival_ticks > wait.survival_ticks else "wait-survival",
        )
    shot_failed = shot.terminated or shot.truncated
    wait_failed = wait.terminated or wait.truncated
    if shot_failed != wait_failed:
        return not shot_failed, "shot-rescue" if wait_failed else "wait-safer"
    gauge_debt = wait.final_gauge - shot.final_gauge
    score_gain = shot.score - wait.score
    if gauge_debt > maximum_gauge_debt and score_gain < rescue_score_margin:
        return False, "wait-reserve"
    return choose_shot(shot, wait, gauge_advantage=1)


class ReserveWaitGate(ExactWaitDominanceGate):
    def __init__(
        self,
        *args: object,
        maximum_gauge_debt: int,
        rescue_score_margin: int,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_gauge_debt = maximum_gauge_debt
        self.rescue_score_margin = rescue_score_margin

    def evaluate(self, *args: object, **kwargs: object) -> GateVerdict:
        verdict = super().evaluate(*args, **kwargs)
        execute, reason = reserve_choice(
            verdict.shot,
            verdict.wait,
            maximum_gauge_debt=self.maximum_gauge_debt,
            rescue_score_margin=self.rescue_score_margin,
        )
        return GateVerdict(
            execute,
            reason,
            verdict.shot,
            verdict.wait,
            verdict.restore_checks,
        )


def primitive_actions(decision: object) -> tuple[object, ...]:
    method = getattr(decision, "primitive_actions", None)
    if not callable(method):
        raise TypeError("policy decision does not expose primitive_actions")
    actions = tuple(method())
    if not actions:
        raise ValueError("policy decision produced no primitive actions")
    return actions


def make_gate(
    horizon: int,
    *,
    wait_ticks: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> ReserveWaitGate:
    return ReserveWaitGate(
        primitive_actions,
        config=WaitDominanceConfig(
            probe_ticks=horizon,
            wait_ticks=wait_ticks,
            gauge_advantage=16,
        ),
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )


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
    short_horizon: int,
    long_horizon: int,
    gauge_threshold: int,
    wait_ticks: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
    target_score: int,
    trace_interval_ticks: int,
    recover_exact_branch_errors: bool = False,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    policy = bundle.factory()
    policy.reset(seed)
    short_gate = make_gate(
        short_horizon,
        wait_ticks=wait_ticks,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    long_gate = make_gate(
        long_horizon,
        wait_ticks=wait_ticks,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    attempted = kept = suppressed = long_queries = 0
    policy_invalid = simulator_invalid = action_count = restore_checks = 0
    reasons: Counter[str] = Counter()
    branch_error_events: list[dict[str, object]] = []
    terminated = truncated = False
    final_info: Mapping[str, Any] = {}
    started = time.monotonic()
    simulation_config = {"max_episode_ticks": maximum_ticks + long_horizon}
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
            before = copy.deepcopy(policy)
            try:
                decision = policy.predict(observation)
                if bool(getattr(decision, "is_shot", False)):
                    attempted += 1
                    gate = short_gate
                    if int(observation["gauge"]) <= gauge_threshold:
                        gate = long_gate
                        long_queries += 1
                    parent_hash_before = int(env.state_hash())
                    try:
                        verdict = gate.evaluate(
                            env, observation, before, policy, decision
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
                        branch_error_events.append(
                            {
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
                            }
                        )
                        reason = "wait-exact-branch-error"
                        reasons[reason] += 1
                        suppressed += 1
                        policy = before
                        decision = gate.wait_decision(reason)
                    else:
                        restore_checks += verdict.restore_checks
                        reasons[verdict.reason] += 1
                        if verdict.execute_shot:
                            kept += 1
                        else:
                            suppressed += 1
                            policy = before
                            decision = gate.wait_decision(verdict.reason)
                actions = primitive_actions(decision)
            except (AttributeError, TypeError, ValueError, OverflowError):
                policy_invalid += 1
                policy = before
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
                    simulator_invalid += int(bool(final_info.get("invalid_action", False)))
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
    invalid = policy_invalid + simulator_invalid
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
        "censored": not bool(terminated or observation.get("terminated", False)),
        "policy_invalid_actions": policy_invalid,
        "simulator_invalid_actions": simulator_invalid,
        "invalid_actions": invalid,
        "action_count": action_count,
        "attempted_shots": attempted,
        "kept_shots": kept,
        "suppressed_shots": suppressed,
        "long_horizon_queries": long_queries,
        "gate_restore_checks": restore_checks,
        "gate_reasons": dict(sorted(reasons.items())),
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


def validate_contract(
    bundle: PolicyBundle,
    declared_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    declared = tuple(sorted(validate_seed_sequence(declared_training_seeds, "declared training")))
    checkpoint_bound = checkpoint_training_seeds(bundle)
    if declared != checkpoint_bound:
        raise ValueError(
            "declared training seeds differ from checkpoint-bound training seeds"
        )
    evaluation = validate_seed_sequence(evaluation_seeds, "evaluation")
    overlap = sorted(set(declared) & set(evaluation))
    if overlap:
        raise ValueError(f"evaluation seeds overlap checkpoint training seeds: {overlap}")
    return declared, evaluation


def evaluate_bundle(
    runtime: ExactTrainingRuntime,
    bundle: PolicyBundle,
    *,
    declared_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    maximum_ticks: int = FULL_GAME_TICKS,
    short_horizon: int = 128,
    long_horizon: int = 256,
    gauge_threshold: int = 30_000,
    wait_ticks: int = 16,
    maximum_gauge_debt: int = 1_000,
    rescue_score_margin: int = 500,
    target_score: int = DEFAULT_TARGET_SCORE,
    required_episode_count: int = DEFAULT_EPISODES,
    required_success_count: int = DEFAULT_SUCCESSES,
    trace_interval_ticks: int = 10_000,
    episode_callback: Callable[[Mapping[str, object]], None] | None = None,
    recover_exact_branch_errors: bool = False,
    report_format: str = FORMAT,
    runner_path: Path | None = None,
) -> dict[str, object]:
    declared, evaluation = validate_contract(
        bundle, declared_training_seeds, evaluation_seeds
    )
    positive = {
        "maximum_ticks": maximum_ticks,
        "short_horizon": short_horizon,
        "long_horizon": long_horizon,
        "wait_ticks": wait_ticks,
        "maximum_gauge_debt": maximum_gauge_debt,
        "rescue_score_margin": rescue_score_margin,
        "required_episode_count": required_episode_count,
        "required_success_count": required_success_count,
        "trace_interval_ticks": trace_interval_ticks,
    }
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in positive.values()):
        raise ValueError("tick counts, margins, and promotion counts must be positive")
    if long_horizon < short_horizon:
        raise ValueError("long_horizon must be at least short_horizon")
    if gauge_threshold < 0 or target_score < 0:
        raise ValueError("gauge threshold and target score must be nonnegative")
    if required_success_count > required_episode_count:
        raise ValueError("required successes cannot exceed required episodes")

    started = time.monotonic()
    results: list[dict[str, object]] = []
    provenance: Mapping[str, Any] | None = None
    provenance_hash: str | None = None
    for seed in evaluation:
        result, episode_provenance = run_episode(
            runtime,
            bundle,
            seed,
            maximum_ticks=maximum_ticks,
            short_horizon=short_horizon,
            long_horizon=long_horizon,
            gauge_threshold=gauge_threshold,
            wait_ticks=wait_ticks,
            maximum_gauge_debt=maximum_gauge_debt,
            rescue_score_margin=rescue_score_margin,
            target_score=target_score,
            trace_interval_ticks=trace_interval_ticks,
            recover_exact_branch_errors=recover_exact_branch_errors,
        )
        current_hash = canonical_sha256(episode_provenance)
        if provenance_hash is not None and current_hash != provenance_hash:
            raise RuntimeError("exact runtime provenance changed between episodes")
        provenance = episode_provenance
        provenance_hash = current_hash
        results.append(result)
        if episode_callback is not None:
            episode_callback(result)

    scores = [int(row["score"]) for row in results]
    successes = sum(bool(row["success"]) for row in results)
    invalid = sum(int(row["invalid_actions"]) for row in results)
    median_score = float(statistics.median(scores))
    contract = {
        "required_episode_count": required_episode_count,
        "required_success_count": required_success_count,
        "target_score": target_score,
        "require_median_at_or_above_target": True,
        "require_zero_invalid_actions": True,
    }
    bound_runner_path = (
        Path(__file__).resolve()
        if runner_path is None
        else runner_path.resolve(strict=True)
    )
    planner_config: dict[str, object] = {
        "short_horizon": short_horizon,
        "long_horizon": long_horizon,
        "gauge_threshold": gauge_threshold,
        "wait_ticks": wait_ticks,
        "maximum_gauge_debt": maximum_gauge_debt,
        "rescue_score_margin": rescue_score_margin,
        "objective": "reserve-aware-wait-dominance-v1",
    }
    if recover_exact_branch_errors:
        planner_config["exact_branch_error_policy"] = {
            "version": "exact-branch-error-conservative-wait-v1",
            "caught_exception": "irisu_env.exact_ipc.ExactWorkerError",
            "action": "restore-policy-and-execute-wait",
            "require_live_parent_state_hash_unchanged": True,
        }
    report: dict[str, object] = {
        "format": report_format,
        "physics_backend": "exact",
        "deterministic_policy": True,
        "promotion_contract": contract,
        "promotion_eligible": len(evaluation) == required_episode_count,
        "training_seeds": list(declared),
        "training_seeds_sha256": canonical_sha256(list(declared)),
        "evaluation_seeds": list(evaluation),
        "evaluation_seeds_sha256": canonical_sha256(list(evaluation)),
        "training_evaluation_overlap": [],
        "maximum_ticks": maximum_ticks,
        "planner_config": planner_config,
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_sha256": bundle.model_sha256,
        "checkpoint_metadata_sha256": canonical_sha256(bundle.metadata),
        "checkpoint_metadata": dict(bundle.metadata),
        "inference_config": dict(bundle.inference_config),
        "exact_runtime": dict(provenance or {}),
        "runtime_hashes": {
            "worker_sha256": runtime.identity.worker_sha256,
            "exact_library_sha256": runtime.identity.exact_library_sha256,
            "identity_config_sha256": runtime.identity.config_sha256,
            "runtime_provenance_sha256": provenance_hash,
        },
        "runner_sha256": file_sha256(bound_runner_path),
        "episodes": results,
        "episode_count": len(results),
        "scores": scores,
        "median_score": median_score,
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(results),
        "invalid_actions": invalid,
        "passed": len(results) == required_episode_count
        and median_score >= target_score
        and successes >= required_success_count
        and invalid == 0,
        "wall_seconds": time.monotonic() - started,
    }
    if recover_exact_branch_errors:
        report["exact_branch_errors"] = sum(
            int(row["exact_branch_errors"]) for row in results
        )
        report["branch_error_recovery_enabled"] = True
        report["evaluator_engine_sha256"] = file_sha256(
            Path(__file__).resolve()
        )
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
    result.add_argument("--episode-dir", type=Path)
    result.add_argument("--maximum-ticks", type=int, default=FULL_GAME_TICKS)
    result.add_argument("--short-horizon", type=int, default=128)
    result.add_argument("--long-horizon", type=int, default=256)
    result.add_argument("--gauge-threshold", type=int, default=30_000)
    result.add_argument("--wait-ticks", type=int, default=16)
    result.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    result.add_argument("--rescue-score-margin", type=int, default=500)
    result.add_argument("--target-score", type=int, default=DEFAULT_TARGET_SCORE)
    result.add_argument("--required-episode-count", type=int, default=DEFAULT_EPISODES)
    result.add_argument("--required-success-count", type=int, default=DEFAULT_SUCCESSES)
    result.add_argument("--trace-interval-ticks", type=int, default=10_000)
    result.add_argument("--act-logit-bias", type=float)
    result.add_argument("--cooldown-ticks", type=int)
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    report_format: str = FORMAT,
    runner_path: Path | None = None,
    recover_exact_branch_errors: bool = False,
) -> int:
    args = parser().parse_args(argv)
    if not args.worker.is_absolute():
        parser().error("--worker must be absolute")
    if args.output.exists():
        parser().error("--output must not already exist")
    if args.episode_dir is not None and args.episode_dir.exists():
        parser().error("--episode-dir must not already exist")
    inference_options = {
        name: value
        for name, value in {
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
    training = load_declared_training_seeds(
        args.training_seed_manifest.resolve(strict=True)
    )
    if args.episode_dir is not None:
        args.episode_dir.mkdir(parents=True)

    episode_artifacts: list[dict[str, object]] = []

    def record_episode(row: Mapping[str, object]) -> None:
        print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
        if args.episode_dir is not None:
            path = (
                args.episode_dir / f"seed-{int(row['seed']):010d}.json"
            ).resolve()
            _write_new(path, row)
            episode_artifacts.append(
                {
                    "seed": int(row["seed"]),
                    "path": str(path),
                    "file_sha256": file_sha256(path),
                    "content_sha256": canonical_sha256(row),
                }
            )

    report = evaluate_bundle(
        ExactTrainingRuntime(args.worker),
        bundle,
        declared_training_seeds=training,
        evaluation_seeds=args.seeds,
        maximum_ticks=args.maximum_ticks,
        short_horizon=args.short_horizon,
        long_horizon=args.long_horizon,
        gauge_threshold=args.gauge_threshold,
        wait_ticks=args.wait_ticks,
        maximum_gauge_debt=args.maximum_gauge_debt,
        rescue_score_margin=args.rescue_score_margin,
        target_score=args.target_score,
        required_episode_count=args.required_episode_count,
        required_success_count=args.required_success_count,
        trace_interval_ticks=args.trace_interval_ticks,
        episode_callback=record_episode,
        recover_exact_branch_errors=recover_exact_branch_errors,
        report_format=report_format,
        runner_path=runner_path,
    )
    if episode_artifacts:
        report["episode_artifacts"] = episode_artifacts
        bind_report_content_sha256(report)
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
