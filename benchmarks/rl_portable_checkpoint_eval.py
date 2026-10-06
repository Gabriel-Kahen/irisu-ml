#!/usr/bin/env python3
"""Development-only portable evaluation for learned steering checkpoints.

Seeds are always caller supplied.  This runner is diagnostic evidence only; it
cannot load or materialize a locked evaluation split.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from irisu_env import Action, ActionKind, IrisuEnv  # noqa: E402
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


PORTABLE_RUNTIME = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/"
    "libirisu_clone.so"
)
FULL_GAME_TICKS = 100_000
MODES = ("model-only", "adaptive-wait-gate")


class ReserveWaitGate(ExactWaitDominanceGate):
    """Wait gate that treats excess gauge spend as a dominated shot."""

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
        shot, wait = verdict.shot, verdict.wait
        execute, reason = reserve_choice(
            shot,
            wait,
            maximum_gauge_debt=self.maximum_gauge_debt,
            rescue_score_margin=self.rescue_score_margin,
        )
        return GateVerdict(
            execute, reason, shot, wait, verdict.restore_checks
        )


def reserve_choice(
    shot: ProbeOutcome,
    wait: ProbeOutcome,
    *,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[bool, str]:
    if shot.survival_ticks != wait.survival_ticks:
        return (
            shot.survival_ticks > wait.survival_ticks,
            "shot-survival"
            if shot.survival_ticks > wait.survival_ticks
            else "wait-survival",
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


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip(), 0) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated uint32 values") from exc
    if (
        not seeds
        or any(not 0 <= seed <= 0xFFFF_FFFF for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise argparse.ArgumentTypeError("seeds must be unique comma-separated uint32 values")
    return seeds


def primitive_actions(decision: object) -> tuple[Action, ...]:
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


def _step_decision(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    decision: object,
    maximum_ticks: int,
) -> tuple[dict[str, Any], bool, bool, int]:
    current = dict(observation)
    terminated = truncated = False
    invalid_actions = 0
    for action in primitive_actions(decision):
        kind = ActionKind(int(action.kind))
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        duration = min(duration, maximum_ticks - int(current["tick"]))
        for _ in range(max(0, duration)):
            primitive = Action.wait(1) if kind is ActionKind.WAIT else action
            current, _reward, terminated, truncated, info = env.step(primitive)
            invalid_actions += int(bool(info.get("invalid_action", False)))
            if terminated or truncated:
                break
        if terminated or truncated or int(current["tick"]) >= maximum_ticks:
            break
    return current, terminated, truncated, invalid_actions


def run_episode(
    seed: int,
    *,
    mode: str,
    policy_factory: Callable[[], GoalConditionedSteeringPolicy],
    runtime: Path,
    maximum_ticks: int,
    probe_ticks: int,
    rescue_probe_ticks: int,
    rescue_gauge_threshold: int,
    wait_ticks: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    policy = policy_factory()
    policy.reset(seed)
    gate = make_gate(
        probe_ticks,
        wait_ticks=wait_ticks,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    rescue_gate = make_gate(
        rescue_probe_ticks,
        wait_ticks=wait_ticks,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    attempted = kept = suppressed = rescue_queries = invalid_actions = 0
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    terminated = truncated = False
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        # Keep the simulator's own truncation beyond the diagnostic horizon so
        # reaching ``maximum_ticks`` is not misreported as an environment fault.
        config={"max_episode_ticks": maximum_ticks + rescue_probe_ticks},
    ) as env:
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("portable reset seed differs")
        while int(observation["tick"]) < maximum_ticks and not (
            terminated or truncated
        ):
            before = copy.deepcopy(policy) if mode == "adaptive-wait-gate" else None
            decision = policy.predict(observation)
            if mode == "adaptive-wait-gate" and bool(getattr(decision, "is_shot", False)):
                attempted += 1
                active_gate = gate
                if int(observation["gauge"]) <= rescue_gauge_threshold:
                    active_gate = rescue_gate
                    rescue_queries += 1
                verdict = active_gate.evaluate(env, observation, before, policy, decision)
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = active_gate.wait_decision(verdict.reason)
            observation, terminated, truncated, invalid = _step_decision(
                env, observation, decision, maximum_ticks
            )
            invalid_actions += invalid
    return {
        "seed": seed,
        "mode": mode,
        "score": int(observation["score"]),
        "tick": int(observation["tick"]),
        "level": int(observation["level"]),
        "gauge": int(observation["gauge"]),
        "highest_chain": int(observation["highest_chain"]),
        "terminated": bool(observation.get("terminated", terminated)),
        "truncated": bool(truncated),
        "invalid_actions": invalid_actions,
        "attempted_shots": attempted,
        "kept_shots": kept,
        "suppressed_shots": suppressed,
        "rescue_queries": rescue_queries,
        "gate_reasons": dict(reasons),
        "wall_seconds": time.monotonic() - started,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    scores = [int(row["score"]) for row in rows]
    return {
        "episodes": len(rows),
        "scores": scores,
        "median_score": statistics.median(scores),
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "scores_at_least_50000": sum(score >= 50_000 for score in scores),
        "success_fraction": sum(score >= 50_000 for score in scores) / len(scores),
        "median_survival_ticks": statistics.median(int(row["tick"]) for row in rows),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expected-checkpoint-sha256")
    result.add_argument("--seeds", type=parse_seeds, required=True)
    result.add_argument("--mode", choices=(*MODES, "both"), default="both")
    result.add_argument("--runtime", type=Path, default=PORTABLE_RUNTIME)
    result.add_argument("--maximum-ticks", type=int, default=FULL_GAME_TICKS)
    result.add_argument("--probe-ticks", type=int, default=128)
    result.add_argument("--rescue-probe-ticks", type=int, default=256)
    result.add_argument("--rescue-gauge-threshold", type=int, default=30_000)
    result.add_argument("--wait-ticks", type=int, default=16)
    result.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    result.add_argument("--rescue-score-margin", type=int, default=500)
    result.add_argument("--cooldown-ticks", type=int, default=16)
    result.add_argument("--act-logit-bias", type=float, default=1.0)
    result.add_argument(
        "--use-kind-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt into learned WEAK/STRONG selection (legacy default: STRONG)",
    )
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = (
        "maximum_ticks",
        "probe_ticks",
        "rescue_probe_ticks",
        "wait_ticks",
        "maximum_gauge_debt",
        "rescue_score_margin",
        "cooldown_ticks",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser().error("tick counts, margins, and cooldown must be positive")
    if args.rescue_gauge_threshold < 0:
        parser().error("rescue gauge threshold must be nonnegative")
    runtime = args.runtime.resolve(strict=True)
    checkpoint = load_steering_checkpoint(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        device="cpu",
    )
    checkpoint.model.eval()
    inference = {
        "cooldown_ticks": args.cooldown_ticks,
        "minimum_pair_closure_sizes": 0.05,
        "impact_side_sizes": 0.5,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 50.0,
        "act_logit_bias": args.act_logit_bias,
        "use_kind_head": args.use_kind_head,
    }

    def policy_factory() -> GoalConditionedSteeringPolicy:
        return GoalConditionedSteeringPolicy(
            checkpoint.model,
            **inference,
            artifact_sha256=checkpoint.sha256,
        )

    modes = MODES if args.mode == "both" else (args.mode,)
    episodes = []
    for mode in modes:
        for seed in args.seeds:
            row = run_episode(
                seed,
                mode=mode,
                policy_factory=policy_factory,
                runtime=runtime,
                maximum_ticks=args.maximum_ticks,
                probe_ticks=args.probe_ticks,
                rescue_probe_ticks=args.rescue_probe_ticks,
                rescue_gauge_threshold=args.rescue_gauge_threshold,
                wait_ticks=args.wait_ticks,
                maximum_gauge_debt=args.maximum_gauge_debt,
                rescue_score_margin=args.rescue_score_margin,
            )
            episodes.append(row)
            print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    report = {
        "schema": "irisu-portable-checkpoint-development-eval-v1",
        "development_only": True,
        "promotion_eligible": False,
        "deterministic_policy": True,
        "physics_backend": "portable",
        "runtime": str(runtime),
        "checkpoint": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_metadata": dict(checkpoint.metadata),
        "seeds": list(args.seeds),
        "maximum_ticks": args.maximum_ticks,
        "inference": inference,
        "adaptive_wait_gate": {
            "probe_ticks": args.probe_ticks,
            "rescue_probe_ticks": args.rescue_probe_ticks,
            "rescue_gauge_threshold": args.rescue_gauge_threshold,
            "wait_ticks": args.wait_ticks,
            "maximum_gauge_debt": args.maximum_gauge_debt,
            "rescue_score_margin": args.rescue_score_margin,
        },
        "episodes": episodes,
        "summary_by_mode": {
            mode: summarize([row for row in episodes if row["mode"] == mode])
            for mode in modes
        },
    }
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
