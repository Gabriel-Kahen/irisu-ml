#!/usr/bin/env python3
"""Development-only portable probe for stronger multi-seed shot teachers.

This deliberately cannot emit promotion evidence.  It reuses the frozen-v5
directed-pair policy and compares its proposed shot with one wait, continuing
both branches closed-loop for a configurable horizon.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3k_sustainable_v3 as frozen  # noqa: E402
from irisu_pointer.shot_necessity import (  # noqa: E402
    ExactWaitDominanceGate,
    GateVerdict,
    ProbeOutcome,
    WaitDominanceConfig,
    choose_shot,
)


DEFAULT_SEEDS = (
    3_939_967_453,
    1_807_785_371,
    1_236_345_145,
    2_273_442_848,
    3_998_699_142,
    3_818_226_755,
    136_083_672,
    3_808_206_320,
    3_576_545_428,
    3_293_020_940,
)


@dataclass(frozen=True, slots=True)
class ReserveObjective:
    """Reject short-term gains which spend substantially more gauge reserve."""

    maximum_gauge_debt: int = 1_000
    rescue_score_margin: int = 500

    def choose(self, shot: ProbeOutcome, wait: ProbeOutcome) -> tuple[bool, str]:
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
            return (
                not shot_failed,
                "shot-rescue" if wait_failed else "wait-safer",
            )
        gauge_debt = wait.final_gauge - shot.final_gauge
        score_gain = shot.score - wait.score
        if (
            gauge_debt > self.maximum_gauge_debt
            and score_gain < self.rescue_score_margin
        ):
            return False, "wait-reserve"
        return choose_shot(shot, wait, gauge_advantage=1)


class ObjectiveGate(ExactWaitDominanceGate):
    def __init__(self, *args: object, objective: ReserveObjective | None, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.objective = objective

    def evaluate(self, *args: object, **kwargs: object) -> GateVerdict:
        verdict = super().evaluate(*args, **kwargs)
        if self.objective is None:
            return verdict
        execute, reason = self.objective.choose(verdict.shot, verdict.wait)
        return GateVerdict(
            execute, reason, verdict.shot, verdict.wait, verdict.restore_checks
        )


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip(), 0) for item in value.split(",") if item.strip())
    if not seeds or any(not 0 <= seed <= 0xFFFF_FFFF for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be comma-separated uint32 values")
    return seeds


def run_episode(
    seed: int,
    *,
    probe_ticks: int,
    rescue_probe_ticks: int | None,
    rescue_gauge_threshold: int,
    maximum_ticks: int,
    objective: ReserveObjective | None,
) -> dict[str, object]:
    core, campaign = frozen._load_external()
    policy = campaign.POLICY_FACTORY()
    policy.reset(seed)
    def make_gate(horizon: int) -> ObjectiveGate:
        return ObjectiveGate(
            lambda decision: frozen._primitive_actions(core, decision),
            config=WaitDominanceConfig(
                probe_ticks=horizon, wait_ticks=16, gauge_advantage=16
            ),
            objective=objective,
        )

    gate = make_gate(probe_ticks)
    rescue_gate = (
        None if rescue_probe_ticks is None else make_gate(rescue_probe_ticks)
    )
    attempted = kept = suppressed = 0
    rescue_queries = 0
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    terminated = truncated = False
    with campaign.IrisuEnv(
        library_path=frozen.RUNTIME,
        physics_backend="portable",
        config={
            "max_episode_ticks": maximum_ticks
            + max(probe_ticks, rescue_probe_ticks or 0)
        },
    ) as env:
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("portable reset seed differs")
        for _ in range(2):
            observation, _reward, terminated, truncated, _info = env.step(
                core.JOINT.Action.wait(1)
            )
        while int(observation["tick"]) < maximum_ticks and not (
            terminated or truncated
        ):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                attempted += 1
                active_gate = gate
                if (
                    rescue_gate is not None
                    and int(observation["gauge"]) <= rescue_gauge_threshold
                ):
                    active_gate = rescue_gate
                    rescue_queries += 1
                verdict = active_gate.evaluate(
                    env, observation, before, policy, decision
                )
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = active_gate.wait_decision(verdict.reason)
            for action in frozen._primitive_actions(core, decision):
                kind = int(action.kind)
                duration = int(action.wait_ticks) if kind == 0 else 1
                remaining = maximum_ticks - int(observation["tick"])
                if remaining <= 0:
                    break
                duration = min(duration, remaining)
                for _ in range(duration):
                    primitive = core.JOINT.Action.wait(1) if kind == 0 else action
                    observation, _reward, terminated, truncated, _info = env.step(
                        primitive
                    )
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
        return {
            "seed": seed,
            "score": int(observation["score"]),
            "tick": int(observation["tick"]),
            "level": int(observation["level"]),
            "gauge": int(observation["gauge"]),
            "highest_chain": int(observation["highest_chain"]),
            "terminated": bool(observation.get("terminated", terminated)),
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
        "scores_at_least_50000": sum(score >= 50_000 for score in scores),
        "median_survival_ticks": statistics.median(
            int(row["tick"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--probe-ticks", type=int, default=256)
    parser.add_argument("--rescue-probe-ticks", type=int)
    parser.add_argument("--rescue-gauge-threshold", type=int, default=30_000)
    parser.add_argument("--maximum-ticks", type=int, default=100_000)
    parser.add_argument("--objective", choices=("baseline", "reserve"), default="reserve")
    parser.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    parser.add_argument("--rescue-score-margin", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.probe_ticks < 1
        or args.maximum_ticks < 1
        or (args.rescue_probe_ticks is not None and args.rescue_probe_ticks < 1)
    ):
        parser.error("tick counts must be positive")
    objective = (
        None
        if args.objective == "baseline"
        else ReserveObjective(args.maximum_gauge_debt, args.rescue_score_margin)
    )
    rows = []
    for seed in args.seeds:
        row = run_episode(
            seed,
            probe_ticks=args.probe_ticks,
            rescue_probe_ticks=args.rescue_probe_ticks,
            rescue_gauge_threshold=args.rescue_gauge_threshold,
            maximum_ticks=args.maximum_ticks,
            objective=objective,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "schema": "irisu-portable-multiseed-teacher-probe-v1",
        "development_only": True,
        "promotion_eligible": False,
        "physics_backend": "portable",
        "model": "frozen-v5 directed-pair policy",
        "probe_ticks": args.probe_ticks,
        "rescue_probe_ticks": args.rescue_probe_ticks,
        "rescue_gauge_threshold": args.rescue_gauge_threshold,
        "maximum_ticks": args.maximum_ticks,
        "objective": args.objective,
        "objective_config": None
        if objective is None
        else {
            "maximum_gauge_debt": objective.maximum_gauge_debt,
            "rescue_score_margin": objective.rescue_score_margin,
        },
        "episodes": rows,
        "summary": summarize(rows),
    }
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
