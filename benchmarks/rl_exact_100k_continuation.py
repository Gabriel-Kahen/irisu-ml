#!/usr/bin/env python3
"""Resume the frozen proposal controller from an exact replay prefix.

This is a development search runner.  Every public state and every branch is
produced by the pinned exact worker, but the proposal checkpoint retains its
historical portable-training lineage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3k_sustainable_v3 as screen  # noqa: E402
from irisu_env import Action  # noqa: E402
from irisu_pointer.shot_necessity import (  # noqa: E402
    ExactWaitDominanceGate,
    WaitDominanceConfig,
)
from irisu_pointer.steering import (  # noqa: E402
    ClosedLoopSteeringExpert,
    SteeringExpertConfig,
)
from irisu_pointer.steering_checkpoint import load_steering_checkpoint  # noqa: E402
from irisu_pointer.steering_learning import GoalConditionedSteeringPolicy  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


WORD = struct.Struct("<I")
HEADER = struct.Struct("<I4i")


def decode(core: object | None, word: int) -> object:
    action_type = Action if core is None else core.JOINT.Action
    kind = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if kind == 1:
        return action_type.weak(x, y)
    if kind == 2:
        return action_type.strong(x, y)
    if kind == 3:
        return action_type.both(x, y)
    return action_type.wait(1)


def validate_exact_pair_metadata(metadata: Any) -> None:
    if (
        not isinstance(metadata, dict)
        or metadata.get("physics_backend") != "exact"
        or metadata.get("portable_checkpoint_loaded") is not False
        or metadata.get("state_producing_backends") != ["exact"]
    ):
        raise ValueError("pair checkpoint is not exact-only training evidence")


def load_exact_pair_policy(
    path: Path,
    *,
    expected_sha256: str | None,
    act_logit_bias: float,
) -> tuple[GoalConditionedSteeringPolicy, str]:
    checkpoint = load_steering_checkpoint(
        path,
        expected_sha256=expected_sha256,
        device="cpu",
    )
    validate_exact_pair_metadata(checkpoint.metadata)
    checkpoint.model.eval()
    return GoalConditionedSteeringPolicy(
        checkpoint.model,
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=act_logit_bias,
        artifact_sha256=checkpoint.sha256,
    ), checkpoint.sha256


def encode(action: object) -> int:
    kind = int(action.kind)
    if kind == 0:
        return 0
    x = min(1023, max(0, int(round(float(action.cursor_x)))))
    y = min(511, max(0, int(round(float(action.cursor_y)))))
    return (y << 12) | (x << 2) | kind


def checkpoint(observation: dict[str, object]) -> dict[str, int | bool]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "level": int(observation["level"]),
        "clears": int(observation["qualifying_clear_count"]),
        "highest_chain": int(observation["highest_chain"]),
        "terminated": bool(observation["terminated"]),
        "truncated": bool(observation["truncated"]),
    }


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3_939_967_453)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument(
        "--controller", choices=("gate", "pair", "expert", "wait"), default="gate"
    )
    parser.add_argument("--pair-checkpoint", type=Path)
    parser.add_argument("--pair-checkpoint-sha256")
    parser.add_argument("--pair-act-logit-bias", type=float, default=1.0)
    parser.add_argument("--probe", type=int, required=True)
    parser.add_argument("--survival-probe", type=int)
    parser.add_argument("--enter-gauge", type=int, default=8_000)
    parser.add_argument("--exit-gauge", type=int, default=20_000)
    parser.add_argument("--wait", type=int, default=16)
    parser.add_argument("--gauge-advantage", type=int, default=8)
    parser.add_argument(
        "--reserve-first",
        action="store_true",
        help="rank equal-horizon branches by gauge reserve before clears and score",
    )
    parser.add_argument("--max-ticks", type=int, default=120_000)
    parser.add_argument("--target-score", type=int, default=500_000)
    parser.add_argument("--observe", type=int, default=16)
    parser.add_argument("--abandon", type=int, default=32)
    parser.add_argument("--impact-side", type=float, default=0.5)
    parser.add_argument("--impact-below", type=float, default=0.75)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be absolute")
    if args.controller == "pair" and args.pair_checkpoint is None:
        parser.error("--controller pair requires --pair-checkpoint")
    if args.target_score < 0:
        parser.error("--target-score must be nonnegative")
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    args.run_root.mkdir(parents=True)

    source = args.trace.read_bytes()
    words = [word for (word,) in struct.iter_unpack("<I", source)]
    if not 0 < args.cutoff < len(words):
        raise ValueError("cutoff must be inside the source trace")
    pair_sha256: str | None = None
    if args.controller == "pair":
        core = None
        policy, pair_sha256 = load_exact_pair_policy(
            args.pair_checkpoint.resolve(strict=True),
            expected_sha256=args.pair_checkpoint_sha256,
            act_logit_bias=args.pair_act_logit_bias,
        )
        primitive_actions = lambda decision: tuple(decision.primitive_actions())
    else:
        core, campaign = screen._load_external()
        policy = (
            campaign.POLICY_FACTORY()
            if args.controller == "gate"
            else ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(
                observe_ticks=args.observe,
                resolution_wait_ticks=4,
                abandon_ticks=args.abandon,
                impact_side_sizes=args.impact_side,
                impact_below_sizes=args.impact_below,
                source_velocity_lead_ticks=1.0,
                minimum_pair_closure_sizes=0.05,
                ticks_per_second=50.0,
                hazard_remaining_ticks=48,
                enable_bonus=False,
                enable_rotten_matching=True,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            )
        )
        )
        primitive_actions = lambda decision: screen._primitive_actions(core, decision)
    policy.reset(args.seed)
    gate = ExactWaitDominanceGate(
        primitive_actions,
        config=WaitDominanceConfig(args.probe, args.wait, args.gauge_advantage),
    )
    survival_gate = (
        ExactWaitDominanceGate(
            primitive_actions,
            config=WaitDominanceConfig(
                args.survival_probe, args.wait, args.gauge_advantage
            ),
        )
        if args.survival_probe is not None
        else gate
    )
    survival_mode = False
    runtime = ExactTrainingRuntime(args.worker)
    actions = list(words[: args.cutoff])
    reasons: Counter[str] = Counter()
    attempted = kept = suppressed = 0
    started = time.monotonic()
    checkpoints: list[dict[str, int | bool]] = []
    with runtime.open_env(
        simulation_config={"max_episode_ticks": args.max_ticks + args.probe}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=args.seed)
        if int(info["seed"]) != args.seed:
            raise RuntimeError("seed mismatch")
        for word in actions:
            observation, _reward, terminated, truncated, _info = env.step(
                decode(core, word)
            )
            if terminated or truncated:
                raise RuntimeError("source terminated before cutoff")
        checkpoints.append(checkpoint(observation))
        print(json.dumps({"event": "prefix", **checkpoints[-1]}), flush=True)
        # The policy's temporal trackers are intentionally reset at the exact
        # prefix state. Its neural proposal is otherwise unchanged.
        while int(observation["tick"]) < args.max_ticks and not (
            terminated or truncated
        ):
            before = copy.deepcopy(policy)
            decision = (
                gate.wait_decision("wait-only natural-terminal completion")
                if args.controller == "wait"
                else policy.predict(observation)
            )
            if args.controller in {"gate", "pair"} and getattr(decision, "is_shot", False):
                attempted += 1
                gauge = int(observation["gauge"])
                if args.survival_probe is not None:
                    if not survival_mode and gauge < args.enter_gauge:
                        survival_mode = True
                    elif survival_mode and gauge > args.exit_gauge:
                        survival_mode = False
                active_gate = survival_gate if survival_mode else gate
                verdict = active_gate.evaluate(
                    env, observation, before, policy, decision
                )
                execute_shot = verdict.execute_shot
                reason = verdict.reason
                if args.reserve_first:
                    shot, wait = verdict.shot, verdict.wait
                    shot_failed = shot.terminated or shot.truncated
                    wait_failed = wait.terminated or wait.truncated
                    shot_rank = (
                        shot.survival_ticks,
                        not shot_failed,
                        shot.final_gauge,
                        shot.minimum_gauge,
                        shot.clears,
                        shot.score,
                    )
                    wait_rank = (
                        wait.survival_ticks,
                        not wait_failed,
                        wait.final_gauge,
                        wait.minimum_gauge,
                        wait.clears,
                        wait.score,
                    )
                    execute_shot = shot_rank > wait_rank
                    reason = "reserve-shot" if execute_shot else "reserve-wait"
                reasons[f"{'survival' if survival_mode else 'score'}:{reason}"] += 1
                if execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = gate.wait_decision(verdict.reason)
            for action in primitive_actions(decision):
                kind = int(action.kind)
                duration = int(action.wait_ticks) if kind == 0 else 1
                duration = min(duration, args.max_ticks - int(observation["tick"]))
                for _ in range(duration):
                    primitive = decode(core, 0) if kind == 0 else action
                    actions.append(encode(primitive))
                    observation, _reward, terminated, truncated, _info = env.step(primitive)
                    tick = int(observation["tick"])
                    if tick % 2_500 == 0:
                        row = checkpoint(observation)
                        checkpoints.append(row)
                        partial = b"".join(WORD.pack(word) for word in actions)
                        replace(args.run_root / "progress.u32le", partial)
                        replace(
                            args.run_root / "progress.json",
                            json.dumps(
                                {
                                    "seed": args.seed,
                                    "source_trace": str(args.trace.resolve()),
                                    "cutoff": args.cutoff,
                                    "action_count": len(actions),
                                    "trace_sha256": hashlib.sha256(partial).hexdigest(),
                                    "checkpoint": row,
                                },
                                sort_keys=True,
                                indent=2,
                            ).encode()
                            + b"\n",
                        )
                        print(json.dumps({"event": "checkpoint", **row}), flush=True)
                    if terminated or truncated or tick >= args.max_ticks:
                        break
                if terminated or truncated or int(observation["tick"]) >= args.max_ticks:
                    break
        final = checkpoint(observation)
        state_hash = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    trace = b"".join(WORD.pack(word) for word in actions)
    replay = HEADER.pack(
        args.seed, int(final["level"]), int(final["score"]),
        int(final["highest_chain"]), 0,
    ) + bytes(32) + trace
    write_new(args.run_root / "continuation.u32le", trace)
    write_new(args.run_root / "continuation.rpy", replay)
    result = {
        "schema": "irisu-exact-100k-continuation-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "controller": args.controller,
        "proposal_lineage": (
            "portable-trained frozen-v5; exact closed-loop continuation"
            if args.controller == "gate"
            else (
                "exact-only neural pair checkpoint; exact closed-loop continuation"
                if args.controller == "pair"
                else (
                    "checkpoint-free exact public-state heuristic"
                    if args.controller == "expert"
                    else "checkpoint-free wait-only terminal completion"
                )
            )
        ),
        "pair_checkpoint": (
            None
            if args.controller != "pair"
            else {
                "path": str(args.pair_checkpoint.resolve()),
                "sha256": pair_sha256,
                "act_logit_bias": args.pair_act_logit_bias,
            }
        ),
        "seed": args.seed,
        "source_trace": str(args.trace.resolve()),
        "source_trace_sha256": hashlib.sha256(source).hexdigest(),
        "cutoff": args.cutoff,
        "gate": {
            "probe_ticks": args.probe,
            "survival_probe_ticks": args.survival_probe,
            "enter_gauge": args.enter_gauge,
            "exit_gauge": args.exit_gauge,
            "wait_ticks": args.wait,
            "gauge_advantage": args.gauge_advantage,
            "reserve_first": args.reserve_first,
        },
        "expert": {
            "observe_ticks": args.observe,
            "abandon_ticks": args.abandon,
            "impact_side_sizes": args.impact_side,
            "impact_below_sizes": args.impact_below,
        },
        "attempted_shots": attempted,
        "kept_shots": kept,
        "suppressed_shots": suppressed,
        "gate_reasons": dict(sorted(reasons.items())),
        "natural_terminal": bool(final["terminated"]) and not bool(final["truncated"]),
        "censored": not bool(final["terminated"]),
        "target_100k_met": bool(final["terminated"]) and int(final["score"]) >= 100_000,
        "target_score": args.target_score,
        "target_met": (
            bool(final["terminated"]) and int(final["score"]) >= args.target_score
        ),
        "final": final,
        "final_state_u64": state_hash,
        "action_count": len(actions),
        "trace_sha256": hashlib.sha256(trace).hexdigest(),
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
        "checkpoints": checkpoints,
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    write_new(
        args.run_root / "result.json",
        json.dumps(result, sort_keys=True, indent=2).encode() + b"\n",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
