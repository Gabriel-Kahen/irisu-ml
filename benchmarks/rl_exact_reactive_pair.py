#!/usr/bin/env python3
"""Train a fresh pair policy from a successful exact replay and gate it exactly."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from irisu_pointer.action import PointerActionSpec
from irisu_pointer.replay_supervision import (
    ReplayEvidenceIdentity,
    ReplayInputFrame,
    collect_replay_steering_supervision,
)
from irisu_pointer.shot_necessity import ExactWaitDominanceGate, WaitDominanceConfig
from irisu_pointer.steering_checkpoint import save_steering_checkpoint
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    steering_examples_from_replay,
    train_goal_conditioned_steering,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.exact_training_runtime import ExactTrainingRuntime
from irisu_rl.schema import TEACHER_V1

import rl_exact_reactive_distill as base


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "artifacts/r3/development/exact-100k-continuation-20260810/c42000-p192/continuation.u32le"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-reactive-pair-20260810-001"
SEED = 3_939_967_453
WORD = struct.Struct("<I")
HEADER = struct.Struct("<I4i")


def train(
    run_root: Path,
    runtime: ExactTrainingRuntime,
    *,
    trace_path: Path = TRACE,
    seed: int = SEED,
    steps: int,
) -> tuple[GoalConditionedSteeringPolicy, dict[str, object]]:
    raw = trace_path.read_bytes()
    packed = [value for (value,) in struct.iter_unpack("<I", raw)]
    frames = [
        ReplayInputFrame(index, bool(word & 1), bool(word & 2), (word >> 2) & 1023, (word >> 12) & 511, word)
        for index, word in enumerate(packed)
    ]
    encoder, pointer = TeacherStateEncoder(), PointerActionSpec()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": max(100_000, len(frames) + 1)}
    ) as session:
        observation, info = session.environment.reset(seed=seed)
        identity = ReplayEvidenceIdentity(
            "exact-reactive-pair-20260810",
            hashlib.sha256(raw).hexdigest(),
            runtime.identity.worker_sha256,
            int(info["config_hash"]),
            encoder.schema.sha256,
            pointer.sha256,
            runtime.identity.exact_library_sha256,
        )
        collection = collect_replay_steering_supervision(
            session.environment, frames, seed=seed, identity=identity, pointer_spec=pointer
        )
        provenance = session.provenance_manifest
    all_examples = steering_examples_from_replay(
        collection,
        encoder=encoder,
        pointer_spec=pointer,
        max_first_hit_delay_ticks=256,
        max_destination_distance_pixels=1_000.0,
        directional_offset_threshold=10.0,
    )
    # Bound the quadratic directed-pair tensor while retaining all game phases.
    count = min(384, len(all_examples))
    examples = tuple(
        all_examples[index * len(all_examples) // count] for index in range(count)
    )
    dataset = SteeringDataset(examples)
    torch.manual_seed(20260810)
    model = GoalConditionedSteeringModel(TEACHER_V1, pointer_spec=pointer)
    report = train_goal_conditioned_steering(
        model, dataset, steps=steps, batch_size=64, learning_rate=3e-4, seed=20260810
    )
    checkpoint = run_root / "checkpoints/exact-reactive-pair.pt"
    checkpoint_sha = save_steering_checkpoint(
        checkpoint,
        model,
        metadata={
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "portable_checkpoint_loaded": False,
            "teacher_trace": str(trace_path),
            "teacher_trace_sha256": hashlib.sha256(raw).hexdigest(),
            "exact_runtime": provenance,
            "available_training_examples": len(all_examples),
            "training_examples": len(examples),
        },
    )
    result = {
        "schema": "irisu-exact-reactive-pair-training-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "portable_checkpoint_loaded": False,
        "teacher_score": collection.metrics.final_score,
        "teacher_trace": str(trace_path),
        "teacher_shots": collection.metrics.shots_fired,
        "teacher_hit_rate": collection.metrics.hit_rate,
        "training_examples": len(examples),
        "available_training_examples": len(all_examples),
        "dataset_sha256": dataset.sha256,
        "training_report": asdict(report),
        "checkpoint": str(checkpoint.relative_to(run_root)),
        "checkpoint_sha256": checkpoint_sha,
        "exact_runtime": provenance,
    }
    (run_root / "training.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    policy = GoalConditionedSteeringPolicy(
        model,
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=1.0,
        artifact_sha256=checkpoint_sha,
    )
    policy.reset(seed)
    return policy, result


def evaluate(
    run_root: Path,
    runtime: ExactTrainingRuntime,
    policy: GoalConditionedSteeringPolicy,
    *,
    seed: int = SEED,
    maximum_ticks: int,
) -> dict[str, object]:
    gate = ExactWaitDominanceGate(
        lambda decision: decision.primitive_actions(),
        config=WaitDominanceConfig(probe_ticks=192, wait_ticks=16, gauge_advantage=8),
    )
    actions: list[int] = []
    attempted = kept = suppressed = 0
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum_ticks + 192}) as session:
        env = session.environment
        observation, _info = env.reset(seed=seed)
        terminated = truncated = False
        for _ in range(2):
            actions.append(0)
            observation, _reward, terminated, truncated, _info = env.step(base.decode(0))
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if decision.is_shot:
                attempted += 1
                verdict = gate.evaluate(env, observation, before, policy, decision)
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = gate.wait_decision(verdict.reason)
            for macro in decision.primitive_actions():
                duration = int(macro.wait_ticks) if int(macro.kind) == 0 else 1
                for _ in range(min(duration, maximum_ticks - int(observation["tick"]))):
                    action = base.decode(0) if int(macro.kind) == 0 else macro
                    word = 0 if int(action.kind) == 0 else (int(action.cursor_y) << 12) | (int(action.cursor_x) << 2) | int(action.kind)
                    actions.append(word)
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    if terminated or truncated:
                        break
                if terminated or truncated or int(observation["tick"]) >= maximum_ticks:
                    break
        final = base.checkpoint(observation)
        provenance = session.provenance_manifest
        state_u64 = f"0x{int(env.state_hash()):016x}"
    trace = b"".join(WORD.pack(value) for value in actions)
    replay = HEADER.pack(
        seed, int(final["level"]), int(final["score"]),
        int(final["highest_chain"]), 0,
    ) + bytes(32) + trace
    (run_root / "evaluation.u32le").write_bytes(trace)
    (run_root / "evaluation.rpy").write_bytes(replay)
    result = {
        "schema": "irisu-exact-reactive-pair-evaluation-v1",
        "physics_backend": "exact",
        "seed": seed,
        "maximum_ticks": maximum_ticks,
        "score": int(final["score"]),
        "natural_terminal": bool(final["terminated"]) and not bool(final["truncated"]),
        "censored": not bool(final["terminated"]),
        "final": final,
        "final_state_u64": state_u64,
        "attempted_shots": attempted,
        "kept_shots": kept,
        "suppressed_shots": suppressed,
        "gate_reasons": dict(reasons),
        "replay": "evaluation.rpy",
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    (run_root / "evaluation.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=base.WORKER)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--maximum-ticks", type=int, default=10_000)
    args = parser.parse_args()
    if args.run_root.exists():
        parser.error("--run-root must not exist")
    args.run_root.mkdir(parents=True)
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    trace_path = args.trace.resolve(strict=True)
    policy, training = train(
        args.run_root, runtime, trace_path=trace_path, seed=args.seed,
        steps=args.steps,
    )
    evaluation = evaluate(
        args.run_root, runtime, policy, seed=args.seed,
        maximum_ticks=args.maximum_ticks,
    )
    summary = {
        "schema": "irisu-exact-reactive-pair-summary-v1",
        "physics_backend": "exact",
        "portable_checkpoint_loaded": False,
        "checkpoint": training["checkpoint"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_examples": training["training_examples"],
        "score": evaluation["score"],
        "tick": evaluation["final"]["tick"],
        "natural_terminal": evaluation["natural_terminal"],
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
