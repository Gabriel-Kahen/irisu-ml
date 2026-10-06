#!/usr/bin/env python3
"""Exact multi-trajectory state distillation with counterfactual DAgger.

This augments the state-reactive policy with exact states visited after one
teacher-shot suppression.  The teacher schedule is used only to label the
exact DAgger rollouts; inference receives public physical state only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_reactive_distill as base  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


BETTER_TRACE = ROOT / "artifacts/r3/development/exact-100k-continuation-20260810/c42000-p192/continuation.u32le"
OLDER_TRACE = ROOT / "artifacts/r3/development/exact-100k-expert-takeover-20260810/c44000-default/continuation.u32le"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-reactive-dagger-20260810-001"
SEED = 3_939_967_453
WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")


@dataclass(frozen=True)
class Rollout:
    label: str
    seed: int
    words: tuple[int, ...]
    states: tuple[tuple[str, np.ndarray, int], ...]
    final: dict[str, object]
    runtime: dict[str, object]
    state_u64: str


def words(path: Path) -> tuple[int, ...]:
    return tuple(value for (value,) in struct.iter_unpack("<I", path.read_bytes()))


def replay(
    worker: Path,
    label: str,
    scheduled: tuple[int, ...],
    *,
    seed: int = SEED,
    maximum_ticks: int = 100_000,
    mutation_tick: int | None = None,
) -> Rollout:
    runtime = ExactTrainingRuntime(worker)
    rows: list[tuple[str, np.ndarray, int]] = []
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum_ticks}) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info["seed"]) != seed:
            raise RuntimeError("DAgger exact seed differs")
        terminated = truncated = False
        step_info: dict[str, Any] = {}
        index = 0
        while not (terminated or truncated):
            teacher_word = scheduled[index] if index < len(scheduled) else 0
            action_word = 0 if mutation_tick == index else teacher_word
            # The perturbed state itself is labeled by the counterfactual action;
            # subsequent states receive the original exact teacher schedule.
            if mutation_tick is None or index >= mutation_tick:
                rows.append((base.state_digest(observation), base.features(observation), action_word))
            observation, _reward, terminated, truncated, step_info = env.step(
                base.decode(action_word)
            )
            index += 1
        final = base.canonical_checkpoint(observation, step_info)
        state_u64 = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    return Rollout(label, seed, scheduled, tuple(rows), final, provenance, state_u64)


def replay_job(
    item: tuple[Path, str, tuple[int, ...], int, int, int | None]
) -> Rollout:
    worker, label, scheduled, seed, maximum_ticks, mutation_tick = item
    return replay(
        worker, label, scheduled, seed=seed,
        maximum_ticks=maximum_ticks, mutation_tick=mutation_tick,
    )


def save_checkpoint(
    run_root: Path,
    primary: Rollout,
    secondary: Rollout,
    branches: list[Rollout],
    *,
    primary_trace: Path = BETTER_TRACE,
    secondary_trace: Path = OLDER_TRACE,
    seed: int = SEED,
    teacher_action_lineage: str = "external exact traces; inspect their manifests",
) -> dict[str, object]:
    state_actions: dict[str, int] = {}
    collisions = 0
    # Higher-scoring primary owns any shared state; the older and perturbed
    # trajectories only add unseen exact states.
    for rollout in (primary, secondary, *branches):
        for digest, _feature, action in rollout.states:
            if digest in state_actions and state_actions[digest] != action:
                collisions += 1
                continue
            state_actions.setdefault(digest, action)
    exemplar_features: list[np.ndarray] = []
    exemplar_words: list[int] = []
    for rollout in (primary, secondary, *branches):
        stride = 16 if rollout in (primary, secondary) else 64
        for index, (_digest, feature, action) in enumerate(rollout.states):
            if action or index % stride == 0:
                exemplar_features.append(feature)
                exemplar_words.append(action)
    matrix = np.stack(exemplar_features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    checkpoint = {
        "schema": "irisu-exact-reactive-public-state-dagger-v1",
        "physics_backend": "exact",
        "state_producing_backends": ("exact",),
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": teacher_action_lineage,
        "inference_inputs": "public physical state; seed/tick/score/counters excluded",
        "feature_schema": "public-combo-topology-v2",
        "state_actions": state_actions,
        "mean": torch.from_numpy(mean),
        "scale": torch.from_numpy(scale),
        "exemplar_features": torch.from_numpy((matrix - mean) / scale),
        "exemplar_words": torch.tensor(exemplar_words, dtype=torch.int64),
        "source_seed_recorded_for_audit_only": seed,
        "teacher_trace_sha256s": (
            base.sha256_file(primary_trace), base.sha256_file(secondary_trace)
        ),
        "dagger_branch_labels": tuple(value.label for value in branches),
    }
    checkpoint_path = run_root / "checkpoints/exact-reactive-dagger.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema": "irisu-exact-reactive-dagger-training-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": teacher_action_lineage,
        "policy_inputs_exclude": sorted(base._EPISODE_ONLY | base._BODY_IDENTITY_OR_CLOCK | {"seed"}),
        "primary_teacher": {
            "seed": primary.seed,
            "trace": str(primary_trace),
            "trace_sha256": base.sha256_file(primary_trace),
            "final": primary.final,
        },
        "secondary_teacher": {
            "seed": secondary.seed,
            "trace": str(secondary_trace),
            "trace_sha256": base.sha256_file(secondary_trace),
            "final": secondary.final,
        },
        "dagger_branches": [
            {"label": item.label, "final": item.final, "state_u64": item.state_u64}
            for item in branches
        ],
        "training_states": len(state_actions),
        "conflicting_labels_kept_from_higher_priority_rollout": collisions,
        "ood_exemplars": len(exemplar_words),
        "shot_exemplars": sum(value != 0 for value in exemplar_words),
        "checkpoint": str(checkpoint_path.relative_to(run_root)),
        "checkpoint_sha256": base.sha256_file(checkpoint_path),
        "exact_runtime": primary.runtime,
    }
    (run_root / "training.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def load_policy(run_root: Path) -> base.ReactiveExactPolicy:
    manifest = json.loads((run_root / "training.json").read_text())
    path = run_root / manifest["checkpoint"]
    if base.sha256_file(path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("DAgger checkpoint bytes differ")
    return base.ReactiveExactPolicy(torch.load(path, map_location="cpu", weights_only=False))


def evaluate(
    run_root: Path,
    worker: Path,
    label: str,
    *,
    seed: int,
    maximum_ticks: int = 100_000,
    forced_tick: int | None = None,
) -> dict[str, object]:
    policy = load_policy(run_root)
    runtime = ExactTrainingRuntime(worker)
    actions: list[int] = []
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum_ticks}) as session:
        env = session.environment
        observation, _info = env.reset(seed=seed)
        terminated = truncated = False
        step_info: dict[str, Any] = {}
        while not (terminated or truncated):
            tick = int(observation["tick"])
            word = 0 if forced_tick == tick else policy.predict_word(observation)
            actions.append(word)
            observation, _reward, terminated, truncated, step_info = env.step(base.decode(word))
        final = base.canonical_checkpoint(observation, step_info)
        state_u64 = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    trace = b"".join(WORD.pack(value) for value in actions)
    replay_bytes = REPLAY_HEADER.pack(
        seed, int(final["level"]), int(final["score"]), int(final["highest_chain"]), 0
    ) + bytes(32) + trace
    output = run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{label}.u32le").write_bytes(trace)
    (output / f"{label}.rpy").write_bytes(replay_bytes)
    result = {
        "schema": "irisu-exact-reactive-dagger-evaluation-v1",
        "physics_backend": "exact",
        "label": label,
        "seed": seed,
        "forced_wait_tick": forced_tick,
        "natural_terminal": bool(final["terminated"]) and not bool(final["truncated"]),
        "score": int(final["score"]),
        "final": final,
        "final_state_u64": state_u64,
        "exact_state_retrievals": policy.exact_retrievals,
        "ood_retrievals": policy.ood_retrievals,
        "trace": f"evaluations/{label}.u32le",
        "trace_sha256": hashlib.sha256(trace).hexdigest(),
        "replay": f"evaluations/{label}.rpy",
        "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "exact_runtime": provenance,
    }
    (output / f"{label}.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=base.WORKER)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--primary-trace", type=Path, default=BETTER_TRACE)
    parser.add_argument("--primary-score", type=int, default=68_750)
    parser.add_argument("--secondary-trace", type=Path, default=OLDER_TRACE)
    parser.add_argument("--secondary-score", type=int, default=53_294)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--secondary-seed",
        type=int,
        help="exact seed for the secondary teacher; defaults to --seed",
    )
    parser.add_argument("--maximum-ticks", type=int, default=100_000)
    parser.add_argument("--target-score", type=int, default=500_000)
    parser.add_argument(
        "--teacher-action-lineage",
        default="external exact traces; inspect their manifests",
    )
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--holdouts", type=int, default=1)
    args = parser.parse_args()
    if args.run_root.exists():
        parser.error("--run-root must not exist")
    worker = args.worker.resolve(strict=True)
    secondary_seed = args.seed if args.secondary_seed is None else args.secondary_seed
    if not 0 <= secondary_seed <= 0xFFFFFFFF:
        parser.error("--secondary-seed is outside uint32")
    primary_trace = args.primary_trace.resolve(strict=True)
    secondary_trace = args.secondary_trace.resolve(strict=True)
    primary_words, secondary_words = words(primary_trace), words(secondary_trace)
    if args.maximum_ticks < max(len(primary_words), len(secondary_words)):
        parser.error("--maximum-ticks must cover both teacher traces")
    if args.branches < 0 or args.holdouts < 0 or args.target_score < 0:
        parser.error("--branches, --holdouts, and --target-score must be nonnegative")
    shot_ticks = [index for index, value in enumerate(primary_words) if value]
    # Spread counterfactuals over the late half, where score-preserving recovery matters.
    candidates = shot_ticks[len(shot_ticks) // 2 :]
    branch_ticks = [
        candidates[(index + 1) * len(candidates) // (args.branches + 1)]
        for index in range(args.branches)
    ]
    args.run_root.mkdir(parents=True)
    started = time.monotonic()
    jobs = [
        (f"primary-{args.primary_score}", primary_words, args.seed, None),
        (f"secondary-{args.secondary_score}", secondary_words, secondary_seed, None),
        *[
            (f"suppress-{tick}", primary_words, args.seed, tick)
            for tick in branch_ticks
        ],
    ]
    process_jobs = [
        (worker, label, scheduled, seed, args.maximum_ticks, mutation)
        for label, scheduled, seed, mutation in jobs
    ]
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        rollouts = list(pool.map(replay_job, process_jobs))
    primary, secondary, *branches = rollouts
    if (
        int(primary.final["score"]) != args.primary_score
        or int(secondary.final["score"]) != args.secondary_score
    ):
        raise RuntimeError("exact teacher reconstruction differs")
    training = save_checkpoint(
        args.run_root, primary, secondary, branches,
        primary_trace=primary_trace, secondary_trace=secondary_trace,
        seed=args.seed, teacher_action_lineage=args.teacher_action_lineage,
    )
    source = evaluate(
        args.run_root, worker, "source", seed=args.seed,
        maximum_ticks=args.maximum_ticks,
    )
    recoveries = [
        evaluate(
            args.run_root, worker, item.label, seed=args.seed,
            maximum_ticks=args.maximum_ticks, forced_tick=tick,
        )
        for item, tick in zip(branches, branch_ticks, strict=True)
    ]
    holdouts = [
        evaluate(
            args.run_root, worker, f"holdout-{seed:08x}", seed=seed,
            maximum_ticks=args.maximum_ticks,
        )
        for seed in base.HOLDOUT_SEEDS[: args.holdouts]
    ]
    evaluator = ROOT / "tools/evaluate-rpy.py"
    accepted = subprocess.run(
        [sys.executable, str(evaluator), str(args.run_root / source["replay"]),
         "--worker", str(worker), "--purpose", "target", "--compact"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    acceptance = json.loads(accepted.stdout)
    (args.run_root / "source-replay-acceptance.json").write_text(
        json.dumps(acceptance, sort_keys=True, indent=2) + "\n"
    )
    summary = {
        "schema": "irisu-exact-reactive-dagger-summary-v1",
        "physics_backend": "exact",
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": training["teacher_action_lineage"],
        "source_score": source["score"],
        "source_natural_terminal": source["natural_terminal"],
        "target_score": args.target_score,
        "target_reached": (
            source["natural_terminal"] and source["score"] >= args.target_score
        ),
        "source_replay_accepted": acceptance.get("status", {}).get("accepted") is True,
        "source_exact_retrieval_fraction": source["exact_state_retrievals"] / source["final"]["tick"],
        "recovery_scores": [item["score"] for item in recoveries],
        "recovery_ood_retrievals": [item["ood_retrievals"] for item in recoveries],
        "holdout_scores": [item["score"] for item in holdouts],
        "generalization_claim": "local exact perturbation recovery measured; unseen seed remains separately reported",
        "checkpoint": training["checkpoint"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "source_replay": source["replay"],
        "source_replay_sha256": source["replay_sha256"],
        "wall_seconds": time.monotonic() - started,
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
