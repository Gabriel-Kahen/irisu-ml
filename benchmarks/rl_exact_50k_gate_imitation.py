#!/usr/bin/env python3
"""Imitate exact wait-dominance labels, then fall back to exact search OOD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import struct
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_50k_search as search  # noqa: E402
import rl_r3k_sustainable_v3 as screen  # noqa: E402
import rl_r3m_shot_restraint as helpers  # noqa: E402
from irisu_pointer.shot_necessity import (  # noqa: E402
    ExactWaitDominanceGate,
    WaitDominanceConfig,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


CONFIG_PATH = ROOT / "configs/rl/experiments/exact-50k-search-gate-v1.toml"
DEFAULT_RUN_ROOT = (
    ROOT / "artifacts/r3/development/exact-50k-search-gate-20260810-001"
)
ACTION_WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")
FEATURE_NAMES = (*search.FEATURE_NAMES, "intent_steer_match")


def config() -> dict[str, Any]:
    value = tomllib.loads(CONFIG_PATH.read_text())
    if value.get("physics_backend") != "exact" or value.get(
        "state_producing_backends"
    ) != ["exact"]:
        raise RuntimeError("gate imitation config is not exact-only")
    return value


def gate_config(value: dict[str, Any]) -> WaitDominanceConfig:
    teacher = value["teacher"]
    return WaitDominanceConfig(
        probe_ticks=int(teacher["probe_ticks"]),
        wait_ticks=int(teacher["wait_ticks"]),
        gauge_advantage=int(teacher["gauge_advantage"]),
    )


def gate_features(observation: dict[str, Any], decision: Any) -> list[float]:
    return [
        *search.features(observation, decision),
        float(decision.intent.value == "steer_match"),
    ]


def source_identity() -> dict[str, object]:
    value = config()
    worker = Path(value["runtime"]["worker_path"])
    unit = Path(value["teacher"]["unit"])
    trace = Path(value["teacher"]["trace"])
    files = (
        Path(__file__).resolve(),
        CONFIG_PATH,
        worker,
        unit,
        trace,
        Path(search.__file__).resolve(),
        Path(screen.__file__).resolve(),
        Path(helpers.__file__).resolve(),
        ROOT / "python/irisu_pointer/shot_necessity.py",
        ROOT / "python/irisu_rl/exact_training_runtime.py",
    )
    identities = {str(path): search.sha256_file(path) for path in files}
    if identities[str(worker)] != value["runtime"]["worker_sha256"]:
        raise RuntimeError("exact worker bytes differ")
    teacher = search.read_json(unit)
    helpers.verify_self_hash(teacher, "exact gate teacher unit")
    if (
        teacher.get("physics_backend") != "exact"
        or int(teacher.get("seed", -1)) != int(value["teacher"]["seed"])
        or identities[str(trace)] != teacher.get("trace_sha256")
    ):
        raise RuntimeError("exact gate teacher provenance differs")
    return search.with_sha(
        {
            "schema": "irisu-exact-gate-imitation-source-v1",
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "promotion_eligible": True,
            "files": identities,
            "teacher_unit_sha256": teacher["sha256"],
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(run_root)
    identity = source_identity()
    value = config()
    plan = search.with_sha(
        {
            "schema": "irisu-exact-gate-imitation-plan-v1",
            "source_identity_sha256": identity["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "teacher": dict(value["teacher"]),
            "full_game": dict(value["full_game"]),
            "policy": (
                "exact nearest-state imitation inside the verified teacher "
                "trajectory; exact wait-dominance branches on every OOD shot"
            ),
        }
    )
    run_root.mkdir(parents=True)
    search.write_json_new(run_root / "source-identity.json", identity)
    search.write_json_new(run_root / "plan.json", plan)
    return plan


def validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = search.read_json(run_root / "source-identity.json")
    plan = search.read_json(run_root / "plan.json")
    search.verify_sha(identity, "gate imitation source")
    search.verify_sha(plan, "gate imitation plan")
    if identity != source_identity() or plan["source_identity_sha256"] != identity["sha256"]:
        raise RuntimeError("gate imitation source identity changed")
    return identity, plan


def _expanded_actions(core: Any, decision: Any) -> list[Any]:
    output = []
    for action in screen._primitive_actions(core, decision):
        kind = core.JOINT.ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
        output.extend(
            core.JOINT.Action.wait(1) if kind is core.JOINT.ActionKind.WAIT else action
            for _ in range(duration)
        )
    return output


def train(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    manifest_path = run_root / "gate-imitation-checkpoint.json"
    if manifest_path.exists():
        result = search.read_json(manifest_path)
        search.verify_sha(result, "gate imitation checkpoint")
        return result
    value = config()
    teacher_path = Path(value["teacher"]["unit"])
    teacher = search.read_json(teacher_path)
    receipts = list(teacher["gate_receipts"])
    trace = Path(value["teacher"]["trace"]).read_bytes()
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    seed = int(value["teacher"]["seed"])
    policy.reset(seed)
    gate = ExactWaitDominanceGate(
        lambda decision: screen._primitive_actions(core, decision),
        config=gate_config(value),
    )
    rows: list[dict[str, object]] = []
    cursor = 0
    with ExactTrainingRuntime(Path(value["runtime"]["worker_path"])).open_env(
        simulation_config={
            "max_episode_ticks": int(value["teacher"]["horizon_ticks"])
            + int(value["teacher"]["probe_ticks"])
        }
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=seed)
        receipt_index = 0
        while cursor < len(words):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                receipt = receipts[receipt_index]
                receipt_index += 1
                if int(receipt["tick"]) != int(observation["tick"]):
                    raise RuntimeError("exact gate receipt tick differs")
                rows.append(
                    {
                        "tick": int(observation["tick"]),
                        "state_u64": f"0x{int(env.state_hash()):016x}",
                        "features": gate_features(observation, decision),
                        "execute_shot": bool(receipt["execute_shot"]),
                        "reason": str(receipt["reason"]),
                    }
                )
                if not receipt["execute_shot"]:
                    policy = before
                    decision = gate.wait_decision(str(receipt["reason"]))
            for expected in _expanded_actions(core, decision):
                if cursor >= len(words):
                    break
                actual = helpers.decode_action(core, words[cursor])
                if helpers.encode_action(expected) != words[cursor]:
                    raise RuntimeError("teacher trace action differs from receipt policy")
                observation, _reward, _terminated, _truncated, _info = env.step(actual)
                cursor += 1
        final = helpers.checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if (
        len(rows) != len(receipts)
        or final != teacher["final"]
        or snapshot_sha != teacher["final_snapshot_sha256"]
    ):
        raise RuntimeError("exact gate teacher reconstruction differs")
    inputs = torch.tensor([row["features"] for row in rows], dtype=torch.float64)
    labels = torch.tensor([row["execute_shot"] for row in rows], dtype=torch.bool)
    mean = inputs.mean(dim=0)
    scale = inputs.std(dim=0, unbiased=False).clamp_min(1e-12)
    checkpoint_path = run_root / "exact-gate-imitation.pt"
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema": "irisu-exact-gate-nearest-state-v1",
            "features": inputs,
            "labels": labels,
            "mean": mean,
            "scale": scale,
            "feature_names": FEATURE_NAMES,
            "teacher_unit_sha256": teacher["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ("exact",),
        },
        temporary,
    )
    os.link(temporary, checkpoint_path)
    temporary.unlink()
    manifest = search.with_sha(
        {
            "schema": "irisu-exact-gate-imitation-checkpoint-manifest-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "checkpoint": str(checkpoint_path.relative_to(run_root)),
            "checkpoint_sha256": search.sha256_file(checkpoint_path),
            "training_examples": len(rows),
            "kept_examples": int(labels.sum()),
            "suppressed_examples": int((~labels).sum()),
            "teacher_final": final,
            "teacher_exact_runtime": provenance,
        }
    )
    search.write_json_new(manifest_path, manifest)
    return manifest


def load_checkpoint(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = train(run_root)
    path = run_root / str(manifest["checkpoint"])
    if search.sha256_file(path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("gate imitation checkpoint bytes changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("physics_backend") != "exact"
        or tuple(checkpoint.get("state_producing_backends", ())) != ("exact",)
        or tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES
    ):
        raise RuntimeError("gate imitation checkpoint provenance differs")
    return checkpoint, manifest


def _imitate(
    checkpoint: dict[str, Any], observation: dict[str, Any], decision: Any
) -> tuple[bool, float]:
    value = torch.tensor(gate_features(observation, decision), dtype=torch.float64)
    normalized = (value - checkpoint["mean"]) / checkpoint["scale"]
    training = (checkpoint["features"] - checkpoint["mean"]) / checkpoint["scale"]
    distances = ((training - normalized) ** 2).sum(dim=1).sqrt()
    index = int(distances.argmin())
    return bool(checkpoint["labels"][index]), float(distances[index])


def play(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    output = run_root / "full-game.json"
    if output.exists():
        result = search.read_json(output)
        search.verify_sha(result, "gate imitation full game")
        return result
    value = config()
    full = value["full_game"]
    checkpoint, checkpoint_manifest = load_checkpoint(run_root)
    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    seed = int(full["seed"])
    policy.reset(seed)
    gate = ExactWaitDominanceGate(
        lambda decision: screen._primitive_actions(core, decision),
        config=gate_config(value),
    )
    maximum_tick = int(full["maximum_ticks"])
    teacher_horizon = int(value["teacher"]["horizon_ticks"])
    tolerance = float(full["exact_match_tolerance"])
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    imitation_count = exact_query_count = kept = suppressed = 0
    started = time.monotonic()
    with ExactTrainingRuntime(Path(value["runtime"]["worker_path"])).open_env(
        simulation_config={
            "max_episode_ticks": maximum_tick + int(value["teacher"]["probe_ticks"])
        }
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=seed)
        checkpoints.append(helpers.checkpoint(env, observation))
        terminated = truncated = False
        next_checkpoint = int(full["checkpoint_stride_ticks"])
        while int(observation["tick"]) < maximum_tick and not (terminated or truncated):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                execute = True
                distance = float("inf")
                if int(observation["tick"]) < teacher_horizon:
                    execute, distance = _imitate(checkpoint, observation, decision)
                if int(observation["tick"]) < teacher_horizon and distance <= tolerance:
                    imitation_count += 1
                    if not execute:
                        policy = before
                        decision = gate.wait_decision("exact-imitation-suppression")
                else:
                    verdict = gate.evaluate(env, observation, before, policy, decision)
                    exact_query_count += 1
                    execute = verdict.execute_shot
                    if not execute:
                        policy = before
                        decision = gate.wait_decision(verdict.reason)
                kept += int(execute)
                suppressed += int(not execute)
            for action in _expanded_actions(core, decision):
                actions.append(helpers.encode_action(action))
                observation, _reward, terminated, truncated, _info = env.step(action)
                tick = int(observation["tick"])
                if tick >= next_checkpoint:
                    checkpoints.append(helpers.checkpoint(env, observation))
                    next_checkpoint += int(full["checkpoint_stride_ticks"])
                    print(
                        json.dumps(
                            {
                                "phase": "gate-imitation-full-game",
                                "tick": tick,
                                "score": int(observation["score"]),
                                "gauge": int(observation["gauge"]),
                                "imitated": imitation_count,
                                "exact_queries": exact_query_count,
                                "elapsed_seconds": time.monotonic() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if terminated or truncated or tick >= maximum_tick:
                    break
        final = helpers.checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "full-game.u32le"
    search.write_new(trace_path, trace)
    replay = (
        REPLAY_HEADER.pack(
            seed,
            int(final["level"]),
            int(final["score"]),
            int(final["highest_chain"]),
            0,
        )
        + bytes(32)
        + trace
    )
    replay_path = run_root / "best-exact-gate-imitation.rpy"
    search.write_new(replay_path, replay)
    result = search.with_sha(
        {
            "schema": "irisu-exact-gate-imitation-full-game-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "promotion_eligible": True,
            "seed": seed,
            "score": int(final["score"]),
            "target_score": int(full["target_score"]),
            "target_reached": int(final["score"]) >= int(full["target_score"]),
            "survival_ticks": int(final["tick"]),
            "terminal": bool(final["terminated"]),
            "censored": int(final["tick"]) >= maximum_tick and not bool(final["terminated"]),
            "level": int(final["level"]),
            "highest_chain": int(final["highest_chain"]),
            "clears": int(final["clears"]),
            "final_gauge": int(final["gauge"]),
            "imitation_decisions": imitation_count,
            "exact_search_queries": exact_query_count,
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "imitation_checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "replay_file": str(replay_path.relative_to(run_root)),
            "replay_sha256": hashlib.sha256(replay).hexdigest(),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha,
            "exact_runtime": provenance,
            "wall_seconds": time.monotonic() - started,
        }
    )
    search.write_json_new(output, result)
    return result


def verify(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    game = play(run_root)
    output = run_root / "verification.json"
    if output.exists():
        result = search.read_json(output)
        search.verify_sha(result, "gate imitation verification")
        return result
    value = config()
    trace = (run_root / str(game["trace_file"])).read_bytes()
    if hashlib.sha256(trace).hexdigest() != game["trace_sha256"]:
        raise RuntimeError("gate imitation trace bytes changed")
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    expected = {int(row["tick"]): row for row in game["checkpoints"]}
    core, _campaign = screen._load_external()
    with ExactTrainingRuntime(Path(value["runtime"]["worker_path"])).open_env(
        simulation_config={
            "max_episode_ticks": int(value["full_game"]["maximum_ticks"])
            + int(value["teacher"]["probe_ticks"])
        }
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=int(game["seed"]))
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                helpers.decode_action(core, word)
            )
            tick = int(observation["tick"])
            if tick in expected and helpers.checkpoint(env, observation) != expected[tick]:
                raise RuntimeError(f"exact replay differs at tick {tick}")
        final = helpers.checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if final != game["final"] or snapshot_sha != game["final_snapshot_sha256"]:
        raise RuntimeError("gate imitation exact replay final closure differs")
    result = search.with_sha(
        {
            "schema": "irisu-exact-gate-imitation-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "full_game_sha256": game["sha256"],
            "physics_backend": "exact",
            "replay_reexecution_backend": "exact",
            "verified": True,
            "score": int(final["score"]),
            "survival_ticks": int(final["tick"]),
            "action_count": len(words),
            "replay_sha256": game["replay_sha256"],
            "exact_runtime": provenance,
        }
    )
    search.write_json_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "train", "play", "verify", "run-all"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.run_root)
    elif args.command == "train":
        result = train(args.run_root)
    elif args.command == "play":
        result = play(args.run_root)
    elif args.command == "verify":
        result = verify(args.run_root)
    else:
        if not args.run_root.exists():
            initialize(args.run_root)
        train(args.run_root)
        play(args.run_root)
        result = verify(args.run_root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
