#!/usr/bin/env python3
"""Checkpoint-free SteeringExpert under an exact wait-dominance gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import time
import tomllib
from collections import Counter
from pathlib import Path

from irisu_env import Action
from irisu_pointer.shot_necessity import ExactWaitDominanceGate, WaitDominanceConfig
from irisu_pointer.steering import ClosedLoopSteeringExpert, SteeringExpertConfig
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rl/experiments/exact-50k-expert-gate-v1.toml"
WORKER = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/exact-runtime-backup/irisu-exact-worker"
WORD = struct.Struct("<I")
HEADER = struct.Struct("<I4i")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(action: Action) -> int:
    kind = int(action.kind)
    return 0 if kind == 0 else (int(action.cursor_y) << 12) | (int(action.cursor_x) << 2) | kind


def checkpoint(env, observation):
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "level": int(observation["level"]),
        "clears": int(observation["qualifying_clear_count"]),
        "highest_chain": int(observation["highest_chain"]),
        "gauge": int(observation["gauge"]),
        "terminated": bool(observation["terminated"]),
        "state_u64": f"0x{int(env.state_hash()):016x}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be absolute")
    if args.run_root.exists() or "exact-50k-ppo-" not in args.run_root.name:
        parser.error("new --run-root must use exact-50k-ppo-*")
    config_path = args.config.resolve(strict=True)
    config = tomllib.loads(config_path.read_text())
    if config.get("physics_backend") != "exact" or config.get("state_producing_backends") != ["exact"]:
        raise RuntimeError("expert-gate config is not exact-only")
    runtime = ExactTrainingRuntime(args.worker)
    expert = ClosedLoopSteeringExpert(config=SteeringExpertConfig(**config["expert"]))
    gate = ExactWaitDominanceGate(
        lambda decision: decision.primitive_actions(),
        config=WaitDominanceConfig(**config["gate"]),
    )
    seed = int(config["seed"])
    maximum_ticks = int(config["maximum_ticks"])
    args.run_root.mkdir(parents=True)
    identity = {
        "version": "exact-50k-expert-gate-identity-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "checkpoint_free": True,
        "portable_model_dependencies": [],
        "worker": str(runtime.worker_path),
        "worker_sha256": sha(runtime.worker_path),
        "config": str(config_path),
        "config_sha256": sha(config_path),
        "sources": {
            "steering": sha(ROOT / "python/irisu_pointer/steering.py"),
            "gate": sha(ROOT / "python/irisu_pointer/shot_necessity.py"),
            "runner": sha(Path(__file__).resolve()),
        },
    }
    (args.run_root / "source-identity.json").write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
    actions: list[int] = []
    checkpoints = []
    reasons: Counter[str] = Counter()
    attempted = kept = suppressed = 0
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": maximum_ticks + int(config["gate"]["probe_ticks"])}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info["seed"]) != seed:
            raise RuntimeError("reset seed differs")
        checkpoints.append(checkpoint(env, observation))
        for _ in range(int(config["replay_warmup_ticks"])):
            actions.append(0)
            observation, _, terminated, truncated, _ = env.step(Action.wait(1))
        next_checkpoint = 5_000
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            before = copy.deepcopy(expert)
            decision = expert.predict(observation)
            if decision.is_shot:
                attempted += 1
                verdict = gate.evaluate(env, observation, before, expert, decision)
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    expert = before
                    decision = gate.wait_decision(verdict.reason)
            for macro in decision.primitive_actions():
                kind = int(macro.kind)
                duration = int(macro.wait_ticks) if kind == 0 else 1
                for _ in range(duration):
                    primitive = Action.wait(1) if kind == 0 else macro
                    actions.append(encode(primitive))
                    observation, _, terminated, truncated, _ = env.step(primitive)
                    tick = int(observation["tick"])
                    if tick >= next_checkpoint:
                        checkpoints.append(checkpoint(env, observation))
                        print(json.dumps({"tick": tick, "score": int(observation["score"]), "gauge": int(observation["gauge"]), "elapsed_seconds": time.monotonic() - started}, sort_keys=True), flush=True)
                        next_checkpoint += 5_000
                    if terminated or truncated or tick >= maximum_ticks:
                        break
                if terminated or truncated or int(observation["tick"]) >= maximum_ticks:
                    break
        final = checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        provenance = session.provenance_manifest
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
    trace = b"".join(WORD.pack(value) for value in actions)
    trace_path = args.run_root / "full-game.u32le"
    trace_path.write_bytes(trace)
    replay = HEADER.pack(seed, final["level"], final["score"], final["highest_chain"], 0) + bytes(32) + trace
    replay_path = args.run_root / "checkpoint-free-expert-gate.rpy"
    replay_path.write_bytes(replay)
    terminal = bool(final["terminated"])
    result = {
        "version": "exact-50k-expert-gate-result-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "checkpoint_free": True,
        "portable_model_dependencies": [],
        "seed": seed,
        "score": final["score"],
        "target_reached": terminal and final["score"] >= int(config["target_score"]),
        "terminal": terminal,
        "censored": not terminal,
        "survival_ticks": final["tick"],
        "attempted_shots": attempted,
        "kept_shots": kept,
        "suppressed_shots": suppressed,
        "gate_reasons": dict(sorted(reasons.items())),
        "trace": trace_path.name,
        "trace_sha256": hashlib.sha256(trace).hexdigest(),
        "replay": replay_path.name,
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
        "checkpoints": checkpoints,
        "final": final,
        "final_snapshot_sha256": snapshot_sha,
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    (args.run_root / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
