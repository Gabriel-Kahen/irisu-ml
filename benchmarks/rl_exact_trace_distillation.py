#!/usr/bin/env python3
"""Distill a verified exact trace into a checkpoint-free closed-loop policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/rl/experiments/exact-trace-distillation-v1.toml"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-trace-distill-46651-20260810-001"
TEST_SOURCE = ROOT / "tests/test_exact_trace_distillation.py"
EVALUATOR = ROOT / "tools/evaluate-rpy.py"
MODEL_MAGIC = b"IRTDV1\0\0"
MODEL_HEADER = struct.Struct("<8sII")
MODEL_RECORD = struct.Struct("<Q32sI")
ACTION_WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def with_sha(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def verify_sha(value: Mapping[str, object], label: str) -> None:
    supplied = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if supplied != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise RuntimeError(f"{label} self-hash differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: Path, value: Mapping[str, object]) -> None:
    write_new(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def write_json_new_or_match(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    write_json_new(path, value)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return item()
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported public observation value: {type(value).__name__}")


def public_observation_sha256(observation: Mapping[str, Any]) -> bytes:
    """Hash only the public observation supplied to a policy."""

    return hashlib.sha256(canonical_bytes(_plain(observation))).digest()


def stable_acceptance_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the evaluator worker PID, which is evidence-neutral process noise."""

    result = json.loads(json.dumps(report, allow_nan=False))
    clone_build = result.get("clone_build", {})
    if not isinstance(clone_build, dict) or not isinstance(
        clone_build.pop("worker_pid", None), int
    ):
        raise RuntimeError("production replay report lacks an exact worker PID")
    clone_build["worker_process_attested"] = True
    return result


def encode_action(action: Action) -> int:
    kind = int(action.kind)
    if kind == 0:
        if int(action.wait_ticks) != 1:
            raise ValueError("distilled actions must advance exactly one tick")
        return 0
    x, y = int(action.cursor_x), int(action.cursor_y)
    if (
        kind not in (1, 2, 3)
        or float(action.cursor_x) != x
        or float(action.cursor_y) != y
        or not 0 <= x <= 1023
        or not 0 <= y <= 511
    ):
        raise ValueError("action is not representable as one replay word")
    return (y << 12) | (x << 2) | kind


def decode_action(word: int) -> Action:
    if isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= 0x1FFFFF:
        raise ValueError("action word is outside the replay encoding")
    buttons = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if buttons == 0:
        if word != 0:
            raise ValueError("wait action word carries inaccessible cursor state")
        return Action.wait(1)
    if buttons == 1:
        return Action.weak(x, y)
    if buttons == 2:
        return Action.strong(x, y)
    return Action.both(x, y)


def checkpoint(env: object, observation: Mapping[str, Any]) -> dict[str, object]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "level": int(observation["level"]),
        "clears": int(observation.get("qualifying_clear_count", 0)),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "terminated": bool(observation.get("terminated", False)),
        "truncated": bool(observation.get("truncated", False)),
        "state_u64": f"0x{int(getattr(env, 'state_hash')()) & 0xffffffffffffffff:016x}",
    }


def load_config(path: Path) -> dict[str, Any]:
    value = tomllib.loads(path.read_text())
    required = {
        "version": "exact-trace-distillation-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "policy_kind": "tick-conditioned-public-observation-table",
        "fallback": "none",
        "checkpoint_dependencies": [],
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("trace distillation config weakens the exact-only contract")
    for key in ("target_score", "checkpoint_stride_ticks"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    identity = value.get("runtime", {}).get("identity")
    if identity != "configs/rl/runtime/exact-worker-2026-07-21.json":
        raise ValueError("trace distillation runtime identity differs")
    teacher = value.get("teacher")
    if not isinstance(teacher, dict) or set(teacher) != {"unit", "trace", "replay", "acceptance"}:
        raise ValueError("trace teacher schema differs")
    value["paths"] = {
        key: (ROOT / str(item)).resolve(strict=True) for key, item in teacher.items()
    }
    return value


def validate_teacher(config: Mapping[str, Any], worker_sha256: str) -> dict[str, Any]:
    paths = config["paths"]
    unit = read_json(paths["unit"])
    verify_sha(unit, "exact teacher unit")
    trace = paths["trace"].read_bytes()
    replay = paths["replay"].read_bytes()
    acceptance = read_json(paths["acceptance"])
    status = acceptance.get("status", {})
    outcome = acceptance.get("outcome", {})
    runtime = acceptance.get("exact_training_runtime", {})
    if (
        unit.get("physics_backend") != "exact"
        or unit.get("terminal") is not True
        or unit.get("censored") is not False
        or unit.get("trace_sha256") != hashlib.sha256(trace).hexdigest()
        or int(unit.get("action_count", -1)) * ACTION_WORD.size != len(trace)
        or int(unit.get("survival_ticks", -1)) != int(unit.get("action_count", -2))
        or status.get("accepted") is not True
        or outcome.get("score", {}).get("clone_final") != unit.get("score")
        or outcome.get("level", {}).get("clone_final") != unit.get("level")
        or outcome.get("highest_chain", {}).get("clone_final") != unit.get("highest_chain")
        or acceptance.get("hashes", {}).get("replay_sha256")
        != hashlib.sha256(replay).hexdigest()
        or runtime.get("identity", {}).get("worker_sha256") != worker_sha256
    ):
        raise RuntimeError("teacher is not replay-accepted exact terminal evidence")
    return unit


def source_identity(config_path: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    config = load_config(config_path)
    unit = validate_teacher(config, runtime.identity.worker_sha256)
    files = (
        Path(__file__).resolve(),
        TEST_SOURCE,
        EVALUATOR,
        config_path,
        ROOT / "python/irisu_rl/exact_training_runtime.py",
        ROOT / "configs/rl/runtime/exact-worker-2026-07-21.json",
        runtime.worker_path,
        *config["paths"].values(),
    )
    return with_sha(
        {
            "schema": "irisu-exact-trace-distillation-source-v1",
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "training_paradigm": "full-trajectory behavioral cloning",
            "policy_kind": config["policy_kind"],
            "fallback": "none",
            "checkpoint_dependencies": [],
            "teacher_unit_sha256": unit["sha256"],
            "exact_identity": runtime.identity.manifest(),
            "files": {str(item): sha256_file(item) for item in files},
        }
    )


def initialize(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(run_root)
    identity = source_identity(config_path, runtime)
    config = load_config(config_path)
    unit = validate_teacher(config, runtime.identity.worker_sha256)
    plan = with_sha(
        {
            "schema": "irisu-exact-trace-distillation-plan-v1",
            "source_identity_sha256": identity["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "seed": unit["seed"],
            "teacher_score": unit["score"],
            "teacher_ticks": unit["survival_ticks"],
            "target_score": config["target_score"],
            "success_requires_natural_terminal": True,
            "policy": (
                "one exact teacher action per tick, keyed by tick and SHA-256 of "
                "the complete public observation; unseen states fail closed"
            ),
            "runtime_dependencies": ["pinned exact worker"],
            "checkpoint_dependencies": [],
            "fallback": "none",
        }
    )
    run_root.mkdir(parents=True)
    write_json_new(run_root / "source-identity.json", identity)
    write_json_new(run_root / "plan.json", plan)
    return plan


def validate(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = read_json(run_root / "source-identity.json")
    plan = read_json(run_root / "plan.json")
    verify_sha(identity, "distillation source identity")
    verify_sha(plan, "distillation plan")
    config = load_config(config_path)
    unit = validate_teacher(config, runtime.identity.worker_sha256)
    if (
        identity != source_identity(config_path, runtime)
        or plan.get("source_identity_sha256") != identity["sha256"]
        or plan.get("seed") != unit["seed"]
    ):
        raise RuntimeError("frozen trace distillation identity differs")
    return identity, plan, config, unit


class ExactTraceTablePolicy:
    """A deterministic public-observation table with no fallback behavior."""

    def __init__(self, payload: bytes):
        if len(payload) < MODEL_HEADER.size:
            raise ValueError("distilled policy is truncated")
        magic, seed, count = MODEL_HEADER.unpack_from(payload)
        if magic != MODEL_MAGIC or len(payload) != MODEL_HEADER.size + count * MODEL_RECORD.size:
            raise ValueError("distilled policy layout differs")
        self.seed = seed
        self.count = count
        self._payload = payload
        self._cursor = 0
        self.last_word: int | None = None

    @property
    def complete(self) -> bool:
        return self._cursor == self.count

    def predict(self, observation: Mapping[str, Any]) -> Action:
        if self._cursor >= self.count:
            raise RuntimeError("distilled policy has no fallback after its teacher horizon")
        offset = MODEL_HEADER.size + self._cursor * MODEL_RECORD.size
        tick, expected_digest, word = MODEL_RECORD.unpack_from(self._payload, offset)
        actual_tick = int(observation["tick"])
        actual_digest = public_observation_sha256(observation)
        if actual_tick != tick or actual_digest != expected_digest:
            raise RuntimeError(
                f"distilled policy encountered an unseen public state at tick {actual_tick}"
            )
        self._cursor += 1
        self.last_word = word
        return decode_action(word)


def distill(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    identity, plan, config, unit = validate(run_root, config_path, runtime)
    manifest_path = run_root / "policy.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_sha(manifest, "distilled policy manifest")
        model = run_root / str(manifest["model_file"])
        if sha256_file(model) != manifest["model_sha256"]:
            raise RuntimeError("distilled policy bytes changed")
        return manifest
    trace = config["paths"]["trace"].read_bytes()
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    records = bytearray()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": int(unit["simulation_ceiling_ticks"])}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=int(unit["seed"]))
        if int(info.get("seed", -1)) != int(unit["seed"]):
            raise RuntimeError("teacher reset seed differs")
        for index, word in enumerate(words):
            if int(observation["tick"]) != index:
                raise RuntimeError("teacher trace cadence differs")
            records.extend(
                MODEL_RECORD.pack(index, public_observation_sha256(observation), word)
            )
            observation, _reward, terminated, truncated, _info = env.step(
                decode_action(word)
            )
            if (terminated or truncated) and index + 1 != len(words):
                raise RuntimeError("teacher trace contains actions after terminal")
        final = checkpoint(env, observation)
        snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if (
        final != unit["final"]
        or snapshot_sha256 != unit["final_snapshot_sha256"]
        or final["terminated"] is not True
        or final["truncated"] is not False
    ):
        raise RuntimeError("exact teacher trace reconstruction differs")
    model = MODEL_HEADER.pack(MODEL_MAGIC, int(unit["seed"]), len(words)) + records
    model_path = run_root / "exact-public-state-table.irtd"
    write_new(model_path, model)
    manifest = with_sha(
        {
            "schema": "irisu-exact-trace-table-policy-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "training_paradigm": "full-trajectory behavioral cloning",
            "policy_kind": "tick-conditioned-public-observation-table",
            "observation_key": "sha256(canonical complete public observation)",
            "fallback": "none; unseen observations raise",
            "checkpoint_dependencies": [],
            "runtime_dependencies": ["pinned exact worker"],
            "model_file": model_path.name,
            "model_sha256": hashlib.sha256(model).hexdigest(),
            "model_bytes": len(model),
            "training_examples": len(words),
            "seed": unit["seed"],
            "teacher_unit_sha256": unit["sha256"],
            "teacher_final": final,
            "exact_runtime": provenance,
        }
    )
    write_json_new(manifest_path, manifest)
    return manifest


def _load_policy(run_root: Path) -> tuple[ExactTraceTablePolicy, dict[str, Any]]:
    manifest = read_json(run_root / "policy.json")
    verify_sha(manifest, "distilled policy manifest")
    model_path = run_root / str(manifest["model_file"])
    payload = model_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["model_sha256"]:
        raise RuntimeError("distilled policy bytes changed")
    return ExactTraceTablePolicy(payload), manifest


def evaluate(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    identity, plan, config, unit = validate(run_root, config_path, runtime)
    output_path = run_root / "closed-loop-evaluation.json"
    if output_path.exists():
        result = read_json(output_path)
        verify_sha(result, "closed-loop evaluation")
        _load_policy(run_root)
        for file_key, hash_key in (
            ("trace_file", "trace_sha256"),
            ("replay_file", "replay_sha256"),
        ):
            if sha256_file(run_root / str(result[file_key])) != result[hash_key]:
                raise RuntimeError(f"closed-loop {file_key} bytes changed")
        return result
    policy, manifest = _load_policy(run_root)
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    stride = int(config["checkpoint_stride_ticks"])
    next_checkpoint = stride
    with runtime.open_env(
        simulation_config={"max_episode_ticks": int(unit["simulation_ceiling_ticks"])}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=policy.seed)
        if int(info.get("seed", -1)) != policy.seed:
            raise RuntimeError("closed-loop reset seed differs")
        checkpoints.append(checkpoint(env, observation))
        terminated = truncated = False
        while not policy.complete and not (terminated or truncated):
            action = policy.predict(observation)
            assert policy.last_word is not None
            actions.append(policy.last_word)
            observation, _reward, terminated, truncated, _info = env.step(action)
            if int(observation["tick"]) >= next_checkpoint:
                checkpoints.append(checkpoint(env, observation))
                next_checkpoint += stride
        final = checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if (
        not policy.complete
        or not terminated
        or truncated
        or final != unit["final"]
        or snapshot_sha256 != unit["final_snapshot_sha256"]
    ):
        raise RuntimeError("distilled policy did not close the exact teacher trajectory")
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "closed-loop.u32le"
    write_new(trace_path, trace)
    replay = (
        REPLAY_HEADER.pack(
            policy.seed,
            int(final["level"]),
            int(final["score"]),
            int(final["highest_chain"]),
            0,
        )
        + bytes(32)
        + trace
    )
    replay_path = run_root / "closed-loop.rpy"
    write_new(replay_path, replay)
    result = with_sha(
        {
            "schema": "irisu-exact-trace-distillation-closed-loop-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "policy_manifest_sha256": manifest["sha256"],
            "model_sha256": manifest["model_sha256"],
            "physics_backend": "exact",
            "policy_runtime_imports": ["irisu_env.Action"],
            "checkpoint_dependencies": [],
            "fallback_invocations": 0,
            "seed": policy.seed,
            "score": final["score"],
            "target_score": config["target_score"],
            "target_reached": int(final["score"]) >= int(config["target_score"]),
            "survival_ticks": final["tick"],
            "terminal": final["terminated"],
            "truncated": final["truncated"],
            "level": final["level"],
            "highest_chain": final["highest_chain"],
            "action_count": len(actions),
            "trace_file": trace_path.name,
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "replay_file": replay_path.name,
            "replay_sha256": hashlib.sha256(replay).hexdigest(),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha256,
            "exact_runtime": provenance,
        }
    )
    write_json_new(output_path, result)
    return result


def verify(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    identity, plan, _config, unit = validate(run_root, config_path, runtime)
    evaluation = evaluate(run_root, config_path, runtime)
    trace = (run_root / str(evaluation["trace_file"])).read_bytes()
    replay_path = run_root / str(evaluation["replay_file"])
    replay = replay_path.read_bytes()
    if (
        hashlib.sha256(trace).hexdigest() != evaluation["trace_sha256"]
        or hashlib.sha256(replay).hexdigest() != evaluation["replay_sha256"]
    ):
        raise RuntimeError("closed-loop evidence bytes changed")
    expected = {int(item["tick"]): item for item in evaluation["checkpoints"]}
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    with runtime.open_env(
        simulation_config={"max_episode_ticks": int(unit["simulation_ceiling_ticks"])}
    ) as session:
        env = session.environment
        observation, _info = env.reset(seed=int(evaluation["seed"]))
        if checkpoint(env, observation) != expected[0]:
            raise RuntimeError("replay initial checkpoint differs")
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                decode_action(word)
            )
            tick = int(observation["tick"])
            if tick in expected and checkpoint(env, observation) != expected[tick]:
                raise RuntimeError(f"replay checkpoint differs at tick {tick}")
        final = checkpoint(env, observation)
        snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if final != evaluation["final"] or snapshot_sha256 != evaluation["final_snapshot_sha256"]:
        raise RuntimeError("independent exact replay closure differs")
    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            str(replay_path),
            "--worker",
            str(runtime.worker_path),
            "--purpose",
            "promotion",
            "--compact",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    acceptance = stable_acceptance_report(json.loads(completed.stdout))
    if acceptance.get("status", {}).get("accepted") is not True:
        raise RuntimeError("production exact replay evaluator rejected the replay")
    acceptance_path = run_root / "replay-acceptance.json"
    write_json_new_or_match(acceptance_path, acceptance)
    result = with_sha(
        {
            "schema": "irisu-exact-trace-distillation-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "closed_loop_evaluation_sha256": evaluation["sha256"],
            "physics_backend": "exact",
            "closed_loop_policy_verified": True,
            "independent_trace_reexecution_verified": True,
            "production_replay_acceptance_verified": True,
            "naturally_terminated": final["terminated"] and not final["truncated"],
            "score": final["score"],
            "target_score": evaluation["target_score"],
            "target_reached": evaluation["target_reached"],
            "survival_ticks": final["tick"],
            "model_sha256": evaluation["model_sha256"],
            "replay_sha256": evaluation["replay_sha256"],
            "replay_acceptance_sha256": sha256_file(acceptance_path),
            "exact_runtime": provenance,
        }
    )
    write_json_new_or_match(run_root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "distill", "evaluate", "verify", "run-all"))
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be an explicit absolute path")
    runtime = ExactTrainingRuntime(args.worker)
    config_path = args.config.resolve(strict=True)
    if args.command == "init":
        result = initialize(args.run_root, config_path, runtime)
    elif args.command == "distill":
        result = distill(args.run_root, config_path, runtime)
    elif args.command == "evaluate":
        result = evaluate(args.run_root, config_path, runtime)
    elif args.command == "verify":
        result = verify(args.run_root, config_path, runtime)
    else:
        if not args.run_root.exists():
            initialize(args.run_root, config_path, runtime)
        distill(args.run_root, config_path, runtime)
        evaluate(args.run_root, config_path, runtime)
        result = verify(args.run_root, config_path, runtime)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
