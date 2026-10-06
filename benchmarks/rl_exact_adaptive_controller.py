#!/usr/bin/env python3
"""Exact full-game hysteresis controller derived from evolution evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import struct
import subprocess
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_controller_evolution as evo
import rl_r3k_sustainable_v3 as screen
from irisu_pointer.shot_necessity import ExactWaitDominanceGate, WaitDominanceConfig
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


DEFAULT_CONFIG = ROOT / "configs/rl/experiments/exact-adaptive-controller-v1.toml"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-50k-evo-adaptive-20260810-001"
TEST_SOURCE = ROOT / "tests/test_exact_adaptive_controller.py"
ACTION_WORD = struct.Struct("<I")


@dataclass(slots=True)
class GaugeHysteresis:
    enter: int
    exit: int
    survival_mode: bool = False

    def __post_init__(self) -> None:
        if self.enter < 1 or self.exit <= self.enter:
            raise ValueError("gauge hysteresis thresholds are malformed")

    def update(self, gauge: int) -> str:
        if not self.survival_mode and gauge < self.enter:
            self.survival_mode = True
        elif self.survival_mode and gauge > self.exit:
            self.survival_mode = False
        return "survival" if self.survival_mode else "score"


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = tomllib.loads(payload.decode())
    if value.get("version") != "exact-adaptive-controller-v1" or value.get("physics_backend") != "exact":
        raise ValueError("adaptive config must be exact-only")
    for key in (
        "seed", "max_ticks", "replay_warmup_ticks", "target_score",
        "enter_survival_gauge", "exit_survival_gauge",
    ):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    GaugeHysteresis(value["enter_survival_gauge"], value["exit_survival_gauge"])
    for mode in ("score_mode", "survival_mode"):
        WaitDominanceConfig(**value[mode])
    return value, payload


def source_identity(config: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    files = (
        Path(__file__).resolve(), TEST_SOURCE, config,
        Path(evo.__file__).resolve(),
        ROOT / "python/irisu_pointer/shot_necessity.py",
        ROOT / "python/irisu_rl/exact_training_runtime.py",
        screen.BASE_CHECKPOINT, screen.CAMPAIGN_SOURCE,
    )
    return evo.evidence.with_sha(
        {
            "schema": "irisu-exact-adaptive-controller-source-v1",
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "physics_backend": "exact",
            "model_lineage": "portable-trained frozen-v5; controller optimized exact closed-loop",
            "exact_identity": runtime.identity.manifest(),
            "files": {str(path): evo.evidence.sha256_file(path) for path in files},
        }
    )


def initialize(run_root: Path, config: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"run path already exists: {run_root}")
    values, payload = load_config(config)
    identity = source_identity(config, runtime)
    plan = evo.evidence.with_sha(
        {
            "schema": "irisu-exact-adaptive-controller-plan-v1",
            "source_identity_sha256": identity["sha256"],
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "physics_backend": "exact",
            "model_lineage": identity["model_lineage"],
            "derivation": "96-tick score gate with 128-tick survival gate below exact observed reserve",
            "config": values,
        }
    )
    run_root.mkdir(parents=True)
    evo.evidence.write_new(run_root / "source-identity.json", identity)
    evo.evidence.write_new(run_root / "plan.json", plan)
    return plan


def validate(
    run_root: Path, config: Path, runtime: ExactTrainingRuntime
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = evo.evidence.read_json(run_root / "source-identity.json")
    plan = evo.evidence.read_json(run_root / "plan.json")
    evo.evidence.verify_self_hash(identity, "adaptive source")
    evo.evidence.verify_self_hash(plan, "adaptive plan")
    values, payload = load_config(config)
    if (
        identity != source_identity(config, runtime)
        or plan["source_identity_sha256"] != identity["sha256"]
        or plan["config_sha256"] != hashlib.sha256(payload).hexdigest()
        or plan["config"] != values
    ):
        raise RuntimeError("adaptive frozen identity differs")
    return identity, plan, values


def _usage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_SELF)


def run(run_root: Path, config: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    identity, plan, values = validate(run_root, config, runtime)
    path = run_root / "unit.json"
    if path.exists():
        result = evo.evidence.read_json(path)
        evo.evidence.verify_self_hash(result, "adaptive unit")
        return result
    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    seed = int(values["seed"])
    policy.reset(seed)
    modes = {
        name: ExactWaitDominanceGate(
            lambda decision: screen._primitive_actions(core, decision),
            config=WaitDominanceConfig(**values[f"{name}_mode"]),
        )
        for name in ("score", "survival")
    }
    hysteresis = GaugeHysteresis(
        int(values["enter_survival_gauge"]), int(values["exit_survival_gauge"])
    )
    max_ticks = int(values["max_ticks"])
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    switches: list[dict[str, object]] = []
    mode_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    attempted = kept = suppressed = 0
    previous_mode = "score"
    started = time.monotonic()
    usage_before = _usage()
    terminated = truncated = False
    ceiling = max_ticks + max(gate.config.probe_ticks for gate in modes.values())
    with runtime.open_env(simulation_config={"max_episode_ticks": ceiling}) as session:
        env = session.environment
        observation, _info = env.reset(seed=seed)
        checkpoints.append(evo.evidence.checkpoint(env, observation))
        for _ in range(int(values["replay_warmup_ticks"])):
            action = core.JOINT.Action.wait(1)
            actions.append(evo.evidence.encode_action(action))
            observation, _reward, terminated, truncated, _info = env.step(action)
        while int(observation["tick"]) < max_ticks and not (terminated or truncated):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                attempted += 1
                mode = hysteresis.update(int(observation["gauge"]))
                if mode != previous_mode:
                    switches.append(
                        {
                            "tick": int(observation["tick"]),
                            "gauge": int(observation["gauge"]),
                            "from": previous_mode,
                            "to": mode,
                        }
                    )
                    previous_mode = mode
                mode_counts[mode] += 1
                gate = modes[mode]
                verdict = gate.evaluate(env, observation, before, policy, decision)
                reasons[f"{mode}:{verdict.reason}"] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = gate.wait_decision(verdict.reason)
            for action in screen._primitive_actions(core, decision):
                kind = int(action.kind)
                duration = int(action.wait_ticks) if kind == 0 else 1
                duration = min(duration, max_ticks - int(observation["tick"]))
                for _ in range(duration):
                    primitive = core.JOINT.Action.wait(1) if kind == 0 else action
                    actions.append(evo.evidence.encode_action(primitive))
                    observation, _reward, terminated, truncated, _info = env.step(primitive)
                    tick = int(observation["tick"])
                    if tick % 5_000 == 0:
                        checkpoints.append(evo.evidence.checkpoint(env, observation))
                        print(
                            json.dumps(
                                {
                                    "tick": tick, "score": int(observation["score"]),
                                    "gauge": int(observation["gauge"]),
                                    "mode": previous_mode,
                                    "switches": len(switches),
                                    "wall_seconds": time.monotonic() - started,
                                },
                                sort_keys=True,
                            ), flush=True,
                        )
                    if terminated or truncated or tick >= max_ticks:
                        break
                if terminated or truncated or int(observation["tick"]) >= max_ticks:
                    break
        final = evo.evidence.checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "trace.u32le"
    evo.evidence.write_bytes_new(trace_path, trace)
    usage_after = _usage()
    terminal = bool(final["terminated"])
    result = evo.evidence.with_sha(
        {
            "schema": "irisu-exact-adaptive-controller-unit-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "model_lineage": plan["model_lineage"],
            "exact_runtime": provenance,
            "seed": seed,
            "simulation_ceiling_ticks": ceiling,
            "terminal": terminal,
            "censored": int(final["tick"]) >= max_ticks and not terminal,
            "target_score": values["target_score"],
            "target_met": terminal and int(final["score"]) >= int(values["target_score"]),
            "score": int(final["score"]),
            "survival_ticks": int(final["tick"]),
            "level": int(final["level"]),
            "clears": int(final["clears"]),
            "highest_chain": int(final["highest_chain"]),
            "attempted_shots": attempted,
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "mode_counts": dict(mode_counts),
            "switches": switches,
            "gate_reasons": dict(sorted(reasons.items())),
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
                "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
                "maximum_rss_kib": usage_after.ru_maxrss,
            },
        }
    )
    evo.evidence.write_new(path, result)
    return result


def verify(run_root: Path, config: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    identity, plan, _values = validate(run_root, config, runtime)
    row = evo.evidence.read_json(run_root / "unit.json")
    evo.evidence.verify_self_hash(row, "adaptive unit")
    trace = (run_root / row["trace_file"]).read_bytes()
    if hashlib.sha256(trace).hexdigest() != row["trace_sha256"]:
        raise RuntimeError("adaptive trace hash differs")
    core, _campaign = screen._load_external()
    expected = {int(item["tick"]): item for item in row["checkpoints"]}
    with runtime.open_env(
        simulation_config={"max_episode_ticks": int(row["simulation_ceiling_ticks"])}
    ) as session:
        env = session.environment
        observation, _info = env.reset(seed=int(row["seed"]))
        if evo.evidence.checkpoint(env, observation) != expected[0]:
            raise RuntimeError("adaptive initial replay differs")
        words = [word for (word,) in struct.iter_unpack("<I", trace)]
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                evo.evidence.decode_action(core, word)
            )
            tick = int(observation["tick"])
            if tick in expected and evo.evidence.checkpoint(env, observation) != expected[tick]:
                raise RuntimeError(f"adaptive replay differs at tick {tick}")
        final = evo.evidence.checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
    if final != row["final"] or snapshot_sha != row["final_snapshot_sha256"]:
        raise RuntimeError("adaptive replay final closure differs")
    result = evo.evidence.with_sha(
        {
            "schema": "irisu-exact-adaptive-controller-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "unit_sha256": row["sha256"],
            "verified": True,
        }
    )
    evo.evidence.write_new(run_root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "verify"))
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be absolute")
    runtime = ExactTrainingRuntime(args.worker)
    config = args.config.resolve(strict=True)
    if args.command == "init":
        value = initialize(args.run_root, config, runtime)
    elif args.command == "run":
        value = run(args.run_root, config, runtime)
    else:
        value = verify(args.run_root, config, runtime)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
