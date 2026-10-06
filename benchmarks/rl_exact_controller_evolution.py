#!/usr/bin/env python3
"""Exact full-game controller evolution with replay-closed evidence."""

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3k_sustainable_v3 as screen
import rl_r3m_shot_restraint as evidence
from irisu_pointer.shot_necessity import ExactWaitDominanceGate, WaitDominanceConfig
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


DEFAULT_CONFIG = ROOT / "configs/rl/experiments/exact-controller-evolution-v1.toml"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-50k-evo-20260810-001"
TEST_SOURCE = ROOT / "tests/test_exact_controller_evolution.py"
ACTION_WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    generation: int
    gate: bool
    parent: str | None = None
    probe_ticks: int | None = None
    wait_ticks: int | None = None
    gauge_advantage: int | None = None

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "Candidate":
        expected = {"id", "generation", "gate"}
        optional = {"parent", "probe_ticks", "wait_ticks", "gauge_advantage"}
        if not isinstance(value, Mapping) or not expected <= set(value) <= expected | optional:
            raise ValueError("candidate schema differs")
        candidate = cls(**value)  # type: ignore[arg-type]
        if (
            not candidate.id
            or isinstance(candidate.generation, bool)
            or candidate.generation < 0
            or type(candidate.gate) is not bool
        ):
            raise ValueError("candidate identity is malformed")
        parameters = (
            candidate.probe_ticks,
            candidate.wait_ticks,
            candidate.gauge_advantage,
        )
        if candidate.gate:
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in parameters):
                raise ValueError("gated candidate parameters must be positive integers")
            WaitDominanceConfig(*parameters)  # type: ignore[arg-type]
        elif any(item is not None for item in parameters) or candidate.parent is not None:
            raise ValueError("ungated control cannot carry gate parameters or a parent")
        return candidate

    def gate_config(self) -> WaitDominanceConfig | None:
        if not self.gate:
            return None
        return WaitDominanceConfig(
            probe_ticks=int(self.probe_ticks),
            wait_ticks=int(self.wait_ticks),
            gauge_advantage=int(self.gauge_advantage),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.id,
            "generation": self.generation,
            "gate": self.gate,
            **({} if self.parent is None else {"parent": self.parent}),
            **(
                {}
                if not self.gate
                else {
                    "probe_ticks": self.probe_ticks,
                    "wait_ticks": self.wait_ticks,
                    "gauge_advantage": self.gauge_advantage,
                }
            ),
        }


def load_config(path: Path) -> tuple[dict[str, Any], tuple[Candidate, ...], bytes]:
    payload = path.read_bytes()
    value = tomllib.loads(payload.decode())
    required = {
        "version": "exact-controller-evolution-v1",
        "physics_backend": "exact",
        "worker_identity": "configs/rl/runtime/exact-worker-2026-07-21.json",
        "objective": "maximum exact natural-terminal score; censored runs are ineligible",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("evolution config weakens the exact-only contract")
    for key in ("seed", "max_ticks", "replay_warmup_ticks", "target_score"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    candidates = tuple(Candidate.parse(item) for item in value.get("candidates", ()))
    if len(candidates) < 2 or len({item.id for item in candidates}) != len(candidates):
        raise ValueError("evolution requires distinct control and treatment candidates")
    ids = {item.id for item in candidates}
    if sum(not item.gate for item in candidates) != 1 or any(
        item.parent is not None and item.parent not in ids for item in candidates
    ):
        raise ValueError("candidate genealogy is malformed")
    return value, candidates, payload


def source_identity(config_path: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        TEST_SOURCE,
        config_path,
        ROOT / "python/irisu_pointer/shot_necessity.py",
        ROOT / "python/irisu_rl/exact_training_runtime.py",
        ROOT / "configs/rl/runtime/exact-worker-2026-07-21.json",
        screen.BASE_CHECKPOINT,
        screen.CAMPAIGN_SOURCE,
    )
    return evidence.with_sha(
        {
            "schema": "irisu-exact-controller-evolution-source-v1",
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "physics_backend": "exact",
            "exact_identity": runtime.identity.manifest(),
            "files": {str(path): evidence.sha256_file(path) for path in files},
        }
    )


def initialize(run_root: Path, config_path: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"run path already exists: {run_root}")
    config, candidates, payload = load_config(config_path)
    identity = source_identity(config_path, runtime)
    plan = evidence.with_sha(
        {
            "schema": "irisu-exact-controller-evolution-plan-v1",
            "run_id": run_root.name,
            "source_identity_sha256": identity["sha256"],
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "physics_backend": "exact",
            "seed": config["seed"],
            "max_ticks": config["max_ticks"],
            "replay_warmup_ticks": config["replay_warmup_ticks"],
            "target_score": config["target_score"],
            "objective": config["objective"],
            "candidates": [item.manifest() for item in candidates],
        }
    )
    run_root.mkdir(parents=True)
    evidence.write_new(run_root / "source-identity.json", identity)
    evidence.write_new(run_root / "plan.json", plan)
    return plan


def validate(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> tuple[dict[str, Any], dict[str, Any], tuple[Candidate, ...]]:
    identity = evidence.read_json(run_root / "source-identity.json")
    plan = evidence.read_json(run_root / "plan.json")
    evidence.verify_self_hash(identity, "evolution source identity")
    evidence.verify_self_hash(plan, "evolution plan")
    config, candidates, payload = load_config(config_path)
    if (
        identity != source_identity(config_path, runtime)
        or plan.get("source_identity_sha256") != identity["sha256"]
        or plan.get("config_sha256") != hashlib.sha256(payload).hexdigest()
        or plan.get("candidates") != [item.manifest() for item in candidates]
        or plan.get("seed") != config["seed"]
    ):
        raise RuntimeError("frozen evolution identity differs")
    return identity, plan, candidates


def unit_path(run_root: Path, candidate: Candidate) -> Path:
    return run_root / "units" / f"{candidate.id}.json"


def _usage() -> dict[str, float | int]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_seconds": own.ru_utime + children.ru_utime,
        "system_cpu_seconds": own.ru_stime + children.ru_stime,
        "maximum_rss_kib": max(own.ru_maxrss, children.ru_maxrss),
    }


def run_candidate(
    run_root: Path,
    config_path: Path,
    runtime: ExactTrainingRuntime,
    candidate_id: str,
) -> dict[str, object]:
    identity, plan, candidates = validate(run_root, config_path, runtime)
    candidate = next((item for item in candidates if item.id == candidate_id), None)
    if candidate is None:
        raise ValueError(f"unknown candidate: {candidate_id}")
    path = unit_path(run_root, candidate)
    if path.exists():
        result = evidence.read_json(path)
        evidence.verify_self_hash(result, "evolution unit")
        return result
    core, campaign = screen._load_external()
    seed = int(plan["seed"])
    max_ticks = int(plan["max_ticks"])
    warmup = int(plan["replay_warmup_ticks"])
    policy = campaign.POLICY_FACTORY()
    policy.reset(seed)
    gate_config = candidate.gate_config()
    gate = (
        None
        if gate_config is None
        else ExactWaitDominanceGate(
            lambda decision: screen._primitive_actions(core, decision),
            config=gate_config,
        )
    )
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    attempted = kept = suppressed = 0
    started = time.monotonic()
    usage_before = _usage()
    terminated = truncated = False
    simulation_ceiling = max_ticks + (0 if gate_config is None else gate_config.probe_ticks)
    with runtime.open_env(
        simulation_config={"max_episode_ticks": simulation_ceiling}
    ) as exact_session:
        env = exact_session.environment
        exact_runtime = exact_session.provenance_manifest
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("evolution reset seed differs")
        checkpoints.append(evidence.checkpoint(env, observation))
        for _ in range(warmup):
            action = core.JOINT.Action.wait(1)
            actions.append(evidence.encode_action(action))
            observation, _reward, terminated, truncated, _info = env.step(action)
        while int(observation["tick"]) < max_ticks and not (terminated or truncated):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if gate is not None and getattr(decision, "is_shot", False):
                attempted += 1
                verdict = gate.evaluate(env, observation, before, policy, decision)
                reasons[verdict.reason] += 1
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
                    actions.append(evidence.encode_action(primitive))
                    observation, _reward, terminated, truncated, _info = env.step(primitive)
                    tick = int(observation["tick"])
                    if tick % 5_000 == 0:
                        checkpoints.append(evidence.checkpoint(env, observation))
                        print(
                            json.dumps(
                                {
                                    "candidate": candidate.id,
                                    "tick": tick,
                                    "score": int(observation["score"]),
                                    "gauge": int(observation["gauge"]),
                                    "wall_seconds": time.monotonic() - started,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if terminated or truncated or tick >= max_ticks:
                        break
                if terminated or truncated or int(observation["tick"]) >= max_ticks:
                    break
        final = evidence.checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "traces" / f"{candidate.id}.u32le"
    evidence.write_bytes_new(trace_path, trace)
    terminal = bool(final["terminated"])
    censored = int(final["tick"]) >= max_ticks and not terminal
    usage_after = _usage()
    result = evidence.with_sha(
        {
            "schema": "irisu-exact-controller-evolution-unit-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "candidate": candidate.manifest(),
            "seed": seed,
            "simulation_ceiling_ticks": simulation_ceiling,
            "exact_runtime": exact_runtime,
            "terminal": terminal,
            "censored": censored,
            "score": int(final["score"]),
            "survival_ticks": int(final["tick"]),
            "level": int(final["level"]),
            "clears": int(final["clears"]),
            "highest_chain": int(final["highest_chain"]),
            "attempted_shots": attempted,
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "gate_reasons": dict(sorted(reasons.items())),
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha256,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": float(usage_after["user_cpu_seconds"])
                - float(usage_before["user_cpu_seconds"]),
                "system_cpu_seconds": float(usage_after["system_cpu_seconds"])
                - float(usage_before["system_cpu_seconds"]),
                "maximum_rss_kib": usage_after["maximum_rss_kib"],
            },
        }
    )
    evidence.write_new(path, result)
    return result


def load_units(run_root: Path, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        path = unit_path(run_root, candidate)
        if not path.exists():
            raise RuntimeError(f"missing candidate unit: {candidate.id}")
        row = evidence.read_json(path)
        evidence.verify_self_hash(row, "evolution unit")
        rows.append(row)
    return rows


def choose_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if row["terminal"] and not row["censored"]]
    if not eligible:
        raise RuntimeError("no naturally terminated candidate is eligible")
    return max(eligible, key=lambda row: (int(row["score"]), int(row["survival_ticks"])))


def summarize(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    identity, plan, candidates = validate(run_root, config_path, runtime)
    rows = load_units(run_root, candidates)
    best = choose_best(rows)
    result = evidence.with_sha(
        {
            "schema": "irisu-exact-controller-evolution-summary-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "candidate_count": len(rows),
            "natural_terminal_count": sum(bool(row["terminal"]) and not bool(row["censored"]) for row in rows),
            "target_score": plan["target_score"],
            "target_met": int(best["score"]) >= int(plan["target_score"]),
            "scores": {row["candidate"]["id"]: row["score"] for row in rows},
            "best": {
                "candidate": best["candidate"],
                "score": best["score"],
                "survival_ticks": best["survival_ticks"],
                "unit_sha256": best["sha256"],
            },
            "total_resources": {
                key: sum(float(row["resources"][key]) for row in rows)
                for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds")
            },
            "unit_sha256s": [row["sha256"] for row in rows],
        }
    )
    path = run_root / "summary.json"
    if path.exists():
        if evidence.read_json(path) != result:
            raise RuntimeError("existing evolution summary differs")
    else:
        evidence.write_new(path, result)
    return result


def verify(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime
) -> dict[str, object]:
    identity, plan, candidates = validate(run_root, config_path, runtime)
    rows = load_units(run_root, candidates)
    core, _campaign = screen._load_external()
    verified = []
    for row in rows:
        trace = (run_root / row["trace_file"]).read_bytes()
        if hashlib.sha256(trace).hexdigest() != row["trace_sha256"]:
            raise RuntimeError("evolution trace hash differs")
        expected = {int(item["tick"]): item for item in row["checkpoints"]}
        with runtime.open_env(
            simulation_config={
                "max_episode_ticks": int(row["simulation_ceiling_ticks"])
            }
        ) as exact_session:
            env = exact_session.environment
            observation, _info = env.reset(seed=int(row["seed"]))
            if evidence.checkpoint(env, observation) != expected[0]:
                raise RuntimeError("initial evolution replay checkpoint differs")
            words = [word for (word,) in struct.iter_unpack("<I", trace)]
            for word in words:
                observation, _reward, _terminated, _truncated, _info = env.step(
                    evidence.decode_action(core, word)
                )
                tick = int(observation["tick"])
                if tick in expected and evidence.checkpoint(env, observation) != expected[tick]:
                    raise RuntimeError(f"evolution replay differs at tick {tick}")
            final = evidence.checkpoint(env, observation)
            snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
        if (
            final != row["final"]
            or snapshot_sha256 != row["final_snapshot_sha256"]
            or len(words) != row["action_count"]
        ):
            raise RuntimeError("evolution replay final closure differs")
        verified.append(row["sha256"])
    summary = summarize(run_root, config_path, runtime)
    result = evidence.with_sha(
        {
            "schema": "irisu-exact-controller-evolution-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "summary_sha256": summary["sha256"],
            "verified_unit_sha256s": verified,
            "verified": True,
        }
    )
    evidence.write_new(run_root / "verification.json", result)
    return result


def export_best(
    run_root: Path, config_path: Path, runtime: ExactTrainingRuntime, output: Path | None
) -> dict[str, object]:
    summary = summarize(run_root, config_path, runtime)
    _identity, _plan, candidates = validate(run_root, config_path, runtime)
    row = choose_best(load_units(run_root, candidates))
    trace = (run_root / row["trace_file"]).read_bytes()
    replay = REPLAY_HEADER.pack(
        int(row["seed"]), int(row["level"]), int(row["score"]),
        int(row["highest_chain"]), 0,
    ) + bytes(32) + trace
    path = output or run_root / f"best-{row['candidate']['id']}.rpy"
    evidence.write_bytes_new(path, replay)
    return {
        "summary_sha256": summary["sha256"],
        "candidate": row["candidate"]["id"],
        "score": row["score"],
        "replay": str(path),
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "summary", "verify", "export-best"))
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be an explicit absolute path")
    runtime = ExactTrainingRuntime(args.worker)
    config_path = args.config.resolve(strict=True)
    if args.command == "init":
        value: object = initialize(args.run_root, config_path, runtime)
    elif args.command == "run":
        if args.candidate is None:
            parser.error("run requires --candidate")
        value = run_candidate(args.run_root, config_path, runtime, args.candidate)
    elif args.command == "summary":
        value = summarize(args.run_root, config_path, runtime)
    elif args.command == "verify":
        value = verify(args.run_root, config_path, runtime)
    else:
        value = export_best(args.run_root, config_path, runtime, args.output)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
