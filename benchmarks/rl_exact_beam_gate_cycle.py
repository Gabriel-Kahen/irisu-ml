#!/usr/bin/env python3
"""Resume-safe exact beam-to-gate continuation cycle.

The teacher trace may have mixed proposal lineage, but every candidate state,
counterfactual branch, terminal, and score is produced by the pinned exact
worker.  Each cycle is append-only: retry attempts and selections receive new
ordinal directories rather than replacing prior evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks/rl_exact_100k_continuation.py"
WORD = struct.Struct("<I")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json_new(path: Path, value: object) -> None:
    write_new(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def outcome(result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if isinstance(result.get("final"), dict):
        return bool(result.get("natural_terminal")), result["final"]
    if isinstance(result.get("winner"), dict):
        return bool(result.get("winner_natural_terminal")), result["winner"]
    raise ValueError("result has neither final nor winner outcome")


def exact_natural(result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    natural, final = outcome(result)
    exact = result.get("physics_backend") == "exact"
    if "state_producing_backends" in result:
        exact = exact and result["state_producing_backends"] == ["exact"]
    terminal = bool(final.get("terminated", natural)) and not bool(
        final.get("truncated", False)
    )
    return natural and exact and terminal, final


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    cutoff: int
    probe: int
    wait: int
    gauge_advantage: int
    reserve_first: bool

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "cutoff": self.cutoff,
            "probe": self.probe,
            "wait": self.wait,
            "gauge_advantage": self.gauge_advantage,
            "reserve_first": self.reserve_first,
        }


def load_config(path: Path) -> tuple[dict[str, Any], list[Arm]]:
    value = tomllib.loads(path.read_text())
    if value.get("version") != "exact-beam-gate-cycle-v1":
        raise ValueError("cycle config version differs")
    if value.get("physics_backend") != "exact":
        raise ValueError("cycle config must require exact physics")
    arms: list[Arm] = []
    names: set[str] = set()
    for ordinal, row in enumerate(value.get("arms", []), 1):
        if not isinstance(row, dict):
            raise ValueError("each arm must be a table")
        name = str(row.get("name", f"arm-{ordinal:02d}"))
        if not name or name in names or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError(f"invalid or duplicate arm name: {name!r}")
        names.add(name)
        integers = {
            key: int(row.get(key, default))
            for key, default in (
                ("cutoff", 0), ("probe", 0), ("wait", 16),
                ("gauge_advantage", 8),
            )
        }
        if any(number < 1 for number in integers.values()):
            raise ValueError(f"arm {name} contains a nonpositive integer")
        arms.append(
            Arm(
                name,
                integers["cutoff"],
                integers["probe"],
                integers["wait"],
                integers["gauge_advantage"],
                bool(row.get("reserve_first", False)),
            )
        )
    if not arms:
        raise ValueError("cycle config requires at least one arm")
    return value, arms


def initialize(
    root: Path,
    config_path: Path,
    teacher_result_path: Path,
    teacher_trace_path: Path,
    worker_path: Path,
) -> dict[str, Any]:
    config, arms = load_config(config_path)
    result = read_json(teacher_result_path)
    accepted, final = exact_natural(result)
    if not accepted:
        raise ValueError("teacher is not a naturally terminating exact result")
    trace_size = teacher_trace_path.stat().st_size
    if trace_size % WORD.size or trace_size // WORD.size != int(final["tick"]):
        raise ValueError("teacher trace length does not match terminal tick")
    if result.get("trace_sha256") not in (None, sha256(teacher_trace_path)):
        raise ValueError("teacher trace hash differs from result")
    seed = result.get("seed", config.get("seed"))
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("teacher seed is missing or invalid")
    max_ticks = int(config.get("max_ticks", 300_000))
    target_score = int(config.get("target_score", 500_000))
    parallelism = int(config.get("parallelism", len(arms)))
    if min(max_ticks, target_score, parallelism) < 1:
        raise ValueError("cycle scalar settings must be positive")
    for arm in arms:
        if arm.cutoff >= trace_size // WORD.size:
            raise ValueError(f"arm {arm.name} cutoff is outside teacher trace")
    plan = {
        "schema": "irisu-exact-beam-gate-cycle-plan-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "proposal_lineage": "portable-trained frozen-v5; exact closed-loop continuation",
        "teacher": {
            "result": str(teacher_result_path.resolve()),
            "result_sha256": sha256(teacher_result_path),
            "trace": str(teacher_trace_path.resolve()),
            "trace_sha256": sha256(teacher_trace_path),
            "score": int(final["score"]),
            "tick": int(final["tick"]),
            "seed": seed,
        },
        "worker": str(worker_path.resolve()),
        "worker_sha256": sha256(worker_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "target_score": target_score,
        "max_ticks": max_ticks,
        "parallelism": min(parallelism, len(arms)),
        "arms": [arm.manifest() for arm in arms],
    }
    plan_path = root / "plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError("existing cycle plan differs")
    else:
        write_json_new(plan_path, plan)
    return plan


def validate_plan(root: Path) -> dict[str, Any]:
    plan = read_json(root / "plan.json")
    for key in ("worker", "config"):
        path = Path(plan[key])
        if sha256(path) != plan[f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} hash differs")
    teacher = plan["teacher"]
    for key in ("result", "trace"):
        if sha256(Path(teacher[key])) != teacher[f"{key}_sha256"]:
            raise RuntimeError(f"frozen teacher {key} hash differs")
    return plan


def next_attempt(root: Path, arm: dict[str, Any], plan: dict[str, Any]) -> tuple[Path, Path, int, int]:
    arm_root = root / "arms" / str(arm["name"])
    attempts = sorted(path for path in arm_root.glob("attempt-*" ) if path.is_dir())
    for attempt in attempts:
        if (attempt / "result.json").exists():
            return attempt, Path(), 0, 0
    source = Path(plan["teacher"]["trace"])
    cutoff = int(arm["cutoff"])
    if attempts:
        progress = attempts[-1] / "progress.json"
        progress_trace = attempts[-1] / "progress.u32le"
        if progress.exists() and progress_trace.exists():
            row = read_json(progress)
            if sha256(progress_trace) != row["trace_sha256"]:
                raise RuntimeError(f"progress trace hash differs: {progress_trace}")
            source = progress_trace
            cutoff = max(1, int(row["action_count"]) - 1)
    ordinal = len(attempts) + 1
    return arm_root / f"attempt-{ordinal:04d}", source, cutoff, ordinal


def command(plan: dict[str, Any], arm: dict[str, Any], attempt: Path, source: Path, cutoff: int) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--worker", plan["worker"],
        "--trace", str(source.resolve()),
        "--seed", str(plan["teacher"]["seed"]),
        "--cutoff", str(cutoff),
        "--controller", "gate",
        "--probe", str(arm["probe"]),
        "--wait", str(arm["wait"]),
        "--gauge-advantage", str(arm["gauge_advantage"]),
        "--max-ticks", str(plan["max_ticks"]),
        "--run-root", str(attempt),
    ]
    if arm.get("reserve_first"):
        cmd.append("--reserve-first")
    return cmd


def run(root: Path) -> dict[str, Any]:
    plan = validate_plan(root)
    lock_path = root / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending: list[tuple[dict[str, Any], Path, Path, int, int]] = []
        completed: list[str] = []
        for arm in plan["arms"]:
            attempt, source, cutoff, ordinal = next_attempt(root, arm, plan)
            if ordinal == 0:
                completed.append(str(arm["name"]))
            else:
                pending.append((arm, attempt, source, cutoff, ordinal))
        launched: list[dict[str, Any]] = []
        while pending:
            batch = pending[: int(plan["parallelism"])]
            del pending[: len(batch)]
            processes: list[tuple[subprocess.Popen[bytes], Any, Path, str]] = []
            for arm, attempt, source, cutoff, ordinal in batch:
                cmd = command(plan, arm, attempt, source, cutoff)
                attempt.parent.mkdir(parents=True, exist_ok=True)
                prefix = attempt.name
                attempt_plan = attempt.parent / f"{prefix}.plan.json"
                write_json_new(
                    attempt_plan,
                    {
                        "schema": "irisu-exact-beam-gate-attempt-v1",
                        "arm": arm,
                        "attempt": ordinal,
                        "source": str(source.resolve()),
                        "source_sha256": sha256(source),
                        "cutoff": cutoff,
                        "command": cmd,
                    },
                )
                log_path = attempt.parent / f"{prefix}.log"
                log = log_path.open("xb")
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                write_json_new(
                    attempt.parent / f"{prefix}.launched.json",
                    {"pid": process.pid, "command": cmd},
                )
                processes.append((process, log, attempt, str(arm["name"])))
            for process, log, attempt, name in processes:
                returncode = process.wait()
                log.close()
                write_json_new(
                    attempt.parent / f"{attempt.name}.exit.json",
                    {"pid": process.pid, "returncode": returncode},
                )
                launched.append(
                    {
                        "arm": name,
                        "attempt": str(attempt),
                        "returncode": returncode,
                        "result": (attempt / "result.json").exists(),
                    }
                )
        return {"completed": completed, "launched": launched}


def candidate(result_path: Path, trace_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    result = read_json(result_path)
    accepted, final = exact_natural(result)
    if not accepted:
        raise ValueError("not a naturally terminating exact result")
    if sha256(trace_path) != result.get("trace_sha256"):
        raise ValueError("trace hash differs from result")
    identity = result.get("exact_runtime", {}).get("identity", {})
    if identity.get("worker_sha256") != plan["worker_sha256"]:
        raise ValueError("exact worker identity differs")
    return {
        "score": int(final["score"]),
        "tick": int(final["tick"]),
        "result": str(result_path.resolve()),
        "result_sha256": sha256(result_path),
        "trace": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "replay": str((result_path.parent / "continuation.rpy").resolve()),
        "proposal_lineage": result.get("proposal_lineage", plan["proposal_lineage"]),
    }


def select(root: Path) -> dict[str, Any]:
    plan = validate_plan(root)
    teacher = plan["teacher"]
    candidates = [
        {
            "score": teacher["score"], "tick": teacher["tick"],
            "result": teacher["result"], "result_sha256": teacher["result_sha256"],
            "trace": teacher["trace"], "trace_sha256": teacher["trace_sha256"],
            "replay": None,
            "proposal_lineage": read_json(Path(teacher["result"])).get(
                "proposal_lineage", "teacher lineage recorded in source evidence"
            ),
        }
    ]
    rejected: list[dict[str, str]] = []
    for result_path in sorted((root / "arms").glob("*/attempt-*/result.json")):
        try:
            candidates.append(candidate(result_path, result_path.parent / "continuation.u32le", plan))
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"result": str(result_path.resolve()), "reason": str(exc)})
    fingerprint = hashlib.sha256(
        json.dumps(
            sorted((row["result_sha256"], row["trace_sha256"]) for row in candidates),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    for manifest_path in sorted((root / "selections").glob("*/selection.json")):
        manifest = read_json(manifest_path)
        if manifest.get("candidate_set_sha256") == fingerprint:
            return manifest
    winner = max(candidates, key=lambda row: (row["score"], row["tick"]))
    selection_root = root / "selections" / f"{len(list((root / 'selections').glob('*'))) + 1:04d}"
    trace_data = Path(winner["trace"]).read_bytes()
    write_new(selection_root / "next-parent.u32le", trace_data)
    replay_output: str | None = None
    if winner["replay"] and Path(winner["replay"]).exists():
        write_new(selection_root / "next-parent.rpy", Path(winner["replay"]).read_bytes())
        replay_output = str((selection_root / "next-parent.rpy").resolve())
    manifest = {
        "schema": "irisu-exact-beam-gate-selection-v1",
        "physics_backend": "exact",
        "natural_terminal_candidates_only": True,
        "candidate_set_sha256": fingerprint,
        "candidate_count": len(candidates),
        "rejected": rejected,
        "winner": winner,
        "next_parent_trace": str((selection_root / "next-parent.u32le").resolve()),
        "next_parent_trace_sha256": hashlib.sha256(trace_data).hexdigest(),
        "next_parent_replay": replay_output,
        "target_score": plan["target_score"],
        "target_reached": winner["score"] >= plan["target_score"],
    }
    write_json_new(selection_root / "selection.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "select"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--teacher-result", type=Path)
    parser.add_argument("--teacher-trace", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        if not all((args.config, args.teacher_result, args.teacher_trace, args.worker)):
            parser.error("init requires --config, --teacher-result, --teacher-trace, and --worker")
        value = initialize(
            args.run_root,
            args.config.resolve(strict=True),
            args.teacher_result.resolve(strict=True),
            args.teacher_trace.resolve(strict=True),
            args.worker.resolve(strict=True),
        )
    elif args.command == "run":
        value = run(args.run_root)
    else:
        value = select(args.run_root)
    print(json.dumps(value, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
