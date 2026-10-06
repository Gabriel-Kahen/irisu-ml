#!/usr/bin/env python3
"""Capture replay-native exact seeds with deterministic trajectory pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "python", ROOT / "benchmarks"):
    sys.path.insert(0, str(item))

from irisu_env import Action, ActionKind  # noqa: E402
from irisu_pointer.fast_multiaction_planner import FastMultiActionConfig  # noqa: E402
from irisu_pointer.trajectory_gate import (  # noqa: E402
    CollapseRiskRule,
    DEFAULT_GATES,
    GATE_CALIBRATIONS,
    GATE_PROFILES,
    TARGET_300_COLLAPSE_RISK,
    TrajectoryGate,
    canonical_sha256,
    collapse_risk_manifest,
    evaluate_collapse_risk,
    evaluate_gate,
    gate_manifest,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402
from rl_exact_adaptive_checkpoint_eval import load_policy_bundle  # noqa: E402
from rl_exact_fast_multiaction_eval import run_episode  # noqa: E402


HEADER = struct.Struct("<I4i")
WORD = struct.Struct("<I")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(
        path,
        (json.dumps(value, sort_keys=True, indent=2) + "\n").encode(),
    )


def encode(action: object) -> int:
    kind = int(ActionKind.parse(getattr(action, "kind")))
    if kind == 0:
        return 0
    x = int(round(float(getattr(action, "cursor_x"))))
    y = int(round(float(getattr(action, "cursor_y"))))
    if kind not in (1, 2, 3) or not (0 <= x < 640 and 0 <= y < 480):
        raise ValueError("live action is not replay-encodable")
    return kind | (x << 2) | (y << 12)


def decode(value: int) -> Action:
    kind = value & 3
    if kind == 0:
        return Action.wait(1)
    return Action(kind, (value >> 2) & 1023, (value >> 12) & 511, 1)


def observation_point(observation: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "level": int(observation.get("level", 0)),
        "gauge": int(observation.get("gauge", 0)),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "terminated": bool(observation.get("terminated", False)),
        "truncated": bool(observation.get("truncated", False)),
    }


class WeakSeedStop(RuntimeError):
    def __init__(self, verdict: Mapping[str, object]) -> None:
        super().__init__("trajectory gate rejected seed")
        self.verdict = dict(verdict)


class GatedEnvironment:
    def __init__(
        self,
        environment: object,
        words: list[int],
        startup_waits: int,
        progress_path: Path,
        policy_manifest: Mapping[str, object],
        mode: str,
        gates: Sequence[TrajectoryGate] = DEFAULT_GATES,
        collapse_risk_rule: CollapseRiskRule | None = None,
        collapse_risk_enforce: bool = True,
    ) -> None:
        self._environment = environment
        self._words = words
        self._startup_waits = startup_waits
        self._progress_path = progress_path
        self._policy_manifest = policy_manifest
        self._mode = mode
        self._gates = tuple(gates)
        self._collapse_risk_rule = collapse_risk_rule
        self._collapse_risk_enforce = collapse_risk_enforce
        self.checkpoints: list[dict[str, object]] = []
        self.checkpoint_scores: dict[int, int] = {}
        self.checkpoint_gauges: dict[int, int] = {}
        self.interval_minimum_gauge: int | None = None
        self.last_observation: Mapping[str, object] | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._environment, name)

    def reset(self, **kwargs):
        observation, reset_info = self._environment.reset(**kwargs)
        for _ in range(self._startup_waits):
            observation, _reward, terminated, truncated, _info = (
                self._environment.step(Action.wait(1))
            )
            self._words.append(0)
            if terminated or truncated:
                raise RuntimeError("startup replay waits terminated the episode")
        self.last_observation = observation
        self.interval_minimum_gauge = int(observation.get("gauge", 0))
        return observation, reset_info

    def step(self, action: object):
        result = self._environment.step(action)
        self._words.append(encode(action))
        observation = result[0]
        self.last_observation = observation
        gauge = int(observation.get("gauge", 0))
        if self.interval_minimum_gauge is None:
            self.interval_minimum_gauge = gauge
        else:
            self.interval_minimum_gauge = min(self.interval_minimum_gauge, gauge)
        tick = int(observation["tick"])
        if tick % 10_000 == 0:
            point = observation_point(observation)
            self.checkpoints.append(point)
            self.checkpoint_scores[tick] = int(observation["score"])
            gate = next((item for item in self._gates if item.tick == tick), None)
            verdict = None
            if gate is not None:
                verdict = evaluate_gate(gate, observation, self.checkpoint_scores)
                if (
                    bool(verdict["passed"])
                    and self._collapse_risk_rule is not None
                    and tick in self._collapse_risk_rule.ticks
                ):
                    collapse = evaluate_collapse_risk(
                        self._collapse_risk_rule,
                        observation,
                        self.checkpoint_scores,
                        self.checkpoint_gauges,
                        int(self.interval_minimum_gauge),
                    )
                    verdict["collapse_risk"] = collapse
                    verdict["collapse_risk_enforced"] = self._collapse_risk_enforce
                    if self._collapse_risk_enforce:
                        verdict["checks"]["collapse_risk"] = collapse["passed"]
                        verdict["passed"] = all(verdict["checks"].values())
            trace = b"".join(WORD.pack(word) for word in self._words)
            progress = {
                "schema": "irisu-exact-trajectory-gated-progress-v1",
                "mode": self._mode,
                "policy": self._policy_manifest,
                "latest": point,
                "checkpoints": self.checkpoints,
                "word_count": len(self._words),
                "prefix_sha256": hashlib.sha256(trace).hexdigest(),
                "gate_verdict": verdict,
            }
            atomic_json(self._progress_path, progress)
            self.checkpoint_gauges[tick] = gauge
            self.interval_minimum_gauge = gauge
            if (
                self._mode == "enforce"
                and verdict is not None
                and not bool(verdict["passed"])
            ):
                raise WeakSeedStop(verdict)
        return result


class GatedRuntime:
    def __init__(
        self,
        runtime: ExactTrainingRuntime,
        words: list[int],
        startup_waits: int,
        progress_path: Path,
        policy_manifest: Mapping[str, object],
        mode: str,
        gates: Sequence[TrajectoryGate] = DEFAULT_GATES,
        collapse_risk_rule: CollapseRiskRule | None = None,
        collapse_risk_enforce: bool = True,
    ) -> None:
        self.runtime = runtime
        self.words = words
        self.startup_waits = startup_waits
        self.progress_path = progress_path
        self.policy_manifest = policy_manifest
        self.mode = mode
        self.gates = tuple(gates)
        self.collapse_risk_rule = collapse_risk_rule
        self.collapse_risk_enforce = collapse_risk_enforce
        self.environment: GatedEnvironment | None = None
        self.provenance: Mapping[str, object] | None = None

    @contextmanager
    def open_env(self, **kwargs):
        with self.runtime.open_env(**kwargs) as session:
            self.provenance = session.provenance_manifest
            self.environment = GatedEnvironment(
                session.environment,
                self.words,
                self.startup_waits,
                self.progress_path,
                self.policy_manifest,
                self.mode,
                self.gates,
                self.collapse_risk_rule,
                self.collapse_risk_enforce,
            )
            yield SimpleNamespace(
                environment=self.environment,
                provenance_manifest=session.provenance_manifest,
            )


def replay_prefix(
    runtime: ExactTrainingRuntime,
    seed: int,
    words: list[int],
    maximum_ticks: int,
    probe_padding: int,
) -> tuple[dict[str, object], int]:
    with runtime.open_env(
        # Match run_episode's live-parent ceiling.  A prefix ending exactly at
        # maximum_ticks is a search cap, not a simulator truncation.
        simulation_config={"max_episode_ticks": maximum_ticks + probe_padding}
    ) as session:
        env = session.environment
        observation, _info = env.reset(seed=seed)
        terminated = truncated = False
        for value in words:
            if terminated or truncated:
                raise RuntimeError("recorded actions continue past exact terminal")
            observation, _reward, terminated, truncated, _info = env.step(decode(value))
        return observation_point(observation), int(env.state_hash())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-waits", type=int, default=2)
    parser.add_argument("--maximum-ticks", type=int, default=100_000)
    parser.add_argument("--probe-ticks", type=int, default=256)
    parser.add_argument("--long-probe-ticks", type=int, default=512)
    parser.add_argument("--top-k-pairs", type=int, default=0)
    parser.add_argument("--long-probe-min-tick", type=int, default=0)
    parser.add_argument(
        "--objective-mode",
        choices=(
            "wait-relative",
            "reserve-band",
            "robust-reserve-tie",
            "robust-reserve-bounded",
            "chain-first",
        ),
        default="wait-relative",
    )
    parser.add_argument("--robust-reserve-margin", type=int, default=1_000)
    parser.add_argument("--robust-score-trade", type=int, default=500)
    parser.add_argument("--gate-mode", choices=("observe", "enforce"), default="enforce")
    parser.add_argument(
        "--collapse-risk-mode",
        choices=("profile", "shadow", "off"),
        default="profile",
        help="enforce the profile rule, record it without pruning, or disable it",
    )
    parser.add_argument(
        "--gate-profile",
        choices=tuple(GATE_PROFILES),
        default="conservative-200",
    )
    args = parser.parse_args()
    if args.startup_waits < 2:
        parser.error("--startup-waits must be at least 2 for replay-native capture")
    return args


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = args.worker.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    bundle = load_policy_bundle(
        checkpoint, expected_sha256=args.checkpoint_sha256
    )
    config = FastMultiActionConfig(
        probe_ticks=args.probe_ticks,
        long_probe_ticks=args.long_probe_ticks,
        low_gauge_threshold=20_000,
        low_gauge_exit_threshold=30_000,
        wait_ticks=16,
        top_k_pairs=args.top_k_pairs,
        long_probe_min_tick=args.long_probe_min_tick,
        objective_mode=args.objective_mode,
        robust_reserve_margin=args.robust_reserve_margin,
        robust_score_trade=args.robust_score_trade,
        maximum_gauge_debt=1_000,
        rescue_score_margin=500,
        gauge_advantage=1,
    )
    gates = GATE_PROFILES[args.gate_profile]
    collapse_risk_rule = (
        TARGET_300_COLLAPSE_RISK
        if args.gate_profile == "target-300" and args.collapse_risk_mode != "off"
        else None
    )
    collapse_risk_enforce = args.collapse_risk_mode == "profile"
    policy = {
        "seed": args.seed,
        "maximum_ticks": args.maximum_ticks,
        "startup_waits": args.startup_waits,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "planner_config": config.manifest(),
        "gate_mode": args.gate_mode,
        "gate_profile": args.gate_profile,
        "gate_calibration": GATE_CALIBRATIONS[args.gate_profile],
        **gate_manifest(gates),
        **collapse_risk_manifest(collapse_risk_rule),
        "collapse_risk_mode": args.collapse_risk_mode,
    }
    policy["policy_sha256"] = canonical_sha256(policy)
    atomic_json(output / "policy.json", policy)

    words: list[int] = []
    runtime = ExactTrainingRuntime(worker)
    gated = GatedRuntime(
        runtime,
        words,
        args.startup_waits,
        output / "progress.json",
        policy,
        args.gate_mode,
        gates,
        collapse_risk_rule,
        collapse_risk_enforce,
    )
    try:
        episode, provenance = run_episode(
            gated,
            bundle,
            args.seed,
            maximum_ticks=args.maximum_ticks,
            planner_config=config,
            target_score=300_000,
            trace_interval_ticks=10_000,
            maximum_logged_queries=8,
            recover_exact_branch_errors=True,
        )
    except WeakSeedStop as stop:
        trace = b"".join(WORD.pack(word) for word in words)
        verified, state_hash = replay_prefix(
            runtime,
            args.seed,
            words,
            args.maximum_ticks,
            max(config.probe_ticks, config.long_probe_ticks),
        )
        live = observation_point(gated.environment.last_observation)  # type: ignore[union-attr]
        if verified != live:
            raise RuntimeError("exact prefix replay disagrees with gated live state")
        atomic_write(output / "screening-prefix.u32le", trace)
        screening = {
            "schema": "irisu-exact-trajectory-screening-v1",
            "screening_only": True,
            "promotable": False,
            "policy": policy,
            "verdict": stop.verdict,
            "live": live,
            "verified": verified,
            "verified_state_u64": state_hash,
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "exact_runtime": gated.provenance,
        }
        atomic_json(output / "screening.json", screening)
        print(json.dumps(screening, sort_keys=True))
        return 0

    trace = b"".join(WORD.pack(word) for word in words)
    verified, state_hash = replay_prefix(
        runtime,
        args.seed,
        words,
        args.maximum_ticks,
        max(config.probe_ticks, config.long_probe_ticks),
    )
    expected = {
        "tick": int(episode["tick"]),
        "score": int(episode["score"]),
        "level": int(episode["level"]),
        "gauge": int(episode["gauge"]),
        "highest_chain": int(episode["highest_chain"]),
        "terminated": bool(episode["terminated"]),
        "truncated": bool(episode["truncated"]),
    }
    if verified != expected:
        raise RuntimeError(
            "independent exact replay disagrees with recorded episode: "
            f"verified={verified!r} expected={expected!r}"
        )
    replay = (
        HEADER.pack(
            args.seed,
            int(episode["level"]),
            int(episode["score"]),
            int(episode["highest_chain"]),
            0,
        )
        + bytes(32)
        + trace
    )
    atomic_write(output / "source.u32le", trace)
    atomic_write(output / "source.rpy", replay)
    summary = {
        "schema": "irisu-exact-trajectory-gated-capture-v1",
        "policy": policy,
        "episode": episode,
        "verified": verified,
        "verified_state_u64": state_hash,
        "trace_sha256": hashlib.sha256(trace).hexdigest(),
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
        "exact_runtime": provenance,
    }
    atomic_json(output / "capture-summary.json", summary)
    if bool(episode["terminated"]) and not bool(episode["truncated"]):
        base_command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "tools/evaluate-rpy.py"),
            "--worker",
            str(worker),
            "--compact",
        ]
        command = base_command + [
            "--purpose", "promotion", str(output / "source.rpy")
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            initial_promotion_failure = result.stderr.strip()
            diagnostic_command = base_command + [
                "--purpose", "diagnostic", str(output / "source.rpy")
            ]
            diagnostic_result = subprocess.run(
                diagnostic_command, cwd=ROOT, text=True, capture_output=True
            )
            if diagnostic_result.returncode != 0:
                raise RuntimeError(
                    "promotion and diagnostic audits failed: "
                    f"{result.stderr.strip()} / {diagnostic_result.stderr.strip()}"
                )
            diagnostic = json.loads(diagnostic_result.stdout)
            outcome = diagnostic["outcome"]
            canonical = {
                "level": int(outcome["level"]["clone_final"]),
                "score": int(outcome["score"]["clone_final"]),
                "highest_chain": int(outcome["highest_chain"]["clone_final"]),
            }
            canonical_replay = (
                HEADER.pack(
                    args.seed,
                    canonical["level"],
                    canonical["score"],
                    canonical["highest_chain"],
                    0,
                )
                + bytes(32)
                + trace
            )
            canonical_path = output / "promotion-canonical.rpy"
            atomic_write(canonical_path, canonical_replay)
            command = base_command + [
                "--purpose", "promotion", str(canonical_path)
            ]
            canonical_result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True
            )
            if canonical_result.returncode != 0:
                raise RuntimeError(
                    "canonical promotion audit failed: "
                    f"{canonical_result.stderr.strip()}"
                )
            result = canonical_result
            atomic_json(
                output / "canonicalization.json",
                {
                    "schema": "irisu-replay-terminal-metadata-canonicalization-v1",
                    "live_header_replay_sha256": hashlib.sha256(replay).hexdigest(),
                    "canonical_header": canonical,
                    "canonical_replay_sha256": hashlib.sha256(
                        canonical_replay
                    ).hexdigest(),
                    "initial_promotion_failure": initial_promotion_failure,
                    "diagnostic_command": diagnostic_command,
                    "diagnostic": diagnostic,
                },
            )
        atomic_json(
            output / "promotion.json",
            {"command": command, "promotion": json.loads(result.stdout)},
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
