#!/usr/bin/env python3
"""Dense exact fast-checkpoint sweep around one late-game shot."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_suffix_beam as common  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


def score_key(final: dict[str, int], natural: bool, target: int) -> tuple[int, ...]:
    score = final.get("canonical_score", final["score"])
    return (
        natural and score >= target,
        score,
        final["qualifying_clear_count"],
        final["tick"],
        final["gauge"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=common.WORKER)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tick", type=int, required=True)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--target-score", type=int, default=250_000)
    parser.add_argument("--maximum-ticks", type=int, default=300_000)
    args = parser.parse_args()

    root = args.run_root.resolve()
    result_path = root / "result.json"
    if result_path.exists():
        print(result_path.read_text(), end="")
        return 0
    source = common.words(args.source.resolve(strict=True))
    if not 0 <= args.tick < len(source):
        raise ValueError("sweep tick is outside source")
    original = source[args.tick]
    center_x = (original >> 2) & 1023
    center_y = (original >> 12) & 511
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    rows: list[dict[str, object]] = []
    started = time.monotonic()

    with runtime.open_env(
        simulation_config={"max_episode_ticks": args.maximum_ticks}
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=common.SEED)
        for value in source[: args.tick]:
            observation, _reward, terminated, truncated, _info = env.step(
                common.decode(value)
            )
            if terminated or truncated:
                raise RuntimeError("source terminated before sweep tick")
        expected_hash = int(env.state_hash())
        with env.fast_checkpoint() as checkpoint:
            for y in range(center_y - args.radius, center_y + args.radius + 1, args.step):
                for x in range(center_x - args.radius, center_x + args.radius + 1, args.step):
                    for kind in (2, 1):
                        value = common.encode(kind, x, y)
                        if value is None:
                            continue
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            _suffix, final, natural = common.advance_suffix(
                                branch,
                                args.tick,
                                source,
                                value,
                                args.maximum_ticks,
                            )
                        rows.append(
                            {
                                "action": value,
                                "kind": kind,
                                "x": x,
                                "y": y,
                                "final": final,
                                "natural_terminal": natural,
                            }
                        )

    best = max(
        rows,
        key=lambda row: score_key(
            row["final"], bool(row["natural_terminal"]), args.target_score
        ),
    )
    winner = list(source[: int(best["final"]["tick"])])
    winner.extend([0] * max(0, int(best["final"]["tick"]) - len(winner)))
    winner[args.tick] = int(best["action"])
    verified, verified_natural, state_hash = common.evaluate_trace(
        runtime, winner, args.maximum_ticks
    )
    if verified != best["final"] or verified_natural != bool(best["natural_terminal"]):
        raise RuntimeError("independent exact winner replay disagrees with branch")

    trace = common.trace_bytes(winner)
    replay = common.HEADER.pack(
        common.SEED,
        verified.get("canonical_level", verified["level"]),
        verified.get("canonical_score", verified["score"]),
        verified.get("canonical_highest_chain", verified["highest_chain"]),
        0,
    ) + bytes(32) + trace
    common.atomic_write(root / "winner.u32le", trace)
    common.atomic_write(root / "winner.rpy", replay)
    result = {
        "schema": "irisu-exact-dense-shot-sweep-v1",
        "physics_backend": "exact",
        "seed": common.SEED,
        "source": str(args.source.resolve()),
        "source_sha256": common.sha256(args.source.read_bytes()),
        "tick": args.tick,
        "center": {"kind": original & 3, "x": center_x, "y": center_y},
        "radius": args.radius,
        "step": args.step,
        "evaluated_candidates": len(rows),
        "candidate_rows": rows,
        "winner": best,
        "verified_final": verified,
        "verified_natural_terminal": verified_natural,
        "target_score": args.target_score,
        "target_reached": verified_natural
        and verified.get("canonical_score", verified["score"]) >= args.target_score,
        "final_state_u64": state_hash,
        "trace_sha256": common.sha256(trace),
        "replay_sha256": common.sha256(replay),
        "exact_runtime": runtime.identity.manifest(),
        "wall_seconds": time.monotonic() - started,
    }
    common.atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
