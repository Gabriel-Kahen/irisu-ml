#!/usr/bin/env python3
"""One-depth exact multi-parent search over an explicit replay interval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_multi_parent_beam as beam  # noqa: E402
import rl_exact_suffix_beam as suffix  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "artifacts/r3/development/exact-300k-multi-parent-beam-20260810-003"
    / "best.u32le"
)
DEFAULT_ROOT = (
    ROOT / "artifacts/r3/development/exact-300k-ranged-100352-102400-20260810-001"
)


def validate_range(start: int, end: int, trace_length: int) -> None:
    if not 0 <= start < end <= trace_length:
        raise ValueError("search range must satisfy 0 <= start < end <= trace length")


def should_search_tick(
    tick: int,
    start: int,
    end: int,
    stride: int,
    incumbent: int,
    *,
    include_incumbent_shots: bool,
) -> bool:
    if stride <= 0:
        raise ValueError("tick stride must be positive")
    return start <= tick < end and (
        (tick - start) % stride == 0
        or (include_incumbent_shots and incumbent != 0)
    )


def expand_range(
    runtime: ExactTrainingRuntime,
    parent: beam.Parent,
    args: argparse.Namespace,
) -> tuple[list[beam.Parent], int]:
    children = [parent]
    evaluated = 0
    trace = list(parent.trace)
    validate_range(args.start_tick, args.end_tick, len(trace))
    with runtime.open_env(
        simulation_config={"max_episode_ticks": args.maximum_ticks}
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=suffix.SEED)
        terminated = truncated = False
        for tick in range(args.end_tick):
            if terminated or truncated:
                raise RuntimeError("source terminated before the search interval ended")
            incumbent = trace[tick]
            if should_search_tick(
                tick,
                args.start_tick,
                args.end_tick,
                args.tick_stride,
                incumbent,
                include_incumbent_shots=args.include_incumbent_shots,
            ):
                expected_hash = int(env.state_hash())
                with env.fast_checkpoint() as checkpoint:
                    for value, replacements in beam.branch_specs(
                        observation,
                        parent.trace,
                        tick,
                        args.candidate_cap,
                        args.timing_jitter_cap,
                    ):
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            _emitted, final, natural = suffix.advance_suffix(
                                branch,
                                tick,
                                trace,
                                value,
                                args.maximum_ticks,
                                replacements,
                            )
                        row: dict[str, Any] = {
                            "tick": tick,
                            "action": value,
                            "replacements": replacements,
                            "final": final,
                        }
                        child_trace = tuple(
                            suffix.reconstructed_candidate(trace, row)
                        )
                        edit: Mapping[str, int] = {
                            str(key): int(replacement)
                            for key, replacement in replacements.items()
                        }
                        children.append(
                            beam.Parent(
                                child_trace,
                                final,
                                natural,
                                parent.lineage + ({"tick": tick, **edit},),
                            )
                        )
                        evaluated += 1
            observation, _reward, terminated, truncated, _info = env.step(
                suffix.decode(incumbent)
            )
    return children, evaluated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worker", type=Path, default=suffix.WORKER)
    parser.add_argument("--start-tick", type=int, required=True)
    parser.add_argument("--end-tick", type=int, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--tick-stride", type=int, default=128)
    parser.add_argument("--candidate-cap", type=int, default=4)
    parser.add_argument("--timing-jitter-cap", type=int, default=0)
    parser.add_argument("--include-incumbent-shots", action="store_true")
    parser.add_argument("--maximum-ticks", type=int, default=120_000)
    parser.add_argument("--target-score", type=int, default=300_000)
    args = parser.parse_args()
    if args.beam_width < 1 or args.candidate_cap < 1:
        parser.error("beam width and candidate cap must be positive")
    args.source = args.source.resolve(strict=True)
    args.run_root = args.run_root.resolve()
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    source = suffix.words(args.source)
    try:
        validate_range(args.start_tick, args.end_tick, len(source))
    except ValueError as error:
        parser.error(str(error))
    args.run_root.mkdir(parents=True, exist_ok=True)

    final, natural, _state = suffix.evaluate_trace(
        runtime, source, args.maximum_ticks
    )
    parent = beam.Parent(tuple(source[: int(final["tick"])]), final, natural)
    started = time.monotonic()
    children, evaluated = expand_range(runtime, parent, args)
    parents = beam.beam_select(
        children,
        args.beam_width,
        args.maximum_ticks,
        args.target_score,
    )
    beam.write_depth(
        args.run_root, 1, parents, evaluated, time.monotonic() - started
    )
    winner = parents[0]
    verified, verified_natural, state_hash = suffix.evaluate_trace(
        runtime, list(winner.trace), args.maximum_ticks
    )
    if dict(verified) != dict(winner.final) or verified_natural != winner.natural:
        raise RuntimeError("ranged winner failed independent exact replay")
    data = suffix.trace_bytes(list(winner.trace))
    replay = beam.replay_bytes(winner)
    suffix.atomic_write(args.run_root / "best.u32le", data)
    suffix.atomic_write(args.run_root / "best.rpy", replay)
    summary = {
        "schema": "irisu-exact-ranged-multi-parent-summary-v1",
        "physics_backend": "exact",
        "source": str(args.source),
        "source_sha256": suffix.sha256(args.source.read_bytes()),
        "search_interval": [args.start_tick, args.end_tick],
        "tick_stride": args.tick_stride,
        "candidate_cap": args.candidate_cap,
        "include_incumbent_shots": args.include_incumbent_shots,
        "evaluated_candidates": evaluated,
        "verified_final": verified,
        "verified_natural_terminal": verified_natural,
        "target_score": args.target_score,
        "target_reached": verified_natural
        and beam.canonical_score(winner) >= args.target_score,
        "final_state_u64": state_hash,
        "lineage": list(winner.lineage),
        "trace_sha256": suffix.sha256(data),
        "replay_sha256": suffix.sha256(replay),
        "exact_runtime": runtime.identity.manifest(),
    }
    suffix.atomic_json(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
