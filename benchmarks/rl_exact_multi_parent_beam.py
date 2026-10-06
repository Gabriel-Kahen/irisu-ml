#!/usr/bin/env python3
"""Exact multi-parent, multi-edit beam search over a verified replay tail."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_suffix_beam as suffix  # noqa: E402
from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "artifacts/r3/development/exact-250k-combo-chain-priority-20260810-001"
    / "reranked.u32le"
)
DEFAULT_ROOT = (
    ROOT / "artifacts/r3/development/exact-300k-multi-parent-beam-20260810-001"
)


class Parent(NamedTuple):
    trace: tuple[int, ...]
    final: Mapping[str, int]
    natural: bool
    lineage: tuple[Mapping[str, int], ...] = ()


def canonical_score(parent: Parent) -> int:
    return int(parent.final.get("canonical_score", parent.final["score"]))


def beam_select(parents: list[Parent], width: int, maximum: int, target: int) -> list[Parent]:
    """Select strong distinct parents instead of collapsing to one suffix edit."""
    ordered = sorted(
        parents,
        key=lambda parent: suffix.objective(parent.final, parent.natural, maximum, target),
        reverse=True,
    )
    selected: list[Parent] = []
    traces: set[str] = set()
    buckets: set[tuple[int, int, int]] = set()
    for distinct_pass in (True, False):
        for parent in ordered:
            digest = suffix.sha256(suffix.trace_bytes(list(parent.trace)))
            if digest in traces:
                continue
            bucket = (
                canonical_score(parent),
                int(parent.final.get("canonical_highest_chain", parent.final["highest_chain"])),
                int(parent.final["gauge"]) // 1000,
            )
            if distinct_pass and bucket in buckets:
                continue
            selected.append(parent)
            traces.add(digest)
            buckets.add(bucket)
            if len(selected) == width:
                return selected
    return selected


def branch_specs(
    observation: Mapping[str, Any], trace: tuple[int, ...], tick: int, cap: int, jitter_cap: int
) -> list[tuple[int, dict[int, int]]]:
    incumbent = trace[tick]
    specs = [(value, {tick: value}) for value in suffix.candidates(observation, cap, incumbent)]
    if incumbent:
        specs.append((0, {tick: 0}))
    specs.extend(suffix.timing_jitter_specs(list(trace), tick)[:jitter_cap])
    output: list[tuple[int, dict[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for value, replacements in specs:
        identity = tuple(sorted(replacements.items()))
        if identity not in seen and any(trace[key] != replacement for key, replacement in identity):
            seen.add(identity)
            output.append((value, replacements))
    return output


def expand(
    runtime: ExactTrainingRuntime, parent: Parent, args: argparse.Namespace
) -> tuple[list[Parent], int]:
    children = [parent]
    evaluated = 0
    start = max(0, len(parent.trace) - args.window_ticks)
    trace = list(parent.trace)
    with runtime.open_env(simulation_config={"max_episode_ticks": args.maximum_ticks}) as session:
        env = session.environment
        observation, _ = env.reset(seed=suffix.SEED)
        terminated = truncated = False
        for tick, incumbent in enumerate(trace):
            if terminated or truncated:
                break
            jitters = suffix.timing_jitter_specs(trace, tick)
            if suffix.should_search_tick(
                tick,
                start,
                args.tick_stride,
                incumbent,
                bool(jitters and not incumbent),
            ):
                expected_hash = int(env.state_hash())
                with env.fast_checkpoint() as checkpoint:
                    for value, replacements in branch_specs(
                        observation, parent.trace, tick, args.candidate_cap, args.timing_jitter_cap
                    ):
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            _emitted, final, natural = suffix.advance_suffix(
                                branch,
                                tick,
                                list(parent.trace),
                                value,
                                args.maximum_ticks,
                                replacements,
                            )
                        row = {"tick": tick, "action": value, "replacements": replacements, "final": final}
                        child_trace = tuple(suffix.reconstructed_candidate(list(parent.trace), row))
                        edit = {str(key): int(replacement) for key, replacement in replacements.items()}
                        children.append(
                            Parent(child_trace, final, natural, parent.lineage + ({"tick": tick, **edit},))
                        )
                        evaluated += 1
            observation, _reward, terminated, truncated, _info = env.step(suffix.decode(incumbent))
    return children, evaluated


def write_depth(root: Path, depth: int, parents: list[Parent], evaluated: int, elapsed: float) -> None:
    directory = root / "depths" / f"{depth:02d}"
    rows = []
    for rank, parent in enumerate(parents):
        data = suffix.trace_bytes(list(parent.trace))
        relative = f"parent-{rank:02d}.u32le"
        suffix.atomic_write(directory / relative, data)
        rows.append(
            {
                "rank": rank,
                "trace": relative,
                "trace_sha256": suffix.sha256(data),
                "final": dict(parent.final),
                "natural_terminal": parent.natural,
                "lineage": list(parent.lineage),
            }
        )
    suffix.atomic_json(
        directory / "result.json",
        {
            "schema": "irisu-exact-multi-parent-depth-v1",
            "depth": depth,
            "evaluated_candidates": evaluated,
            "parents": rows,
            "wall_seconds": elapsed,
        },
    )


def replay_bytes(parent: Parent) -> bytes:
    final = parent.final
    return suffix.HEADER.pack(
        suffix.SEED,
        int(final.get("canonical_level", final["level"])),
        canonical_score(parent),
        int(final.get("canonical_highest_chain", final["highest_chain"])),
        0,
    ) + bytes(32) + suffix.trace_bytes(list(parent.trace))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worker", type=Path, default=suffix.WORKER)
    parser.add_argument("--depths", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--window-ticks", type=int, default=512)
    parser.add_argument("--tick-stride", type=int, default=16)
    parser.add_argument("--candidate-cap", type=int, default=6)
    parser.add_argument("--timing-jitter-cap", type=int, default=4)
    parser.add_argument("--maximum-ticks", type=int, default=120_000)
    parser.add_argument("--target-score", type=int, default=300_000)
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    args.run_root = args.run_root.resolve()
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    args.run_root.mkdir(parents=True, exist_ok=True)

    source = suffix.words(args.source)
    final, natural, _state = suffix.evaluate_trace(runtime, source, args.maximum_ticks)
    beam = [Parent(tuple(source[: int(final["tick"])]), final, natural)]
    total = 0
    for depth in range(1, args.depths + 1):
        started = time.monotonic()
        pool: list[Parent] = []
        evaluated = 0
        for parent in beam:
            children, count = expand(runtime, parent, args)
            pool.extend(children)
            evaluated += count
        beam = beam_select(pool, args.beam_width, args.maximum_ticks, args.target_score)
        total += evaluated
        write_depth(args.run_root, depth, beam, evaluated, time.monotonic() - started)
        print(json.dumps({"depth": depth, "evaluated": evaluated, "scores": [canonical_score(p) for p in beam]}), flush=True)
        if canonical_score(beam[0]) >= args.target_score and beam[0].natural:
            break

    winner = beam[0]
    verified, verified_natural, state_hash = suffix.evaluate_trace(
        runtime, list(winner.trace), args.maximum_ticks
    )
    if dict(verified) != dict(winner.final) or verified_natural != winner.natural:
        raise RuntimeError("winner failed independent exact replay verification")
    data = suffix.trace_bytes(list(winner.trace))
    replay = replay_bytes(winner)
    suffix.atomic_write(args.run_root / "best.u32le", data)
    suffix.atomic_write(args.run_root / "best.rpy", replay)
    summary = {
        "schema": "irisu-exact-multi-parent-beam-summary-v1",
        "physics_backend": "exact",
        "source": str(args.source),
        "source_sha256": suffix.sha256(args.source.read_bytes()),
        "target_score": args.target_score,
        "target_reached": verified_natural and canonical_score(winner) >= args.target_score,
        "evaluated_candidates": total,
        "beam_width": args.beam_width,
        "depths_completed": depth,
        "verified_final": verified,
        "verified_natural_terminal": verified_natural,
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
