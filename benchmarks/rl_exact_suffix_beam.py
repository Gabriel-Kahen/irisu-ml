#!/usr/bin/env python3
"""Resume-safe exact-backend suffix beam search over late-game shots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


WORKER = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/exact-runtime-backup/irisu-exact-worker"
SOURCE = ROOT / "artifacts/r3/development/exact-seed-tick-distill-20260810-004/dataset/exact-teacher.u32le"
DEFAULT_ROOT = ROOT / "artifacts/r3/development/exact-100k-suffix-beam-20260810-001"
SEED = 3_939_967_453
WORD = struct.Struct("<I")
HEADER = struct.Struct("<I4i")
TIMING_JITTERS = (1, 2, 4, 8)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


def words(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % WORD.size:
        raise ValueError(f"unaligned action trace: {path}")
    return [value for (value,) in struct.iter_unpack("<I", data)]


def trace_bytes(values: list[int]) -> bytes:
    return b"".join(WORD.pack(value) for value in values)


def reconstructed_candidate(incumbent: list[int], row: Mapping[str, Any]) -> list[int]:
    """Materialize a one-shot branch recorded by :func:`generation`."""
    tick = int(row["tick"])
    final_tick = int(row["final"]["tick"])
    values = list(incumbent[:final_tick])
    values.extend([0] * max(0, final_tick - len(values)))
    replacements = row.get("replacements", {str(tick): int(row["action"])})
    assert isinstance(replacements, Mapping)
    for replacement_tick, value in replacements.items():
        replacement_tick = int(replacement_tick)
        if replacement_tick < final_tick:
            values[replacement_tick] = int(value)
    return values


def timing_jitter_specs(
    incumbent: list[int], tick: int
) -> list[tuple[int, dict[int, int]]]:
    """Return bounded two-edit mutations that move a proven shot in time."""
    value = incumbent[tick]
    specs: list[tuple[int, dict[int, int]]] = []
    if value:
        for delta in TIMING_JITTERS:
            destination = tick + delta
            if destination < len(incumbent) and incumbent[destination] == 0:
                specs.append((0, {tick: 0, destination: value}))
    else:
        for delta in TIMING_JITTERS:
            source = tick + delta
            if source < len(incumbent) and incumbent[source] != 0:
                specs.append(
                    (incumbent[source], {tick: incumbent[source], source: 0})
                )
    return specs


def pareto_rows(rows: list[dict[str, object]], width: int) -> list[dict[str, object]]:
    """Keep diverse, non-dominated score/survival branches.

    Survival and score can trade off sharply near a cascade.  Retaining only
    the lexicographic winner loses high-scoring parents that are one later
    repair away from becoming the better survivor.
    """
    if width <= 0:
        return []

    def coordinates(row: Mapping[str, object]) -> tuple[int, int, int, int]:
        final = row["final"]
        assert isinstance(final, Mapping)
        return (
            int(final["tick"]),
            int(final.get("canonical_score", final["score"])),
            int(final["qualifying_clear_count"]),
            int(final["gauge"]),
        )

    frontier: list[dict[str, object]] = []
    for candidate in rows:
        point = coordinates(candidate)
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            alternate = coordinates(other)
            if all(left >= right for left, right in zip(alternate, point)) and any(
                left > right for left, right in zip(alternate, point)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    # Alternating order guarantees that both survival and high-scoring ends of
    # the frontier remain represented when the archive is bounded.
    by_survival = sorted(frontier, key=coordinates, reverse=True)
    by_score = sorted(
        frontier,
        key=lambda row: (
            coordinates(row)[1],
            coordinates(row)[0],
            coordinates(row)[2],
            coordinates(row)[3],
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for rank in range(max(len(by_survival), len(by_score))):
        for ordered in (by_survival, by_score):
            if rank >= len(ordered):
                continue
            row = ordered[rank]
            identity = (
                int(row["tick"]),
                int(row["action"]),
                json.dumps(row.get("replacements", {}), sort_keys=True),
            )
            if identity not in seen:
                seen.add(identity)
                selected.append(row)
                if len(selected) >= width:
                    return selected
    return selected


def decode(value: int) -> Action:
    kind = value & 3
    x, y = (value >> 2) & 1023, (value >> 12) & 511
    if kind == 1:
        return Action.weak(x, y)
    if kind == 2:
        return Action.strong(x, y)
    if kind == 3:
        return Action.both(x, y)
    return Action.wait(1)


def encode(kind: int, x: float, y: float) -> int | None:
    px, py = int(round(x)), int(round(y))
    if kind not in (1, 2) or not (0 <= px < 640 and 0 <= py < 480):
        return None
    return (py << 12) | (px << 2) | kind


def snapshot(
    observation: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None
) -> dict[str, int]:
    result = {
        name: int(observation.get(name, 0))
        for name in (
            "tick", "score", "gauge", "level", "highest_chain",
            "qualifying_clear_count", "terminated", "truncated",
        )
    }
    if diagnostics is not None and diagnostics.get("terminal_metadata_recorded"):
        result.update(
            {
                "canonical_score": int(diagnostics["recorded_final_score"]),
                "canonical_level": int(diagnostics["recorded_final_level"]),
                "canonical_highest_chain": int(
                    diagnostics["recorded_final_highest_chain"]
                ),
                "canonical_clears": int(diagnostics["recorded_final_clears"]),
            }
        )
    return result


def should_search_tick(
    tick: int,
    start_tick: int,
    stride: int,
    incumbent_value: int,
    has_future_shot: bool = False,
) -> bool:
    """Search the regular grid plus every incumbent shot in the death window."""
    return tick >= start_tick and (
        (tick - start_tick) % stride == 0
        or incumbent_value != 0
        or has_future_shot
    )


def candidates(
    observation: Mapping[str, Any], cap: int, incumbent_value: int = 0
) -> list[int]:
    pieces = [
        body for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and body.get("kind") == "piece"
    ]
    sources = [
        body for body in pieces
        if str(body.get("lifecycle", "")) in {
            "scripted_falling", "dynamic_fresh", "falling", "fresh"
        }
        and int(body.get("chain_id", 0)) == 0
        and int(body.get("projectile_hits", 0)) == 0
    ]
    # Bottom-most pristine pieces are the immediate survival risk.
    sources.sort(key=lambda body: (-float(body.get("y", 0.0)), int(body.get("id", 0))))
    ranked: list[tuple[tuple[float, ...], int]] = []
    for source_rank, source in enumerate(sources):
        sx = float(source.get("effect_x", source.get("x", 0.0)))
        sy = float(source.get("effect_y", source.get("y", 0.0)))
        size = float(source.get("size", 0.0))
        color = int(source.get("color", -1))
        if size <= 0.0:
            continue
        destinations = [
            body for body in pieces
            if int(body.get("id", -1)) != int(source.get("id", -1))
            and int(body.get("color", -2)) == color
            and str(body.get("lifecycle", "")) != "deleted"
        ]
        destinations.sort(
            key=lambda body: (
                -int(int(body.get("chain_id", 0)) > 0),
                (float(body.get("x", 0.0)) - sx) ** 2
                + (float(body.get("y", 0.0)) - sy) ** 2,
                int(body.get("id", 0)),
            )
        )
        for destination_rank, destination in enumerate(destinations[:4]):
            dx = float(destination.get("effect_x", destination.get("x", 0.0)))
            direction = 1.0 if dx > sx else -1.0
            if abs(dx - sx) < 1e-9:
                direction = 1.0 if sx <= 320.0 else -1.0
            for strength in (2, 1):
                for side in (0.25, 0.5, 0.75, 1.0, 0.0):
                    for below in (0.5, 0.75, 1.0, 1.25):
                        value = encode(
                            strength,
                            sx - direction * side * size,
                            sy + below * size,
                        )
                        if value is not None:
                            ranked.append(
                                (
                                    (
                                        source_rank,
                                        destination_rank,
                                        strength != 2,
                                        abs(side - 0.5),
                                        abs(below - 0.75),
                                    ),
                                    value,
                                )
                            )
    output: list[int] = []
    incumbent_kind = incumbent_value & 3
    if incumbent_kind in (1, 2):
        incumbent_x = (incumbent_value >> 2) & 1023
        incumbent_y = (incumbent_value >> 12) & 511
        # Preserve fine-grained impact variants around an already useful shot.
        # These often change the length of a cascade without changing its
        # intended source/destination pairing.
        for radius in (4, 8, 12, 16, 24):
            for x_offset, y_offset in (
                (-radius, 0),
                (radius, 0),
                (0, -radius),
                (0, radius),
                (-radius, radius),
                (radius, radius),
            ):
                for kind in (incumbent_kind, 3 - incumbent_kind):
                    value = encode(kind, incumbent_x + x_offset, incumbent_y + y_offset)
                    if value is not None and value not in output:
                        output.append(value)
                        if len(output) >= cap:
                            return output
    for _rank, value in sorted(ranked, key=lambda item: item[0]):
        if value not in output:
            output.append(value)
            if len(output) >= cap:
                break
    return output


def advance_suffix(
    env: Any,
    start: int,
    incumbent: list[int],
    replacement: int,
    maximum: int,
    replacements: Mapping[int, int] | None = None,
) -> tuple[list[int], dict[str, int], bool]:
    emitted = [replacement]
    observation, _reward, terminated, truncated, info = env.step(decode(replacement))
    tick = start + 1
    while not (terminated or truncated) and int(observation["tick"]) < maximum:
        value = (
            int(replacements[tick])
            if replacements is not None and tick in replacements
            else (incumbent[tick] if tick < len(incumbent) else 0)
        )
        emitted.append(value)
        observation, _reward, terminated, truncated, info = env.step(decode(value))
        tick += 1
    return (
        emitted,
        snapshot(observation, info.get("diagnostics")),
        bool(terminated and not truncated),
    )


def objective(
    final: Mapping[str, int], natural: bool, maximum: int, target_score: int
) -> tuple[int, ...]:
    # Completed games are scored before terminal timing.  A late death is useful
    # survival evidence, but must not displace a level-100 replay merely because
    # it happens a few ticks later.  This distinction matters in the endgame,
    # where every branch is naturally terminal (completion or death).
    score = int(final.get("canonical_score", final["score"]))
    level = int(final.get("canonical_level", final.get("level", 0)))
    tick = int(final["tick"])
    completed = natural and level >= 100
    clears = int(final["qualifying_clear_count"])
    gauge = int(final["gauge"])
    return (
        completed and score >= target_score,
        completed,
        score if completed else 0,
        clears if completed else 0,
        gauge if completed else 0,
        -tick if completed else int(tick >= maximum),
        0 if completed else tick,
        score,
        clears,
        gauge,
        natural,
    )


def evaluate_trace(runtime: ExactTrainingRuntime, values: list[int], maximum: int) -> tuple[dict[str, int], bool, str]:
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum}) as session:
        env = session.environment
        observation, _ = env.reset(seed=SEED)
        terminated = truncated = False
        info: Mapping[str, Any] = {}
        for value in values:
            observation, _reward, terminated, truncated, info = env.step(decode(value))
            if terminated or truncated:
                break
        while not (terminated or truncated) and int(observation["tick"]) < maximum:
            observation, _reward, terminated, truncated, info = env.step(Action.wait(1))
        return (
            snapshot(observation, info.get("diagnostics")),
            bool(terminated and not truncated),
            f"0x{int(env.state_hash()):016x}",
        )


def generation(runtime: ExactTrainingRuntime, root: Path, ordinal: int, incumbent: list[int], args: argparse.Namespace) -> list[int]:
    directory = root / "generations" / f"{ordinal:04d}"
    result_path = directory / "result.json"
    winner_path = directory / "winner.u32le"
    if result_path.exists() and winner_path.exists():
        result = json.loads(result_path.read_text())
        data = winner_path.read_bytes()
        if sha256(data) != result["winner_trace_sha256"]:
            raise RuntimeError(f"generation {ordinal} winner trace hash differs")
        return [value for (value,) in struct.iter_unpack("<I", data)]

    rows: list[dict[str, object]] = []
    start_tick = max(0, len(incumbent) - args.window_ticks)
    started = time.monotonic()
    with runtime.open_env(simulation_config={"max_episode_ticks": args.maximum_ticks}) as session:
        env = session.environment
        observation, _ = env.reset(seed=SEED)
        terminated = truncated = False
        info: Mapping[str, Any] = {}
        for tick in range(len(incumbent)):
            if terminated or truncated:
                break
            incumbent_value = incumbent[tick] if tick < len(incumbent) else 0
            timing_specs = timing_jitter_specs(incumbent, tick)
            if should_search_tick(
                tick,
                start_tick,
                args.tick_stride,
                incumbent_value,
                bool(timing_specs and not incumbent_value),
            ):
                actions = candidates(observation, args.candidate_cap, incumbent_value)
                if incumbent_value:
                    actions.append(0)
                branch_specs: list[tuple[int, dict[int, int]]] = [
                    (value, {tick: value}) for value in actions
                ]
                branch_specs.extend(timing_specs[: args.timing_jitter_cap])
                expected_hash = int(env.state_hash())
                with env.fast_checkpoint() as checkpoint:
                    for value, replacements in branch_specs:
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            _suffix, final, natural = advance_suffix(
                                branch,
                                tick,
                                incumbent,
                                value,
                                args.maximum_ticks,
                                replacements,
                            )
                        row = {
                            "tick": tick,
                            "action": value,
                            "replacements": {
                                str(key): replacement
                                for key, replacement in sorted(replacements.items())
                            },
                            "final": final,
                            "natural_terminal": natural,
                        }
                        rows.append(row)
            value = incumbent[tick] if tick < len(incumbent) else 0
            observation, _reward, terminated, truncated, info = env.step(decode(value))
        while not (terminated or truncated) and int(observation["tick"]) < args.maximum_ticks:
            observation, _reward, terminated, truncated, info = env.step(Action.wait(1))

    baseline_final = snapshot(observation, info.get("diagnostics"))
    baseline_natural = bool(terminated and not truncated)
    best_values = list(incumbent[: int(baseline_final["tick"])])
    best_values.extend([0] * max(0, int(baseline_final["tick"]) - len(best_values)))
    best_final, best_natural = baseline_final, baseline_natural
    for row in rows:
        final = row["final"]
        assert isinstance(final, Mapping)
        natural = bool(row["natural_terminal"])
        if objective(
            final, natural, args.maximum_ticks, args.target_score
        ) > objective(
            best_final,
            best_natural,
            args.maximum_ticks,
            args.target_score,
        ):
            best_final = {key: int(value) for key, value in final.items()}
            best_natural = natural
            best_values = reconstructed_candidate(incumbent, row)

    if best_values != incumbent:
        print(
            json.dumps(
                {
                    "generation": ordinal,
                    "winner": best_final,
                    "natural_terminal": best_natural,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    data = trace_bytes(best_values)
    atomic_write(winner_path, data)
    archive: list[dict[str, object]] = []
    for rank, row in enumerate(pareto_rows(rows, args.pareto_width)):
        values = reconstructed_candidate(incumbent, row)
        candidate_data = trace_bytes(values)
        relative = Path("pareto") / f"{rank:02d}.u32le"
        atomic_write(directory / relative, candidate_data)
        archive.append(
            {
                **row,
                "trace": str(relative),
                "trace_sha256": sha256(candidate_data),
            }
        )
    result = {
        "schema": "irisu-exact-suffix-beam-generation-v1",
        "physics_backend": "exact",
        "candidate_strategy": "shot-inclusive-timing-jitter-chain-priority-v5",
        "prefix_replays_per_generation": 1,
        "seed": SEED,
        "generation": ordinal,
        "baseline": baseline_final,
        "baseline_natural_terminal": baseline_natural,
        "search_start_tick": start_tick,
        "evaluated_candidates": len(rows),
        "candidate_rows": rows,
        "pareto_archive": archive,
        "winner": best_final,
        "winner_natural_terminal": best_natural,
        "winner_trace_sha256": sha256(data),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(result_path, result)
    return best_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--window-ticks", type=int, default=768)
    parser.add_argument("--tick-stride", type=int, default=8)
    parser.add_argument("--candidate-cap", type=int, default=48)
    parser.add_argument("--maximum-ticks", type=int, default=120_000)
    parser.add_argument("--target-score", type=int, default=100_000)
    parser.add_argument(
        "--pareto-width",
        type=int,
        default=8,
        help="durable non-dominated score/survival branches retained per generation",
    )
    parser.add_argument("--timing-jitter-cap", type=int, default=8)
    parser.add_argument(
        "--source-replacement",
        action="append",
        default=[],
        metavar="TICK:WORD",
        help="reconstruct a persisted Pareto branch from an evaluated action",
    )
    parser.add_argument("--source-pad-to", type=int)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    source = args.source.resolve(strict=True)
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    args.run_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.run_root / "plan.json"
    incumbent = words(source)
    replacements: dict[int, int] = {}
    for item in args.source_replacement:
        tick_text, word_text = item.split(":", 1)
        replacements[int(tick_text)] = int(word_text)
    if replacements:
        end = max(replacements) + 1
        incumbent.extend([0] * max(0, end - len(incumbent)))
        for tick, value in replacements.items():
            incumbent[tick] = value
    if args.source_pad_to is not None:
        incumbent.extend([0] * max(0, args.source_pad_to - len(incumbent)))
    transformed_source_sha256 = sha256(trace_bytes(incumbent))
    plan = {
        "schema": "irisu-exact-suffix-beam-plan-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "seed": SEED,
        "source": str(source),
        "source_sha256": sha256(source.read_bytes()),
        "source_replacements": {str(key): value for key, value in replacements.items()},
        "source_pad_to": args.source_pad_to,
        "transformed_source_sha256": transformed_source_sha256,
        "worker": str(runtime.worker_path),
        "worker_sha256": runtime.identity.worker_sha256,
        "window_ticks": args.window_ticks,
        "tick_stride": args.tick_stride,
        "candidate_cap": args.candidate_cap,
        "maximum_ticks": args.maximum_ticks,
        "target_score": args.target_score,
        "pareto_width": args.pareto_width,
        "timing_jitter_cap": args.timing_jitter_cap,
    }
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("resume plan differs from existing run")
    if not plan_path.exists():
        atomic_json(plan_path, plan)
    for ordinal in range(1, args.generations + 1):
        incumbent = generation(runtime, args.run_root, ordinal, incumbent, args)
        generation_result = json.loads(
            (args.run_root / "generations" / f"{ordinal:04d}" / "result.json").read_text()
        )
        final = generation_result["winner"]
        natural = bool(generation_result["winner_natural_terminal"])
        if natural and int(final.get("canonical_score", final["score"])) >= args.target_score:
            break
    final, natural, state_hash = evaluate_trace(runtime, incumbent, args.maximum_ticks)
    data = trace_bytes(incumbent[: int(final["tick"])])
    replay = HEADER.pack(
        SEED,
        final.get("canonical_level", final["level"]),
        final.get("canonical_score", final["score"]),
        final.get("canonical_highest_chain", final["highest_chain"]),
        0,
    ) + bytes(32) + data
    atomic_write(args.run_root / "best.u32le", data)
    atomic_write(args.run_root / "best.rpy", replay)
    summary = {
        "schema": "irisu-exact-suffix-beam-summary-v1",
        "physics_backend": "exact",
        "seed": SEED,
        "natural_terminal": natural,
        "target_score": args.target_score,
        "target_reached": natural
        and int(final.get("canonical_score", final["score"])) >= args.target_score,
        "final": final,
        "final_state_u64": state_hash,
        "trace_sha256": sha256(data),
        "replay_sha256": sha256(replay),
        "exact_runtime": runtime.identity.manifest(),
    }
    atomic_json(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
