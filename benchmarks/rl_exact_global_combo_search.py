#!/usr/bin/env python3
"""Exact two-stage search for longer combos throughout a replay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_suffix_beam as common  # noqa: E402
from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "artifacts/r3/development/exact-300k-multi-parent-beam-20260810-001"
    / "best.u32le"
)
DEFAULT_ROOT = (
    ROOT / "artifacts/r3/development/exact-300k-global-combo-20260810-001"
)
DELAY_TICKS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
INJECTION_LEADS = (8, 16, 24, 32, 48, 64, 96)


def materialize(values: list[int], replacements: Mapping[int, int], end: int) -> list[int]:
    result = list(values[:end])
    result.extend([0] * max(0, end - len(result)))
    for tick, value in replacements.items():
        if 0 <= tick < end:
            result[tick] = value
    return result


def combo_power(confirmations: list[tuple[int, int]], start: int, end: int) -> int:
    return sum(chain * chain for tick, chain in confirmations if start < tick <= end)


def diverse_ticks(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Round-robin already-ranked rows across 10k-tick bands."""
    bands: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bands[int(row["tick"]) // 10_000].append(row)
    selected: list[dict[str, Any]] = []
    rank = 0
    while len(selected) < cap:
        added = False
        for key in sorted(bands):
            if rank < len(bands[key]):
                selected.append(bands[key][rank])
                added = True
                if len(selected) >= cap:
                    break
        if not added:
            break
        rank += 1
    return selected


def select_opportunities(
    values: list[int],
    confirmations: list[tuple[int, int]],
    *,
    lookahead: int,
    cap: int,
    excluded_suffix: int,
    minimum_tick: int = 0,
) -> list[dict[str, int | str]]:
    """Rank shots and empty pre-clear frames near chain-1/2 confirmations."""
    latest = max(0, len(values) - excluded_suffix)
    rows: dict[tuple[str, int], dict[str, int | str]] = {}
    low = [(tick, chain) for tick, chain in confirmations if chain <= 2]
    for tick, value in enumerate(values[:latest]):
        if tick < minimum_tick:
            continue
        if not value:
            continue
        nearby = [
            (clear_tick, chain)
            for clear_tick, chain in low
            if 0 < clear_tick - (tick + 1) <= lookahead
        ]
        if nearby:
            weight = sum((4 - chain) ** 2 for _clear_tick, chain in nearby)
            distance = min(clear_tick - (tick + 1) for clear_tick, _chain in nearby)
            rows[("shot", tick)] = {
                "kind": "shot",
                "tick": tick,
                "weight": weight,
                "distance": distance,
            }
    for clear_tick, chain in low:
        for lead in INJECTION_LEADS:
            tick = clear_tick - lead - 1
            if minimum_tick <= tick < latest and values[tick] == 0:
                key = ("inject", tick)
                weight = (4 - chain) ** 2
                prior = rows.get(key)
                if prior is None or int(prior["weight"]) < weight:
                    rows[key] = {
                        "kind": "inject",
                        "tick": tick,
                        "weight": weight,
                        "distance": lead,
                    }
    ranked = sorted(
        rows.values(),
        key=lambda row: (
            -int(row["weight"]),
            int(row["distance"]),
            int(row["tick"]),
        ),
    )
    shot_rows = [row for row in ranked if row["kind"] == "shot"]
    injection_rows = [row for row in ranked if row["kind"] == "inject"]
    injection_cap = min(len(injection_rows), max(1, cap // 3))
    selected = diverse_ticks(shot_rows, cap - injection_cap)
    selected.extend(diverse_ticks(injection_rows, injection_cap))
    return selected


def local_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        not bool(row["terminated"]),
        int(row["combo_power_gain"]),
        int(row["max_chain"]),
        int(row["score_gain"]),
        int(row["clear_gain"]),
        int(row["gauge_gain"]),
    )


def final_key(final: Mapping[str, int], natural: bool) -> tuple[int, ...]:
    score = int(final.get("canonical_score", final["score"]))
    level = int(final.get("canonical_level", final["level"]))
    highest = int(final.get("canonical_highest_chain", final["highest_chain"]))
    clears = int(final.get("canonical_clears", final["qualifying_clear_count"]))
    complete = natural and level >= 100
    return (
        complete,
        score if complete else 0,
        highest if complete else 0,
        clears if complete else 0,
        int(final["gauge"]) if complete else 0,
        int(final["tick"]),
        score,
    )


def profile(
    runtime: ExactTrainingRuntime, values: list[int], maximum: int
) -> tuple[list[dict[str, int]], list[tuple[int, int]], dict[str, int], bool]:
    states: list[dict[str, int]] = []
    confirmations: list[tuple[int, int]] = []
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum}) as session:
        env = session.environment
        observation, _ = env.reset(seed=common.SEED)
        states.append(common.snapshot(observation))
        terminated = truncated = False
        info: Mapping[str, Any] = {}
        for value in values:
            if terminated or truncated:
                break
            observation, _reward, terminated, truncated, info = env.step(common.decode(value))
            for event in info.get("events", ()):
                if event.get("kind_name") == "confirmed":
                    confirmations.append((int(event["tick"]), int(event["value"])))
            states.append(common.snapshot(observation, info.get("diagnostics")))
        while not (terminated or truncated) and int(observation["tick"]) < maximum:
            observation, _reward, terminated, truncated, info = env.step(Action.wait(1))
            for event in info.get("events", ()):
                if event.get("kind_name") == "confirmed":
                    confirmations.append((int(event["tick"]), int(event["value"])))
            states.append(common.snapshot(observation, info.get("diagnostics")))
        return states, confirmations, states[-1], bool(terminated and not truncated)


def mutation_specs(
    observation: Mapping[str, Any],
    values: list[int],
    opportunity: Mapping[str, int | str],
    candidate_cap: int,
) -> list[tuple[str, dict[int, int]]]:
    tick = int(opportunity["tick"])
    output: list[tuple[str, dict[int, int]]] = []
    if opportunity["kind"] == "shot":
        original = values[tick]
        for delay in DELAY_TICKS:
            destination = tick + delay
            if destination < len(values) and values[destination] == 0:
                output.append((f"delay-{delay}", {tick: 0, destination: original}))
        for value in common.candidates(observation, candidate_cap, original):
            if value != original:
                output.append(("retarget", {tick: value}))
    else:
        for value in common.candidates(observation, candidate_cap, 0):
            output.append(("pair-injection", {tick: value}))
    unique: list[tuple[str, dict[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for label, replacements in output:
        identity = tuple(sorted(replacements.items()))
        if identity not in seen:
            seen.add(identity)
            unique.append((label, replacements))
    return unique


def advance(
    env: Any,
    values: list[int],
    start: int,
    end: int,
    replacements: Mapping[int, int],
    collect_confirmations: bool,
) -> tuple[dict[str, int], bool, list[tuple[int, int]]]:
    observation: Mapping[str, Any] = {}
    info: Mapping[str, Any] = {}
    terminated = truncated = False
    confirmations: list[tuple[int, int]] = []
    for tick in range(start, end):
        value = replacements.get(tick, values[tick] if tick < len(values) else 0)
        observation, _reward, terminated, truncated, info = env.step(common.decode(value))
        if collect_confirmations:
            for event in info.get("events", ()):
                if event.get("kind_name") == "confirmed":
                    confirmations.append((int(event["tick"]), int(event["value"])))
        if terminated or truncated:
            break
    return (
        common.snapshot(observation, info.get("diagnostics")),
        bool(terminated and not truncated),
        confirmations,
    )


def search_generation(
    runtime: ExactTrainingRuntime,
    values: list[int],
    args: argparse.Namespace,
    directory: Path,
) -> tuple[list[int], dict[str, Any]]:
    started = time.monotonic()
    states, confirmations, baseline_final, baseline_natural = profile(
        runtime, values, args.maximum_ticks
    )
    opportunities = select_opportunities(
        values,
        confirmations,
        lookahead=args.lookahead,
        cap=args.opportunity_cap,
        excluded_suffix=args.excluded_suffix,
        minimum_tick=args.minimum_opportunity_tick,
    )
    by_tick = {int(row["tick"]): row for row in opportunities}
    local_rows: list[dict[str, Any]] = []
    with runtime.open_env(simulation_config={"max_episode_ticks": args.maximum_ticks}) as session:
        env = session.environment
        observation, _ = env.reset(seed=common.SEED)
        for tick, source_value in enumerate(values):
            if tick in by_tick:
                horizon = min(len(states) - 1, tick + args.local_horizon)
                baseline_power = combo_power(confirmations, tick, horizon)
                expected_hash = int(env.state_hash())
                specs = mutation_specs(
                    observation, values, by_tick[tick], args.candidate_cap
                )
                with env.fast_checkpoint() as checkpoint:
                    for label, replacements in specs:
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            final, natural, branch_confirms = advance(
                                branch,
                                values,
                                tick,
                                horizon,
                                replacements,
                                True,
                            )
                        baseline = states[min(horizon, len(states) - 1)]
                        branch_power = combo_power(branch_confirms, tick, horizon)
                        local_rows.append(
                            {
                                "kind": label,
                                "tick": tick,
                                "opportunity": by_tick[tick],
                                "replacements": {
                                    str(key): value
                                    for key, value in sorted(replacements.items())
                                },
                                "horizon": horizon,
                                "terminated": natural,
                                "score_gain": int(final["score"]) - int(baseline["score"]),
                                "clear_gain": int(final["qualifying_clear_count"])
                                - int(baseline["qualifying_clear_count"]),
                                "gauge_gain": int(final["gauge"]) - int(baseline["gauge"]),
                                "combo_power": branch_power,
                                "combo_power_gain": branch_power - baseline_power,
                                "max_chain": max(
                                    (chain for _event_tick, chain in branch_confirms),
                                    default=0,
                                ),
                                "local_final": final,
                            }
                        )
            observation, _reward, terminated, truncated, _info = env.step(
                common.decode(source_value)
            )
            if terminated or truncated:
                break

    promising = [
        row
        for row in local_rows
        if not bool(row["terminated"])
        and (int(row["combo_power_gain"]) > 0 or int(row["score_gain"]) > 0)
    ]
    ranked_promising = sorted(promising, key=local_key, reverse=True)
    shortlisted = diverse_ticks(ranked_promising, args.full_cap)
    full_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in shortlisted:
        full_by_tick[int(row["tick"])].append(row)

    full_rows: list[dict[str, Any]] = []
    with runtime.open_env(simulation_config={"max_episode_ticks": args.maximum_ticks}) as session:
        env = session.environment
        observation, _ = env.reset(seed=common.SEED)
        for tick, source_value in enumerate(values):
            if tick in full_by_tick:
                expected_hash = int(env.state_hash())
                with env.fast_checkpoint() as checkpoint:
                    for local in full_by_tick[tick]:
                        replacements = {
                            int(key): int(value)
                            for key, value in local["replacements"].items()
                        }
                        with checkpoint.branch() as branch:
                            if int(branch.state_hash()) != expected_hash:
                                raise RuntimeError("exact fast branch state mismatch")
                            final, natural, _ = advance(
                                branch,
                                values,
                                tick,
                                args.maximum_ticks,
                                replacements,
                                False,
                            )
                        full_rows.append(
                            {**local, "final": final, "natural_terminal": natural}
                        )
            observation, _reward, terminated, truncated, _info = env.step(
                common.decode(source_value)
            )
            if terminated or truncated:
                break

    winner_final = baseline_final
    winner_natural = baseline_natural
    winner_replacements: dict[int, int] = {}
    for row in full_rows:
        if final_key(row["final"], bool(row["natural_terminal"])) > final_key(
            winner_final, winner_natural
        ):
            winner_final = row["final"]
            winner_natural = bool(row["natural_terminal"])
            winner_replacements = {
                int(key): int(value) for key, value in row["replacements"].items()
            }
    winner_values = materialize(values, winner_replacements, int(winner_final["tick"]))
    verified, verified_natural, state_hash = common.evaluate_trace(
        runtime, winner_values, args.maximum_ticks
    )
    if verified != winner_final or verified_natural != winner_natural:
        raise RuntimeError("independent exact replay disagrees with selected winner")
    result = {
        "schema": "irisu-exact-global-combo-generation-v1",
        "physics_backend": "exact",
        "baseline": baseline_final,
        "baseline_natural_terminal": baseline_natural,
        "confirmation_count": len(confirmations),
        "opportunities": opportunities,
        "local_candidate_count": len(local_rows),
        "promising_candidate_count": len(promising),
        "full_candidate_count": len(full_rows),
        "local_rows": local_rows,
        "full_rows": full_rows,
        "winner_replacements": {
            str(key): value for key, value in sorted(winner_replacements.items())
        },
        "winner": verified,
        "winner_natural_terminal": verified_natural,
        "final_state_u64": state_hash,
        "wall_seconds": time.monotonic() - started,
    }
    common.atomic_write(directory / "winner.u32le", common.trace_bytes(winner_values))
    common.atomic_json(directory / "result.json", result)
    return winner_values, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worker", type=Path, default=common.WORKER)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--maximum-ticks", type=int, default=130_000)
    parser.add_argument("--target-score", type=int, default=300_000)
    parser.add_argument("--excluded-suffix", type=int, default=768)
    parser.add_argument("--lookahead", type=int, default=192)
    parser.add_argument("--opportunity-cap", type=int, default=96)
    parser.add_argument("--candidate-cap", type=int, default=8)
    parser.add_argument("--local-horizon", type=int, default=8_192)
    parser.add_argument("--full-cap", type=int, default=48)
    parser.add_argument("--minimum-opportunity-tick", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be an unsigned 32-bit integer")
    if args.minimum_opportunity_tick < 0:
        parser.error("--minimum-opportunity-tick must be nonnegative")
    common.SEED = args.seed

    args.source = args.source.resolve(strict=True)
    args.run_root = args.run_root.resolve()
    args.worker = args.worker.resolve(strict=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    runtime = ExactTrainingRuntime(args.worker)
    plan = {
        "schema": "irisu-exact-global-combo-plan-v1",
        "physics_backend": "exact",
        "seed": common.SEED,
        "source": str(args.source),
        "source_sha256": common.sha256(args.source.read_bytes()),
        "worker": str(args.worker),
        "worker_sha256": runtime.identity.worker_sha256,
        "generations": args.generations,
        "maximum_ticks": args.maximum_ticks,
        "target_score": args.target_score,
        "excluded_suffix": args.excluded_suffix,
        "lookahead": args.lookahead,
        "opportunity_cap": args.opportunity_cap,
        "candidate_cap": args.candidate_cap,
        "local_horizon": args.local_horizon,
        "full_cap": args.full_cap,
        "minimum_opportunity_tick": args.minimum_opportunity_tick,
    }
    plan_path = args.run_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("resume plan differs from existing run")
    if not plan_path.exists():
        common.atomic_json(plan_path, plan)

    values = common.words(args.source)
    final_result: dict[str, Any] = {}
    for ordinal in range(1, args.generations + 1):
        directory = args.run_root / "generations" / f"{ordinal:04d}"
        result_path = directory / "result.json"
        winner_path = directory / "winner.u32le"
        if result_path.exists() and winner_path.exists():
            final_result = json.loads(result_path.read_text())
            values = common.words(winner_path)
        else:
            values, final_result = search_generation(runtime, values, args, directory)
        winner = final_result["winner"]
        score = int(winner.get("canonical_score", winner["score"]))
        print(
            json.dumps(
                {
                    "generation": ordinal,
                    "score": score,
                    "highest_chain": winner.get(
                        "canonical_highest_chain", winner["highest_chain"]
                    ),
                    "replacement": final_result["winner_replacements"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if score >= args.target_score or not final_result["winner_replacements"]:
            break

    verified = final_result["winner"]
    replay = common.HEADER.pack(
        common.SEED,
        int(verified.get("canonical_level", verified["level"])),
        int(verified.get("canonical_score", verified["score"])),
        int(verified.get("canonical_highest_chain", verified["highest_chain"])),
        0,
    ) + bytes(32) + common.trace_bytes(values[: int(verified["tick"])])
    common.atomic_write(args.run_root / "best.u32le", common.trace_bytes(values))
    common.atomic_write(args.run_root / "best.rpy", replay)
    summary = {
        "schema": "irisu-exact-global-combo-summary-v1",
        "physics_backend": "exact",
        "seed": common.SEED,
        "final": verified,
        "natural_terminal": final_result["winner_natural_terminal"],
        "target_score": args.target_score,
        "target_reached": int(verified.get("canonical_score", verified["score"]))
        >= args.target_score,
        "trace_sha256": common.sha256(common.trace_bytes(values)),
        "replay_sha256": common.sha256(replay),
        "exact_runtime": runtime.identity.manifest(),
    }
    common.atomic_json(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
