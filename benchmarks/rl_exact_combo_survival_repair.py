#!/usr/bin/env python3
"""Repair downstream deaths caused by a locally profitable combo mutation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_global_combo_search as combo  # noqa: E402
import rl_exact_suffix_beam as beam  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


DEFAULT_GLOBAL_ROOT = (
    ROOT / "artifacts/r3/development/exact-300k-global-combo-253487-20260810-001"
)
DEFAULT_SOURCE = (
    ROOT
    / "artifacts/r3/development/exact-300k-multi-parent-beam-20260810-001"
    / "best.u32le"
)
DEFAULT_ROOT = (
    ROOT / "artifacts/r3/development/exact-300k-combo-survival-repair-20260810-001"
)


def select_global_mutation(
    rows: list[dict[str, Any]], *, minimum_tick: int = 0, maximum_tick: int | None = None
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if not bool(row.get("terminated", False))
        and int(row.get("combo_power_gain", 0)) > 0
        and int(row.get("score_gain", 0)) > 0
        and int(row["tick"]) >= minimum_tick
        and (maximum_tick is None or int(row["tick"]) <= maximum_tick)
    ]
    if not eligible:
        raise ValueError("global result has no locally profitable surviving mutation")
    return max(eligible, key=combo.local_key)


def repair_should_continue(
    before: Mapping[str, int], after: Mapping[str, int], natural: bool
) -> bool:
    after_level = int(after.get("canonical_level", after["level"]))
    if natural and after_level >= 100:
        return False
    return int(after["tick"]) > int(before["tick"])


def replay(
    runtime: ExactTrainingRuntime,
    values: list[int],
    maximum: int,
) -> tuple[dict[str, int], bool, str, list[int]]:
    final, natural, state_hash = beam.evaluate_trace(runtime, values, maximum)
    return final, natural, state_hash, values[: int(final["tick"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", type=Path, default=DEFAULT_GLOBAL_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--worker", type=Path, default=beam.WORKER)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--window-ticks", type=int, default=768)
    parser.add_argument("--tick-stride", type=int, default=8)
    parser.add_argument("--candidate-cap", type=int, default=24)
    parser.add_argument("--timing-jitter-cap", type=int, default=8)
    parser.add_argument("--pareto-width", type=int, default=8)
    parser.add_argument("--maximum-ticks", type=int, default=130_000)
    parser.add_argument("--target-score", type=int, default=300_000)
    parser.add_argument("--minimum-mutation-tick", type=int, default=0)
    parser.add_argument("--maximum-mutation-tick", type=int)
    args = parser.parse_args()

    args.global_root = args.global_root.resolve(strict=True)
    args.source = args.source.resolve(strict=True)
    args.worker = args.worker.resolve(strict=True)
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    global_result_path = args.global_root / "generations/0001/result.json"
    global_result = json.loads(global_result_path.read_text())
    selected = select_global_mutation(
        global_result["local_rows"],
        minimum_tick=args.minimum_mutation_tick,
        maximum_tick=args.maximum_mutation_tick,
    )
    replacements = {
        int(key): int(value) for key, value in selected["replacements"].items()
    }
    source = beam.words(args.source)
    mutated = combo.materialize(source, replacements, len(source))
    runtime = ExactTrainingRuntime(args.worker)
    seed_final, seed_natural, seed_hash, incumbent = replay(
        runtime, mutated, args.maximum_ticks
    )
    if not seed_natural or int(seed_final.get("canonical_level", seed_final["level"])) >= 100:
        raise RuntimeError("selected mutation did not reproduce the expected pre-completion death")

    plan = {
        "schema": "irisu-exact-combo-survival-repair-plan-v1",
        "physics_backend": "exact",
        "seed": beam.SEED,
        "source": str(args.source),
        "source_sha256": beam.sha256(args.source.read_bytes()),
        "global_result": str(global_result_path),
        "global_result_sha256": beam.sha256(global_result_path.read_bytes()),
        "selected_mutation": {
            "kind": selected["kind"],
            "tick": selected["tick"],
            "combo_power_gain": selected["combo_power_gain"],
            "score_gain": selected["score_gain"],
            "max_chain": selected["max_chain"],
            "replacements": selected["replacements"],
        },
        "seed_death": seed_final,
        "seed_state_u64": seed_hash,
        "worker": str(args.worker),
        "worker_sha256": runtime.identity.worker_sha256,
        "generations": args.generations,
        "window_ticks": args.window_ticks,
        "tick_stride": args.tick_stride,
        "candidate_cap": args.candidate_cap,
        "timing_jitter_cap": args.timing_jitter_cap,
        "pareto_width": args.pareto_width,
        "maximum_ticks": args.maximum_ticks,
        "target_score": args.target_score,
        "minimum_mutation_tick": args.minimum_mutation_tick,
        "maximum_mutation_tick": args.maximum_mutation_tick,
    }
    plan_path = args.run_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("resume plan differs from existing run")
    if not plan_path.exists():
        beam.atomic_json(plan_path, plan)
        beam.atomic_write(args.run_root / "mutated-death.u32le", beam.trace_bytes(incumbent))

    search_args = SimpleNamespace(
        window_ticks=args.window_ticks,
        tick_stride=args.tick_stride,
        candidate_cap=args.candidate_cap,
        maximum_ticks=args.maximum_ticks,
        target_score=args.target_score,
        pareto_width=args.pareto_width,
        timing_jitter_cap=args.timing_jitter_cap,
    )
    history: list[dict[str, Any]] = []
    current_final = seed_final
    current_natural = seed_natural
    for ordinal in range(1, args.generations + 1):
        before = current_final
        incumbent = beam.generation(
            runtime, args.run_root / "repair", ordinal, incumbent, search_args
        )
        current_final, current_natural, state_hash, incumbent = replay(
            runtime, incumbent, args.maximum_ticks
        )
        row = {
            "generation": ordinal,
            "before": before,
            "after": current_final,
            "natural_terminal": current_natural,
            "final_state_u64": state_hash,
            "trace_sha256": beam.sha256(beam.trace_bytes(incumbent)),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not repair_should_continue(before, current_final, current_natural):
            break

    final_score = int(current_final.get("canonical_score", current_final["score"]))
    final_level = int(current_final.get("canonical_level", current_final["level"]))
    final_highest = int(
        current_final.get("canonical_highest_chain", current_final["highest_chain"])
    )
    trace = beam.trace_bytes(incumbent)
    replay_bytes = (
        beam.HEADER.pack(beam.SEED, final_level, final_score, final_highest, 0)
        + bytes(32)
        + trace
    )
    beam.atomic_write(args.run_root / "best.u32le", trace)
    beam.atomic_write(args.run_root / "best.rpy", replay_bytes)
    summary = {
        "schema": "irisu-exact-combo-survival-repair-summary-v1",
        "physics_backend": "exact",
        "seed_death": seed_final,
        "selected_mutation": plan["selected_mutation"],
        "repairs": history,
        "final": current_final,
        "natural_terminal": current_natural,
        "survival_gain_ticks": int(current_final["tick"]) - int(seed_final["tick"]),
        "target_score": args.target_score,
        "target_reached": current_natural
        and final_level >= 100
        and final_score >= args.target_score,
        "trace_sha256": beam.sha256(trace),
        "replay_sha256": beam.sha256(replay_bytes),
        "exact_runtime": runtime.identity.manifest(),
    }
    beam.atomic_json(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
