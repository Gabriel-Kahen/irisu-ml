#!/usr/bin/env python3
"""Measure the R1 teacher-state collection path, including owned storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import numpy as np

from irisu_env import PaddedVectorEnv
from irisu_rl import (
    ACTOR_VISION_V1,
    MacroVectorAdapter,
    RolloutBuffer,
    SeedAllocator,
    SemanticAction,
    TeacherStateEncoder,
    TEACHER_V1,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--worker")
    parser.add_argument("--library")
    parser.add_argument(
        "--diagnostic-portable",
        action="store_true",
        help="use approximate portable physics for throughput diagnostics only",
    )
    args = parser.parse_args()
    backend = "portable" if args.diagnostic_portable else "exact"
    exact_runtime = None
    if backend == "exact":
        if args.library is not None:
            parser.error("--library is only valid with --diagnostic-portable")
        if not args.worker or not Path(args.worker).is_absolute():
            parser.error("exact collection requires an explicit absolute --worker path")
        exact_runtime = ExactTrainingRuntime(args.worker)
    else:
        if args.worker is not None:
            parser.error("--worker cannot be combined with --diagnostic-portable")
        if args.library is None or not Path(args.library).is_absolute():
            parser.error(
                "--diagnostic-portable requires an explicit absolute --library path"
            )
    rng = np.random.default_rng(20260722)
    exact_training_provenance = None
    with ExitStack() as stack:
        if exact_runtime is not None:
            exact_session = stack.enter_context(
                exact_runtime.open_vector(
                    args.lanes,
                    simulation_config={"max_episode_ticks": 100_000},
                )
            )
            vector = exact_session.environment
            exact_training_provenance = exact_session.provenance_manifest
        else:
            vector = stack.enter_context(
                PaddedVectorEnv(
                    args.lanes,
                    library_path=args.library,
                    physics_backend="portable",
                    config={"max_episode_ticks": 100_000},
                )
            )
        adapter = MacroVectorAdapter(
            vector,
            encoder=TeacherStateEncoder(),
            seed_allocator=SeedAllocator(key=20260722),
        )
        initial = adapter.reset()
        buffer = RolloutBuffer(args.lanes * args.iterations, initial.schema)
        invalid = 0
        native_ticks = 0
        primitive_actions = 0
        config_hashes: set[int] = set()
        started = time.perf_counter()
        for _ in range(args.iterations):
            actions = []
            for draw in rng.integers(0, 10, size=args.lanes):
                if draw < 7:
                    actions.append(SemanticAction.wait((1, 2, 4, 8, 16, 32)[draw % 6]))
                elif draw < 9:
                    actions.append(
                        SemanticAction.weak(float(rng.random()), float(rng.random()))
                    )
                else:
                    actions.append(
                        SemanticAction.strong(float(rng.random()), float(rng.random()))
                    )
            transitions = adapter.step(actions)
            for transition in transitions:
                buffer.append(transition)
                invalid += int(transition.diagnostics.invalid_action)
                native_ticks += transition.elapsed_ticks
                primitive_actions += len(transition.primitive_trace)
                config_hashes.add(transition.diagnostics.config_hash)
        buffer.seal(adapter.current_observation)
        elapsed = time.perf_counter() - started
    decisions = args.lanes * args.iterations
    result = {
        "schema": "rl-r1-teacher-rollout-benchmark-v1",
        "recorded_at": date.today().isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "parameters": {
            "backend": backend,
            "diagnostic_portable": args.diagnostic_portable,
            "lanes": args.lanes,
            "iterations": args.iterations,
            "seed": 20260722,
        },
        "results": {
            "semantic_decisions": decisions,
            "elapsed_seconds": elapsed,
            "semantic_decisions_per_second": decisions / elapsed,
            "native_ticks": native_ticks,
            "native_ticks_per_second": native_ticks / elapsed,
            "primitive_actions": primitive_actions,
            "primitive_actions_per_second": primitive_actions / elapsed,
            "invalid_actions": invalid,
            "stored_transitions": buffer.size,
            "unique_initial_and_autoreset_seeds": adapter.seed_allocator.cursor,
        },
        "identity": {
            "exact_training_provenance": exact_training_provenance,
            "actor_schema_sha256": ACTOR_VISION_V1.sha256,
            "teacher_schema_sha256": TEACHER_V1.sha256,
            "config_hashes": sorted(config_hashes),
            "uv_lock_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "uv.lock").read_bytes()
            ).hexdigest(),
        },
        "scope": f"{backend} padded vector + semantic macros + vectorized teacher encoding + owned rollout writes",
        "interpretation": "R1 teacher-state engineering throughput only; excludes actor tracking and model inference and is not evidence of policy quality or sim-to-game fidelity",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
