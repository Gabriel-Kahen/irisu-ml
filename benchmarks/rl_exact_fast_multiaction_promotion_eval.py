#!/usr/bin/env python3
"""Evidence-bound exact promotion shards for FastMultiActionPlanner.

All evaluation seeds are supplied by the caller.  A shard may contain any
nonempty subset, but only a report with exactly 20 episodes can itself pass the
canonical 16-of-20, median-100k, zero-live-invalid promotion contract.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    FastMultiActionConfig,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402
from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    FULL_GAME_TICKS,
    PolicyBundle,
    bind_report_content_sha256,
    canonical_sha256,
    file_sha256,
    load_declared_training_seeds,
    load_policy_bundle,
    parse_seeds,
    validate_contract,
)
from rl_exact_fast_multiaction_eval import run_episode  # noqa: E402


FORMAT = "irisu-exact-fast-multiaction-promotion-shard-v1"
TARGET_SCORE = 100_000
REQUIRED_EPISODES = 20
REQUIRED_SUCCESSES = 16
CONTRACT = {
    "required_episode_count": REQUIRED_EPISODES,
    "required_success_count": REQUIRED_SUCCESSES,
    "target_score": TARGET_SCORE,
    "require_median_at_or_above_target": True,
    "require_zero_live_invalid_actions": True,
}
BRANCH_ERROR_POLICY = {
    "version": "exact-branch-error-conservative-wait-v1",
    "caught_exception": "irisu_env.exact_ipc.ExactWorkerError",
    "action": "restore-policy-and-execute-wait",
    "require_live_parent_state_hash_unchanged": True,
}


def evaluate_promotion_shard(
    runtime: ExactTrainingRuntime,
    bundle: PolicyBundle,
    *,
    declared_training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    maximum_ticks: int = FULL_GAME_TICKS,
    planner_config: FastMultiActionConfig | None = None,
    trace_interval_ticks: int = 10_000,
    maximum_logged_queries: int = 128,
    episode_callback: Callable[[Mapping[str, object]], None] | None = None,
    runner_path: Path | None = None,
    evaluator_engine_path: Path | None = None,
) -> dict[str, object]:
    declared, evaluation = validate_contract(
        bundle, declared_training_seeds, evaluation_seeds
    )
    config = FastMultiActionConfig() if planner_config is None else planner_config
    for name, value in (
        ("maximum_ticks", maximum_ticks),
        ("trace_interval_ticks", trace_interval_ticks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(maximum_logged_queries, bool)
        or not isinstance(maximum_logged_queries, int)
        or maximum_logged_queries < 0
    ):
        raise ValueError("maximum_logged_queries must be nonnegative")

    started = time.monotonic()
    episodes: list[dict[str, object]] = []
    runtime_provenance: Mapping[str, Any] | None = None
    runtime_provenance_sha256: str | None = None
    for seed in evaluation:
        episode, current_provenance = run_episode(
            runtime,
            bundle,
            seed,
            maximum_ticks=maximum_ticks,
            planner_config=config,
            target_score=TARGET_SCORE,
            trace_interval_ticks=trace_interval_ticks,
            maximum_logged_queries=maximum_logged_queries,
            recover_exact_branch_errors=True,
        )
        current_sha = canonical_sha256(current_provenance)
        if (
            runtime_provenance_sha256 is not None
            and runtime_provenance_sha256 != current_sha
        ):
            raise RuntimeError("exact runtime provenance changed between episodes")
        runtime_provenance = current_provenance
        runtime_provenance_sha256 = current_sha
        episodes.append(episode)
        if episode_callback is not None:
            episode_callback(episode)

    scores = [int(value["score"]) for value in episodes]
    successes = sum(score >= TARGET_SCORE for score in scores)
    invalid = sum(int(value["invalid_actions"]) for value in episodes)
    median_score = float(statistics.median(scores))
    aggregate_probe_modes: Counter[str] = Counter()
    aggregate_probe_horizons: Counter[str] = Counter()
    for episode in episodes:
        aggregate_probe_modes.update(episode["planner_probe_mode_counts"])
        aggregate_probe_horizons.update(episode["planner_probe_horizon_counts"])
    bound_runner = (
        Path(__file__).resolve()
        if runner_path is None
        else runner_path.resolve(strict=True)
    )
    bound_engine = (
        SCRIPT_DIR / "rl_exact_fast_multiaction_eval.py"
        if evaluator_engine_path is None
        else evaluator_engine_path.resolve(strict=True)
    )
    planner_source = ROOT / "python/irisu_pointer/fast_multiaction_planner.py"
    report: dict[str, object] = {
        "format": FORMAT,
        "physics_backend": "exact",
        "deterministic_policy": True,
        "promotion_contract": dict(CONTRACT),
        "promotion_eligible": len(episodes) == REQUIRED_EPISODES,
        "training_seeds": list(declared),
        "training_seeds_sha256": canonical_sha256(list(declared)),
        "evaluation_seeds": list(evaluation),
        "evaluation_seeds_sha256": canonical_sha256(list(evaluation)),
        "training_evaluation_overlap": [],
        "maximum_ticks": maximum_ticks,
        "planner_config": config.manifest(),
        "planner_objective": (
            "wait-relative-reserve-filter-then-survival-clears-score-gauge-v1"
        ),
        "branch_error_policy": dict(BRANCH_ERROR_POLICY),
        "branch_error_recovery_enabled": True,
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_sha256": bundle.model_sha256,
        "checkpoint_metadata": dict(bundle.metadata),
        "checkpoint_metadata_sha256": canonical_sha256(bundle.metadata),
        "inference_config": dict(bundle.inference_config),
        "exact_runtime": dict(runtime_provenance or {}),
        "runtime_hashes": {
            "worker_sha256": runtime.identity.worker_sha256,
            "exact_library_sha256": runtime.identity.exact_library_sha256,
            "identity_config_sha256": runtime.identity.config_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
        },
        "planner_source_sha256": file_sha256(planner_source),
        "evaluator_engine_sha256": file_sha256(bound_engine),
        "runner_sha256": file_sha256(bound_runner),
        "episodes": episodes,
        "episode_count": len(episodes),
        "scores": scores,
        "median_score": median_score,
        "mean_score": statistics.fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "success_count": successes,
        "success_fraction_at_or_above_target": successes / len(episodes),
        "invalid_actions": invalid,
        "counterfactual_invalid_actions": sum(
            int(value["counterfactual_invalid_actions"]) for value in episodes
        ),
        "planner_probe_mode_counts": dict(sorted(aggregate_probe_modes.items())),
        "planner_probe_horizon_counts": dict(
            sorted(aggregate_probe_horizons.items())
        ),
        "planner_probe_mode_switch_count": sum(
            len(value["planner_probe_mode_switches"]) for value in episodes
        ),
        "exact_branch_errors": sum(
            int(value["exact_branch_errors"]) for value in episodes
        ),
        "passed": len(episodes) == REQUIRED_EPISODES
        and successes >= REQUIRED_SUCCESSES
        and median_score >= TARGET_SCORE
        and invalid == 0,
        "wall_seconds": time.monotonic() - started,
    }
    bind_report_content_sha256(report)
    return report


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--training-seed-manifest", type=Path, required=True)
    result.add_argument("--seeds", type=parse_seeds, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--episode-dir", type=Path)
    result.add_argument("--maximum-ticks", type=int, default=FULL_GAME_TICKS)
    result.add_argument("--probe-ticks", type=int, default=256)
    result.add_argument("--long-probe-ticks", type=int, default=512)
    result.add_argument("--wait-ticks", type=int, default=16)
    result.add_argument("--low-gauge-threshold", type=int, default=20_000)
    result.add_argument("--low-gauge-exit-threshold", type=int, default=30_000)
    result.add_argument("--top-k-pairs", type=int, default=2)
    result.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    result.add_argument("--rescue-score-margin", type=int, default=500)
    result.add_argument("--gauge-advantage", type=int, default=1)
    result.add_argument("--trace-interval-ticks", type=int, default=10_000)
    result.add_argument("--maximum-logged-queries", type=int, default=128)
    result.add_argument("--act-logit-bias", type=float)
    result.add_argument("--cooldown-ticks", type=int)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.worker.is_absolute():
        parser().error("--worker must be absolute")
    if args.output.exists():
        parser().error("--output must not already exist")
    if args.episode_dir is not None and args.episode_dir.exists():
        parser().error("--episode-dir must not already exist")
    inference_options = {
        key: value
        for key, value in {
            "act_logit_bias": args.act_logit_bias,
            "cooldown_ticks": args.cooldown_ticks,
        }.items()
        if value is not None
    }
    bundle = load_policy_bundle(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        inference_options=inference_options,
    )
    training = load_declared_training_seeds(
        args.training_seed_manifest.resolve(strict=True)
    )
    if args.episode_dir is not None:
        args.episode_dir.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []

    def record_episode(row: Mapping[str, object]) -> None:
        print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
        if args.episode_dir is None:
            return
        path = (args.episode_dir / f"seed-{int(row['seed']):010d}.json").resolve()
        _write_new(path, row)
        artifacts.append(
            {
                "seed": int(row["seed"]),
                "path": str(path),
                "file_sha256": file_sha256(path),
                "content_sha256": canonical_sha256(row),
            }
        )

    report = evaluate_promotion_shard(
        ExactTrainingRuntime(args.worker),
        bundle,
        declared_training_seeds=training,
        evaluation_seeds=args.seeds,
        maximum_ticks=args.maximum_ticks,
        planner_config=FastMultiActionConfig(
            probe_ticks=args.probe_ticks,
            long_probe_ticks=args.long_probe_ticks,
            wait_ticks=args.wait_ticks,
            low_gauge_threshold=args.low_gauge_threshold,
            low_gauge_exit_threshold=args.low_gauge_exit_threshold,
            top_k_pairs=args.top_k_pairs,
            maximum_gauge_debt=args.maximum_gauge_debt,
            rescue_score_margin=args.rescue_score_margin,
            gauge_advantage=args.gauge_advantage,
        ),
        trace_interval_ticks=args.trace_interval_ticks,
        maximum_logged_queries=args.maximum_logged_queries,
        episode_callback=record_episode,
    )
    if artifacts:
        report["episode_artifacts"] = artifacts
        bind_report_content_sha256(report)
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
