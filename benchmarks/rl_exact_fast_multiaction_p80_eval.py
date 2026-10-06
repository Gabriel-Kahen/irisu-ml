#!/usr/bin/env python3
"""Exact fixed-512 shards for the predeclared 200k-p80 promotion gate."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rl_exact_fast_multiaction_promotion_eval as base  # noqa: E402
from irisu_pointer.fast_multiaction_planner import FastMultiActionConfig  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402
from rl_exact_adaptive_checkpoint_eval import (  # noqa: E402
    FULL_GAME_TICKS,
    PolicyBundle,
    bind_report_content_sha256,
    canonical_sha256,
    file_sha256,
    load_declared_training_seeds,
    load_policy_bundle,
)
from rl_exact_p80_contract import CONTRACT, evaluate_contract  # noqa: E402


FORMAT = "irisu-exact-fast-multiaction-p80-promotion-shard-v1"


def evaluate_p80_shard(
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
) -> dict[str, object]:
    config = (
        FastMultiActionConfig(probe_ticks=512, long_probe_ticks=512, top_k_pairs=0)
        if planner_config is None
        else planner_config
    )
    report = base.evaluate_promotion_shard(
        runtime,
        bundle,
        declared_training_seeds=declared_training_seeds,
        evaluation_seeds=evaluation_seeds,
        maximum_ticks=maximum_ticks,
        planner_config=config,
        trace_interval_ticks=trace_interval_ticks,
        maximum_logged_queries=maximum_logged_queries,
        episode_callback=episode_callback,
        runner_path=Path(__file__) if runner_path is None else runner_path,
    )
    gate = evaluate_contract(
        [int(value) for value in report["scores"]],
        invalid_actions=int(report["invalid_actions"]),
    )
    report.update(
        {
            "format": FORMAT,
            "promotion_contract": CONTRACT,
            "baseline_success_count": gate["baseline_success_count"],
            "baseline_success_fraction_at_or_above_target": gate[
                "baseline_success_fraction_at_or_above_target"
            ],
            "p80_score_linear_type7": gate["p80_score_linear_type7"],
            "p80_score_linear_type7_exact": gate[
                "p80_score_linear_type7_exact"
            ],
            "p80_nearest_rank_score": gate["p80_nearest_rank_score"],
            "scores_at_or_above_200k": gate["scores_at_or_above_200k"],
            "promotion_eligible": gate["promotion_eligible"],
            "passed": gate["passed"],
        }
    )
    bind_report_content_sha256(report)
    return report


def parser():
    result = base.parser()
    result.set_defaults(probe_ticks=512, long_probe_ticks=512, top_k_pairs=0)
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
        base._write_new(path, row)
        artifacts.append(
            {
                "seed": int(row["seed"]),
                "path": str(path),
                "file_sha256": file_sha256(path),
                "content_sha256": canonical_sha256(row),
            }
        )

    report = evaluate_p80_shard(
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
    base._write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
