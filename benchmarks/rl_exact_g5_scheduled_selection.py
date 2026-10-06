#!/usr/bin/env python3
"""Paired TRAIN-only selection for the deterministic G5 exact scheduler."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "python", ROOT / "benchmarks"):
    sys.path.insert(0, str(value))

from irisu_env import Action, ActionKind, ExactWorkerError  # noqa: E402
from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    FastMultiActionConfig,
    FastMultiActionPlanner,
)
from irisu_pointer.g5_hazard_scheduler import evaluate_scheduled_exact  # noqa: E402
from irisu_pointer.g5_solvency_trigger import sha256  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402

import rl_exact_adaptive_checkpoint_eval as checkpoint_eval  # noqa: E402
import rl_exact_fast_multiaction_eval as base_eval  # noqa: E402
import rl_exact_onpolicy_pair_dagger as seed_support  # noqa: E402
import rl_g5_hazard_shadow_audit as scheduler_support  # noqa: E402
import rl_g5_solvency_trigger as io_support  # noqa: E402


SCHEMA = "irisu-g5-scheduled-paired-selection-config-v1"


def runtime_core(provenance: Mapping[str, Any]) -> dict[str, object]:
    """Return identities that must match despite horizon-specific env config."""

    return {
        "version": provenance.get("version"),
        "physics_backend": provenance.get("physics_backend"),
        "identity": provenance.get("identity"),
        "runtime_attestation_sha256": provenance.get("runtime_attestation_sha256"),
        "runtime_attestation": provenance.get("runtime_attestation"),
    }


def validate_selection_seeds(
    plan: Mapping[str, object],
    model_training_seeds: Sequence[int],
    scheduler_training_seeds: Sequence[int],
) -> tuple[int, int]:
    raw = plan.get("seeds")
    if plan.get("split") != "train" or type(raw) is not list or len(raw) != 2:
        raise ValueError("G5 paired selection requires exactly two TRAIN seeds")
    seeds = tuple(int(value) for value in raw)
    if len(set(seeds)) != 2 or any(not 0 <= seed < 2**30 for seed in seeds):
        raise ValueError("G5 paired selection seeds are not unique TRAIN-split seeds")
    if (set(model_training_seeds) | set(scheduler_training_seeds)) & set(seeds):
        raise ValueError("G5 paired selection seeds overlap training lineage")
    return seeds


def load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text())
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise ValueError("G5 paired selection config schema mismatch")
    if value.get("development_only") is not True or value.get("promotion_eligible") is not False:
        raise ValueError("G5 paired selection must remain development-only")
    if int(value.get("maximum_ticks", 0)) != 80_000:
        raise ValueError("G5 paired selection maximum_ticks must remain 80000")
    if value.get("base_planner") != {
        "probe_ticks": 512,
        "long_probe_ticks": 512,
        "top_k_pairs": 0,
        "maximum_gauge_debt": 1_000,
        "rescue_score_margin": 500,
    }:
        raise ValueError("G5 immutable base planner contract mismatch")
    if value.get("scheduled_planner") != {
        "short_horizon": 2_048,
        "long_horizons": [8_192, 12_288],
        "top_k_pairs": 2,
        "maximum_gauge_debt": 1_000,
        "rescue_score_margin": 500,
        "compute_trigger_only": True,
        "exact_rule_only": True,
    }:
        raise ValueError("G5 scheduled planner contract mismatch")
    if value.get("hard_reject") != {
        "minimum_score_delta_per_seed": 0,
        "minimum_survival_tick_delta_per_seed": 0,
        "maximum_final_gauge_regression_per_seed": 1_000,
        "maximum_invalid_actions": 0,
        "maximum_exact_stage_failures": 0,
        "preserve_target_success_per_seed": True,
        "require_any_strict_improvement": True,
    }:
        raise ValueError("G5 hard-reject contract mismatch")
    scheduler_path = Path(str(value["scheduler_checkpoint"])).resolve(strict=True)
    if io_support.file_sha(scheduler_path) != value["scheduler_checkpoint_file_sha256"]:
        raise ValueError("G5 scheduler checkpoint file hash mismatch")
    _scheduler, checkpoint = scheduler_support.load_scheduler(scheduler_path)
    if checkpoint["checkpoint_sha256"] != value["scheduler_checkpoint_content_sha256"]:
        raise ValueError("G5 scheduler checkpoint content identity mismatch")
    return value


def hard_reject_report(
    base: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    *,
    target_score: int,
) -> dict[str, object]:
    if len(base) != len(candidate) or not base:
        raise ValueError("paired selection requires equal nonempty episode inventories")
    maximum_gauge_regression = int(contract["maximum_final_gauge_regression_per_seed"])
    minimum_score_delta = int(contract["minimum_score_delta_per_seed"])
    minimum_tick_delta = int(contract["minimum_survival_tick_delta_per_seed"])
    maximum_invalid = int(contract["maximum_invalid_actions"])
    maximum_failures = int(contract["maximum_exact_stage_failures"])
    pairs = []
    reasons = []
    any_strict = False
    for left, right in zip(base, candidate, strict=True):
        seed = int(left["seed"])
        if int(right["seed"]) != seed:
            raise ValueError("paired selection seed order differs")
        score_delta = int(right["score"]) - int(left["score"])
        tick_delta = int(right["tick"]) - int(left["tick"])
        gauge_delta = int(right["gauge"]) - int(left["gauge"])
        invalid = int(right.get("invalid_actions", 0))
        failures = int(right.get("exact_stage_failures", 0))
        failures += int(right.get("exact_branch_errors", 0))
        row_reasons = []
        if score_delta < minimum_score_delta:
            row_reasons.append("score-regression")
        if tick_delta < minimum_tick_delta:
            row_reasons.append("survival-regression")
        if gauge_delta < -maximum_gauge_regression:
            row_reasons.append("final-gauge-regression")
        if invalid > maximum_invalid:
            row_reasons.append("invalid-actions")
        if failures > maximum_failures:
            row_reasons.append("exact-stage-failure")
        if (
            contract.get("preserve_target_success_per_seed") is True
            and int(left["score"]) >= target_score
            and int(right["score"]) < target_score
        ):
            row_reasons.append("target-success-regression")
        any_strict |= score_delta > 0 or tick_delta > 0 or gauge_delta > 0
        reasons.extend(f"seed-{seed}:{reason}" for reason in row_reasons)
        pairs.append({
            "seed": seed,
            "base": {key: left[key] for key in ("score", "tick", "gauge")},
            "candidate": {key: right[key] for key in ("score", "tick", "gauge")},
            "score_delta": score_delta,
            "tick_delta": tick_delta,
            "gauge_delta": gauge_delta,
            "invalid_actions": invalid,
            "exact_failures": failures,
            "hard_reject_reasons": row_reasons,
        })
    if contract.get("require_any_strict_improvement") is True and not any_strict:
        reasons.append("aggregate:no-strict-improvement")
    return {
        "schema": "irisu-g5-paired-hard-reject-report-v1",
        "accepted": not reasons,
        "hard_reject_reasons": reasons,
        "pairs": pairs,
        "pairs_sha256": sha256(pairs),
    }


def run_scheduled_episode(
    runtime: ExactTrainingRuntime,
    bundle: checkpoint_eval.PolicyBundle,
    scheduler: object,
    seed: int,
    *,
    maximum_ticks: int,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    policy = bundle.factory()
    policy.reset(seed)
    planner = FastMultiActionPlanner(
        base_eval.primitive_actions,
        config=FastMultiActionConfig(
            probe_ticks=2_048,
            long_probe_ticks=2_048,
            low_gauge_threshold=1_000_000,
            low_gauge_exit_threshold=1_000_000,
            top_k_pairs=2,
            maximum_gauge_debt=1_000,
            rescue_score_margin=500,
        ),
        action_spec=policy.action_spec,
    )
    selected: Counter[str] = Counter()
    queries = branch_checks = invalid = branch_errors = stage_failures = 0
    terminated = truncated = False
    final_info: Mapping[str, Any] = {}
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": maximum_ticks + 12_288}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("G5 scheduled exact reset seed mismatch")
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            before = copy.deepcopy(policy)
            try:
                decision = policy.predict(observation)
            except (AttributeError, TypeError, ValueError, OverflowError):
                invalid += 1
                policy = before
                decision = planner.wait_decision("policy-error")
            if decision.is_shot:
                queries += 1
                source_hash = int(env.state_hash())
                try:
                    verdict = evaluate_scheduled_exact(
                        planner, env, observation, before, policy, decision, scheduler
                    )
                except ExactWorkerError:
                    if int(env.state_hash()) != source_hash:
                        raise RuntimeError("G5 exact failure changed live parent")
                    branch_errors += 1
                    policy = before
                    decision = planner.wait_decision("short-exact-worker-error")
                    selected["short-exact-worker-error"] += 1
                else:
                    branch_checks += verdict.branch_checks
                    stage_failures += int(verdict.reason == "long-exact-failure-retains-base")
                    selected[verdict.reason] += 1
                    policy = verdict.selected.continuation_policy
                    decision = verdict.selected.decision
            for raw_action in base_eval.primitive_actions(decision):
                remaining = maximum_ticks - int(observation["tick"])
                if remaining <= 0 or terminated or truncated:
                    break
                try:
                    action = base_eval._validated_action(raw_action, remaining)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    invalid += 1
                    action = Action.wait(1)
                kind = ActionKind.parse(action.kind)
                duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
                for _ in range(duration):
                    primitive = Action.wait(1) if kind is ActionKind.WAIT else action
                    observation, _reward, terminated, truncated, final_info = env.step(primitive)
                    invalid += int(bool(final_info.get("invalid_action", False)))
                    if terminated or truncated:
                        break
        provenance = session.provenance_manifest
    return {
        "seed": seed,
        "score": base_eval._effective_score(observation, final_info),
        "tick": int(observation["tick"]),
        "gauge": int(observation.get("gauge", 0)),
        "terminated": bool(terminated or observation.get("terminated", False)),
        "truncated": bool(truncated or observation.get("truncated", False)),
        "invalid_actions": invalid,
        "exact_branch_errors": branch_errors,
        "exact_stage_failures": stage_failures,
        "planner_queries": queries,
        "planner_branch_checks": branch_checks,
        "selection_reasons": dict(sorted(selected.items())),
        "wall_seconds": time.monotonic() - started,
    }, provenance


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("G5 paired selection output must be new")
    config = load_config(args.config)
    plan = seed_support.load_seed_plan(args.seed_plan)
    scheduler, scheduler_checkpoint = scheduler_support.load_scheduler(
        Path(str(config["scheduler_checkpoint"]))
    )
    bundle = checkpoint_eval.load_policy_bundle(
        Path(str(config["base_checkpoint"])),
        expected_sha256=str(config["base_checkpoint_sha256"]),
    )
    training = checkpoint_eval.checkpoint_training_seeds(bundle)
    seeds = validate_selection_seeds(plan, training, scheduler.training_seeds)
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    base_config = FastMultiActionConfig(**config["base_planner"])
    base_rows = []
    candidate_rows = []
    runtime_manifests = []
    for seed in seeds:
        base_row, base_runtime = base_eval.run_episode(
            runtime,
            bundle,
            seed,
            maximum_ticks=int(config["maximum_ticks"]),
            planner_config=base_config,
            target_score=int(config["target_score"]),
            trace_interval_ticks=10_000,
            maximum_logged_queries=0,
        )
        candidate_row, candidate_runtime = run_scheduled_episode(
            runtime, bundle, scheduler, seed,
            maximum_ticks=int(config["maximum_ticks"]),
        )
        if sha256(runtime_core(base_runtime)) != sha256(runtime_core(candidate_runtime)):
            raise RuntimeError("exact runtime identity changed within paired seed")
        base_rows.append(base_row)
        candidate_rows.append(candidate_row)
        runtime_manifests.append({
            "base": base_runtime,
            "candidate": candidate_runtime,
        })
        print(json.dumps({
            "seed": seed,
            "base_score": base_row["score"],
            "candidate_score": candidate_row["score"],
            "base_tick": base_row["tick"],
            "candidate_tick": candidate_row["tick"],
        }, sort_keys=True), flush=True)
    verdict = hard_reject_report(
        base_rows,
        candidate_rows,
        config["hard_reject"],
        target_score=int(config["target_score"]),
    )
    report = {
        "schema": "irisu-g5-scheduled-paired-selection-report-v1",
        "development_only": True,
        "promotion_eligible": False,
        "config": config,
        "config_file_sha256": io_support.file_sha(args.config.resolve(strict=True)),
        "seed_plan": plan,
        "seed_plan_file_sha256": io_support.file_sha(args.seed_plan.resolve(strict=True)),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_sha256": bundle.model_sha256,
        "checkpoint_metadata_sha256": sha256(bundle.metadata),
        "inference_config": dict(bundle.inference_config),
        "scheduler_checkpoint_sha256": scheduler_checkpoint["checkpoint_sha256"],
        "scheduler_sha256": scheduler.sha256,
        "runtime_manifests_sha256": sha256(runtime_manifests),
        "runtime_core_sha256": sha256([
            runtime_core(value["base"]) for value in runtime_manifests
        ]),
        "runner_source_sha256": io_support.file_sha(Path(__file__).resolve()),
        "scheduler_runtime_source_sha256": io_support.file_sha(
            ROOT / "python/irisu_pointer/g5_hazard_scheduler.py"
        ),
        "base_episodes": base_rows,
        "candidate_episodes": candidate_rows,
        "hard_reject": verdict,
    }
    report["content_sha256"] = sha256(report)
    io_support.write_new(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--worker", type=Path, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--seed-plan", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
