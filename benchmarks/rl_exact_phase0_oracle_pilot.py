#!/usr/bin/env python3
"""Development-only delayed-horizon oracle viability pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_onpolicy_pair_dagger as prior  # noqa: E402
import rl_exact_proposal_residual_dagger as residual  # noqa: E402
import rl_portable_gate_distill as gate  # noqa: E402
from irisu_env import ExactWorkerError  # noqa: E402
from irisu_pointer.branching import TransactionalBranches  # noqa: E402
from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
)
from irisu_pointer.shot_necessity import ProbeOutcome  # noqa: E402
from irisu_pointer.steering_checkpoint import load_steering_checkpoint  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


SCHEMA = "irisu-exact-phase0-delayed-horizon-oracle-v2"
SEED_SCHEMA = "irisu-exact-phase0-seed-durable-v1"
HORIZONS = (2_048, 8_192, 12_288)
STRATA = ("low-gauge", "early", "middle", "late")


@dataclass(frozen=True, slots=True)
class OracleOutcome:
    value: CandidateOutcome
    endpoint: dict[str, object]
    endpoint_state_hash: int


def public_value(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return public_value(item())
    if isinstance(value, Mapping):
        return {str(key): public_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [public_value(child) for child in value]
    raise TypeError(f"unsupported public value: {type(value).__name__}")


def observation_manifest(observation: Mapping[str, Any]) -> dict[str, object]:
    result = public_value(observation)
    if not isinstance(result, dict):
        raise TypeError("public observation is not a dictionary")
    return result


def state_stratum(
    observation: Mapping[str, Any], counts: Mapping[str, int], *, cap: int
) -> str | None:
    tick = int(observation["tick"])
    gauge = int(observation["gauge"])
    tick_stratum = "early" if tick < 10_000 else "middle" if tick < 30_000 else "late"
    eligible = (["low-gauge"] if gauge <= 12_000 else []) + [tick_stratum]
    return next((name for name in eligible if counts.get(name, 0) < cap), None)


def unchanged_parent_error_event(
    env: object,
    observation: Mapping[str, Any],
    source_hash: int,
    *,
    stage: str,
    error: ExactWorkerError,
) -> dict[str, object]:
    current_hash = int(env.state_hash())
    if current_hash != source_hash:
        raise RuntimeError("phase-0 exact worker error altered live parent") from error
    message = str(error)
    return {
        "stage": stage,
        "tick": int(observation["tick"]),
        "source_state_hash": source_hash,
        "parent_state_unchanged": True,
        "exception_type": "irisu_env.exact_ipc.ExactWorkerError",
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "recovery": (
            "no-label-then-continue-live-planner"
            if stage == "oracle-query"
            else "restore-preprediction-policy-and-execute-wait"
        ),
    }


def advance(
    planner: FastMultiActionPlanner,
    env: object,
    observation: Mapping[str, Any],
    policy: object,
    first: object,
    horizon: int,
) -> tuple[ProbeOutcome, dict[str, object], int]:
    start = int(observation["tick"])
    current = observation
    minimum = int(current["gauge"])
    terminated = truncated = False
    decision: object | None = first
    while int(current["tick"]) - start < horizon and not (terminated or truncated):
        if decision is None:
            decision = policy.predict(current)
        for raw in prior.primitive_actions(decision):
            remaining = start + horizon - int(current["tick"])
            if remaining <= 0:
                break
            action = prior._validated_action(raw, remaining)
            kind = prior.ActionKind.parse(action.kind)
            duration = int(action.wait_ticks) if kind is prior.ActionKind.WAIT else 1
            for _ in range(duration):
                primitive = prior.Action.wait(1) if kind is prior.ActionKind.WAIT else action
                current, _reward, terminated, truncated, _info = env.step(primitive)
                minimum = min(minimum, int(current["gauge"]))
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        decision = None
    probe = ProbeOutcome(
        int(current["tick"]) - start,
        int(current.get("score", 0)),
        int(current.get("qualifying_clear_count", 0)),
        int(current["gauge"]),
        minimum,
        bool(terminated or current.get("terminated", False)),
        bool(truncated or current.get("truncated", False)),
    )
    return probe, observation_manifest(current), int(env.state_hash())


def evaluate_subset(
    planner: FastMultiActionPlanner,
    env: object,
    observation: Mapping[str, Any],
    candidates: Sequence[object],
    *,
    horizon: int,
) -> tuple[OracleOutcome, ...]:
    source_hash = int(env.state_hash())
    outcomes: list[OracleOutcome] = []
    with TransactionalBranches(env, observation) as branches:
        if not branches.uses_fast_checkpoint:
            raise RuntimeError("phase-0 requires exact fast checkpoints")
        for candidate in candidates:
            with branches.branch() as (branch, state):
                probe, endpoint, endpoint_hash = advance(
                    planner,
                    branch,
                    state,
                    copy.deepcopy(candidate.continuation_policy),
                    candidate.decision,
                    horizon,
                )
                outcomes.append(
                    OracleOutcome(CandidateOutcome(candidate, probe, 0), endpoint, endpoint_hash)
                )
    if int(env.state_hash()) != source_hash:
        raise RuntimeError("phase-0 oracle changed the live parent")
    return tuple(outcomes)


def outcome_manifest(value: OracleOutcome) -> dict[str, object]:
    return {
        **value.value.manifest(),
        "endpoint_public_observation": value.endpoint,
        "endpoint_public_observation_sha256": gate._sha(value.endpoint),
        "endpoint_state_hash": value.endpoint_state_hash,
    }


def base_choice(
    planner: FastMultiActionPlanner, outcomes: Sequence[OracleOutcome]
) -> OracleOutcome:
    wait = next(value for value in outcomes if value.value.candidate.category == "wait")
    eligible = []
    for value in outcomes:
        if value is wait or value.value.candidate.category.startswith("top-"):
            continue
        allowed, _ = planner._eligible(value.value, wait.value)
        if allowed:
            eligible.append(value)
    return max(eligible, key=lambda value: planner._objective(value.value)) if eligible else wait


def contender_choices(
    planner: FastMultiActionPlanner,
    outcomes: Sequence[OracleOutcome],
    base: OracleOutcome,
) -> tuple[OracleOutcome, ...]:
    values = [
        value for value in outcomes
        if value.value.candidate.ordinal != base.value.candidate.ordinal
    ]
    return tuple(
        sorted(values, key=lambda value: planner._objective(value.value), reverse=True)[:2]
    )


def robust_better(
    candidate: ProbeOutcome, base: ProbeOutcome, *, horizon: int
) -> bool:
    if candidate.survival_ticks > base.survival_ticks:
        return candidate.minimum_gauge >= base.minimum_gauge
    return bool(
        candidate.survival_ticks == base.survival_ticks == horizon
        and candidate.minimum_gauge >= base.minimum_gauge + 1_000
        and candidate.final_gauge >= base.final_gauge + 1_000
        and candidate.clears >= base.clears
        and candidate.score >= base.score
    )


def safe_delayed_disagreement(
    short_candidate: ProbeOutcome,
    short_base: ProbeOutcome,
    long8_candidate: ProbeOutcome,
    long8_base: ProbeOutcome,
    long12_candidate: ProbeOutcome,
    long12_base: ProbeOutcome,
) -> bool:
    return bool(
        not robust_better(short_candidate, short_base, horizon=2_048)
        and robust_better(long8_candidate, long8_base, horizon=8_192)
        and robust_better(long12_candidate, long12_base, horizon=12_288)
    )


def go_decision(
    episodes: Sequence[Mapping[str, object]], *, minimum_queries_per_seed: int
) -> tuple[bool, int, int, int]:
    total = sum(int(value["safe_delayed_disagreements"]) for value in episodes)
    seeds_with = sum(int(value["safe_delayed_disagreements"]) > 0 for value in episodes)
    late = sum(int(value["late_safe_delayed_disagreements"]) for value in episodes)
    complete = all(
        int(value["query_count"]) >= minimum_queries_per_seed for value in episodes
    )
    return complete and total >= 8 and seeds_with >= 3 and late > 0, total, seeds_with, late


def oracle_query(
    planner: FastMultiActionPlanner,
    env: object,
    observation: Mapping[str, Any],
    policy_before: object,
    policy_after: object,
    prediction: object,
    *,
    seed: int,
    query_index: int,
    stratum: str,
) -> dict[str, object]:
    source_hash = int(env.state_hash())
    candidates = planner.candidates(
        observation, policy_before, policy_after, prediction
    )
    if not 3 <= len(candidates) <= 7:
        raise RuntimeError("phase-0 candidate inventory is outside [3,7]")
    short = evaluate_subset(planner, env, observation, candidates, horizon=HORIZONS[0])
    base = base_choice(planner, short)
    contenders = contender_choices(planner, short, base)
    selected = (
        base.value.candidate,
        *(value.value.candidate for value in contenders),
    )
    long_by_horizon: dict[int, tuple[OracleOutcome, ...]] = {}
    for horizon in HORIZONS[1:]:
        long_by_horizon[horizon] = evaluate_subset(
            planner, env, observation, selected, horizon=horizon
        )
    long8 = {
        value.value.candidate.ordinal: value for value in long_by_horizon[8_192]
    }
    long12 = {
        value.value.candidate.ordinal: value for value in long_by_horizon[12_288]
    }
    disagreements = []
    for contender in contenders:
        ordinal = contender.value.candidate.ordinal
        if safe_delayed_disagreement(
            contender.value.probe,
            base.value.probe,
            long8[ordinal].value.probe,
            long8[base.value.candidate.ordinal].value.probe,
            long12[ordinal].value.probe,
            long12[base.value.candidate.ordinal].value.probe,
        ):
            disagreements.append(ordinal)
    source_observation = observation_manifest(observation)
    return {
        "schema": "irisu-phase0-oracle-query-v1",
        "seed": seed,
        "query_index": query_index,
        "stratum": stratum,
        "tick": int(observation["tick"]),
        "gauge": int(observation["gauge"]),
        "source_state_hash": source_hash,
        "source_public_observation": source_observation,
        "source_public_observation_sha256": gate._sha(source_observation),
        "candidate_count": len(candidates),
        "base_ordinal": base.value.candidate.ordinal,
        "contender_ordinals": [value.value.candidate.ordinal for value in contenders],
        "safe_delayed_disagreement_ordinals": disagreements,
        "horizons": {
            "2048": [outcome_manifest(value) for value in short],
            "8192": [outcome_manifest(value) for value in long_by_horizon[8_192]],
            "12288": [outcome_manifest(value) for value in long_by_horizon[12_288]],
        },
    }


def collect_seed(
    runtime: ExactTrainingRuntime,
    model: object,
    model_sha: str,
    seed: int,
    *,
    maximum_ticks: int,
    states_per_stratum: int,
    act_logit_bias: float,
) -> dict[str, object]:
    policy = gate._make_policy(model, model_sha, act_logit_bias)
    policy.reset(seed)
    oracle = FastMultiActionPlanner(
        prior.primitive_actions,
        config=FastMultiActionConfig(
            probe_ticks=2_048, long_probe_ticks=2_048,
            low_gauge_threshold=1_000_000, low_gauge_exit_threshold=1_000_000,
            top_k_pairs=2, maximum_gauge_debt=1_000, rescue_score_margin=500,
        ),
        action_spec=policy.action_spec,
    )
    live = FastMultiActionPlanner(
        prior.primitive_actions,
        config=FastMultiActionConfig(
            probe_ticks=512, long_probe_ticks=512, top_k_pairs=0,
            maximum_gauge_debt=1_000, rescue_score_margin=500,
        ),
        action_spec=policy.action_spec,
    )
    counts: Counter[str] = Counter()
    queries = []
    exact_worker_errors = []
    terminated = truncated = False
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": maximum_ticks + max(HORIZONS)}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("phase-0 exact reset seed mismatch")
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            before = copy.deepcopy(policy)
            prediction = policy.predict(observation)
            executed = prediction
            if prediction.is_shot:
                stratum = state_stratum(observation, counts, cap=states_per_stratum)
                if stratum is not None:
                    source_hash = int(env.state_hash())
                    try:
                        query = oracle_query(
                            oracle, env, observation, before, policy, prediction,
                            seed=seed, query_index=len(queries), stratum=stratum,
                        )
                    except ExactWorkerError as error:
                        exact_worker_errors.append(unchanged_parent_error_event(
                            env, observation, source_hash,
                            stage="oracle-query", error=error,
                        ))
                    else:
                        queries.append(query)
                        counts[stratum] += 1
                source_hash = int(env.state_hash())
                try:
                    verdict = live.evaluate(
                        env, observation, before, policy, prediction
                    )
                except ExactWorkerError as error:
                    exact_worker_errors.append(unchanged_parent_error_event(
                        env, observation, source_hash,
                        stage="live-planner", error=error,
                    ))
                    policy = before
                    executed = live.wait_decision("exact-worker-error-safe-wait")
                else:
                    executed = verdict.selected.decision
                    policy = verdict.selected.continuation_policy
            observation, terminated, truncated = prior._step(
                env, observation, executed, maximum_ticks=maximum_ticks
            )
            if len(queries) >= len(STRATA) * states_per_stratum:
                break
        provenance = session.provenance_manifest
    disagreements = sum(
        len(value["safe_delayed_disagreement_ordinals"]) for value in queries
    )
    late = sum(
        len(value["safe_delayed_disagreement_ordinals"])
        for value in queries if value["stratum"] == "late"
    )
    return {
        "seed": seed,
        "queries": queries,
        "query_count": len(queries),
        "stratum_counts": dict(sorted(counts.items())),
        "safe_delayed_disagreements": disagreements,
        "late_safe_delayed_disagreements": late,
        "final_tick": int(observation["tick"]),
        "final_score": int(observation.get("score", 0)),
        "final_gauge": int(observation.get("gauge", 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "runtime": provenance,
        "exact_worker_error_count": len(exact_worker_errors),
        "exact_worker_errors": exact_worker_errors,
        "exact_worker_errors_sha256": gate._sha(exact_worker_errors),
        "wall_seconds": time.monotonic() - started,
    }


def durable_seed_envelope(
    binding: Mapping[str, object], episode: Mapping[str, object]
) -> dict[str, object]:
    stable_episode = {
        key: value for key, value in episode.items() if key != "wall_seconds"
    }
    result = {
        "schema": SEED_SCHEMA,
        "binding": dict(binding),
        "binding_sha256": gate._sha(dict(binding)),
        "episode": stable_episode,
        "episode_sha256": gate._sha(stable_episode),
    }
    result["content_sha256"] = gate._sha(result)
    return result


def load_durable_seed(
    path: Path, expected_binding: Mapping[str, object]
) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text())
    if type(value) is not dict or value.get("schema") != SEED_SCHEMA:
        raise ValueError("phase-0 durable seed schema mismatch")
    claimed = value.get("content_sha256")
    payload = {key: child for key, child in value.items() if key != "content_sha256"}
    if claimed != gate._sha(payload):
        raise ValueError("phase-0 durable seed content hash mismatch")
    if value.get("binding") != dict(expected_binding):
        raise ValueError("phase-0 durable seed binding mismatch")
    if value.get("binding_sha256") != gate._sha(dict(expected_binding)):
        raise ValueError("phase-0 durable seed binding hash mismatch")
    episode = value.get("episode")
    if type(episode) is not dict or value.get("episode_sha256") != gate._sha(episode):
        raise ValueError("phase-0 durable seed episode hash mismatch")
    return episode


def run(args: argparse.Namespace) -> dict[str, object]:
    final_path = args.output / "oracle-pilot.json"
    if args.resume:
        if not args.output.is_dir() or final_path.exists():
            raise FileNotFoundError(
                "phase-0 resume requires an unfinished durable output directory"
            )
    else:
        if args.output.exists():
            raise FileExistsError("phase-0 output directory must be new")
        args.output.mkdir(parents=True)
        (args.output / "episodes").mkdir()
    plan = prior.load_seed_plan(args.seed_plan)
    artifact = load_steering_checkpoint(args.base_checkpoint, expected_sha256=args.base_sha256)
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    seeds = tuple(int(value) for value in plan["seeds"])
    if len(seeds) != args.expected_seed_count or set(seeds) & set(upstream):
        raise RuntimeError("phase-0 seed count/freshness contract differs")
    model_sha = gate._model_state_sha(artifact.model)
    inference = artifact.metadata.get("inference_config")
    if not isinstance(inference, dict) or inference != gate.inference_config(
        float(inference.get("act_logit_bias", 0.0))
    ):
        raise RuntimeError("phase-0 base inference configuration is unsupported")
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    collection_config = {
        "maximum_ticks": args.maximum_ticks,
        "states_per_stratum": args.states_per_stratum,
        "strata": list(STRATA),
        "shortlist_top_k_pairs": 2,
        "maximum_candidates": 7,
        "long_contender_cap": 2,
        "maximum_gauge_debt": 1_000,
        "rescue_score_margin": 500,
    }
    source_sha = gate._file_sha(Path(__file__).resolve())
    dependency_sha = {
        "rl_exact_onpolicy_pair_dagger": gate._file_sha(Path(prior.__file__).resolve()),
        "rl_exact_proposal_residual_dagger": gate._file_sha(Path(residual.__file__).resolve()),
        "rl_portable_gate_distill": gate._file_sha(Path(gate.__file__).resolve()),
    }
    common_binding = {
        "collector_schema": SCHEMA,
        "collector_source_sha256": source_sha,
        "dependency_source_sha256": dependency_sha,
        "base_checkpoint_sha256": artifact.sha256,
        "base_model_sha256": model_sha,
        "base_metadata_sha256": gate._sha(dict(artifact.metadata)),
        "base_inference_config": inference,
        "seed_plan_sha256": gate._file_sha(args.seed_plan.resolve(strict=True)),
        "worker_identity": runtime.identity.manifest(),
        "collection_config": collection_config,
    }
    expected_seed_files = {
        f"seed-{index:02d}-{seed}.json" for index, seed in enumerate(seeds)
    }
    episode_root = args.output / "episodes"
    if args.resume:
        if not episode_root.is_dir():
            raise FileNotFoundError("phase-0 resume episode directory is absent")
        unexpected = {path.name for path in episode_root.iterdir()} - expected_seed_files
        if unexpected:
            raise RuntimeError("phase-0 resume contains unexpected episode files")
    episodes = []
    for index, seed in enumerate(seeds):
        binding = {**common_binding, "seed_index": index, "seed": seed}
        seed_path = episode_root / f"seed-{index:02d}-{seed}.json"
        resumed = seed_path.exists()
        if resumed:
            episode = load_durable_seed(seed_path, binding)
        else:
            collected = collect_seed(
                runtime, artifact.model, model_sha, seed,
                maximum_ticks=args.maximum_ticks,
                states_per_stratum=args.states_per_stratum,
                act_logit_bias=float(inference["act_logit_bias"]),
            )
            envelope = durable_seed_envelope(binding, collected)
            gate._write_json_new(seed_path, envelope)
            episode = envelope["episode"]
        episodes.append(episode)
        print(json.dumps({
            **{
                key: child for key, child in episode.items()
                if key not in {"queries", "runtime", "wall_seconds"}
            },
            "durable_seed_file": str(seed_path.resolve()),
            "resumed": resumed,
        }, sort_keys=True), flush=True)
    minimum_queries = len(STRATA) * args.states_per_stratum
    go, total, seeds_with, late = go_decision(
        episodes, minimum_queries_per_seed=minimum_queries
    )
    result = {
        "schema": SCHEMA,
        "development_only": True,
        "promotion_eligible": False,
        "decision": (
            "SUPPLEMENTAL-G5-DATA" if args.expected_seed_count < 3
            else "GO-G5" if go else "NO-GO-ADD-TEMPORAL-MACROS"
        ),
        "go_criteria": {
            "minimum_disagreements": 8,
            "minimum_seeds": 3,
            "minimum_queries_per_seed": minimum_queries,
            "require_late": True,
        },
        "safe_delayed_disagreements": total,
        "seeds_with_disagreement": seeds_with,
        "late_safe_delayed_disagreements": late,
        "horizons": list(HORIZONS),
        "base_checkpoint_sha256": artifact.sha256,
        "base_model_sha256": model_sha,
        "base_metadata_sha256": gate._sha(dict(artifact.metadata)),
        "base_inference_config": inference,
        "seed_plan": plan,
        "seed_plan_sha256": common_binding["seed_plan_sha256"],
        "training_seeds": sorted((*upstream, *seeds)),
        "pilot_seeds": list(seeds),
        "collection_config": collection_config,
        "exact_worker_error_recovery": {
            "exception_type": "ExactWorkerError-only",
            "require_unchanged_parent_hash": True,
            "oracle_query": "no-label-continue",
            "live_planner": "restore-preprediction-policy-execute-wait",
        },
        "durability": {
            "schema": SEED_SCHEMA,
            "seed_files": [
                str((episode_root / f"seed-{index:02d}-{seed}.json").resolve())
                for index, seed in enumerate(seeds)
            ],
            "resumable": True,
        },
        "episodes": episodes,
        "worker_identity": runtime.identity.manifest(),
        "source_sha256": source_sha,
        "dependency_source_sha256": dependency_sha,
    }
    result["content_sha256"] = gate._sha(result)
    gate._write_json_new(final_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--worker", type=Path, required=True)
    value.add_argument("--base-checkpoint", type=Path, required=True)
    value.add_argument("--base-sha256", required=True)
    value.add_argument("--seed-plan", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--maximum-ticks", type=int, default=60_000)
    value.add_argument("--states-per-stratum", type=int, default=4)
    value.add_argument("--expected-seed-count", type=int, default=4)
    value.add_argument("--resume", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parser().parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
