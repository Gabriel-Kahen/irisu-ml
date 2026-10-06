#!/usr/bin/env python3
"""Collect strict-survival labels for a proposal-only pair residual."""

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

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_exact_onpolicy_pair_dagger as prior  # noqa: E402
import rl_portable_gate_distill as gate  # noqa: E402
import rl_portable_pair_rank_distill as ranking  # noqa: E402
from irisu_env import ExactWorkerError  # noqa: E402
from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
    MultiActionVerdict,
)
from irisu_pointer.proposal_only_pair_residual import (  # noqa: E402
    ProposalOnlyResidualConfig,
)
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_rl.encoding import EncodedBatch  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


SCHEMA = "irisu-exact-proposal-only-residual-collection-v1"


@dataclass(frozen=True, slots=True)
class SurvivalPreferenceConfig:
    capped_minimum_gauge_margin: int = 1_000
    capped_final_gauge_margin: int = 1_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.capped_minimum_gauge_margin, bool)
            or isinstance(self.capped_final_gauge_margin, bool)
            or self.capped_minimum_gauge_margin < 0
            or self.capped_final_gauge_margin < 0
        ):
            raise ValueError("capped-tie gauge margins must be nonnegative integers")


def survival_preference_assessment(
    planner: FastMultiActionPlanner,
    verdict: MultiActionVerdict,
    *,
    config: SurvivalPreferenceConfig = SurvivalPreferenceConfig(),
) -> tuple[tuple[CandidateOutcome, CandidateOutcome] | None, str]:
    """Assess a strict-survival or robust long-horizon reserve preference."""
    if not verdict.used_fast_checkpoint:
        return None, "nonfast-checkpoint"
    teacher = next(
        value
        for value in verdict.outcomes
        if value.candidate.ordinal == verdict.selected.ordinal
    )
    if not teacher.candidate.category.startswith("top-"):
        return None, "selected-not-alternate-pair"
    incumbent = prior.incumbent_choice(planner, verdict)
    wait = next(value for value in verdict.outcomes if value.candidate.category == "wait")
    horizon = int(verdict.probe_ticks)
    controls = (incumbent.probe, wait.probe)
    if teacher.probe.survival_ticks != horizon:
        return None, "teacher-incomplete"
    if teacher.probe.survival_ticks > max(value.survival_ticks for value in controls):
        if teacher.probe.minimum_gauge < max(value.minimum_gauge for value in controls):
            return None, "strict-survival-minimum-gauge-dominated"
        return (teacher, incumbent), "strict-survival"
    if any(value.survival_ticks != horizon for value in controls):
        return None, "no-strict-survival-improvement"
    if (
        verdict.probe_mode != "low-gauge-long"
        or horizon != planner.config.long_probe_ticks
    ):
        return None, "capped-tie-not-long-horizon"
    if teacher.probe.minimum_gauge < (
        max(value.minimum_gauge for value in controls)
        + config.capped_minimum_gauge_margin
    ):
        return None, "capped-tie-minimum-gauge-margin"
    if teacher.probe.final_gauge < (
        max(value.final_gauge for value in controls)
        + config.capped_final_gauge_margin
    ):
        return None, "capped-tie-final-gauge-margin"
    if teacher.probe.clears < max(value.clears for value in controls):
        return None, "capped-tie-clear-regression"
    if teacher.probe.score < max(value.score for value in controls):
        return None, "capped-tie-score-regression"
    return (teacher, incumbent), "capped-tie-robust-reserve"


def strict_full_horizon_preference(
    planner: FastMultiActionPlanner,
    verdict: MultiActionVerdict,
    *,
    config: SurvivalPreferenceConfig = SurvivalPreferenceConfig(),
) -> tuple[CandidateOutcome, CandidateOutcome] | None:
    return survival_preference_assessment(
        planner, verdict, config=config
    )[0]


def immutable_base_choice(
    planner: FastMultiActionPlanner, verdict: MultiActionVerdict
) -> CandidateOutcome:
    """Choose the deployment/base action without any teacher-only top pair."""
    wait = next(value for value in verdict.outcomes if value.candidate.category == "wait")
    eligible: list[CandidateOutcome] = []
    for value in verdict.outcomes:
        if value.candidate.category.startswith("top-") or value is wait:
            continue
        allowed, _reason = planner._eligible(value, wait)
        if allowed:
            eligible.append(value)
    return max(eligible, key=planner._objective) if eligible else wait


def encoded_copy(policy: object, observation: Mapping[str, Any]) -> EncodedBatch:
    value = policy.encoder.encode([observation])
    return EncodedBatch(
        np.array(value.global_features, copy=True, order="C"),
        np.array(value.body_features, copy=True, order="C"),
        np.array(value.body_mask, copy=True, order="C"),
        np.array(value.source_tick, copy=True, order="C"),
        np.array(value.health_flags, copy=True, order="C"),
        value.schema,
    )


def encoded_record(value: EncodedBatch) -> dict[str, object]:
    """Persist an owned observation together with its semantic tensor hashes."""
    tensors = {
        "global_features": torch.from_numpy(np.array(value.global_features, copy=True, order="C")),
        "body_features": torch.from_numpy(np.array(value.body_features, copy=True, order="C")),
        "body_mask": torch.from_numpy(np.array(value.body_mask, copy=True, order="C")),
        "source_tick": torch.from_numpy(np.array(value.source_tick, copy=True, order="C")),
        "health_flags": torch.from_numpy(np.array(value.health_flags, copy=True, order="C")),
    }
    hashes = {
        name: {
            "dtype": str(tensor.numpy().dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
        }
        for name, tensor in tensors.items()
    }
    return {"tensors": tensors, "hashes": hashes, "schema_sha256": value.schema.sha256}


def collection_content_sha256(payload: Mapping[str, object]) -> str:
    semantic = dict(payload)
    semantic["positives"] = [
        {"manifest": value["manifest"], "observation": value["observation"]["hashes"]}
        for value in payload["positives"]
    ]
    semantic["anchors"] = [
        {"hashes": value["hashes"], "schema_sha256": value["schema_sha256"]}
        for value in payload["anchors"]
    ]
    return gate._sha(semantic)


def collect_seed(
    *,
    runtime: ExactTrainingRuntime,
    model: torch.nn.Module,
    model_sha256: str,
    seed: int,
    maximum_ticks: int,
    maximum_queries: int,
    maximum_positives: int,
    maximum_anchors: int,
    planner_config: FastMultiActionConfig,
    trigger: ProposalOnlyResidualConfig,
    act_logit_bias: float,
    preference_config: SurvivalPreferenceConfig,
) -> tuple[list[ranking.PairPreference], list[EncodedBatch], dict[str, object]]:
    policy = gate._make_policy(model, model_sha256, act_logit_bias)
    policy.reset(seed)
    planner = FastMultiActionPlanner(
        prior.primitive_actions, config=planner_config, action_spec=policy.action_spec
    )
    positives: list[ranking.PairPreference] = []
    anchors: list[EncodedBatch] = []
    queries = errors = 0
    assessments: Counter[str] = Counter()
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": maximum_ticks + planner_config.long_probe_ticks}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("exact reset seed mismatch")
        terminated = truncated = False
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            policy_before = copy.deepcopy(policy)
            prediction = policy.predict(observation)
            executed = prediction
            if prediction.is_shot and trigger.active(observation) and queries < maximum_queries:
                queries += 1
                parent = int(env.state_hash())
                try:
                    verdict = planner.evaluate(
                        env, observation, policy_before, policy, prediction
                    )
                except ExactWorkerError:
                    errors += 1
                    policy, executed, _ = prior.conservative_exact_error(
                        parent_hash=parent,
                        current_hash=int(env.state_hash()),
                        policy_before=policy_before,
                    )
                else:
                    if int(env.state_hash()) != parent:
                        raise RuntimeError("exact planner changed parent state")
                    base_choice = immutable_base_choice(planner, verdict)
                    executed = base_choice.candidate.decision
                    policy = base_choice.candidate.continuation_policy
                    strict, assessment = survival_preference_assessment(
                        planner, verdict, config=preference_config
                    )
                    assessments[assessment] += 1
                    if strict is not None and len(positives) < maximum_positives:
                        teacher, incumbent = strict
                        provenance = {
                            "schema": "irisu-strict-survival-proposal-pair-v1",
                            "seed": seed,
                            "tick": int(observation["tick"]),
                            "gauge": int(observation["gauge"]),
                            "horizon": verdict.probe_ticks,
                            "preference_basis": assessment,
                            "parent_state_hash": parent,
                            "planner_manifest": verdict.manifest(),
                            "teacher_pair": list(ranking._decision_key(teacher.candidate.decision)),
                            "incumbent_pair": list(ranking._decision_key(incumbent.candidate.decision)),
                            "strict_survival_ticks": teacher.probe.survival_ticks - max(
                                incumbent.probe.survival_ticks,
                                next(value.probe.survival_ticks for value in verdict.outcomes if value.candidate.category == "wait"),
                            ),
                        }
                        positives.append(
                            ranking._preference(
                                observation,
                                teacher.candidate.decision,
                                incumbent.candidate.decision,
                                provenance,
                                encoder=policy.encoder,
                                pointer_spec=policy.pointer_spec,
                            )
                        )
                    else:
                        if len(anchors) < maximum_anchors:
                            anchors.append(encoded_copy(policy, observation))
            observation, terminated, truncated = prior._step(
                env, observation, executed, maximum_ticks=maximum_ticks
            )
            if queries >= maximum_queries and len(positives) >= maximum_positives:
                break
        runtime_manifest = session.provenance_manifest
    return positives, anchors, {
        "seed": seed,
        "final_tick": int(observation["tick"]),
        "final_score": int(observation.get("score", 0)),
        "final_gauge": int(observation.get("gauge", 0)),
        "queries": queries,
        "strict_positives": len(positives),
        "functional_anchors": len(anchors),
        "preference_assessments": dict(sorted(assessments.items())),
        "exact_branch_errors": errors,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "runtime": runtime_manifest,
        "wall_seconds": time.monotonic() - started,
    }


def _batch_encoded(values: Sequence[EncodedBatch]) -> tuple[torch.Tensor, ...]:
    active = max(int(np.flatnonzero(v.body_mask[0])[-1]) + 1 for v in values)
    return (
        torch.from_numpy(np.concatenate([v.global_features for v in values])),
        torch.from_numpy(np.concatenate([v.body_features[:, :active] for v in values])),
        torch.from_numpy(np.concatenate([v.body_mask[:, :active] for v in values])),
    )


def train_residual(
    residual: torch.nn.Module,
    base: torch.nn.Module,
    positives: Sequence[ranking.PairPreference],
    anchors: Sequence[EncodedBatch],
    *,
    steps: int,
    learning_rate: float,
    functional_weight: float,
    seed: int,
) -> dict[str, float | int]:
    if not positives or not anchors:
        raise ValueError("strict positives and functional anchors are required")
    trainable = prior.configure_trainable(residual)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    residual.eval(); base.eval()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in residual.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    initial = copy.deepcopy(residual.state_dict())
    for _ in range(steps):
        pids = torch.randint(len(positives), (min(32, len(positives)),), generator=generator).tolist()
        gf, bf, bm, ps, pd, ns, nd = ranking._preference_tensors(positives, pids)
        output = residual(gf, bf, bm); rows = torch.arange(len(ps))
        preference_loss = F.softplus(-(
            output.pair_logits[rows, ps, pd] - output.pair_logits[rows, ns, nd]
        )).mean()
        aids = torch.randint(len(anchors), (min(8, len(anchors)),), generator=generator).tolist()
        ag, ab, am = _batch_encoded([anchors[index] for index in aids])
        current = residual(ag, ab, am)
        with torch.no_grad():
            target = base(ag, ab, am)
        functional_loss = (
            (current.pair_logits - target.pair_logits).square()
            .masked_select(current.legal_pair_mask)
            .mean()
        )
        loss = preference_loss + functional_weight * functional_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        ag, ab, am = _batch_encoded(anchors)
        current = residual(ag, ab, am); target = base(ag, ab, am)
        anchor_delta = (current.pair_logits - target.pair_logits).abs().masked_select(current.legal_pair_mask)
    changed = [name for name, value in residual.state_dict().items() if not torch.equal(value, initial[name])]
    if set(changed) - set(trainable):
        raise RuntimeError("proposal residual changed preserved parameters")
    return {
        "steps": steps,
        "strict_positive_count": len(positives),
        "functional_anchor_count": len(anchors),
        "maximum_anchor_logit_delta": float(anchor_delta.max()),
        "mean_anchor_logit_delta": float(anchor_delta.mean()),
        "changed_parameters": changed,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    seed_plan = prior.load_seed_plan(args.seed_plan)
    seeds = tuple(int(seed) for seed in seed_plan["seeds"])
    selection_seed_plan = prior.load_seed_plan(args.selection_seed_plan)
    selection_seeds = tuple(int(seed) for seed in selection_seed_plan["seeds"])
    if set(seeds) & set(selection_seeds):
        raise RuntimeError("collection and selection seed plans overlap")
    artifact = load_steering_checkpoint(args.base_checkpoint, expected_sha256=args.base_sha256)
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    if (set(seeds) | set(selection_seeds)) & set(upstream):
        raise RuntimeError("fresh seed plans overlap base lineage")
    base_inference = artifact.metadata.get("inference_config")
    if not isinstance(base_inference, dict) or base_inference != gate.inference_config(
        float(base_inference.get("act_logit_bias", 0.0))
    ):
        raise RuntimeError("base checkpoint inference configuration is missing or unsupported")
    act_logit_bias = float(base_inference["act_logit_bias"])
    base = artifact.model.eval()
    residual = copy.deepcopy(base)
    planner_config = FastMultiActionConfig(
        probe_ticks=args.probe_ticks,
        long_probe_ticks=args.low_gauge_probe_ticks,
        low_gauge_threshold=args.low_gauge,
        low_gauge_exit_threshold=args.low_gauge_exit,
        top_k_pairs=2,
        maximum_gauge_debt=1_000,
        rescue_score_margin=500,
    )
    trigger = ProposalOnlyResidualConfig(
        minimum_tick=args.minimum_tick,
        low_reserve_minimum_tick=args.low_reserve_minimum_tick,
        low_reserve_gauge=args.low_gauge,
    )
    preference_config = SurvivalPreferenceConfig(
        capped_minimum_gauge_margin=args.capped_minimum_gauge_margin,
        capped_final_gauge_margin=args.capped_final_gauge_margin,
    )
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    positives: list[ranking.PairPreference] = []
    anchors: list[EncodedBatch] = []
    episodes = []
    model_sha = gate._model_state_sha(base)
    for seed in seeds:
        current_positive, current_anchors, episode = collect_seed(
            runtime=runtime, model=base, model_sha256=model_sha, seed=seed,
            maximum_ticks=args.maximum_ticks, maximum_queries=args.maximum_queries,
            maximum_positives=args.maximum_positives,
            maximum_anchors=args.maximum_anchors,
            planner_config=planner_config, trigger=trigger,
            act_logit_bias=act_logit_bias,
            preference_config=preference_config,
        )
        positives.extend(current_positive); anchors.extend(current_anchors); episodes.append(episode)
        print(json.dumps({k: v for k, v in episode.items() if k not in {"runtime", "wall_seconds"}}, sort_keys=True), flush=True)
    if any(int(row["strict_positives"]) < args.minimum_positives for row in episodes):
        raise RuntimeError("strict full-horizon positive minimum was not met")
    report = train_residual(
        residual, base, positives, anchors, steps=args.training_steps,
        learning_rate=args.learning_rate, functional_weight=args.functional_weight,
        seed=args.training_seed,
    )
    if gate._model_state_sha(base) != model_sha:
        raise RuntimeError("immutable base model changed during residual fitting")
    if report["maximum_anchor_logit_delta"] > args.maximum_anchor_logit_delta:
        raise RuntimeError("functional anchor trust region exceeded")
    positive_records = [
        {"manifest": value.manifest(), "observation": encoded_record(value.observation)}
        for value in positives
    ]
    anchor_records = [encoded_record(value) for value in anchors]
    collection = {
        "schema": SCHEMA,
        "base_checkpoint_sha256": artifact.sha256,
        "base_model_sha256": model_sha,
        "seed_plan": seed_plan,
        "selection_seed_plan": selection_seed_plan,
        "planner_config": planner_config.manifest(),
        "proposal_trigger": trigger.manifest(),
        "preference_config": {
            "capped_minimum_gauge_margin": preference_config.capped_minimum_gauge_margin,
            "capped_final_gauge_margin": preference_config.capped_final_gauge_margin,
        },
        "episodes": [{k: v for k, v in row.items() if k != "wall_seconds"} for row in episodes],
        "positives": positive_records,
        "anchors": anchor_records,
    }
    collection_sha = collection_content_sha256(collection)
    metadata = {
        "schema": "irisu-proposal-only-pair-residual-v1",
        "development_only": True,
        "promotion_eligible": False,
        "base_checkpoint_sha256": artifact.sha256,
        "base_model_sha256": model_sha,
        "base_metadata_sha256": gate._sha(dict(artifact.metadata)),
        "base_inference_config": base_inference,
        "immutable_base_role": "all live decisions, base candidates, and all branch continuations",
        "residual_role": "append at most one strong directed-pair proposal under trigger",
        "upstream_training_seeds": list(upstream),
        "residual_training_seeds": list(seeds),
        "training_seeds": sorted((*upstream, *seeds)),
        "seed_plan": seed_plan,
        "planner_config": planner_config.manifest(),
        "proposal_trigger": trigger.manifest(),
        "preference_config": collection["preference_config"],
        "label_rule": (
            "full-horizon strict survival with minimum-gauge nondomination, or "
            "4096-tick low-reserve capped tie with configured minimum+final gauge "
            "margins and no clear/score regression"
        ),
        "capped_ties_are_positive": "robust-long-horizon-reserve-only",
        "episodes": [{k: v for k, v in row.items() if k != "wall_seconds"} for row in episodes],
        "training_report": report,
        "selection_seed_plan": selection_seed_plan,
        "collection_content_sha256": collection_sha,
        "source_sha256": gate._file_sha(Path(__file__).resolve()),
    }
    args.output.mkdir(parents=True)
    checkpoint = args.output / "proposal-residual.pt"
    checkpoint_sha = save_steering_checkpoint(checkpoint, residual, metadata=metadata)
    gate._save_torch_new(
        args.output / "strict-labels.pt",
        {**collection, "content_sha256": collection_sha},
    )
    result = {**metadata, "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha}
    result["sha256"] = gate._sha(result)
    gate._write_json_new(args.output / "provenance.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--worker", type=Path, required=True)
    value.add_argument("--base-checkpoint", type=Path, required=True)
    value.add_argument("--base-sha256", required=True)
    value.add_argument("--seed-plan", type=Path, required=True)
    value.add_argument("--selection-seed-plan", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--maximum-ticks", type=int, default=60_000)
    value.add_argument("--maximum-queries", type=int, default=128)
    value.add_argument("--maximum-positives", type=int, default=32)
    value.add_argument("--minimum-positives", type=int, default=2)
    value.add_argument("--maximum-anchors", type=int, default=128)
    value.add_argument("--minimum-tick", type=int, default=40_000)
    value.add_argument("--low-reserve-minimum-tick", type=int, default=0)
    value.add_argument("--low-gauge", type=int, default=12_000)
    value.add_argument("--low-gauge-exit", type=int, default=16_000)
    value.add_argument("--probe-ticks", type=int, default=2048)
    value.add_argument("--low-gauge-probe-ticks", type=int, default=4096)
    value.add_argument("--training-steps", type=int, default=100)
    value.add_argument("--learning-rate", type=float, default=1e-5)
    value.add_argument("--functional-weight", type=float, default=10.0)
    value.add_argument("--maximum-anchor-logit-delta", type=float, default=0.05)
    value.add_argument("--capped-minimum-gauge-margin", type=int, default=1_000)
    value.add_argument("--capped-final-gauge-margin", type=int, default=1_000)
    value.add_argument("--training-seed", type=int, default=2026081301)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
