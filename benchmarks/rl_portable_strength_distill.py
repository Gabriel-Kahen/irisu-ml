#!/usr/bin/env python3
"""Development-only STRONG-versus-WEAK counterfactual distillation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_portable_gate_distill as gate  # noqa: E402
import rl_portable_pair_rank_distill as ranking  # noqa: E402
from irisu_env import IrisuEnv  # noqa: E402
from irisu_pointer.steering import SteeringDecision, SteeringIntent  # noqa: E402
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (  # noqa: E402
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    SteeringExample,
    steering_example_from_decision,
)
from irisu_rl.actions import SemanticAction, SemanticActionKind  # noqa: E402
from irisu_rl.encoding import TeacherStateEncoder  # noqa: E402


@dataclass(frozen=True, slots=True)
class StrengthLabel:
    example: SteeringExample
    confidence: float
    kind: str
    provenance: Mapping[str, object]

    @property
    def sha256(self) -> str:
        return gate._sha(self.manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-portable-strength-label-v1",
            "kind": self.kind,
            "confidence": self.confidence,
            "example_sha256": self.example.sha256,
            "provenance": dict(self.provenance),
            "provenance_sha256": self.example.provenance_sha256,
        }


@dataclass(frozen=True, slots=True)
class StrengthTrainingReport:
    steps: int
    labels: int
    strong_labels: int
    weak_labels: int
    initial_loss: float
    final_loss: float
    accuracy: float
    strong_recall: float
    weak_recall: float


def inference_config(act_logit_bias: float) -> dict[str, int | float | bool]:
    return {
        **gate.inference_config(act_logit_bias),
        "use_kind_head": True,
    }


def make_policy(
    model: GoalConditionedSteeringModel,
    model_sha256: str,
    act_logit_bias: float,
) -> Any:
    model.eval()
    return gate._SharedModelPolicy(
        GoalConditionedSteeringPolicy(
            model,
            **inference_config(act_logit_bias),
            artifact_sha256=model_sha256,
        )
    )


def with_strength(
    proposal: SteeringDecision, *, strong: bool
) -> SteeringDecision:
    if not proposal.is_shot:
        raise ValueError("strength counterfactual requires a shot")
    constructor = SemanticAction.strong if strong else SemanticAction.weak
    return SteeringDecision(
        constructor(proposal.action.x_norm, proposal.action.y_norm),
        proposal.intent,
        source_body_id=proposal.source_body_id,
        destination_body_id=proposal.destination_body_id,
        destination_chain_id=proposal.destination_chain_id,
        impact_x_sizes=proposal.impact_x_sizes,
        impact_y_sizes=proposal.impact_y_sizes,
        correction_index=proposal.correction_index,
        reason=("counterfactual STRONG" if strong else "counterfactual WEAK"),
    )


def choose_strength(
    strong: Any,
    weak: Any,
    *,
    incumbent_is_strong: bool,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[bool, str]:
    strong_wins, strong_reason = gate.reserve_choice(
        strong,
        weak,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    weak_wins, weak_reason = gate.reserve_choice(
        weak,
        strong,
        maximum_gauge_debt=maximum_gauge_debt,
        rescue_score_margin=rescue_score_margin,
    )
    if strong_wins and not weak_wins:
        return True, f"strong:{strong_reason}"
    if weak_wins and not strong_wins:
        return False, f"weak:{weak_reason}"
    return incumbent_is_strong, "incumbent-strength-tie-anchor"


def configure_trainable_scope(
    model: GoalConditionedSteeringModel, scope: str
) -> tuple[str, ...]:
    prefixes = {
        "kind-head": ("kind_head.",),
        "heads": (
            "act_head.",
            "wait_head.",
            "pair_head.",
            "kind_head.",
            "template_head.",
            "intent_head.",
        ),
        "all": ("",),
    }
    if scope not in prefixes:
        raise ValueError(f"unknown trainable scope: {scope}")
    selected = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes[scope])
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise RuntimeError("trainable scope selected no parameters")
    return tuple(selected)


def balanced_labels(labels: Sequence[StrengthLabel]) -> tuple[StrengthLabel, ...]:
    weak = sorted(
        (value for value in labels if value.example.kind_index == 0),
        key=lambda value: value.sha256,
    )
    strong = sorted(
        (value for value in labels if value.example.kind_index == 1),
        key=lambda value: value.sha256,
    )
    if not weak or not strong:
        return tuple(weak or strong)
    count = max(len(weak), len(strong))
    output = []
    for index in range(count):
        output.extend((weak[index % len(weak)], strong[index % len(strong)]))
    return tuple(output)


def train_strength_head(
    model: GoalConditionedSteeringModel,
    labels: Sequence[StrengthLabel],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> StrengthTrainingReport:
    view = balanced_labels(labels)
    if not view:
        raise ValueError("strength training requires labels")
    dataset = SteeringDataset([value.example for value in view])
    confidence = torch.tensor([value.confidence for value in view], dtype=torch.float32)
    device = next(model.parameters()).device
    full = dataset.as_tensors().to(device)
    confidence = confidence.to(device)

    def metrics() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model.eval()
        output = model(full.global_features, full.body_features, full.body_mask)
        rows = torch.arange(len(view), device=device)
        logits = output.kind_logits[
            rows, full.source_index, full.destination_index
        ]
        losses = F.cross_entropy(logits, full.kind_index, reduction="none")
        prediction = logits.argmax(-1)
        accuracy = (prediction == full.kind_index).float().mean()
        strong = full.kind_index == 1
        weak = ~strong
        strong_recall = (
            (prediction[strong] == 1).float().mean()
            if bool(strong.any())
            else accuracy.detach() * 0.0
        )
        weak_recall = (
            (prediction[weak] == 0).float().mean()
            if bool(weak.any())
            else accuracy.detach() * 0.0
        )
        return (confidence * losses).sum() / confidence.sum(), accuracy, strong_recall, weak_recall

    with torch.no_grad():
        initial, *_ = metrics()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=learning_rate,
    )
    model.eval()
    for _ in range(steps):
        indices = torch.randint(
            len(view), (min(batch_size, len(view)),), generator=generator
        )
        batch = dataset.as_tensors(indices.tolist()).to(device)
        weights = confidence[indices.to(device)]
        output = model(batch.global_features, batch.body_features, batch.body_mask)
        rows = torch.arange(len(indices), device=device)
        logits = output.kind_logits[
            rows, batch.source_index, batch.destination_index
        ]
        losses = F.cross_entropy(logits, batch.kind_index, reduction="none")
        loss = (weights * losses).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad], 5.0
        )
        optimizer.step()
    with torch.no_grad():
        final, accuracy, strong_recall, weak_recall = metrics()
    return StrengthTrainingReport(
        steps,
        len(view),
        sum(value.example.kind_index == 1 for value in view),
        sum(value.example.kind_index == 0 for value in view),
        float(initial),
        float(final),
        float(accuracy),
        float(strong_recall),
        float(weak_recall),
    )


def collect_episode(
    *,
    runtime: Path,
    model: GoalConditionedSteeringModel,
    model_sha256: str,
    seed: int,
    maximum_ticks: int,
    short_horizon: int,
    long_horizon: int,
    gauge_threshold: int,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
    wait_ticks: int,
    act_logit_bias: float,
    iteration: int,
) -> tuple[list[StrengthLabel], dict[str, object], Mapping[str, Any]]:
    policy = make_policy(model, model_sha256, act_logit_bias)
    policy.reset(seed)
    encoder = TeacherStateEncoder()
    labels: list[StrengthLabel] = []
    reasons: Counter[str] = Counter()
    proposed = deployment_kept = deployment_suppressed = strong_labels = weak_labels = 0
    started = time.monotonic()
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": maximum_ticks + long_horizon + 64},
    ) as env:
        runtime_manifest = env.runner_identity_manifest()
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("portable reset seed differs")
        terminated = truncated = False
        decisions = 0
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            policy_before = copy.deepcopy(policy)
            proposal = policy.predict(observation)
            executed = proposal
            if proposal.is_shot:
                proposed += 1
                horizon = gate.adaptive_horizon(
                    int(observation["gauge"]),
                    short_horizon=short_horizon,
                    long_horizon=long_horizon,
                    gauge_threshold=gauge_threshold,
                )
                wait = SteeringDecision(
                    SemanticAction.wait(wait_ticks),
                    SteeringIntent.WAIT,
                    reason="strength teacher WAIT",
                )
                strong = with_strength(proposal, strong=True)
                weak = with_strength(proposal, strong=False)
                outcomes = ranking.evaluate_candidates(
                    env,
                    observation,
                    policy_before,
                    (wait, strong, weak),
                    horizon=horizon,
                    wait_ticks=wait_ticks,
                    maximum_gauge_debt=maximum_gauge_debt,
                    rescue_score_margin=rescue_score_margin,
                )
                incumbent_strong = (
                    SemanticActionKind(proposal.action.kind)
                    is SemanticActionKind.FIRE_STRONG
                )
                choose_strong, reason = choose_strength(
                    outcomes[1],
                    outcomes[2],
                    incumbent_is_strong=incumbent_strong,
                    maximum_gauge_debt=maximum_gauge_debt,
                    rescue_score_margin=rescue_score_margin,
                )
                reasons[reason] += 1
                decision = strong if choose_strong else weak
                strong_labels += int(choose_strong)
                weak_labels += int(not choose_strong)
                provenance = {
                    "schema": "irisu-portable-strength-label-provenance-v1",
                    "seed": seed,
                    "tick": int(observation["tick"]),
                    "iteration": iteration,
                    "horizon": horizon,
                    "reason": reason,
                    "incumbent_is_strong": incumbent_strong,
                    "strong_outcome": outcomes[1].manifest(),
                    "weak_outcome": outcomes[2].manifest(),
                }
                example = steering_example_from_decision(
                    observation,
                    decision,
                    episode_identity=f"strength:{seed:08x}:{int(observation['tick'])}",
                    provenance_sha256=gate._sha(provenance),
                    encoder=encoder,
                    pointer_spec=policy.pointer_spec,
                    require_representable_template=False,
                )
                if example is None:
                    raise RuntimeError("strength teacher decision is not representable")
                labels.append(
                    StrengthLabel(
                        example,
                        ranking.causal_confidence(
                            outcomes[1],
                            outcomes[2],
                            horizon=horizon,
                            maximum_gauge_debt=maximum_gauge_debt,
                        ),
                        "strong" if choose_strong else "weak",
                        provenance,
                    )
                )
                proposal_index = 1 if incumbent_strong else 2
                keep, deployment_reason = gate.reserve_choice(
                    outcomes[proposal_index],
                    outcomes[0],
                    maximum_gauge_debt=maximum_gauge_debt,
                    rescue_score_margin=rescue_score_margin,
                )
                if keep:
                    deployment_kept += 1
                else:
                    deployment_suppressed += 1
                    policy = policy_before
                    executed = wait
            observation, terminated, truncated = gate._step_decision(
                env, observation, executed, maximum_tick=maximum_ticks
            )
            decisions += 1
        episode = {
            "seed": seed,
            "iteration": iteration,
            "final_tick": int(observation["tick"]),
            "final_score": int(observation.get("score", 0)),
            "final_gauge": int(observation.get("gauge", 0)),
            "terminated": bool(terminated or observation.get("terminated", False)),
            "truncated": bool(truncated or observation.get("truncated", False)),
            "decisions": decisions,
            "proposed_shots": proposed,
            "deployment_kept": deployment_kept,
            "deployment_suppressed": deployment_suppressed,
            "strength_labels": len(labels),
            "strong_labels": strong_labels,
            "weak_labels": weak_labels,
            "strength_reasons": dict(sorted(reasons.items())),
            "wall_seconds": time.monotonic() - started,
        }
    return labels, episode, runtime_manifest


def _source_identity() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        Path(gate.__file__).resolve(),
        Path(ranking.__file__).resolve(),
        ROOT / "python/irisu_pointer/steering_learning.py",
        ROOT / "python/irisu_pointer/steering_checkpoint.py",
    )
    value = {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {str(path.relative_to(ROOT)): gate._file_sha(path) for path in paths},
    }
    return {**value, "sha256": gate._sha(value)}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    if (
        not args.train_seeds
        or len(set(args.train_seeds)) != len(args.train_seeds)
        or any(type(seed) is not int or not 0 <= seed <= 0xFFFF_FFFF for seed in args.train_seeds)
    ):
        raise ValueError("caller-supplied train seeds must be unique uint32 values")
    artifact = load_steering_checkpoint(
        args.base_checkpoint, expected_sha256=args.base_sha256
    )
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    strength_seeds = tuple(sorted(args.train_seeds))
    training_seeds = tuple(sorted(set(upstream) | set(strength_seeds)))
    runtime = args.runtime.resolve(strict=True)
    bound_inference = inference_config(args.act_logit_bias)
    planner = {
        "probe_ticks": args.short_horizon,
        "rescue_probe_ticks": args.long_horizon,
        "rescue_gauge_threshold": args.gauge_threshold,
        "wait_ticks": args.wait_ticks,
        "maximum_gauge_debt": args.maximum_gauge_debt,
        "rescue_score_margin": args.rescue_score_margin,
    }
    warm_start = {
        "path": str(artifact.path),
        "checkpoint_sha256": artifact.sha256,
        "metadata_sha256": gate._sha(dict(artifact.metadata)),
        "model_state_sha256": gate._model_state_sha(artifact.model),
        "architecture_sha256": artifact.model.architecture_sha256,
    }
    args.output.mkdir(parents=True)
    model = artifact.model
    trainable = configure_trainable_scope(model, args.trainable_scope)
    labels: list[StrengthLabel] = []
    episodes: list[dict[str, object]] = []
    runtimes: list[Mapping[str, Any]] = []
    training_reports: list[dict[str, object]] = []
    started = time.monotonic()
    for iteration in range(args.iterations):
        rollout_model_sha256 = gate._model_state_sha(model)
        for seed in args.train_seeds:
            new, episode, runtime_manifest = collect_episode(
                runtime=runtime,
                model=model,
                model_sha256=rollout_model_sha256,
                seed=seed,
                maximum_ticks=args.maximum_ticks,
                short_horizon=args.short_horizon,
                long_horizon=args.long_horizon,
                gauge_threshold=args.gauge_threshold,
                maximum_gauge_debt=args.maximum_gauge_debt,
                rescue_score_margin=args.rescue_score_margin,
                wait_ticks=args.wait_ticks,
                act_logit_bias=args.act_logit_bias,
                iteration=iteration,
            )
            labels.extend(new)
            episode["rollout_model_state_sha256"] = rollout_model_sha256
            episodes.append(episode)
            runtimes.append(runtime_manifest)
            print(json.dumps(episode, sort_keys=True), flush=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.training_seed + iteration)
            training_reports.append(
                asdict(
                    train_strength_head(
                        model,
                        labels,
                        steps=args.training_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        seed=args.training_seed + iteration,
                    )
                )
            )
    raw_dataset = SteeringDataset([value.example for value in labels])
    training_view = balanced_labels(labels)
    balanced_dataset = SteeringDataset([value.example for value in training_view])
    payload = {
        "schema": "irisu-portable-strength-labels-v1",
        "base_checkpoint_sha256": artifact.sha256,
        "upstream_training_seeds": list(upstream),
        "strength_training_seeds": list(strength_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": bound_inference,
        "adaptive_wait_gate": planner,
        "raw_dataset_manifest": raw_dataset.manifest(),
        "balanced_dataset_manifest": balanced_dataset.manifest(),
        "labels": [
            {
                "manifest": value.manifest(),
                "example": value.example.manifest(),
                "global_features": torch.from_numpy(value.example.observation.global_features),
                "body_features": torch.from_numpy(value.example.observation.body_features),
                "body_mask": torch.from_numpy(value.example.observation.body_mask),
            }
            for value in labels
        ],
    }
    labels_path = args.output / "strength-labels.pt"
    gate._save_torch_new(labels_path, payload)
    labels_sha256 = gate._file_sha(labels_path)
    source = _source_identity()
    metadata = {
        "schema": "irisu-portable-strength-distillation-v1",
        "development_only": True,
        "promotion_eligible": False,
        "held_out_seeds_used": False,
        "base_checkpoint_sha256": artifact.sha256,
        "warm_start": warm_start,
        "upstream_training_seeds": list(upstream),
        "strength_training_seeds": list(strength_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": bound_inference,
        "adaptive_wait_gate": planner,
        "runtime": {"path": str(runtime), "sha256": gate._file_sha(runtime)},
        "source": source,
        "raw_dataset_sha256": raw_dataset.sha256,
        "balanced_dataset_sha256": balanced_dataset.sha256,
        "labels_artifact_sha256": labels_sha256,
        "label_count": len(labels),
        "strong_labels": sum(value.example.kind_index == 1 for value in labels),
        "weak_labels": sum(value.example.kind_index == 0 for value in labels),
        "episodes": [
            {key: value for key, value in row.items() if key != "wall_seconds"}
            for row in episodes
        ],
        "scores": [int(row["final_score"]) for row in episodes],
        "median_score": statistics.median(int(row["final_score"]) for row in episodes),
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": list(trainable),
        "training_reports": training_reports,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    }
    checkpoint_path = args.output / "strength-distilled.pt"
    checkpoint_sha256 = save_steering_checkpoint(
        checkpoint_path, model, metadata=metadata
    )
    report = {
        **metadata,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "labels_artifact": labels_path.name,
        "runtime_manifests": runtimes,
        "operational": {
            "output": str(args.output.resolve()),
            "episode_wall_seconds": [float(row["wall_seconds"]) for row in episodes],
        },
        "wall_seconds": time.monotonic() - started,
    }
    report["sha256"] = gate._sha(report)
    gate._write_json_new(args.output / "provenance.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--train-seeds", type=gate.parse_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=gate.PORTABLE)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--maximum-ticks", type=int, default=20_000)
    parser.add_argument("--short-horizon", type=int, default=128)
    parser.add_argument("--long-horizon", type=int, default=256)
    parser.add_argument("--gauge-threshold", type=int, default=30_000)
    parser.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    parser.add_argument("--rescue-score-margin", type=int, default=500)
    parser.add_argument("--wait-ticks", type=int, default=16)
    parser.add_argument("--act-logit-bias", type=float, default=1.0)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--training-seed", type=int, default=2026081102)
    parser.add_argument(
        "--trainable-scope", choices=("kind-head", "heads", "all"), default="kind-head"
    )
    args = parser.parse_args(argv)
    positive = (
        "iterations",
        "maximum_ticks",
        "short_horizon",
        "long_horizon",
        "maximum_gauge_debt",
        "rescue_score_margin",
        "wait_ticks",
        "training_steps",
        "batch_size",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("counts and horizons must be positive")
    if args.long_horizon < args.short_horizon:
        parser.error("long horizon must be at least short horizon")
    if args.gauge_threshold < 0:
        parser.error("gauge threshold must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if len(args.base_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in args.base_sha256
    ):
        parser.error("base SHA-256 must be lowercase hexadecimal")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "labels": report["label_count"],
                "strong": report["strong_labels"],
                "weak": report["weak_labels"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
