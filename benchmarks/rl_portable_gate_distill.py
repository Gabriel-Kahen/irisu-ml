#!/usr/bin/env python3
"""Development-only DAgger distillation of an adaptive portable wait gate.

The learner visits every rollout state.  Each shot it proposes is compared
against WAIT from a byte-identical portable snapshot at 128 ticks normally and
256 ticks when gauge is low.  The gate verdict becomes an act-head label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from irisu_env import Action, ActionKind, IrisuEnv  # noqa: E402
from irisu_pointer.shot_necessity import (  # noqa: E402
    ExactWaitDominanceGate,
    GateVerdict,
    ProbeOutcome,
    WaitDominanceConfig,
    choose_shot,
)
from irisu_pointer.steering import SteeringDecision  # noqa: E402
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
    train_goal_conditioned_steering,
)
from irisu_rl.encoding import TeacherStateEncoder  # noqa: E402


PORTABLE = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/"
    "libirisu_clone.so"
)

INFERENCE_CONFIG = {
    "cooldown_ticks": 16,
    "minimum_pair_closure_sizes": 0.05,
    "impact_side_sizes": 0.5,
    "impact_below_sizes": 0.75,
    "source_velocity_lead_ticks": 1.0,
    "ticks_per_second": 50.0,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha(model: GoalConditionedSteeringModel) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip(), 0) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train seeds must be comma-separated uint32 values") from exc
    if not seeds or any(type(seed) is not int or not 0 <= seed <= 0xFFFF_FFFF for seed in seeds):
        raise argparse.ArgumentTypeError("train seeds must be comma-separated uint32 values")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("train seeds must be unique")
    return seeds


def checkpoint_training_seeds(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    """Fail closed unless a checkpoint declares its complete seed lineage."""

    values: set[int] = set()
    found = False
    for key in ("training_seeds", "demonstration_seeds"):
        raw = metadata.get(key)
        if raw is None:
            continue
        found = True
        if (
            not isinstance(raw, list)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 0xFFFF_FFFF
                for seed in raw
            )
            or len(set(raw)) != len(raw)
        ):
            raise ValueError(f"checkpoint {key} must contain unique uint32 integers")
        values.update(raw)
    if not found:
        raise ValueError("warm-start checkpoint does not declare its training seeds")
    return tuple(sorted(values))


def inference_config(act_logit_bias: float) -> dict[str, int | float]:
    return {**INFERENCE_CONFIG, "act_logit_bias": float(act_logit_bias)}


def reserve_choice(
    shot: ProbeOutcome,
    wait: ProbeOutcome,
    *,
    maximum_gauge_debt: int,
    rescue_score_margin: int,
) -> tuple[bool, str]:
    """Deployment-matched survival/score/gauge-reserve gate objective."""

    if shot.survival_ticks != wait.survival_ticks:
        return (
            shot.survival_ticks > wait.survival_ticks,
            "shot-survival"
            if shot.survival_ticks > wait.survival_ticks
            else "wait-survival",
        )
    shot_failed = shot.terminated or shot.truncated
    wait_failed = wait.terminated or wait.truncated
    if shot_failed != wait_failed:
        return not shot_failed, "shot-rescue" if wait_failed else "wait-safer"
    gauge_debt = wait.final_gauge - shot.final_gauge
    score_gain = shot.score - wait.score
    if gauge_debt > maximum_gauge_debt and score_gain < rescue_score_margin:
        return False, "wait-reserve"
    return choose_shot(shot, wait, gauge_advantage=1)


class ReserveWaitGate(ExactWaitDominanceGate):
    def __init__(
        self,
        *args: object,
        maximum_gauge_debt: int,
        rescue_score_margin: int,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_gauge_debt = maximum_gauge_debt
        self.rescue_score_margin = rescue_score_margin

    def evaluate(self, *args: object, **kwargs: object) -> GateVerdict:
        verdict = super().evaluate(*args, **kwargs)
        execute, reason = reserve_choice(
            verdict.shot,
            verdict.wait,
            maximum_gauge_debt=self.maximum_gauge_debt,
            rescue_score_margin=self.rescue_score_margin,
        )
        return GateVerdict(
            execute,
            reason,
            verdict.shot,
            verdict.wait,
            verdict.restore_checks,
        )


def adaptive_horizon(
    gauge: int, *, short_horizon: int, long_horizon: int, gauge_threshold: int
) -> int:
    return long_horizon if gauge <= gauge_threshold else short_horizon


def configure_trainable_scope(
    model: GoalConditionedSteeringModel, scope: str
) -> tuple[str, ...]:
    prefixes = {
        "act-head": ("act_head.",),
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
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes[scope])
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise RuntimeError("trainable scope selected no parameters")
    return tuple(selected)


def balanced_examples(examples: Sequence[SteeringExample]) -> tuple[SteeringExample, ...]:
    """Return a deterministic, exactly action-balanced oversampled view."""

    shot = sorted((value for value in examples if value.is_shot), key=lambda value: value.sha256)
    wait = sorted((value for value in examples if not value.is_shot), key=lambda value: value.sha256)
    if not shot or not wait:
        return tuple(shot or wait)
    count = max(len(shot), len(wait))
    output: list[SteeringExample] = []
    for index in range(count):
        output.extend((shot[index % len(shot)], wait[index % len(wait)]))
    return tuple(output)


def _copy_policy(policy: GoalConditionedSteeringPolicy) -> GoalConditionedSteeringPolicy:
    return copy.deepcopy(policy, {id(policy.model): policy.model})


class _SharedModelPolicy:
    """Give gate branch copies private controller state but shared frozen weights."""

    def __init__(self, inner: GoalConditionedSteeringPolicy):
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __deepcopy__(self, memo: dict[int, object]) -> "_SharedModelPolicy":
        copied = _SharedModelPolicy(_copy_policy(self.inner))
        memo[id(self)] = copied
        return copied


def _make_policy(
    model: GoalConditionedSteeringModel, checkpoint_sha256: str, act_logit_bias: float
) -> _SharedModelPolicy:
    model.eval()
    return _SharedModelPolicy(
        GoalConditionedSteeringPolicy(
            model,
            **inference_config(act_logit_bias),
            artifact_sha256=checkpoint_sha256,
        )
    )


@dataclass(frozen=True, slots=True)
class Label:
    example: SteeringExample
    kind: str
    provenance: Mapping[str, object]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-portable-adaptive-gate-label-v1",
            "kind": self.kind,
            "provenance": dict(self.provenance),
            "provenance_sha256": self.example.provenance_sha256,
            "example_sha256": self.example.sha256,
        }


def _label(
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    kind: str,
    provenance: Mapping[str, object],
    encoder: TeacherStateEncoder,
    pointer_spec: Any,
) -> Label:
    owned = dict(provenance)
    provenance_sha256 = _sha(owned)
    example = steering_example_from_decision(
        observation,
        decision,
        episode_identity=(
            f"portable-gate:{kind}:{int(owned['seed']):08x}:{int(owned['tick'])}"
        ),
        provenance_sha256=provenance_sha256,
        encoder=encoder,
        pointer_spec=pointer_spec,
        require_representable_template=False,
    )
    if example is None:
        raise RuntimeError("gate decision could not be represented as steering supervision")
    return Label(example, kind, owned)


def _step_decision(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    maximum_tick: int,
) -> tuple[Mapping[str, Any], bool, bool]:
    current = observation
    terminated = bool(current.get("terminated", False))
    truncated = bool(current.get("truncated", False))
    for action in decision.primitive_actions():
        remaining = maximum_tick - int(current["tick"])
        if terminated or truncated or remaining <= 0:
            break
        if ActionKind.parse(action.kind) is ActionKind.WAIT:
            action = Action.wait(min(int(action.wait_ticks), remaining))
        current, _reward, terminated, truncated, _info = env.step(action)
    return current, terminated, truncated


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
    wait_anchor_stride: int,
    maximum_wait_anchors: int,
    act_logit_bias: float,
    iteration: int,
) -> tuple[list[Label], dict[str, object], Mapping[str, Any]]:
    policy = _make_policy(model, model_sha256, act_logit_bias)
    policy.reset(seed)
    encoder = TeacherStateEncoder()
    gates = {
        short_horizon: ReserveWaitGate(
            lambda decision: decision.primitive_actions(),
            config=WaitDominanceConfig(short_horizon, wait_ticks, 16),
            maximum_gauge_debt=maximum_gauge_debt,
            rescue_score_margin=rescue_score_margin,
        ),
        long_horizon: ReserveWaitGate(
            lambda decision: decision.primitive_actions(),
            config=WaitDominanceConfig(long_horizon, wait_ticks, 16),
            maximum_gauge_debt=maximum_gauge_debt,
            rescue_score_margin=rescue_score_margin,
        ),
    }
    labels: list[Label] = []
    reasons: Counter[str] = Counter()
    proposed = kept = suppressed = long_queries = 0
    wait_seen = wait_anchors = decisions = 0
    started = time.monotonic()
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": maximum_ticks + long_horizon + 64},
    ) as env:
        runtime_manifest = env.runner_identity_manifest()
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("portable reset seed differs from requested train seed")
        terminated = truncated = False
        while int(observation["tick"]) < maximum_ticks and not (terminated or truncated):
            policy_before = copy.deepcopy(policy)
            proposal = policy.predict(observation)
            executed = proposal
            if proposal.is_shot:
                proposed += 1
                horizon = adaptive_horizon(
                    int(observation["gauge"]),
                    short_horizon=short_horizon,
                    long_horizon=long_horizon,
                    gauge_threshold=gauge_threshold,
                )
                long_queries += horizon == long_horizon and long_horizon != short_horizon
                gate = gates[horizon]
                verdict = gate.evaluate(env, observation, policy_before, policy, proposal)
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                    kind = "gate-kept-shot"
                else:
                    suppressed += 1
                    kind = "gate-suppressed-wait"
                    policy = policy_before
                    executed = gate.wait_decision(verdict.reason)
                labels.append(
                    _label(
                        observation,
                        executed,
                        kind=kind,
                        provenance={
                            "schema": "irisu-portable-adaptive-gate-label-provenance-v1",
                            "seed": seed,
                            "tick": int(observation["tick"]),
                            "iteration": iteration,
                            "kind": kind,
                            "horizon": horizon,
                            "gauge": int(observation["gauge"]),
                            "gate_verdict": verdict.manifest(),
                        },
                        encoder=encoder,
                        pointer_spec=policy.pointer_spec,
                    )
                )
            else:
                wait_seen += 1
                if (
                    wait_seen % wait_anchor_stride == 0
                    and wait_anchors < maximum_wait_anchors
                ):
                    wait_anchors += 1
                    labels.append(
                        _label(
                            observation,
                            proposal,
                            kind="learner-wait-anchor",
                            provenance={
                                "schema": "irisu-portable-adaptive-gate-label-provenance-v1",
                                "seed": seed,
                                "tick": int(observation["tick"]),
                                "iteration": iteration,
                                "kind": "learner-wait-anchor",
                                "wait_ordinal": wait_seen,
                            },
                            encoder=encoder,
                            pointer_spec=policy.pointer_spec,
                        )
                    )
            observation, terminated, truncated = _step_decision(
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
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "long_horizon_queries": long_queries,
            "learner_waits_seen": wait_seen,
            "wait_anchors": wait_anchors,
            "gate_reasons": dict(sorted(reasons.items())),
            "labels": len(labels),
            "wall_seconds": time.monotonic() - started,
        }
    return labels, episode, runtime_manifest


def _write_json_new(path: Path, value: Mapping[str, object]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_torch_new(path: Path, value: object) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(value, temporary)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_identity() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "python/irisu_pointer/shot_necessity.py",
        ROOT / "python/irisu_pointer/steering_learning.py",
    )
    value = {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {str(path.relative_to(ROOT)): _file_sha(path) for path in paths},
    }
    return {**value, "sha256": _sha(value)}


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
    upstream_training_seeds = checkpoint_training_seeds(artifact.metadata)
    dagger_training_seeds = tuple(sorted(args.train_seeds))
    training_seeds = tuple(
        sorted(set(upstream_training_seeds) | set(dagger_training_seeds))
    )
    bound_inference_config = inference_config(args.act_logit_bias)
    planner_config = {
        "probe_ticks": args.short_horizon,
        "rescue_probe_ticks": args.long_horizon,
        "rescue_gauge_threshold": args.gauge_threshold,
        "wait_ticks": args.wait_ticks,
        "maximum_gauge_debt": args.maximum_gauge_debt,
        "rescue_score_margin": args.rescue_score_margin,
    }
    warm_start = {
        "path": str(args.base_checkpoint.resolve()),
        "checkpoint_sha256": artifact.sha256,
        "metadata_sha256": _sha(dict(artifact.metadata)),
        "model_state_sha256": _model_state_sha(artifact.model),
        "architecture_sha256": artifact.model.architecture_sha256,
    }
    args.runtime.resolve(strict=True)
    args.output.mkdir(parents=True)
    model = artifact.model
    trainable_parameters = configure_trainable_scope(model, args.trainable_scope)
    labels: list[Label] = []
    episodes: list[dict[str, object]] = []
    runtimes: list[Mapping[str, Any]] = []
    training_reports: list[dict[str, object]] = []
    started = time.monotonic()
    for iteration in range(args.iterations):
        rollout_model_sha256 = _model_state_sha(model)
        for seed in args.train_seeds:
            new, episode, runtime = collect_episode(
                runtime=args.runtime,
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
                wait_anchor_stride=args.wait_anchor_stride,
                maximum_wait_anchors=args.maximum_wait_anchors,
                act_logit_bias=args.act_logit_bias,
                iteration=iteration,
            )
            labels.extend(new)
            episodes.append(episode)
            episode["rollout_model_state_sha256"] = rollout_model_sha256
            runtimes.append(runtime)
            print(json.dumps(episode, sort_keys=True), flush=True)
        view = balanced_examples([value.example for value in labels])
        if not view:
            raise RuntimeError("gate collection produced no labels")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.training_seed + iteration)
            training_reports.append(
                asdict(
                    train_goal_conditioned_steering(
                        model,
                        SteeringDataset(view),
                        steps=args.training_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        seed=args.training_seed + iteration,
                    )
                )
            )

    raw_dataset = SteeringDataset([value.example for value in labels])
    training_view = SteeringDataset(balanced_examples([value.example for value in labels]))
    label_payload = {
        "schema": "irisu-portable-adaptive-gate-labels-v1",
        "base_checkpoint_sha256": artifact.sha256,
        "upstream_training_seeds": list(upstream_training_seeds),
        "dagger_training_seeds": list(dagger_training_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": bound_inference_config,
        "adaptive_wait_gate": planner_config,
        "raw_dataset_manifest": raw_dataset.manifest(),
        "balanced_dataset_manifest": training_view.manifest(),
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
    labels_path = args.output / "labels.pt"
    _save_torch_new(labels_path, label_payload)
    labels_sha256 = _file_sha(labels_path)
    gate_totals = {
        "proposed_shots": sum(int(row["proposed_shots"]) for row in episodes),
        "kept_shots": sum(int(row["kept_shots"]) for row in episodes),
        "suppressed_shots": sum(int(row["suppressed_shots"]) for row in episodes),
        "long_horizon_queries": sum(int(row["long_horizon_queries"]) for row in episodes),
        "wait_anchors": sum(int(row["wait_anchors"]) for row in episodes),
    }
    if gate_totals["proposed_shots"] != gate_totals["kept_shots"] + gate_totals["suppressed_shots"]:
        raise RuntimeError("not every proposed shot received exactly one gate label")
    label_counts = dict(sorted(Counter(value.kind for value in labels).items()))
    scores = [int(row["final_score"]) for row in episodes]
    source = _source_identity()
    deterministic_episodes = [
        {key: value for key, value in row.items() if key != "wall_seconds"}
        for row in episodes
    ]
    training_manifest = {
        "schema": "irisu-portable-adaptive-gate-distillation-v1",
        "development_only": True,
        "promotion_eligible": False,
        "held_out_seeds_used": False,
        "base_checkpoint_sha256": artifact.sha256,
        "warm_start": warm_start,
        "upstream_training_seeds": list(upstream_training_seeds),
        "dagger_training_seeds": list(dagger_training_seeds),
        "training_seeds": list(training_seeds),
        "inference_config": bound_inference_config,
        "adaptive_wait_gate": planner_config,
        "runtime": {"path": str(args.runtime.resolve()), "sha256": _file_sha(args.runtime)},
        "source": source,
        "raw_dataset_sha256": raw_dataset.sha256,
        "balanced_dataset_sha256": training_view.sha256,
        "labels_artifact_sha256": labels_sha256,
        "episodes": deterministic_episodes,
        "gate_totals": gate_totals,
        "label_counts": label_counts,
        "label_count": len(labels),
        "scores": scores,
        "median_score": statistics.median(scores),
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": list(trainable_parameters),
        "training_reports": training_reports,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    }
    checkpoint_path = args.output / "gate-distilled.pt"
    checkpoint_sha256 = save_steering_checkpoint(
        checkpoint_path, model, metadata=training_manifest
    )
    report = {
        **training_manifest,
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
    report["sha256"] = _sha(report)
    _write_json_new(args.output / "provenance.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--train-seeds", type=parse_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=PORTABLE)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--maximum-ticks", type=int, default=20_000)
    parser.add_argument("--short-horizon", type=int, default=128)
    parser.add_argument("--long-horizon", type=int, default=256)
    parser.add_argument("--gauge-threshold", type=int, default=30_000)
    parser.add_argument("--maximum-gauge-debt", type=int, default=1_000)
    parser.add_argument("--rescue-score-margin", type=int, default=500)
    parser.add_argument("--wait-ticks", type=int, default=16)
    parser.add_argument("--wait-anchor-stride", type=int, default=16)
    parser.add_argument("--maximum-wait-anchors", type=int, default=64)
    parser.add_argument("--act-logit-bias", type=float, default=1.0)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--training-seed", type=int, default=2026081002)
    parser.add_argument(
        "--trainable-scope", choices=("act-head", "heads", "all"), default="act-head"
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
        "wait_anchor_stride",
        "training_steps",
        "batch_size",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("counts and horizons must be positive")
    if args.long_horizon < args.short_horizon:
        parser.error("long horizon must be at least the short horizon")
    if args.gauge_threshold < 0 or args.maximum_wait_anchors < 0:
        parser.error("gauge threshold and maximum wait anchors must be nonnegative")
    if len(args.base_sha256) != 64 or any(value not in "0123456789abcdef" for value in args.base_sha256):
        parser.error("base SHA-256 must be 64 lowercase hexadecimal characters")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "dataset_sha256": report["raw_dataset_sha256"],
                "labels": report["label_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
