#!/usr/bin/env python3
"""Multi-seed snapshot-search expert iteration for frozen-v5 steering.

This is a development trainer, not a promotion claim.  Rollouts visit states
under a deterministic learner/frozen-v5 mixture.  At queried states the exact
same transactional search evaluates WAIT and legal directed-pair shots, then
distils only strict survival/score improvements back into the learner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, IrisuEnv
from irisu_pointer.branching import TransactionalBranches
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_checkpoint import (
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    SteeringExample,
    steering_example_from_decision,
    train_goal_conditioned_steering,
)
from irisu_pointer.policy import encoded_body_ids
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/r3/development/r3d-survival-v5-20260729/long-development.pt"
BASE_SHA256 = "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
PORTABLE = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/libirisu_clone.so"
EXACT_WORKER = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/exact-runtime-backup/irisu-exact-worker"

INFERENCE_CONFIG = {
    "cooldown_ticks": 16,
    "minimum_pair_closure_sizes": 0.05,
    "impact_side_sizes": 0.5,
    "impact_below_sizes": 0.75,
    "source_velocity_lead_ticks": 1.0,
    "ticks_per_second": 50.0,
    "act_logit_bias": 1.0,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seeds(label: str, count: int) -> tuple[int, ...]:
    """Stable, unique uint32 seeds with an auditable namespace."""

    if not label or count < 1:
        raise ValueError("seed namespace and count must be positive")
    result: list[int] = []
    ordinal = 0
    while len(result) < count:
        value = int.from_bytes(hashlib.sha256(f"{label}:{ordinal}".encode()).digest()[:4], "big")
        if value not in result:
            result.append(value)
        ordinal += 1
    return tuple(result)


def _policy(model: GoalConditionedSteeringModel, identity: str) -> GoalConditionedSteeringPolicy:
    model.eval()
    return GoalConditionedSteeringPolicy(
        model,
        **INFERENCE_CONFIG,
        artifact_sha256=identity,
    )


def checkpoint_training_seeds(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the complete declared environment-seed lineage of a checkpoint."""

    values: set[int] = set()
    found = False
    for key in ("training_seeds", "demonstration_seeds"):
        raw = metadata.get(key)
        if raw is None:
            continue
        found = True
        if not isinstance(raw, list) or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFF_FFFF
            for seed in raw
        ):
            raise ValueError(f"checkpoint {key} must contain uint32 integers")
        values.update(raw)
    if not found:
        raise ValueError("checkpoint does not declare its training seeds")
    return tuple(sorted(values))


def _copy_policy_state(policy: Any) -> Any:
    """Copy mutable controller state while sharing immutable model weights."""

    model = getattr(policy, "model", None)
    memo = {} if model is None else {id(model): model}
    return copy.deepcopy(policy, memo)


@dataclass(frozen=True, slots=True)
class MixturePrediction:
    selected: SteeringDecision
    learner_reference: SteeringDecision
    base_policy: Any
    learner_policy: Any
    base_continuation: Any


def predict_mixture(
    base_policy: Any,
    learner_policy: Any,
    observation: Mapping[str, Any],
    *,
    execute_base: bool,
) -> MixturePrediction:
    """Predict both shadows but advance state only for the executed policy."""

    base_before = _copy_policy_state(base_policy)
    learner_before = _copy_policy_state(learner_policy)
    learner_decision = learner_policy.predict(observation)
    base_decision = base_policy.predict(observation)
    return MixturePrediction(
        base_decision if execute_base else learner_decision,
        learner_decision,
        base_policy if execute_base else base_before,
        learner_before if execute_base else learner_policy,
        base_before,
    )


def _bodies(observation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(body["id"]): body
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and isinstance(body.get("id"), int)
    }


@torch.no_grad()
def legal_candidates(
    policy: GoalConditionedSteeringPolicy,
    observation: Mapping[str, Any],
    *,
    wait_ticks: int,
    maximum_pairs: int | None,
) -> tuple[SteeringDecision, ...]:
    """Return WAIT plus model-ranked, public-semantics-legal pair shots."""

    encoded = policy.encoder.encode([observation])
    active = np.flatnonzero(encoded.body_mask[0])
    width = int(active[-1]) + 1 if active.size else 1
    device = next(policy.model.parameters()).device
    output = policy.model(
        torch.from_numpy(encoded.global_features).to(device),
        torch.from_numpy(encoded.body_features[:, :width]).to(device),
        torch.from_numpy(encoded.body_mask[:, :width]).to(device),
    )
    identifiers = list(encoded_body_ids(encoded, observation)[:width])
    bodies = _bodies(observation)
    wait = SteeringDecision(
        policy.action_spec.validate(SemanticAction.wait(wait_ticks)),
        SteeringIntent.WAIT,
        reason="snapshot-search WAIT candidate",
    )
    flat = output.legal_pair_mask[0].flatten().nonzero(as_tuple=False).reshape(-1)
    ranked = flat[output.pair_logits[0].flatten()[flat].argsort(descending=True)]
    candidates: list[SteeringDecision] = [wait]
    for raw in ranked.tolist():
        source_index, destination_index = divmod(int(raw), width)
        source_id, destination_id = identifiers[source_index], identifiers[destination_index]
        if source_id is None or destination_id is None:
            continue
        source, destination = bodies.get(source_id), bodies.get(destination_id)
        if source is None or destination is None:
            continue
        analytic = policy._analytic_action(source, destination)
        if analytic is None:
            continue
        action, impact_x, impact_y = analytic
        candidates.append(
            SteeringDecision(
                action,
                SteeringIntent.MATCH_ROTTEN
                if str(destination.get("lifecycle", "")) == "rotten"
                else SteeringIntent.STEER_MATCH,
                source_body_id=source_id,
                destination_body_id=destination_id,
                destination_chain_id=int(destination.get("chain_id", 0)),
                impact_x_sizes=impact_x,
                impact_y_sizes=impact_y,
                reason="snapshot-search legal directed-pair candidate",
            )
        )
        if maximum_pairs is not None and len(candidates) - 1 >= maximum_pairs:
            break
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    candidate_index: int
    survival_ticks: int
    score_delta: int
    final_gauge: int
    terminated: bool
    truncated: bool
    invalid_actions: int = 0

    @property
    def objective(self) -> tuple[int, int, int, int, int]:
        # A branch that survives the whole probe always beats a terminal branch.
        return (
            self.invalid_actions == 0,
            not self.terminated,
            self.survival_ticks,
            self.score_delta,
            self.final_gauge,
        )

    @property
    def improvement_objective(self) -> tuple[int, int, int, int]:
        """Only survival and score can turn a search result into a label."""

        return (
            self.invalid_actions == 0,
            not self.terminated,
            self.survival_ticks,
            self.score_delta,
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    seed: int
    tick: int
    reference: BranchOutcome
    outcomes: tuple[BranchOutcome, ...]
    winner: int | None
    strict_improvement: bool
    label_reason: str | None
    sha256: str


@dataclass(frozen=True, slots=True)
class DistillationLabel:
    example: SteeringExample
    label_kind: str
    search_sha256: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.label_kind not in {"correction", "anchor"}:
            raise ValueError("distillation label kind must be correction or anchor")
        if len(self.search_sha256) != 64:
            raise ValueError("distillation search identity must be a SHA-256")
        if _sha(self.provenance) != self.example.provenance_sha256:
            raise ValueError("distillation label provenance identity differs")
        if self.provenance.get("schema") != "irisu-expert-iteration-label-provenance-v1":
            raise ValueError("distillation label provenance schema differs")
        if self.provenance.get("label_kind") != self.label_kind:
            raise ValueError("distillation label kind differs from provenance")
        if self.provenance.get("search_sha256") != self.search_sha256:
            raise ValueError("distillation search identity differs from provenance")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-expert-iteration-distillation-label-v1",
            "label_kind": self.label_kind,
            "label_reason": self.provenance.get("label_reason"),
            "search_sha256": self.search_sha256,
            "provenance_sha256": self.example.provenance_sha256,
            "provenance": dict(self.provenance),
            "example_sha256": self.example.sha256,
        }


def _decision_manifest(decision: SteeringDecision) -> dict[str, object]:
    return {
        "action": asdict(decision.action),
        "intent": decision.intent.value,
        "source_body_id": decision.source_body_id,
        "destination_body_id": decision.destination_body_id,
        "destination_chain_id": decision.destination_chain_id,
        "impact_x_sizes": decision.impact_x_sizes,
        "impact_y_sizes": decision.impact_y_sizes,
    }


def make_distillation_label(
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    result: SearchResult,
    *,
    label_kind: str,
    encoder: TeacherStateEncoder,
    pointer_spec: Any,
) -> DistillationLabel | None:
    provenance = {
        "schema": "irisu-expert-iteration-label-provenance-v1",
        "label_kind": label_kind,
        "search_sha256": result.sha256,
        "seed": result.seed,
        "tick": result.tick,
        "label_reason": (
            result.label_reason
            if label_kind == "correction"
            else "self_distillation_anchor"
        ),
        "decision": _decision_manifest(decision),
    }
    provenance_sha256 = _sha(provenance)
    example = steering_example_from_decision(
        observation,
        decision,
        episode_identity=(
            f"expert-iteration:{label_kind}:{result.seed:08x}:{result.tick}"
        ),
        provenance_sha256=provenance_sha256,
        encoder=encoder,
        pointer_spec=pointer_spec,
        require_representable_template=False,
    )
    if example is None:
        return None
    return DistillationLabel(example, label_kind, result.sha256, provenance)


def label_search_result(
    observation: Mapping[str, Any],
    learner_reference: SteeringDecision,
    candidates: Sequence[SteeringDecision],
    result: SearchResult,
    *,
    self_distillation_anchors: bool,
    encoder: TeacherStateEncoder,
    pointer_spec: Any,
) -> DistillationLabel | None:
    if result.winner is not None:
        return make_distillation_label(
            observation,
            candidates[result.winner],
            result,
            label_kind="correction",
            encoder=encoder,
            pointer_spec=pointer_spec,
        )
    if not self_distillation_anchors:
        return None
    return make_distillation_label(
        observation,
        learner_reference,
        result,
        label_kind="anchor",
        encoder=encoder,
        pointer_spec=pointer_spec,
    )


def balanced_training_examples(
    labels: Sequence[DistillationLabel],
) -> tuple[SteeringExample, ...]:
    """Stratify the training view across label kind and action kind."""

    groups: dict[tuple[str, bool], list[SteeringExample]] = {}
    for label in labels:
        groups.setdefault(
            (label.label_kind, label.example.is_shot), []
        ).append(label.example)
    if not groups:
        return ()
    target = max(len(values) for values in groups.values())
    output: list[SteeringExample] = []
    for key in sorted(groups):
        values = groups[key]
        output.extend(values[index % len(values)] for index in range(target))
    return tuple(output)


def correction_reason_counts(
    labels: Sequence[DistillationLabel],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for label in labels:
        if label.label_kind != "correction":
            continue
        reason = str(label.provenance.get("label_reason", "unknown"))
        output[reason] = output.get(reason, 0) + 1
    return dict(sorted(output.items()))


def configure_trainable_scope(
    model: GoalConditionedSteeringModel, scope: str
) -> tuple[str, ...]:
    """Limit sparse DAgger updates to the parameters they supervise."""

    prefixes = {
        "all": ("",),
        "heads": (
            "act_head.",
            "wait_head.",
            "pair_head.",
            "kind_head.",
            "template_head.",
            "intent_head.",
        ),
        "act-head": ("act_head.",),
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


def _step_decision(
    env: Any,
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    stop_tick: int | None = None,
) -> tuple[Mapping[str, Any], bool, bool, int]:
    current = observation
    terminated = bool(current.get("terminated", False))
    truncated = bool(current.get("truncated", False))
    invalid_actions = 0
    for action in decision.primitive_actions():
        if terminated or truncated:
            break
        if stop_tick is not None:
            remaining = stop_tick - int(current["tick"])
            if remaining <= 0:
                break
            if ActionKind.parse(action.kind) is ActionKind.WAIT:
                action = Action.wait(min(int(action.wait_ticks), remaining))
        current, _reward, terminated, truncated, info = env.step(action)
        invalid_actions += int(bool(info.get("invalid_action", False)))
    return current, terminated, truncated, invalid_actions


def _branch_outcome(
    env: Any,
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    candidate_index: int,
    continuation_policy: GoalConditionedSteeringPolicy,
    horizon_ticks: int,
) -> BranchOutcome:
    start_tick, start_score = int(observation["tick"]), int(observation.get("score", 0))
    deadline = start_tick + horizon_ticks
    current, terminated, truncated, invalid_actions = _step_decision(
        env, observation, decision, stop_tick=deadline
    )
    # The branch starts from the live controller's current cooldown/progress
    # state.  Model weights are immutable during search and can be shared.
    continuation = _copy_policy_state(continuation_policy)
    guard = 0
    while int(current["tick"]) - start_tick < horizon_ticks and not (terminated or truncated):
        current, terminated, truncated, invalid = _step_decision(
            env,
            current,
            continuation.predict(current),
            stop_tick=deadline,
        )
        invalid_actions += invalid
        guard += 1
        if guard > horizon_ticks * 2 + 64:
            raise RuntimeError("branch continuation made no bounded progress")
    return BranchOutcome(
        candidate_index,
        min(horizon_ticks, int(current["tick"]) - start_tick),
        int(current.get("score", 0)) - start_score,
        int(current.get("gauge", 0)),
        bool(terminated),
        bool(truncated),
        invalid_actions,
    )


def search_improvement(
    env: Any,
    observation: Mapping[str, Any],
    reference_decision: SteeringDecision,
    candidates: Sequence[SteeringDecision],
    *,
    continuation_policy: GoalConditionedSteeringPolicy,
    seed: int,
    horizon_ticks: int,
    gauge_correction_threshold: int = 1_000,
) -> SearchResult:
    """Transactionally compare learner action with WAIT/legal pair actions."""

    if not candidates:
        raise ValueError("snapshot search requires candidates")
    if gauge_correction_threshold < 0:
        raise ValueError("gauge correction threshold must be nonnegative")
    with TransactionalBranches(env, observation) as branches:
        with branches.branch() as (branch, restored):
            reference = _branch_outcome(
                branch, restored, reference_decision, candidate_index=-1,
                continuation_policy=continuation_policy,
                horizon_ticks=horizon_ticks,
            )
        outcomes: list[BranchOutcome] = []
        for index, candidate in enumerate(candidates):
            with branches.branch() as (branch, restored):
                outcomes.append(
                    _branch_outcome(
                        branch, restored, candidate, candidate_index=index,
                        continuation_policy=continuation_policy,
                        horizon_ticks=horizon_ticks,
                    )
                )
    best = max(outcomes, key=lambda value: (value.objective, -value.candidate_index))
    primary_improved = best.improvement_objective > reference.improvement_objective
    gauge_improved = (
        best.improvement_objective == reference.improvement_objective
        and best.final_gauge - reference.final_gauge >= gauge_correction_threshold
        and best.final_gauge > reference.final_gauge
    )
    improved = primary_improved or gauge_improved
    if not improved:
        label_reason = None
    elif gauge_improved:
        label_reason = "gauge_reserve"
    elif best.invalid_actions != reference.invalid_actions:
        label_reason = "validity"
    elif (
        best.terminated != reference.terminated
        or best.survival_ticks != reference.survival_ticks
    ):
        label_reason = "survival"
    else:
        label_reason = "score"
    manifest = {
        "schema": "irisu-expert-iteration-branch-search-v1",
        "seed": seed,
        "tick": int(observation["tick"]),
        "horizon_ticks": horizon_ticks,
        "gauge_correction_threshold": gauge_correction_threshold,
        "reference_decision": _decision_manifest(reference_decision),
        "candidate_decisions": [_decision_manifest(value) for value in candidates],
        "reference": asdict(reference),
        "outcomes": [asdict(value) for value in outcomes],
        "winner": best.candidate_index if improved else None,
        "strict_improvement": improved,
        "label_reason": label_reason,
    }
    return SearchResult(
        seed, int(observation["tick"]), reference, tuple(outcomes),
        best.candidate_index if improved else None, improved, label_reason,
        _sha(manifest),
    )


@contextmanager
def open_environment(
    backend: str, runtime: Path, maximum_ticks: int
) -> Iterator[tuple[Any, Mapping[str, Any]]]:
    if backend == "portable":
        with IrisuEnv(
            library_path=runtime,
            physics_backend="portable",
            config={"max_episode_ticks": maximum_ticks},
        ) as env:
            yield env, env.runner_identity_manifest()
        return
    exact = ExactTrainingRuntime(runtime.resolve(strict=True))
    with exact.open_env(simulation_config={"max_episode_ticks": maximum_ticks}) as session:
        yield session.environment, session.provenance_manifest


def collect_iteration(
    *,
    backend: str,
    runtime: Path,
    base_model: GoalConditionedSteeringModel,
    learner_model: GoalConditionedSteeringModel,
    base_identity: str,
    learner_identity: str,
    seeds: Sequence[int],
    episode_ticks: int,
    query_stride: int,
    maximum_queries: int,
    branch_horizon: int,
    maximum_pairs: int | None,
    base_probability: float,
    wait_ticks: int,
    random_seed: int,
    self_distillation_anchors: bool = True,
    gauge_correction_threshold: int = 1_000,
) -> tuple[list[DistillationLabel], list[dict[str, object]], list[dict[str, object]], list[Mapping[str, Any]]]:
    labels: list[DistillationLabel] = []
    searches: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    runtimes: list[Mapping[str, Any]] = []
    encoder = TeacherStateEncoder()
    for seed in seeds:
        label_start = len(labels)
        rng = random.Random((random_seed << 32) ^ seed)
        with open_environment(backend, runtime, episode_ticks + branch_horizon + 64) as (env, identity):
            runtimes.append(identity)
            observation, _info = env.reset(seed=int(seed))
            base, learner = _policy(base_model, base_identity), _policy(learner_model, learner_identity)
            base.reset(int(seed))
            learner.reset(int(seed))
            decisions = queries = base_actions = learner_actions = invalid_actions = 0
            terminated = truncated = False
            while int(observation["tick"]) < episode_ticks and not (terminated or truncated):
                execute_base = rng.random() < base_probability
                mixture = predict_mixture(
                    base, learner, observation, execute_base=execute_base
                )
                learner_decision = mixture.learner_reference
                if decisions % query_stride == 0 and queries < maximum_queries:
                    candidates = legal_candidates(
                        learner, observation, wait_ticks=wait_ticks,
                        maximum_pairs=maximum_pairs,
                    )
                    result = search_improvement(
                        env, observation, learner_decision, candidates,
                        continuation_policy=mixture.base_continuation,
                        seed=int(seed), horizon_ticks=branch_horizon,
                        gauge_correction_threshold=gauge_correction_threshold,
                    )
                    searches.append({
                        "seed": seed,
                        "tick": result.tick,
                        "sha256": result.sha256,
                        "strict_improvement": result.strict_improvement,
                        "label_reason": result.label_reason,
                        "winner": result.winner,
                        "reference": asdict(result.reference),
                        "outcomes": [asdict(value) for value in result.outcomes],
                    })
                    queries += 1
                    label = label_search_result(
                        observation,
                        learner_decision,
                        candidates,
                        result,
                        self_distillation_anchors=self_distillation_anchors,
                        encoder=encoder,
                        pointer_spec=learner.pointer_spec,
                    )
                    if label is not None:
                        labels.append(label)
                base, learner = mixture.base_policy, mixture.learner_policy
                if execute_base:
                    base_actions += 1
                else:
                    learner_actions += 1
                observation, terminated, truncated, invalid = _step_decision(
                    env,
                    observation,
                    mixture.selected,
                    stop_tick=episode_ticks,
                )
                invalid_actions += invalid
                decisions += 1
            episodes.append({
                "seed": seed,
                "final_tick": int(observation["tick"]),
                "final_score": int(observation.get("score", 0)),
                "final_gauge": int(observation.get("gauge", 0)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "decisions": decisions,
                "queries": queries,
                "corrections": sum(
                    value.label_kind == "correction"
                    for value in labels[label_start:]
                ),
                "anchors": sum(
                    value.label_kind == "anchor"
                    for value in labels[label_start:]
                ),
                "base_actions": base_actions,
                "learner_actions": learner_actions,
                "invalid_actions": invalid_actions,
            })
    return labels, searches, episodes, runtimes


def _source_identity() -> dict[str, object]:
    files = [Path(__file__).resolve(), ROOT / "python/irisu_pointer/steering_learning.py", ROOT / "python/irisu_pointer/branching.py"]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    value = {
        "git_revision": revision,
        "files": {str(path.relative_to(ROOT)): _file_sha(path) for path in files},
    }
    return {**value, "sha256": _sha(value)}


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish canonical JSON without replacing prior evidence."""

    if path.exists():
        raise FileExistsError(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and fails with
        # EEXIST instead of replacing evidence created by another process.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("output directory must not already exist")
    if _file_sha(args.base) != BASE_SHA256:
        raise ValueError("base checkpoint is not frozen-v5")
    args.output.mkdir(parents=True)
    artifact = load_steering_checkpoint(args.base, expected_sha256=BASE_SHA256)
    upstream_training_seeds = checkpoint_training_seeds(artifact.metadata)
    base_model = artifact.model
    learner_model = copy.deepcopy(base_model)
    trainable_parameters = configure_trainable_scope(
        learner_model, args.trainable_scope
    )
    all_labels: list[DistillationLabel] = []
    iterations: list[dict[str, object]] = []
    runtime_manifests: list[Mapping[str, Any]] = []
    expert_iteration_training_seeds: list[int] = []
    started = time.monotonic()
    for iteration in range(args.iterations):
        seeds = derive_seeds(f"{args.seed_namespace}:iteration-{iteration}", args.seeds_per_iteration)
        expert_iteration_training_seeds.extend(seeds)
        beta = args.initial_base_probability * (args.base_probability_decay ** iteration)
        new, searches, episodes, runtimes = collect_iteration(
            backend=args.backend,
            runtime=args.runtime,
            base_model=base_model,
            learner_model=learner_model,
            base_identity=BASE_SHA256,
            learner_identity=BASE_SHA256 if iteration == 0 else f"{iteration:064x}",
            seeds=seeds,
            episode_ticks=args.episode_ticks,
            query_stride=args.query_stride,
            maximum_queries=args.maximum_queries,
            branch_horizon=args.branch_horizon,
            maximum_pairs=None if args.maximum_pairs == 0 else args.maximum_pairs,
            base_probability=beta,
            wait_ticks=args.wait_ticks,
            random_seed=args.training_seed + iteration,
            self_distillation_anchors=args.self_distillation_anchors,
            gauge_correction_threshold=args.gauge_correction_threshold,
        )
        all_labels.extend(new)
        runtime_manifests.extend(runtimes)
        training: dict[str, object] | None = None
        training_examples = balanced_training_examples(all_labels)
        if training_examples:
            dataset = SteeringDataset(training_examples)
            training = asdict(train_goal_conditioned_steering(
                learner_model,
                dataset,
                steps=args.training_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.training_seed + iteration,
            ))
        iterations.append({
            "iteration": iteration,
            "seeds": list(seeds),
            "base_probability": beta,
            "new_labels": len(new),
            "new_corrections": sum(value.label_kind == "correction" for value in new),
            "new_anchors": sum(value.label_kind == "anchor" for value in new),
            "new_correction_reasons": correction_reason_counts(new),
            "aggregate_labels": len(all_labels),
            "aggregate_corrections": sum(
                value.label_kind == "correction" for value in all_labels
            ),
            "aggregate_anchors": sum(
                value.label_kind == "anchor" for value in all_labels
            ),
            "aggregate_correction_reasons": correction_reason_counts(all_labels),
            "balanced_training_examples": len(training_examples),
            "searches": searches,
            "episodes": episodes,
            "training": training,
        })
    if not all_labels:
        raise RuntimeError("expert iteration produced no correction or anchor labels")
    raw_dataset = SteeringDataset([value.example for value in all_labels])
    training_examples = balanced_training_examples(all_labels)
    dataset = SteeringDataset(training_examples)
    labels_path = args.output / "labels.pt"
    training_seeds = sorted(
        set(upstream_training_seeds) | set(expert_iteration_training_seeds)
    )
    torch.save({
        "schema": "irisu-expert-iteration-labels-v1",
        "base_checkpoint_sha256": BASE_SHA256,
        "upstream_training_seeds": list(upstream_training_seeds),
        "expert_iteration_training_seeds": sorted(expert_iteration_training_seeds),
        "training_seeds": training_seeds,
        "raw_dataset_manifest": raw_dataset.manifest(),
        "balanced_training_dataset_manifest": dataset.manifest(),
        "examples": [
            {
                "label": value.manifest(),
                "example": value.example.manifest(),
                "global_features": torch.from_numpy(value.example.observation.global_features),
                "body_features": torch.from_numpy(value.example.observation.body_features),
                "body_mask": torch.from_numpy(value.example.observation.body_mask),
            }
            for value in all_labels
        ],
    }, labels_path)
    labels_sha = _file_sha(labels_path)
    source = _source_identity()
    metadata = {
        "schema": "irisu-multiseed-expert-iteration-checkpoint-v1",
        "warm_start": {"path": str(args.base), "sha256": BASE_SHA256},
        "backend": args.backend,
        "runtime_path": str(args.runtime),
        "runtime_sha256": _file_sha(args.runtime),
        "raw_dataset_sha256": raw_dataset.sha256,
        "balanced_training_dataset_sha256": dataset.sha256,
        "labels_artifact_sha256": labels_sha,
        "source": source,
        "seed_namespace": args.seed_namespace,
        "held_out_seeds_used": False,
        "upstream_training_seeds": list(upstream_training_seeds),
        "expert_iteration_training_seeds": sorted(expert_iteration_training_seeds),
        "training_seeds": training_seeds,
        "inference_config": INFERENCE_CONFIG,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": list(trainable_parameters),
    }
    checkpoint = args.output / "expert-iteration.pt"
    checkpoint_sha = save_steering_checkpoint(checkpoint, learner_model, metadata=metadata)
    report = {
        "schema": "irisu-multiseed-expert-iteration-report-v1",
        "development_only": True,
        "backend": args.backend,
        "base_checkpoint_sha256": BASE_SHA256,
        "training_seeds": training_seeds,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": raw_dataset.sha256,
        "balanced_training_dataset_sha256": dataset.sha256,
        "correction_labels": sum(
            value.label_kind == "correction" for value in all_labels
        ),
        "anchor_labels": sum(
            value.label_kind == "anchor" for value in all_labels
        ),
        "correction_reasons": correction_reason_counts(all_labels),
        "labels_artifact": labels_path.name,
        "labels_artifact_sha256": labels_sha,
        "source": source,
        "runtime_manifests": runtime_manifests,
        "config": vars(args) | {"output": str(args.output), "base": str(args.base), "runtime": str(args.runtime)},
        "trainable_parameters": list(trainable_parameters),
        "iterations": iterations,
        "wall_seconds": time.monotonic() - started,
    }
    _write_json_once(args.output / "provenance.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("portable", "exact"), default="portable")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-namespace", default="irisu-high-score-expert-iteration-v1")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seeds-per-iteration", type=int, default=32)
    parser.add_argument("--episode-ticks", type=int, default=20_000)
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--maximum-queries", type=int, default=128)
    parser.add_argument("--branch-horizon", type=int, default=512)
    parser.add_argument("--maximum-pairs", type=int, default=32, help="0 evaluates every legal pair")
    parser.add_argument("--wait-ticks", type=int, default=16)
    parser.add_argument("--initial-base-probability", type=float, default=0.5)
    parser.add_argument("--base-probability-decay", type=float, default=0.5)
    parser.add_argument("--training-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--trainable-scope",
        choices=("act-head", "heads", "all"),
        default="act-head",
        help="freeze the pretrained representation for sparse DAgger labels",
    )
    parser.add_argument("--training-seed", type=int, default=2026081001)
    parser.add_argument("--gauge-correction-threshold", type=int, default=1_000)
    parser.add_argument(
        "--self-distillation-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="anchor no-improvement queries to the current learner (default: on)",
    )
    args = parser.parse_args(argv)
    args.runtime = args.runtime or (PORTABLE if args.backend == "portable" else EXACT_WORKER)
    integer_fields = ("iterations", "seeds_per_iteration", "episode_ticks", "query_stride", "maximum_queries", "branch_horizon", "wait_ticks", "training_steps", "batch_size")
    if (
        any(getattr(args, name) < 1 for name in integer_fields)
        or args.maximum_pairs < 0
        or args.gauge_correction_threshold < 0
    ):
        parser.error("counts and horizons must be positive; maximum-pairs may be zero")
    if not 0.0 <= args.initial_base_probability <= 1.0 or not 0.0 <= args.base_probability_decay <= 1.0:
        parser.error("base policy probabilities must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "checkpoint_sha256": report["checkpoint_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "labels": sum(value["new_labels"] for value in report["iterations"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
