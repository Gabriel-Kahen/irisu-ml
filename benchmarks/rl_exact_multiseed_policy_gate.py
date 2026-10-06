#!/usr/bin/env python3
"""Evaluate one learned steering checkpoint on a locked fresh-seed exact gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from irisu_env import Action, ActionKind
from irisu_pointer.steering_checkpoint import load_steering_checkpoint
from irisu_pointer.steering_learning import GoalConditionedSteeringPolicy
from irisu_rl.exact_training_runtime import ExactTrainingRuntime


CALIBRATION_SEEDS = (
    1_439_993_096,
    1_363_400_495,
    1_555_243_350,
    1_478_650_749,
    1_402_058_148,
    1_593_901_003,
    1_517_308_402,
    1_440_715_801,
    1_364_123_200,
    1_555_966_055,
    1_479_373_454,
    1_402_780_853,
    1_594_623_708,
    1_518_031_107,
    1_441_438_506,
    1_364_845_905,
    1_556_688_760,
    1_480_096_159,
    1_403_503_558,
    1_595_346_413,
)
TARGET_SCORE = 50_000
DEFAULT_REQUIRED_SUCCESS_FRACTION = 0.8
FORMAT = "irisu-exact-multiseed-learned-policy-gate-v1"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_manifest(declared_training_seeds: Sequence[int]) -> dict[str, object]:
    training = tuple(declared_training_seeds)
    for label, values in (
        ("training", training),
        ("calibration", CALIBRATION_SEEDS),
    ):
        if (
            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values)
            or any(not 0 <= seed <= 0xFFFFFFFF for seed in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{label} seeds must be unique uint32 values")
    overlap = sorted(set(training) & set(CALIBRATION_SEEDS))
    if overlap:
        raise ValueError(f"calibration seeds overlap declared training seeds: {overlap}")
    manifest: dict[str, object] = {
        "version": "irisu-exact-50k-calibration-seeds-v1",
        "training_seeds": sorted(training),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "overlap": [],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def load_declared_training_seeds(path: Path) -> tuple[int, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        seeds = value
    elif isinstance(value, dict) and isinstance(value.get("training_seeds"), list):
        seeds = value["training_seeds"]
    else:
        raise ValueError(
            "training seed manifest must be a JSON list or contain training_seeds"
        )
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("declared training seeds must be integers")
    return tuple(seeds)


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0" + value.numpy().tobytes())
    return digest.hexdigest()


class Policy(Protocol):
    def reset(self, seed: int = 0) -> None: ...

    def predict(self, observation: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    format: str
    checkpoint_sha256: str
    model_sha256: str
    metadata: Mapping[str, Any]
    factory: Callable[[], Policy]
    inference_config: Mapping[str, Any]


def checkpoint_training_seeds(bundle: PolicyBundle) -> tuple[int, ...]:
    raw = bundle.metadata.get("training_seeds")
    if not isinstance(raw, list):
        raise ValueError("checkpoint must declare its complete training_seeds")
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw)
        or any(not 0 <= seed <= 0xFFFFFFFF for seed in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("checkpoint training_seeds must be unique uint32 values")
    return tuple(sorted(raw))


CheckpointLoader = Callable[[Path, str | None, Mapping[str, Any]], PolicyBundle]
_CHECKPOINT_LOADERS: dict[str, CheckpointLoader] = {}


def register_checkpoint_loader(name: str, loader: CheckpointLoader) -> None:
    if not name or name in _CHECKPOINT_LOADERS:
        raise ValueError(f"checkpoint loader is already registered: {name!r}")
    _CHECKPOINT_LOADERS[name] = loader


def _load_goal_conditioned_steering(
    path: Path,
    expected_sha256: str | None,
    options: Mapping[str, Any],
) -> PolicyBundle:
    checkpoint = load_steering_checkpoint(
        path, expected_sha256=expected_sha256, device="cpu"
    )
    checkpoint.model.eval()
    recorded = checkpoint.metadata.get("inference_config", {})
    if not isinstance(recorded, dict):
        raise ValueError("checkpoint inference_config must be a mapping")
    defaults = {
        "cooldown_ticks": 16,
        "minimum_pair_closure_sizes": 0.05,
        "impact_side_sizes": 0.5,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 50.0,
        "act_logit_bias": 0.0,
    }
    for name, value in options.items():
        if name in recorded and value != recorded[name]:
            raise ValueError(
                f"inference option {name} differs from checkpoint metadata"
            )
    resolved = defaults | recorded | dict(options)
    config = {
        "cooldown_ticks": int(resolved["cooldown_ticks"]),
        "minimum_pair_closure_sizes": float(
            resolved["minimum_pair_closure_sizes"]
        ),
        "impact_side_sizes": float(resolved["impact_side_sizes"]),
        "impact_below_sizes": float(resolved["impact_below_sizes"]),
        "source_velocity_lead_ticks": float(
            resolved["source_velocity_lead_ticks"]
        ),
        "ticks_per_second": float(resolved["ticks_per_second"]),
        "act_logit_bias": float(resolved["act_logit_bias"]),
    }

    def factory() -> GoalConditionedSteeringPolicy:
        return GoalConditionedSteeringPolicy(
            checkpoint.model,
            **config,
            artifact_sha256=checkpoint.sha256,
        )

    return PolicyBundle(
        format="irisu-goal-conditioned-steering-checkpoint-v1",
        checkpoint_sha256=checkpoint.sha256,
        model_sha256=state_dict_sha256(checkpoint.model),
        metadata=dict(checkpoint.metadata),
        factory=factory,
        inference_config=config,
    )


register_checkpoint_loader(
    "goal-conditioned-steering-v1", _load_goal_conditioned_steering
)


def load_policy_bundle(
    path: Path,
    *,
    expected_sha256: str | None = None,
    checkpoint_loader: str = "goal-conditioned-steering-v1",
    options: Mapping[str, Any] | None = None,
) -> PolicyBundle:
    try:
        loader = _CHECKPOINT_LOADERS[checkpoint_loader]
    except KeyError as exc:
        raise ValueError(f"unknown checkpoint loader: {checkpoint_loader}") from exc
    return loader(path.resolve(strict=True), expected_sha256, options or {})


def _observation_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    convert = getattr(value, "to_dict", None)
    if not callable(convert):
        raise TypeError("vector observation is not convertible to a public mapping")
    result = convert()
    if not isinstance(result, dict):
        raise TypeError("vector observation conversion did not return a mapping")
    return result


def _effective_score(observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
    diagnostics = info.get("diagnostics")
    if (
        diagnostics is not None
        and bool(getattr(diagnostics, "terminal_metadata_recorded", False))
    ):
        return int(getattr(diagnostics, "recorded_final_score"))
    return int(observation["score"])


def _primitive_actions(decision: Any) -> tuple[Any, ...]:
    method = getattr(decision, "primitive_actions", None)
    if not callable(method):
        raise TypeError("policy decision does not expose primitive_actions")
    actions = tuple(method())
    if not actions:
        raise ValueError("policy decision produced no primitive actions")
    return actions


def _validated_action(value: Any, remaining_ticks: int) -> Action:
    kind = ActionKind.parse(getattr(value, "kind"))
    x = float(getattr(value, "cursor_x"))
    y = float(getattr(value, "cursor_y"))
    wait_ticks = int(getattr(value, "wait_ticks"))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("policy cursor coordinates must be finite")
    if kind == ActionKind.WAIT:
        if wait_ticks < 1:
            raise ValueError("policy wait duration must be positive")
        return Action.wait(min(wait_ticks, remaining_ticks))
    if wait_ticks != 1:
        raise ValueError("shot actions must span exactly one tick")
    return Action(kind, x, y, 1)


@dataclass(slots=True)
class _Lane:
    seed: int
    policy: Policy
    observation: dict[str, Any]
    pending: deque[Any]
    policy_invalid_actions: int = 0
    simulator_invalid_actions: int = 0
    action_count: int = 0


def evaluate_bundle(
    runtime: ExactTrainingRuntime,
    bundle: PolicyBundle,
    *,
    declared_training_seeds: Sequence[int],
    lanes: int = 4,
    workers: int | None = None,
    maximum_ticks: int = 120_000,
    target_score: int = TARGET_SCORE,
    required_success_fraction: float = DEFAULT_REQUIRED_SUCCESS_FRACTION,
) -> dict[str, object]:
    declared = tuple(sorted(declared_training_seeds))
    if declared != checkpoint_training_seeds(bundle):
        raise ValueError(
            "declared training seeds differ from checkpoint-bound training seeds"
        )
    seeds = seed_manifest(declared)
    if not 1 <= lanes <= len(CALIBRATION_SEEDS):
        raise ValueError("lanes must be between one and the calibration seed count")
    if maximum_ticks < 1 or target_score < 0:
        raise ValueError("maximum_ticks must be positive and target_score nonnegative")
    if not 0.0 <= required_success_fraction <= 1.0:
        raise ValueError("required_success_fraction must lie in [0, 1]")

    started = time.monotonic()
    results: list[dict[str, object]] = []
    initial = list(CALIBRATION_SEEDS[:lanes])
    next_seed = lanes
    simulation_config = {"max_episode_ticks": maximum_ticks}
    with runtime.open_vector(
        lanes, simulation_config=simulation_config, workers=workers
    ) as session:
        vector = session.environment
        observations, infos = vector.reset(seed=initial)
        if [int(info["seed"]) for info in infos] != initial:
            raise RuntimeError("exact vector reset returned different seeds")
        active: dict[int, _Lane] = {}
        for lane, (seed, observation) in enumerate(zip(initial, observations)):
            policy = bundle.factory()
            policy.reset(seed)
            active[lane] = _Lane(
                seed, policy, _observation_dict(observation), deque()
            )
        provenance = session.provenance_manifest

        while active:
            indices: list[int] = []
            actions: list[Action] = []
            for lane in sorted(active):
                state = active[lane]
                remaining = maximum_ticks - int(state.observation["tick"])
                if remaining <= 0:
                    raise RuntimeError("completed lane was not retired")
                try:
                    if not state.pending:
                        state.pending.extend(
                            _primitive_actions(state.policy.predict(state.observation))
                        )
                    action = _validated_action(state.pending.popleft(), remaining)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    state.policy_invalid_actions += 1
                    state.pending.clear()
                    action = Action.wait(1)
                indices.append(lane)
                actions.append(action)
            stepped = vector.step_many(indices, actions)
            observations, _rewards, terminated, truncated, infos = stepped
            completed: list[int] = []
            for lane, observation, ended, cut, info in zip(
                indices, observations, terminated, truncated, infos
            ):
                state = active[lane]
                state.action_count += 1
                state.simulator_invalid_actions += int(bool(info["invalid_action"]))
                state.observation = _observation_dict(observation)
                at_limit = int(state.observation["tick"]) >= maximum_ticks
                if ended or cut or at_limit:
                    score = _effective_score(state.observation, info)
                    invalid = (
                        state.policy_invalid_actions
                        + state.simulator_invalid_actions
                    )
                    results.append(
                        {
                            "seed": state.seed,
                            "score": score,
                            "success": score >= target_score,
                            "tick": int(state.observation["tick"]),
                            "terminated": bool(ended),
                            "truncated": bool(cut),
                            "censored": not bool(ended),
                            "policy_invalid_actions": state.policy_invalid_actions,
                            "simulator_invalid_actions": state.simulator_invalid_actions,
                            "invalid_actions": invalid,
                            "action_count": state.action_count,
                        }
                    )
                    completed.append(lane)

            replacements = min(len(completed), len(CALIBRATION_SEEDS) - next_seed)
            reset_lanes = completed[:replacements]
            reset_seeds = list(
                CALIBRATION_SEEDS[next_seed : next_seed + replacements]
            )
            next_seed += replacements
            if reset_lanes:
                reset_observations = vector.reset_many(reset_lanes, seeds=reset_seeds)
                for lane, seed, observation in zip(
                    reset_lanes, reset_seeds, reset_observations
                ):
                    policy = bundle.factory()
                    policy.reset(seed)
                    active[lane] = _Lane(
                        seed, policy, _observation_dict(observation), deque()
                    )
            for lane in completed[replacements:]:
                del active[lane]

    results.sort(key=lambda item: CALIBRATION_SEEDS.index(int(item["seed"])))
    scores = [int(item["score"]) for item in results]
    successes = sum(bool(item["success"]) for item in results)
    success_fraction = successes / len(results)
    median_score = float(statistics.median(scores))
    runner_path = Path(__file__).resolve()
    runtime_hashes = {
        "worker_sha256": runtime.identity.worker_sha256,
        "exact_library_sha256": runtime.identity.exact_library_sha256,
        "identity_config_sha256": runtime.identity.config_sha256,
        "runtime_attestation_sha256": provenance["runtime_attestation_sha256"],
        "runner_identity_sha256": canonical_sha256(provenance["runner_identity"]),
    }
    return {
        "format": FORMAT,
        "physics_backend": "exact",
        "seed_manifest": seeds,
        "target_score": target_score,
        "required_success_fraction": required_success_fraction,
        "maximum_ticks": maximum_ticks,
        "lanes": lanes,
        "workers": workers,
        "checkpoint_loader": bundle.format,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_sha256": bundle.model_sha256,
        "checkpoint_metadata_sha256": canonical_sha256(bundle.metadata),
        "inference_config": dict(bundle.inference_config),
        "runtime_hashes": runtime_hashes,
        "runner_sha256": file_sha256(runner_path),
        "episodes": results,
        "median_score": median_score,
        "success_count": successes,
        "success_fraction_at_or_above_50k": success_fraction,
        "invalid_actions": sum(int(item["invalid_actions"]) for item in results),
        "passed": median_score >= target_score
        and success_fraction >= required_success_fraction
        and not any(int(item["invalid_actions"]) for item in results),
        "wall_seconds": time.monotonic() - started,
    }


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-loader", default="goal-conditioned-steering-v1")
    parser.add_argument("--training-seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--maximum-ticks", type=int, default=120_000)
    parser.add_argument("--required-success-fraction", type=float, default=0.8)
    parser.add_argument("--act-logit-bias", type=float)
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be absolute")
    if args.output.exists():
        parser.error("--output must not already exist")
    training = load_declared_training_seeds(
        args.training_seed_manifest.resolve(strict=True)
    )
    bundle = load_policy_bundle(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        checkpoint_loader=args.checkpoint_loader,
        options=(
            {} if args.act_logit_bias is None
            else {"act_logit_bias": args.act_logit_bias}
        ),
    )
    report = evaluate_bundle(
        ExactTrainingRuntime(args.worker),
        bundle,
        declared_training_seeds=training,
        lanes=args.lanes,
        workers=args.workers,
        maximum_ticks=args.maximum_ticks,
        required_success_fraction=args.required_success_fraction,
    )
    _write_new(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
