#!/usr/bin/env python3
"""Distill exact high-score trajectory shots into the directed-pair head.

The expert trace supplies actions, not body IDs.  Each sampled shot is bound
to the unique legal analytic directed pair nearest its cursor coordinate.
Ambiguous or distant matches fail closed.  The matched pair is ranked over a
bounded set of current-model hard negatives, and only ``pair_head`` is fit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_expert_iteration_dagger as search  # noqa: E402
import rl_portable_gate_distill as gate  # noqa: E402
import rl_portable_pair_rank_distill as ranking  # noqa: E402
from irisu_env import Action  # noqa: E402
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import SteeringExample  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


WORD = struct.Struct("<I")
TRACE_MANIFEST_SCHEMA = "irisu-exact-pair-trace-training-manifest-v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace_manifest(path: Path) -> tuple[dict[str, object], ...]:
    """Load caller-owned trace inventory and bind its source evidence."""

    resolved = path.resolve(strict=True)
    raw = json.loads(resolved.read_text())
    if not isinstance(raw, dict) or set(raw) != {"schema", "traces"}:
        raise ValueError("trace manifest has an unexpected schema")
    traces = raw["traces"]
    if raw["schema"] != TRACE_MANIFEST_SCHEMA or not isinstance(traces, list) or not traces:
        raise ValueError("trace manifest must contain a nonempty trace list")
    required = {
        "label",
        "seed",
        "trace",
        "trace_sha256",
        "source_metadata",
        "source_metadata_sha256",
        "expected_score",
        "expected_ticks",
    }
    entries: list[dict[str, object]] = []
    seen_seeds: set[int] = set()
    seen_traces: set[str] = set()
    for index, value in enumerate(traces):
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"trace manifest entry {index} has an unexpected schema")
        seed = value["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFF_FFFF
            or seed in seen_seeds
        ):
            raise ValueError("trace manifest seeds must be unique uint32 values")
        trace = (resolved.parent / str(value["trace"])).resolve(strict=True)
        source = (resolved.parent / str(value["source_metadata"])).resolve(strict=True)
        trace_sha = str(value["trace_sha256"])
        source_sha = str(value["source_metadata_sha256"])
        if trace_sha in seen_traces or _file_sha256(trace) != trace_sha:
            raise RuntimeError("trace manifest trace identity mismatch or duplicate")
        if _file_sha256(source) != source_sha:
            raise RuntimeError("trace manifest source metadata SHA-256 mismatch")
        source_value = json.loads(source.read_text())
        episode = source_value.get("episode") if isinstance(source_value, dict) else None
        if not isinstance(episode, dict):
            raise ValueError("trace source metadata lacks an episode")
        expected_score = value["expected_score"]
        expected_ticks = value["expected_ticks"]
        if (
            episode.get("seed") != seed
            or episode.get("score") != expected_score
            or episode.get("tick") != expected_ticks
            or trace.stat().st_size != int(expected_ticks) * WORD.size
        ):
            raise RuntimeError("trace manifest expected episode does not match source evidence")
        entries.append(
            {
                **value,
                "trace": trace,
                "source_metadata": source,
                "source_metadata_schema": source_value.get("schema"),
            }
        )
        seen_seeds.add(seed)
        seen_traces.add(trace_sha)
    return tuple(entries)


def trace_balanced(
    groups: Sequence[Sequence[Any]], *, seed: int
) -> tuple[Any, ...]:
    """Deterministically give each trace exactly equal dataset weight."""

    if not groups or any(not group for group in groups):
        raise ValueError("trace balancing requires nonempty groups")
    count = min(len(group) for group in groups)
    selected: list[list[Any]] = []
    for ordinal, group in enumerate(groups):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + ordinal)
        indices = torch.randperm(len(group), generator=generator)[:count].tolist()
        selected.append([group[index] for index in indices])
    return tuple(
        selected[trace_index][item_index]
        for item_index in range(count)
        for trace_index in range(len(selected))
    )


def normalize_public_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Normalize exact NumPy identifiers for shared public-policy helpers."""

    normalized = dict(observation)
    normalized["bodies"] = tuple(
        {
            **dict(body),
            "id": int(body["id"]),
            "color": int(body["color"]),
            "chain_id": int(body["chain_id"]),
            "projectile_hits": int(body["projectile_hits"]),
            "age_ticks": int(body["age_ticks"]),
            "remaining_lifetime": int(body["remaining_lifetime"]),
            "rot_timer": int(body["rot_timer"]),
        }
        for body in observation.get("bodies", ())
    )
    return normalized


def decode_action(word: int) -> Action:
    buttons = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if buttons == 0:
        if word != 0:
            raise ValueError("WAIT replay word carries cursor state")
        return Action.wait(1)
    if buttons == 1:
        return Action.weak(x, y)
    if buttons == 2:
        return Action.strong(x, y)
    return Action.both(x, y)


def bind_shot_to_pair(
    candidates: tuple[Any, ...],
    *,
    cursor_x: int,
    cursor_y: int,
    client_width: int,
    client_height: int,
    maximum_error_pixels: float,
    ambiguity_gap_pixels: float,
) -> tuple[Any | None, dict[str, object]]:
    ranked: list[tuple[float, Any]] = []
    for candidate in candidates:
        if not candidate.is_shot:
            continue
        x = min(math.floor(candidate.action.x_norm * client_width), client_width - 1)
        y = min(math.floor(candidate.action.y_norm * client_height), client_height - 1)
        ranked.append((math.hypot(x - cursor_x, y - cursor_y), candidate))
    ranked.sort(
        key=lambda value: (
            value[0],
            value[1].source_body_id,
            value[1].destination_body_id,
        )
    )
    manifest: dict[str, object] = {
        "candidate_count": len(ranked),
        "best_error_pixels": ranked[0][0] if ranked else None,
        "second_error_pixels": ranked[1][0] if len(ranked) > 1 else None,
    }
    if not ranked or ranked[0][0] > maximum_error_pixels:
        return None, {**manifest, "rejection": "coordinate-error"}
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < ambiguity_gap_pixels:
        return None, {**manifest, "rejection": "ambiguous-coordinate"}
    return ranked[0][1], {**manifest, "rejection": None}


def collect(
    *,
    worker: Path,
    trace: Path,
    expected_trace_sha256: str,
    seed: int,
    model: Any,
    model_sha256: str,
    shot_stride: int,
    maximum_examples: int,
    maximum_tick: int | None,
    hard_negatives: int,
    maximum_error_pixels: float,
    ambiguity_gap_pixels: float,
) -> tuple[list[SteeringExample], list[ranking.PairPreference], dict[str, object]]:
    data = trace.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_trace_sha256:
        raise RuntimeError("exact expert trace SHA-256 mismatch")
    if len(data) % WORD.size:
        raise ValueError("exact expert trace is not packed uint32 words")
    words = [value[0] for value in WORD.iter_unpack(data)]
    policy = gate._make_policy(model, model_sha256, 1.0)
    policy.reset(seed)
    labels: list[SteeringExample] = []
    preferences: list[ranking.PairPreference] = []
    rejection_counts: dict[str, int] = {}
    accepted_errors: list[float] = []
    accepted_gaps: list[float] = []
    shot_ordinal = accepted = 0
    runtime = ExactTrainingRuntime(worker)
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": len(words) + 1}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("exact expert reset returned a different seed")
        terminated = truncated = False
        for tick, word in enumerate(words):
            if maximum_tick is not None and tick >= maximum_tick:
                break
            buttons = word & 3
            if buttons in (1, 2):
                shot_ordinal += 1
                if shot_ordinal % shot_stride == 0 and accepted < maximum_examples:
                    sample_observation = normalize_public_observation(observation)
                    candidates = search.legal_candidates(
                        policy.inner,
                        sample_observation,
                        wait_ticks=16,
                        maximum_pairs=None,
                    )[1:]
                    expert_x, expert_y = (word >> 2) & 1023, (word >> 12) & 511
                    positive, evidence = bind_shot_to_pair(
                        candidates,
                        cursor_x=expert_x,
                        cursor_y=expert_y,
                        client_width=int(policy.action_spec.client_width),
                        client_height=int(policy.action_spec.client_height),
                        maximum_error_pixels=maximum_error_pixels,
                        ambiguity_gap_pixels=ambiguity_gap_pixels,
                    )
                    rejection = evidence["rejection"]
                    if positive is None:
                        key = str(rejection)
                        rejection_counts[key] = rejection_counts.get(key, 0) + 1
                    else:
                        provenance = {
                            "schema": "irisu-exact-trace-pair-binding-v1",
                            "seed": seed,
                            "tick": tick,
                            "shot_ordinal": shot_ordinal,
                            "trace_sha256": expected_trace_sha256,
                            "cursor": [expert_x, expert_y],
                            "binding": evidence,
                            "positive_pair": [
                                positive.source_body_id,
                                positive.destination_body_id,
                            ],
                        }
                        example = ranking.steering_example_from_decision(
                            sample_observation,
                            positive,
                            episode_identity="exact-pair-trace",
                            provenance_sha256=gate._sha(provenance),
                            encoder=policy.inner.encoder,
                            pointer_spec=policy.pointer_spec,
                            require_representable_template=False,
                        )
                        if example is None:
                            raise RuntimeError("bound exact pair is not representable")
                        labels.append(example)
                        accepted_errors.append(float(evidence["best_error_pixels"]))
                        second_error = evidence["second_error_pixels"]
                        if second_error is not None:
                            accepted_gaps.append(
                                float(second_error) - float(evidence["best_error_pixels"])
                            )
                        negatives = [
                            candidate
                            for candidate in candidates
                            if ranking._decision_key(candidate)
                            != ranking._decision_key(positive)
                        ][:hard_negatives]
                        for ordinal, negative in enumerate(negatives):
                            preferences.append(
                                ranking._preference(
                                    sample_observation,
                                    positive,
                                    negative,
                                    {**provenance, "hard_negative_ordinal": ordinal},
                                    encoder=policy.inner.encoder,
                                    pointer_spec=policy.pointer_spec,
                                )
                            )
                        accepted += 1
            observation, _reward, terminated, truncated, _info = env.step(
                decode_action(word)
            )
            if terminated or truncated:
                break
        runtime_manifest = session.provenance_manifest
    def distribution(values: list[float]) -> dict[str, float | int | None]:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "minimum": ordered[0] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "maximum": ordered[-1] if ordered else None,
        }

    sampled = accepted + sum(rejection_counts.values())
    return labels, preferences, {
        "seed": seed,
        "trace_words": len(words),
        "replayed_ticks": int(observation["tick"]),
        "final_score": int(observation.get("score", 0)),
        "shots_seen": shot_ordinal,
        "accepted_pair_labels": len(labels),
        "sampled_shots": sampled,
        "mapping_yield": len(labels) / sampled if sampled else 0.0,
        "accepted_residual_pixels": distribution(accepted_errors),
        "accepted_ambiguity_gap_pixels": distribution(accepted_gaps),
        "pair_preferences": len(preferences),
        "rejections": dict(sorted(rejection_counts.items())),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "exact_runtime": runtime_manifest,
        "wall_seconds": time.monotonic() - started,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    artifact = load_steering_checkpoint(
        args.base_checkpoint, expected_sha256=args.base_sha256
    )
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    trace_manifest: dict[str, object] | None = None
    if args.trace_manifest is not None:
        manifest_path = args.trace_manifest.resolve(strict=True)
        entries = load_trace_manifest(manifest_path)
        trace_manifest = {
            "path": str(manifest_path),
            "sha256": _file_sha256(manifest_path),
            "schema": TRACE_MANIFEST_SCHEMA,
        }
    else:
        if args.trace is None or args.trace_sha256 is None or args.seed is None:
            raise ValueError("single-trace mode requires trace, trace SHA-256, and seed")
        trace = args.trace.resolve(strict=True)
        entries = (
            {
                "label": "single-trace",
                "seed": args.seed,
                "trace": trace,
                "trace_sha256": args.trace_sha256,
                "source_metadata": None,
                "source_metadata_sha256": None,
                "source_metadata_schema": None,
                "expected_score": None,
                "expected_ticks": len(trace.read_bytes()) // WORD.size,
            },
        )
    pair_trace_seeds = tuple(int(value["seed"]) for value in entries)
    if any(seed in upstream for seed in pair_trace_seeds):
        raise ValueError("exact expert seed is already in warm-start lineage")
    model = artifact.model
    trainable = ranking.configure_trainable_scope(model, "pair-head")
    before = copy.deepcopy(model.state_dict())
    label_groups: list[list[SteeringExample]] = []
    preference_groups: list[list[ranking.PairPreference]] = []
    collections: list[dict[str, object]] = []
    worker = args.worker.resolve(strict=True)
    rollout_model_sha256 = gate._model_state_sha(model)
    for entry in entries:
        labels, preferences, collection = collect(
            worker=worker,
            trace=entry["trace"],
            expected_trace_sha256=str(entry["trace_sha256"]),
            seed=int(entry["seed"]),
            model=model,
            model_sha256=rollout_model_sha256,
            shot_stride=args.shot_stride,
            maximum_examples=args.maximum_examples,
            maximum_tick=args.maximum_tick,
            hard_negatives=args.hard_negatives,
            maximum_error_pixels=args.maximum_error_pixels,
            ambiguity_gap_pixels=args.ambiguity_gap_pixels,
        )
        if entry["expected_score"] is not None and (
            collection["final_score"] != entry["expected_score"]
            or collection["replayed_ticks"] != entry["expected_ticks"]
        ):
            raise RuntimeError("exact replay result differs from trace manifest evidence")
        label_groups.append(labels)
        preference_groups.append(preferences)
        collections.append(collection)
    labels = list(trace_balanced(label_groups, seed=args.training_seed + 10_000))
    preferences = list(
        trace_balanced(preference_groups, seed=args.training_seed + 20_000)
    )
    balanced_labels_per_trace = len(labels) // len(entries)
    balanced_preferences_per_trace = len(preferences) // len(entries)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.training_seed)
        training = ranking.train_ranked_heads(
            model,
            labels,
            (),
            preferences,
            steps=args.training_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            causal_weight=0.0,
            ranking_weight=1.0,
            seed=args.training_seed,
        )
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(before[name], value)
    ]
    if not changed or any(not name.startswith("pair_head.") for name in changed):
        raise RuntimeError("pair trace distillation changed an unexpected parameter set")
    args.output.mkdir(parents=True)
    training_seeds = sorted((*upstream, *pair_trace_seeds))
    expert_traces = [
        {
            "label": entry["label"],
            "seed": entry["seed"],
            "path": str(entry["trace"]),
            "sha256": entry["trace_sha256"],
            "bytes": entry["trace"].stat().st_size,
            "source_metadata": None
            if entry["source_metadata"] is None
            else {
                "path": str(entry["source_metadata"]),
                "sha256": entry["source_metadata_sha256"],
                "schema": entry["source_metadata_schema"],
            },
            "expected_score": entry["expected_score"],
            "expected_ticks": entry["expected_ticks"],
        }
        for entry in entries
    ]
    metadata = {
        "schema": "irisu-exact-pair-trace-distillation-v2",
        "development_only": True,
        "promotion_eligible": False,
        "base_checkpoint_sha256": artifact.sha256,
        "warm_start_metadata_sha256": gate._sha(dict(artifact.metadata)),
        "upstream_training_seeds": list(upstream),
        "pair_trace_training_seeds": list(pair_trace_seeds),
        "training_seeds": training_seeds,
        "inference_config": gate.inference_config(1.0),
        "trace_manifest": trace_manifest,
        "expert_traces": expert_traces,
        "source": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": gate._file_sha(Path(__file__).resolve()),
            "pair_trainer_sha256": gate._file_sha(Path(ranking.__file__).resolve()),
            "steering_learning_sha256": gate._file_sha(
                ROOT / "python/irisu_pointer/steering_learning.py"
            ),
        },
        "collections": [
            {k: v for k, v in collection.items() if k != "wall_seconds"}
            for collection in collections
        ],
        "trace_balancing": {
            "schema": "irisu-deterministic-equal-trace-weight-v1",
            "seed": args.training_seed,
            "labels_per_trace": balanced_labels_per_trace,
            "preferences_per_trace": balanced_preferences_per_trace,
            "total_labels": len(labels),
            "total_preferences": len(preferences),
            "interleaving": "round-robin-after-per-trace-seeded-permutation",
        },
        "trainable_scope": "pair-head",
        "trainable_parameters": list(trainable),
        "changed_parameters": changed,
        "training_report": asdict(training),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    }
    checkpoint = args.output / "exact-pair-distilled.pt"
    checkpoint_sha256 = save_steering_checkpoint(checkpoint, model, metadata=metadata)
    labels_path = args.output / "pair-preferences.pt"
    gate._save_torch_new(
        labels_path,
        {
            "schema": "irisu-exact-pair-trace-preferences-v2",
            "metadata": metadata,
            "preferences": [
                {
                    "manifest": value.manifest(),
                    "global_features": torch.from_numpy(
                        value.observation.global_features
                    ),
                    "body_features": torch.from_numpy(
                        value.observation.body_features
                    ),
                    "body_mask": torch.from_numpy(value.observation.body_mask),
                }
                for value in preferences
            ],
        },
    )
    report = {
        **metadata,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "labels_artifact": labels_path.name,
        "labels_artifact_sha256": gate._file_sha(labels_path),
        "operational": {
            "collection_wall_seconds": [
                collection["wall_seconds"] for collection in collections
            ]
        },
    }
    report["sha256"] = gate._sha(report)
    gate._write_json_new(args.output / "provenance.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", type=Path, required=True)
    result.add_argument("--base-checkpoint", type=Path, required=True)
    result.add_argument("--base-sha256", required=True)
    traces = result.add_mutually_exclusive_group(required=True)
    traces.add_argument("--trace", type=Path)
    traces.add_argument("--trace-manifest", type=Path)
    result.add_argument("--trace-sha256")
    result.add_argument("--seed", type=int)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--shot-stride", type=int, default=4)
    result.add_argument("--maximum-examples", type=int, default=1_250)
    result.add_argument("--maximum-tick", type=int)
    result.add_argument("--hard-negatives", type=int, default=3)
    result.add_argument("--maximum-error-pixels", type=float, default=64.0)
    result.add_argument("--ambiguity-gap-pixels", type=float, default=4.0)
    result.add_argument("--training-steps", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=5e-5)
    result.add_argument("--training-seed", type=int, default=2026081201)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = parser().parse_args(argv)
    if args.trace is not None and (args.trace_sha256 is None or args.seed is None):
        parser().error("--trace requires --trace-sha256 and --seed")
    if args.trace_manifest is not None and (
        args.trace_sha256 is not None or args.seed is not None
    ):
        parser().error("--trace-manifest cannot be combined with single-trace identity")
    for name in (
        "shot_stride",
        "maximum_examples",
        "hard_negatives",
        "training_steps",
        "batch_size",
    ):
        if getattr(args, name) < 1:
            parser().error(f"--{name.replace('_', '-')} must be positive")
    if args.maximum_tick is not None and args.maximum_tick < 1:
        parser().error("--maximum-tick must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
