#!/usr/bin/env python3
"""Exact on-policy DAgger for late/low-reserve directed-pair disagreements."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_portable_gate_distill as gate  # noqa: E402
import rl_portable_pair_rank_distill as ranking  # noqa: E402
from irisu_env import Action, ActionKind, ExactWorkerError  # noqa: E402
from irisu_pointer.fast_multiaction_planner import (  # noqa: E402
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
    MultiActionVerdict,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent  # noqa: E402
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_rl.actions import SemanticAction  # noqa: E402
from irisu_rl.encoding import EncodedBatch  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402
from irisu_rl.seeds import SeedAllocator  # noqa: E402


SEED_PLAN_SCHEMA = "irisu-onpolicy-pair-dagger-seed-plan-v1"
TRAINABLE_PREFIXES = ("pair_head.",)


def primitive_actions(decision: object) -> tuple[object, ...]:
    actions = tuple(decision.primitive_actions())
    if not actions:
        raise ValueError("empty decision macro")
    return actions


def load_seed_plan(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text())
    required = {
        "schema",
        "namespace",
        "namespace_sha256",
        "allocator_key",
        "allocator_manifest_sha256",
        "split",
        "cursor",
        "count",
        "seeds",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("seed plan has an unexpected schema")
    namespace = value["namespace"]
    digest = hashlib.sha256(str(namespace).encode()).hexdigest()
    key = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    allocator = SeedAllocator(str(value["split"]), key=key, cursor=int(value["cursor"]))
    expected = allocator.take(int(value["count"]))
    if (
        value["schema"] != SEED_PLAN_SCHEMA
        or value["namespace_sha256"] != digest
        or value["allocator_key"] != key
        or value["allocator_manifest_sha256"] != allocator.manifest_sha256
        or list(expected) != value["seeds"]
        or any(not 0 <= seed < 1 << 30 for seed in expected)
    ):
        raise RuntimeError("seed plan does not match the canonical TRAIN allocator")
    return {**value, "path": str(resolved), "sha256": gate._file_sha(resolved)}


def configure_trainable(model: torch.nn.Module) -> tuple[str, ...]:
    selected = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    expected = {"pair_head.weight", "pair_head.bias"}
    if set(selected) != expected:
        raise RuntimeError("unexpected pair DAgger trainable parameter set")
    return tuple(selected)


def incumbent_choice(
    planner: FastMultiActionPlanner, verdict: MultiActionVerdict
) -> CandidateOutcome:
    wait = next(value for value in verdict.outcomes if value.candidate.category == "wait")
    eligible: list[CandidateOutcome] = []
    for value in verdict.outcomes:
        if not value.candidate.category.startswith("predicted-pair-"):
            continue
        allowed, _reason = planner._eligible(value, wait)
        if allowed:
            eligible.append(value)
    return max(eligible, key=planner._objective) if eligible else wait


def safe_alternate_preference(
    planner: FastMultiActionPlanner,
    verdict: MultiActionVerdict,
    *,
    branch_complete: bool = True,
    parent_unchanged: bool = True,
) -> tuple[CandidateOutcome, CandidateOutcome] | None:
    """Return one strictly better alternate without trading survival."""

    if not branch_complete or not parent_unchanged or not verdict.used_fast_checkpoint:
        return None
    teacher = next(
        value
        for value in verdict.outcomes
        if value.candidate.ordinal == verdict.selected.ordinal
    )
    if not teacher.candidate.category.startswith("top-"):
        return None
    wait = next(value for value in verdict.outcomes if value.candidate.category == "wait")
    incumbent = incumbent_choice(planner, verdict)
    if incumbent is wait:
        return None
    if (
        teacher.probe.survival_ticks < incumbent.probe.survival_ticks
        or teacher.probe.survival_ticks < wait.probe.survival_ticks
        or planner._objective(teacher) <= planner._objective(incumbent)
    ):
        return None
    return teacher, incumbent


def _validated_action(value: object, remaining: int) -> Action:
    kind = ActionKind.parse(getattr(value, "kind"))
    x, y = float(getattr(value, "cursor_x")), float(getattr(value, "cursor_y"))
    wait = int(getattr(value, "wait_ticks"))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("nonfinite action")
    if kind is ActionKind.WAIT:
        return Action.wait(min(wait, remaining))
    return Action(kind, x, y, 1)


def _step(
    env: object,
    observation: Mapping[str, Any],
    decision: object,
    *,
    maximum_ticks: int,
) -> tuple[Mapping[str, Any], bool, bool]:
    current = observation
    terminated = truncated = False
    for raw in primitive_actions(decision):
        remaining = maximum_ticks - int(current["tick"])
        if remaining <= 0:
            break
        action = _validated_action(raw, remaining)
        duration = int(action.wait_ticks) if ActionKind.parse(action.kind) is ActionKind.WAIT else 1
        for _ in range(duration):
            primitive = Action.wait(1) if ActionKind.parse(action.kind) is ActionKind.WAIT else action
            current, _reward, terminated, truncated, _info = env.step(primitive)
            if terminated or truncated:
                return current, terminated, truncated
    return current, terminated, truncated


def conservative_exact_error(
    *, parent_hash: int, current_hash: int, policy_before: object
) -> tuple[object, SteeringDecision, None]:
    """Recover from an incomplete exact branch without emitting supervision."""
    if current_hash != parent_hash:
        raise RuntimeError("failed exact branch changed parent state")
    return (
        policy_before,
        SteeringDecision(
            SemanticAction.wait(16),
            SteeringIntent.WAIT,
            reason="conservative exact branch recovery",
        ),
        None,
    )


def collect_seed(
    *,
    runtime: ExactTrainingRuntime,
    model: torch.nn.Module,
    model_sha256: str,
    seed: int,
    maximum_ticks: int,
    low_gauge: int,
    maximum_disagreements: int,
    planner_config: FastMultiActionConfig,
) -> tuple[list[ranking.PairPreference], dict[str, object]]:
    policy = gate._make_policy(model, model_sha256, 1.0)
    policy.reset(seed)
    planner = FastMultiActionPlanner(
        primitive_actions, config=planner_config, action_spec=policy.action_spec
    )
    preferences: list[ranking.PairPreference] = []
    queries = disagreements = branch_errors = rejected_safety = 0
    band_edges = (maximum_ticks // 4, maximum_ticks // 2, 3 * maximum_ticks // 4)
    band_cap = max(1, maximum_disagreements // 4)
    band_counts = [0, 0, 0, 0]
    query_band_counts = [0, 0, 0, 0]
    low_gauge_queries = 0
    low_gauge_labels = 0
    started = time.monotonic()
    with runtime.open_env(
        simulation_config={"max_episode_ticks": maximum_ticks + 512}
    ) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("exact reset seed mismatch")
        terminated = truncated = False
        while (
            int(observation["tick"]) < maximum_ticks
            and len(preferences) < maximum_disagreements
            and not (terminated or truncated)
        ):
            policy_before = copy.deepcopy(policy)
            prediction = policy.predict(observation)
            executed = prediction
            if prediction.is_shot:
                queries += 1
                tick = int(observation["tick"])
                band = sum(tick >= edge for edge in band_edges)
                is_low = int(observation["gauge"]) < low_gauge
                query_band_counts[band] += 1
                low_gauge_queries += int(is_low)
                parent_hash = int(env.state_hash())
                try:
                    verdict = planner.evaluate(
                        env, observation, policy_before, policy, prediction
                    )
                except ExactWorkerError:
                    branch_errors += 1
                    policy, executed, _label = conservative_exact_error(
                        parent_hash=parent_hash,
                        current_hash=int(env.state_hash()),
                        policy_before=policy_before,
                    )
                else:
                    if int(env.state_hash()) != parent_hash:
                        raise RuntimeError("exact planner changed parent state")
                    if not verdict.used_fast_checkpoint:
                        raise RuntimeError("on-policy label query did not use fast checkpoint")
                    executed = verdict.selected.decision
                    policy = verdict.selected.continuation_policy
                    safe = safe_alternate_preference(
                        planner,
                        verdict,
                        branch_complete=True,
                        parent_unchanged=int(env.state_hash()) == parent_hash,
                    )
                    if safe is not None and len(preferences) < maximum_disagreements:
                        teacher, incumbent = safe
                        # Equal per-band caps prevent front-loaded data. Low
                        # gauge coverage is checked globally before fitting.
                        if band_counts[band] < band_cap:
                            provenance = {
                                "schema": "irisu-exact-neutral-onpolicy-pair-preference-v1",
                                "seed": seed,
                                "tick": tick,
                                "gauge": int(observation["gauge"]),
                                "learner_pair": list(ranking._decision_key(prediction)),
                                "teacher_pair": list(ranking._decision_key(teacher.candidate.decision)),
                                "teacher_category": teacher.candidate.category,
                                "teacher_reason": verdict.reason,
                                "planner_manifest": verdict.manifest(),
                                "parent_state_hash": parent_hash,
                                "parent_state_hash_after": int(env.state_hash()),
                                "fast_checkpoint": verdict.used_fast_checkpoint,
                                "tick_band": band,
                                "low_gauge": is_low,
                                "survival_constraint": {
                                    "teacher": teacher.probe.survival_ticks,
                                    "incumbent": incumbent.probe.survival_ticks,
                                    "wait": next(value.probe.survival_ticks for value in verdict.outcomes if value.candidate.category == "wait"),
                                },
                            }
                            preferences.append(
                                ranking._preference(
                                    observation,
                                    teacher.candidate.decision,
                                    incumbent.candidate.decision,
                                    provenance,
                                    encoder=policy.encoder,
                                    pointer_spec=policy.pointer_spec,
                                )
                            )
                            band_counts[band] += 1
                            low_gauge_labels += int(is_low)
                            disagreements += 1
                    elif verdict.selected.category.startswith("top-"):
                        rejected_safety += 1
            observation, terminated, truncated = _step(
                env, observation, executed, maximum_ticks=maximum_ticks
            )
        runtime_manifest = session.provenance_manifest
    return preferences, {
        "seed": seed,
        "final_tick": int(observation["tick"]),
        "final_score": int(observation.get("score", 0)),
        "final_gauge": int(observation.get("gauge", 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "learner_shot_queries": queries,
        "pair_disagreements": disagreements,
        "tick_band_edges": list(band_edges),
        "tick_band_counts": band_counts,
        "query_tick_band_counts": query_band_counts,
        "low_gauge_queries": low_gauge_queries,
        "low_gauge_labels": low_gauge_labels,
        "rejected_alternate_safety": rejected_safety,
        "exact_branch_errors": branch_errors,
        "runtime": runtime_manifest,
        "wall_seconds": time.monotonic() - started,
    }


def restore_preferences(path: Path, schema: Any) -> list[ranking.PairPreference]:
    payload = torch.load(path, weights_only=False)
    output = []
    for value in payload["preferences"]:
        manifest = value["manifest"]
        # torch.Tensor.numpy() is a non-owning view. EncodedBatch deliberately
        # rejects such arrays, so the persisted collection boundary requires
        # owned C-contiguous copies.
        global_features = np.array(value["global_features"].numpy(), copy=True, order="C")
        body_features = np.array(value["body_features"].numpy(), copy=True, order="C")
        body_mask = np.array(value["body_mask"].numpy(), copy=True, order="C")
        observation = EncodedBatch(
            global_features,
            body_features,
            body_mask,
            np.zeros((1,), dtype=np.uint64),
            np.zeros((1,), dtype=np.uint32),
            schema,
        )
        output.append(
            ranking.PairPreference(
                observation,
                int(manifest["positive"][0]),
                int(manifest["positive"][1]),
                int(manifest["negative"][0]),
                int(manifest["negative"][1]),
                manifest["provenance"],
            )
        )
    return output


def collection_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash collection semantics independently of torch serialization bytes."""
    preferences = []
    for value in payload.get("preferences", ()):
        tensors = {}
        for name in ("global_features", "body_features", "body_mask"):
            tensor = value[name].detach().cpu().contiguous()
            array = tensor.numpy()
            tensors[name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        preferences.append({"manifest": value["manifest"], "tensors": tensors})
    semantic = {
        "schema": payload.get("schema"),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "seed_plan": payload.get("seed_plan"),
        "planner_config": payload.get("planner_config"),
        "episodes": payload.get("episodes"),
        "preferences": preferences,
    }
    return gate._sha(semantic)


def validate_resume_collection(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    base_sha256: str,
    seed_plan: Mapping[str, Any],
    planner_config: Mapping[str, Any],
    runtime: ExactTrainingRuntime,
    maximum_ticks: int,
    maximum_disagreements: int,
) -> tuple[list[dict[str, Any]], int, int]:
    if collection_content_sha256(payload) != expected_content_sha256:
        raise RuntimeError("collection semantic content SHA-256 mismatch")
    episodes = payload.get("episodes")
    preferences = payload.get("preferences")
    seeds = [int(value) for value in seed_plan["seeds"]]
    if (
        payload.get("schema") != "irisu-exact-neutral-onpolicy-pair-collection-v1"
        or payload.get("base_checkpoint_sha256") != base_sha256
        or payload.get("seed_plan") != seed_plan
        or payload.get("planner_config") != planner_config
        or not isinstance(episodes, list)
        or not isinstance(preferences, list)
        or len(episodes) != len(seeds)
        or len(preferences) != len(seeds) * maximum_disagreements
    ):
        raise RuntimeError("collection binding/configuration mismatch")
    edges = [maximum_ticks // 4, maximum_ticks // 2, 3 * maximum_ticks // 4]
    identity = runtime.identity.manifest()
    runtime_sha = None
    occupied = set()
    low_gauge = 0
    for expected_seed, episode in zip(seeds, episodes):
        provenance = episode.get("runtime", {})
        attestation = provenance.get("runtime_attestation", {})
        artifact = attestation.get("runtime_artifact", {})
        if (
            int(episode.get("seed", -1)) != expected_seed
            or episode.get("tick_band_edges") != edges
            or sum(int(value) for value in episode.get("tick_band_counts", ()))
            != maximum_disagreements
            or provenance.get("version") != "exact-training-runtime-provenance-v1"
            or provenance.get("physics_backend") != "exact"
            or provenance.get("identity") != identity
            or artifact.get("sha256") != runtime.identity.worker_sha256
            or provenance.get("runtime_attestation_sha256") is None
        ):
            raise RuntimeError("collection episode/runtime provenance mismatch")
        current_runtime_sha = gate._sha(provenance)
        if runtime_sha is not None and current_runtime_sha != runtime_sha:
            raise RuntimeError("collection episodes used different exact runtimes")
        runtime_sha = current_runtime_sha
        occupied.update(
            index
            for index, count in enumerate(episode["tick_band_counts"])
            if int(count) > 0
        )
        low_gauge += int(episode.get("low_gauge_labels", 0))
    by_seed = {seed: 0 for seed in seeds}
    for value in preferences:
        manifest = value.get("manifest", {})
        provenance = manifest.get("provenance", {})
        seed = int(provenance.get("seed", -1))
        if (
            seed not in by_seed
            or provenance.get("schema")
            != "irisu-exact-neutral-onpolicy-pair-preference-v1"
            or provenance.get("fast_checkpoint") is not True
            or provenance.get("parent_state_hash")
            != provenance.get("parent_state_hash_after")
        ):
            raise RuntimeError("collection preference provenance mismatch")
        by_seed[seed] += 1
    if any(count != maximum_disagreements for count in by_seed.values()):
        raise RuntimeError("collection is not seed balanced")
    return episodes, len(occupied), low_gauge


def balanced_new_old(
    new: Sequence[Any], old: Sequence[Any], *, seed: int
) -> tuple[Any, ...]:
    if not new or not old:
        raise ValueError("replay balancing requires new and old preferences")
    count = min(len(new), len(old))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_indices = torch.randperm(len(new), generator=generator)[:count].tolist()
    old_indices = torch.randperm(len(old), generator=generator)[:count].tolist()
    return tuple(
        value
        for pair in zip(
            (new[index] for index in new_indices),
            (old[index] for index in old_indices),
        )
        for value in pair
    )


def balanced_groups(groups: Sequence[Sequence[Any]], *, seed: int) -> tuple[Any, ...]:
    if not groups or any(not group for group in groups):
        raise ValueError("each on-policy training seed must produce preferences")
    count = min(len(group) for group in groups)
    selected = []
    for ordinal, group in enumerate(groups):
        generator = torch.Generator(device="cpu").manual_seed(seed + ordinal)
        indices = torch.randperm(len(group), generator=generator)[:count].tolist()
        selected.append([group[index] for index in indices])
    return tuple(
        selected[group][index]
        for index in range(count)
        for group in range(len(selected))
    )


def train(
    model: torch.nn.Module,
    preferences: Sequence[ranking.PairPreference],
    anchor: Mapping[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    anchor_weight: float,
    seed: int,
) -> dict[str, float | int]:
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    trainable = [value for value in named.values() if value.requires_grad]

    def metrics() -> tuple[float, float, float]:
        losses = correct = count = 0.0
        with torch.no_grad():
            for start in range(0, len(preferences), batch_size):
                indices = range(start, min(start + batch_size, len(preferences)))
                tensors = tuple(x.to(device) for x in ranking._preference_tensors(preferences, indices))
                gf, bf, bm, ps, pd, ns, nd = tensors
                output = model(gf, bf, bm); rows = torch.arange(len(ps), device=device)
                difference = output.pair_logits[rows, ps, pd] - output.pair_logits[rows, ns, nd]
                losses += float(F.softplus(-difference).sum()); correct += float((difference > 0).sum()); count += len(ps)
        penalty = sum(float((named[name] - anchor[name].to(device)).square().sum()) for name in anchor)
        return losses / count, correct / count, penalty

    initial_loss, initial_accuracy, _ = metrics()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    model.eval()
    for _ in range(steps):
        indices = torch.randint(len(preferences), (min(batch_size, len(preferences)),), generator=generator).tolist()
        tensors = tuple(x.to(device) for x in ranking._preference_tensors(preferences, indices))
        gf, bf, bm, ps, pd, ns, nd = tensors
        output = model(gf, bf, bm); rows = torch.arange(len(ps), device=device)
        difference = output.pair_logits[rows, ps, pd] - output.pair_logits[rows, ns, nd]
        penalty = sum((named[name] - anchor[name].to(device)).square().sum() for name in anchor)
        loss = F.softplus(-difference).mean() + anchor_weight * penalty
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
    final_loss, final_accuracy, final_penalty = metrics()
    return {
        "steps": steps, "preference_count": len(preferences),
        "initial_pair_loss": initial_loss, "final_pair_loss": final_loss,
        "initial_pair_accuracy": initial_accuracy, "final_pair_accuracy": final_accuracy,
        "final_anchor_squared_l2": final_penalty,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    resume = args.resume_collection_sha256 is not None
    if resume != (args.resume_collection_content_sha256 is not None):
        raise ValueError("fit-only resume requires both collection hashes")
    if args.output.exists() and not resume:
        raise FileExistsError("output directory must be new")
    if resume and not args.output.is_dir():
        raise FileNotFoundError("fit-only resume output directory is missing")
    seed_plan = load_seed_plan(args.seed_plan)
    seeds = tuple(int(seed) for seed in seed_plan["seeds"])
    artifact = load_steering_checkpoint(args.base_checkpoint, expected_sha256=args.base_sha256)
    upstream = gate.checkpoint_training_seeds(artifact.metadata)
    if set(seeds) & set(upstream):
        raise ValueError("on-policy seeds overlap warm-start lineage")
    model = artifact.model; trainable = configure_trainable(model)
    original = copy.deepcopy(model.state_dict())
    anchor = {name: original[name].clone() for name in trainable}
    planner_config = FastMultiActionConfig(
        probe_ticks=512, long_probe_ticks=512, top_k_pairs=2,
        low_gauge_threshold=0x7FFF_FFFF, low_gauge_exit_threshold=0x7FFF_FFFF,
        maximum_gauge_debt=1_000, rescue_score_margin=500,
    )
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    collected_path = args.output / "collected-onpolicy-preferences.pt"
    if resume:
        for name in ("onpolicy-pair-dagger.pt", "onpolicy-preferences.pt", "provenance.json"):
            if (args.output / name).exists():
                raise FileExistsError("fit-only resume found an existing output product")
        if gate._file_sha(collected_path.resolve(strict=True)) != args.resume_collection_sha256:
            raise RuntimeError("collection file SHA-256 mismatch")
        payload = torch.load(collected_path, weights_only=False)
        episodes, occupied_bands, accepted_low_gauge = validate_resume_collection(
            payload,
            expected_content_sha256=args.resume_collection_content_sha256,
            base_sha256=artifact.sha256,
            seed_plan=seed_plan,
            planner_config=planner_config.manifest(),
            runtime=runtime,
            maximum_ticks=args.maximum_ticks,
            maximum_disagreements=args.maximum_disagreements,
        )
    else:
        groups = []; episodes = []
        rollout_sha = gate._model_state_sha(model)
        for seed in seeds:
            preferences, episode = collect_seed(
                runtime=runtime, model=model, model_sha256=rollout_sha, seed=seed,
                maximum_ticks=args.maximum_ticks,
                low_gauge=args.low_gauge, maximum_disagreements=args.maximum_disagreements,
                planner_config=planner_config,
            )
            groups.append(preferences); episodes.append(episode)
            print(json.dumps({k: v for k, v in episode.items() if k not in ("runtime",)}, sort_keys=True), flush=True)
        new = list(balanced_groups(groups, seed=args.training_seed + 1))
        args.output.mkdir(parents=True)
        payload = {
            "schema": "irisu-exact-neutral-onpolicy-pair-collection-v1",
            "base_checkpoint_sha256": artifact.sha256,
            "seed_plan": seed_plan,
            "planner_config": planner_config.manifest(),
            "episodes": [{k: v for k, v in row.items() if k != "wall_seconds"} for row in episodes],
            "preferences": [
                {
                    "manifest": value.manifest(),
                    "global_features": torch.from_numpy(value.observation.global_features),
                    "body_features": torch.from_numpy(value.observation.body_features),
                    "body_mask": torch.from_numpy(value.observation.body_mask),
                }
                for value in new
            ],
        }
        gate._save_torch_new(collected_path, payload)
        occupied_bands = sum(
            any(int(row["tick_band_counts"][band]) > 0 for row in episodes)
            for band in range(4)
        )
        accepted_low_gauge = sum(int(row["low_gauge_labels"]) for row in episodes)
    if accepted_low_gauge < 1:
        raise RuntimeError("collection has no survival-safe low-gauge positive")
    if occupied_bands < 3:
        raise RuntimeError("collection occupies fewer than three tick bands")

    # Enforce a real collection/fit boundary: fitting consumes only the
    # just-persisted, content-bound collection artifact.
    new = restore_preferences(collected_path, model.schema)
    old_path = args.replay_preferences.resolve(strict=True)
    if gate._file_sha(old_path) != args.replay_sha256:
        raise RuntimeError("old replay preference SHA-256 mismatch")
    old = restore_preferences(old_path, model.schema)
    combined = balanced_new_old(new, old, seed=args.training_seed + 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.training_seed)
        report = train(
            model, combined, anchor, steps=args.training_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            anchor_weight=args.anchor_weight, seed=args.training_seed,
        )
    changed = [name for name, value in model.state_dict().items() if not torch.equal(original[name], value)]
    if not changed or set(changed) - set(trainable):
        raise RuntimeError("on-policy training changed preserved parameters")
    metadata = {
        "schema": "irisu-exact-onpolicy-pair-dagger-v1", "development_only": True,
        "promotion_eligible": False, "base_checkpoint_sha256": artifact.sha256,
        "warm_start_metadata_sha256": gate._sha(dict(artifact.metadata)),
        "upstream_training_seeds": list(upstream), "onpolicy_training_seeds": list(seeds),
        "training_seeds": sorted((*upstream, *seeds)), "seed_plan": seed_plan,
        "inference_config": gate.inference_config(1.0), "planner_config": planner_config.manifest(),
        "collection_gate": {
            "tick_stratification": "four equal rollout quarters",
            "maximum_per_tick_band": args.maximum_disagreements // 4,
            "minimum_low_gauge_labels": 1,
            "minimum_occupied_tick_bands": 3,
            "occupied_tick_bands": occupied_bands,
            "accepted_low_gauge_labels": accepted_low_gauge,
            "low_gauge_exclusive": args.low_gauge,
            "maximum_disagreements_per_seed": args.maximum_disagreements,
            "label_rule": "one survival-neutral strict alternate-pair winner over best proposal pair",
            "no_improvement_behavior": "actual planner rollout, no pair label",
        },
        "episodes": [{k: v for k, v in row.items() if k != "wall_seconds"} for row in episodes],
        "replay": {"path": str(old_path), "sha256": args.replay_sha256, "available": len(old)},
        "collection_artifact": {
            "path": collected_path.name,
            "sha256": gate._file_sha(collected_path),
            "content_sha256": collection_content_sha256(payload),
            "stage_boundary": "collection persisted before fit",
            "fit_only_resume": resume,
        },
        "replay_balance": {"new": len(combined) // 2, "old": len(combined) // 2, "order": "alternating"},
        "trainable_parameters": list(trainable), "changed_parameters": changed,
        "anchor": {"kind": "original-base-parameter-squared-l2", "weight": args.anchor_weight},
        "training_report": report,
        "source": {"trainer_sha256": gate._file_sha(Path(__file__).resolve()), "planner_sha256": gate._file_sha(ROOT / "python/irisu_pointer/fast_multiaction_planner.py")},
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "output"},
    }
    checkpoint = args.output / "onpolicy-pair-dagger.pt"
    checkpoint_sha = save_steering_checkpoint(checkpoint, model, metadata=metadata)
    labels = args.output / "onpolicy-preferences.pt"
    gate._save_torch_new(labels, {"schema": "irisu-exact-onpolicy-pair-preferences-v1", "metadata": metadata, "preferences": [value.manifest() for value in new]})
    result = {**metadata, "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha, "labels_artifact": labels.name, "labels_artifact_sha256": gate._file_sha(labels), "operational": {"episode_wall_seconds": [row.get("wall_seconds") for row in episodes]}}
    result["sha256"] = gate._sha(result); gate._write_json_new(args.output / "provenance.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path, required=True); p.add_argument("--base-checkpoint", type=Path, required=True); p.add_argument("--base-sha256", required=True)
    p.add_argument("--seed-plan", type=Path, required=True); p.add_argument("--replay-preferences", type=Path, required=True); p.add_argument("--replay-sha256", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-ticks", type=int, default=10_000); p.add_argument("--low-gauge", type=int, default=20_000); p.add_argument("--maximum-disagreements", type=int, default=128)
    p.add_argument("--training-steps", type=int, default=125); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=1e-5); p.add_argument("--anchor-weight", type=float, default=1e-3); p.add_argument("--training-seed", type=int, default=2026081202)
    p.add_argument("--resume-collection-sha256"); p.add_argument("--resume-collection-content-sha256")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parser().parse_args(argv)); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
