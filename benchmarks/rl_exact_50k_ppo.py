#!/usr/bin/env python3
"""Exact-trace warm start and exact-backend recurrent PPO development runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import struct
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_env import Action
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind
from irisu_rl.collector import (
    CollectorConfig,
    R3ATrainingSession,
    RecurrentCollector,
    ScoreTaskContract,
    model_state_sha256,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.exact_training_runtime import ExactTrainingRuntime
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.seeds import SeedAllocator
from irisu_rl.torch_distribution import TorchConditionalActionDistribution
from irisu_rl.vector_adapter import MacroVectorAdapter


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "python", ROOT / "benchmarks"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
DEFAULT_CONFIG = ROOT / "configs/rl/experiments/exact-50k-ppo-v1.toml"
WORKER = ROOT / "build-physics-integration-exact-multiworld-2/irisu-exact-worker"
TRACE_WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


@dataclass(slots=True)
class EpisodeData:
    global_features: np.ndarray
    body_features: np.ndarray
    body_mask: np.ndarray
    kind: np.ndarray
    wait_index: np.ndarray
    xy: np.ndarray
    source_index: int
    source_seed: int

    @property
    def length(self) -> int:
        return int(self.kind.shape[0])


def decode_word(word: int) -> Action:
    kind = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if kind == 1:
        return Action.weak(x, y)
    if kind == 2:
        return Action.strong(x, y)
    if kind == 3:
        return Action.both(x, y)
    return Action.wait(1)


def encode_action(action: Action) -> int:
    kind = int(action.kind)
    if kind == 0:
        return 0
    x, y = int(action.cursor_x), int(action.cursor_y)
    return (y << 12) | (x << 2) | kind


def collect_teacher_data(
    config: dict[str, Any], run_root: Path, worker: Path, seeds: list[int]
) -> tuple[list[EpisodeData], dict[str, object]]:
    # This controller is only a warm start. Every state, action outcome, target,
    # and exported trace below is newly generated against the attested worker.
    import rl_r3m_shot_restraint as r3m

    core, campaign = r3m.screen._load_external()
    encoder = TeacherStateEncoder()
    action_spec = ActionSpec()
    episodes: list[EpisodeData] = []
    source_records = []
    runtime = ExactTrainingRuntime(worker)
    with runtime.open_env(
        simulation_config={"max_episode_ticks": int(config["maximum_episode_ticks"])}
    ) as session:
        env = session.environment
        provenance = session.provenance_manifest
        for source_index, seed in enumerate(seeds):
            policy = campaign.POLICY_FACTORY()
            policy.reset(seed)
            gate = None
            if bool(config["teacher_gate"]):
                from irisu_pointer.shot_necessity import ExactWaitDominanceGate

                gate = ExactWaitDominanceGate(
                    lambda decision: r3m.screen._primitive_actions(core, decision),
                    config=r3m.GATE_CONFIG,
                )
            observation, info = env.reset(seed=seed)
            if int(info.get("seed", -1)) != seed:
                raise RuntimeError("exact teacher reset seed differs")
            global_rows: list[np.ndarray] = []
            body_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            kinds: list[int] = []
            waits: list[int] = []
            coordinates: list[tuple[float, float]] = []
            words: list[int] = []
            reasons: Counter[str] = Counter()
            attempted = kept = suppressed = 0
            terminated = truncated = False
            started = time.perf_counter()

            def record_label(kind: int, wait_index: int, xy: tuple[float, float]) -> None:
                encoded = encoder.encode([observation])
                global_rows.append(encoded.global_features[0])
                body_rows.append(encoded.body_features[0].astype(np.float16))
                mask_rows.append(encoded.body_mask[0])
                kinds.append(kind)
                waits.append(wait_index)
                coordinates.append(xy)

            # Match the real replay's two neutral startup ticks and expose them
            # as one legal semantic action to the recurrent clone.
            record_label(int(SemanticActionKind.WAIT), 1, (0.0, 0.0))
            for _ in range(2):
                words.append(0)
                observation, _, terminated, truncated, _ = env.step(Action.wait(1))
            while not (terminated or truncated):
                before = copy.deepcopy(policy)
                decision = policy.predict(observation)
                if gate is not None and getattr(decision, "is_shot", False):
                    attempted += 1
                    verdict = gate.evaluate(env, observation, before, policy, decision)
                    reasons[verdict.reason] += 1
                    if verdict.execute_shot:
                        kept += 1
                    else:
                        suppressed += 1
                        policy = before
                        decision = gate.wait_decision(verdict.reason)
                primitives = r3m.screen._primitive_actions(core, decision)
                if not primitives:
                    raise RuntimeError("exact teacher produced no primitive action")
                first = primitives[0]
                first_kind = int(first.kind)
                if first_kind == 0:
                    duration = int(first.wait_ticks)
                    if len(primitives) != 1 or duration not in action_spec.wait_choices:
                        raise RuntimeError("teacher wait is outside PPO action vocabulary")
                    record_label(
                        int(SemanticActionKind.WAIT),
                        action_spec.wait_choices.index(duration),
                        (0.0, 0.0),
                    )
                else:
                    if first_kind == 3:
                        raise RuntimeError("deployment-v1 cannot represent teacher BOTH")
                    x, y = int(first.cursor_x), int(first.cursor_y)
                    if not (0 <= x < 640 and 0 <= y < 480):
                        raise RuntimeError("teacher shot lies outside client bounds")
                    record_label(
                        int(SemanticActionKind.FIRE_WEAK)
                        if first_kind == 1
                        else int(SemanticActionKind.FIRE_STRONG),
                        0,
                        ((x + 0.5) / 640.0, (y + 0.5) / 480.0),
                    )
                for primitive in primitives:
                    kind = int(primitive.kind)
                    duration = int(primitive.wait_ticks) if kind == 0 else 1
                    for _ in range(duration):
                        action = Action.wait(1) if kind == 0 else primitive
                        words.append(encode_action(action))
                        observation, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
                    if terminated or truncated:
                        break
                if int(observation["tick"]) % 5_000 < 20:
                    print(
                        json.dumps(
                            {
                                "exact_teacher": source_index,
                                "seed": seed,
                                "tick": int(observation["tick"]),
                                "score": int(observation["score"]),
                                "elapsed_seconds": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if truncated or not terminated:
                raise RuntimeError("exact teacher did not reach natural GAME_OVER")
            episode = EpisodeData(
                np.stack(global_rows),
                np.stack(body_rows),
                np.stack(mask_rows),
                np.asarray(kinds, dtype=np.int64),
                np.asarray(waits, dtype=np.int64),
                np.asarray(coordinates, dtype=np.float32),
                source_index,
                seed,
            )
            episodes.append(episode)
            dataset_path = run_root / "dataset" / f"teacher-{source_index:02d}.npz"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dataset_path,
                global_features=episode.global_features,
                body_features=episode.body_features,
                body_mask=episode.body_mask,
                kind=episode.kind,
                wait_index=episode.wait_index,
                xy=episode.xy,
            )
            trace = b"".join(TRACE_WORD.pack(word) for word in words)
            trace_path = run_root / "teacher-traces" / f"{source_index:02d}.u32le"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_bytes(trace)
            replay = REPLAY_HEADER.pack(
                seed,
                int(observation["level"]),
                int(observation["score"]),
                int(observation.get("highest_chain", 0)),
                0,
            ) + bytes(32) + trace
            replay_path = trace_path.with_suffix(".rpy")
            replay_path.write_bytes(replay)
            source_records.append(
                {
                    "index": source_index,
                    "seed": seed,
                    "score": int(observation["score"]),
                    "ticks": int(observation["tick"]),
                    "terminal": True,
                    "semantic_decisions": episode.length,
                    "trace": str(trace_path.relative_to(run_root)),
                    "trace_sha256": sha256_bytes(trace),
                    "replay": str(replay_path.relative_to(run_root)),
                    "replay_sha256": sha256_bytes(replay),
                    "gate": bool(gate),
                    "attempted_shots": attempted,
                    "kept_shots": kept,
                    "suppressed_shots": suppressed,
                    "gate_reasons": dict(sorted(reasons.items())),
                    "wall_seconds": time.perf_counter() - started,
                    "dataset": str(dataset_path.relative_to(run_root)),
                    "dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
                }
            )
    manifest = {
        "version": "exact-50k-ppo-teacher-dataset-v1",
        "physics_backend": "exact",
        "exact_runtime": provenance,
        "source_controller": "portable-frozen-v5 warm start, executed closed-loop on exact",
        "source_controller_checkpoint": str(r3m.screen.BASE_CHECKPOINT),
        "source_controller_checkpoint_sha256": sha256_bytes(
            r3m.screen.BASE_CHECKPOINT.read_bytes()
        ),
        "targets_derived_exclusively_from_exact_rollouts": True,
        "episodes": source_records,
        "total_semantic_decisions": sum(value.length for value in episodes),
    }
    write_json(run_root / "dataset-manifest.json", manifest)
    return episodes, manifest


def load_exact_teacher_data(
    source_root: Path, run_root: Path
) -> tuple[list[EpisodeData], dict[str, object]]:
    source = source_root.resolve(strict=True)
    manifest_path = source / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("physics_backend") != "exact"
        or manifest.get("targets_derived_exclusively_from_exact_rollouts") is not True
        or manifest.get("exact_runtime", {}).get("physics_backend") != "exact"
    ):
        raise RuntimeError("teacher dataset lacks fail-closed exact provenance")
    episodes = []
    for record in manifest.get("episodes", []):
        if record.get("terminal") is not True:
            raise RuntimeError("teacher dataset contains a nonterminal episode")
        for file_key, hash_key in (
            ("dataset", "dataset_sha256"),
            ("trace", "trace_sha256"),
            ("replay", "replay_sha256"),
        ):
            path = source / str(record[file_key])
            if sha256_bytes(path.read_bytes()) != record[hash_key]:
                raise RuntimeError(f"teacher {file_key} hash mismatch")
        values = np.load(source / str(record["dataset"]), allow_pickle=False)
        episodes.append(
            EpisodeData(
                values["global_features"],
                values["body_features"],
                values["body_mask"],
                values["kind"],
                values["wait_index"],
                values["xy"],
                int(record["index"]),
                int(record["seed"]),
            )
        )
    if not episodes:
        raise RuntimeError("exact teacher dataset is empty")
    derived: dict[str, object] = {
        "version": "exact-50k-ppo-reused-teacher-dataset-v1",
        "physics_backend": "exact",
        "targets_derived_exclusively_from_exact_rollouts": True,
        "source_root": str(source),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "exact_runtime": manifest["exact_runtime"],
        "episodes": manifest["episodes"],
        "total_semantic_decisions": sum(value.length for value in episodes),
    }
    write_json(run_root / "dataset-manifest.json", derived)
    return episodes, derived


def model_from_config(config: dict[str, Any]) -> RecurrentActorCritic:
    value = config["model"]
    return RecurrentActorCritic(
        TeacherStateEncoder().schema,
        config=RecurrentModelConfig(
            int(value["global_hidden"]),
            int(value["body_hidden"]),
            int(value["fused_hidden"]),
            int(value["recurrent_hidden"]),
            int(value["recurrent_layers"]),
        ),
    )


def fit_behavioral_clone(
    model: RecurrentActorCritic,
    episodes: list[EpisodeData],
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, object]:
    bc = config["behavioral_cloning"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(bc["learning_rate"]), eps=1e-5)
    length = int(bc["sequence_length"])
    batch_size = int(bc["batch_size"])
    reports = []
    started = time.perf_counter()
    model.train()
    for step in range(1, int(bc["steps"]) + 1):
        chosen = [episodes[int(rng.integers(len(episodes)))] for _ in range(batch_size)]
        starts = [int(rng.integers(max(1, row.length - length + 1))) for row in chosen]
        global_features = torch.stack(
            [torch.from_numpy(row.global_features[start : start + length]) for row, start in zip(chosen, starts)],
            dim=1,
        )
        body_features = torch.stack(
            [torch.from_numpy(row.body_features[start : start + length]).float() for row, start in zip(chosen, starts)],
            dim=1,
        )
        body_mask = torch.stack(
            [torch.from_numpy(row.body_mask[start : start + length]) for row, start in zip(chosen, starts)],
            dim=1,
        )
        kinds = torch.stack(
            [torch.from_numpy(row.kind[start : start + length]) for row, start in zip(chosen, starts)],
            dim=1,
        )
        waits = torch.stack(
            [torch.from_numpy(row.wait_index[start : start + length]) for row, start in zip(chosen, starts)],
            dim=1,
        )
        xy = torch.stack(
            [torch.from_numpy(row.xy[start : start + length]) for row, start in zip(chosen, starts)],
            dim=1,
        )
        output = model(
            global_features,
            body_features,
            body_mask,
            model.initial_state(batch_size),
            reset_before=torch.zeros((length, batch_size), dtype=torch.bool),
        )
        per_kind = F.cross_entropy(
            output.kind_logits.reshape(-1, 3), kinds.reshape(-1), reduction="none"
        ).reshape_as(kinds)
        kind_weight = torch.where(
            kinds == 0,
            torch.ones_like(per_kind),
            torch.full_like(per_kind, float(bc["shot_kind_weight"])),
        )
        kind_loss = (per_kind * kind_weight).mean()
        wait_rows = kinds == 0
        wait_loss = F.cross_entropy(output.wait_logits[wait_rows], waits[wait_rows])
        shot_rows = kinds != 0
        branch = (kinds[shot_rows] - 1).long()
        means = output.coordinate_alpha / (
            output.coordinate_alpha + output.coordinate_beta
        )
        selected_means = means[shot_rows][torch.arange(int(shot_rows.sum())), branch]
        coordinate_loss = F.mse_loss(selected_means, xy[shot_rows])
        loss = (
            kind_loss
            + float(bc["wait_loss_weight"]) * wait_loss
            + float(bc["coordinate_mse_weight"]) * coordinate_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == int(bc["steps"]):
            reports.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "kind_loss": float(kind_loss.detach()),
                    "wait_loss": float(wait_loss.detach()),
                    "coordinate_mse": float(coordinate_loss.detach()),
                    "gradient_norm": float(gradient),
                }
            )
            print(json.dumps({"bc": reports[-1]}, sort_keys=True), flush=True)
    return {
        "steps": int(bc["steps"]),
        "wall_seconds": time.perf_counter() - started,
        "reports": reports,
    }


@torch.no_grad()
def policy_action(
    model: RecurrentActorCritic,
    encoded: Any,
    recurrent_state: Tensor,
    *,
    deterministic: bool,
) -> tuple[SemanticAction, Tensor]:
    output = model(
        torch.from_numpy(encoded.global_features).unsqueeze(0),
        torch.from_numpy(encoded.body_features).unsqueeze(0),
        torch.from_numpy(encoded.body_mask).unsqueeze(0),
        recurrent_state,
        reset_before=torch.zeros((1, 1), dtype=torch.bool),
    )
    distribution = TorchConditionalActionDistribution(
        output.kind_logits,
        output.wait_logits,
        output.coordinate_alpha,
        output.coordinate_beta,
        spec=model.action_spec,
    )
    selected = distribution.deterministic() if deterministic else distribution.sample()
    action = model.action_spec.decode(
        int(selected.kind[0, 0]),
        int(selected.wait_index[0, 0]),
        float(selected.xy[0, 0, 0]),
        float(selected.xy[0, 0, 1]),
    )
    return action, output.recurrent_state


def evaluate_model(
    model: RecurrentActorCritic,
    config: dict[str, Any],
    run_root: Path,
    worker: Path,
    *,
    stage: str,
    seeds: list[int],
    deterministic: bool,
    maximum_ticks: int | None = None,
) -> dict[str, object]:
    encoder = TeacherStateEncoder()
    spec = model.action_spec
    results = []
    was_training = model.training
    training_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    started_all = time.perf_counter()
    with ExactTrainingRuntime(worker).open_env(
        simulation_config={"max_episode_ticks": int(config["maximum_episode_ticks"])}
    ) as session:
        provenance = session.provenance_manifest
        env = session.environment
        for ordinal, seed in enumerate(seeds):
            torch.manual_seed(seed ^ 0x50C0FFEE)
            observation, _ = env.reset(seed=seed)
            recurrent_state = model.initial_state(1)
            trace: list[int] = []
            decisions = 0
            next_progress_tick = 5_000
            started = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated) and (
                maximum_ticks is None or int(observation["tick"]) < maximum_ticks
            ):
                encoded = encoder.encode([observation])
                semantic, recurrent_state = policy_action(
                    model,
                    encoded,
                    recurrent_state,
                    deterministic=deterministic,
                )
                primitive = spec.press(semantic)
                if semantic.kind == SemanticActionKind.WAIT:
                    trace.extend([0] * semantic.wait_ticks)
                    observation, _, terminated, truncated, _ = env.step(primitive)
                else:
                    trace.append(encode_action(primitive))
                    observation, _, terminated, truncated, _ = env.step(primitive)
                    if not (terminated or truncated):
                        trace.append(0)
                        observation, _, terminated, truncated, _ = env.step(spec.release())
                decisions += 1
                if int(observation["tick"]) >= next_progress_tick:
                    print(
                        json.dumps(
                            {
                                "evaluation_progress": stage,
                                "seed": seed,
                                "tick": int(observation["tick"]),
                                "score": int(observation["score"]),
                                "decisions": decisions,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    next_progress_tick += 5_000
            trace_bytes = b"".join(TRACE_WORD.pack(word) for word in trace)
            trace_path = run_root / "evaluations" / stage / f"{ordinal:02d}.u32le"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_bytes(trace_bytes)
            replay = REPLAY_HEADER.pack(
                seed,
                int(observation["level"]),
                int(observation["score"]),
                int(observation.get("highest_chain", 0)),
                0,
            ) + bytes(32) + trace_bytes
            replay_path = trace_path.with_suffix(".rpy")
            replay_path.write_bytes(replay)
            result = {
                "ordinal": ordinal,
                "seed": seed,
                "score": int(observation["score"]),
                "level": int(observation["level"]),
                "highest_chain": int(observation.get("highest_chain", 0)),
                "ticks": int(observation["tick"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "censored": not (terminated or truncated),
                "decisions": decisions,
                "wall_seconds": time.perf_counter() - started,
                "trace": str(trace_path.relative_to(run_root)),
                "trace_sha256": sha256_bytes(trace_bytes),
                "replay": str(replay_path.relative_to(run_root)),
                "replay_sha256": sha256_bytes(replay),
                "final_state_u64": f"0x{int(env.state_hash()) & 0xffffffffffffffff:016x}",
            }
            results.append(result)
            print(json.dumps({"evaluation": stage, **result}, sort_keys=True), flush=True)
    model.train(was_training)
    torch.set_num_threads(training_threads)
    report = {
        "version": "exact-50k-ppo-evaluation-v1",
        "stage": stage,
        "physics_backend": "exact",
        "exact_runtime": provenance,
        "model_sha256": model_state_sha256(model),
        "deterministic": deterministic,
        "evaluation_maximum_ticks": maximum_ticks,
        "all_natural_terminal": all(not value["censored"] for value in results),
        "promotion_eligible": all(not value["censored"] for value in results),
        "wall_seconds": time.perf_counter() - started_all,
        "scores": [value["score"] for value in results],
        "mean_score": statistics.fmean(value["score"] for value in results),
        "best_score": max(value["score"] for value in results),
        "results": results,
    }
    write_json(run_root / "evaluations" / stage / "report.json", report)
    return report


def save_model(run_root: Path, stage: str, model: RecurrentActorCritic, extra: object) -> Path:
    path = run_root / "checkpoints" / f"{stage}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": "exact-50k-ppo-checkpoint-v1",
            "stage": stage,
            "model_manifest": model.manifest(),
            "model_state": model.state_dict(),
            "model_sha256": model_state_sha256(model),
            "extra": extra,
        },
        path,
    )
    return path


def run_ppo(
    model: RecurrentActorCritic,
    config: dict[str, Any],
    run_root: Path,
    worker: Path,
    *,
    seed: int,
    evaluation_seeds: list[int],
    evaluation_maximum_ticks: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    value = config["ppo"]
    lanes = int(value["lanes"])
    updates = int(value["updates"])
    reports = []
    evaluations = []
    runtime = ExactTrainingRuntime(worker)
    with runtime.open_vector(
        lanes,
        simulation_config={"max_episode_ticks": int(config["maximum_episode_ticks"])},
    ) as exact_session:
        vector = exact_session.environment
        adapter = MacroVectorAdapter(
            vector,
            encoder=TeacherStateEncoder(),
            seed_allocator=SeedAllocator("train", key=seed),
        )
        task = ScoreTaskContract(lanes, reward_scale=float(value["reward_scale"]))
        collector = RecurrentCollector(
            model,
            adapter,
            task,
            config=CollectorConfig(
                max_decisions=int(value["max_decisions"]),
                target_simulated_ticks=int(value["target_simulated_ticks"]),
                gamma_tick=1.0,
                lambda_tick=0.9862327044933592,
            ),
            policy_sampler_seed=seed ^ 0xA5A5,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                learning_rate=float(value["learning_rate"]),
                final_learning_rate_fraction=0.2,
                epochs=int(value["epochs"]),
                lane_minibatch_size=int(value["lane_minibatch_size"]),
                clip_ratio=0.2,
                value_clip=0.2,
                value_coefficient=0.5,
                entropy_coefficient=float(value["entropy_coefficient"]),
                max_gradient_norm=0.5,
                target_kl=float(value["target_kl"]),
            ),
            total_updates=updates,
            sampler_seed=seed ^ 0x5A5A,
        )
        training = R3ATrainingSession(collector, trainer, numpy_seed=seed ^ 17)
        training.initialize()
        for update in range(1, updates + 1):
            started = time.perf_counter()
            result = training.run_update()
            report: dict[str, object] = {
                "update": update,
                "wall_seconds": time.perf_counter() - started,
                "transitions": result.collection.transitions,
                "simulated_ticks": result.collection.simulated_ticks,
                "raw_reward": result.collection.raw_reward,
                "completed_episodes": result.collection.completed_episodes,
                "invalid_actions": result.collection.invalid_actions,
                "skipped_reason": result.skipped_reason,
            }
            if result.optimizer is not None:
                report.update(
                    approximate_kl=result.optimizer.approximate_kl,
                    gradient_norm=result.optimizer.gradient_norm,
                    learning_rate=result.optimizer.learning_rate,
                    entropy=result.optimizer.entropy,
                )
            reports.append(report)
            print(json.dumps({"ppo": report}, sort_keys=True), flush=True)
            if update in {8, updates}:
                stage = f"ppo-{update:04d}"
                save_model(run_root, stage, model, report)
                evaluations.append(
                    evaluate_model(
                        model,
                        config,
                        run_root,
                        worker,
                        stage=stage,
                        seeds=evaluation_seeds,
                        deterministic=True,
                        maximum_ticks=evaluation_maximum_ticks,
                    )
                )
        provenance = exact_session.provenance_manifest
    write_json(
        run_root / "ppo-report.json",
        {
            "version": "exact-50k-ppo-training-v1",
            "physics_backend": "exact",
            "exact_runtime": provenance,
            "reports": reports,
        },
    )
    return reports, evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081001)
    parser.add_argument("--teacher-count", type=int)
    parser.add_argument("--teacher-dataset-root", type=Path)
    parser.add_argument("--bc-steps", type=int)
    parser.add_argument("--ppo-updates", type=int)
    parser.add_argument("--evaluation-count", type=int)
    parser.add_argument("--evaluation-maximum-ticks", type=int)
    parser.add_argument("--skip-bc-evaluation", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()
    if args.run_root.exists():
        raise SystemExit("run root already exists")
    if "exact-50k-ppo-" not in args.run_root.name:
        raise SystemExit("run root must use the exact-50k-ppo-* namespace")
    config_path = args.config.resolve(strict=True)
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    if args.bc_steps is not None:
        config["behavioral_cloning"]["steps"] = args.bc_steps
    if args.ppo_updates is not None:
        config["ppo"]["updates"] = args.ppo_updates
    worker = args.worker.resolve(strict=True)
    args.run_root.mkdir(parents=True)
    torch.set_num_threads(4)
    teacher_seeds = [int(value) for value in config["teacher_seeds"]]
    if args.teacher_count is not None:
        teacher_seeds = teacher_seeds[: args.teacher_count]
    identity = {
        "version": "exact-50k-ppo-run-v1",
        "physics_backend": "exact",
        "config": str(config_path),
        "config_sha256": sha256_bytes(config_bytes),
        "worker": str(worker),
        "worker_sha256": sha256_bytes(worker.read_bytes()),
        "seed": args.seed,
        "teacher_seeds": teacher_seeds,
        "teacher_source": config["teacher_source"],
    }
    write_json(args.run_root / "run-identity.json", identity)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    model = model_from_config(config)
    if args.skip_teacher:
        dataset: dict[str, object] = {
            "version": "exact-50k-ppo-no-teacher-v1",
            "physics_backend": "exact",
            "episodes": [],
            "targets_derived_exclusively_from_exact_rollouts": True,
        }
        write_json(args.run_root / "dataset-manifest.json", dataset)
        bc: dict[str, object] = {"skipped": True, "reason": "exact PPO scratch baseline"}
    else:
        if args.teacher_dataset_root is None:
            episodes, dataset = collect_teacher_data(
                config, args.run_root, worker, teacher_seeds
            )
        else:
            episodes, dataset = load_exact_teacher_data(
                args.teacher_dataset_root, args.run_root
            )
        bc = fit_behavioral_clone(model, episodes, config, seed=args.seed)
    bc_checkpoint = save_model(args.run_root, "bc", model, bc)
    evaluation_seeds = [int(value) for value in config["evaluation"]["seeds"]]
    if args.evaluation_count is not None:
        evaluation_seeds = evaluation_seeds[: args.evaluation_count]
    evaluations = []
    if not args.skip_bc_evaluation:
        evaluations.append(
            evaluate_model(
                model,
                config,
                args.run_root,
                worker,
                stage="bc",
                seeds=evaluation_seeds,
                deterministic=bool(config["evaluation"]["deterministic"]),
                maximum_ticks=args.evaluation_maximum_ticks,
            )
        )
    ppo_reports = []
    if not args.skip_ppo:
        ppo_reports, added = run_ppo(
            model,
            config,
            args.run_root,
            worker,
            seed=args.seed,
            evaluation_seeds=evaluation_seeds,
            evaluation_maximum_ticks=args.evaluation_maximum_ticks,
        )
        evaluations.extend(added)
    summary = {
        "version": "exact-50k-ppo-summary-v1",
        "physics_backend": "exact",
        "run_identity": identity,
        "dataset_manifest": dataset,
        "behavioral_cloning": bc,
        "bc_checkpoint": str(bc_checkpoint.relative_to(args.run_root)),
        "ppo_updates": len(ppo_reports),
        "evaluations": [
            {
                "stage": value["stage"],
                "model_sha256": value["model_sha256"],
                "scores": value["scores"],
                "mean_score": value["mean_score"],
                "best_score": value["best_score"],
                "all_natural_terminal": value["all_natural_terminal"],
                "promotion_eligible": value["promotion_eligible"],
            }
            for value in evaluations
        ],
        "best_score": max((value["best_score"] for value in evaluations), default=0),
        "target_reached": any(
            value["promotion_eligible"] and value["best_score"] >= 50_000
            for value in evaluations
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
