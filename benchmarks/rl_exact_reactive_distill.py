#!/usr/bin/env python3
"""Distill exact trajectories into a seed/tick-free public-state policy.

The deployment policy first retrieves an action by a permutation-invariant
digest of the physical public state.  Unseen states are handled by a compact
nearest-neighbour classifier fitted only to exact-backend observations.  Seed,
tick, score, and cumulative episode counters are deliberately unavailable to
both paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


WORKER = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/exact-runtime-backup/irisu-exact-worker"
TRACE = ROOT / "artifacts/r3/development/exact-seed-tick-distill-20260810-004/dataset/exact-teacher.u32le"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-reactive-distill-20260810-001"
SOURCE_SEED = 3_939_967_453
HOLDOUT_SEEDS = (0x13579BDF, 0x2468ACE0, 0x5A17C0DE)
WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")
LEGACY_FEATURE_COUNT = 218
COMBO_FEATURE_COUNT = 226

_EPISODE_ONLY = {
    "tick", "score", "highest_chain", "qualifying_clear_count",
    "terminated", "truncated", "left_held", "right_held",
}
_BODY_IDENTITY_OR_CLOCK = {"id", "age_ticks"}
_LIFECYCLES = (
    "scripted_falling", "falling", "dynamic_fresh", "fresh",
    "confirmed", "rotten", "projectile",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return _plain(item())
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public state contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child) for child in value]
    raise TypeError(f"unsupported public state value: {type(value).__name__}")


def physical_state(observation: Mapping[str, Any]) -> dict[str, object]:
    """Permutation-invariant public physics state without seed/tick proxies."""

    result = {
        str(key): _plain(value)
        for key, value in observation.items()
        if key not in _EPISODE_ONLY and key != "bodies"
    }
    bodies = []
    for body in observation.get("bodies", ()):
        if not isinstance(body, Mapping):
            continue
        row = {
            str(key): _plain(value)
            for key, value in body.items()
            if key not in _BODY_IDENTITY_OR_CLOCK
        }
        bodies.append(row)
    bodies.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    result["bodies"] = bodies
    return result


def state_digest(observation: Mapping[str, Any]) -> str:
    payload = json.dumps(
        physical_state(observation), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def features(
    observation: Mapping[str, Any], *, combo_aware: bool = True
) -> np.ndarray:
    """Small permutation-invariant features for OOD nearest-state inference."""

    gauge_max = max(1, int(observation.get("gauge_max", 1)))
    difficulty = observation.get("difficulty", {})
    values = [
        int(observation.get("gauge", 0)) / gauge_max,
        min(1.0, int(observation.get("level", 1)) / 100.0),
        int(difficulty.get("active_colors", 0)) / 6.0,
        min(1.0, int(difficulty.get("spawn_interval_ticks", 0)) / 100.0),
    ]
    bodies = [body for body in observation.get("bodies", ()) if isinstance(body, Mapping)]
    values.append(min(1.0, len(bodies) / 196.0))
    pieces = [body for body in bodies if str(body.get("kind", "")) == "piece"]

    # Describe active combo topology without using chain IDs as identities.
    # The histogram is invariant to body order and arbitrary chain-ID labels,
    # while retaining the distinction between scattered pairs and one large
    # clear-ready group that the earlier population means erased.
    chain_sizes = Counter(
        int(body.get("chain_id", 0))
        for body in pieces
        if int(body.get("chain_id", 0)) > 0
    )
    grouped = sum(chain_sizes.values())
    confirmed_grouped = sum(
        int(body.get("chain_id", 0)) > 0
        and str(body.get("lifecycle", "")) == "confirmed"
        for body in pieces
    )
    touched = sum(int(body.get("projectile_hits", 0)) > 0 for body in pieces)
    if combo_aware:
        values.extend(
            (
                min(1.0, len(chain_sizes) / 32.0),
                min(1.0, max(chain_sizes.values(), default=0) / 32.0),
                min(1.0, sum(size >= 2 for size in chain_sizes.values()) / 16.0),
                min(1.0, sum(size >= 3 for size in chain_sizes.values()) / 16.0),
                min(1.0, grouped / 196.0),
                min(1.0, confirmed_grouped / 196.0),
                min(1.0, touched / 196.0),
                min(
                    1.0,
                    max(
                        (int(body.get("projectile_hits", 0)) for body in pieces),
                        default=0,
                    )
                    / 8.0,
                ),
            )
        )
    # Color/lifecycle population and geometry moments.  No body ID or clock is used.
    for color in range(6):
        for lifecycle in _LIFECYCLES:
            selected = [
                body for body in bodies
                if int(body.get("color", -1)) == color
                and str(body.get("lifecycle", "")) == lifecycle
            ]
            values.extend((
                min(1.0, len(selected) / 32.0),
                sum(float(body.get("x", 0.0)) for body in selected) / max(1, len(selected)) / 640.0,
                sum(float(body.get("y", 0.0)) for body in selected) / max(1, len(selected)) / 480.0,
                sum(float(body.get("vx", 0.0)) for body in selected) / max(1, len(selected)) / 100.0,
                sum(float(body.get("vy", 0.0)) for body in selected) / max(1, len(selected)) / 100.0,
            ))
    for kind in ("piece", "bonus", "projectile"):
        selected = [body for body in bodies if str(body.get("kind", "")) == kind]
        values.append(min(1.0, len(selected) / 196.0))
    return np.asarray(values, dtype=np.float32)


def decode(word: int) -> Action:
    kind = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if kind == 1:
        return Action.weak(x, y)
    if kind == 2:
        return Action.strong(x, y)
    if kind == 3:
        return Action.both(x, y)
    return Action.wait(1)


def checkpoint(observation: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: int(observation[key])
        for key in (
            "tick", "score", "gauge", "level", "highest_chain",
            "qualifying_clear_count", "terminated", "truncated",
        )
    }


def canonical_checkpoint(
    observation: Mapping[str, Any], step_info: Mapping[str, Any]
) -> dict[str, int | bool]:
    live = checkpoint(observation)
    diagnostics = step_info.get("diagnostics", {})
    recorded = (
        isinstance(diagnostics, Mapping)
        and bool(diagnostics.get("terminal_metadata_recorded", False))
    )
    return {
        **live,
        "score": int(
            diagnostics.get("recorded_final_score", live["score"])
            if recorded else live["score"]
        ),
        "level": int(
            diagnostics.get("recorded_final_level", live["level"])
            if recorded else live["level"]
        ),
        "highest_chain": int(
            diagnostics.get("recorded_final_highest_chain", live["highest_chain"])
            if recorded else live["highest_chain"]
        ),
        "terminal_metadata_recorded": recorded,
    }


class ReactiveExactPolicy:
    """Physical-state retrieval with a learned nearest-state OOD path."""

    def __init__(
        self,
        checkpoint_value: Mapping[str, Any],
        *,
        minimum_shot_neighbors: int | None = None,
    ) -> None:
        self.actions = dict(checkpoint_value["state_actions"])
        self.mean = np.asarray(checkpoint_value["mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint_value["scale"], dtype=np.float32)
        self.exemplar_features = np.asarray(checkpoint_value["exemplar_features"], dtype=np.float32)
        self.exemplar_words = np.asarray(checkpoint_value["exemplar_words"], dtype=np.uint32)
        self.feature_count = int(self.mean.shape[0])
        if self.feature_count not in (LEGACY_FEATURE_COUNT, COMBO_FEATURE_COUNT):
            raise ValueError(f"unsupported reactive feature count: {self.feature_count}")
        self.minimum_shot_neighbors = int(
            checkpoint_value.get("minimum_shot_neighbors", 4)
            if minimum_shot_neighbors is None
            else minimum_shot_neighbors
        )
        if not 1 <= self.minimum_shot_neighbors <= 7:
            raise ValueError("minimum_shot_neighbors must be in [1, 7]")
        self.exact_retrievals = 0
        self.ood_retrievals = 0

    def predict_word(self, observation: Mapping[str, Any]) -> int:
        digest = state_digest(observation)
        if digest in self.actions:
            self.exact_retrievals += 1
            return int(self.actions[digest])
        self.ood_retrievals += 1
        query = (
            features(
                observation,
                combo_aware=self.feature_count == COMBO_FEATURE_COUNT,
            )
            - self.mean
        ) / self.scale
        distance = np.square(self.exemplar_features - query).mean(axis=1)
        # Seven-neighbour vote makes the unseen-state path intentionally conservative.
        indices = np.argpartition(distance, min(6, len(distance) - 1))[:7]
        words = self.exemplar_words[indices]
        shots = words[words != 0]
        if len(shots) < self.minimum_shot_neighbors:
            return 0
        shot_indices = indices[words != 0]
        return int(self.exemplar_words[shot_indices[np.argmin(distance[shot_indices])]])


def collect_and_train(
    run_root: Path,
    runtime: ExactTrainingRuntime,
    *,
    trace_path: Path,
    source_seed: int,
    expected_score: int,
    maximum_ticks: int = 100_000,
    teacher_action_lineage: str = "external source trace; inspect its manifest",
) -> dict[str, object]:
    words = [word for (word,) in struct.iter_unpack("<I", trace_path.read_bytes())]
    state_actions: dict[str, int] = {}
    exemplar_x: list[np.ndarray] = []
    exemplar_y: list[int] = []
    collisions: Counter[str] = Counter()
    started = time.monotonic()
    step_info: Mapping[str, Any] = {}
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum_ticks}) as session:
        env = session.environment
        observation, info = env.reset(seed=source_seed)
        if int(info["seed"]) != source_seed:
            raise RuntimeError("source exact seed differs")
        for index, word in enumerate(words):
            digest = state_digest(observation)
            prior = state_actions.setdefault(digest, word)
            if prior != word:
                collisions[f"{prior}->{word}"] += 1
            # Keep every shot and a deterministic 1/16 sample of waits for OOD fitting.
            if word or index % 16 == 0:
                exemplar_x.append(features(observation))
                exemplar_y.append(word)
            observation, _reward, terminated, truncated, step_info = env.step(decode(word))
            if terminated or truncated:
                if index + 1 != len(words):
                    raise RuntimeError("source exact trace terminated before its declared end")
                break
        final = checkpoint(observation)
        canonical_final = canonical_checkpoint(observation, step_info)
        provenance = session.provenance_manifest
    if collisions:
        raise RuntimeError(f"physical-state action collisions: {dict(collisions)}")
    if (
        not final["terminated"]
        or final["truncated"]
        or canonical_final["score"] != expected_score
    ):
        raise RuntimeError("source trajectory exact reconstruction differs")
    matrix = np.stack(exemplar_x)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    checkpoint_value = {
        "schema": "irisu-exact-reactive-public-state-knn-v1",
        "physics_backend": "exact",
        "state_producing_backends": ("exact",),
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": teacher_action_lineage,
        "inference_inputs": "public physical state; seed/tick/score/counters excluded",
        "feature_schema": "public-combo-topology-v2",
        "state_actions": state_actions,
        "mean": torch.from_numpy(mean),
        "scale": torch.from_numpy(scale),
        "exemplar_features": torch.from_numpy(normalized),
        "exemplar_words": torch.tensor(exemplar_y, dtype=torch.int64),
        "source_seed_recorded_for_audit_only": source_seed,
        "source_trace_sha256": sha256_file(trace_path),
        "maximum_ticks": maximum_ticks,
    }
    checkpoint_path = run_root / "checkpoints/exact-reactive-state-knn.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint_value, checkpoint_path)
    manifest = {
        "schema": "irisu-exact-reactive-training-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": teacher_action_lineage,
        "source_seed": source_seed,
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256_file(trace_path),
        "training_states": len(state_actions),
        "ood_exemplars": len(exemplar_y),
        "shot_exemplars": sum(value != 0 for value in exemplar_y),
        "feature_count": int(matrix.shape[1]),
        "excluded_inputs": sorted(_EPISODE_ONLY | _BODY_IDENTITY_OR_CLOCK | {"seed"}),
        "checkpoint": str(checkpoint_path.relative_to(run_root)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_final": final,
        "source_canonical_final": canonical_final,
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    (run_root / "training.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def load_policy(
    run_root: Path, *, minimum_shot_neighbors: int | None = None
) -> ReactiveExactPolicy:
    manifest = json.loads((run_root / "training.json").read_text())
    path = run_root / manifest["checkpoint"]
    if sha256_file(path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("reactive checkpoint bytes differ")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("physics_backend") != "exact" or value.get("portable_checkpoint_loaded") is not False:
        raise RuntimeError("reactive checkpoint provenance differs")
    return ReactiveExactPolicy(
        value, minimum_shot_neighbors=minimum_shot_neighbors
    )


def evaluate_seed(
    run_root: Path,
    runtime: ExactTrainingRuntime,
    seed: int,
    *,
    source: bool,
    maximum_ticks: int = 100_000,
    minimum_shot_neighbors: int | None = None,
) -> dict[str, object]:
    policy = load_policy(
        run_root, minimum_shot_neighbors=minimum_shot_neighbors
    )
    actions: list[int] = []
    started = time.monotonic()
    step_info: Mapping[str, Any] = {}
    with runtime.open_env(simulation_config={"max_episode_ticks": maximum_ticks}) as session:
        env = session.environment
        observation, info = env.reset(seed=seed)
        if int(info["seed"]) != seed:
            raise RuntimeError("evaluation exact seed differs")
        terminated = truncated = False
        while not (terminated or truncated):
            word = policy.predict_word(observation)
            actions.append(word)
            observation, _reward, terminated, truncated, step_info = env.step(decode(word))
        final = checkpoint(observation)
        canonical_final = canonical_checkpoint(observation, step_info)
        state_u64 = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    trace = b"".join(WORD.pack(word) for word in actions)
    replay = REPLAY_HEADER.pack(
        seed, int(canonical_final["level"]), int(canonical_final["score"]),
        int(canonical_final["highest_chain"]), 0,
    ) + bytes(32) + trace
    label = "source" if source else f"holdout-{seed:08x}"
    if minimum_shot_neighbors is not None:
        label += f"-vote{policy.minimum_shot_neighbors}"
    trace_path = run_root / "evaluations" / f"{label}.u32le"
    replay_path = run_root / "evaluations" / f"{label}.rpy"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_bytes(trace)
    replay_path.write_bytes(replay)
    result = {
        "schema": "irisu-exact-reactive-evaluation-v1",
        "physics_backend": "exact",
        "seed": seed,
        "split": label,
        "natural_terminal": bool(final["terminated"]) and not bool(final["truncated"]),
        "score": int(canonical_final["score"]),
        "live_score": int(final["score"]),
        "final": final,
        "canonical_final": canonical_final,
        "final_state_u64": state_u64,
        "exact_state_retrievals": policy.exact_retrievals,
        "ood_retrievals": policy.ood_retrievals,
        "minimum_shot_neighbors": policy.minimum_shot_neighbors,
        "trace": str(trace_path.relative_to(run_root)),
        "trace_sha256": hashlib.sha256(trace).hexdigest(),
        "replay": str(replay_path.relative_to(run_root)),
        "replay_sha256": hashlib.sha256(replay).hexdigest(),
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    (run_root / "evaluations" / f"{label}.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source-trace", type=Path, default=TRACE)
    parser.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    parser.add_argument("--expected-score", type=int, default=50_663)
    parser.add_argument(
        "--maximum-ticks", type=int,
        help="exact episode cap; defaults to at least one tick beyond the trace",
    )
    parser.add_argument("--holdouts", type=int, default=len(HOLDOUT_SEEDS))
    parser.add_argument(
        "--teacher-action-lineage",
        default="external source trace; inspect its manifest",
    )
    args = parser.parse_args()
    if args.run_root.exists():
        parser.error("--run-root must not exist")
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    args.run_root.mkdir(parents=True)
    trace_path = args.source_trace.resolve(strict=True)
    action_count = trace_path.stat().st_size // WORD.size
    maximum_ticks = (
        args.maximum_ticks
        if args.maximum_ticks is not None
        else max(100_000, action_count + 1)
    )
    if not 0 <= args.source_seed <= 0xFFFFFFFF or args.expected_score < 0:
        parser.error("source seed/score are outside their valid ranges")
    if trace_path.stat().st_size % WORD.size or maximum_ticks < action_count:
        parser.error("--maximum-ticks must cover the aligned source trace")
    training = collect_and_train(
        args.run_root,
        runtime,
        trace_path=trace_path,
        source_seed=args.source_seed,
        expected_score=args.expected_score,
        maximum_ticks=maximum_ticks,
        teacher_action_lineage=args.teacher_action_lineage,
    )
    source = evaluate_seed(
        args.run_root, runtime, args.source_seed, source=True,
        maximum_ticks=maximum_ticks,
    )
    holdouts = [
        evaluate_seed(
            args.run_root, runtime, seed, source=False,
            maximum_ticks=maximum_ticks,
        )
        for seed in HOLDOUT_SEEDS[: args.holdouts]
    ]
    evaluator = ROOT / "tools/evaluate-rpy.py"
    completed = subprocess.run(
        [
            sys.executable, str(evaluator), str(args.run_root / source["replay"]),
            "--worker", str(runtime.worker_path), "--purpose", "target", "--compact",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    acceptance = json.loads(completed.stdout)
    (args.run_root / "source-replay-acceptance.json").write_text(
        json.dumps(acceptance, sort_keys=True, indent=2) + "\n"
    )
    summary = {
        "schema": "irisu-exact-reactive-distillation-summary-v1",
        "physics_backend": "exact",
        "portable_checkpoint_loaded": False,
        "teacher_action_lineage": training["teacher_action_lineage"],
        "policy_input": "public physical state without seed/tick/score/cumulative counters",
        "source_score": source["score"],
        "maximum_ticks": maximum_ticks,
        "source_natural_terminal": source["natural_terminal"],
        "source_exact_retrieval_fraction": source["exact_state_retrievals"] / source["final"]["tick"],
        "source_replay_accepted": acceptance.get("status", {}).get("accepted") is True,
        "holdout_scores": [item["score"] for item in holdouts],
        "holdout_natural_terminals": [item["natural_terminal"] for item in holdouts],
        "holdout_ood_retrieval_fractions": [
            item["ood_retrievals"] / item["final"]["tick"] for item in holdouts
        ],
        "generalization_claim": (
            "source trajectory score is verified; unseen-seed performance is "
            "claimed only for measured holdouts"
        ),
        "checkpoint": training["checkpoint"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "source_replay": source["replay"],
        "source_replay_sha256": source["replay_sha256"],
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
