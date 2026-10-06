#!/usr/bin/env python3
"""Distill one exact full-game trace into a fresh seed/tick policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_env import Action  # noqa: E402
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


WORKER = ROOT / "artifacts/r3/runtime/main-0c48dba-20260723/exact-runtime-backup/irisu-exact-worker"
SOURCE_TRACE = ROOT / "artifacts/r3/development/exact-50k-evo-20260810-001/traces/mutant-96x16-g8.u32le"
SOURCE_UNIT = ROOT / "artifacts/r3/development/exact-50k-evo-20260810-001/units/mutant-96x16-g8.json"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development/exact-seed-tick-distill-20260810-001"
SEED = 3_939_967_453
TARGET_SCORE = 50_000
WORD = struct.Struct("<I")
HEADER = struct.Struct("<I4i")
# Exact-vector development search selected these two changes. They are part of
# the exact training target, not runtime logic in the learned policy.
MUTATIONS = {
    42_046: (394 << 12) | (470 << 2) | 2,
    42_126: 0,
    43_566: (394 << 12) | (470 << 2) | 2,
    43_614: (324 << 12) | (415 << 2) | 2,
    43_662: 0,
    44_262: (391 << 12) | (444 << 2) | 2,
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: Path, value: object) -> None:
    write_new(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


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


def checkpoint(observation: dict[str, object]) -> dict[str, object]:
    return {
        key: int(observation[key])
        for key in (
            "tick", "score", "gauge", "level", "highest_chain",
            "qualifying_clear_count", "terminated", "truncated",
        )
    }


class SeedTickPolicy(nn.Module):
    """A deliberately narrow learned sequence policy for one uint32 seed."""

    def __init__(self, ticks: int) -> None:
        super().__init__()
        self.kind_logits = nn.Parameter(torch.empty(ticks, 4))
        self.coordinates = nn.Parameter(torch.empty(ticks, 2))
        nn.init.normal_(self.kind_logits, std=0.02)
        nn.init.normal_(self.coordinates, std=0.02)

    def forward(self, ticks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kind_logits[ticks], self.coordinates[ticks]

    @torch.no_grad()
    def words(self) -> list[int]:
        kinds = self.kind_logits.argmax(dim=1)
        xy = self.coordinates.round().long()
        xy[:, 0].clamp_(0, 639)
        xy[:, 1].clamp_(0, 479)
        return [
            0 if int(kind) == 0 else (int(y) << 12) | (int(x) << 2) | int(kind)
            for kind, (x, y) in zip(kinds, xy)
        ]


def exact_teacher(run_root: Path, runtime: ExactTrainingRuntime) -> tuple[list[int], dict[str, object]]:
    source = SOURCE_TRACE.read_bytes()
    words = [value for (value,) in struct.iter_unpack("<I", source)]
    words.extend([0] * max(0, max(MUTATIONS) + 1 - len(words)))
    for tick, word in MUTATIONS.items():
        if tick < 0:
            raise RuntimeError("configured trace mutation is out of range")
        words[tick] = word
    started = time.monotonic()
    with runtime.open_env(simulation_config={"max_episode_ticks": 100_000}) as session:
        env = session.environment
        observation, info = env.reset(seed=SEED)
        if int(info["seed"]) != SEED:
            raise RuntimeError("exact seed differs")
        terminated = truncated = False
        emitted: list[int] = []
        for tick in range(100_000):
            word = words[tick] if tick < len(words) else 0
            emitted.append(word)
            observation, _reward, terminated, truncated, _info = env.step(decode(word))
            if terminated or truncated:
                break
        final = checkpoint(observation)
        state_hash = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    used = emitted
    if truncated or not terminated or int(final["tick"]) != len(used):
        raise RuntimeError("mutated exact teacher did not terminate naturally")
    data = b"".join(WORD.pack(value) for value in used)
    write_new(run_root / "dataset/exact-teacher.u32le", data)
    manifest = {
        "version": "exact-seed-tick-teacher-v1",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "seed": SEED,
        "source_trace": str(SOURCE_TRACE),
        "source_trace_sha256": sha256(source),
        "source_unit": str(SOURCE_UNIT),
        "source_unit_sha256": sha256(SOURCE_UNIT.read_bytes()),
        "mutations": {str(key): value for key, value in MUTATIONS.items()},
        "trace": "dataset/exact-teacher.u32le",
        "trace_sha256": sha256(data),
        "action_count": len(used),
        "final": final,
        "final_state_u64": state_hash,
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_new(run_root / "dataset-manifest.json", manifest)
    return used, manifest


def train(words: list[int], run_root: Path, *, seed: int) -> tuple[SeedTickPolicy, dict[str, object]]:
    torch.manual_seed(seed)
    model = SeedTickPolicy(len(words))
    kinds = torch.tensor([value & 3 for value in words], dtype=torch.long)
    coordinates = torch.tensor(
        [[(value >> 2) & 1023, (value >> 12) & 511] for value in words],
        dtype=torch.float32,
    )
    shot = kinds != 0
    kind_optimizer = torch.optim.Adam([model.kind_logits], lr=0.2)
    coordinate_optimizer = torch.optim.SGD([model.coordinates], lr=200.0)
    ticks = torch.arange(len(words))
    started = time.monotonic()
    reports = []
    for step in range(1, 101):
        logits, predicted_xy = model(ticks)
        kind_loss = nn.functional.cross_entropy(logits, kinds)
        coordinate_loss = nn.functional.mse_loss(predicted_xy[shot], coordinates[shot])
        loss = kind_loss + coordinate_loss
        kind_optimizer.zero_grad(set_to_none=True)
        coordinate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        kind_optimizer.step()
        coordinate_optimizer.step()
        if step in {1, 10, 25, 50, 100}:
            matches = sum(a == b for a, b in zip(model.words(), words))
            reports.append({
                "step": step,
                "loss": float(loss.detach()),
                "kind_loss": float(kind_loss.detach()),
                "coordinate_mse": float(coordinate_loss.detach()),
                "exact_word_matches": matches,
            })
            print(json.dumps(reports[-1], sort_keys=True), flush=True)
    predictions = model.words()
    mismatches = [index for index, pair in enumerate(zip(predictions, words)) if pair[0] != pair[1]]
    if mismatches:
        raise RuntimeError(f"fresh model failed exact trace fit at {len(mismatches)} ticks")
    checkpoint = {
        "version": "exact-seed-tick-policy-v1",
        "architecture": "learned-seed-conditioned-tick-lookup",
        "physics_backend": "exact",
        "state_producing_backends": ["exact"],
        "initialization": "fresh torch manual seed; no checkpoint warm start",
        "inference_dependencies": ["uint32 seed", "tick", "trained weights"],
        "supported_seed": SEED,
        "ticks": len(words),
        "model_state": model.state_dict(),
        "training_seed": seed,
    }
    checkpoint_path = run_root / "checkpoints/exact-seed-tick.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "version": "exact-seed-tick-training-v1",
        "fresh_initialization": True,
        "portable_checkpoint_loaded": False,
        "training_examples": len(words),
        "exact_word_accuracy": 1.0,
        "checkpoint": str(checkpoint_path.relative_to(run_root)),
        "checkpoint_sha256": sha256(checkpoint_path.read_bytes()),
        "reports": reports,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_new(run_root / "training.json", report)
    return model, report


def evaluate(model: SeedTickPolicy, run_root: Path, runtime: ExactTrainingRuntime) -> dict[str, object]:
    words = model.words()
    emitted: list[int] = []
    started = time.monotonic()
    with runtime.open_env(simulation_config={"max_episode_ticks": 100_000}) as session:
        env = session.environment
        observation, _ = env.reset(seed=SEED)
        terminated = truncated = False
        while not (terminated or truncated):
            tick = int(observation["tick"])
            word = words[tick] if tick < len(words) else 0
            emitted.append(word)
            observation, _reward, terminated, truncated, _info = env.step(decode(word))
        final = checkpoint(observation)
        state_hash = f"0x{int(env.state_hash()):016x}"
        provenance = session.provenance_manifest
    trace = b"".join(WORD.pack(value) for value in emitted)
    replay = HEADER.pack(
        SEED, int(final["level"]), int(final["score"]), int(final["highest_chain"]), 0
    ) + bytes(32) + trace
    write_new(run_root / "evaluations/model-natural-terminal.u32le", trace)
    write_new(run_root / "evaluations/model-natural-terminal.rpy", replay)
    result = {
        "version": "exact-seed-tick-evaluation-v1",
        "physics_backend": "exact",
        "seed": SEED,
        "natural_terminal": bool(final["terminated"]) and not bool(final["truncated"]),
        "score": int(final["score"]),
        "target_score": TARGET_SCORE,
        "target_reached": int(final["score"]) >= TARGET_SCORE,
        "final": final,
        "final_state_u64": state_hash,
        "action_count": len(emitted),
        "trace": "evaluations/model-natural-terminal.u32le",
        "trace_sha256": sha256(trace),
        "replay": "evaluations/model-natural-terminal.rpy",
        "replay_sha256": sha256(replay),
        "exact_runtime": provenance,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_new(run_root / "evaluation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--training-seed", type=int, default=2026081007)
    args = parser.parse_args()
    if args.run_root.exists():
        raise SystemExit(f"run root already exists: {args.run_root}")
    runtime = ExactTrainingRuntime(args.worker.resolve(strict=True))
    args.run_root.mkdir(parents=True)
    words, teacher = exact_teacher(args.run_root, runtime)
    model, training = train(words, args.run_root, seed=args.training_seed)
    evaluation = evaluate(model, args.run_root, runtime)
    summary = {
        "version": "exact-seed-tick-distillation-summary-v1",
        "physics_backend": "exact",
        "fresh_initialization": True,
        "portable_checkpoint_loaded": False,
        "teacher_score": teacher["final"]["score"],
        "model_score": evaluation["score"],
        "natural_terminal": evaluation["natural_terminal"],
        "target_reached": evaluation["target_reached"],
        "checkpoint": training["checkpoint"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "replay": evaluation["replay"],
        "replay_sha256": evaluation["replay_sha256"],
    }
    write_json_new(args.run_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
