#!/usr/bin/env python3
"""Exact joint-search teacher, imitation, and replay-verified full game."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3k_sustainable_v3 as screen  # noqa: E402
import rl_r3m_shot_restraint as replay_helpers  # noqa: E402
from irisu_pointer.joint_planner import (  # noqa: E402
    JointPairGeometrySearch,
    JointPlannerConfig,
    SteeringDecision,
    _commit_base_decision,
)
from irisu_rl.exact_training_runtime import ExactTrainingRuntime  # noqa: E402


CONFIG_PATH = ROOT / "configs/rl/experiments/exact-50k-search-v1.toml"
DEFAULT_RUN_ROOT = (
    ROOT / "artifacts/r3/development/exact-50k-search-20260810-001"
)
ACTION_WORD = struct.Struct("<I")
REPLAY_HEADER = struct.Struct("<I4i")
FEATURE_NAMES = (
    "tick_100k",
    "score_50k",
    "gauge_fraction",
    "level_50",
    "highest_chain_10",
    "piece_count_100",
    "fresh_count_50",
    "confirmed_count_50",
    "rotten_count_50",
    "source_x",
    "source_y",
    "destination_x",
    "destination_y",
    "delta_x",
    "delta_y",
    "same_color",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def with_sha(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def verify_sha(value: Mapping[str, object], label: str) -> None:
    supplied = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if supplied != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise RuntimeError(f"{label} self-hash differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: Path, value: Mapping[str, object]) -> None:
    write_new(
        path,
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
        + b"\n",
    )


def load_config() -> dict[str, Any]:
    value = tomllib.loads(CONFIG_PATH.read_text())
    if (
        value.get("physics_backend") != "exact"
        or value.get("state_producing_backends") != ["exact"]
    ):
        raise RuntimeError("exact search config permits a non-exact backend")
    return value


def planner_config(config: Mapping[str, Any]) -> JointPlannerConfig:
    teacher = config["teacher"]
    return JointPlannerConfig(
        pair_cap=int(teacher["pair_cap"]),
        geometry_cap=int(teacher["geometry_cap"]),
        horizons=tuple(int(value) for value in teacher["rollout_horizons"]),
        cooldown_ticks=int(teacher["cooldown_ticks"]),
    )


def source_identity() -> dict[str, object]:
    config = load_config()
    worker = Path(config["runtime"]["worker_path"])
    checkpoint = Path(config["warm_start"]["checkpoint"])
    core, campaign = screen._load_external()
    files = (
        Path(__file__).resolve(),
        CONFIG_PATH,
        worker,
        checkpoint,
        ROOT / "python/irisu_rl/exact_training_runtime.py",
        ROOT / "python/irisu_pointer/joint_planner.py",
        Path(screen.__file__).resolve(),
        Path(replay_helpers.__file__).resolve(),
        Path(core.__file__).resolve(),
        Path(campaign.__file__).resolve(),
    )
    identities = {str(path): sha256_file(path) for path in files}
    if identities[str(worker)] != config["runtime"]["worker_sha256"]:
        raise RuntimeError("configured exact worker bytes differ")
    if identities[str(checkpoint)] != config["warm_start"]["checkpoint_sha256"]:
        raise RuntimeError("configured warm-start bytes differ")
    return with_sha(
        {
            "schema": "irisu-exact-50k-search-source-v1",
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "promotion_eligible": True,
            "warm_start_role": config["warm_start"]["role"],
            "files": identities,
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(run_root)
    identity = source_identity()
    config = load_config()
    plan = with_sha(
        {
            "schema": "irisu-exact-50k-search-plan-v1",
            "source_identity_sha256": identity["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "teacher": dict(config["teacher"]),
            "imitation": dict(config["imitation"]),
            "full_game": dict(config["full_game"]),
            "pipeline": [
                "exact joint-search teacher collection",
                "supervised imitation checkpoint",
                "imitation-guided exact branch validation",
                "exact full-game replay re-execution",
            ],
        }
    )
    run_root.mkdir(parents=True)
    write_json_new(run_root / "source-identity.json", identity)
    write_json_new(run_root / "plan.json", plan)
    return plan


def validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = read_json(run_root / "source-identity.json")
    plan = read_json(run_root / "plan.json")
    verify_sha(identity, "source identity")
    verify_sha(plan, "plan")
    if identity != source_identity() or plan["source_identity_sha256"] != identity["sha256"]:
        raise RuntimeError("exact search source identity changed")
    return identity, plan


def _body_map(observation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(body["id"]): body
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and body.get("kind") == "piece"
    }


def features(
    observation: Mapping[str, Any], incumbent: SteeringDecision
) -> list[float]:
    bodies = _body_map(observation)
    source = bodies.get(int(incumbent.source_body_id or -1), {})
    destination = bodies.get(int(incumbent.destination_body_id or -1), {})

    def coordinate(body: Mapping[str, Any], name: str, scale: float) -> float:
        return float(body.get(name, 0.0)) / scale

    sx, sy = coordinate(source, "x", 640.0), coordinate(source, "y", 480.0)
    dx = coordinate(destination, "x", 640.0)
    dy = coordinate(destination, "y", 480.0)
    lifecycles = [str(body.get("lifecycle", "")) for body in bodies.values()]
    gauge_max = max(int(observation.get("gauge_max", 1)), 1)
    return [
        int(observation.get("tick", 0)) / 100_000.0,
        int(observation.get("score", 0)) / 50_000.0,
        int(observation.get("gauge", 0)) / gauge_max,
        int(observation.get("level", 0)) / 50.0,
        int(observation.get("highest_chain", 0)) / 10.0,
        len(bodies) / 100.0,
        sum(value in {"scripted_falling", "dynamic_fresh", "falling", "fresh"} for value in lifecycles) / 50.0,
        lifecycles.count("confirmed") / 50.0,
        lifecycles.count("rotten") / 50.0,
        sx,
        sy,
        dx,
        dy,
        dx - sx,
        dy - sy,
        float(source.get("color", -1) == destination.get("color", -2)),
    ]


def _searcher(campaign: object, config: Mapping[str, Any]) -> JointPairGeometrySearch:
    return JointPairGeometrySearch(
        campaign.POLICY_FACTORY,
        config=planner_config(config),
    )


def _winner(outcomes: Sequence[Any]) -> Any:
    incumbent = outcomes[0]
    eligible = [value for value in outcomes if value.selectable_against(incumbent)]
    winner = max(
        eligible,
        key=lambda value: (value.objective, -value.candidate.ordinal),
    )
    return winner if winner.objective > incumbent.objective else incumbent


def _teacher_query(
    env: Any,
    observation: Mapping[str, Any],
    policy: object,
    incumbent: SteeringDecision,
    searcher: JointPairGeometrySearch,
) -> tuple[SteeringDecision, dict[str, object]]:
    candidates = searcher._candidates(observation, incumbent)
    started = time.monotonic()
    outcomes = []
    with env.fast_checkpoint() as checkpoint:
        for candidate in candidates:
            with checkpoint.branch() as branch:
                outcomes.append(searcher._evaluate(branch, observation, candidate))
    selected = _winner(outcomes)
    decision = selected.candidate.decision
    rebound = _commit_base_decision(policy, observation, incumbent, decision)
    if not rebound:
        selected = outcomes[0]
        decision = incumbent
    return decision, {
        "tick": int(observation["tick"]),
        "source_state_hash": f"0x{int(env.state_hash()):016x}",
        "features": features(observation, incumbent),
        "candidate_count": len(candidates),
        "candidates": [candidate.manifest() for candidate in candidates],
        "selected_ordinal": int(selected.candidate.ordinal),
        "strictly_improved": selected is not outcomes[0],
        "rebind_succeeded": bool(rebound),
        "outcomes": [outcome.manifest() for outcome in outcomes],
        "wall_seconds": time.monotonic() - started,
    }


def _step_decision(
    core: object,
    env: Any,
    observation: Mapping[str, Any],
    decision: object,
    *,
    maximum_tick: int,
    action_sink: list[int] | None = None,
) -> tuple[Mapping[str, Any], bool, bool]:
    current = observation
    terminated = truncated = False
    for action in screen._primitive_actions(core, decision):
        kind = core.JOINT.ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
        duration = min(duration, maximum_tick - int(current["tick"]))
        for _ in range(duration):
            primitive = core.JOINT.Action.wait(1) if kind is core.JOINT.ActionKind.WAIT else action
            if action_sink is not None:
                action_sink.append(replay_helpers.encode_action(primitive))
            current, _reward, terminated, truncated, _info = env.step(primitive)
            if terminated or truncated or int(current["tick"]) >= maximum_tick:
                return current, bool(terminated), bool(truncated)
    return current, bool(terminated), bool(truncated)


def collect(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    output = run_root / "teacher-dataset.json"
    if output.exists():
        value = read_json(output)
        verify_sha(value, "teacher dataset")
        return value
    config = load_config()
    teacher = config["teacher"]
    worker = Path(config["runtime"]["worker_path"])
    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    seed = int(teacher["seed"])
    policy.reset(seed)
    searcher = _searcher(campaign, config)
    horizon = int(teacher["horizon_ticks"])
    rows: list[dict[str, object]] = []
    shot_count = 0
    started = time.monotonic()
    with ExactTrainingRuntime(worker).open_env(
        simulation_config={"max_episode_ticks": horizon + max(searcher.config.horizons)}
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while int(observation["tick"]) < horizon and not (terminated or truncated):
            incumbent = policy.predict(observation)
            decision = incumbent
            if isinstance(incumbent, SteeringDecision) and incumbent.is_shot:
                shot_count += 1
                if (
                    (shot_count - 1) % int(teacher["query_stride_shots"]) == 0
                    and len(rows) < int(teacher["maximum_queries"])
                ):
                    decision, row = _teacher_query(
                        env, observation, policy, incumbent, searcher
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "phase": "teacher",
                                "query": len(rows),
                                "tick": row["tick"],
                                "selected": row["selected_ordinal"],
                                "score": int(observation["score"]),
                                "gauge": int(observation["gauge"]),
                                "seconds": row["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            observation, terminated, truncated = _step_decision(
                core, env, observation, decision, maximum_tick=horizon
            )
        final_snapshot = hashlib.sha256(env.clone_state()).hexdigest()
        final = replay_helpers.checkpoint(env, observation)
        provenance = session.provenance_manifest
    dataset = with_sha(
        {
            "schema": "irisu-exact-joint-search-imitation-dataset-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "seed": seed,
            "feature_names": list(FEATURE_NAMES),
            "planner": searcher.identity_manifest(),
            "query_count": len(rows),
            "rows": rows,
            "final": final,
            "final_snapshot_sha256": final_snapshot,
            "exact_runtime": provenance,
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_json_new(output, dataset)
    return dataset


class ImitationModel(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.Tanh(),
            nn.Linear(width, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def train(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    manifest_path = run_root / "imitation-checkpoint.json"
    if manifest_path.exists():
        value = read_json(manifest_path)
        verify_sha(value, "imitation checkpoint manifest")
        return value
    dataset = collect(run_root)
    config = load_config()
    imitation = config["imitation"]
    classes = int(config["teacher"]["pair_cap"]) * int(
        config["teacher"]["geometry_cap"]
    )
    inputs = torch.tensor(
        [row["features"] for row in dataset["rows"]], dtype=torch.float32
    )
    targets = torch.tensor(
        [row["selected_ordinal"] for row in dataset["rows"]], dtype=torch.long
    )
    mean = inputs.mean(dim=0)
    scale = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    torch.manual_seed(20260810)
    model = ImitationModel(int(imitation["hidden_width"]), classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(imitation["learning_rate"]),
        weight_decay=float(imitation["weight_decay"]),
    )
    for _ in range(int(imitation["epochs"])):
        logits = model((inputs - mean) / scale)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model((inputs - mean) / scale)
        accuracy = float((logits.argmax(dim=1) == targets).float().mean())
        final_loss = float(nn.functional.cross_entropy(logits, targets))
    checkpoint_path = run_root / "exact-imitation.pt"
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema": "irisu-exact-search-imitation-checkpoint-v1",
            "state_dict": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "feature_names": FEATURE_NAMES,
            "width": int(imitation["hidden_width"]),
            "classes": classes,
            "teacher_dataset_sha256": dataset["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ("exact",),
        },
        temporary,
    )
    os.link(temporary, checkpoint_path)
    temporary.unlink()
    manifest = with_sha(
        {
            "schema": "irisu-exact-search-imitation-checkpoint-manifest-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "teacher_dataset_sha256": dataset["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "checkpoint": str(checkpoint_path.relative_to(run_root)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_examples": len(targets),
            "training_accuracy": accuracy,
            "training_loss": final_loss,
        }
    )
    write_json_new(manifest_path, manifest)
    return manifest


def load_model(run_root: Path) -> tuple[ImitationModel, torch.Tensor, torch.Tensor, dict[str, Any]]:
    manifest = train(run_root)
    path = run_root / str(manifest["checkpoint"])
    if sha256_file(path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("imitation checkpoint bytes changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("physics_backend") != "exact"
        or tuple(value.get("state_producing_backends", ())) != ("exact",)
        or tuple(value.get("feature_names", ())) != FEATURE_NAMES
    ):
        raise RuntimeError("imitation checkpoint provenance differs")
    model = ImitationModel(int(value["width"]), int(value["classes"]))
    model.load_state_dict(value["state_dict"])
    model.eval()
    return model, value["mean"], value["scale"], manifest


def _student_query(
    env: Any,
    observation: Mapping[str, Any],
    policy: object,
    incumbent: SteeringDecision,
    searcher: JointPairGeometrySearch,
    model: ImitationModel,
    mean: torch.Tensor,
    scale: torch.Tensor,
    top_k: int,
) -> tuple[SteeringDecision, dict[str, object]]:
    candidates = searcher._candidates(observation, incumbent)
    encoded = torch.tensor([features(observation, incumbent)], dtype=torch.float32)
    with torch.no_grad():
        logits = model((encoded - mean) / scale)[0]
    predicted = [
        int(value)
        for value in logits.argsort(descending=True).tolist()
        if int(value) < len(candidates)
    ][:top_k]
    ordinals = [0, *(value for value in predicted if value != 0)]
    started = time.monotonic()
    outcomes = []
    with env.fast_checkpoint() as checkpoint:
        for ordinal in ordinals:
            with checkpoint.branch() as branch:
                outcomes.append(
                    searcher._evaluate(branch, observation, candidates[ordinal])
                )
    selected = _winner(outcomes)
    decision = selected.candidate.decision
    rebound = _commit_base_decision(policy, observation, incumbent, decision)
    if not rebound:
        selected = outcomes[0]
        decision = incumbent
    return decision, {
        "tick": int(observation["tick"]),
        "source_state_hash": f"0x{int(env.state_hash()):016x}",
        "features": features(observation, incumbent),
        "candidate_count": len(candidates),
        "predicted_ordinals": predicted,
        "evaluated_ordinals": ordinals,
        "selected_ordinal": int(selected.candidate.ordinal),
        "strictly_improved": selected is not outcomes[0],
        "rebind_succeeded": bool(rebound),
        "outcomes": [outcome.manifest() for outcome in outcomes],
        "wall_seconds": time.monotonic() - started,
    }


def play(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    output = run_root / "full-game.json"
    if output.exists():
        value = read_json(output)
        verify_sha(value, "full game")
        return value
    config = load_config()
    full = config["full_game"]
    worker = Path(config["runtime"]["worker_path"])
    model, mean, scale, checkpoint_manifest = load_model(run_root)
    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    seed = int(full["seed"])
    policy.reset(seed)
    searcher = _searcher(campaign, config)
    maximum_tick = int(full["maximum_ticks"])
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    shot_count = 0
    started = time.monotonic()
    with ExactTrainingRuntime(worker).open_env(
        simulation_config={"max_episode_ticks": maximum_tick + max(searcher.config.horizons)}
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=seed)
        checkpoints.append(replay_helpers.checkpoint(env, observation))
        terminated = truncated = False
        for _ in range(int(full["replay_warmup_ticks"])):
            action = core.JOINT.Action.wait(1)
            actions.append(replay_helpers.encode_action(action))
            observation, _reward, terminated, truncated, _info = env.step(action)
        next_checkpoint = int(full["checkpoint_stride_ticks"])
        while int(observation["tick"]) < maximum_tick and not (terminated or truncated):
            incumbent = policy.predict(observation)
            decision = incumbent
            if isinstance(incumbent, SteeringDecision) and incumbent.is_shot:
                shot_count += 1
                remaining = maximum_tick - int(observation["tick"])
                if (
                    (shot_count - 1) % int(full["query_stride_shots"]) == 0
                    and len(queries) < int(full["maximum_queries"])
                    and remaining >= max(searcher.config.horizons)
                ):
                    decision, query = _student_query(
                        env,
                        observation,
                        policy,
                        incumbent,
                        searcher,
                        model,
                        mean,
                        scale,
                        int(config["imitation"]["student_top_k"]),
                    )
                    queries.append(query)
                    print(
                        json.dumps(
                            {
                                "phase": "full-game-search",
                                "query": len(queries),
                                "tick": query["tick"],
                                "selected": query["selected_ordinal"],
                                "score": int(observation["score"]),
                                "gauge": int(observation["gauge"]),
                                "seconds": query["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            observation, terminated, truncated = _step_decision(
                core,
                env,
                observation,
                decision,
                maximum_tick=maximum_tick,
                action_sink=actions,
            )
            if int(observation["tick"]) >= next_checkpoint:
                checkpoints.append(replay_helpers.checkpoint(env, observation))
                next_checkpoint += int(full["checkpoint_stride_ticks"])
                print(
                    json.dumps(
                        {
                            "phase": "full-game",
                            "tick": int(observation["tick"]),
                            "score": int(observation["score"]),
                            "gauge": int(observation["gauge"]),
                            "queries": len(queries),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        final = replay_helpers.checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "full-game.u32le"
    write_new(trace_path, trace)
    replay = (
        REPLAY_HEADER.pack(
            seed,
            int(final["level"]),
            int(final["score"]),
            int(final["highest_chain"]),
            0,
        )
        + bytes(32)
        + trace
    )
    replay_path = run_root / "best-exact-search.rpy"
    write_new(replay_path, replay)
    result = with_sha(
        {
            "schema": "irisu-exact-50k-search-full-game-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "physics_backend": "exact",
            "state_producing_backends": ["exact"],
            "promotion_eligible": True,
            "seed": seed,
            "terminal": bool(final["terminated"]),
            "censored": int(final["tick"]) >= maximum_tick and not bool(final["terminated"]),
            "target_score": int(full["target_score"]),
            "target_reached": int(final["score"]) >= int(full["target_score"]),
            "score": int(final["score"]),
            "survival_ticks": int(final["tick"]),
            "level": int(final["level"]),
            "highest_chain": int(final["highest_chain"]),
            "final_gauge": int(final["gauge"]),
            "clears": int(final["clears"]),
            "shot_count": shot_count,
            "query_count": len(queries),
            "strict_improvements": sum(bool(row["strictly_improved"]) for row in queries),
            "queries": queries,
            "imitation_checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "replay_file": str(replay_path.relative_to(run_root)),
            "replay_sha256": hashlib.sha256(replay).hexdigest(),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha,
            "exact_runtime": provenance,
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_json_new(output, result)
    return result


def verify(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    game = play(run_root)
    output = run_root / "verification.json"
    if output.exists():
        value = read_json(output)
        verify_sha(value, "verification")
        return value
    config = load_config()
    worker = Path(config["runtime"]["worker_path"])
    trace = (run_root / str(game["trace_file"])).read_bytes()
    if hashlib.sha256(trace).hexdigest() != game["trace_sha256"]:
        raise RuntimeError("full-game trace bytes changed")
    words = [word for (word,) in struct.iter_unpack("<I", trace)]
    expected = {int(row["tick"]): row for row in game["checkpoints"]}
    core, _campaign = screen._load_external()
    with ExactTrainingRuntime(worker).open_env(
        simulation_config={
            "max_episode_ticks": int(config["full_game"]["maximum_ticks"])
            + max(planner_config(config).horizons)
        }
    ) as session:
        env = session.environment
        observation, _ = env.reset(seed=int(game["seed"]))
        if replay_helpers.checkpoint(env, observation) != expected[0]:
            raise RuntimeError("initial exact replay checkpoint differs")
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                replay_helpers.decode_action(core, word)
            )
            tick = int(observation["tick"])
            if tick in expected and replay_helpers.checkpoint(env, observation) != expected[tick]:
                raise RuntimeError(f"exact replay checkpoint differs at tick {tick}")
        final = replay_helpers.checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        provenance = session.provenance_manifest
    if (
        len(words) != game["action_count"]
        or final != game["final"]
        or snapshot_sha != game["final_snapshot_sha256"]
    ):
        raise RuntimeError("exact replay final closure differs")
    result = with_sha(
        {
            "schema": "irisu-exact-50k-search-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "full_game_sha256": game["sha256"],
            "physics_backend": "exact",
            "replay_reexecution_backend": "exact",
            "verified": True,
            "score": int(final["score"]),
            "survival_ticks": int(final["tick"]),
            "action_count": len(words),
            "replay_sha256": game["replay_sha256"],
            "exact_runtime": provenance,
        }
    )
    write_json_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("init", "collect", "train", "play", "verify", "run-all")
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.run_root)
    elif args.command == "collect":
        result = collect(args.run_root)
    elif args.command == "train":
        result = train(args.run_root)
    elif args.command == "play":
        result = play(args.run_root)
    elif args.command == "verify":
        result = verify(args.run_root)
    else:
        if not args.run_root.exists():
            initialize(args.run_root)
        collect(args.run_root)
        train(args.run_root)
        play(args.run_root)
        result = verify(args.run_root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
