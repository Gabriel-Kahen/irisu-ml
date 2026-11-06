"""Command-line entry point for durable high-score campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from . import PLANNER_VERSION
from .config import CampaignConfig, default_paths
from .coordinator import Coordinator, campaign_lock, collection_models
from .diagnostics import trace_replay
from .engine import Engine
from .neural import NeuralGuide, train_guide, validate_guide
from .store import Store
from .verification import replay_bytes, sha256_file


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def parser():
    defaults = default_paths()
    p = argparse.ArgumentParser(description="Persistent multi-seed, combo-aware Irisu high-score search")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create a frozen campaign and optional incumbent archive")
    init.add_argument("root", type=Path)
    init.add_argument("--config", type=Path, help="complete JSON CampaignConfig; overrides init settings")
    init.add_argument("--engine", default=defaults["engine"])
    init.add_argument("--engine-arg", action="append", default=[])
    init.add_argument("--cuda-device", default="0", help="CUDA visibility for this worker only; empty inherits environment")
    init.add_argument("--oracle", default=defaults["oracle"])
    init.add_argument("--oracle-library", default=defaults["oracle_library"])
    init.add_argument("--seed-count", type=int, default=128)
    init.add_argument("--seed-generator", type=int, default=0)
    init.add_argument("--incumbent", type=Path)
    init.add_argument("--stride", type=int, default=4096)
    init.add_argument("--combo-events", type=Path, help="replay-bound genealogy receipt for combo event roots")
    run = sub.add_parser("run", help="run bounded jobs, resuming existing state")
    run.add_argument("root", type=Path)
    run.add_argument("--seconds", type=float, default=3600.)
    run.add_argument("--jobs", type=int)
    for command in ("status", "verify", "export", "import-replay", "train", "activate-model", "diagnose", "compare"):
        item = sub.add_parser(command)
        item.add_argument("root", type=Path)
        if command == "verify":
            item.add_argument("checkpoint", nargs="?")
        elif command == "export":
            item.add_argument("output", type=Path)
        elif command == "import-replay":
            item.add_argument("replay", type=Path)
            item.add_argument("--stride", type=int, default=4096)
        elif command == "train":
            item.add_argument("output", type=Path)
            item.add_argument("--epochs", type=int, default=80)
            item.add_argument("--hidden", type=int, default=24)
            item.add_argument("--seed", type=int, default=0)
        elif command == "activate-model":
            item.add_argument("model", type=Path)
        elif command == "diagnose":
            item.add_argument("replay", type=Path)
            item.add_argument("--stride", type=int, default=256)
            item.add_argument("--events", action="store_true", help="stream full native group genealogy")
            item.add_argument("--output", type=Path)
        elif command == "compare":
            item.add_argument("others", type=Path, nargs="+")
    return p


def open_coordinator(store, config):
    environment = dict(os.environ)
    if config.cuda_device:
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    engine = Engine(config.engine, timeout_seconds=config.engine_timeout, env=environment)
    try:
        coordinator = Coordinator(store, engine, config)
    except BaseException:
        engine.close()
        raise
    return engine, coordinator


def train(store, output, *, epochs=80, hidden=24, seed=0):
    examples = store.examples()
    if not examples:
        raise ValueError("campaign has no training examples")
    identity = store.get_meta("engine_identity")
    if any(row.get("engine_identity") != identity for row in examples):
        raise ValueError("training data mixes engine identities")
    if any(row.get("planner_version") != PLANNER_VERSION for row in examples):
        raise ValueError("training data mixes planner versions")
    model, report = train_guide(examples, epochs=epochs, hidden=hidden, seed=seed)
    report["collection_models"] = collection_models(examples)
    model.save(output, {"engine_identity": identity, "collection_models": report["collection_models"]})
    store.event("model_trained", {"path": str(output.resolve()), "sha256": sha256_file(output), "report": report})
    return report


def comparison(roots):
    rows = []
    for root in roots:
        with Store(root) as store:
            report = store.report()
            points = []
            # Completed jobs carry actual all-in costs; promotions have timestamps
            # so time curves can be reconstructed without treating capped scores as records.
            events = list(store.db.execute("SELECT created,payload FROM events WHERE kind='champion' ORDER BY created"))
            jobs = store.jobs()
            for event in events:
                payload = json.loads(event["payload"])
                spent = sum(job.get("cost_seconds", 0) for job in jobs if job["updated"] <= event["created"])
                points.append({"search_seconds_before_promotion_job": spent,
                               "wall_since_init": event["created"] - store.get_meta("created", event["created"]),
                               "score": payload["score"]})
            rows.append({"root": str(Path(root).resolve()), "search_seconds": report["search_seconds"],
                         "champion": report["champion"], "seed_count": len(report["seeds"]),
                         "simulated_ticks": report["simulated_ticks"], "verified_score_curve": points,
                         "config": store.get_meta("campaign_config")})
    return {"campaigns": rows, "note": "Compare shared wall-time budgets and whole-seed suites; this report does not infer a speedup."}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "compare":
            emit(comparison([args.root, *args.others]))
            return 0
        with Store(args.root) as store:
            if args.command == "init":
                if store.get_meta("campaign_config"):
                    raise ValueError("campaign already initialized; use run to resume")
                config = CampaignConfig.parse(json.loads(args.config.read_text())) if args.config else CampaignConfig(
                    engine=(args.engine, *args.engine_arg), oracle=args.oracle,
                    oracle_library=args.oracle_library, cuda_device=args.cuda_device, initial_seeds=args.seed_count,
                    seed_generator=args.seed_generator)
            else:
                manifest = store.get_meta("campaign_config")
                if not manifest:
                    raise ValueError("campaign is not initialized")
                config = CampaignConfig.parse(manifest)
            if args.command == "status":
                emit(store.report())
                return 0
            if args.command == "train":
                with campaign_lock(store.root):
                    emit(train(store, args.output, epochs=args.epochs, hidden=args.hidden, seed=args.seed))
                return 0
            if args.command == "activate-model":
                guide = NeuralGuide.load(args.model)
                validate_guide(guide, store.get_meta("engine_identity"), PLANNER_VERSION)
                with campaign_lock(store.root):
                    store.set_meta("active_model", {"path": str(args.model.resolve()), "sha256": sha256_file(args.model)})
                emit({"active_model": store.get_meta("active_model"), "exploration_fraction": config.exploration_fraction})
                return 0
            if args.command == "export":
                champion = store.champion()
                if not champion:
                    raise ValueError("campaign has no verified champion")
                checkpoint = store.get_checkpoint(champion["checkpoint_id"])
                data = replay_bytes(checkpoint["seed"], checkpoint["words"], checkpoint["summary"])
                if hashlib.sha256(data).hexdigest() != champion["receipt"]["replay_sha256"]:
                    raise ValueError("champion replay differs from verification receipt")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation prevents accidental destruction of an existing replay.
                with args.output.open("xb") as stream:
                    stream.write(data)
                emit({"output": str(args.output.resolve()), "score": champion["score"], "sha256": sha256_file(args.output)})
                return 0
            if args.command == "diagnose" and args.events:
                with campaign_lock(store.root):
                    emit(trace_replay(store, config, args.replay, args.output))
                return 0
            engine, coordinator = open_coordinator(store, config)
            try:
                if args.command == "init":
                    with campaign_lock(store.root):
                        if args.incumbent:
                            coordinator.import_replay(args.incumbent, args.stride, events_path=args.combo_events)
                    emit(store.report())
                elif args.command == "run":
                    emit(coordinator.run(seconds=args.seconds, max_jobs=args.jobs, progress=emit))
                elif args.command in ("import-replay", "diagnose"):
                    with campaign_lock(store.root):
                        ids = coordinator.import_replay(args.replay, args.stride)
                    if args.command == "diagnose":
                        rows = [store.get_checkpoint(key) for key in ids]
                        emit({"seed": rows[0]["seed"], "samples": [
                            {"tick": row["summary"]["tick"], "score": row["summary"]["score"],
                             "qualifying": row["summary"]["qualifying"], "groups": row["summary"]["groups"]}
                            for row in rows], "scope": "checkpoint samples; group IDs persist, events between samples are not reconstructed"})
                    else:
                        emit({"imported": ids, "champion": store.champion()})
                elif args.command == "verify":
                    champion = store.champion()
                    key = args.checkpoint or (champion and champion["checkpoint_id"])
                    if not key:
                        raise ValueError("specify a checkpoint to verify")
                    summary = store.get_checkpoint(key)["summary"]
                    if not summary.get("terminated") or summary.get("truncated"):
                        raise ValueError("only naturally terminated checkpoints can be verified")
                    with campaign_lock(store.root):
                        coordinator._verify(key, force=True)
                    emit({"checkpoint": key, "verified": True, "champion": store.champion()})
            finally:
                try:
                    coordinator.close()
                finally:
                    engine.close()
        return 0
    except KeyboardInterrupt:
        print("Search interrupted; completed batches and recipes are retained.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"irisu-search: {error}", file=sys.stderr)
        return 1
