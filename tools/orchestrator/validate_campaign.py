#!/usr/bin/env python3
"""Bounded end-to-end acceptance run, retaining reproducible local receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from irisu_search.cli import train
from irisu_search.config import CampaignConfig, default_paths
from irisu_search.coordinator import Coordinator
from irisu_search.diagnostics import trace_replay
from irisu_search.engine import Engine
from irisu_search.store import Store
from irisu_search.verification import replay_bytes, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    args.output.mkdir(parents=True)
    defaults = default_paths()
    receipts = {"schema": "irisu-search-acceptance-v1", "backends": {}}
    engines = {"native": (defaults["engine"],)}
    if args.cuda:
        engines["cuda"] = (str(args.cuda.resolve()), "--host-prefix", "--max-checkpoints", "16")
    for name, command in engines.items():
        cfg = CampaignConfig(engine=command, oracle=defaults["oracle"], oracle_library=defaults["oracle_library"],
            initial_seeds=2, attempts_per_seed=1, candidates=4, survivors=2, bootstrap_ticks=96, horizon=48,
            maximum_ticks=768, job_seconds=.15, minimum_seed_seconds=.05, finalist_seconds=.1,
            lanes={"seed": .8, "combo": .1, "probe": .05, "wildcard": .05})
        environment = None
        if name == "cuda":
            import os
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": cfg.cuda_device}
        with Store(args.output / name) as store:
            with Engine(command, env=environment) as engine:
                coordinator = Coordinator(store, engine, cfg)
                if name == "native":
                    coordinator.import_replay(defaults["incumbent"], stride=48000)
                    trace = trace_replay(store, cfg, defaults["incumbent"])
                    (args.output / "genealogy-receipt.json").write_text(json.dumps(trace, indent=2))
                before_manifest = store.get_meta("seed_manifest")
                coordinator.run(seconds=10, max_jobs=12)
                coordinator.close()
            count_before = len(store.jobs("done"))
            checkpoint_count = len(store.checkpoints(active=None))
            if name == "native":
                model_path = args.output / "guide.json"
                learned = train(store, model_path, epochs=5, hidden=8)
                store.set_meta("active_model", {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)})
                receipts["neural_training"] = learned
            # New engine process exercises reconstruction from durable recipes.
            with Engine(command, env=environment) as engine:
                coordinator = Coordinator(store, engine, cfg)
                coordinator.run(seconds=5, max_jobs=2)
                coordinator.close()
            assert store.get_meta("seed_manifest") == before_manifest
            assert len(store.jobs("done")) == count_before + 2
            assert len(store.checkpoints(active=None)) >= checkpoint_count
            assert len({j["seed"] for j in store.jobs("done")}) >= 2
            result = store.report()
            if name == "native":
                champion = store.champion()
                assert champion["score"] == 4393101
                cp = store.get_checkpoint(champion["checkpoint_id"])
                export = args.output / "verified-record.rpy"
                export.write_bytes(replay_bytes(cp["seed"], cp["words"], cp["summary"]))
                assert sha256_file(export) == sha256_file(defaults["incumbent"])
                receipts["record_replay_sha256"] = sha256_file(export)
            receipts["backends"][name] = result
        (args.output / "receipt.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps({"receipt": str((args.output / 'receipt.json').resolve()),
                      "backends": list(receipts["backends"]),
                      "record_replay_sha256": receipts["record_replay_sha256"]}))


if __name__ == "__main__":
    main()
