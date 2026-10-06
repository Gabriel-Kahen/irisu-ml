#!/usr/bin/env python3
"""Fit the development-only deterministic G5 exact-compute scheduler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from irisu_pointer.g5_hazard_scheduler import fit_scheduler, hazard_counts  # noqa: E402
from irisu_pointer.g5_solvency_trigger import sha256  # noqa: E402
import rl_g5_solvency_trigger as source  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("G5 scheduler output directory must be new")
    input_manifest, artifacts = source.load_inputs(args.inputs)
    queries = [
        query for artifact in artifacts
        for episode in artifact["episodes"] for query in episode["queries"]
    ]
    rows = [{
        "seed": int(query["seed"]),
        "source_state_hash": int(query["source_state_hash"]),
        "source_public_observation_sha256": query["source_public_observation_sha256"],
        "moving_body_count": hazard_counts(query["source_public_observation"])[0],
        "large_body_count": hazard_counts(query["source_public_observation"])[1],
        "positive": bool(query["safe_delayed_disagreement_ordinals"]),
    } for query in queries]
    dataset = {
        "schema": "irisu-g5-hazard-scheduler-dataset-v1",
        "rows": rows,
    }
    dataset_sha = sha256(dataset)
    provenance = {
        "input-manifest": source.file_sha(args.inputs.resolve(strict=True)),
        "trainer-source": source.file_sha(Path(__file__).resolve()),
        "scheduler-source": source.file_sha(
            (ROOT / "python/irisu_pointer/g5_hazard_scheduler.py").resolve()
        ),
        **{
            f"oracle-{index}-file": item["sha256"]
            for index, item in enumerate(input_manifest["artifacts"])
        },
        **{
            f"oracle-{index}-content": item["content_sha256"]
            for index, item in enumerate(input_manifest["artifacts"])
        },
    }
    scheduler, oof = fit_scheduler(
        [query["source_public_observation"] for query in queries],
        [bool(query["safe_delayed_disagreement_ordinals"]) for query in queries],
        [int(query["seed"]) for query in queries],
        dataset_sha256=dataset_sha,
        provenance=provenance,
    )
    triggers = [
        scheduler.should_compute(query["source_public_observation"])
        for query in queries
    ]
    short_ticks = sum(query["candidate_count"] * 2_048 for query in queries)
    long_ticks = len(queries) * 3 * (8_192 + 12_288)
    scheduled_long_ticks = sum(triggers) * 3 * (8_192 + 12_288)
    report = {
        "schema": "irisu-g5-hazard-scheduler-report-v1",
        "development_only": True,
        "promotion_eligible": False,
        "scheduler_sha256": scheduler.sha256,
        "dataset_sha256": dataset_sha,
        "input_manifest": input_manifest,
        "oof": oof,
        "cost_estimate": {
            "always_short_branch_ticks": short_ticks,
            "always_long_branch_ticks": long_ticks,
            "scheduled_long_branch_ticks": scheduled_long_ticks,
            "always_staged_branch_ticks": short_ticks + long_ticks,
            "scheduled_staged_branch_ticks": short_ticks + scheduled_long_ticks,
            "scheduled_staged_fraction": (
                (short_ticks + scheduled_long_ticks) / (short_ticks + long_ticks)
            ),
            "scheduler_query_fraction": sum(triggers) / len(triggers),
            "oof_scheduler_query_fraction": oof["compute_fraction"],
        },
    }
    checkpoint = {
        "schema": "irisu-g5-hazard-scheduler-checkpoint-v1",
        "scheduler": scheduler.manifest(),
        "scheduler_sha256": scheduler.sha256,
        "report": report,
        "report_sha256": sha256(report),
    }
    checkpoint["checkpoint_sha256"] = sha256(checkpoint)
    args.output.mkdir(parents=True)
    source.write_new(args.output / "dataset.json", dataset)
    source.write_new(args.output / "report.json", report)
    source.write_new(args.output / "checkpoint.json", checkpoint)
    return {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "scheduler_sha256": scheduler.sha256,
        "thresholds": [scheduler.moving_threshold, scheduler.large_threshold],
        "oof": oof,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--inputs", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
