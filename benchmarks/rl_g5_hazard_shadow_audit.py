#!/usr/bin/env python3
"""Audit a frozen G5 hazard scheduler on fresh TRAIN oracle queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from irisu_pointer.g5_hazard_scheduler import HazardScheduler, SCHEMA  # noqa: E402
from irisu_pointer.g5_solvency_trigger import sha256  # noqa: E402
import rl_g5_solvency_trigger as support  # noqa: E402


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text())
    if type(value) is not dict:
        raise ValueError("G5 shadow input must be a JSON object")
    return value


def load_scheduler(path: Path) -> tuple[HazardScheduler, dict[str, object]]:
    checkpoint = _read(path)
    claimed = checkpoint.get("checkpoint_sha256")
    payload = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if claimed != sha256(payload):
        raise ValueError("G5 scheduler checkpoint content hash mismatch")
    manifest = checkpoint.get("scheduler")
    if type(manifest) is not dict or manifest.get("schema") != SCHEMA:
        raise ValueError("G5 scheduler schema mismatch")
    if checkpoint.get("scheduler_sha256") != sha256(manifest):
        raise ValueError("G5 scheduler manifest hash mismatch")
    provenance = manifest.get("provenance")
    if type(provenance) is not list or any(
        type(row) is not list or len(row) != 2 for row in provenance
    ):
        raise ValueError("G5 scheduler provenance is malformed")
    scheduler = HazardScheduler(
        int(manifest["moving_threshold"]),
        int(manifest["large_threshold"]),
        tuple(int(value) for value in manifest["training_seeds"]),
        str(manifest["dataset_sha256"]),
        tuple((str(row[0]), str(row[1])) for row in provenance),
    )
    if scheduler.manifest() != manifest:
        raise ValueError("G5 scheduler manifest is noncanonical")
    return scheduler, checkpoint


def audit(
    scheduler: HazardScheduler,
    oracle: dict[str, object],
    *, maximum_compute_fraction: float = 0.90,
) -> dict[str, object]:
    claimed = oracle.get("content_sha256")
    payload = {key: value for key, value in oracle.items() if key != "content_sha256"}
    if claimed != sha256(payload):
        raise ValueError("G5 shadow oracle content hash mismatch")
    plan = oracle.get("seed_plan")
    if type(plan) is not dict or plan.get("split") != "train":
        raise ValueError("G5 shadow oracle is not canonical TRAIN data")
    pilot_seeds = tuple(int(value) for value in oracle.get("pilot_seeds", ()))
    if not pilot_seeds or set(pilot_seeds) & set(scheduler.training_seeds):
        raise ValueError("G5 shadow seeds overlap scheduler training seeds")
    episodes = oracle.get("episodes")
    if type(episodes) is not list or len(episodes) != len(pilot_seeds):
        raise ValueError("G5 shadow episode inventory is malformed")
    rows = []
    for episode in episodes:
        for query in episode["queries"]:
            positive = bool(query["safe_delayed_disagreement_ordinals"])
            trigger = scheduler.should_compute(query["source_public_observation"])
            rows.append({
                "seed": int(query["seed"]),
                "query_index": int(query["query_index"]),
                "stratum": str(query["stratum"]),
                "positive": positive,
                "trigger": trigger,
                "candidate_count": int(query["candidate_count"]),
            })
    positives = sum(row["positive"] for row in rows)
    recalled = sum(row["positive"] and row["trigger"] for row in rows)
    triggers = sum(row["trigger"] for row in rows)
    if positives < 1:
        raise RuntimeError("G5 shadow has no positive query and cannot establish recall")
    recall = recalled / positives
    compute_fraction = triggers / len(rows)
    short_ticks = sum(row["candidate_count"] * 2_048 for row in rows)
    always_long = len(rows) * 3 * (8_192 + 12_288)
    scheduled_long = triggers * 3 * (8_192 + 12_288)
    passed = recall == 1.0 and compute_fraction <= maximum_compute_fraction
    by_seed = []
    for seed in sorted(set(row["seed"] for row in rows)):
        selected = [row for row in rows if row["seed"] == seed]
        by_seed.append({
            "seed": seed,
            "queries": len(selected),
            "positives": sum(row["positive"] for row in selected),
            "recalled": sum(row["positive"] and row["trigger"] for row in selected),
            "triggers": sum(row["trigger"] for row in selected),
        })
    return {
        "schema": "irisu-g5-hazard-shadow-audit-v1",
        "development_only": True,
        "promotion_eligible": False,
        "pass": passed,
        "gate": {
            "minimum_recall": 1.0,
            "maximum_compute_fraction": maximum_compute_fraction,
        },
        "queries": len(rows),
        "positives": positives,
        "recalled": recalled,
        "recall": recall,
        "triggers": triggers,
        "compute_fraction": compute_fraction,
        "cost": {
            "short_branch_ticks": short_ticks,
            "always_long_branch_ticks": always_long,
            "scheduled_long_branch_ticks": scheduled_long,
            "scheduled_staged_fraction": (
                (short_ticks + scheduled_long) / (short_ticks + always_long)
            ),
        },
        "by_seed": by_seed,
        "rows_sha256": sha256(rows),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("G5 shadow audit output directory must be new")
    scheduler, checkpoint = load_scheduler(args.scheduler_checkpoint)
    oracle = _read(args.oracle)
    report = audit(scheduler, oracle)
    report.update({
        "scheduler_sha256": scheduler.sha256,
        "scheduler_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "scheduler_checkpoint_file_sha256": support.file_sha(
            args.scheduler_checkpoint.resolve(strict=True)
        ),
        "oracle_content_sha256": oracle["content_sha256"],
        "oracle_file_sha256": support.file_sha(args.oracle.resolve(strict=True)),
        "source_sha256": support.file_sha(Path(__file__).resolve()),
    })
    report["report_sha256"] = sha256(report)
    args.output.mkdir(parents=True)
    support.write_new(args.output / "shadow-audit.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scheduler-checkpoint", type=Path, required=True)
    value.add_argument("--oracle", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
