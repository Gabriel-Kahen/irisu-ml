#!/usr/bin/env python3
"""Fit the development-only G5 staged-compute trigger from exact pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer.g5_solvency_trigger import (  # noqa: E402
    TARGET_NAMES,
    canonical_bytes,
    checkpoint_envelope,
    phase0_public_entry,
    phase0_targets,
    query_features,
    sha256,
    train_g5,
)


SCHEMA = "irisu-g5-training-input-manifest-v1"


def load_unique_json(path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in {path}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=pairs)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = load_unique_json(path)
    if (
        type(manifest) is not dict
        or set(manifest) != {
            "schema", "purpose", "expected_base_checkpoint_sha256",
            "expected_base_model_sha256", "artifacts",
        }
        or manifest.get("schema") != SCHEMA
        or manifest.get("purpose") != "g5-compute-trigger-development-training-only"
    ):
        raise ValueError("G5 input manifest is malformed")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list or not artifacts:
        raise ValueError("G5 input manifest has no artifacts")
    loaded = []
    for item in artifacts:
        if type(item) is not dict or set(item) != {"path", "sha256", "content_sha256"}:
            raise ValueError("G5 input artifact binding is malformed")
        declared_path = Path(item["path"])
        if not declared_path.is_absolute() or declared_path.is_symlink():
            raise RuntimeError("G5 input artifact path is not direct and absolute")
        artifact_path = declared_path.resolve(strict=True)
        if file_sha(artifact_path) != item["sha256"]:
            raise RuntimeError("G5 input artifact file identity differs")
        value = load_unique_json(artifact_path)
        content = value.pop("content_sha256", None)
        if content != item["content_sha256"] or sha256(value) != content:
            raise RuntimeError("G5 input artifact content identity differs")
        value["content_sha256"] = content
        loaded.append(value)
    all_seeds: list[int] = []
    for value in loaded:
        pilot = value.get("pilot_seeds")
        seed_plan = value.get("seed_plan")
        if (
            value.get("schema") != "irisu-exact-phase0-delayed-horizon-oracle-v1"
            or value.get("development_only") is not True
            or value.get("promotion_eligible") is not False
            or value.get("base_checkpoint_sha256")
            != manifest["expected_base_checkpoint_sha256"]
            or value.get("base_model_sha256") != manifest["expected_base_model_sha256"]
            or type(pilot) is not list
            or any(type(seed) is not int or not 0 <= seed < 1 << 30 for seed in pilot)
            or type(seed_plan) is not dict
            or seed_plan.get("split") != "train"
            or seed_plan.get("seeds") != pilot
            or seed_plan.get("count") != len(pilot)
        ):
            raise RuntimeError("G5 oracle artifact lineage differs")
        episode_seeds = [episode.get("seed") for episode in value.get("episodes", [])]
        if (
            episode_seeds != pilot
            or any(
                query.get("seed") != episode.get("seed")
                for episode in value["episodes"]
                for query in episode.get("queries", [])
            )
        ):
            raise RuntimeError("G5 oracle artifact episode identities differ")
        all_seeds.extend(pilot)
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("G5 oracle artifacts reuse a training seed")
    return manifest, loaded


def dataset(artifacts: list[dict[str, object]]) -> tuple[
    np.ndarray, dict[str, np.ndarray], np.ndarray, list[dict[str, object]]
]:
    rows = []
    features = []
    labels = {name: [] for name in TARGET_NAMES}
    seeds = []
    seen = set()
    for artifact in artifacts:
        for episode in artifact["episodes"]:
            for query in episode["queries"]:
                key = (int(query["seed"]), int(query["source_state_hash"]))
                if key in seen:
                    raise RuntimeError("G5 training query is duplicated")
                seen.add(key)
                entry = phase0_public_entry(query)
                feature, board = query_features(entry)
                target = phase0_targets(query)
                features.append(feature)
                seeds.append(key[0])
                for name in TARGET_NAMES:
                    labels[name].append(target[name])
                rows.append({
                    "seed": key[0],
                    "source_state_hash": key[1],
                    "stratum": query["stratum"],
                    "source_observation_sha256": query["source_public_observation_sha256"],
                    "normalized_g4_inventory_sha256": board.feature_inventory_sha256,
                    "target": target,
                })
    return (
        np.asarray(features, dtype=np.float64),
        {name: np.asarray(value, dtype=np.float64) for name, value in labels.items()},
        np.asarray(seeds, dtype=np.int64),
        rows,
    )


def write_new(path: Path, value: object) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.write(b"\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError("G5 output directory must be new")
    input_manifest, artifacts = load_inputs(args.inputs)
    features, targets, seeds, rows = dataset(artifacts)
    dataset_manifest = {
        "schema": "irisu-g5-query-training-dataset-v1",
        "target_formulas": {
            "delayed_disagreement": "any(safe_delayed_disagreement_ordinals)",
            "terminal_h": "mean_branch_I(terminated_or_truncated_by_h)",
            "runway_q20": "q20_all_branch_horizons(min(survival,h)/h)",
            "minimum_effective_gauge_q20": "q20_all_branch_horizons(max(minimum_gauge,0)/gauge_max)",
            "final_effective_gauge_q20": "q20_all_branch_horizons(max(final_gauge,0)/gauge_max)",
        },
        "rows": rows,
    }
    dataset_sha = sha256(dataset_manifest)
    feature_sha = sha256({
        "rows": [
            {"identity": [row["seed"], row["source_state_hash"]], "features": list(map(float, feature))}
            for row, feature in zip(rows, features, strict=True)
        ]
    })
    provenance = {
        "input-manifest": file_sha(args.inputs.resolve(strict=True)),
        "trainer-source": file_sha(Path(__file__).resolve()),
        "feature-source": file_sha((ROOT / "python/irisu_pointer/g5_solvency_trigger.py").resolve()),
        **{
            f"oracle-{index}-file": item["sha256"]
            for index, item in enumerate(input_manifest["artifacts"])
        },
        **{
            f"oracle-{index}-content": item["content_sha256"]
            for index, item in enumerate(input_manifest["artifacts"])
        },
    }
    model, calibration = train_g5(
        features, targets, seeds, dataset_sha256=dataset_sha,
        feature_inventory_sha256=feature_sha, provenance=provenance,
        strata=[str(row["stratum"]) for row in rows],
        rounds=args.rounds,
    )
    report = {
        "schema": "irisu-g5-solvency-trigger-training-report-v1",
        "development_only": True,
        "promotion_eligible": False,
        "limited_sample_warning": f"{len(set(seeds))} whole seeds / {len(seeds)} queries",
        "role": "compute-trigger-only; exact planner retains all selection authority",
        "model_sha256": model.sha256,
        "dataset_sha256": dataset_sha,
        "feature_inventory_sha256": feature_sha,
        "training_seeds": sorted(set(int(value) for value in seeds)),
        "query_count": len(seeds),
        "positive_queries": int(np.sum(targets["delayed_disagreement"])),
        "calibration": calibration,
        "calibration_sha256": sha256({
            key: value for key, value in calibration.items()
            if key not in {"oof_predictions", "oof_metrics"}
        }),
        "input_manifest": input_manifest,
    }
    checkpoint = checkpoint_envelope(model, report)
    args.output.mkdir(parents=True)
    write_new(args.output / "dataset.json", dataset_manifest)
    write_new(args.output / "checkpoint.json", checkpoint)
    write_new(args.output / "report.json", report)
    return {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "model_sha256": model.sha256,
        "report_sha256": checkpoint["report_sha256"],
        "threshold": model.threshold,
        "selected_calibration": calibration["selected"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--inputs", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--rounds", type=int, default=40)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
