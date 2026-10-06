#!/usr/bin/env python3
"""Refit frozen-v5 from a persisted expert-iteration label set."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import rl_expert_iteration_dagger as campaign
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.steering_checkpoint import (
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (
    SteeringDataset,
    SteeringExample,
    train_goal_conditioned_steering,
)
from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TEACHER_V1


def load_labels(path: Path) -> tuple[campaign.DistillationLabel, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "irisu-expert-iteration-labels-v1":
        raise ValueError("expert-iteration label schema differs")
    pointer_spec = PointerActionSpec()
    output: list[campaign.DistillationLabel] = []
    for row in payload.get("examples", ()):
        label, manifest = row["label"], row["example"]
        provenance = label["provenance"]
        observation = EncodedBatch(
            np.array(row["global_features"].numpy(), dtype=np.float32, copy=True),
            np.array(row["body_features"].numpy(), dtype=np.float32, copy=True),
            np.array(row["body_mask"].numpy(), dtype=np.bool_, copy=True),
            np.array([int(provenance["tick"])], dtype=np.uint64),
            np.zeros(1, dtype=np.uint32),
            TEACHER_V1,
        )
        targets = manifest["labels"]
        example = SteeringExample(
            manifest["episode_identity"],
            manifest["provenance_sha256"],
            observation,
            int(targets["source_index"]),
            int(targets["destination_index"]),
            int(targets["kind_index"]),
            int(targets["template_index"]),
            int(targets["intent_index"]),
            pointer_spec,
            int(targets["act_index"]),
            int(targets["wait_index"]),
        )
        if example.sha256 != label["example_sha256"]:
            raise RuntimeError("reconstructed label identity differs")
        output.append(
            campaign.DistillationLabel(
                example,
                str(label["label_kind"]),
                str(label["search_sha256"]),
                provenance,
            )
        )
    if not output:
        raise RuntimeError("label artifact is empty")
    raw_dataset = SteeringDataset([value.example for value in output])
    if raw_dataset.manifest() != payload.get("raw_dataset_manifest"):
        raise RuntimeError("raw label dataset manifest differs")
    balanced_dataset = SteeringDataset(campaign.balanced_training_examples(output))
    if balanced_dataset.manifest() != payload.get(
        "balanced_training_dataset_manifest"
    ):
        raise RuntimeError("balanced label dataset manifest differs")
    actual_expert_seeds = sorted(
        {int(value.provenance["seed"]) for value in output}
    )
    declared_expert_seeds = payload.get("expert_iteration_training_seeds")
    if declared_expert_seeds is not None:
        if (
            not isinstance(declared_expert_seeds, list)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 0xFFFF_FFFF
                for seed in declared_expert_seeds
            )
            or len(set(declared_expert_seeds)) != len(declared_expert_seeds)
            or not set(actual_expert_seeds).issubset(declared_expert_seeds)
        ):
            raise RuntimeError("label artifact training-seed lineage differs")
    return tuple(output)


def load_label_seed_lineage(
    path: Path,
    labels: tuple[campaign.DistillationLabel, ...],
) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    """Preserve rollout seeds even when a seed emitted no representable label."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    actual = tuple(sorted({int(value.provenance["seed"]) for value in labels}))
    declared_expert = payload.get("expert_iteration_training_seeds")
    expert = (
        actual
        if declared_expert is None
        else campaign.checkpoint_training_seeds(
            {"training_seeds": declared_expert}
        )
    )
    declared_upstream = payload.get("upstream_training_seeds")
    upstream = (
        None
        if declared_upstream is None
        else campaign.checkpoint_training_seeds(
            {"training_seeds": declared_upstream}
        )
    )
    declared_all = payload.get("training_seeds")
    if declared_all is not None:
        all_seeds = campaign.checkpoint_training_seeds(
            {"training_seeds": declared_all}
        )
        if upstream is None or all_seeds != tuple(sorted(set(upstream) | set(expert))):
            raise RuntimeError("label artifact complete training-seed lineage differs")
    return upstream, expert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-scope", choices=("act-head", "heads", "all"),
        default="act-head",
    )
    parser.add_argument("--training-seed", type=int, default=2026081002)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must not already exist")
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("training parameters must be positive")

    labels_path = args.labels.resolve(strict=True)
    labels = load_labels(labels_path)
    artifact = load_steering_checkpoint(
        campaign.BASE, expected_sha256=campaign.BASE_SHA256
    )
    upstream_training_seeds = campaign.checkpoint_training_seeds(artifact.metadata)
    label_upstream_seeds, expert_iteration_training_seeds = load_label_seed_lineage(
        labels_path, labels
    )
    if (
        label_upstream_seeds is not None
        and label_upstream_seeds != upstream_training_seeds
    ):
        raise RuntimeError("label artifact warm-start seed lineage differs")
    training_seeds = sorted(
        set(upstream_training_seeds) | set(expert_iteration_training_seeds)
    )
    model = copy.deepcopy(artifact.model)
    trainable = campaign.configure_trainable_scope(model, args.trainable_scope)
    balanced = campaign.balanced_training_examples(labels)
    dataset = SteeringDataset(balanced)
    report = train_goal_conditioned_steering(
        model,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.training_seed,
    )
    args.output.mkdir(parents=True)
    metadata = {
        "schema": "irisu-expert-iteration-refit-checkpoint-v1",
        "development_only": True,
        "held_out_seeds_used": False,
        "base_checkpoint_sha256": campaign.BASE_SHA256,
        "upstream_training_seeds": list(upstream_training_seeds),
        "expert_iteration_training_seeds": list(expert_iteration_training_seeds),
        "training_seeds": training_seeds,
        "inference_config": campaign.INFERENCE_CONFIG,
        "labels_path": str(labels_path),
        "labels_sha256": campaign._file_sha(labels_path),
        "balanced_dataset_sha256": dataset.sha256,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": list(trainable),
        "training": asdict(report),
        "source_sha256": campaign._file_sha(Path(__file__).resolve()),
    }
    checkpoint = args.output / "expert-iteration-refit.pt"
    checkpoint_sha = save_steering_checkpoint(checkpoint, model, metadata=metadata)
    provenance = metadata | {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "labels": len(labels),
        "corrections": sum(value.label_kind == "correction" for value in labels),
        "anchors": sum(value.label_kind == "anchor" for value in labels),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "training_seed": args.training_seed,
        },
    }
    campaign._write_json_once(args.output / "provenance.json", provenance)
    print(json.dumps({"checkpoint_sha256": checkpoint_sha, "training": asdict(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
