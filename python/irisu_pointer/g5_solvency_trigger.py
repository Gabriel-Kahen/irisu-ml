"""Development-only delayed-horizon compute trigger.

G5 never returns an action or candidate ordinal.  It only decides whether a
caller should buy a staged exact search; the exact planner keeps final say.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .resolution_first_g3r3 import (
    BoostConfigR3,
    HistogramNewtonBoostR3,
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
)
from .resolution_proposal_g4 import (
    INFERENCE_SCHEMA,
    G4InferenceBoard,
    g4_inference_board_from_entries,
)


MODEL_SCHEMA = "irisu-g5-solvency-compute-trigger-v1"
CHECKPOINT_SCHEMA = "irisu-g5-solvency-trigger-checkpoint-v1"
TARGET_NAMES = (
    "delayed_disagreement",
    "terminal_2048",
    "terminal_8192",
    "terminal_12288",
    "runway_q20",
    "minimum_effective_gauge_q20",
    "final_effective_gauge_q20",
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_GEOMETRY = {
    "strong": (0, "analytic-strong", 0.50, 0.75, 2),
    "weak": (4, "analytic-weak", 0.50, 0.75, 1),
}
_INTENT_CATEGORY = {
    "match_rotten": "rotten-hazard",
    "extend_anchor": "viable-anchor",
    "steer_match": "fresh-match",
}
_STRATUM_PHASE = {"early": 0, "middle": 1, "late": 2, "low-gauge": 3}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _body_by_id(observation: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    raw = observation.get("bodies")
    if type(raw) is not list:
        raise ValueError("G5 observation bodies are malformed")
    result = {int(row["id"]): row for row in raw if type(row) is dict}
    if len(result) != len(raw):
        raise ValueError("G5 observation body identities are malformed")
    return result


def _pair_manifest(
    observation: Mapping[str, object], decision: Mapping[str, object], incumbent: bool
) -> dict[str, object]:
    bodies = _body_by_id(observation)
    source_id = int(decision["source_body_id"])
    destination_id = int(decision["destination_body_id"])
    source, destination = bodies[source_id], bodies[destination_id]
    sx, sy, ss = float(source["x"]), float(source["y"]), float(source["size"])
    dx, dy, ds = (
        float(destination["x"]),
        float(destination["y"]),
        float(destination["size"]),
    )
    intent = str(decision["intent"])
    if intent not in _INTENT_CATEGORY:
        raise ValueError("G5 decision intent is unsupported")
    return {
        "source_body_id": source_id,
        "destination_body_id": destination_id,
        "destination_chain_id": int(destination["chain_id"]),
        "category": _INTENT_CATEGORY[intent],
        "intent": intent,
        "distance_sizes": math.hypot(dx - sx, dy - sy) / max((ss + ds) / 2.0, 1e-9),
        "incumbent": incumbent,
    }


def phase0_public_entry(query: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct a strict pre-exact G4 shot inventory from a Phase-0 query."""

    source_observation = query["source_public_observation"]
    horizons = query["horizons"]
    if type(source_observation) is not dict or type(horizons) is not dict:
        raise ValueError("G5 Phase-0 query is malformed")
    if sha256(source_observation) != query["source_public_observation_sha256"]:
        raise ValueError("G5 Phase-0 source observation SHA differs")
    # G4 predates exact bonus color=-2 but otherwise admits bonus bodies.
    # Retain their public kinematics and candidate bindings while mapping only
    # that unsupported sentinel into G4's identity-free color=-1 bucket.
    observation = json.loads(canonical_bytes(source_observation))
    for body in observation["bodies"]:
        if body["kind"] == "bonus" and int(body["color"]) < -1:
            body["color"] = -1
    short = horizons.get("2048")
    if type(short) is not list:
        raise ValueError("G5 Phase-0 short outcomes are malformed")
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in short:
        if type(row) is not dict or row.get("category") == "wait":
            continue
        decision = row.get("decision")
        if type(decision) is not dict:
            raise ValueError("G5 Phase-0 decision is malformed")
        grouped.setdefault(
            (int(decision["source_body_id"]), int(decision["destination_body_id"])), []
        ).append(row)
    if not 1 <= len(grouped) <= 3:
        raise ValueError("G5 Phase-0 shot inventory is outside G4 bounds")
    candidates: list[dict[str, object]] = []
    for pair_ordinal, rows in enumerate(grouped.values()):
        ordered = sorted(rows, key=lambda row: int(row["decision"]["kind"]), reverse=True)
        pair = _pair_manifest(observation, ordered[0]["decision"], pair_ordinal == 0)
        for row in ordered:
            decision = row["decision"]
            strength = "strong" if int(decision["kind"]) == 2 else "weak"
            geometry_ordinal, name, side, below, kind = _GEOMETRY[strength]
            candidates.append(
                {
                    "ordinal": len(candidates),
                    "pair_ordinal": pair_ordinal,
                    "geometry_ordinal": geometry_ordinal,
                    "pair": pair,
                    "geometry": {
                        "name": name,
                        "strength": strength,
                        "side_sizes": side,
                        "below_sizes": below,
                    },
                    "action": {
                        "kind": kind,
                        "x_norm": float(decision["x_norm"]),
                        "y_norm": float(decision["y_norm"]),
                    },
                }
            )
    observation_sha = sha256(observation)
    stratum = str(query["stratum"])
    if stratum not in _STRATUM_PHASE:
        raise ValueError("G5 Phase-0 stratum is unsupported")
    return {
        "schema": INFERENCE_SCHEMA,
        "seed": int(query["seed"]),
        "query_id": f"phase0-{int(query['source_state_hash']):016x}",
        "query_index": _STRATUM_PHASE[stratum],
        "shot_index": min(int(query["query_index"]) + 1, 19),
        "tick": int(query["tick"]),
        "pre_query_public_observation": observation,
        "pre_query_public_observation_sha256": observation_sha,
        "candidates": candidates,
    }


def query_features(entry: Mapping[str, object]) -> tuple[np.ndarray, G4InferenceBoard]:
    """Mean-pool the strict identity-free G4 candidate inventory."""

    board = g4_inference_board_from_entries([entry])
    if len(board.features) == 0:
        raise ValueError("G5 requires at least one nonincumbent shot")
    features = np.asarray(board.features.mean(axis=0), dtype=np.float64)
    if features.shape != (WIDE_FEATURE_WIDTH,) or not np.isfinite(features).all():
        raise RuntimeError("G5 pooled feature row is malformed")
    return features, board


def phase0_targets(query: Mapping[str, object]) -> dict[str, float]:
    horizons = query["horizons"]
    if type(horizons) is not dict:
        raise ValueError("G5 Phase-0 horizons are malformed")
    values: dict[str, float] = {
        "delayed_disagreement": float(bool(query["safe_delayed_disagreement_ordinals"]))
    }
    runway: list[float] = []
    minimum: list[float] = []
    final: list[float] = []
    gauge_max = max(int(query["source_public_observation"]["gauge_max"]), 1)
    for horizon in (2_048, 8_192, 12_288):
        rows = horizons[str(horizon)]
        probes = [row["probe"] for row in rows]
        values[f"terminal_{horizon}"] = float(
            sum(bool(probe["terminated"] or probe["truncated"]) for probe in probes)
            / len(probes)
        )
        runway.extend(min(int(probe["survival_ticks"]), horizon) / horizon for probe in probes)
        minimum.extend(max(0, int(probe["minimum_gauge"])) / gauge_max for probe in probes)
        final.extend(max(0, int(probe["final_gauge"])) / gauge_max for probe in probes)
    values["runway_q20"] = float(np.quantile(runway, 0.2))
    values["minimum_effective_gauge_q20"] = float(np.quantile(minimum, 0.2))
    values["final_effective_gauge_q20"] = float(np.quantile(final, 0.2))
    if set(values) != set(TARGET_NAMES) or any(not 0 <= value <= 1 for value in values.values()):
        raise RuntimeError("G5 target construction failed")
    return values


@dataclass(frozen=True, slots=True)
class G5Fold:
    heldout_seeds: tuple[int, ...]
    heads: tuple[tuple[str, HistogramNewtonBoostR3], ...]

    def __post_init__(self) -> None:
        if (
            not self.heldout_seeds
            or tuple(sorted(set(self.heldout_seeds))) != self.heldout_seeds
            or tuple(name for name, _head in self.heads) != TARGET_NAMES
        ):
            raise ValueError("G5 fold is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "heads": {name: head.manifest() for name, head in self.heads},
        }

    @classmethod
    def from_manifest(cls, value: object) -> "G5Fold":
        if type(value) is not dict or set(value) != {"heldout_seeds", "heads"}:
            raise RuntimeError("G5 fold manifest is malformed")
        raw_seeds, raw_heads = value["heldout_seeds"], value["heads"]
        if type(raw_seeds) is not list or type(raw_heads) is not dict:
            raise RuntimeError("G5 fold manifest is malformed")
        try:
            result = cls(
                tuple(int(seed) for seed in raw_seeds),
                tuple(
                    (name, HistogramNewtonBoostR3.from_manifest(raw_heads[name]))
                    for name in TARGET_NAMES
                ),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError("G5 fold manifest is malformed") from exc
        if set(raw_heads) != set(TARGET_NAMES) or result.manifest() != value:
            raise RuntimeError("G5 fold manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G5Prediction:
    model_sha256: str
    feature_inventory_sha256: str
    member_scores: tuple[float, ...]
    mean: float
    std: float

    def manifest(self) -> dict[str, object]:
        return {
            "model_sha256": self.model_sha256,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "member_scores": list(self.member_scores),
            "mean": self.mean,
            "std": self.std,
        }


@dataclass(frozen=True, slots=True)
class G5SolvencyTrigger:
    folds: tuple[G5Fold, ...]
    training_seeds: tuple[int, ...]
    threshold: float
    training_dataset_sha256: str
    training_feature_inventory_sha256: str
    calibration_sha256: str
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            len(self.folds) != 5
            or len(self.training_seeds) < 6
            or tuple(sorted(set(self.training_seeds))) != self.training_seeds
            or not 0.0 <= self.threshold <= 1.0
            or any(_SHA_RE.fullmatch(value) is None for value in (
                self.training_dataset_sha256,
                self.training_feature_inventory_sha256,
                self.calibration_sha256,
            ))
            or tuple(sorted(set(self.provenance))) != self.provenance
            or len({name for name, _value in self.provenance}) != len(self.provenance)
            or any(not name for name, _value in self.provenance)
            or any(_SHA_RE.fullmatch(value) is None for _name, value in self.provenance)
        ):
            raise ValueError("G5 trigger model is malformed")
        heldout = tuple(seed for fold in self.folds for seed in fold.heldout_seeds)
        if tuple(sorted(heldout)) != self.training_seeds or len(set(heldout)) != len(heldout):
            raise ValueError("G5 whole-seed partition is malformed")
        if any(tuple(name for name, _ in fold.heads) != TARGET_NAMES for fold in self.folds):
            raise ValueError("G5 fold heads are malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "role": "compute-trigger-only-never-selects-or-prunes",
            "feature_source": (
                "mean-pooled-public-identity-free-g4-inventory;"
                "map-bonus-color-below-minus-one-to-minus-one-v2"
            ),
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "target_names": list(TARGET_NAMES),
            "training_seeds": list(self.training_seeds),
            "threshold": self.threshold,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_feature_inventory_sha256": self.training_feature_inventory_sha256,
            "calibration_sha256": self.calibration_sha256,
            "provenance": [list(row) for row in self.provenance],
            "folds": [fold.manifest() for fold in self.folds],
        }

    @property
    def sha256(self) -> str:
        return sha256(self.manifest())

    def predict(self, entry: Mapping[str, object]) -> G5Prediction:
        features, board = query_features(entry)
        matrix = features.reshape(1, -1)
        scores = tuple(
            float(dict(fold.heads)["delayed_disagreement"].probabilities(matrix)[0])
            for fold in self.folds
        )
        return G5Prediction(
            self.sha256,
            board.feature_inventory_sha256,
            scores,
            float(np.mean(scores)),
            float(np.std(scores)),
        )

    @classmethod
    def from_manifest(cls, value: object) -> "G5SolvencyTrigger":
        if type(value) is not dict:
            raise RuntimeError("G5 model manifest is malformed")
        required = {
            "schema", "role", "feature_source", "feature_names", "feature_width",
            "target_names", "training_seeds", "threshold",
            "training_dataset_sha256", "training_feature_inventory_sha256",
            "calibration_sha256", "provenance", "folds",
        }
        if set(value) != required or value["schema"] != MODEL_SCHEMA:
            raise RuntimeError("G5 model manifest is malformed")
        try:
            result = cls(
                tuple(G5Fold.from_manifest(row) for row in value["folds"]),
                tuple(int(seed) for seed in value["training_seeds"]),
                float(value["threshold"]),
                str(value["training_dataset_sha256"]),
                str(value["training_feature_inventory_sha256"]),
                str(value["calibration_sha256"]),
                tuple((str(row[0]), str(row[1])) for row in value["provenance"]),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("G5 model manifest is malformed") from exc
        if result.manifest() != value:
            raise RuntimeError("G5 model manifest is malformed")
        return result


def seed_partition(seeds: Sequence[int], folds: int = 5) -> tuple[tuple[int, ...], ...]:
    unique = tuple(sorted(set(int(seed) for seed in seeds)))
    if folds != 5 or len(unique) <= folds:
        raise ValueError("G5 five-fold whole-seed training requires at least six seeds")
    return tuple(tuple(unique[index::folds]) for index in range(folds))


def calibration_report(
    scores: Sequence[float], labels: Sequence[float], seeds: Sequence[int],
    *, strata: Sequence[str] | None = None, minimum_recall: float = 0.90,
    maximum_trigger_fraction: float = 0.80,
) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    seed = np.asarray(seeds, dtype=np.int64)
    stratum = np.asarray(
        ["unspecified"] * len(score) if strata is None else strata, dtype=object
    )
    if (
        score.shape != label.shape or seed.shape != label.shape
        or stratum.shape != label.shape or len(score) == 0
        or set(np.unique(label)) != {0.0, 1.0}
    ):
        raise ValueError("G5 calibration inputs are malformed")
    thresholds = sorted({0.0, 1.0, *(float(value) for value in score)}, reverse=True)
    rows = []
    for threshold in thresholds:
        predicted = score >= threshold
        positive = label == 1.0
        tp = int(np.sum(predicted & positive))
        fn = int(np.sum(~predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        tn = int(np.sum(~predicted & ~positive))
        recall = tp / max(tp + fn, 1)
        fraction = float(np.mean(predicted))
        rows.append({
            "threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "trigger_fraction": fraction,
            "by_seed": [
                {
                    "seed": int(value),
                    "positives": int(np.sum(positive & (seed == value))),
                    "recalled": int(np.sum(predicted & positive & (seed == value))),
                    "triggers": int(np.sum(predicted & (seed == value))),
                    "queries": int(np.sum(seed == value)),
                }
                for value in np.unique(seed)
            ],
            "by_stratum": [
                {
                    "stratum": str(value),
                    "positives": int(np.sum(positive & (stratum == value))),
                    "recalled": int(np.sum(predicted & positive & (stratum == value))),
                    "triggers": int(np.sum(predicted & (stratum == value))),
                    "queries": int(np.sum(stratum == value)),
                }
                for value in sorted(set(stratum))
            ],
        })
    eligible = [
        row for row in rows
        if row["recall"] >= minimum_recall
        and row["trigger_fraction"] <= maximum_trigger_fraction
    ]
    if not eligible:
        raise RuntimeError("G5 OOF calibration failed recall/compute constraints")
    selected = max(eligible, key=lambda row: (row["threshold"], -row["trigger_fraction"]))
    return {
        "schema": "irisu-g5-oof-calibration-v1",
        "cross_fit_unit": "whole-seed",
        "minimum_recall": minimum_recall,
        "maximum_trigger_fraction": maximum_trigger_fraction,
        "selected": selected,
        "sweep": rows,
    }


def train_g5(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    seeds: np.ndarray,
    *,
    dataset_sha256: str,
    feature_inventory_sha256: str,
    provenance: Mapping[str, str],
    strata: Sequence[str] | None = None,
    rounds: int = 40,
) -> tuple[G5SolvencyTrigger, dict[str, object]]:
    x = np.asarray(features, dtype=np.float64)
    seed = np.asarray(seeds, dtype=np.int64)
    if x.shape != (len(seed), WIDE_FEATURE_WIDTH) or set(targets) != set(TARGET_NAMES):
        raise ValueError("G5 training matrix is malformed")
    partition = seed_partition(tuple(int(value) for value in seed))
    folds: list[G5Fold] = []
    oof = {
        name: np.full(len(seed), np.nan, dtype=np.float64) for name in TARGET_NAMES
    }
    for heldout in partition:
        test = np.isin(seed, heldout)
        train = ~test
        heads = []
        for name in TARGET_NAMES:
            config = BoostConfigR3(
                rounds=rounds, depth=2, learning_rate=0.03, l2=16.0,
                minimum_leaf=4, maximum_features=96, preserved_features=48,
                bins=8, balance_classes=name == "delayed_disagreement",
            )
            head = HistogramNewtonBoostR3.fit(
                x[train], np.asarray(targets[name], dtype=np.float64)[train], seed[train], config
            )
            heads.append((name, head))
        fold = G5Fold(tuple(heldout), tuple(heads))
        folds.append(fold)
        for name, head in fold.heads:
            oof[name][test] = head.probabilities(x[test])
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("G5 OOF prediction coverage is incomplete")
    calibration = calibration_report(
        oof["delayed_disagreement"], targets["delayed_disagreement"], seed,
        strata=strata,
    )
    model = G5SolvencyTrigger(
        tuple(folds), tuple(sorted(set(int(value) for value in seed))),
        float(calibration["selected"]["threshold"]), dataset_sha256,
        feature_inventory_sha256, sha256(calibration), tuple(sorted(provenance.items())),
    )
    return model, {
        **calibration,
        "oof_predictions": {
            name: [float(value) for value in values] for name, values in oof.items()
        },
        "oof_metrics": {
            name: {
                "mean_absolute_error": float(np.mean(np.abs(
                    values - np.asarray(targets[name], dtype=np.float64)
                ))),
                "mean_squared_error": float(np.mean(np.square(
                    values - np.asarray(targets[name], dtype=np.float64)
                ))),
            }
            for name, values in oof.items()
        },
    }


@dataclass(frozen=True, slots=True)
class G5TriggerDecision:
    compute: bool
    observation_sha256: str
    candidate_inventory_sha256: str
    prediction: G5Prediction
    threshold: float
    reason: str

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-g5-compute-trigger-decision-v1",
            "compute": self.compute,
            "observation_sha256": self.observation_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "prediction": self.prediction.manifest(),
            "threshold": self.threshold,
            "reason": self.reason,
        }


def should_compute(model: G5SolvencyTrigger, entry: Mapping[str, object]) -> G5TriggerDecision:
    prediction = model.predict(entry)
    observation = entry["pre_query_public_observation"]
    candidates = entry["candidates"]
    compute = prediction.mean >= model.threshold
    return G5TriggerDecision(
        compute,
        sha256(observation),
        sha256(candidates),
        prediction,
        model.threshold,
        "buy-staged-exact-search" if compute else "immutable-base-fallback",
    )


def checkpoint_envelope(
    model: G5SolvencyTrigger, report: Mapping[str, object]
) -> dict[str, object]:
    """Build the complete content-addressed G5 checkpoint envelope."""

    canonical_report = json.loads(canonical_bytes(report))
    envelope = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.manifest(),
        "model_sha256": model.sha256,
        "training_dataset_sha256": model.training_dataset_sha256,
        "training_feature_inventory_sha256": model.training_feature_inventory_sha256,
        "calibration_sha256": model.calibration_sha256,
        "provenance": [list(row) for row in model.provenance],
        "report": canonical_report,
        "report_sha256": sha256(canonical_report),
    }
    envelope["checkpoint_sha256"] = sha256(envelope)
    return envelope


def _decode_unique_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("G5 checkpoint has duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("G5 checkpoint is not canonical JSON") from exc


def load_checkpoint_g5(
    path: str | os.PathLike[str],
    *,
    expected_checkpoint_sha256: str,
    expected_model_sha256: str,
    expected_dataset_sha256: str,
    expected_feature_inventory_sha256: str,
    expected_calibration_sha256: str,
    expected_provenance: Mapping[str, str],
) -> tuple[G5SolvencyTrigger, dict[str, object]]:
    """Load G5 only when every training identity is supplied and matches."""

    target = os.fspath(path)
    if os.path.islink(target):
        raise RuntimeError("G5 checkpoint path must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read()
    except OSError as exc:
        raise RuntimeError("G5 checkpoint cannot be read safely") from exc
    value = _decode_unique_json(raw)
    if canonical_bytes(value) + b"\n" != raw:
        raise RuntimeError("G5 checkpoint is not canonical JSON")
    required = {
        "schema", "model", "model_sha256", "training_dataset_sha256",
        "training_feature_inventory_sha256", "calibration_sha256", "provenance",
        "report", "report_sha256", "checkpoint_sha256",
    }
    if type(value) is not dict or set(value) != required or value["schema"] != CHECKPOINT_SCHEMA:
        raise RuntimeError("G5 checkpoint envelope is malformed")
    expected = (
        expected_checkpoint_sha256, expected_model_sha256, expected_dataset_sha256,
        expected_feature_inventory_sha256, expected_calibration_sha256,
    )
    if any(type(item) is not str or _SHA_RE.fullmatch(item) is None for item in expected):
        raise RuntimeError("G5 checkpoint expectations are malformed")
    provenance = tuple(sorted((str(key), str(item)) for key, item in expected_provenance.items()))
    if any(_SHA_RE.fullmatch(item) is None for _key, item in provenance):
        raise RuntimeError("G5 provenance expectations are malformed")
    body = {key: item for key, item in value.items() if key != "checkpoint_sha256"}
    if (
        value["checkpoint_sha256"] != sha256(body)
        or value["checkpoint_sha256"] != expected_checkpoint_sha256
        or value["model_sha256"] != expected_model_sha256
        or value["training_dataset_sha256"] != expected_dataset_sha256
        or value["training_feature_inventory_sha256"] != expected_feature_inventory_sha256
        or value["calibration_sha256"] != expected_calibration_sha256
        or value["report_sha256"] != sha256(value["report"])
        or value["provenance"] != [list(row) for row in provenance]
    ):
        raise RuntimeError("G5 checkpoint expectation or identity mismatch")
    model = G5SolvencyTrigger.from_manifest(value["model"])
    if (
        model.sha256 != expected_model_sha256
        or model.training_dataset_sha256 != expected_dataset_sha256
        or model.training_feature_inventory_sha256 != expected_feature_inventory_sha256
        or model.calibration_sha256 != expected_calibration_sha256
        or model.provenance != provenance
    ):
        raise RuntimeError("G5 checkpoint model binding differs")
    return model, value["report"]


__all__ = [
    "G5Fold",
    "G5Prediction",
    "G5SolvencyTrigger",
    "G5TriggerDecision",
    "TARGET_NAMES",
    "checkpoint_envelope",
    "load_checkpoint_g5",
    "phase0_public_entry",
    "phase0_targets",
    "query_features",
    "should_compute",
]
