"""Small, dependency-free neural guidance for exact high-score search.

Predictions order experiments; they never establish a record or prune a state.
Numeric entity/group/seed identifiers are not features.  Training uses whole
game-seed holdouts and never treats censored returns as completed outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .proposals import target_group

SCHEMA = "irisu-search-neural-guide-v1"
FEATURE_SCHEMA = "irisu-search-group-features-v1"
OPERATORS = (
    "recruitment", "preservation", "handoff", "payout", "timing",
    "repair", "bootstrap", "completion", "wait", "baseline", "other",
)
FEATURE_NAMES = (
    "level", "qualifying", "score_log", "gauge", "body_count",
    "fresh_count", "grouped_count", "damaged_count", "rot_count",
    "pending_count", "special_count", "body_x_mean", "body_y_mean",
    "body_y_max", "speed_mean", "size_mean", "lifetime_min",
    "live_group_count", "extendable_group_count", "chain_max_log",
    "num_max_log", "multiplier_max_log", "multiplier_sum_log",
    "best_carriers", "best_safe_carriers", "best_carrier_y",
    "best_carrier_hits", "best_carrier_rot", "best_fresh_distance",
    "second_multiplier_log", "third_multiplier_log", "best_group_fraction",
    "terminated", "truncated", "mode_replay", "mode_recruit",
    "mode_preserve", "mode_payout",
    *("operator_" + name for name in OPERATORS),
    "action_count_log", "shot_fraction", "first_shot_x", "first_shot_y",
    "horizon_log", "budget_log",
)
HEADS = ("gain", "extension", "survival", "compute_gain")
GAIN_SCALE = 10_000.0


def _number(value: Any, name: str = "feature") -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _log(value: Any) -> float:
    return math.log1p(max(0.0, _number(value)))


def _mean(values: Sequence[float], fallback: float = 0.0) -> float:
    return math.fsum(sorted(values)) / len(values) if values else fallback


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def features(summary: Mapping[str, Any], candidate: Mapping[str, Any] | None = None) -> list[float]:
    """Encode observable geometry and actual live-group history invariantly.

    ``chain``/``num`` come from group summaries, not Confirmed event values or
    historical highest-chain. IDs are used only for membership joins. Budget
    is an explicit feature because search returns depend on allocated effort.
    """
    candidate = candidate or {}
    bodies = list(summary.get("bodies", ()))
    by_id = {body["id"]: body for body in bodies}
    ordinary = [body for body in bodies if body.get("color", -1) >= 0 and not body.get("special", False)]
    fresh = [body for body in ordinary if not body.get("chain_id", 0)]
    grouped = [body for body in ordinary if body.get("chain_id", 0)]

    def safe(body: Mapping[str, Any]) -> bool:
        return (not body.get("pending_clear", False) and not body.get("pending_delete", False)
                and _number(body.get("hits", 0)) < 2 and _number(body.get("lifetime", 100_000)) > 1)

    groups = []
    for group in summary.get("groups", ()):
        members = [by_id[identifier] for identifier in group.get("members", ()) if identifier in by_id]
        if not members:
            continue
        carriers = [body for body in members if safe(body)]
        chain, num = _number(group.get("chain", 0)), _number(group.get("num", 0))
        multiplier = max(0.0, chain) * (max(0.0, num) + 1)
        distance = min((math.hypot(_number(a.get("x", 0)) - _number(b.get("x", 0)),
                                   _number(a.get("y", 0)) - _number(b.get("y", 0)))
                        for a in carriers for b in fresh if a.get("color") == b.get("color")), default=640.0)
        # Include all aggregate geometry in ties; never depend on ID/order.
        descriptor = (multiplier, chain, num, len(carriers), len(members),
                      -_mean([_number(b.get("y", 0)) for b in members]), -distance,
                      -_mean([_number(b.get("y", 0)) for b in carriers]),
                      -_mean([_number(b.get("hits", 0)) for b in members]),
                      -_mean([min(_number(b.get("rot_timer", 0)), 41) for b in members]))
        groups.append((descriptor, members, carriers, distance, group))
    groups.sort(key=lambda row: row[0], reverse=True)
    chosen = target_group(summary)
    best = next((row for row in groups if chosen and row[4] is chosen[0]), None)
    descriptor, members, carriers, distance, _group = best or ((0,) * 10, [], [], 640.0, None)
    multipliers = [row[0][0] for row in groups]
    lifetimes = [_number(b.get("lifetime", 100_000)) for b in ordinary]
    result = [
        _number(summary.get("level", 1)) / 100,
        _number(summary.get("qualifying", 0)) / 990,
        _log(summary.get("score", 0)) / 16,
        _number(summary.get("gauge", 0)) / 40_000,
        len(bodies) / 196, len(fresh) / 196, len(grouped) / 196,
        sum(_number(b.get("hits", 0)) > 0 for b in grouped) / 196,
        sum(_number(b.get("rot_timer", 0)) > 0 for b in ordinary) / 196,
        sum(bool(b.get("pending_clear") or b.get("pending_delete")) for b in bodies) / 196,
        sum(bool(b.get("special")) for b in bodies) / 196,
        _mean([_number(b.get("x", 0)) / 640 for b in ordinary]),
        _mean([_number(b.get("y", 0)) / 480 for b in ordinary]),
        max((_number(b.get("y", 0)) / 480 for b in ordinary), default=0),
        _mean([math.hypot(_number(b.get("vx", 0)), _number(b.get("vy", 0))) / 50 for b in ordinary]),
        _mean([_number(b.get("size", 0)) / 140 for b in ordinary]),
        min(lifetimes, default=100_000) / 100_000,
        len(groups) / 100, sum(bool(row[2]) for row in groups) / 100,
        _log(max((row[0][1] for row in groups), default=0)) / 7,
        _log(max((row[0][2] for row in groups), default=0)) / 7,
        _log(descriptor[0]) / 14, _log(math.fsum(sorted(multipliers))) / 14,
        len(members) / 196, len(carriers) / 196,
        _mean([_number(b.get("y", 0)) / 480 for b in carriers]),
        _mean([_number(b.get("hits", 0)) / 2 for b in members]),
        _mean([min(_number(b.get("rot_timer", 0)), 41) / 41 for b in members]),
        min(distance, 1280) / 640,
        _log(multipliers[1] if len(multipliers) > 1 else 0) / 14,
        _log(multipliers[2] if len(multipliers) > 2 else 0) / 14,
        len(members) / max(1, len(grouped)),
        float(bool(summary.get("terminated"))), float(bool(summary.get("truncated"))),
    ]
    mode = candidate.get("mode", 0)
    result.extend(float(mode == value) for value in range(4))
    operator = candidate.get("operator", "other")
    operator = {"recruit": "recruitment", "preserve": "preservation",
                "prepare": "timing", "incumbent": "baseline", "complete": "completion"}.get(operator, operator)
    if operator not in OPERATORS:
        operator = "other"
    result.extend(float(operator == name) for name in OPERATORS)
    words = list(candidate.get("words", ()))
    if any(type(word) is not int or not 0 <= word < 1 << 21 for word in words):
        raise ValueError("candidate words must be legal 21-bit replay inputs")
    shots = [word for word in words if word & 3]
    first = shots[0] if shots else 0
    result.extend((_log(len(words)) / 12, len(shots) / max(1, len(words)),
                   ((first >> 2) & 1023) / 640 if shots else 0,
                   ((first >> 12) & 511) / 480 if shots else 0,
                   _log(candidate.get("horizon", len(words))) / 12,
                   _log(candidate.get("budget_seconds", 0)) / 12))
    assert len(result) == len(FEATURE_NAMES)
    return [_number(value) for value in result]


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1 / (1 + math.exp(-value))


def _latent(value: float) -> float:
    return math.asinh(value / GAIN_SCALE)


def _gain(value: float) -> float:
    return GAIN_SCALE * math.sinh(max(-20.0, min(20.0, value)))


class NeuralGuide:
    """Tanh MLP with return, extension, survival and compute-benefit heads."""

    @classmethod
    def create(cls, seed: int = 0, hidden: int = 24) -> NeuralGuide:
        if type(hidden) is not int or not 1 <= hidden <= 512:
            raise ValueError("hidden width must be in 1..512")
        self = cls()
        rng = random.Random(seed)
        self.hidden = hidden
        width = len(FEATURE_NAMES)
        self.w1 = [[rng.uniform(-1, 1) * math.sqrt(3 / width) for _ in range(width)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-1, 1) * math.sqrt(3 / hidden) for _ in range(hidden)] for _ in HEADS]
        self.b2 = [0.0] * len(HEADS)
        self.mean = [0.0] * width
        self.scale = [1.0] * width
        self.head_counts = [0] * len(HEADS)
        self.metadata: dict[str, Any] = {}
        return self

    def _forward(self, row: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        if len(row) != len(FEATURE_NAMES):
            raise ValueError("neural feature schema width differs")
        normalized = [max(-8.0, min(8.0, (_number(x) - mean) / scale))
                      for x, mean, scale in zip(row, self.mean, self.scale)]
        hidden = [math.tanh(bias + math.fsum(w * x for w, x in zip(weights, normalized)))
                  for weights, bias in zip(self.w1, self.b1)]
        output = [bias + math.fsum(w * x for w, x in zip(weights, hidden))
                  for weights, bias in zip(self.w2, self.b2)]
        return normalized, hidden, output

    def predict_features(self, row: Sequence[float]) -> dict[str, float]:
        output = self._forward(row)[2]
        # Unobserved return heads must not inject random preference when a
        # campaign has produced only censored, partial-outcome supervision.
        output = [value if count else 0.0 for value, count in zip(output, self.head_counts)]
        extension, survival = _sigmoid(output[1]), _sigmoid(output[2])
        return {
            "predicted_gain": _gain(output[0]),
            "extension_probability": extension,
            "survival_probability": survival,
            "compute_gain": _gain(output[3]),
            "priority": output[0] + 2 * (extension - .5) + (survival - .5) + .25 * output[3],
        }

    def predict(self, summary: Mapping[str, Any], candidate: Mapping[str, Any] | None = None) -> dict[str, float]:
        return self.predict_features(features(summary, candidate))

    def rank(self, summary: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return every candidate, stably ordered, with attached predictions."""
        rows = [(self.predict(summary, candidate), dict(candidate)) for candidate in candidates]
        rows.sort(key=lambda row: row[0]["priority"], reverse=True)
        return [{**candidate, "neural": prediction} for prediction, candidate in rows]

    @staticmethod
    def _rows(examples: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        rows, skipped = [], 0
        for example in examples:
            seed = example.get("seed")
            if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError("training seed must be uint32")
            values = list(example.get("features", ()))
            if len(values) != len(FEATURE_NAMES):
                raise ValueError("training feature schema width differs")
            values = [_number(value) for value in values]
            if "budget_seconds" in example:
                values[-1] = _log(example["budget_seconds"]) / 12
            targets = example.get("targets", {})
            supervised: dict[int, float] = {}
            for index, name in enumerate(HEADS):
                if name not in targets or targets[name] is None:
                    continue
                if example.get("censored", False) and name in {"gain", "compute_gain"}:
                    continue
                value = _number(targets[name], f"target {name}")
                if index in {1, 2}:
                    if not 0 <= value <= 1:
                        raise ValueError("probability targets must be in [0,1]")
                else:
                    value = _latent(value)
                supervised[index] = value
            if not supervised:
                skipped += 1
                continue
            rows.append({"seed": seed, "features": values, "targets": supervised,
                         "planner_version": str(example.get("planner_version", "unspecified"))})
        return rows, skipped

    def _loss(self, rows: Sequence[Mapping[str, Any]]) -> float | None:
        loss, count = 0.0, 0
        for row in rows:
            output = self._forward(row["features"])[2]
            for index, target in row["targets"].items():
                if index in {1, 2}:
                    value = output[index]
                    loss += max(value, 0) - value * target + math.log1p(math.exp(-abs(value)))
                else:
                    loss += .5 * (output[index] - target) ** 2
                count += 1
        return loss / count if count else None

    def train(self, examples: Sequence[Mapping[str, Any]], *, epochs: int = 80, seed: int = 0,
              validation_fraction: float = .25, learning_rate: float = .025,
              batch_size: int = 16) -> dict[str, Any]:
        if (type(epochs) is not int or not 1 <= epochs <= 100_000 or type(batch_size) is not int
                or batch_size < 1 or not 0 <= validation_fraction < 1
                or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1):
            raise ValueError("invalid neural training configuration")
        rows, skipped = self._rows(examples)
        if not rows:
            raise ValueError("no observed supervision; untested or censored returns are not failures")
        versions = sorted({row["planner_version"] for row in rows})
        if len(versions) != 1:
            raise ValueError("train separate neural guides for different planner versions")
        seeds = sorted({row["seed"] for row in rows})
        rng = random.Random(seed)
        rng.shuffle(seeds)
        holdout_count = min(len(seeds) - 1, max(1, round(len(seeds) * validation_fraction))) if validation_fraction and len(seeds) > 1 else 0
        heldout = set(seeds[:holdout_count])
        training = [row for row in rows if row["seed"] not in heldout]
        validation = [row for row in rows if row["seed"] in heldout]
        # Stable initial ordering keeps training reproducible when a database
        # query returns the same examples in a different order.
        training.sort(key=lambda row: _canonical(row))
        validation.sort(key=lambda row: _canonical(row))
        self.head_counts = [sum(index in row["targets"] for row in training) for index in range(len(HEADS))]
        width = len(FEATURE_NAMES)
        self.mean = [_mean([row["features"][column] for row in training]) for column in range(width)]
        self.scale = [max(.05, math.sqrt(_mean([(row["features"][column] - self.mean[column]) ** 2
                                               for row in training]))) for column in range(width)]
        initial = self._loss(training)
        validation_initial = self._loss(validation)
        for _epoch in range(epochs):
            rng.shuffle(training)
            for begin in range(0, len(training), batch_size):
                batch = training[begin:begin + batch_size]
                g1 = [[0.0] * width for _ in range(self.hidden)]
                gb1 = [0.0] * self.hidden
                g2 = [[0.0] * self.hidden for _ in HEADS]
                gb2 = [0.0] * len(HEADS)
                count = 0
                for row in batch:
                    normalized, hidden, output = self._forward(row["features"])
                    delta = [0.0] * len(HEADS)
                    for index, target in row["targets"].items():
                        error = (_sigmoid(output[index]) - target if index in {1, 2} else output[index] - target)
                        delta[index] = max(-5.0, min(5.0, error))
                        count += 1
                    for index in range(len(HEADS)):
                        gb2[index] += delta[index]
                        for j in range(self.hidden):
                            g2[index][j] += delta[index] * hidden[j]
                    for j in range(self.hidden):
                        back = math.fsum(delta[index] * self.w2[index][j] for index in range(len(HEADS))) * (1 - hidden[j] ** 2)
                        gb1[j] += back
                        for column in range(width):
                            g1[j][column] += back * normalized[column]
                rate = learning_rate / max(1, count)
                for index in range(len(HEADS)):
                    self.b2[index] -= rate * gb2[index]
                    for j in range(self.hidden):
                        self.w2[index][j] -= rate * g2[index][j]
                for j in range(self.hidden):
                    self.b1[j] -= rate * gb1[j]
                    for column in range(width):
                        self.w1[j][column] -= rate * g1[j][column]
        report = {
            "schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
            "planner_version": versions[0], "epochs": epochs, "random_seed": seed,
            "train_examples": len(training), "validation_examples": len(validation),
            "skipped_unsupervised_examples": skipped,
            "observed_train_targets": dict(zip(HEADS, self.head_counts)),
            "train_seeds": sorted(set(seeds) - heldout), "validation_seeds": sorted(heldout),
            "initial_train_loss": initial, "train_loss": self._loss(training),
            "initial_validation_loss": validation_initial, "validation_loss": self._loss(validation),
            "evaluation_scope": "whole-game-seed-held-out-prediction; not search-score improvement",
            "returns": "asinh-scaled; censored gain and compute labels excluded",
            "dataset_sha256": hashlib.sha256(_canonical(sorted(rows, key=_canonical))).hexdigest(),
        }
        self.metadata.update({"planner_version": versions[0], "training": report})
        return report

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        extra = dict(metadata or {})
        if any(key in self.metadata and extra.get(key, self.metadata[key]) != self.metadata[key]
               for key in ("training", "planner_version")):
            raise ValueError("extra metadata cannot replace neural training provenance")
        payload = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
                   "feature_names": list(FEATURE_NAMES), "heads": list(HEADS), "hidden": self.hidden,
                   "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
                   "mean": self.mean, "scale": self.scale, "head_counts": self.head_counts,
                   "metadata": {**self.metadata, **extra}}
        envelope = {"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".guide-", delete=False) as stream:
                name = stream.name
                stream.write(_canonical(envelope))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, destination)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path: str | Path) -> NeuralGuide:
        try:
            envelope = json.loads(Path(path).read_bytes())
            if set(envelope) != {"payload", "sha256"}:
                raise ValueError("invalid neural envelope")
            payload = envelope["payload"]
            if envelope["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest():
                raise ValueError("neural guide checksum mismatch")
            expected = {"schema", "feature_schema", "feature_names", "heads", "hidden", "w1", "b1", "w2", "b2", "mean", "scale", "head_counts", "metadata"}
            if (set(payload) != expected or payload["schema"] != SCHEMA or payload["feature_schema"] != FEATURE_SCHEMA
                    or payload["feature_names"] != list(FEATURE_NAMES) or payload["heads"] != list(HEADS)):
                raise ValueError("incompatible neural guide schema")
            self = cls.create(hidden=payload["hidden"])
            shapes = {"w1": (self.hidden, len(FEATURE_NAMES)), "w2": (len(HEADS), self.hidden),
                      "b1": (self.hidden,), "b2": (len(HEADS),), "mean": (len(FEATURE_NAMES),), "scale": (len(FEATURE_NAMES),)}
            for name, shape in shapes.items():
                value = payload[name]
                if not isinstance(value, list) or len(value) != shape[0]:
                    raise ValueError(f"invalid neural {name} shape")
                values = value
                if len(shape) == 2:
                    if any(not isinstance(row, list) or len(row) != shape[1] for row in value):
                        raise ValueError(f"invalid neural {name} shape")
                    values = [entry for row in value for entry in row]
                if any(abs(_number(entry, name)) > 1e12 for entry in values):
                    raise ValueError("neural parameter outside numeric limits")
                if name == "scale" and any(entry <= 0 for entry in values):
                    raise ValueError("normalization scales must be positive")
                setattr(self, name, value)
            if not isinstance(payload["metadata"], dict):
                raise TypeError("neural metadata must be an object")
            counts = payload["head_counts"]
            if not isinstance(counts, list) or len(counts) != len(HEADS) or any(type(count) is not int or count < 0 for count in counts):
                raise ValueError("invalid observed-head counts")
            self.head_counts = counts
            self.metadata = payload["metadata"]
            return self
        except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("invalid neural guide file") from error


def train_guide(examples: Sequence[Mapping[str, Any]], *, seed: int = 0, hidden: int = 24,
                **kwargs: Any) -> tuple[NeuralGuide, dict[str, Any]]:
    guide = NeuralGuide.create(seed=seed, hidden=hidden)
    return guide, guide.train(examples, seed=seed, **kwargs)


def validate_guide(guide: NeuralGuide, identity: dict[str, Any], planner_version: str) -> None:
    """Validate the same activation boundary for configuration and CLI models."""
    metadata = guide.metadata
    if metadata.get("planner_version") != planner_version or metadata.get("engine_identity") != identity:
        raise ValueError("neural guide planner/runtime provenance differs")
    report = metadata.get("training", {})
    train_seeds = report.get("train_seeds", [])
    validation_seeds = report.get("validation_seeds", [])
    if not train_seeds or not validation_seeds or set(train_seeds) & set(validation_seeds):
        raise ValueError("neural guide requires disjoint whole-seed training and validation")
    for name in ("initial_validation_loss", "validation_loss"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("neural guide validation metrics must be finite")
