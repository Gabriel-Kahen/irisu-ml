from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_multiseed_policy_gate.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_multiseed_policy_gate", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_locked_calibration_manifest_is_fresh_and_hash_bound() -> None:
    assert len(MODULE.CALIBRATION_SEEDS) == 20
    assert len(set(MODULE.CALIBRATION_SEEDS)) == 20
    manifest = MODULE.seed_manifest([1, 2, 3])
    assert manifest["training_seeds"] == [1, 2, 3]
    assert manifest["calibration_seeds"] == list(MODULE.CALIBRATION_SEEDS)
    digest_input = dict(manifest)
    del digest_input["manifest_sha256"]
    assert manifest["manifest_sha256"] == MODULE.canonical_sha256(digest_input)


def test_calibration_overlap_and_duplicate_declarations_fail_closed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MODULE.seed_manifest([MODULE.CALIBRATION_SEEDS[4]])
    with pytest.raises(ValueError, match="unique"):
        MODULE.seed_manifest([7, 7])


def test_existing_goal_conditioned_checkpoint_is_first_class_loader(tmp_path) -> None:
    from irisu_pointer.steering_checkpoint import save_steering_checkpoint
    from irisu_pointer.steering_learning import GoalConditionedSteeringModel
    from irisu_rl.schema import TEACHER_V1

    checkpoint = tmp_path / "policy.pt"
    model = GoalConditionedSteeringModel(TEACHER_V1)
    digest = save_steering_checkpoint(
        checkpoint,
        model,
        metadata={
            "training_seeds": [11, 12],
            "inference_config": {"act_logit_bias": 1.0},
        },
    )
    bundle = MODULE.load_policy_bundle(checkpoint, expected_sha256=digest)
    assert bundle.format == "irisu-goal-conditioned-steering-checkpoint-v1"
    assert bundle.checkpoint_sha256 == digest
    assert len(bundle.model_sha256) == 64
    assert bundle.metadata["training_seeds"] == [11, 12]
    assert bundle.inference_config["act_logit_bias"] == 1.0
    assert bundle.factory().artifact_sha256 == digest
    with pytest.raises(ValueError, match="differs from checkpoint metadata"):
        MODULE.load_policy_bundle(
            checkpoint,
            expected_sha256=digest,
            options={"act_logit_bias": 0.0},
        )


class _Decision:
    def __init__(self, wait_ticks: int = 1) -> None:
        self.wait_ticks = wait_ticks

    def primitive_actions(self):
        return (MODULE.Action.wait(self.wait_ticks),)


class _Policy:
    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.calls = 0

    def predict(self, observation):
        del observation
        self.calls += 1
        if self.seed == MODULE.CALIBRATION_SEEDS[0] and self.calls == 1:
            return _Decision(0)
        return _Decision()


def _observation(seed: int, tick: int, score: int = 0):
    return {
        "seed": seed,
        "tick": tick,
        "score": score,
        "terminated": tick >= 2,
        "truncated": False,
    }


class _FakeVector:
    def __init__(self, lanes: int) -> None:
        self.lanes = lanes
        self.states = {}
        self.reset_many_calls = 0

    def reset(self, *, seed):
        for lane, value in enumerate(seed):
            self.states[lane] = _observation(value, 0)
        return (
            [self.states[lane] for lane in range(self.lanes)],
            [{"seed": value} for value in seed],
        )

    def reset_many(self, indices, *, seeds):
        self.reset_many_calls += 1
        values = []
        for lane, seed in zip(indices, seeds):
            self.states[lane] = _observation(seed, 0)
            values.append(self.states[lane])
        return values

    def step_many(self, indices, actions):
        observations, terminated, infos = [], [], []
        for lane, action in zip(indices, actions):
            previous = self.states[lane]
            tick = previous["tick"] + int(action.wait_ticks)
            rank = MODULE.CALIBRATION_SEEDS.index(previous["seed"])
            score = (60_000 if rank < 16 else 40_000) if tick >= 2 else 0
            value = _observation(previous["seed"], tick, score)
            self.states[lane] = value
            observations.append(value)
            terminated.append(tick >= 2)
            infos.append(
                {
                    "invalid_action": previous["seed"]
                    == MODULE.CALIBRATION_SEEDS[1]
                    and tick == 1,
                    "diagnostics": SimpleNamespace(
                        terminal_metadata_recorded=False
                    ),
                }
            )
        return observations, [0] * len(indices), terminated, [False] * len(indices), infos


class _Session:
    def __init__(self, vector) -> None:
        self.environment = vector
        self.provenance_manifest = {
            "runtime_attestation_sha256": "d" * 64,
            "runner_identity": {"physics_backend": "exact", "lanes": vector.lanes},
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _Runtime:
    def __init__(self) -> None:
        self.identity = SimpleNamespace(
            worker_sha256="a" * 64,
            exact_library_sha256="b" * 64,
            config_sha256="c" * 64,
        )
        self.vector = None

    def open_vector(self, lanes, *, simulation_config, workers):
        assert simulation_config == {"max_episode_ticks": 2}
        assert workers == 2
        self.vector = _FakeVector(lanes)
        return _Session(self.vector)


def test_vector_gate_recycles_lanes_and_reports_50k_success_and_invalids() -> None:
    runtime = _Runtime()
    bundle = MODULE.PolicyBundle(
        format="test-policy-v1",
        checkpoint_sha256="e" * 64,
        model_sha256="f" * 64,
        metadata={"training_seeds": [1, 2, 3]},
        factory=_Policy,
        inference_config={},
    )
    report = MODULE.evaluate_bundle(
        runtime,
        bundle,
        declared_training_seeds=(1, 2, 3),
        lanes=3,
        workers=2,
        maximum_ticks=2,
    )
    assert [row["seed"] for row in report["episodes"]] == list(
        MODULE.CALIBRATION_SEEDS
    )
    assert report["median_score"] == 60_000.0
    assert report["success_count"] == 16
    assert report["success_fraction_at_or_above_50k"] == 0.8
    assert report["invalid_actions"] == 2
    assert report["passed"] is False
    assert runtime.vector.reset_many_calls > 0
    assert report["runtime_hashes"]["worker_sha256"] == "a" * 64


def test_gate_rejects_training_manifest_not_bound_to_checkpoint() -> None:
    runtime = _Runtime()
    bundle = MODULE.PolicyBundle(
        format="test-policy-v1",
        checkpoint_sha256="e" * 64,
        model_sha256="f" * 64,
        metadata={"training_seeds": [1, 2]},
        factory=_Policy,
        inference_config={},
    )
    with pytest.raises(ValueError, match="checkpoint-bound"):
        MODULE.evaluate_bundle(
            runtime,
            bundle,
            declared_training_seeds=(1, 2, 3),
            maximum_ticks=2,
        )


def test_terminal_recorded_score_overrides_post_finish_observation() -> None:
    info = {
        "diagnostics": SimpleNamespace(
            terminal_metadata_recorded=True, recorded_final_score=51_234
        )
    }
    assert MODULE._effective_score({"score": 0}, info) == 51_234
