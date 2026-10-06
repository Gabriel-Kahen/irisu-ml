from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_adaptive_checkpoint_eval.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_exact_adaptive_checkpoint_eval", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _outcome(
    *,
    survival: int = 128,
    score: int = 100,
    gauge: int = 20_000,
    terminated: bool = False,
):
    return MODULE.ProbeOutcome(
        survival,
        score,
        0,
        gauge,
        gauge,
        terminated,
        False,
    )


def test_reserve_objective_matches_survival_and_gauge_contract() -> None:
    execute, reason = MODULE.reserve_choice(
        _outcome(survival=127, score=1000),
        _outcome(survival=128, score=0),
        maximum_gauge_debt=1000,
        rescue_score_margin=500,
    )
    assert (execute, reason) == (False, "wait-survival")

    execute, reason = MODULE.reserve_choice(
        _outcome(score=200, gauge=10_000),
        _outcome(score=100, gauge=12_000),
        maximum_gauge_debt=1000,
        rescue_score_margin=500,
    )
    assert (execute, reason) == (False, "wait-reserve")

    execute, reason = MODULE.reserve_choice(
        _outcome(score=700, gauge=10_000),
        _outcome(score=100, gauge=12_000),
        maximum_gauge_debt=1000,
        rescue_score_margin=500,
    )
    assert (execute, reason) == (True, "shot-score")


def _bundle(training=(1, 2, 3)):
    return MODULE.PolicyBundle(
        checkpoint_path="/tmp/model.pt",
        checkpoint_sha256="a" * 64,
        model_sha256="b" * 64,
        metadata={"training_seeds": list(training)},
        inference_config={"act_logit_bias": 1.0},
        factory=lambda: None,
    )


def test_seed_contract_rejects_checkpoint_mismatch_and_eval_overlap() -> None:
    with pytest.raises(ValueError, match="checkpoint-bound"):
        MODULE.validate_contract(_bundle(), [1, 2], [10])
    with pytest.raises(ValueError, match="overlap"):
        MODULE.validate_contract(_bundle(), [1, 2, 3], [3, 10])
    declared, evaluation = MODULE.validate_contract(
        _bundle(), [3, 2, 1], [10, 11]
    )
    assert declared == (1, 2, 3)
    assert evaluation == (10, 11)


def test_checkpoint_loader_requires_sha_and_binds_inference(tmp_path) -> None:
    from irisu_pointer.steering_checkpoint import save_steering_checkpoint
    from irisu_pointer.steering_learning import GoalConditionedSteeringModel
    from irisu_rl.schema import TEACHER_V1

    path = tmp_path / "policy.pt"
    digest = save_steering_checkpoint(
        path,
        GoalConditionedSteeringModel(TEACHER_V1),
        metadata={
            "training_seeds": [7, 8],
            "inference_config": {"act_logit_bias": 1.0},
        },
    )
    bundle = MODULE.load_policy_bundle(path, expected_sha256=digest)
    assert bundle.checkpoint_sha256 == digest
    assert bundle.inference_config["act_logit_bias"] == 1.0
    assert bundle.factory().artifact_sha256 == digest
    with pytest.raises(ValueError, match="differs from checkpoint metadata"):
        MODULE.load_policy_bundle(
            path,
            expected_sha256=digest,
            inference_options={"act_logit_bias": 0.0},
        )
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE.load_policy_bundle(path, expected_sha256="bad")


class _Runtime:
    identity = SimpleNamespace(
        worker_sha256="c" * 64,
        exact_library_sha256="d" * 64,
        config_sha256="e" * 64,
    )


def _fake_run_episode(runtime, bundle, seed, *, target_score, **kwargs):
    del runtime, bundle
    score = 60_000 if seed % 5 else 40_000
    row = {
        "seed": seed,
        "score": score,
        "success": score >= target_score,
        "invalid_actions": 0,
    }
    if kwargs.get("recover_exact_branch_errors", False):
        row["exact_branch_errors"] = 0
        row["exact_branch_error_events"] = []
    provenance = {
        "physics_backend": "exact",
        "runtime_attestation_sha256": "f" * 64,
        "runner_identity": {"physics_backend": "exact"},
    }
    return row, provenance


def test_promotion_contract_requires_n_median_successes_and_zero_invalid(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "run_episode", _fake_run_episode)
    seeds = tuple(range(101, 121))
    report = MODULE.evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=(1, 2, 3),
        evaluation_seeds=seeds,
    )
    assert report["episode_count"] == 20
    assert report["success_count"] == 16
    assert report["median_score"] == 60_000
    assert report["success_fraction_at_or_above_target"] == 0.8
    assert report["passed"] is True
    assert report["evaluation_seeds"] == list(seeds)
    assert len(report["report_content_sha256"]) == 64

    shard = MODULE.evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=(1, 2, 3),
        evaluation_seeds=(101,),
    )
    assert shard["promotion_eligible"] is False
    assert shard["passed"] is False


def test_one_invalid_action_fails_an_otherwise_passing_report(monkeypatch) -> None:
    def invalid_run(*args, **kwargs):
        row, provenance = _fake_run_episode(*args, **kwargs)
        row["invalid_actions"] = 1 if row["seed"] == 101 else 0
        return row, provenance

    monkeypatch.setattr(MODULE, "run_episode", invalid_run)
    report = MODULE.evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=(1, 2, 3),
        evaluation_seeds=tuple(range(101, 121)),
    )
    assert report["invalid_actions"] == 1
    assert report["passed"] is False


def test_no_evaluation_seed_suite_is_embedded() -> None:
    assert not hasattr(MODULE, "CALIBRATION_SEEDS")
    assert "--seeds" in MODULE.parser().format_help()


class _Decision:
    is_shot = True

    def primitive_actions(self):
        return (MODULE.Action(MODULE.ActionKind.LEFT_CLICK, 1.0, 1.0, 1),)


class _WaitDecision:
    is_shot = False

    def primitive_actions(self):
        return (MODULE.Action.wait(16),)


class _Policy:
    def reset(self, seed=0):
        self.seed = seed
        self.calls = 0

    def predict(self, observation):
        del observation
        self.calls += 1
        return _Decision()


class _ErrorGate:
    def __init__(self, mutate_parent=False):
        self.mutate_parent = mutate_parent

    def evaluate(self, env, *args):
        del args
        if self.mutate_parent:
            env.tick += 1
        raise MODULE.ExactWorkerError("forked counterfactual worker exited on signal 11")

    def wait_decision(self, reason):
        assert reason == "wait-exact-branch-error"
        return _WaitDecision()


class _Env:
    def __init__(self):
        self.tick = 0
        self.actions = []

    def reset(self, *, seed):
        self.tick = 0
        return self.observation(seed), {"seed": seed}

    def observation(self, seed=99):
        return {
            "seed": seed,
            "tick": self.tick,
            "score": 0,
            "level": 1,
            "gauge": 3000 - self.tick,
            "highest_chain": 0,
            "terminated": False,
            "truncated": False,
        }

    def state_hash(self):
        return 10_000 + self.tick

    def step(self, action):
        self.actions.append(action)
        assert MODULE.ActionKind.parse(action.kind) is MODULE.ActionKind.WAIT
        self.tick += 1
        return self.observation(), 0, False, False, {"invalid_action": False}


class _Session:
    def __init__(self, env):
        self.environment = env
        self.provenance_manifest = {"physics_backend": "exact"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _EpisodeRuntime:
    def __init__(self):
        self.env = _Env()

    def open_env(self, *, simulation_config):
        assert simulation_config["max_episode_ticks"] > 1
        return _Session(self.env)


def _episode_bundle():
    return MODULE.PolicyBundle(
        checkpoint_path="/tmp/model.pt",
        checkpoint_sha256="a" * 64,
        model_sha256="b" * 64,
        metadata={"training_seeds": [1]},
        inference_config={},
        factory=_Policy,
    )


def _episode_kwargs():
    return {
        "maximum_ticks": 1,
        "short_horizon": 2,
        "long_horizon": 2,
        "gauge_threshold": 3000,
        "wait_ticks": 16,
        "maximum_gauge_debt": 1000,
        "rescue_score_margin": 500,
        "target_score": 50_000,
        "trace_interval_ticks": 1,
    }


def test_v2_recovers_exact_branch_error_with_parent_preserving_wait(monkeypatch) -> None:
    gate = _ErrorGate()
    monkeypatch.setattr(MODULE, "make_gate", lambda *args, **kwargs: gate)
    runtime = _EpisodeRuntime()
    row, _ = MODULE.run_episode(
        runtime,
        _episode_bundle(),
        99,
        recover_exact_branch_errors=True,
        **_episode_kwargs(),
    )
    assert row["exact_branch_errors"] == 1
    assert row["gate_reasons"] == {"wait-exact-branch-error": 1}
    assert row["kept_shots"] == 0
    assert row["suppressed_shots"] == 1
    assert row["invalid_actions"] == 0
    event = row["exact_branch_error_events"][0]
    assert event["parent_state_unchanged"] is True
    assert event["state_hash_before"] == event["state_hash_after"] == 10_000
    assert len(event["message_sha256"]) == 64
    assert runtime.env.actions and all(
        MODULE.ActionKind.parse(action.kind) is MODULE.ActionKind.WAIT
        for action in runtime.env.actions
    )


def test_v1_still_fails_fast_and_v2_rejects_changed_parent(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE, "make_gate", lambda *args, **kwargs: _ErrorGate()
    )
    with pytest.raises(MODULE.ExactWorkerError):
        MODULE.run_episode(
            _EpisodeRuntime(), _episode_bundle(), 99, **_episode_kwargs()
        )

    monkeypatch.setattr(
        MODULE, "make_gate", lambda *args, **kwargs: _ErrorGate(True)
    )
    with pytest.raises(RuntimeError, match="altered the live parent"):
        MODULE.run_episode(
            _EpisodeRuntime(),
            _episode_bundle(),
            99,
            recover_exact_branch_errors=True,
            **_episode_kwargs(),
        )


def test_v2_report_binds_recovery_policy_and_runner(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "run_episode", _fake_run_episode)
    report = MODULE.evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=(1, 2, 3),
        evaluation_seeds=(101,),
        recover_exact_branch_errors=True,
        report_format="irisu-exact-adaptive-learned-planner-eval-v2",
        runner_path=SOURCE,
    )
    assert report["branch_error_recovery_enabled"] is True
    assert report["exact_branch_errors"] == 0
    assert report["evaluator_engine_sha256"] == MODULE.file_sha256(SOURCE)
    policy = report["planner_config"]["exact_branch_error_policy"]
    assert policy["caught_exception"] == "irisu_env.exact_ipc.ExactWorkerError"
    assert policy["require_live_parent_state_hash_unchanged"] is True
    assert report["runner_sha256"] == MODULE.file_sha256(SOURCE)
