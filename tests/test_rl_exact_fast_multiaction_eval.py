from __future__ import annotations

from copy import deepcopy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from irisu_env import ActionKind
from irisu_pointer.fast_multiaction_planner import FastMultiActionConfig
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_progress import DirectedPairProgressTracker
from irisu_rl.actions import ActionSpec, SemanticAction


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_exact_fast_multiaction_eval.py"
SPEC = importlib.util.spec_from_file_location("rl_exact_fast_multiaction_eval", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PolicyBundle = MODULE.PolicyBundle
evaluate_bundle = MODULE.evaluate_bundle
parser = MODULE.parser


OBSERVATION = {
    "tick": 0,
    "score": 0,
    "level": 1,
    "gauge": 40_000,
    "highest_chain": 0,
    "qualifying_clear_count": 0,
    "bodies": [
        {
            "id": 1,
            "kind": "piece",
            "lifecycle": "fresh",
            "color": 1,
            "chain_id": 0,
            "x": 100.0,
            "y": 100.0,
            "size": 20.0,
        },
        {
            "id": 2,
            "kind": "piece",
            "lifecycle": "fresh",
            "color": 1,
            "chain_id": 0,
            "x": 200.0,
            "y": 100.0,
            "size": 20.0,
        },
    ],
}


class _Policy:
    def __init__(self) -> None:
        self.action_spec = ActionSpec()
        self._progress = DirectedPairProgressTracker()
        self._last_decision = None

    def reset(self, _seed: int = 0) -> None:
        self._progress.reset()
        self._last_decision = None

    def predict(self, observation):
        if int(observation["tick"]) > 0:
            decision = SteeringDecision(
                SemanticAction.wait(1), SteeringIntent.WAIT, reason="test wait"
            )
        else:
            decision = SteeringDecision(
                SemanticAction.strong(0.25, 0.25),
                SteeringIntent.STEER_MATCH,
                source_body_id=1,
                destination_body_id=2,
                destination_chain_id=0,
            )
            self._progress.begin(observation, 1, 2)
        self._last_decision = decision
        return decision


class _Branch:
    physics_backend = "exact"

    def __init__(self, state):
        self.state = deepcopy(state)

    def state_hash(self):
        return hash(
            tuple(
                sorted(
                    (key, value)
                    for key, value in self.state.items()
                    if key != "bodies"
                )
            )
        )

    def step(self, action):
        kind = ActionKind.parse(action.kind)
        self.state["tick"] += 1
        if kind is ActionKind.STRONG_SHOT:
            self.state["score"] += 20
        elif kind is ActionKind.WEAK_SHOT:
            self.state["score"] += 10
        return deepcopy(self.state), 0.0, False, False, {"invalid_action": False}

    def close(self):
        pass


class _Checkpoint:
    def __init__(self, env):
        self.env = env

    def branch(self):
        return _Branch(self.env.state)

    def close(self):
        pass


class _Env(_Branch):
    def __init__(self):
        super().__init__(OBSERVATION)

    def reset(self, *, seed):
        self.state = deepcopy(OBSERVATION)
        return deepcopy(self.state), {"seed": seed}

    def fast_checkpoint(self):
        return _Checkpoint(self)


class _Session:
    def __init__(self):
        self.environment = _Env()
        self.provenance_manifest = {
            "physics_backend": "exact",
            "runtime": "fake-exact",
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class _Runtime:
    identity = SimpleNamespace(
        worker_sha256="a" * 64,
        exact_library_sha256="b" * 64,
        config_sha256="c" * 64,
    )

    def open_env(self, *, simulation_config):
        assert simulation_config["max_episode_ticks"] == 4
        return _Session()


class _LongRuntime(_Runtime):
    def open_env(self, *, simulation_config):
        assert simulation_config["max_episode_ticks"] == 6
        return _Session()


def _bundle() -> PolicyBundle:
    return PolicyBundle(
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="d" * 64,
        model_sha256="e" * 64,
        metadata={"training_seeds": [1]},
        inference_config={},
        factory=_Policy,
    )


def test_development_report_binds_planner_and_never_claims_promotion() -> None:
    report = evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=[1],
        evaluation_seeds=[2],
        maximum_ticks=2,
        planner_config=FastMultiActionConfig(
            probe_ticks=2,
            long_probe_ticks=2,
            wait_ticks=1,
            low_gauge_threshold=0,
            top_k_pairs=0,
            maximum_gauge_debt=100,
            rescue_score_margin=5,
            gauge_advantage=1,
        ),
        target_score=10,
        trace_interval_ticks=1,
        maximum_logged_queries=1,
    )

    assert report["evidence_class"] == "development-only"
    assert report["promotion_eligible"] is False
    assert report["training_evaluation_overlap"] == []
    assert len(report["planner_source_sha256"]) == 64
    assert report["planner_config"]["objective_mode"] == "wait-relative"
    assert report["planner_objective"].startswith("wait-relative")
    assert report["reserve_comparator_source_sha256"] is None
    assert report["episodes"][0]["selected_categories"] == {
        "predicted-pair-strong": 1
    }
    assert report["invalid_actions"] == 0
    assert report["planner_probe_mode_counts"] == {"normal": 1}
    assert report["planner_probe_horizon_counts"] == {"2": 1}
    assert report["episodes"][0]["query_log"][0]["probe_mode"] == "normal"
    assert report["episodes"][0]["query_log"][0]["probe_ticks"] == 2


def test_cli_requires_caller_supplied_seeds() -> None:
    action = parser().get_default("seeds")
    assert action is None
    assert parser().get_default("objective_mode") == "wait-relative"
    assert parser().get_default("probe_ticks") == 256
    assert parser().get_default("long_probe_ticks") == 512
    assert parser().get_default("low_gauge_threshold") == 20_000
    assert parser().get_default("low_gauge_exit_threshold") == 30_000


def test_development_report_binds_low_gauge_long_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(OBSERVATION, "gauge", 19_999)
    report = evaluate_bundle(
        _LongRuntime(),
        _bundle(),
        declared_training_seeds=[1],
        evaluation_seeds=[2],
        maximum_ticks=2,
        planner_config=FastMultiActionConfig(
            probe_ticks=2,
            long_probe_ticks=4,
            wait_ticks=1,
            low_gauge_threshold=20_000,
            low_gauge_exit_threshold=30_000,
            top_k_pairs=0,
            maximum_gauge_debt=100,
            rescue_score_margin=5,
            gauge_advantage=1,
        ),
        target_score=10,
        trace_interval_ticks=1,
        maximum_logged_queries=1,
    )

    episode = report["episodes"][0]
    assert episode["planner_probe_mode_counts"] == {"low-gauge-long": 1}
    assert episode["planner_probe_horizon_counts"] == {"4": 1}
    assert episode["planner_probe_mode_switches"] == [
        {
            "tick": 0,
            "gauge": 19_999,
            "from": "normal",
            "to": "low-gauge-long",
            "probe_ticks": 4,
        }
    ]
    assert episode["planner_probe_schedule"] == [
        {
            "tick": 0,
            "gauge": 19_999,
            "probe_mode": "low-gauge-long",
            "probe_ticks": 4,
        }
    ]
    assert len(episode["planner_probe_schedule_sha256"]) == 64
    assert episode["query_log"][0]["probe_ticks"] == 4
    assert report["planner_probe_mode_counts"] == {"low-gauge-long": 1}
    assert report["planner_probe_horizon_counts"] == {"4": 1}
    assert report["planner_probe_mode_switch_count"] == 1


def test_development_report_exposes_reserve_band_objective() -> None:
    report = evaluate_bundle(
        _Runtime(),
        _bundle(),
        declared_training_seeds=[1],
        evaluation_seeds=[2],
        maximum_ticks=2,
        planner_config=FastMultiActionConfig(
            probe_ticks=2,
            long_probe_ticks=2,
            wait_ticks=1,
            low_gauge_threshold=0,
            top_k_pairs=0,
            maximum_gauge_debt=100,
            rescue_score_margin=5,
            gauge_advantage=1,
            objective_mode="reserve-band",
            reserve_contingency_gauge=3_000,
        ),
        target_score=10,
        trace_interval_ticks=1,
        maximum_logged_queries=1,
    )
    assert report["planner_config"]["objective_mode"] == "reserve-band"
    assert report["planner_objective"].startswith("liability-adjusted")
    assert len(report["reserve_comparator_source_sha256"]) == 64
    query = report["episodes"][0]["query_log"][0]
    assert query["objective_mode"] == "reserve-band"
    assert "source_visible_rot_liability" in query["objective_evidence"]
