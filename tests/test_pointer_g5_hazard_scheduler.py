from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from irisu_pointer.fast_multiaction_planner import (
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
    PlannerCandidate,
)
from irisu_pointer.g5_hazard_scheduler import (
    HazardScheduler,
    evaluate_scheduled_exact,
    fit_scheduler,
    staged_exact_select,
)
from irisu_pointer.shot_necessity import ProbeOutcome
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_progress import DirectedPairProgressTracker
from irisu_rl.actions import ActionSpec, SemanticAction


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT / "artifacts/r3/development/exact-phase0-oracle-pilot-train4-20260813-001/oracle-pilot.json",
    ROOT / "artifacts/r3/development/exact-phase0-g5-expansion-train2-20260813-001/oracle-pilot.json",
)


def test_scheduler_has_perfect_leave_one_seed_out_recall_under_budget() -> None:
    queries = [
        query
        for path in ARTIFACTS
        for episode in json.loads(path.read_text())["episodes"]
        for query in episode["queries"]
    ]
    scheduler, report = fit_scheduler(
        [query["source_public_observation"] for query in queries],
        [bool(query["safe_delayed_disagreement_ordinals"]) for query in queries],
        [query["seed"] for query in queries],
        dataset_sha256="a" * 64,
        provenance={"test": "b" * 64},
    )
    assert (scheduler.moving_threshold, scheduler.large_threshold) == (1, 1)
    assert report["recall"] == 1.0
    assert report["compute_fraction"] == 71 / 96
    assert all(row["recalled"] == row["positives"] for row in report["folds"])


def _candidate(ordinal: int, category: str) -> PlannerCandidate:
    wait = category == "wait"
    decision = SteeringDecision(
        SemanticAction.wait(16) if wait else SemanticAction.strong(0.5, 0.5),
        SteeringIntent.WAIT if wait else SteeringIntent.STEER_MATCH,
        source_body_id=None if wait else 1,
        destination_body_id=None if wait else 2,
    )
    return PlannerCandidate(ordinal, category, decision, object())


def _outcome(candidate: PlannerCandidate, horizon: int) -> CandidateOutcome:
    values = {
        "predicted-pair-strong": (100, 10_000, 9_000),
        "predicted-pair-weak": (90, 9_000, 8_000),
        "wait": (0, 8_000, 7_000),
        "top-1-pair-strong": (80, 9_500, 8_500),
    }
    score, final, minimum = values[candidate.category]
    if horizon > 2_048 and candidate.category == "top-1-pair-strong":
        score, final, minimum = 200, 13_000, 12_000
    return CandidateOutcome(
        candidate,
        ProbeOutcome(horizon, score, 1, final, minimum, False, False),
        0,
    )


def _scheduler(moving: int, large: int) -> HazardScheduler:
    return HazardScheduler(moving, large, (1, 2, 3, 4, 5, 6), "a" * 64, (("test", "b" * 64),))


def _planner() -> FastMultiActionPlanner:
    return FastMultiActionPlanner(
        lambda _decision: (),
        config=FastMultiActionConfig(probe_ticks=2_048, long_probe_ticks=2_048),
        action_spec=ActionSpec(),
    )


def test_scheduler_false_runs_only_short_and_returns_exact_base() -> None:
    candidates = tuple(_candidate(index, category) for index, category in enumerate((
        "predicted-pair-strong", "predicted-pair-weak", "wait", "top-1-pair-strong"
    )))
    calls = []
    def evaluate(subset, horizon):
        calls.append((subset, horizon))
        return tuple(_outcome(candidate, horizon) for candidate in subset)
    observation = {"bodies": [{"vx": 20.0, "vy": 0.0, "size": 60.0}] * 2}
    verdict = staged_exact_select(_planner(), observation, candidates, evaluate, _scheduler(1, 1))
    assert not verdict.compute_long
    assert verdict.selected.category == "predicted-pair-strong"
    assert [horizon for _subset, horizon in calls] == [2_048]


def test_scheduler_true_never_selects_until_long_exact_rule() -> None:
    candidates = tuple(_candidate(index, category) for index, category in enumerate((
        "predicted-pair-strong", "predicted-pair-weak", "wait", "top-1-pair-strong"
    )))
    calls = []
    def evaluate(subset, horizon):
        calls.append((subset, horizon))
        return tuple(_outcome(candidate, horizon) for candidate in subset)
    observation = {"bodies": []}
    verdict = staged_exact_select(_planner(), observation, candidates, evaluate, _scheduler(1, 1))
    assert verdict.compute_long
    assert verdict.selected.category == "top-1-pair-strong"
    assert [horizon for _subset, horizon in calls] == [2_048, 8_192, 12_288]
    assert all(candidate is frozen for candidate, frozen in zip(calls[0][0], candidates))


def test_malformed_scheduler_features_buy_long_compute_and_long_failure_keeps_base() -> None:
    candidates = tuple(_candidate(index, category) for index, category in enumerate((
        "predicted-pair-strong", "predicted-pair-weak", "wait"
    )))
    calls = []
    def evaluate(subset, horizon):
        calls.append(horizon)
        if horizon > 2_048:
            raise RuntimeError("worker failed")
        return tuple(_outcome(candidate, horizon) for candidate in subset)
    verdict = staged_exact_select(
        _planner(), {"bodies": [{"size": 60.0}]}, candidates, evaluate,
        _scheduler(1, 1),
    )
    assert verdict.compute_long
    assert verdict.reason == "long-exact-failure-retains-base"
    assert verdict.selected is verdict.base
    assert calls == [2_048, 8_192]


class _RuntimePolicy:
    def __init__(self, observation, decision) -> None:
        self._progress = DirectedPairProgressTracker()
        self._progress.begin(observation, decision.source_body_id, decision.destination_body_id)
        self._last_decision = decision

    def predict(self, _observation):
        return SteeringDecision(SemanticAction.wait(128), SteeringIntent.WAIT)


class _RuntimeBranch:
    physics_backend = "exact"

    def __init__(self, state, owner) -> None:
        self.state = deepcopy(state)
        self.owner = owner

    def state_hash(self):
        return hash(tuple(sorted(self.state.items())))

    def step(self, action):
        self.state["tick"] += 1
        if int(action.kind) in (1, 2) and not self.state["shot_seen"]:
            self.state["shot_seen"] = 1
            self.state["score"] += 20 if int(action.kind) == 2 else 10
        return dict(self.state), 0.0, False, False, {"invalid_action": False}

    def close(self):
        self.owner.closed += 1


class _RuntimeCheckpoint:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.state = deepcopy(owner.state)

    def branch(self):
        self.owner.opened += 1
        return _RuntimeBranch(self.state, self.owner)

    def close(self):
        self.owner.checkpoint_closed = True


class _RuntimeEnv:
    physics_backend = "exact"

    def __init__(self) -> None:
        self.state = {
            "tick": 0, "score": 0, "gauge": 40_000,
            "qualifying_clear_count": 0, "shot_seen": 0,
        }
        self.opened = self.closed = 0
        self.checkpoint_closed = False

    def state_hash(self):
        return hash(tuple(sorted(self.state.items())))

    def fast_checkpoint(self):
        return _RuntimeCheckpoint(self)


def _runtime_observation():
    bodies = [
        {
            "id": body_id, "kind": "piece", "lifecycle": "fresh", "color": 2,
            "chain_id": 0, "x": 100.0 * body_id, "y": 100.0, "size": 60.0,
            "vx": 20.0, "vy": 0.0,
        }
        for body_id in (1, 2, 3)
    ]
    return {
        "tick": 0, "score": 0, "gauge": 40_000,
        "qualifying_clear_count": 0, "bodies": bodies,
    }


def test_exact_runtime_freezes_candidates_once_and_preserves_parent(monkeypatch) -> None:
    observation = _runtime_observation()
    prediction = SteeringDecision(
        SemanticAction.strong(0.25, 0.25), SteeringIntent.STEER_MATCH,
        source_body_id=1, destination_body_id=2, destination_chain_id=0,
    )
    before = _RuntimePolicy(observation, prediction)
    after = deepcopy(before)
    env = _RuntimeEnv()
    planner = _planner()
    original = planner.candidates
    calls = 0
    def candidates(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "candidates", candidates)
    monkeypatch.setattr(
        planner,
        "_advance",
        lambda _env, _observation, _policy, decision, horizon: (
            ProbeOutcome(
                horizon,
                20 if decision.is_shot else 0,
                0,
                40_000,
                40_000,
                False,
                False,
            ),
            0,
            0,
        ),
    )

    verdict = evaluate_scheduled_exact(
        planner, env, observation, before, after, prediction, _scheduler(1, 1)
    )

    assert calls == 1
    assert not verdict.compute_long
    assert verdict.used_fast_checkpoint
    assert verdict.branch_checks == 4
    assert env.opened == env.closed == 3
    assert env.checkpoint_closed
    assert env.state["tick"] == 0


def test_exact_runtime_trigger_uses_same_frozen_objects_for_all_stages(monkeypatch) -> None:
    observation = _runtime_observation()
    for body in observation["bodies"]:
        body.update({"vx": 0.0, "vy": 0.0, "size": 20.0})
    prediction = SteeringDecision(
        SemanticAction.strong(0.25, 0.25), SteeringIntent.STEER_MATCH,
        source_body_id=1, destination_body_id=2, destination_chain_id=0,
    )
    before = _RuntimePolicy(observation, prediction)
    env = _RuntimeEnv()
    planner = _planner()
    monkeypatch.setattr(
        planner,
        "_advance",
        lambda _env, _observation, _policy, decision, horizon: (
            ProbeOutcome(horizon, int(decision.is_shot), 0, 40_000, 40_000, False, False),
            0,
            0,
        ),
    )

    verdict = evaluate_scheduled_exact(
        planner, env, observation, before, deepcopy(before), prediction,
        _scheduler(1, 1),
    )

    assert verdict.compute_long
    assert tuple(horizon for horizon, _ordinals in verdict.evaluated_ordinals) == (
        2_048, 8_192, 12_288,
    )
    assert verdict.branch_checks == 10
    assert env.opened == env.closed == 9
    assert env.state["tick"] == 0
