from __future__ import annotations

from copy import deepcopy

from irisu_env import ActionKind, ExactWorkerError
from irisu_pointer.development_proposal_blend import (
    FailClosedProposalBlendPlanner,
    ProposalBlendConfig,
)
from irisu_pointer.fast_multiaction_planner import (
    FastMultiActionConfig,
    FastMultiActionPlanner,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_progress import DirectedPairProgressTracker
from irisu_rl.actions import ActionSpec, SemanticAction


def _observation(*, tick: int = 50_001, gauge: int = 40_000):
    return {
        "tick": tick,
        "score": 0,
        "gauge": gauge,
        "qualifying_clear_count": 0,
        "bodies": [
            {
                "id": value,
                "kind": "piece",
                "lifecycle": "fresh",
                "color": 1,
                "chain_id": 0,
                "x": float(value * 100),
                "y": 100.0,
                "size": 20.0,
            }
            for value in (1, 2, 3)
        ],
    }


def _shot(source: int = 1, destination: int = 2, x: float = 0.25):
    return SteeringDecision(
        SemanticAction.strong(x, 0.25),
        SteeringIntent.STEER_MATCH,
        source_body_id=source,
        destination_body_id=destination,
        destination_chain_id=0,
    )


class _Policy:
    def __init__(self, observation, decision):
        self._progress = DirectedPairProgressTracker()
        self._progress.begin(
            observation, decision.source_body_id, decision.destination_body_id
        )
        self._last_decision = decision

    def _analytic_action(self, source, _destination):
        x = float(source["id"]) / 4.0
        return SemanticAction.strong(x, 0.25), -0.5, 0.75

    def predict(self, _observation):
        return SteeringDecision(
            SemanticAction.wait(1), SteeringIntent.WAIT, reason="test continuation"
        )


class _Ranker:
    def __init__(self, pair=(3, 2), *, fail=False):
        self.pair = pair
        self.fail = fail
        self.calls = 0

    def ranked_pairs(self, _observation, *, excluded_pairs, maximum_pairs):
        self.calls += 1
        assert maximum_pairs == 1
        if self.fail:
            raise RuntimeError("ranker failed")
        return () if self.pair in excluded_pairs else (self.pair,)


class _Branch:
    physics_backend = "exact"

    def __init__(self, state, owner):
        self.state = deepcopy(state)
        self.owner = owner

    def state_hash(self):
        return hash(tuple(sorted(self.state.items())))

    def step(self, action):
        kind = ActionKind.parse(action.kind)
        self.state["tick"] += 1
        if kind is not ActionKind.WAIT and not self.state["shot_seen"]:
            self.state["shot_seen"] = 1
            residual = float(action.cursor_x) > 300.0
            if residual and self.owner.fail_residual:
                raise ExactWorkerError("residual branch worker failed")
            if residual:
                self.state["score"] += self.owner.residual_score
                self.state["gauge"] -= self.owner.residual_gauge_cost
            elif kind is ActionKind.STRONG_SHOT:
                self.state["score"] += 20
                self.state["gauge"] -= 200
            else:
                self.state["score"] += 10
                self.state["gauge"] -= 10
        return dict(self.state), 0.0, False, False, {"invalid_action": False}

    def close(self):
        return None


class _Checkpoint:
    def __init__(self, env):
        self.env = env
        self.state = deepcopy(env.state)

    def branch(self):
        return _Branch(self.state, self.env)

    def close(self):
        return None


class _Env:
    physics_backend = "exact"

    def __init__(
        self,
        observation,
        *,
        residual_score=100,
        residual_gauge_cost=500,
        fail_residual=False,
    ):
        self.state = {
            "tick": observation["tick"],
            "score": observation["score"],
            "gauge": observation["gauge"],
            "qualifying_clear_count": 0,
            "shot_seen": 0,
        }
        self.residual_score = residual_score
        self.residual_gauge_cost = residual_gauge_cost
        self.fail_residual = fail_residual

    def state_hash(self):
        return hash(tuple(sorted(self.state.items())))

    def fast_checkpoint(self):
        return _Checkpoint(self)


def _config():
    return FastMultiActionConfig(
        probe_ticks=2,
        long_probe_ticks=2,
        wait_ticks=1,
        low_gauge_threshold=20_000,
        top_k_pairs=0,
        maximum_gauge_debt=100,
        rescue_score_margin=5,
        gauge_advantage=1,
    )


def _planner(ranker):
    spec = ActionSpec()
    return FailClosedProposalBlendPlanner(
        lambda decision: decision.primitive_actions(spec),
        ranker,
        config=_config(),
        blend_config=ProposalBlendConfig(),
        action_spec=spec,
    )


def test_base_candidates_are_an_exact_prefix_and_base_state_is_immutable():
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    after = deepcopy(before)
    spec = ActionSpec()
    base = FastMultiActionPlanner(
        lambda decision: decision.primitive_actions(spec),
        config=_config(),
        action_spec=spec,
    ).candidates(observation, before, after, prediction)

    blended = _planner(_Ranker()).candidates(
        observation, before, after, prediction
    )

    assert [value.decision for value in blended[: len(base)]] == [
        value.decision for value in base
    ]
    assert [value.category for value in blended[len(base) :]] == [
        "residual-pair-weak",
        "residual-pair-strong",
    ]
    pending = after._progress.pending_pair
    assert (pending.source_id, pending.destination_id) == (1, 2)
    for value in blended[len(base) :]:
        pending = value.continuation_policy._progress.pending_pair
        assert (pending.source_id, pending.destination_id) == (3, 2)


def test_activation_boundaries_and_ranker_error_fail_closed_to_base():
    prediction = _shot()
    ranker = _Ranker(fail=True)
    planner = _planner(ranker)
    for tick, gauge, expected_calls in (
        (50_000, 20_000, 0),
        (50_001, 20_000, 1),
        (50_000, 19_999, 2),
    ):
        observation = _observation(tick=tick, gauge=gauge)
        before = _Policy(observation, prediction)
        candidates = planner.candidates(
            observation, before, deepcopy(before), prediction
        )
        assert [value.category for value in candidates] == [
            "predicted-pair-strong",
            "predicted-pair-weak",
            "wait",
        ]
        assert ranker.calls == expected_calls
    assert planner.proposal_counts["ranker-error-base-fallback"] == 2


def test_exact_residual_branch_is_vetoed_when_it_spends_more_reserve():
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner(_Ranker()).evaluate(
        _Env(observation, residual_score=100, residual_gauge_cost=500),
        observation,
        before,
        deepcopy(before),
        prediction,
    )

    assert verdict.selected.category == "predicted-pair-strong"
    assert verdict.reason.endswith("residual-safety-veto")
    assert verdict.objective_evidence == {"residual_safety_veto": True}


def test_exact_residual_branch_can_win_only_when_reserve_noninferior():
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner(_Ranker()).evaluate(
        _Env(observation, residual_score=100, residual_gauge_cost=0),
        observation,
        before,
        deepcopy(before),
        prediction,
    )

    assert verdict.selected.category.startswith("residual-pair-")
    assert verdict.reason.endswith("residual-safe-override")
    assert verdict.objective_evidence == {"residual_safety_veto": False}


def test_residual_exact_worker_error_discards_residuals_not_base_verdict():
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner(_Ranker()).evaluate(
        _Env(observation, fail_residual=True),
        observation,
        before,
        deepcopy(before),
        prediction,
    )

    assert verdict.selected.category == "predicted-pair-strong"
    assert verdict.reason.endswith("residual-branch-error-base-fallback")
    assert verdict.objective_evidence == {
        "residual_branch_error_base_fallback": True
    }


def test_blend_manifest_binds_base_only_continuation_and_pair_budget():
    manifest = _planner(_Ranker()).manifest()
    assert manifest["continuation_policy"] == "base-only"
    assert manifest["base_candidate_invariant"] == "exact-prefix-subset"
    assert manifest["residual_pair_budget"] == 1
