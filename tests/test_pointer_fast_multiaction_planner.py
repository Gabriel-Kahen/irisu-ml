from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from irisu_env import ActionKind, IrisuEnv
from irisu_pointer.fast_multiaction_planner import (
    FastMultiActionConfig,
    FastMultiActionPlanner,
    _safe_pair,
    visible_rot_liability,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_progress import DirectedPairProgressTracker
from irisu_rl.actions import ActionSpec, SemanticAction


RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/libirisu_clone.so"
)


class _Policy:
    def __init__(self, observation: dict[str, object], decision: SteeringDecision) -> None:
        self._progress = DirectedPairProgressTracker()
        self._progress.begin(
            observation,
            int(decision.source_body_id),
            int(decision.destination_body_id),
        )
        self._last_decision = decision

    def predict(self, _observation: dict[str, object]) -> SteeringDecision:
        return SteeringDecision(
            SemanticAction.wait(1), SteeringIntent.WAIT, reason="test continuation"
        )


class _Branch:
    physics_backend = "exact"

    def __init__(self, state: dict[str, int], owner: "_ExactEnv") -> None:
        self.state = deepcopy(state)
        self.owner = owner

    def state_hash(self) -> int:
        return hash(tuple(sorted(self.state.items())))

    def step(self, action: object):
        kind = ActionKind.parse(getattr(action, "kind"))
        self.state["tick"] += 1
        invalid = False
        if kind is ActionKind.STRONG_SHOT and not self.state["shot_seen"]:
            self.state["shot_seen"] = 1
            self.state["score"] += 20
            self.state["gauge"] -= 200
            self.state["highest_chain"] = 5
            invalid = self.owner.invalidate_strong
        elif kind is ActionKind.WEAK_SHOT and not self.state["shot_seen"]:
            self.state["shot_seen"] = 1
            self.state["score"] += 10
            self.state["gauge"] -= 10
            self.state["highest_chain"] = 6
        observation = dict(self.state)
        return observation, 0.0, False, False, {"invalid_action": invalid}

    def close(self) -> None:
        self.owner.closed += 1


class _Checkpoint:
    def __init__(self, env: "_ExactEnv") -> None:
        self.env = env
        self.state = deepcopy(env.state)

    def branch(self) -> _Branch:
        self.env.opened += 1
        return _Branch(self.state, self.env)

    def close(self) -> None:
        self.env.checkpoint_closed = True


class _ExactEnv:
    physics_backend = "exact"

    def __init__(self, *, invalidate_strong: bool = False) -> None:
        self.state = {
            "tick": 0,
            "score": 0,
            "gauge": 40_000,
            "qualifying_clear_count": 0,
            "shot_seen": 0,
            "highest_chain": 0,
        }
        self.invalidate_strong = invalidate_strong
        self.opened = self.closed = self.clone_calls = 0
        self.checkpoint_closed = False

    def state_hash(self) -> int:
        return hash(tuple(sorted(self.state.items())))

    def fast_checkpoint(self) -> _Checkpoint:
        return _Checkpoint(self)

    def clone_state(self) -> bytes:
        self.clone_calls += 1
        raise AssertionError("fast exact planning must not clone/replay")


def _observation(gauge: int = 40_000) -> dict[str, object]:
    return {
        "tick": 0,
        "score": 0,
        "gauge": gauge,
        "qualifying_clear_count": 0,
        "bodies": [
            {
                "id": 1,
                "kind": "piece",
                "lifecycle": "fresh",
                "color": 2,
                "chain_id": 0,
                "x": 100.0,
                "y": 100.0,
                "size": 20.0,
            },
            {
                "id": 2,
                "kind": "piece",
                "lifecycle": "fresh",
                "color": 2,
                "chain_id": 0,
                "x": 200.0,
                "y": 100.0,
                "size": 20.0,
            },
            {
                "id": 3,
                "kind": "piece",
                "lifecycle": "fresh",
                "color": 2,
                "chain_id": 0,
                "x": 300.0,
                "y": 100.0,
                "size": 20.0,
            },
        ],
    }


def _shot(source: int = 1, destination: int = 2) -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(0.25, 0.25),
        SteeringIntent.STEER_MATCH,
        source_body_id=source,
        destination_body_id=destination,
        destination_chain_id=0,
    )


def _planner(**overrides: object) -> FastMultiActionPlanner:
    values = {
        "probe_ticks": 2,
        "long_probe_ticks": 2,
        "wait_ticks": 1,
        "low_gauge_threshold": 30_000,
        "top_k_pairs": 1,
        "maximum_gauge_debt": 100,
        "rescue_score_margin": 5,
        "gauge_advantage": 1,
    }
    values.update(overrides)
    spec = ActionSpec()
    return FastMultiActionPlanner(
        lambda decision: decision.primitive_actions(spec),
        config=FastMultiActionConfig(**values),
        action_spec=spec,
    )


def test_exact_evaluation_uses_cow_and_rejects_invalid_strong() -> None:
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    after = deepcopy(before)
    env = _ExactEnv(invalidate_strong=True)

    verdict = _planner().evaluate(env, observation, before, after, prediction)

    assert verdict.used_fast_checkpoint
    assert verdict.selected.category == "predicted-pair-weak"
    assert env.state["tick"] == 0
    assert env.clone_calls == 0
    assert env.opened == env.closed == 3
    assert env.checkpoint_closed
    strong = next(
        value for value in verdict.outcomes if value.candidate.category.endswith("strong")
    )
    assert strong.invalid_actions == 1


def test_low_gauge_adds_ranked_pair_strengths_and_rebinds_only_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(10_000)
    prediction = _shot()
    before = _Policy(observation, prediction)
    after = deepcopy(before)
    alternate = _shot(1, 3)
    monkeypatch.setattr(
        "irisu_pointer.fast_multiaction_planner._ranked_alternatives",
        lambda *_args, **_kwargs: (alternate,),
    )

    candidates = _planner().candidates(observation, before, after, prediction)

    assert [value.category for value in candidates] == [
        "predicted-pair-strong",
        "predicted-pair-weak",
        "wait",
        "top-1-pair-weak",
        "top-1-pair-strong",
    ]
    assert (
        after._progress.pending_pair.source_id,
        after._progress.pending_pair.destination_id,
    ) == (1, 2)
    alternative_state = next(
        value.continuation_policy
        for value in candidates
        if value.category == "top-1-pair-weak"
    )
    pending = alternative_state._progress.pending_pair
    assert (pending.source_id, pending.destination_id) == (1, 3)


def test_exact_numpy_body_ids_are_available_to_ranked_alternatives() -> None:
    observation = _observation(10_000)
    for body in observation["bodies"]:
        body["id"] = np.uint32(body["id"])
        body["chain_id"] = np.uint32(body["chain_id"])
        body["color"] = np.int32(body["color"])

    pair = _safe_pair(object(), observation, 1, 3)

    assert pair is not None
    assert tuple(int(body["id"]) for body in pair) == (1, 3)


def test_high_gauge_candidate_set_is_deterministic_and_bounded() -> None:
    observation = _observation(40_000)
    prediction = _shot()
    before = _Policy(observation, prediction)
    after = deepcopy(before)
    planner = _planner()

    first = planner.candidates(observation, before, after, prediction)
    second = planner.candidates(observation, before, after, prediction)

    assert [value.category for value in first] == [
        "predicted-pair-strong",
        "predicted-pair-weak",
        "wait",
    ]
    assert [value.decision for value in first] == [value.decision for value in second]


def test_visible_rot_liability_uses_public_due_timers_and_level_penalty() -> None:
    observation = _observation(10_000)
    observation["level"] = 35
    observation["bodies"] = [
        {"kind": "piece", "lifecycle": "fresh", "rot_timer": 40},
        {"kind": "bonus", "lifecycle": "fresh", "rot_timer": 39},
        {"kind": "piece", "lifecycle": "rotten", "rot_timer": 40},
        {"kind": "projectile", "lifecycle": "fresh", "rot_timer": 40},
        {"kind": "piece", "lifecycle": "fresh", "rot_timer": 1},
    ]
    liability, count = visible_rot_liability(
        observation, horizon_ticks=2, rot_delay_ticks=40
    )
    assert count == 2
    assert liability == 2 * (1800 + 20 * 35)


def test_optional_reserve_band_selects_solvent_weak_and_binds_evidence() -> None:
    observation = _observation(3_100)
    observation["gauge_max"] = 40_000
    observation["level"] = 35
    prediction = _shot()
    before = _Policy(observation, prediction)
    after = deepcopy(before)
    env = _ExactEnv()
    env.state["gauge"] = 3_100
    verdict = _planner(
        long_probe_ticks=4,
        low_gauge_threshold=20_000,
        low_gauge_exit_threshold=30_000,
        objective_mode="reserve-band",
        reserve_contingency_gauge=3_000,
        top_k_pairs=0,
    ).evaluate(env, observation, before, after, prediction)

    assert verdict.selected.category == "predicted-pair-weak"
    assert verdict.objective_mode == "reserve-band"
    assert verdict.reason == "predicted-pair-weak:reserve-band"
    assert (verdict.probe_mode, verdict.probe_ticks) == ("low-gauge-long", 4)
    assert verdict.objective_evidence["config"]["horizon_ticks"] == 4
    assert verdict.objective_evidence["config"]["contingency_gauge"] == 3_000
    assert verdict.manifest()["objective_evidence"]["winner"][
        "reserve_solvent"
    ] is True


def test_default_objective_mode_is_unchanged() -> None:
    config = FastMultiActionConfig()
    assert config.objective_mode == "wait-relative"
    assert (config.probe_ticks, config.long_probe_ticks) == (256, 512)
    assert config.long_probe_min_tick == 0
    assert config.robust_reserve_margin == 1_000
    assert (
        config.low_gauge_threshold,
        config.low_gauge_exit_threshold,
    ) == (20_000, 30_000)
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner().evaluate(
        _ExactEnv(), observation, before, deepcopy(before), prediction
    )
    assert verdict.objective_mode == "wait-relative"
    assert verdict.selected.category == "predicted-pair-strong"


def test_probe_horizon_hysteresis_is_shared_by_every_candidate_branch() -> None:
    planner = _planner(
        probe_ticks=2,
        long_probe_ticks=4,
        low_gauge_threshold=20_000,
        low_gauge_exit_threshold=30_000,
        top_k_pairs=0,
    )
    prediction = _shot()

    def evaluate(gauge: int):
        observation = _observation(gauge)
        env = _ExactEnv()
        env.state["gauge"] = gauge
        before = _Policy(observation, prediction)
        return planner.evaluate(
            env, observation, before, deepcopy(before), prediction
        )

    at_enter = evaluate(20_000)
    low = evaluate(19_999)
    inside_band = evaluate(25_000)
    at_exit = evaluate(30_000)
    recovered = evaluate(30_001)

    assert (at_enter.probe_mode, at_enter.probe_ticks) == ("normal", 2)
    for verdict in (low, inside_band, at_exit):
        assert (verdict.probe_mode, verdict.probe_ticks) == (
            "low-gauge-long",
            4,
        )
        assert {value.probe.survival_ticks for value in verdict.outcomes} == {4}
    assert (recovered.probe_mode, recovered.probe_ticks) == ("normal", 2)
    assert {value.probe.survival_ticks for value in recovered.outcomes} == {2}
    assert inside_band.manifest()["probe_ticks"] == 4


def test_equal_probe_horizons_preserve_stateless_legacy_behavior() -> None:
    planner = _planner(
        probe_ticks=2,
        long_probe_ticks=2,
        low_gauge_threshold=40_000,
        low_gauge_exit_threshold=30_000,
        top_k_pairs=0,
    )
    observation = _observation(1)
    env = _ExactEnv()
    env.state["gauge"] = 1
    prediction = _shot()
    before = _Policy(observation, prediction)

    verdict = planner.evaluate(
        env, observation, before, deepcopy(before), prediction
    )

    assert (verdict.probe_mode, verdict.probe_ticks) == ("normal", 2)
    assert {value.probe.survival_ticks for value in verdict.outcomes} == {2}


def test_long_probe_tick_floor_delays_low_gauge_hysteresis() -> None:
    planner = _planner(
        probe_ticks=2,
        long_probe_ticks=4,
        low_gauge_threshold=20_000,
        low_gauge_exit_threshold=30_000,
        long_probe_min_tick=40_000,
        top_k_pairs=0,
    )
    prediction = _shot()

    def evaluate(tick: int):
        observation = _observation(1)
        observation["tick"] = tick
        env = _ExactEnv()
        env.state["tick"] = tick
        env.state["gauge"] = 1
        before = _Policy(observation, prediction)
        return planner.evaluate(
            env, observation, before, deepcopy(before), prediction
        )

    before_floor = evaluate(39_999)
    at_floor = evaluate(40_000)
    assert (before_floor.probe_mode, before_floor.probe_ticks) == ("normal", 2)
    assert (at_floor.probe_mode, at_floor.probe_ticks) == ("low-gauge-long", 4)


def test_adaptive_probe_config_rejects_malformed_horizons() -> None:
    with pytest.raises(ValueError, match="at least probe_ticks"):
        FastMultiActionConfig(probe_ticks=4, long_probe_ticks=2)
    with pytest.raises(ValueError, match="must exceed"):
        FastMultiActionConfig(
            probe_ticks=2,
            long_probe_ticks=4,
            low_gauge_threshold=20_000,
            low_gauge_exit_threshold=20_000,
        )
    with pytest.raises(ValueError, match="long_probe_min_tick"):
        FastMultiActionConfig(long_probe_min_tick=-1)
    with pytest.raises(ValueError, match="objective_mode"):
        FastMultiActionConfig(objective_mode="unknown")


def test_robust_reserve_tie_is_a_manifest_bound_objective() -> None:
    config = FastMultiActionConfig(objective_mode="robust-reserve-tie")
    assert config.manifest()["objective_mode"] == "robust-reserve-tie"
    assert config.manifest()["robust_reserve_margin"] == 1_000
    assert config.manifest()["robust_score_trade"] == 500


def test_robust_reserve_tie_can_trade_bounded_score_for_reserve() -> None:
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner(
        objective_mode="robust-reserve-bounded",
        robust_reserve_margin=100,
        robust_score_trade=15,
        rescue_score_margin=15,
        top_k_pairs=0,
    ).evaluate(
        _ExactEnv(), observation, before, deepcopy(before), prediction
    )

    assert verdict.selected.category == "predicted-pair-weak"
    assert verdict.reason.endswith("robust-reserve-bounded-trade")
    assert verdict.objective_evidence["score_trade"] == 10
    assert verdict.objective_evidence["minimum_gauge_gain"] == 190
    assert verdict.objective_evidence["final_gauge_gain"] == 190


def test_chain_first_prefers_a_higher_chain_over_probe_score() -> None:
    observation = _observation()
    prediction = _shot()
    before = _Policy(observation, prediction)
    verdict = _planner(
        objective_mode="chain-first",
        rescue_score_margin=5,
        top_k_pairs=0,
    ).evaluate(
        _ExactEnv(), observation, before, deepcopy(before), prediction
    )

    assert verdict.selected.category == "predicted-pair-weak"
    assert verdict.objective_mode == "chain-first"
    weak = next(
        value for value in verdict.outcomes
        if value.candidate.category == "predicted-pair-weak"
    )
    assert weak.probe.highest_chain == 6


@pytest.mark.skipif(not RUNTIME.is_file(), reason="portable runtime artifact unavailable")
def test_portable_transactional_smoke() -> None:
    spec = ActionSpec()
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": 16},
    ) as env:
        observation, _info = env.reset(seed=3939967453)
        body_ids = [int(body["id"]) for body in observation["bodies"][:2]]
        if len(body_ids) < 2:
            pytest.skip("portable reset has fewer than two bodies")
        prediction = SteeringDecision(
            SemanticAction.strong(0.5, 0.5),
            SteeringIntent.STEER_MATCH,
            source_body_id=body_ids[0],
            destination_body_id=body_ids[1],
            destination_chain_id=0,
        )
        before = _Policy(observation, prediction)
        after = deepcopy(before)
        source_hash = env.state_hash()
        planner = FastMultiActionPlanner(
            lambda decision: decision.primitive_actions(spec),
            config=FastMultiActionConfig(
                probe_ticks=2,
                long_probe_ticks=2,
                wait_ticks=1,
                low_gauge_threshold=0,
                top_k_pairs=0,
                maximum_gauge_debt=100,
                rescue_score_margin=5,
                gauge_advantage=1,
            ),
            action_spec=spec,
        )
        verdict = planner.evaluate(env, observation, before, after, prediction)
        assert len(verdict.outcomes) == 3
        assert not verdict.used_fast_checkpoint
        assert env.state_hash() == source_hash
        reserve_planner = FastMultiActionPlanner(
            lambda decision: decision.primitive_actions(spec),
            config=FastMultiActionConfig(
                probe_ticks=2,
                long_probe_ticks=2,
                wait_ticks=1,
                low_gauge_threshold=0,
                top_k_pairs=0,
                maximum_gauge_debt=100,
                rescue_score_margin=5,
                gauge_advantage=1,
                objective_mode="reserve-band",
            ),
            action_spec=spec,
        )
        reserve = reserve_planner.evaluate(
            env, observation, before, after, prediction
        )
        assert reserve.objective_mode == "reserve-band"
        assert not reserve.used_fast_checkpoint
        assert env.state_hash() == source_hash
