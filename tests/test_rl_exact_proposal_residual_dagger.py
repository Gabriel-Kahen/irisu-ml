from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from irisu_pointer.fast_multiaction_planner import (
    CandidateOutcome,
    FastMultiActionConfig,
    FastMultiActionPlanner,
    MultiActionVerdict,
    PlannerCandidate,
)
from irisu_pointer.shot_necessity import ProbeOutcome
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import GoalConditionedSteeringModel, SteeringModelConfig
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rl_exact_proposal_residual_dagger as campaign  # noqa: E402


def decision(source: int | None, destination: int | None) -> SteeringDecision:
    if source is None:
        return SteeringDecision(SemanticAction.wait(16), SteeringIntent.WAIT)
    return SteeringDecision(
        SemanticAction.strong(0.5, 0.5), SteeringIntent.STEER_MATCH,
        source_body_id=source, destination_body_id=destination,
    )


def outcome(
    ordinal: int, category: str, pair: tuple[int, int] | None, *,
    survival: int, minimum: int, final: int | None = None,
    score: int = 10, clears: int = 1,
) -> CandidateOutcome:
    value = decision(None, None) if pair is None else decision(*pair)
    return CandidateOutcome(
        PlannerCandidate(ordinal, category, value, object()),
        ProbeOutcome(
            survival, score, clears, minimum,
            minimum if final is None else final, survival < 2048, False,
        ),
        0,
    )


def test_strict_label_rejects_capped_survival_ties() -> None:
    planner = FastMultiActionPlanner(lambda value: value.primitive_actions())
    incumbent = outcome(0, "predicted-pair-strong", (1, 2), survival=2048, minimum=100)
    wait = outcome(1, "wait", None, survival=2048, minimum=100)
    teacher = outcome(2, "top-1-pair-strong", (3, 4), survival=2048, minimum=100)
    verdict = MultiActionVerdict(teacher.candidate, "top", (incumbent, wait, teacher), 4, True, probe_ticks=2048)
    assert campaign.strict_full_horizon_preference(planner, verdict) is None


def test_capped_tie_requires_long_horizon_robust_reserve_without_reward_regression() -> None:
    planner = FastMultiActionPlanner(
        lambda value: value.primitive_actions(),
        config=FastMultiActionConfig(probe_ticks=2048, long_probe_ticks=4096),
    )
    incumbent = outcome(0, "predicted-pair-strong", (1, 2), survival=4096, minimum=100)
    wait = outcome(1, "wait", None, survival=4096, minimum=90)
    teacher = outcome(2, "top-1-pair-strong", (3, 4), survival=4096, minimum=1100)
    verdict = MultiActionVerdict(
        teacher.candidate, "top", (incumbent, wait, teacher), 4, True,
        probe_mode="low-gauge-long", probe_ticks=4096,
    )
    assert campaign.strict_full_horizon_preference(planner, verdict) == (teacher, incumbent)
    short = MultiActionVerdict(
        teacher.candidate, "top", (incumbent, wait, teacher), 4, True,
        probe_mode="normal", probe_ticks=4096,
    )
    assert campaign.strict_full_horizon_preference(planner, short) is None
    weak_reserve = outcome(2, "top-1-pair-strong", (3, 4), survival=4096, minimum=1099)
    verdict = MultiActionVerdict(
        weak_reserve.candidate, "top", (incumbent, wait, weak_reserve), 4, True,
        probe_mode="low-gauge-long", probe_ticks=4096,
    )
    assert campaign.strict_full_horizon_preference(planner, verdict) is None
    weak_final = outcome(
        2, "top-1-pair-strong", (3, 4), survival=4096, minimum=1100,
        final=1099,
    )
    verdict = MultiActionVerdict(
        weak_final.candidate, "top", (incumbent, wait, weak_final), 4, True,
        probe_mode="low-gauge-long", probe_ticks=4096,
    )
    assert campaign.strict_full_horizon_preference(planner, verdict) is None
    reward_regression = outcome(
        2, "top-1-pair-strong", (3, 4), survival=4096, minimum=1100,
        score=9,
    )
    verdict = MultiActionVerdict(
        reward_regression.candidate, "top", (incumbent, wait, reward_regression), 4, True,
        probe_mode="low-gauge-long", probe_ticks=4096,
    )
    assert campaign.strict_full_horizon_preference(planner, verdict) is None


def test_strict_label_requires_full_survival_and_minimum_gauge_nondomination() -> None:
    planner = FastMultiActionPlanner(lambda value: value.primitive_actions())
    incumbent = outcome(0, "predicted-pair-strong", (1, 2), survival=1900, minimum=100)
    wait = outcome(1, "wait", None, survival=1800, minimum=90)
    teacher = outcome(2, "top-1-pair-strong", (3, 4), survival=2048, minimum=100)
    verdict = MultiActionVerdict(teacher.candidate, "top", (incumbent, wait, teacher), 4, True, probe_ticks=2048)
    assert campaign.strict_full_horizon_preference(planner, verdict) == (teacher, incumbent)
    dominated = outcome(2, "top-1-pair-strong", (3, 4), survival=2048, minimum=99)
    verdict = MultiActionVerdict(dominated.candidate, "top", (incumbent, wait, dominated), 4, True, probe_ticks=2048)
    assert campaign.strict_full_horizon_preference(planner, verdict) is None


def test_teacher_alternate_never_changes_immutable_base_rollout() -> None:
    planner = FastMultiActionPlanner(lambda value: value.primitive_actions())
    incumbent = outcome(0, "predicted-pair-strong", (1, 2), survival=1900, minimum=100)
    wait = outcome(1, "wait", None, survival=1800, minimum=90)
    teacher = outcome(2, "top-1-pair-strong", (3, 4), survival=2048, minimum=100)
    verdict = MultiActionVerdict(
        teacher.candidate, "top-1-pair-strong:shot-survival",
        (incumbent, wait, teacher), 4, True, probe_ticks=2048,
    )
    assert campaign.strict_full_horizon_preference(planner, verdict) == (teacher, incumbent)
    assert campaign.immutable_base_choice(planner, verdict) is incumbent


def test_fresh_seed_plans_are_canonical_train_split_and_disjoint() -> None:
    collect = campaign.prior.load_seed_plan(ROOT / "configs/rl/experiments/exact-proposal-residual-train-v1.json")
    select = campaign.prior.load_seed_plan(ROOT / "configs/rl/experiments/exact-proposal-residual-select-v1.json")
    assert set(collect["seeds"]).isdisjoint(select["seeds"])
    assert all(0 <= int(seed) < 1 << 30 for seed in (*collect["seeds"], *select["seeds"]))


def test_default_horizons_and_budget_are_bounded() -> None:
    parser = campaign.parser()
    assert parser.get_default("probe_ticks") == 2048
    assert parser.get_default("low_gauge_probe_ticks") == 4096
    assert parser.get_default("maximum_queries") == 128
    assert parser.get_default("maximum_positives") == 32
    assert parser.get_default("low_reserve_minimum_tick") == 0


def body(identifier: int, x: float) -> dict[str, object]:
    return {
        "id": identifier, "kind": "piece", "shape": "circle",
        "lifecycle": "dynamic_fresh", "color": 0, "x": x, "y": 200.0,
        "vx": 0.0, "vy": 0.0, "angle": 0.0, "angular_velocity": 0.0,
        "size": 40.0, "chain_id": 0, "projectile_hits": 0,
        "age_ticks": 20, "remaining_lifetime": 1000, "rot_timer": 0,
    }


def observation() -> dict[str, object]:
    return {
        "tick": 50_000, "score": 0, "gauge": 10_000, "gauge_max": 50_000,
        "level": 1, "highest_chain": 0, "qualifying_clear_count": 0,
        "left_held": False, "right_held": False, "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 100},
        "bodies": (body(1, 160.0), body(2, 260.0), body(3, 360.0)),
    }


def test_residual_fit_is_deterministic_and_pair_head_only() -> None:
    torch.manual_seed(9)
    encoder = TeacherStateEncoder()
    config = SteeringModelConfig(body_hidden=16, global_hidden=8, pair_hidden=24)
    base = GoalConditionedSteeringModel(encoder.schema, config=config)
    first = copy.deepcopy(base)
    second = copy.deepcopy(base)
    preference = campaign.ranking._preference(
        observation(), decision(1, 2), decision(2, 1),
        {"schema": "test-strict-survival"}, encoder=encoder,
        pointer_spec=base.pointer_spec,
    )
    anchor = campaign.encoded_copy(
        campaign.gate._make_policy(base, "1" * 64, 1.0), observation()
    )
    base_sha = campaign.gate._model_state_sha(base)
    first_report = campaign.train_residual(
        first, base, (preference,), (anchor,), steps=4, learning_rate=1e-3,
        functional_weight=10.0, seed=17,
    )
    second_report = campaign.train_residual(
        second, base, (preference,), (anchor,), steps=4, learning_rate=1e-3,
        functional_weight=10.0, seed=17,
    )
    assert campaign.gate._model_state_sha(first) == campaign.gate._model_state_sha(second)
    assert campaign.gate._model_state_sha(base) == base_sha
    assert set(first_report["changed_parameters"]) == {"pair_head.weight", "pair_head.bias"}
    assert first_report == second_report


def test_collection_hash_binds_positive_and_anchor_tensors() -> None:
    encoded = TeacherStateEncoder().encode([observation()])
    record = campaign.encoded_record(encoded)
    payload = {
        "schema": campaign.SCHEMA, "base_checkpoint_sha256": "a" * 64,
        "base_model_sha256": "b" * 64, "seed_plan": {"seeds": [1]},
        "selection_seed_plan": {"seeds": [2]}, "planner_config": {},
        "proposal_trigger": {}, "episodes": [],
        "positives": [{"manifest": {"positive": [0, 1]}, "observation": record}],
        "anchors": [record],
    }
    first = campaign.collection_content_sha256(payload)
    changed = copy.deepcopy(payload)
    changed["anchors"][0]["hashes"]["global_features"]["sha256"] = "0" * 64
    assert campaign.collection_content_sha256(changed) != first


def test_trainer_embeds_no_fresh_or_eval_seed_constants() -> None:
    source = (ROOT / "benchmarks/rl_exact_proposal_residual_dagger.py").read_text()
    for seed in (484286044, 344580395, 155356647, 716105428):
        assert str(seed) not in source
    assert "locked" + "_seeds" not in source.lower()
