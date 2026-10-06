from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    SteeringModelConfig,
    steering_example_from_decision,
)
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rl_portable_pair_rank_distill as campaign  # noqa: E402


def body(identifier: int, x: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 0,
        "x": x,
        "y": 200.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def observation() -> dict[str, object]:
    return {
        "tick": 32,
        "score": 0,
        "gauge": 10_000,
        "gauge_max": 50_000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 100},
        "bodies": (body(1, 160.0), body(2, 260.0), body(3, 360.0)),
    }


def outcome(score: int, gauge: int) -> campaign.ProbeOutcome:
    return campaign.ProbeOutcome(128, score, 1, gauge, gauge, False, False)


def shot(source: int, destination: int) -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=source,
        destination_body_id=destination,
    )


def test_tournament_emits_only_observed_shot_preferences() -> None:
    candidates = (
        SteeringDecision(SemanticAction.wait(16), SteeringIntent.WAIT),
        shot(1, 2),
        SteeringDecision(
            SemanticAction.weak(0.5, 0.5),
            SteeringIntent.STEER_MATCH,
            source_body_id=1,
            destination_body_id=2,
        ),
        shot(2, 1),
    )
    winner, preferences = campaign.tournament_preferences(
        candidates,
        (
            outcome(0, 10_000),
            outcome(100, 9_500),
            outcome(200, 9_500),
            outcome(150, 9_500),
        ),
        maximum_gauge_debt=1_000,
        rescue_score_margin=500,
    )
    assert winner == 2
    assert [(positive, negative) for positive, negative, _ in preferences] == [(2, 3)]
    assert all(positive > 0 and negative > 0 for positive, negative, _ in preferences)


def test_default_scope_and_teacher_budget_are_scalable(tmp_path: Path) -> None:
    args = campaign.parse_args(
        [
            "--base-checkpoint",
            str(tmp_path / "base.pt"),
            "--base-sha256",
            "a" * 64,
            "--train-seeds",
            "7,8",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.maximum_pairs == 4
    assert args.trainable_scope == "pair-head"
    model = GoalConditionedSteeringModel(TeacherStateEncoder().schema)
    selected = campaign.configure_trainable_scope(model, args.trainable_scope)
    assert selected
    assert all(name.startswith("pair_head.") for name in selected)
    assert {name for name, value in model.named_parameters() if value.requires_grad} == set(selected)


def test_explicit_pair_ranking_loss_improves_without_false_negative_ce() -> None:
    state = observation()
    provenance = {"schema": "test-pair-preference", "seed": 7, "tick": 32}
    positive, negative = shot(1, 2), shot(2, 1)
    preference = campaign._preference(
        state,
        positive,
        negative,
        provenance,
        encoder=TeacherStateEncoder(),
        pointer_spec=campaign.gate._make_policy(
            GoalConditionedSteeringModel(TeacherStateEncoder().schema), "1" * 64, 1.0
        ).pointer_spec,
    )
    identity = hashlib.sha256(b"act-label").hexdigest()
    shot_example = steering_example_from_decision(
        state,
        positive,
        episode_identity="shot",
        provenance_sha256=identity,
        require_representable_template=False,
    )
    wait_example = steering_example_from_decision(
        state,
        SteeringDecision(SemanticAction.wait(16), SteeringIntent.WAIT),
        episode_identity="wait",
        provenance_sha256=identity,
        require_representable_template=False,
    )
    assert shot_example is not None and wait_example is not None
    model = GoalConditionedSteeringModel(
        TeacherStateEncoder().schema,
        config=SteeringModelConfig(body_hidden=16, global_hidden=8, pair_hidden=24),
    )
    before = copy.deepcopy(model.state_dict())
    campaign.configure_trainable_scope(model, "pair-head")
    report = campaign.train_ranked_heads(
        model,
        (shot_example, wait_example),
        (),
        (preference,),
        steps=40,
        batch_size=2,
        learning_rate=1e-2,
        causal_weight=1.0,
        ranking_weight=1.0,
        seed=17,
    )
    assert report.final_pair_loss < report.initial_pair_loss
    assert report.pair_accuracy == 1.0
    assert any(
        not campaign.torch.equal(before[name], value)
        for name, value in model.state_dict().items()
        if name.startswith("pair_head.")
    )
    assert all(
        campaign.torch.equal(before[name], value)
        for name, value in model.state_dict().items()
        if not name.startswith("pair_head.")
    )


def test_causal_confidence_is_bounded_and_increases_with_survival_margin() -> None:
    tied = campaign.causal_confidence(
        outcome(0, 10_000),
        outcome(0, 10_000),
        horizon=128,
        maximum_gauge_debt=1_000,
    )
    terminal = campaign.ProbeOutcome(32, 0, 0, 1_000, 1_000, True, False)
    survival = campaign.causal_confidence(
        outcome(1_000, 10_000),
        terminal,
        horizon=128,
        maximum_gauge_debt=1_000,
    )
    assert tied == 1.0
    assert tied < survival <= 8.0


def test_pair_only_training_rejects_an_empty_preference_set() -> None:
    state = observation()
    identity = hashlib.sha256(b"empty-pair-label").hexdigest()
    example = steering_example_from_decision(
        state,
        shot(1, 2),
        episode_identity="shot",
        provenance_sha256=identity,
        require_representable_template=False,
    )
    assert example is not None
    model = GoalConditionedSteeringModel(TeacherStateEncoder().schema)
    campaign.configure_trainable_scope(model, "pair-head")
    with pytest.raises(ValueError, match="cross-pair preference"):
        campaign.train_ranked_heads(
            model,
            (example,),
            (),
            (),
            steps=1,
            batch_size=1,
            learning_rate=1e-3,
            causal_weight=1.0,
            ranking_weight=1.0,
            seed=1,
        )


def test_pair_preference_manifest_is_content_bound() -> None:
    state = observation()
    provenance = {"schema": "test-pair-preference", "seed": 7, "tick": 32}
    policy = campaign.gate._make_policy(
        GoalConditionedSteeringModel(TeacherStateEncoder().schema), "1" * 64, 1.0
    )
    first = campaign._preference(
        state,
        shot(1, 2),
        shot(2, 1),
        provenance,
        encoder=TeacherStateEncoder(),
        pointer_spec=policy.pointer_spec,
    )
    second = campaign._preference(
        state,
        shot(1, 2),
        shot(2, 1),
        provenance,
        encoder=TeacherStateEncoder(),
        pointer_spec=policy.pointer_spec,
    )
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_trainer_embeds_no_locked_seed_constants() -> None:
    source = (ROOT / "benchmarks/rl_portable_pair_rank_distill.py").read_text()
    assert "CALIBRATION" + "_SEEDS" not in source
    assert "locked" + "_seeds" not in source.lower()
