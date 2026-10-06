from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringModelConfig,
    steering_example_from_decision,
)
from irisu_rl.actions import SemanticAction, SemanticActionKind
from irisu_rl.encoding import TeacherStateEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rl_portable_strength_distill as campaign  # noqa: E402
import rl_portable_checkpoint_eval as evaluation  # noqa: E402


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


def observation(*, gauge: int = 10_000) -> dict[str, object]:
    return {
        "tick": 32,
        "score": 0,
        "gauge": gauge,
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
        "bodies": (body(1, 160.0), body(2, 260.0)),
    }


def shot(*, strong: bool) -> SteeringDecision:
    constructor = SemanticAction.strong if strong else SemanticAction.weak
    return SteeringDecision(
        constructor(0.25, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
    )


def test_policy_kind_head_is_opt_in_and_legacy_default_stays_strong() -> None:
    model = GoalConditionedSteeringModel(TeacherStateEncoder().schema)
    with torch.no_grad():
        model.act_head.weight.zero_()
        model.act_head.bias.copy_(torch.tensor([-10.0, 10.0]))
        model.kind_head.weight.zero_()
        model.kind_head.bias.copy_(torch.tensor([10.0, -10.0]))
    legacy = GoalConditionedSteeringPolicy(model)
    learned = GoalConditionedSteeringPolicy(model, use_kind_head=True)
    legacy.reset(1)
    learned.reset(1)
    assert SemanticActionKind(legacy.predict(observation()).action.kind) is SemanticActionKind.FIRE_STRONG
    assert SemanticActionKind(learned.predict(observation()).action.kind) is SemanticActionKind.FIRE_WEAK


def test_strength_counterfactual_preserves_pair_and_geometry() -> None:
    proposal = shot(strong=True)
    weak = campaign.with_strength(proposal, strong=False)
    assert not (SemanticActionKind(weak.action.kind) is SemanticActionKind.FIRE_STRONG)
    assert weak.source_body_id == proposal.source_body_id
    assert weak.destination_body_id == proposal.destination_body_id
    assert (weak.action.x_norm, weak.action.y_norm) == (
        proposal.action.x_norm,
        proposal.action.y_norm,
    )


def test_strength_choice_anchors_exact_ties_to_incumbent() -> None:
    tied = campaign.gate.ProbeOutcome(128, 100, 1, 10_000, 9_000, False, False)
    assert campaign.choose_strength(
        tied,
        tied,
        incumbent_is_strong=False,
        maximum_gauge_debt=1_000,
        rescue_score_margin=500,
    ) == (False, "incumbent-strength-tie-anchor")


def test_default_training_scope_is_only_kind_head(tmp_path: Path) -> None:
    args = campaign.parse_args(
        [
            "--base-checkpoint",
            str(tmp_path / "base.pt"),
            "--base-sha256",
            "a" * 64,
            "--train-seeds",
            "7",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.trainable_scope == "kind-head"
    assert campaign.inference_config(args.act_logit_bias)["use_kind_head"] is True
    model = GoalConditionedSteeringModel(TeacherStateEncoder().schema)
    selected = campaign.configure_trainable_scope(model, args.trainable_scope)
    assert selected
    assert all(name.startswith("kind_head.") for name in selected)
    assert {name for name, value in model.named_parameters() if value.requires_grad} == set(selected)


def test_development_evaluator_requires_explicit_kind_head_opt_in(
    tmp_path: Path,
) -> None:
    common = [
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--seeds",
        "7",
    ]
    assert evaluation.parser().parse_args(common).use_kind_head is False
    assert evaluation.parser().parse_args([*common, "--use-kind-head"]).use_kind_head is True


def test_weighted_strength_training_fits_weak_label() -> None:
    provenance = {"schema": "strength-test", "seed": 7, "tick": 32}
    example = steering_example_from_decision(
        observation(),
        shot(strong=False),
        episode_identity="weak",
        provenance_sha256=hashlib.sha256(b"strength-test").hexdigest(),
        require_representable_template=False,
    )
    assert example is not None
    label = campaign.StrengthLabel(example, 2.0, "weak", provenance)
    model = GoalConditionedSteeringModel(
        TeacherStateEncoder().schema,
        config=SteeringModelConfig(body_hidden=16, global_hidden=8, pair_hidden=24),
    )
    campaign.configure_trainable_scope(model, "kind-head")
    report = campaign.train_strength_head(
        model,
        (label,),
        steps=25,
        batch_size=1,
        learning_rate=1e-2,
        seed=19,
    )
    assert report.final_loss < report.initial_loss
    assert report.weak_recall == 1.0


def test_strength_trainer_has_no_seed_suite_constants() -> None:
    source = (ROOT / "benchmarks/rl_portable_strength_distill.py").read_text()
    assert "CALIBRATION" + "_SEEDS" not in source
    assert "locked" + "_seeds" not in source.lower()
