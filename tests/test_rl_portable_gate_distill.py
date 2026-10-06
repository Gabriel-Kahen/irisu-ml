from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rl_portable_gate_distill as campaign  # noqa: E402
import rl_portable_checkpoint_eval as deployment  # noqa: E402


def outcome(*, score: int, gauge: int, ticks: int = 128):
    return campaign.ProbeOutcome(ticks, score, 1, gauge, gauge, False, False)


def test_train_seeds_are_explicit_unique_uint32() -> None:
    assert campaign.parse_seeds("1, 0x2, 4294967295") == (1, 2, 0xFFFF_FFFF)
    with pytest.raises(Exception):
        campaign.parse_seeds("1,1")
    with pytest.raises(Exception):
        campaign.parse_seeds("")
    with pytest.raises(Exception):
        campaign.parse_seeds("4294967296")


def test_cli_defaults_to_adaptive_gate_and_act_head(tmp_path: Path) -> None:
    args = campaign.parse_args(
        [
            "--base-checkpoint",
            str(tmp_path / "base.pt"),
            "--base-sha256",
            "a" * 64,
            "--train-seeds",
            "7,8",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert (args.short_horizon, args.long_horizon, args.gauge_threshold) == (
        128,
        256,
        30_000,
    )
    assert args.trainable_scope == "act-head"
    assert args.train_seeds == (7, 8)
    assert (args.maximum_gauge_debt, args.rescue_score_margin) == (1_000, 500)
    assert campaign.inference_config(args.act_logit_bias) == {
        "cooldown_ticks": 16,
        "minimum_pair_closure_sizes": 0.05,
        "impact_side_sizes": 0.5,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 50.0,
        "act_logit_bias": 1.0,
    }


def test_adaptive_horizon_uses_long_probe_at_threshold() -> None:
    config = {"short_horizon": 128, "long_horizon": 256, "gauge_threshold": 20_000}
    assert campaign.adaptive_horizon(20_001, **config) == 128
    assert campaign.adaptive_horizon(20_000, **config) == 256
    assert campaign.adaptive_horizon(0, **config) == 256


def test_action_balancing_is_exact_and_deterministic() -> None:
    shot = [SimpleNamespace(is_shot=True, sha256=value) for value in ("c", "a", "b")]
    wait = [SimpleNamespace(is_shot=False, sha256="d")]
    first = campaign.balanced_examples((*shot, *wait))
    second = campaign.balanced_examples((*reversed(shot), *wait))
    assert [value.sha256 for value in first] == [value.sha256 for value in second]
    assert sum(value.is_shot for value in first) == 3
    assert sum(not value.is_shot for value in first) == 3


def test_default_scope_freezes_everything_except_act_head() -> None:
    model = campaign.GoalConditionedSteeringModel(
        campaign.TeacherStateEncoder().schema
    )
    selected = campaign.configure_trainable_scope(model, "act-head")
    assert selected
    assert all(name.startswith("act_head.") for name in selected)
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(selected)


def test_gate_policy_copies_share_model_but_not_controller_state() -> None:
    model = campaign.GoalConditionedSteeringModel(
        campaign.TeacherStateEncoder().schema
    )
    policy = campaign._make_policy(model, "1" * 64, 1.0)
    cloned = copy.deepcopy(policy)
    assert cloned is not policy
    assert cloned.inner is not policy.inner
    assert cloned.model is policy.model is model


def test_warm_start_lineage_unions_declared_seed_fields() -> None:
    assert campaign.checkpoint_training_seeds(
        {"demonstration_seeds": [9, 3], "training_seeds": [7, 3]}
    ) == (3, 7, 9)
    with pytest.raises(ValueError, match="does not declare"):
        campaign.checkpoint_training_seeds({})
    with pytest.raises(ValueError, match="unique uint32"):
        campaign.checkpoint_training_seeds({"training_seeds": [1, 1]})


def test_run_fails_closed_before_output_for_lineage_free_checkpoint(
    tmp_path: Path,
) -> None:
    model = campaign.GoalConditionedSteeringModel(
        campaign.TeacherStateEncoder().schema
    )
    checkpoint = tmp_path / "lineage-free.pt"
    checkpoint_sha256 = campaign.save_steering_checkpoint(
        checkpoint, model, metadata={}
    )
    output = tmp_path / "output"
    args = campaign.parse_args(
        [
            "--base-checkpoint",
            str(checkpoint),
            "--base-sha256",
            checkpoint_sha256,
            "--train-seeds",
            "7",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="does not declare"):
        campaign.run(args)
    assert not output.exists()


def test_reserve_objective_exactly_matches_deployment_planner() -> None:
    cases = (
        (outcome(score=100, gauge=10_000), outcome(score=0, gauge=12_000)),
        (outcome(score=600, gauge=10_000), outcome(score=0, gauge=12_000)),
        (outcome(score=0, gauge=12_001), outcome(score=0, gauge=12_000)),
    )
    config = {"maximum_gauge_debt": 1_000, "rescue_score_margin": 500}
    for shot, wait in cases:
        assert campaign.reserve_choice(shot, wait, **config) == deployment.reserve_choice(
            shot, wait, **config
        )
    assert campaign.reserve_choice(*cases[0], **config) == (False, "wait-reserve")
