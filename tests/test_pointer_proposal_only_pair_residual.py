from __future__ import annotations

from types import SimpleNamespace

import pytest

from irisu_pointer.fast_multiaction_planner import FastMultiActionConfig
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.proposal_only_pair_residual import (
    ProposalOnlyPairResidualPlanner,
    ProposalOnlyResidualConfig,
)
from irisu_rl.actions import SemanticAction


def test_trigger_is_late_or_late_low_reserve() -> None:
    config = ProposalOnlyResidualConfig(
        minimum_tick=40_000,
        low_reserve_minimum_tick=20_000,
        low_reserve_gauge=12_000,
    )
    assert not config.active({"tick": 19_999, "gauge": 1})
    assert not config.active({"tick": 30_000, "gauge": 12_001})
    assert config.active({"tick": 20_000, "gauge": 12_000})
    assert config.active({"tick": 40_000, "gauge": 40_000})


def test_default_trigger_has_no_tick_floor_for_low_reserve() -> None:
    config = ProposalOnlyResidualConfig()
    assert config.active({"tick": 3_000, "gauge": 12_000})
    assert not config.active({"tick": 3_000, "gauge": 12_001})


def test_planner_rejects_base_top_k_candidates() -> None:
    with pytest.raises(ValueError, match="top_k_pairs=0"):
        ProposalOnlyPairResidualPlanner(
            lambda value: value.primitive_actions(),
            residual_model=SimpleNamespace(eval=lambda: None),
            config=FastMultiActionConfig(top_k_pairs=1),
        )


def test_inactive_trigger_returns_immutable_base_candidate_set(monkeypatch) -> None:
    residual = SimpleNamespace(eval=lambda: residual)
    planner = ProposalOnlyPairResidualPlanner(
        lambda value: value.primitive_actions(),
        residual_model=residual,
        residual_config=ProposalOnlyResidualConfig(
            minimum_tick=10, low_reserve_minimum_tick=10
        ),
        config=FastMultiActionConfig(top_k_pairs=0),
    )
    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        "irisu_pointer.fast_multiaction_planner.FastMultiActionPlanner.candidates",
        lambda *args, **kwargs: sentinel,
    )
    result = planner.candidates(
        {"tick": 9, "gauge": 1}, object(), object(), object()
    )
    assert result is sentinel


def test_active_trigger_appends_exactly_one_separate_proposal(monkeypatch) -> None:
    residual = SimpleNamespace(eval=lambda: residual)
    planner = ProposalOnlyPairResidualPlanner(
        lambda value: value.primitive_actions(),
        residual_model=residual,
        residual_config=ProposalOnlyResidualConfig(minimum_tick=10),
        config=FastMultiActionConfig(top_k_pairs=0),
    )
    base = (object(), object(), object())
    proposal = SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
    )
    continuation = object()
    monkeypatch.setattr(
        "irisu_pointer.fast_multiaction_planner.FastMultiActionPlanner.candidates",
        lambda *args, **kwargs: base,
    )
    monkeypatch.setattr(
        "irisu_pointer.proposal_only_pair_residual.residual_pair_proposal",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        "irisu_pointer.proposal_only_pair_residual._copy_policy",
        lambda policy: continuation,
    )
    monkeypatch.setattr(
        "irisu_pointer.proposal_only_pair_residual._rebind_policy",
        lambda *args: True,
    )
    result = planner.candidates(
        {"tick": 10, "gauge": 1}, object(), object(), proposal
    )
    assert result[:3] == base
    assert len(result) == len(base) + 1
    assert result[-1].decision is proposal
    assert result[-1].continuation_policy is continuation
