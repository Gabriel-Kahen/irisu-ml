from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from irisu_pointer.steering_learning import GoalConditionedSteeringModel
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.encoding import EncodedBatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import rl_exact_onpolicy_pair_dagger as campaign  # noqa: E402


def decision(source: int | None, destination: int | None) -> SteeringDecision:
    if source is None:
        return SteeringDecision(SemanticAction.wait(16), SteeringIntent.WAIT)
    return SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=source,
        destination_body_id=destination,
    )


def outcome(
    ordinal: int, category: str, source: int | None, destination: int | None,
    *, survival: int = 512, clears: int = 1, score: int = 10, gauge: int = 5000,
) -> CandidateOutcome:
    return CandidateOutcome(
        PlannerCandidate(ordinal, category, decision(source, destination), object()),
        ProbeOutcome(survival, score, clears, gauge, gauge, False, False),
        0,
    )


def test_seed_plan_uses_canonical_train_allocator() -> None:
    value = campaign.load_seed_plan(
        ROOT / "configs/rl/experiments/exact-onpolicy-pair-dagger-train-v1.json"
    )
    assert value["seeds"] == [94547060, 489632993]
    assert all(0 <= seed < 1 << 30 for seed in value["seeds"])


def test_only_pair_head_is_trainable() -> None:
    model = GoalConditionedSteeringModel(TeacherStateEncoder().schema)
    selected = campaign.configure_trainable(model)
    assert set(selected) == {"pair_head.weight", "pair_head.bias"}
    assert all(
        parameter.requires_grad == name.startswith("pair_head.")
        for name, parameter in model.named_parameters()
    )


def test_safe_alternate_must_beat_incumbent_without_survival_trade() -> None:
    planner = FastMultiActionPlanner(
        lambda value: value.primitive_actions(),
        config=FastMultiActionConfig(probe_ticks=512, long_probe_ticks=512),
    )
    wait = outcome(2, "wait", None, None, clears=0, score=0)
    incumbent = outcome(0, "predicted-pair-strong", 1, 2, score=10)
    alternate = outcome(3, "top-1-pair-strong", 3, 4, score=20)
    verdict = MultiActionVerdict(
        alternate.candidate, "top-1-pair-strong:shot-score",
        (incumbent, wait, alternate), 4, True,
    )
    assert campaign.safe_alternate_preference(planner, verdict) == (
        alternate, incumbent
    )
    unsafe = outcome(
        3, "top-1-pair-strong", 3, 4, survival=511, score=1000
    )
    verdict = MultiActionVerdict(
        unsafe.candidate, "top-1-pair-strong:shot-score",
        (incumbent, wait, unsafe), 4, True,
    )
    assert campaign.safe_alternate_preference(planner, verdict) is None
    assert campaign.safe_alternate_preference(
        planner, verdict, branch_complete=False
    ) is None


def test_incomplete_or_nonfast_branch_never_emits_a_label() -> None:
    planner = FastMultiActionPlanner(
        lambda value: value.primitive_actions(),
        config=FastMultiActionConfig(probe_ticks=512, long_probe_ticks=512),
    )
    wait = outcome(2, "wait", None, None, clears=0, score=0)
    incumbent = outcome(0, "predicted-pair-strong", 1, 2, score=10)
    alternate = outcome(3, "top-1-pair-strong", 3, 4, score=20)
    nonfast = MultiActionVerdict(
        alternate.candidate, "top-1-pair-strong:shot-score",
        (incumbent, wait, alternate), 4, False,
    )
    assert campaign.safe_alternate_preference(planner, nonfast) is None
    fast = MultiActionVerdict(
        alternate.candidate, "top-1-pair-strong:shot-score",
        (incumbent, wait, alternate), 4, True,
    )
    assert campaign.safe_alternate_preference(
        planner, fast, parent_unchanged=False
    ) is None


def test_exact_worker_error_waits_without_label_or_policy_mutation() -> None:
    policy = object()
    restored, fallback, label = campaign.conservative_exact_error(
        parent_hash=123, current_hash=123, policy_before=policy
    )
    assert restored is policy
    assert fallback.action.kind.name == "WAIT"
    assert fallback.action.wait_ticks == 16
    assert label is None


def test_exact_worker_error_fails_if_parent_changed() -> None:
    with np.testing.assert_raises_regex(RuntimeError, "changed parent"):
        campaign.conservative_exact_error(
            parent_hash=123, current_hash=124, policy_before=object()
        )


def test_new_old_balance_is_deterministic_and_equal() -> None:
    first = campaign.balanced_new_old(range(5), range(10, 20), seed=7)
    second = campaign.balanced_new_old(range(5), range(10, 20), seed=7)
    assert first == second
    assert len(first) == 10
    assert all(value < 10 for value in first[0::2])
    assert all(value >= 10 for value in first[1::2])


def test_semantic_preference_hash_is_deterministic() -> None:
    schema = TeacherStateEncoder().schema
    encoded = EncodedBatch(
        np.zeros((1, len(schema.global_features)), dtype=np.float32),
        np.zeros((1, schema.capacity, len(schema.body_features)), dtype=np.float32),
        np.ones((1, schema.capacity), dtype=np.bool_),
        np.array([123], dtype=np.uint64),
        np.zeros((1,), dtype=np.uint32),
        schema,
    )
    provenance = {
        "schema": "irisu-exact-neutral-onpolicy-pair-preference-v1",
        "seed": 7,
        "tick": 123,
        "teacher_pair": [1, 2],
        "learner_pair": [2, 1],
    }
    first = campaign.ranking.PairPreference(encoded, 1, 2, 2, 1, provenance)
    second = campaign.ranking.PairPreference(encoded, 1, 2, 2, 1, provenance)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_persisted_preferences_restore_owned_contiguous_arrays(tmp_path: Path) -> None:
    schema = TeacherStateEncoder().schema
    encoded = EncodedBatch(
        np.zeros((1, len(schema.global_features)), dtype=np.float32),
        np.zeros((1, schema.capacity, len(schema.body_features)), dtype=np.float32),
        np.ones((1, schema.capacity), dtype=np.bool_),
        np.array([123], dtype=np.uint64),
        np.zeros((1,), dtype=np.uint32),
        schema,
    )
    preference = campaign.ranking.PairPreference(
        encoded, 1, 2, 2, 1,
        {"schema": "irisu-exact-neutral-onpolicy-pair-preference-v1"},
    )
    path = tmp_path / "collection.pt"
    torch.save(
        {
            "preferences": [{
                "manifest": preference.manifest(),
                "global_features": torch.from_numpy(encoded.global_features),
                "body_features": torch.from_numpy(encoded.body_features),
                "body_mask": torch.from_numpy(encoded.body_mask),
            }]
        },
        path,
    )
    restored = campaign.restore_preferences(path, schema)[0].observation
    for array in (restored.global_features, restored.body_features, restored.body_mask):
        assert array.flags.c_contiguous
        assert array.flags.owndata


def test_collection_content_hash_binds_tensors_and_manifest() -> None:
    base = {
        "schema": "irisu-exact-neutral-onpolicy-pair-collection-v1",
        "base_checkpoint_sha256": "a" * 64,
        "seed_plan": {"seeds": [1]},
        "planner_config": {"probe_ticks": 512},
        "episodes": [{"seed": 1}],
        "preferences": [{
            "manifest": {"positive": [1, 2], "negative": [2, 1]},
            "global_features": torch.zeros((1, 2)),
            "body_features": torch.zeros((1, 2, 3)),
            "body_mask": torch.ones((1, 2), dtype=torch.bool),
        }],
    }
    first = campaign.collection_content_sha256(base)
    second = campaign.collection_content_sha256(base)
    assert first == second
    changed = dict(base)
    changed["preferences"] = [dict(base["preferences"][0])]
    changed["preferences"][0]["global_features"] = torch.ones((1, 2))
    assert campaign.collection_content_sha256(changed) != first


def test_trainer_embeds_no_eval_seed_constants() -> None:
    source = (ROOT / "benchmarks/rl_exact_onpolicy_pair_dagger.py").read_text()
    assert "94547060" not in source
    assert "489632993" not in source
    assert "locked" + "_seeds" not in source.lower()
