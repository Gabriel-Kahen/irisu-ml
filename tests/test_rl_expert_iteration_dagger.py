from __future__ import annotations

import copy
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from irisu_env import ActionKind
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rl_expert_iteration_dagger as campaign  # noqa: E402


class _WaitPolicy:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation: dict[str, object]) -> SteeringDecision:
        return SteeringDecision(
            SemanticAction.wait(1), SteeringIntent.WAIT, reason="test continuation"
        )


class _StatefulPolicy:
    def __init__(self, calls: int = 0, armed: bool = False) -> None:
        self.calls = calls
        self.armed = armed

    def reset(self, seed: int) -> None:
        self.calls = 0
        self.armed = False

    def predict(self, observation: dict[str, object]) -> SteeringDecision:
        self.calls += 1
        if self.armed:
            self.armed = False
            return SteeringDecision(
                SemanticAction.strong(0.5, 0.5),
                SteeringIntent.STEER_MATCH,
                source_body_id=1,
                destination_body_id=2,
            )
        return SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT)


class _BranchEnv:
    physics_backend = "portable"

    def __init__(self, *, shot_score: int = 10, shot_gauge: int = 0) -> None:
        self.tick = 10
        self.score = 7
        self.gauge = 100
        self.terminated = False
        self.shot_score = shot_score
        self.shot_gauge = shot_gauge

    def observation(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "score": self.score,
            "gauge": self.gauge,
            "terminated": self.terminated,
            "truncated": False,
            "bodies": (),
        }

    def clone_state(self) -> bytes:
        return struct.pack("<iii?", self.tick, self.score, self.gauge, self.terminated)

    def restore_state(self, snapshot: bytes) -> dict[str, object]:
        self.tick, self.score, self.gauge, self.terminated = struct.unpack("<iii?", snapshot)
        return self.observation()

    def state_hash(self) -> int:
        return hash((self.tick, self.score, self.gauge, self.terminated))

    def step(self, action: object):
        kind = ActionKind.parse(getattr(action, "kind"))
        self.tick += int(getattr(action, "wait_ticks", 1)) if kind is ActionKind.WAIT else 1
        if kind is ActionKind.STRONG_SHOT:
            self.score += self.shot_score
            self.gauge += self.shot_gauge
        return self.observation(), 0.0, self.terminated, False, {}


def test_seed_derivation_is_stable_unique_and_namespaced() -> None:
    first = campaign.derive_seeds("train-a", 64)
    assert first == campaign.derive_seeds("train-a", 64)
    assert len(first) == len(set(first)) == 64
    assert set(first).isdisjoint(campaign.derive_seeds("held-out", 64))


def test_sparse_dagger_defaults_to_training_only_the_act_head(tmp_path: Path) -> None:
    args = campaign.parse_args(["--output", str(tmp_path / "run")])
    assert args.trainable_scope == "act-head"
    model = campaign.GoalConditionedSteeringModel(
        campaign.TeacherStateEncoder().schema
    )
    selected = campaign.configure_trainable_scope(model, args.trainable_scope)
    assert selected
    assert all(name.startswith("act_head.") for name in selected)
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(selected)


def test_branch_objective_prioritizes_survival_then_score() -> None:
    terminal_rich = campaign.BranchOutcome(0, 9, 1_000, 100, True, False)
    surviving = campaign.BranchOutcome(1, 10, 0, 1, False, False)
    richer = campaign.BranchOutcome(2, 10, 20, 1, False, False)
    assert surviving.objective > terminal_rich.objective
    assert richer.objective > surviving.objective
    gauge_only = campaign.BranchOutcome(3, 10, 0, 99, False, False)
    assert gauge_only.objective > surviving.objective
    assert gauge_only.improvement_objective == surviving.improvement_objective


def test_snapshot_search_labels_strict_score_improvement_and_restores_source() -> None:
    env = _BranchEnv()
    source = copy.deepcopy(env.observation())
    reference = SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT)
    wait = SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT)
    shot = SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
    )
    result = campaign.search_improvement(
        env,
        source,
        reference,
        (wait, shot),
        continuation_policy=_WaitPolicy(),
        seed=123,
        horizon_ticks=4,
    )
    assert result.strict_improvement
    assert result.winner == 1
    assert result.label_reason == "score"
    assert result.outcomes[1].score_delta == 10
    assert env.observation() == source
    assert len(result.sha256) == 64


def test_branch_continuation_copies_live_internal_state_without_reset() -> None:
    env = _BranchEnv()
    live = _StatefulPolicy(calls=11, armed=True)
    outcome = campaign._branch_outcome(
        env,
        env.observation(),
        SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT),
        candidate_index=0,
        continuation_policy=live,
        horizon_ticks=3,
    )
    assert outcome.score_delta == 10
    assert live.calls == 11
    assert live.armed


def test_branch_probe_caps_waits_at_the_declared_horizon() -> None:
    env = _BranchEnv()
    outcome = campaign._branch_outcome(
        env,
        env.observation(),
        SteeringDecision(SemanticAction.wait(16), SteeringIntent.WAIT),
        candidate_index=0,
        continuation_policy=_WaitPolicy(),
        horizon_ticks=4,
    )
    assert outcome.survival_ticks == 4
    assert env.tick == 14


def test_large_primary_tie_gauge_gain_is_a_reserve_correction() -> None:
    reference = SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT)
    shot = SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
    )
    env = _BranchEnv(shot_score=0, shot_gauge=1_500)
    result = campaign.search_improvement(
        env,
        env.observation(),
        reference,
        (reference, shot),
        continuation_policy=_WaitPolicy(),
        seed=123,
        horizon_ticks=4,
        gauge_correction_threshold=1_000,
    )
    assert result.strict_improvement
    assert result.winner == 1
    assert result.label_reason == "gauge_reserve"

    small = _BranchEnv(shot_score=0, shot_gauge=999)
    below_threshold = campaign.search_improvement(
        small,
        small.observation(),
        reference,
        (reference, shot),
        continuation_policy=_WaitPolicy(),
        seed=123,
        horizon_ticks=4,
        gauge_correction_threshold=1_000,
    )
    assert not below_threshold.strict_improvement
    assert below_threshold.winner is None
    assert below_threshold.label_reason is None


def test_mixture_advances_only_the_executed_policy_shadow() -> None:
    observation = _BranchEnv().observation()
    base = _StatefulPolicy(calls=5)
    learner = _StatefulPolicy(calls=9)
    execute_base = campaign.predict_mixture(
        base, learner, observation, execute_base=True
    )
    assert execute_base.base_policy.calls == 6
    assert execute_base.learner_policy.calls == 9
    assert execute_base.base_continuation.calls == 5

    execute_learner = campaign.predict_mixture(
        _StatefulPolicy(calls=5),
        _StatefulPolicy(calls=9),
        observation,
        execute_base=False,
    )
    assert execute_learner.base_policy.calls == 5
    assert execute_learner.learner_policy.calls == 10
    assert execute_learner.base_continuation.calls == 5


def test_cli_selects_backend_specific_default_runtime(tmp_path: Path) -> None:
    portable = campaign.parse_args(["--output", str(tmp_path / "p")])
    exact = campaign.parse_args(
        ["--backend", "exact", "--output", str(tmp_path / "e")]
    )
    assert portable.runtime == campaign.PORTABLE
    assert exact.runtime == campaign.EXACT_WORKER
    assert portable.self_distillation_anchors
    ablation = campaign.parse_args(
        ["--no-self-distillation-anchors", "--output", str(tmp_path / "a")]
    )
    assert not ablation.self_distillation_anchors


def test_no_improvement_becomes_anchor_unless_ablation_disables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = SteeringDecision(SemanticAction.wait(1), SteeringIntent.WAIT)
    outcome = campaign.BranchOutcome(0, 4, 0, 100, False, False)
    result = campaign.SearchResult(
        7, 11, outcome, (outcome,), None, False, None, "a" * 64
    )
    seen: list[tuple[SteeringDecision, str]] = []

    def fake_make(
        observation: object,
        decision: SteeringDecision,
        result: object,
        *,
        label_kind: str,
        encoder: object,
        pointer_spec: object,
    ) -> str:
        seen.append((decision, label_kind))
        return label_kind

    monkeypatch.setattr(campaign, "make_distillation_label", fake_make)
    assert campaign.label_search_result(
        {"tick": 11},
        reference,
        (reference,),
        result,
        self_distillation_anchors=True,
        encoder=object(),
        pointer_spec=object(),
    ) == "anchor"
    assert seen == [(reference, "anchor")]
    assert campaign.label_search_result(
        {"tick": 11},
        reference,
        (reference,),
        result,
        self_distillation_anchors=False,
        encoder=object(),
        pointer_spec=object(),
    ) is None


def test_training_view_balances_label_kind_and_action_kind() -> None:
    labels: list[campaign.DistillationLabel] = []
    counts = {
        ("correction", True): 1,
        ("correction", False): 2,
        ("anchor", True): 3,
        ("anchor", False): 1,
    }
    for (kind, shot), count in counts.items():
        for index in range(count):
            provenance = {
                "schema": "irisu-expert-iteration-label-provenance-v1",
                "label_kind": kind,
                "search_sha256": "b" * 64,
                "shot": shot,
                "index": index,
            }
            example = SimpleNamespace(
                is_shot=shot,
                provenance_sha256=campaign._sha(provenance),
                sha256=f"{kind}:{shot}:{index}",
                group=(kind, shot),
            )
            labels.append(
                campaign.DistillationLabel(
                    example, kind, "b" * 64, provenance
                )
            )
    balanced = campaign.balanced_training_examples(labels)
    assert Counter(value.group for value in balanced) == {
        key: 3 for key in counts
    }


def test_checkpoint_seed_lineage_includes_upstream_and_rejects_malformed() -> None:
    assert campaign.checkpoint_training_seeds(
        {"demonstration_seeds": [3, 1], "training_seeds": [2, 3]}
    ) == (1, 2, 3)
    with pytest.raises(ValueError, match="training seeds"):
        campaign.checkpoint_training_seeds({})
    with pytest.raises(ValueError, match="uint32"):
        campaign.checkpoint_training_seeds({"training_seeds": [True]})


def test_provenance_publish_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "provenance.json"
    campaign._write_json_once(target, {"schema": "test-v1", "value": 1})
    original = target.read_bytes()
    assert not tuple(tmp_path.glob(".provenance.json.*.tmp"))
    with pytest.raises(FileExistsError):
        campaign._write_json_once(target, {"schema": "test-v2", "value": 2})
    assert target.read_bytes() == original
