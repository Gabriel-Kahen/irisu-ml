from __future__ import annotations

import copy
import math
import random

import pytest

from irisu_search.neural import FEATURE_NAMES, features
from irisu_search.proposals import (
    encode, generate, live_groups, potential, select_candidates, select_survivors, target_group,
)


def board():
    return {
        "seed": 30, "tick": 10000, "level": 30, "qualifying": 290, "score": 12000,
        "gauge": 10000, "terminated": False,
        "groups": [
            {"id": 20, "chain": 12, "num": 5, "members": [1, 2]},
            {"id": 10, "chain": 12, "num": 5, "members": [6]},
        ],
        "bodies": [
            {"id": 1, "chain_id": 20, "color": 1, "x": 220, "y": 200, "size": 30, "hits": 1},
            {"id": 2, "chain_id": 20, "color": 1, "x": 310, "y": 240, "size": 30, "hits": 0},
            {"id": 3, "chain_id": 0, "color": 1, "x": 280, "y": 80, "size": 30},
            {"id": 4, "chain_id": 0, "color": 1, "x": 340, "y": 100, "size": 30},
            {"id": 5, "chain_id": 0, "color": 3, "x": 240, "y": 150, "size": 30},
            {"id": 6, "chain_id": 10, "color": 2, "x": 340, "y": 200, "size": 30, "hits": 0},
        ],
    }


def permute(summary):
    result = copy.deepcopy(summary)
    result["seed"] = 999
    for group in result["groups"]:
        group["id"] += 1000
        group["members"] = list(reversed([i + 2000 for i in group["members"]]))
    for body in result["bodies"]:
        body["id"] += 2000
        if body["chain_id"]:
            body["chain_id"] += 1000
    result["groups"].reverse()
    result["bodies"].reverse()
    return result


def test_proposals_are_physical_order_and_identifier_invariant():
    original = board()
    changed = permute(original)
    for operator in ("recruit", "preserve", "handoff", "payout", "prepare", "timing", "repair"):
        assert generate(original, 48, 712, operator) == generate(changed, 48, 712, operator)
    assert potential(original) == potential(changed)
    assert features(original) == features(changed)
    assert target_group(original)[0]["id"] + 1000 == target_group(changed)[0]["id"]


def test_canonical_target_group_and_features_agree_on_tied_groups():
    summary = board()
    summary["groups"][0]["members"] = [1]
    # Same counter product and number of members; geometry resolves the tie.
    target, members = target_group(summary)
    encoded = features(summary)
    assert encoded[FEATURE_NAMES.index("best_carrier_y")] == members[0]["y"] / 480
    assert encoded[FEATURE_NAMES.index("best_carrier_hits")] == members[0].get("hits", 0) / 2
    assert target["id"] == target_group(permute(summary))[0]["id"] - 1000


@pytest.mark.parametrize("horizon", [1, 2, 3, 8, 32])
def test_every_prefix_fits_short_horizon_and_replay_encoding(horizon):
    rows = generate(board(), 64, 99, "timing", baseline=[0, 1, 0] * 10, horizon=horizon)
    assert len(rows) == 64
    assert rows[0]["operator"] == "incumbent"
    for row in rows:
        assert len(row["words"]) <= horizon
        assert all(type(word) is int and 0 <= word < 1 << 21 for word in row["words"])


def test_does_not_propose_merging_groups_or_second_hit_on_carrier():
    summary = board()
    summary["bodies"] = summary["bodies"][:2]
    summary["groups"] = [{"id": 20, "chain": 12, "num": 5, "members": [1]},
                         {"id": 21, "chain": 5, "num": 2, "members": [2]}]
    summary["bodies"][1]["chain_id"] = 21
    for row in generate(summary, 40, 12, "recruit"):
        assert not any(word & 3 for word in row["words"])
    summary["bodies"][1]["hits"] = 1
    for row in generate(summary, 40, 12, "preserve"):
        assert not any(word & 3 for word in row["words"])


def test_pending_or_expiring_carrier_is_not_a_live_exploration_target():
    summary = board()
    for body in summary["bodies"]:
        if body.get("chain_id"):
            body["pending_clear"] = True
    assert live_groups(summary) == []
    assert target_group(summary) is None
    assert potential(summary) == 0
    summary = board()
    for body in summary["bodies"]:
        body["lifetime"] = 1
    assert target_group(summary) is None


def test_candidate_capacity_one_retains_only_incumbent():
    rows = generate(board(), 16, 30)
    assert select_candidates(board(), rows, 1) == rows[:1]


def test_neural_ranker_cannot_remove_incumbent_or_random_floor():
    class Guide:
        def predict(self, summary, candidate):
            return {"priority": candidate["rank"]}

    rows = [{"rank": i} for i in range(20)]
    selected = select_candidates({}, rows, 5, Guide(), exploration_fraction=.5, seed=17)
    floor = random.Random(17).sample(rows[1:], math.ceil(4 * .5))
    assert len(selected) == 5
    assert selected[0] is rows[0]
    assert all(any(row is item for row in selected) for item in floor)
    assert len({id(row) for row in selected}) == 5
    assert rows[19] in selected


def test_single_survivor_retains_live_progress_and_records_have_no_completion_bias():
    live = {"handle": 1, "summary": board()}
    complete = {"handle": 2, "summary": {"terminated": True, "level": 100, "canonical_score": 400}}
    gameover = {"handle": 3, "summary": {"terminated": True, "level": 80, "canonical_score": 900}}
    assert select_survivors([complete, gameover, live], 1) == [live]
    assert select_survivors([complete, gameover, live], 2) == [live, gameover]
    assert select_survivors([complete, gameover], 1) == [gameover]


def test_trained_value_head_retains_delayed_payoff_without_displacing_all_heuristics():
    rows = []
    for handle, chain, score in ((1, 60, 10), (2, 2, 100), (3, 3, 20)):
        summary = board()
        summary["groups"] = [{"id": 20, "chain": chain, "num": 5, "members": [1]}]
        summary["score"] = score
        summary["scenario"] = handle
        rows.append({"handle": handle, "summary": summary})

    class Guide:
        head_counts = [32, 0, 0, 0]

        def predict(self, summary, candidate):
            assert not summary["terminated"]
            assert candidate["mode"] == 1
            assert candidate["horizon"] == 800
            assert candidate["budget_seconds"] == 12
            return {"predicted_gain": 1000 if summary["scenario"] == 3 else 0}

    original = copy.deepcopy(rows)
    assert [r["handle"] for r in select_survivors(rows, 2)] == [1, 2]
    assert [r["handle"] for r in select_survivors(rows, 2, Guide(), horizon=800, budget_seconds=12)] == [1, 3]
    assert [r["handle"] for r in select_survivors(rows, 3, Guide(), horizon=800, budget_seconds=12)] == [1, 3, 2]
    assert rows == original  # Predictions cannot replace realized game fields.


def test_sparse_or_untrained_value_head_and_width_one_preserve_old_behavior():
    live = {"handle": 1, "summary": board()}
    terminal = {"handle": 2, "summary": {"terminated": True, "canonical_score": 999}}

    class Untrained:
        head_counts = [31, 1000, 1000, 1000]

        def predict(self, *args):
            raise AssertionError("sparse value head must not guide live selection")

    assert select_survivors([terminal, live], 2, Untrained()) == select_survivors([terminal, live], 2)
    Untrained.head_counts[0] = 32
    assert select_survivors([terminal, live], 1, Untrained()) == [live]


def test_value_head_never_ranks_terminal_records_by_predicted_score():
    low = {"handle": 1, "summary": {"terminated": True, "canonical_score": 100, "score": 100000}}
    high = {"handle": 2, "summary": {"terminated": True, "canonical_score": 900, "score": 900}}

    class Guide:
        head_counts = [100, 100, 100, 100]

        def predict(self, *args):
            raise AssertionError("a terminal record must never be valued by a network")

    assert select_survivors([low, high], 2, Guide()) == [high]


def test_short_operator_aliases_have_distinct_neural_features():
    for short, long in (("recruit", "recruitment"), ("preserve", "preservation"),
                        ("prepare", "timing"), ("incumbent", "baseline")):
        assert features(board(), {"operator": short}) == features(board(), {"operator": long})
    assert features(board(), {"operator": "recruit"}) != features(board(), {"operator": "preserve"})


def test_training_label_uses_same_canonical_group_as_features():
    from irisu_search.coordinator import Coordinator

    coordinator = Coordinator.__new__(Coordinator)
    coordinator.identity = {"planner": "test", "engine": "fixture"}
    before = board()
    before["groups"][0]["members"] = [1]
    chosen = target_group(before)[0]["id"]
    after = copy.deepcopy(before)
    for group in after["groups"]:
        if group["id"] == chosen:
            group["chain"] += 1
    row = coordinator._example(before, after, {
        "operator": "recruit", "mode": 1, "budget_seconds": 2, "horizon": 40,
    }, .1)
    assert row["targets"]["extension"] == 1
    assert row["censored"] is True
    assert "compute_gain" not in row["targets"]


def test_extending_then_fully_paying_out_is_positive_extension_supervision():
    from irisu_search.coordinator import Coordinator

    coordinator = Coordinator.__new__(Coordinator)
    coordinator.identity = {"planner": "test", "engine": "fixture"}
    before = board()
    chosen = target_group(before)[0]
    after = copy.deepcopy(before)
    after["groups"] = []
    after["bodies"] = []
    row = coordinator._example(before, after, {
        "operator": "payout", "mode": 3, "budget_seconds": 2, "horizon": 40,
    }, .1, [{"id": chosen["id"], "chain_before": chosen["chain"], "num_before": chosen["num"],
             "chain_after": chosen["chain"] + 3, "num_after": chosen["num"] + 2, "members_after": 0}])
    assert row["targets"]["extension"] == 1
    assert row["targets"]["survival"] == 0


def test_invalid_counts_and_nonfinite_coordinates_reject():
    with pytest.raises(ValueError):
        generate({}, 0, 0)
    with pytest.raises(ValueError):
        select_candidates({}, [], 0)
    with pytest.raises(ValueError):
        select_survivors([], 0)
    with pytest.raises(ValueError):
        encode(2, float("nan"), 50)


@pytest.mark.parametrize("operator", ["preserve", "handoff", "payout", "repair"])
def test_dense_pair_fanout_preserves_carrier_intervention_budget(operator):
    carrier = {"id": 1, "kind": 0, "color": 0, "x": 150, "y": 220,
               "size": 30, "chain_id": 1, "hits": 0, "lifetime": 90000}
    fresh = [{"id": i + 2, "kind": 0, "color": 0, "x": 200 + i * 5, "y": 100,
              "size": 30, "chain_id": 0, "hits": 0, "lifetime": 90000} for i in range(70)]
    summary = {"bodies": [carrier, *fresh],
               "groups": [{"id": 1, "chain": 30, "num": 20, "members": [1]}]}
    rows = generate(summary, 256, 123, operator)
    assert len(rows) == 256
    assert rows[0]["operator"] == "incumbent"
    # Fresh-piece shots have y=116; a real carrier intervention has y=236.
    is_carrier = lambda row: row["words"] and row["words"][0] & 3 and (row["words"][0] >> 12 & 511) > 200
    assert any(is_carrier(row) for row in rows)
    assert any(row["words"] and row["words"][0] & 3 and not is_carrier(row) for row in rows)
    assert rows == generate(permute(summary), 256, 123, operator)
    # Damaged carriers are never made into second direct-hit targets.
    carrier["hits"] = 1
    assert not any(is_carrier(row) for row in generate(summary, 256, 123, operator))


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_small_operator_budgets_keep_baseline_and_fit_capacity(count):
    rows = generate(board(), count, 123, "preserve")
    assert len(rows) == count
    assert rows[0]["operator"] == "incumbent"
    if count > 1:
        assert any(row["words"] and row["words"][0] & 3 for row in rows[1:])
