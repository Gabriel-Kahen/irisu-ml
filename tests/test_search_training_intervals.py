"""Macro-interval labels must distinguish observation from endpoint absence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import irisu_search.coordinator as module
from irisu_search.coordinator import Coordinator
from irisu_search.neural import FEATURE_NAMES
from irisu_search.store import Store


def state(seed=7, score=0):
    return {"seed": seed, "tick": 0, "score": score, "gauge": 1000,
            "level": 10, "qualifying": 90, "highest_chain": 0,
            "terminated": False, "truncated": False, "canonical_score": 0,
            "groups": [], "bodies": []}


def planner():
    result = Coordinator.__new__(Coordinator)
    result.identity = {"engine": "interval-test"}
    return result


@pytest.mark.parametrize("max_new_chain,expected", [(3, 1), (2, 0)])
def test_fully_paid_new_group_is_labeled_from_observed_interval(max_new_chain, expected):
    before, after = state(), state(score=120)
    after["tick"] = 80
    # The group recruited and paid out entirely within the macro. Nothing in
    # the final board alone establishes whether a third member ever joined.
    row = planner()._example(before, after,
                             {"operator": "recruit", "budget_seconds": 1, "horizon": 80},
                             .1, metrics={"recruitment_events": max_new_chain,
                                          "new_groups": 1, "max_new_chain": max_new_chain,
                                          "qualifications": 2})
    assert row["targets"]["extension"] == expected
    assert row["targets"]["survival"] is None
    assert row["censored"] is True


def test_missing_interval_observations_do_not_make_absent_group_a_failure():
    before, after = state(), state(score=120)
    after["tick"] = 80
    row = planner()._example(before, after,
                             {"operator": "recruit", "budget_seconds": 1, "horizon": 80}, .1)
    assert row["targets"]["extension"] is None
    assert row["targets"]["survival"] is None


def test_extinct_parent_group_extension_uses_its_own_counter_delta():
    before, after = state(), state(score=120)
    before["groups"] = [{"id": 9, "chain": 3, "num": 1, "members": [4]}]
    before["bodies"] = [{"id": 4, "chain_id": 9, "color": 0, "x": 150, "y": 200,
                         "lifetime": 1000, "hits": 0}]
    after["tick"] = 80
    row = planner()._example(before, after,
                             {"operator": "recruit", "budget_seconds": 1, "horizon": 80}, .1,
                             group_deltas=[{"id": 9, "chain_before": 3, "chain_after": 4,
                                            "num_before": 1, "num_after": 3}],
                             metrics={"recruitment_events": 1, "new_groups": 0,
                                      "max_new_chain": 0, "qualifications": 2})
    assert row["targets"]["extension"] == 1
    assert row["targets"]["survival"] == 0


def test_shadow_training_skips_censored_unobserved_rows_across_many_seeds(tmp_path, monkeypatch):
    with Store(tmp_path) as store:
        for seed in range(8):
            store.add_example({"seed": seed, "features": [0.] * len(FEATURE_NAMES),
                               "targets": {"gain": 1_000_000, "compute_gain": 1_000_000,
                                           "extension": None, "survival": None},
                               "censored": True})
        coordinator = planner()
        coordinator.store = store
        coordinator.config = SimpleNamespace(seed_generator=0, max_training_examples=100,
                                             training_epochs=1)
        def never_train(*args, **kwargs):
            raise AssertionError("unobserved outcomes reached neural training")
        monkeypatch.setattr(module, "train_guide", never_train)
        assert coordinator.train_shadow() is None
        assert store.get_meta("latest_shadow_model") is None
        assert not (tmp_path / "models").exists()
