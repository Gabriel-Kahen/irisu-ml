from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from irisu_search.coordinator import Coordinator
from irisu_search.proposals import encode, generate, select_candidates
from test_search_proposals import board, permute


def reference_words(length):
    return [encode(1 if i % 2 else 0, 10 + i, 30 + i) for i in range(length)]


@pytest.mark.parametrize("horizon", [1, 2, 3, 8, 64])
def test_replay_repairs_fit_horizon_and_keep_absolute_suffix_timing(horizon):
    reference = reference_words(horizon)
    rows = generate(board(), 32, 71, "repair", reference, horizon, reference)
    assert rows[0]["operator"] == "incumbent"
    assert rows[0]["words"] == reference
    repairs = [row for row in rows if row.get("replay_repair")]
    assert repairs
    for row in repairs:
        assert row["mode"] == 0
        assert len(row["words"]) == horizon
        assert all(type(word) is int and 0 <= word < 1 << 21 for word in row["words"])
        differences = [i for i, (a, b) in enumerate(zip(row["words"], reference)) if a != b]
        assert differences
        # Geometric repair prefixes are short; original future ticks are kept,
        # rather than appended after a shifted copy of the original suffix.
        assert row["words"][max(differences) + 1:] == reference[max(differences) + 1:]
        assert max(differences) < 3


def test_diverged_branch_can_follow_reference_without_claiming_incumbent():
    reference = reference_words(20)
    rows = generate(board(), 32, 12, "repair", [], 20, reference)
    assert not any(row["operator"] == "incumbent" for row in rows)
    continuation = next(row for row in rows if row["operator"] == "replay-reference")
    assert continuation["words"] == reference and continuation["mode"] == 0
    assert any(row["operator"] == "replay-repair" for row in rows)


def test_injected_replay_shot_releases_held_button_without_shifting_suffix():
    reference = reference_words(12)
    rows = generate(board(), 64, 19, "repair", reference, 12, reference, held=3)
    geometric = [row for row in rows if row.get("replay_repair")
                 and len(row["words"]) > 2 and row["words"][1] & 3]
    assert geometric
    for row in geometric:
        assert row["words"][0] == 0
        assert row["words"][2] == 0
        assert row["words"][3:] == reference[3:]


def test_replay_repair_capacity_survives_hostile_ranker_and_random_floor():
    reference = reference_words(16)
    rows = generate(board(), 64, 48, "repair", [], 16, reference)

    class Guide:
        def predict(self, summary, candidate):
            return {"priority": -1000 if candidate.get("replay_repair") else 1000}

    selected = select_candidates(board(), rows, 8, Guide(), exploration_fraction=.25, seed=17)
    assert len(selected) == len({id(row) for row in selected}) == 8
    assert selected[0] is rows[0]
    assert any(row["operator"] == "replay-reference" for row in selected)
    assert any(row["operator"] == "replay-repair" for row in selected)
    random_floor = random.Random(17).sample(rows[1:], 2)
    assert all(any(row is chosen for chosen in selected) for row in random_floor)


def test_reference_can_suppress_one_shot_without_a_geometric_target():
    reference = [0, encode(2, 100, 200), 0, 0, encode(1, 200, 100), 0]
    rows = generate({}, 16, 12, "repair", reference, len(reference), reference)
    mutation = next(row for row in rows if row.get("replay_repair"))
    assert mutation["words"] == [0, 0, *reference[2:]]
    assert mutation["mode"] == 0


def test_reference_cache_is_shared_across_divergent_parents_and_invalidated_on_import():
    reference = reference_words(20)
    loaded = []
    baseline = ["first"]

    def get_checkpoint(key):
        loaded.append(key)
        return {"words": reference if key == "first" else [0, *reference]}

    coordinator = Coordinator.__new__(Coordinator)
    coordinator.store = SimpleNamespace(get_meta=lambda _: baseline[0], get_checkpoint=get_checkpoint)
    coordinator.replay_references = {}
    parent = {"seed": 7, "summary": {"tick": 4}, "words": reference[:4]}
    assert coordinator._replay_inputs(parent, 8) == (reference[4:12], reference[4:12])
    changed = {**parent, "words": [encode(2, 50, 50), *reference[1:4]]}
    assert coordinator._replay_inputs(changed, 8) == ([], reference[4:12])
    advanced = {"seed": 7, "summary": {"tick": 12},
                "words": changed["words"] + reference[4:12]}
    assert coordinator._replay_inputs(advanced, 8) == ([], reference[12:20])
    assert loaded == ["first"]
    baseline[0] = "second"
    assert coordinator._replay_inputs(changed, 8) == ([], [0, *reference][4:12])
    assert loaded == ["first", "second"]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_reference_repairs_respect_small_catalog_and_selection_capacity(count):
    reference = reference_words(12)
    rows = generate(board(), count, 11, "repair", reference, 12, reference)
    assert len(rows) == count
    assert rows[0]["words"] == reference
    selected = select_candidates(board(), rows, count)
    assert len(selected) == count
    assert selected[0] is rows[0]


def test_reference_repairs_do_not_depend_on_entity_identifiers():
    reference = reference_words(12)
    assert generate(board(), 32, 11, "repair", reference, 12, reference) == generate(
        permute(board()), 32, 11, "repair", reference, 12, reference)
