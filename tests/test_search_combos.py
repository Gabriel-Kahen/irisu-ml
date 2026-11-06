from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from irisu_search.combos import combo_key, descriptor, event_ticks, outcome, select_frontier, terminal_candidates
from irisu_search.proposals import encode


def board(chain=20, num=10, carriers=1, score=0, x=200, hits=0):
    return {"seed": 7, "tick": 1000, "score": score, "gauge": 12000,
            "terminated": False, "truncated": False,
            "groups": [{"id": 9, "chain": chain, "num": num,
                        "members": list(range(1, carriers + 1))}],
            "bodies": [{"id": i, "chain_id": 9, "kind": "piece", "color": 2,
                        "x": x + i, "y": 180, "size": 30, "hits": hits,
                        "rot_timer": 0, "lifetime": 90000}
                       for i in range(1, carriers + 1)]}


def test_dominant_combo_outranks_fragmented_potential():
    large = board(chain=40, num=25)
    fragments = board(chain=5, num=3)
    for n in range(10, 25):
        fragments["groups"].append({"id": n, "chain": 5, "num": 3, "members": [n]})
        fragments["bodies"].append({**fragments["bodies"][0], "id": n, "chain_id": n})
    assert descriptor(large)["group_value"] == 1040
    assert combo_key(large) > combo_key(fragments)


def test_descriptor_separates_cumulative_counters_from_carrier_count():
    summary = board(chain=70, num=54, carriers=2, hits=1)
    d = descriptor(summary)
    assert (d["chain"], d["num"], d["carriers"]) == (70, 54, 2)
    assert d["one_hit_carriers"] == 2 and d["safe_carriers"] == 0
    assert d["last_carrier_risk"] >= .65
    assert "predicted_score" not in d
    healthy = board(chain=70, num=54, carriers=2)
    assert combo_key(healthy) > combo_key(summary)


def test_pending_payout_members_are_not_usable_carriers():
    summary = board(carriers=2)
    summary["bodies"][0]["pending_clear"] = True
    summary["bodies"][1]["pending_delete"] = True
    assert descriptor(summary)["carriers"] == 0


def test_descriptor_and_key_are_invariant_under_entity_renumbering():
    original = board(carriers=2)
    changed = deepcopy(original)
    changed["groups"][0].update(id=999, members=[102, 101])
    for body in changed["bodies"]:
        body.update(id=body["id"] + 100, chain_id=999)
    changed["bodies"].reverse()
    assert descriptor(original) == descriptor(changed)
    assert combo_key(original) == combo_key(changed)


def test_outcome_credits_extension_and_payout_after_extinction():
    before = board(chain=20, num=10, score=100)
    after = {**board(score=2000), "groups": [], "bodies": []}
    result = outcome(before, after, [{"id": 9, "chain_before": 20, "num_before": 10,
                                    "chain_after": 23, "num_after": 13, "members_after": 0}])
    assert result == {"target_group_id": 9, "chain_gain": 3, "num_gain": 3,
                      "carrier_survived": False, "extinct": True,
                      "realized_score_gain": 1900}
    assert outcome(before, after)["chain_gain"] is None
    assert outcome({"groups": [], "bodies": []}, after)["extinct"] is None


def test_frontier_keeps_live_combo_score_and_safe_alternative():
    rows = [{"handle": 1, "words": [1], "summary": board(chain=50, num=30, hits=1)},
            {"handle": 2, "words": [2], "summary": board(chain=20, num=10, score=5000)},
            {"handle": 3, "words": [3], "summary": board(chain=30, num=20, carriers=3)},
            {"handle": 4, "words": [4], "summary": {**board(chain=1000), "terminated": True}}]
    selected = select_frontier(rows, 3)
    assert [row["handle"] for row in selected] == [1, 2, 3]
    assert select_frontier(rows, 1)[0]["handle"] == 1


def test_frontier_recipe_dedup_does_not_merge_different_parents():
    row = {"handle": 1, "words": [1, 0], "parent_id": "a", "summary": board()}
    duplicate = {**row, "handle": 2}
    other = {**row, "handle": 3, "parent_id": "b"}
    assert len(select_frontier([row, duplicate, other], 4)) == 2
    assert select_frontier([{**row, "summary": {**board(), "truncated": True}}], 2) == []
    with pytest.raises(ValueError):
        select_frontier([row], 0)


def genealogy(tmp_path):
    identity = {"kind": "identity", "schema": "irisu-search-genealogy-v1",
                "seed": 7, "header_score": 1000, "input_words": 10000}
    summary = {"kind": "summary", "verified": True, "seed": 7,
               "canonical_score": 1000, "canonical_attributed_score": 1000,
               "unassigned_score_events": 0, "tick": 10000,
               "groups": [{"id": 1, "chain": 70, "num": 54, "score_canonical": 990,
                           "first_tick": 1000, "last_tick": 9000},
                          {"id": 2, "chain": 5, "num": 2, "score_canonical": 10,
                           "first_tick": 100, "last_tick": 200}]}
    events = [{"kind": "join", "group_id": 1, "tick": tick, "chain": n + 2, "num": n}
              for n, tick in enumerate(range(1000, 9500, 500))]
    events += [{"kind": "carriers", "group_id": 1, "tick": 6000,
                "chain": 40, "num": 30, "before": 2, "after": 1},
               {"kind": "join", "group_id": 2, "tick": 100, "chain": 2, "num": 0}]
    trace = tmp_path / "genealogy.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in [identity, *events, summary]))
    receipt = {"output": trace.name, "replay_sha256": "a" * 64,
               "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
               "identity": identity, "summary": summary}
    path = tmp_path / "genealogy.receipt.json"
    path.write_text(json.dumps(receipt, indent=2))
    return path, trace, receipt


def test_event_roots_cover_combo_lifetime_and_precede_risky_events(tmp_path):
    path, _, _ = genealogy(tmp_path)
    roots = event_ticks(path, "a" * 64, 7, 1000, limit=8, lead=48)
    assert len(roots) == 8 and roots == sorted(set(roots))
    assert 952 in roots and 8952 in roots
    assert min(roots) >= 952  # unrelated small group does not consume roots
    assert event_ticks(path, "a" * 64, 7, 1000, limit=8, lead=48) == roots


@pytest.mark.parametrize("key,value", [("seed", 8), ("header_score", 999), ("schema", "wrong")])
def test_event_roots_require_matching_verified_genealogy(tmp_path, key, value):
    path, trace, receipt = genealogy(tmp_path)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[0][key] = value
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    receipt["identity"] = rows[0]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        event_ticks(path, "a" * 64, 7, 1000)


def test_event_roots_reject_wrong_replay_tampering_and_unbound_raw_trace(tmp_path):
    path, trace, _ = genealogy(tmp_path)
    with pytest.raises(ValueError, match="different replay"):
        event_ticks(path, "b" * 64, 7, 1000)
    with pytest.raises(ValueError, match="matching replay hash"):
        event_ticks(trace, "a" * 64, 7, 1000)
    trace.write_text(trace.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        event_ticks(path, "a" * 64, 7, 1000)


def reference():
    return [encode(1 if index % 8 == 3 else 0, 100 + index, 200)
            for index in range(96)]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 16, 32])
def test_terminal_candidates_are_bounded_legal_distinct_and_reproducible(count):
    words = reference()
    rows = terminal_candidates(board(), count, 41, words, len(words))
    assert 1 <= len(rows) <= count
    assert rows[0]["words"] == words and rows[0]["operator"] == "combo-reference"
    assert all(row["mode"] == 0 and len(row["words"]) <= len(words) for row in rows)
    assert all(type(word) is int and 0 <= word < 1 << 21 for row in rows for word in row["words"])
    assert len({tuple(row["words"]) for row in rows}) == len(rows)
    assert rows == terminal_candidates(board(), count, 41, words, len(words))
    rows[0]["words"][0] = 0
    assert words == reference()


def test_terminal_mutations_cover_late_shots_and_small_spatial_recovery():
    words = reference()
    rows = terminal_candidates(board(), 16, 17, words, len(words))
    edits = [row["combo_edit"] for row in rows if row["combo_edit"]]
    assert {edit["amount"] for edit in edits if edit["kind"] == "x" and edit["tick"] == 3} >= {-2, 2}
    assert {edit["amount"] for edit in edits if edit["kind"] == "y" and edit["tick"] == 3} >= {-2, 2}
    assert {edit["amount"] for edit in edits if edit["kind"] == "timing" and edit["tick"] == 3} >= {-2, 2}
    assert any(edit["tick"] > len(words) // 2 for edit in edits)
    assert len({edit["tick"] for edit in edits}) >= 3
    for row in rows:
        edit = row["combo_edit"]
        if edit and edit["kind"] == "timing":
            target = edit["tick"] + edit["amount"]
            kind = words[edit["tick"]] & 3
            assert row["words"][target] & kind
            assert target == 0 or not row["words"][target - 1] & kind


def test_combo_shot_deletion_preserves_already_held_other_button():
    words = [encode(3, 100, 200), encode(3, 100, 200), encode(2, 100, 200), 0]
    rows = terminal_candidates(board(), 32, 1, words, 4, held=2)
    deleted = next(row for row in rows if row["combo_edit"] == {"kind": "delete", "tick": 0, "amount": 0})
    assert [word & 3 for word in deleted["words"]] == [2, 2, 2, 0]


def test_advanced_shot_never_reuses_an_already_held_button_edge():
    words = [0, 0, encode(1, 100, 200), 0, 0, encode(1, 120, 200), 0, 0]
    rows = terminal_candidates(board(), 32, 7, words, len(words), held=1)
    assert not any(row["combo_edit"] == {"kind": "timing", "tick": 2, "amount": -2}
                   for row in rows)
    for row in rows:
        edit = row["combo_edit"]
        if edit and edit["kind"] == "timing":
            destination = edit["tick"] + edit["amount"]
            kind = row["words"][destination] & 3
            previous = (row["words"][destination - 1] & 3) if destination else 1
            assert kind & ~previous


def test_empty_reference_uses_diverse_reactive_combo_continuations():
    rows = terminal_candidates(board(), 16, 3, [], 512)
    assert len(rows) == 16 and rows[0]["mode"] == 0
    assert any(row["mode"] for row in rows[1:])
    assert len({(row["mode"], row["search_seed"]) for row in rows}) == len(rows)


def test_handoff_injection_releases_held_edge_without_shifting_reference():
    summary = board(carriers=2)
    words = reference()
    rows = terminal_candidates(summary, 16, 2, words, len(words), held=3)
    handoff = next(row for row in rows if row["operator"] == "combo-handoff")
    assert handoff["words"][0] == 0 and handoff["words"][1] & 3
    assert handoff["words"][2] == 0
    assert handoff["words"][3:] == words[3:]
