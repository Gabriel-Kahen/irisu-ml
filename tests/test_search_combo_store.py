from __future__ import annotations

import pytest

from irisu_search.store import Store
from test_search_combo_scheduler import checkpoint, champion, config


def test_frontier_persists_and_idempotently_records_expansions(tmp_path):
    with Store(tmp_path) as store:
        cfg = config()
        store.init_campaign(cfg.manifest(), {})
        root = checkpoint(store, 7)
        siblings = [checkpoint(store, 7, tick=3, word=n, chain=n+4) for n in range(1, 5)]
        first = store.update_combo_frontier(7, siblings, root, 4, "batch1")
        assert {row["checkpoint_id"] for row in first} == set(siblings)
        assert store.mark_combo_expanded(siblings[0], "job1:batch1")
        assert not store.mark_combo_expanded(siblings[0], "job1:batch1")
        assert not store.mark_combo_expanded(root, "root-not-frontier")
        store.update_combo_frontier(7, siblings, root, 4, "batch1")
        assert sum(row["visits"] for row in store.combo_frontier(7)) == 1
    with Store(tmp_path) as store:
        assert len(store.combo_frontier(7)) == 4
        assert store.combo_stats()["frontier_expansions"] == 1
        with pytest.raises(ValueError, match="different work"):
            store.update_combo_frontier(7, siblings[:1], root, 4, "batch1")


def test_frontier_keeps_unexpanded_alternatives_before_more_new_siblings(tmp_path):
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        old = [checkpoint(store, 7, tick=3, word=n, chain=10+n) for n in range(1, 5)]
        store.update_combo_frontier(7, old, root, 4)
        new = [checkpoint(store, 7, tick=4, word=n, chain=1) for n in range(5, 9)]
        rows = store.update_combo_frontier(7, new, root, 4)
        assert {row["checkpoint_id"] for row in rows} == set(old)
        for key in old:
            store.mark_combo_expanded(key)
        rows = store.update_combo_frontier(7, new, root, 4)
        assert any(row["checkpoint_id"] in new for row in rows)
        assert max(row["summary"]["groups"][0]["chain"] for row in rows) == 14


def test_frontier_rejects_cross_seed_and_terminal_states(tmp_path):
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        wrong = checkpoint(store, 8)
        terminal = checkpoint(store, 7, tick=4, terminal=True)
        with pytest.raises(ValueError, match="another seed"):
            store.update_combo_frontier(7, [wrong], root, 4)
        with pytest.raises(ValueError, match="live"):
            store.update_combo_frontier(7, [terminal], root, 4)


def test_frontier_is_protected_and_record_archive_has_capacity(tmp_path):
    cfg = config(control_seed=7)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        root = checkpoint(store, 7, operator="combo-root")
        store.set_meta("combo_roots:7", [root])
        siblings = [checkpoint(store, 7, tick=3, word=n, chain=10+n) for n in range(1, 5)]
        store.update_combo_frontier(7, siblings, root, 4)
        for seed in range(20, 60):
            checkpoint(store, seed, chain=100)
        result = store.prune_archive(per_cell=10, max_checkpoints=10)
        assert all(store.get_checkpoint(key)["active"] for key in siblings)
        assert store.get_checkpoint(root)["active"]
        assert result["active"] <= 10
        assert sum(row["seed"] == 7 for row in store.checkpoints()) >= 6


def test_combo_telemetry_distinguishes_root_coverage_and_novel_results(tmp_path):
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        store.set_meta("combo_roots:7", [root])
        store.enqueue("combo_terminal", 7, checkpoint_id=root)
        store.event("combo_terminal_evaluated", {"novel_terminals": 3, "baseline_ties": 1, "gain": 10})
        report = store.combo_stats()
        assert report["roots"] == report["roots_scheduled"] == 1
        assert report["novel_terminals"] == 3 and report["baseline_ties"] == 1
        assert report["improving_terminal_batches"] == 1


def test_stronger_arrivals_cannot_discard_all_unexpanded_siblings(tmp_path):
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        old = [checkpoint(store, 7, tick=3, word=n, chain=n+3) for n in range(1, 5)]
        store.update_combo_frontier(7, old, root, 4)
        stronger = checkpoint(store, 7, tick=4, word=10, chain=100)
        assert {row["checkpoint_id"] for row in store.update_combo_frontier(7, [stronger], root, 4)} == set(old)
        store.mark_combo_expanded(old[0])
        rows = store.update_combo_frontier(7, [stronger], root, 4)
        assert stronger in {row["checkpoint_id"] for row in rows}
        assert set(old[1:]) <= {row["checkpoint_id"] for row in rows}


def test_unbounded_discovery_does_not_protect_every_dormant_frontier(tmp_path):
    cfg = config(control_seed=7, promoted_seeds=2)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        root = checkpoint(store, 7, operator="combo-root")
        record = [checkpoint(store, 7, tick=3, word=n, chain=5) for n in range(1, 5)]
        store.update_combo_frontier(7, record, root, 4)
        for seed in range(20, 45):
            origin = checkpoint(store, seed)
            children = [checkpoint(store, seed, tick=3, word=n, chain=seed) for n in range(1, 5)]
            store.update_combo_frontier(seed, children, origin, 4)
        result = store.prune_archive(per_cell=4, max_checkpoints=24)
        assert result["active"] <= 24 and not result["protected_over_budget"]
        assert all(store.get_checkpoint(key)["active"] for key in record)
        assert store.combo_stats()["frontier_states"] == 104
        assert store.combo_stats()["active_frontier_states"] < 104


def test_candidate_dedup_is_root_scoped_and_mode_zero_ignores_unused_seed(tmp_path):
    with Store(tmp_path) as store:
        first = checkpoint(store, 7)
        second = checkpoint(store, 7, tick=3)
        pure = {"mode": 0, "words": [1, 0], "search_seed": 1}
        renamed = {**pure, "search_seed": 999, "operator": "another-label"}
        assert not store.combo_candidate_seen(first, pure, 8)
        assert store.record_combo_candidates(first, [pure, renamed], 8) == 1
        assert store.combo_candidate_seen(first, renamed, 8)
        assert not store.combo_candidate_seen(second, pure, 8)
        assert not store.combo_candidate_seen(first, pure, 9)
        assert store.record_combo_candidates(first, [renamed], 8) == 0
        assert store.combo_stats()["tried_candidates"] == 1


def test_reactive_candidate_dedup_keeps_different_controller_seeds(tmp_path):
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        reactive = {"mode": 1, "words": [], "search_seed": 1}
        other = {**reactive, "search_seed": 2}
        assert store.record_combo_candidates(root, [reactive], 8) == 1
        assert store.combo_candidate_seen(root, reactive, 8)
        assert not store.combo_candidate_seen(root, other, 8)
        assert store.record_combo_candidates(root, [other], 8) == 1
        assert store.combo_stats()["tried_candidates"] == 2


def test_terminal_plans_survive_restart_and_allow_compounding_edits(tmp_path):
    import struct
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        first = store.put_blob(struct.pack('<4I', 1, 0, 0, 0))
        second = store.put_blob(struct.pack('<4I', 1, 0, 2, 0))
        store.record_combo_plans(root, [{"plan_blob": first, "score": 100}], 4, "first")
        assert store.mark_combo_plan_expanded(root, first, "first-expand")
        assert not store.mark_combo_plan_expanded(root, first, "first-expand")
        store.record_combo_plans(root, [{"plan_blob": second, "score": 150}], 4, "second")
        # The planned suffix retains future words, rather than truncating to
        # the first plan's observed termination before the second edit.
        assert struct.unpack('<4I', store.get_blob(second)) == (1, 0, 2, 0)
    with Store(tmp_path) as store:
        rows = store.combo_plans(root)
        assert {row['plan_blob'] for row in rows} == {first, second}
        assert next(row for row in rows if row['plan_blob'] == first)['visits'] == 1
        assert store.combo_stats()['plan_expansions'] == 1
        assert store.combo_stats()['plan_states'] == 2


def test_terminal_plan_elites_preserve_unvisited_alternatives_and_root_scope(tmp_path):
    import struct
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        other_root = checkpoint(store, 8)
        blobs = [store.put_blob(struct.pack('<2I', n, 0)) for n in range(1, 6)]
        store.record_combo_plans(root, [{'plan_blob': blob, 'score': n} for n, blob in enumerate(blobs[:4])], 4)
        assert {row['plan_blob'] for row in store.record_combo_plans(root, [{'plan_blob': blobs[4], 'score': 100}], 4)} == set(blobs[:4])
        store.mark_combo_plan_expanded(root, blobs[0])
        assert blobs[4] in {row['plan_blob'] for row in store.record_combo_plans(root, [{'plan_blob': blobs[4], 'score': 100}], 4)}
        assert store.combo_plans(other_root) == []
        store.record_combo_plans(other_root, [{'plan_blob': blobs[4], 'score': 3}], 4)
        assert store.combo_plans(other_root)[0]['score'] == 3


def test_terminal_plan_validation_and_observation_idempotence(tmp_path):
    import struct
    with Store(tmp_path) as store:
        root = checkpoint(store, 7)
        blob = store.put_blob(struct.pack('<2I', 1, 0))
        entries = [{'plan_blob': blob, 'score': 10}]
        store.record_combo_plans(root, entries, 4, 'batch1')
        store.mark_combo_plan_expanded(root, blob)
        store.record_combo_plans(root, entries, 4, 'batch1')
        assert store.combo_plans(root)[0]['visits'] == 1
        with pytest.raises(ValueError, match='different work'):
            store.record_combo_plans(root, [{'plan_blob': blob, 'score': 11}], 4, 'batch1')
        incomplete = store.put_blob(b'abc')
        invalid = store.put_blob(struct.pack('<I', 0x200000))
        for bad in (incomplete, invalid):
            with pytest.raises(ValueError):
                store.record_combo_plans(root, [{'plan_blob': bad, 'score': 1}], 4)
