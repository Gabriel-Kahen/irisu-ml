from __future__ import annotations

from irisu_search.coordinator import Coordinator
from irisu_search.store import Store
from test_search_coordinator import FakeEngine, FakeVerifier, config


def coordinator(store, limit=4):
    engine = FakeEngine(terminal_tick=10)
    cfg = config(combo_search=True, combo_frontier_width=2,
                 combo_root_limit=limit, maximum_ticks=20)
    return Coordinator(store, engine, cfg, FakeVerifier(engine)), engine


def add(store, engine, seed, words, parent=None, operator="explore"):
    return store.add_checkpoint(seed, words, engine.summary(seed, words), parent, operator=operator)


def test_new_champion_rebuilds_later_event_roots_on_changed_recipe(tmp_path):
    with Store(tmp_path) as store:
        worker, engine = coordinator(store)
        original = [0] * 10
        old = []
        parent = None
        for tick in (2, 6, 8):
            parent = add(store, engine, 7, original[:tick], parent, "import")
            old.append(parent)
        terminal = add(store, engine, 7, original, parent, "import")
        assert worker._verify(terminal, force=True)
        store.set_meta("combo_roots:7", old)
        changed = original.copy()
        changed[3] = 1
        winner = add(store, engine, 7, changed, old[0], "combo-edit")
        assert worker._verify(winner)
        roots = [store.get_checkpoint(key) for key in store.get_meta("combo_roots:7")]
        assert [row["summary"]["tick"] for row in roots] == [2, 6, 8]
        assert all(row["words"] == changed[:row["summary"]["tick"]] for row in roots)
        assert all(row["summary"] == engine.summary(7, row["words"]) for row in roots)
        assert all(row["id"] not in old for row in roots if row["summary"]["tick"] > 3)
        assert all(store.get_checkpoint(key)["words"] == original[:tick]
                   for key, tick in zip(old, (2, 6, 8)))
        assert store.get_meta("baseline:7") == winner
        assert [len(words) for _, words in engine.loads] == [2, 6, 8]
        worker.close()


def test_fresh_seed_champion_gets_bounded_temporal_grid_and_ancestors(tmp_path):
    with Store(tmp_path) as store:
        worker, engine = coordinator(store)
        old = add(store, engine, 7, [0] * 10, operator="import")
        worker._verify(old, force=True)
        old_root = add(store, engine, 7, [0, 0], operator="import")
        store.set_meta("combo_roots:7", [old_root])
        start = add(store, engine, 11, [])
        ancestor = add(store, engine, 11, [0, 0], start)
        winner = add(store, engine, 11, [0] * 10, ancestor, "combo-edit")
        assert worker._verify(winner)
        roots = [store.get_checkpoint(key) for key in store.get_meta("combo_roots:11")]
        assert len(roots) == 4
        assert {row["summary"]["tick"] for row in roots} >= {0, 2, 9}
        assert all(row["seed"] == 11 and not row["summary"]["terminated"] for row in roots)
        assert store.get_meta("combo_roots:7") == [old_root]
        worker.close()


def test_per_seed_improvement_below_global_champion_does_not_rebuild_roots(tmp_path):
    with Store(tmp_path) as store:
        worker, engine = coordinator(store)
        old = add(store, engine, 7, [3] * 10, operator="import")
        worker._verify(old, force=True)
        winner = add(store, engine, 8, [0] * 10, operator="combo-edit")
        assert worker._verify(winner) is False
        assert store.get_meta("baseline:8") == winner
        assert store.get_meta("combo_roots:8") is None
        assert engine.loads == []
        worker.close()


def test_import_promotion_leaves_root_preparation_to_import_replay(tmp_path):
    with Store(tmp_path) as store:
        worker, engine = coordinator(store)
        terminal = add(store, engine, 7, [0] * 10, operator="import")
        assert worker._verify(terminal, force=True)
        assert store.get_meta("combo_roots:7") is None
        assert engine.loads == []
        worker.close()


def test_interrupted_winning_root_refresh_resumes_on_reopen(tmp_path, monkeypatch):
    import pytest
    with Store(tmp_path) as store:
        worker, engine = coordinator(store)
        old = add(store, engine, 7, [0] * 10, operator="import")
        worker._verify(old, force=True)
        root = add(store, engine, 7, [0, 0], operator="import")
        store.set_meta("combo_roots:7", [root])
        changed = [0, 1] + [0] * 8
        winner = add(store, engine, 7, changed, operator="combo-edit")

        def interrupted(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(engine, "load", interrupted)
        with pytest.raises(KeyboardInterrupt):
            worker._verify(winner)
        assert store.champion()["checkpoint_id"] == winner
        assert store.get_meta("combo_roots_pending") == winner
        worker.close()
    with Store(tmp_path) as store:
        resumed, _ = coordinator(store)
        roots = store.get_meta("combo_roots:7")
        assert store.get_meta("combo_roots_pending") is None
        assert store.get_checkpoint(roots[0])["words"] == changed[:2]
        resumed.close()
