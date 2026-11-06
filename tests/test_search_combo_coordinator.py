from __future__ import annotations

import json

import pytest

from irisu_search.coordinator import Coordinator
from irisu_search.store import Store
from test_search_coordinator import FakeEngine, FakeVerifier, config


def cfg(**changes):
    return config(combo_search=True, combo_frontier_width=2, **changes)


def proposals(*args):
    horizon = args[4]
    return [{"mode": 0, "words": [word, 0][:horizon], "search_seed": word,
             "operator": "combo-reference" if word == 0 else "combo-edit", "horizon": horizon}
            for word in (0, 1, 2, 3)]


def test_terminal_evidence_counts_novel_verified_gain_without_baseline_credit(tmp_path, monkeypatch):
    import irisu_search.coordinator as module
    monkeypatch.setattr(module, "terminal_candidates", proposals)
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, cfg(), FakeVerifier(engine))
        root = engine.load(7, [])
        key = store.add_checkpoint(7, [], root["summary"])
        engine.release([root["handle"]])
        store.enqueue("combo_terminal", 7, key, {"search_seed": 1, "horizon": 2})
        coordinator.run(seconds=1, max_jobs=1)
        events = [json.loads(row[0]) for row in store.db.execute(
            "SELECT payload FROM events WHERE kind='combo_terminal_evaluated'")]
        assert len(events) == 1
        event = events[0]
        assert event["terminals"] == 4 and event["novel_terminals"] == 3
        assert event["gain"] == 3 and event["censored"] == 0
        assert store.champion()["score"] == event["best_novel_score"]
        assert len(coordinator.verifier.verified) == 2  # Novel winner and paired reference.
        coordinator.close()


def test_capped_payout_is_unresolved_and_frontier_survives_restart(tmp_path, monkeypatch):
    import irisu_search.coordinator as module
    monkeypatch.setattr(module, "terminal_candidates", proposals)
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=6)
        coordinator = Coordinator(store, engine, cfg(), FakeVerifier(engine))
        root = engine.load(7, [])
        key = store.add_checkpoint(7, [], root["summary"])
        engine.release([root["handle"]])
        store.enqueue("combo_terminal", 7, key, {"search_seed": 1, "horizon": 2})
        coordinator.run(seconds=1, max_jobs=1)
        event = json.loads(store.db.execute(
            "SELECT payload FROM events WHERE kind='combo_terminal_evaluated'").fetchone()[0])
        assert event["censored"] == 4
        assert event["gain"] is None and event["best_novel_score"] is None
        assert not coordinator.verifier.verified and store.champion() is None
        frontier = {row["checkpoint_id"] for row in store.combo_frontier(7)}
        assert len(frontier) == 2
        coordinator.close()
    with Store(tmp_path) as store:
        assert {row["checkpoint_id"] for row in store.combo_frontier(7)} == frontier
        assert all(store.get_checkpoint(key)["active"] for key in frontier)


def test_verification_failure_still_accounts_completed_rollout_work(tmp_path):
    class BrokenVerifier(FakeVerifier):
        def verify(self, checkpoint):
            raise RuntimeError("independent mismatch")

    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, cfg(), BrokenVerifier(engine))
        store.enqueue("bootstrap", 7, payload={"search_seed": 1, "horizon": 2})
        with pytest.raises(RuntimeError, match="independent mismatch"):
            coordinator.run(seconds=1, max_jobs=1)
        assert store.jobs()[0]["status"] == "failed"
        assert store.report()["simulated_ticks"] == 8
        coordinator.close()


def test_terminal_plan_retains_unexecuted_inputs_and_is_used_for_next_edit(tmp_path, monkeypatch):
    import struct
    import irisu_search.coordinator as module
    references = []

    def planned(summary, count, seed, reference, horizon, *args):
        references.append(list(reference))
        return [{"mode": 0, "words": [0] * horizon, "search_seed": 0, "operator": "combo-reference", "horizon": horizon},
                {"mode": 0, "words": [1, 0, 2, 0], "search_seed": 1, "operator": "combo-edit", "horizon": horizon}]

    monkeypatch.setattr(module, "terminal_candidates", planned)
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, cfg(), FakeVerifier(engine))
        row = engine.load(7, [])
        root = store.add_checkpoint(7, [], row["summary"])
        engine.release([row["handle"]])
        store.enqueue("combo_terminal", 7, root, {"search_seed": 1, "horizon": 4})
        coordinator.run(seconds=1, max_jobs=1)
        plan = store.combo_plans(root)[0]
        assert struct.unpack('<4I', store.get_blob(plan['plan_blob'])) == (1, 0, 2, 0)
        store.enqueue("combo_terminal", 7, root, {"search_seed": 2, "horizon": 4,
                                                  "plan_blob": plan['plan_blob']})
        coordinator.run(seconds=1, max_jobs=1)
        assert [1, 0, 2, 0] in references
        assert store.combo_stats()["plan_expansions"] == 1
        coordinator.close()


def test_interrupted_observation_resumes_with_dedup_and_accounts_both_batches(tmp_path, monkeypatch):
    import irisu_search.coordinator as module
    monkeypatch.setattr(module, "terminal_candidates", proposals)
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, cfg(), FakeVerifier(engine))
        row = engine.load(7, [])
        root = store.add_checkpoint(7, [], row["summary"])
        engine.release([row["handle"]])
        job = store.enqueue("combo_terminal", 7, root, {"search_seed": 1, "horizon": 2})
        original = store.record_combo_plans

        def interrupt_after_observation(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(store, "record_combo_plans", interrupt_after_observation)
        with pytest.raises(KeyboardInterrupt):
            coordinator.run(seconds=1, max_jobs=1)
        assert store.jobs()[0]["status"] == "pending"
        assert store.get_meta(f"progress:{job}")["simulated_ticks"] == 8
        coordinator.close()
        monkeypatch.setattr(store, "record_combo_plans", original)
        engine2 = FakeEngine(terminal_tick=2)
        resumed = Coordinator(store, engine2, cfg(), FakeVerifier(engine2))
        resumed.run(seconds=1, max_jobs=1)
        assert store.jobs()[0]["status"] == "done"
        assert store.report()["simulated_ticks"] == 10  # Four original futures, one retained reference retry.
        resumed.close()
