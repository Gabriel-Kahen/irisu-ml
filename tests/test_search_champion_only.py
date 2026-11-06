from __future__ import annotations

import pytest

from irisu_search.cli import parser
from irisu_search.coordinator import Coordinator
from irisu_search.scheduler import choose_lane, next_job
from irisu_search.store import Store
from test_search_combo_scheduler import champion, checkpoint, last_job
from test_search_coordinator import FakeEngine, FakeVerifier, config


@pytest.mark.parametrize("record_seed", [0, 11])
@pytest.mark.parametrize("combo", [False, True])
@pytest.mark.parametrize("lane", ["seed", "combo", "probe", "wildcard"])
def test_every_lane_works_only_on_verified_champion(tmp_path, monkeypatch, combo, lane, record_seed):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: lane)
    cfg = config(champion_only=True, combo_search=combo, combo_frontier_width=2,
                 maximum_ticks=1000)
    with Store(tmp_path) as store:
        champion(store, record_seed)
        checkpoint(store, record_seed, operator="import")
        checkpoint(store, 7, chain=100)
        checkpoint(store, 13, chain=200)
        known = store.seeds()
        for _ in range(8):
            job = last_job(store, next_job(store, cfg))
            assert job["seed"] == record_seed
            assert job["kind"] != "bootstrap"
            assert job["lane"] != "seed"
        assert [row["seed"] for row in store.seeds()] == [row["seed"] for row in known]


@pytest.mark.parametrize("combo", [False, True])
def test_no_live_state_restarts_only_the_champion_seed(tmp_path, combo):
    cfg = config(champion_only=True, combo_search=combo, combo_frontier_width=2)
    with Store(tmp_path) as store:
        champion(store, 11)
        store.update_seed(11, attempts=10)
        checkpoint(store, 7, chain=100)
        for _ in range(4):
            job = last_job(store, next_job(store, cfg))
            assert job["seed"] == 11 and job["kind"] == "bootstrap"
        assert {row["seed"] for row in store.seeds()} == {7, 11}


def test_seed_budget_goes_to_improving_champion(tmp_path):
    cfg = config(champion_only=True, lanes={"seed": .9, "combo": .04, "probe": .03, "wildcard": .03})
    with Store(tmp_path) as store:
        assert choose_lane(store, cfg) == "combo"


def test_mode_requires_champion_and_does_not_initialize_other_seeds(tmp_path):
    cfg = config(champion_only=True)
    engine = FakeEngine()
    with Store(tmp_path) as store:
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        assert store.seeds() == [] and store.get_meta("seed_manifest") == []
        with pytest.raises(ValueError, match="verified champion"):
            coordinator.run(seconds=1, max_jobs=1)
        assert not engine.loads and not engine.batches and not store.jobs()
        with pytest.raises(ValueError, match="verified champion"):
            next_job(store, cfg)
        assert store.get_meta("schedule_sequence") is None
        coordinator.close()


def test_run_skips_other_pending_jobs_and_keeps_focus_after_restart(tmp_path):
    cfg = config(champion_only=True, maximum_ticks=20)
    engine = FakeEngine()
    with Store(tmp_path) as store:
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        root = engine.summary(11, [])
        store.add_checkpoint(11, [], root)
        words = [0] * engine.terminal_tick
        key = store.add_checkpoint(11, words, engine.summary(11, words))
        coordinator._verify(key, force=True)
        other = store.enqueue("bootstrap", 7, priority=100, payload={"horizon": 2})
        coordinator.run(seconds=30, max_jobs=2)
        assert last_job(store, other)["status"] == "pending"
        assert all(row["seed"] == 11 for row in store.jobs("done"))
        coordinator.close()
    with Store(tmp_path) as store:
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        coordinator.run(seconds=30, max_jobs=1)
        assert last_job(store, other)["status"] == "pending"
        assert len(store.jobs("done")) == 3
        assert all(row["seed"] == 11 for row in store.jobs("done"))
        assert all(seed == 11 for seed, _ in engine.loads)
        coordinator.close()


def test_champion_only_flag_and_config_validation():
    assert parser().parse_args(["init", "campaign", "--champion-only"]).champion_only
    assert config().champion_only is False
    assert config(champion_only=True).manifest()["champion_only"] is True
    with pytest.raises(ValueError, match="champion_only"):
        config(champion_only=1)


@pytest.mark.parametrize("combo", [False, True])
def test_scheduler_follows_a_new_verified_maximum(tmp_path, combo):
    cfg = config(champion_only=True, combo_search=combo, combo_frontier_width=2,
                 maximum_ticks=1000)
    with Store(tmp_path) as store:
        champion(store, 11)
        checkpoint(store, 11)
        assert last_job(store, next_job(store, cfg))["seed"] == 11
        from test_search_combo_scheduler import state
        summary = state(0, 100, terminal=True)
        summary["canonical_score"] = 200
        key = store.add_checkpoint(0, [0] * 100, summary)
        store.promote(key, {"verified": True, "seed": 0, "tick": 100, "canonical_score": 200})
        checkpoint(store, 0)
        assert last_job(store, next_job(store, cfg))["seed"] == 0
