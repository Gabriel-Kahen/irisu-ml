from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from irisu_search.config import CampaignConfig
from irisu_search.scheduler import next_job
from irisu_search.store import Store


def test_discovery_seeds_receive_development_before_all_attempts_finish(tmp_path, monkeypatch):
    import irisu_search.scheduler as scheduler
    monkeypatch.setattr(scheduler, "choose_lane", lambda *_: "combo")
    cfg = CampaignConfig(engine=("fake",), oracle="fake", control_seed=7,
                         minimum_seed_seconds=30, attempts_per_seed=3)
    with Store(tmp_path) as store:
        for seed in (7, 8, 9):
            store.ensure_seed(seed, control=seed == 7)
            store.update_seed(seed, attempts=1, search_seconds=31)
            store.add_checkpoint(seed, [0], {"seed": seed, "tick": 1, "score": 0,
                                            "terminated": False, "truncated": False})
        selected = set()
        for _ in range(30):
            key = next_job(store, cfg)
            selected.add(next(row["seed"] for row in store.jobs() if row["id"] == key))
        assert selected == {7, 8, 9}
        assert {row["status"] for row in store.seeds()} == {"discovery"}


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("invested,expected", [(119.99, 2048), (120, 8192)])
def test_probe_depth_uses_investment_before_tournament_promotion(
        tmp_path, monkeypatch, control, invested, expected):
    import irisu_search.scheduler as scheduler
    monkeypatch.setattr(scheduler, "choose_lane", lambda *_: "probe")
    cfg = CampaignConfig(engine=("fake",), oracle="fake", control_seed=7,
                         attempts_per_seed=3, finalist_seconds=120)
    with Store(tmp_path) as store:
        store.ensure_seed(7, control=control)
        store.update_seed(7, attempts=1, search_seconds=invested)
        store.add_checkpoint(7, [0], {"seed": 7, "tick": 1, "score": 0,
                                     "terminated": False, "truncated": False})
        key = next_job(store, cfg)
        job = next(row for row in store.jobs() if row["id"] == key)
        assert job["payload"]["horizon"] == expected
        assert job["lane"] == "probe"
        # Depth is an allocation decision, not a claim of tournament maturity.
        assert store.seeds()[0]["status"] == "discovery"
        assert store.seeds()[0]["attempts"] == 1


@pytest.mark.parametrize("bootstrap,maximum,expected", [
    (2048, 1537, 1536),
    (100000, 200000, 100000),
])
def test_deep_probe_still_obeys_remaining_ticks_and_protocol_horizon(
        tmp_path, monkeypatch, bootstrap, maximum, expected):
    import irisu_search.scheduler as scheduler
    monkeypatch.setattr(scheduler, "choose_lane", lambda *_: "probe")
    cfg = CampaignConfig(engine=("fake",), oracle="fake", control_seed=7,
                         bootstrap_ticks=bootstrap, maximum_ticks=maximum)
    with Store(tmp_path) as store:
        store.ensure_seed(7)
        store.update_seed(7, attempts=1, search_seconds=cfg.finalist_seconds)
        store.add_checkpoint(7, [0], {"seed": 7, "tick": 1, "score": 0,
                                     "terminated": False, "truncated": False})
        key = next_job(store, cfg)
        job = next(row for row in store.jobs() if row["id"] == key)
        assert job["payload"]["horizon"] == expected
