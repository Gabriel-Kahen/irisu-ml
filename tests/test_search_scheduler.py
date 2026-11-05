from __future__ import annotations

import sys
from pathlib import Path

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
