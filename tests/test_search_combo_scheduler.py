from __future__ import annotations

from irisu_search.config import CampaignConfig
from irisu_search.scheduler import next_job
from irisu_search.store import Store


def state(seed, tick, chain=5, terminal=False):
    return {"seed": seed, "tick": tick, "score": 100, "canonical_score": 100 if terminal else 0,
            "terminated": terminal, "truncated": False, "gauge": 1000,
            "groups": [{"id": 1, "chain": chain, "num": 3, "members": [8]}],
            "bodies": [{"id": 8, "chain_id": 1, "color": 0, "hits": 0,
                        "x": 150, "y": 200, "lifetime": 1000}]}


def config(**changes):
    return CampaignConfig(engine=("fake",), oracle="fake", combo_search=True,
                          survivors=4, **changes)


def checkpoint(store, seed, tick=2, chain=5, word=0, operator="explore", terminal=False):
    return store.add_checkpoint(seed, [word] * tick, state(seed, tick, chain, terminal), operator=operator)


def champion(store, seed):
    key = checkpoint(store, seed, tick=100, terminal=True)
    store.promote(key, {"verified": True, "seed": seed, "tick": 100, "canonical_score": 100})
    store.set_meta(f"baseline:{seed}", key)
    return key


def last_job(store, key):
    return next(job for job in store.jobs() if job["id"] == key)


def test_record_lane_uses_champion_not_hardcoded_control_and_covers_roots(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "combo")
    cfg = config(control_seed=7, combo_terminal_horizon=1000, maximum_ticks=10000)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        reference = champion(store, 11)
        roots = [checkpoint(store, 11, tick=tick, operator="combo-root") for tick in (5, 10, 15)]
        store.set_meta("combo_roots:11", roots)
        checkpoint(store, 7, chain=100)
        selected = [last_job(store, next_job(store, cfg)) for _ in roots]
        assert {job["checkpoint_id"] for job in selected} == set(roots)
        assert all(job["seed"] == 11 and job["kind"] == "combo_terminal" for job in selected)
        assert all(job["payload"]["reference_id"] == reference for job in selected)
        assert all(job["payload"]["horizon"] == 1000 for job in selected)


def test_record_frontier_siblings_receive_expansion_after_root_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "combo")
    cfg = config(control_seed=7)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        root = checkpoint(store, 7, operator="combo-root")
        store.set_meta("combo_roots:7", [root])
        siblings = [checkpoint(store, 7, tick=3, chain=5+n, word=n) for n in range(1, 5)]
        store.update_combo_frontier(7, siblings, root, 4)
        assert last_job(store, next_job(store, cfg))["checkpoint_id"] == root
        expanded = set()
        for _ in range(8):
            job = last_job(store, next_job(store, cfg))
            if job["checkpoint_id"] in siblings:
                expanded.add(job["checkpoint_id"])
                store.mark_combo_expanded(job["checkpoint_id"], job["id"])
        assert expanded == set(siblings)


def test_bootstrap_finishes_small_cohort_before_admitting_next(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "seed")
    cfg = config(seed_cohort_size=2, attempts_per_seed=3)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        store.set_meta("seed_manifest", [101, 102, 103, 104])
        for seed in (101, 102, 103, 104):
            checkpoint(store, seed)
        selected = []
        for _ in range(7):
            job = last_job(store, next_job(store, cfg))
            selected.append(job["seed"])
            store.update_seed(job["seed"], attempts=job["payload"]["attempt"] + 1)
        assert selected[:6] == [101, 102, 101, 102, 101, 102]
        assert selected[6] == 103


def test_challenger_lane_is_bounded_and_pays_out_developed_combo(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "probe")
    cfg = config(control_seed=7, promoted_seeds=2)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        for seed in range(20, 40):
            checkpoint(store, seed, chain=seed)
        jobs = [last_job(store, next_job(store, cfg)) for _ in range(8)]
        assert {job["seed"] for job in jobs} == {38, 39}
        assert len([job for job in jobs if job["kind"] == "combo_terminal"]) == 2
        assert all(job["payload"]["reference_id"] is None for job in jobs if job["kind"] == "combo_terminal")


def test_terminal_schedule_clamps_to_remaining_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "combo")
    cfg = config(control_seed=7, maximum_ticks=100, combo_terminal_horizon=32768)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        root = checkpoint(store, 7, tick=91, operator="combo-root")
        store.set_meta("combo_roots:7", [root])
        job = last_job(store, next_job(store, cfg))
        assert job["payload"]["horizon"] == 9


def test_record_reference_follows_new_champion_instead_of_stale_import(tmp_path, monkeypatch):
    monkeypatch.setattr("irisu_search.scheduler.choose_lane", lambda *_: "combo")
    cfg = config(control_seed=7)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        reference = champion(store, 11)
        stale = checkpoint(store, 11, tick=99, word=1, terminal=True)
        store.set_meta("baseline:11", stale)
        root = checkpoint(store, 11, operator="combo-root")
        store.set_meta("combo_roots:11", [root])
        job = last_job(store, next_job(store, cfg))
        assert job["seed"] == 11
        assert job["payload"]["reference_id"] == reference


def test_plan_and_live_frontier_followup_do_not_wait_for_full_root_sweep(tmp_path, monkeypatch):
    import struct
    monkeypatch.setattr('irisu_search.scheduler.choose_lane', lambda *_: 'combo')
    cfg = config(control_seed=7)
    with Store(tmp_path) as store:
        store.init_campaign(cfg.manifest(), {})
        champion(store, 7)
        roots = [checkpoint(store, 7, tick=tick, operator='combo-root') for tick in (2, 4, 6, 8)]
        store.set_meta('combo_roots:7', roots)
        first = last_job(store, next_job(store, cfg))
        assert first['checkpoint_id'] == roots[0]
        plan = store.put_blob(struct.pack('<4I', 1, 0, 2, 0))
        store.record_combo_plans(roots[0], [{'plan_blob': plan, 'score': 90}], 4)
        second = last_job(store, next_job(store, cfg))
        assert second['checkpoint_id'] == roots[0] and second['payload']['plan_blob'] == plan
        assert last_job(store, next_job(store, cfg))['checkpoint_id'] == roots[1]
        sibling = checkpoint(store, 7, tick=3, word=3)
        store.update_combo_frontier(7, [sibling], roots[0], 4)
        scheduled = [last_job(store, next_job(store, cfg)) for _ in range(5)]
        assert any(job['checkpoint_id'] == sibling for job in scheduled)
        assert {job['checkpoint_id'] for job in scheduled} >= set(roots[2:])
