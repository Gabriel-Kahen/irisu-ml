from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from irisu_search.store import Store, archive_cell


def summary(seed=7, tick=2, score=100, terminal=False, chain=10):
    return {"seed": seed, "tick": tick, "score": score + (50 if terminal else 0),
            "canonical_score": score if terminal else 0, "gauge": 1000, "level": 99,
            "qualifying": 980, "terminated": terminal, "truncated": False,
            "groups": [{"id": 1, "chain": chain, "num": 5, "members": [8]}],
            "bodies": [{"id": 8, "x": 150, "y": 200}]}


def initialized(tmp_path):
    store = Store(tmp_path)
    store.init_campaign({"seeds": [7]}, {"backend": "native", "sha256": "engine"})
    return store


def test_campaign_identity_frozen_on_resume(tmp_path):
    with initialized(tmp_path):
        pass
    with Store(tmp_path) as store:
        store.init_campaign({"seeds": [7]}, {"backend": "native", "sha256": "engine"})
        with pytest.raises(ValueError, match="immutable"):
            store.init_campaign({"seeds": [8]}, {"backend": "native", "sha256": "engine"})
        assert store.get_meta("campaign_config") == {"seeds": [7]}


def test_checkpoint_recipe_idempotence_and_integrity(tmp_path):
    with initialized(tmp_path) as store:
        first = store.add_checkpoint(7, [0, 0], summary())
        assert first == store.add_checkpoint(7, [0, 0], summary())
        assert store.get_checkpoint(first)["words"] == [0, 0]
        assert len(store.checkpoints()) == 1
        with pytest.raises(ValueError, match="different state"):
            store.add_checkpoint(7, [0, 0], summary(score=101))
        with pytest.raises(ValueError, match="extend its parent"):
            store.add_checkpoint(7, [1, 0, 0], summary(tick=3), parent_id=first)
        second = store.add_checkpoint(7, [0, 0, 0], summary(tick=3), parent_id=first)
        assert store.get_checkpoint(second)["parent_id"] == first
        digest = store.get_checkpoint(first)["trace_blob"]
        store._blob_path(digest).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="checksum"):
            store.get_checkpoint(first)


def test_concurrent_claims_never_claim_same_job(tmp_path):
    with initialized(tmp_path) as store:
        for n in range(12):
            store.enqueue("bootstrap", n)
    def claim(n):
        with Store(tmp_path) as store:
            return store.claim(str(n))["id"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        claimed = list(pool.map(claim, range(12)))
    assert len(set(claimed)) == 12


def test_expired_attempt_is_fenced_and_settlements_exactly_once(tmp_path, monkeypatch):
    with initialized(tmp_path) as store:
        clock = [100.0]
        monkeypatch.setattr(time, "time", lambda: clock[0])
        key = store.enqueue("bootstrap", 7)
        first = store.claim("old", lease_seconds=10)
        clock[0] = 111
        second = store.claim("new", lease_seconds=10)
        assert second["id"] == first["id"] == key
        assert second["token"] != first["token"]
        with pytest.raises(ValueError, match="stale"):
            store.finish(key, first["token"], {}, cost_seconds=9)
        store.finish(key, second["token"], {"value": 1}, cost_seconds=3, simulated_ticks=10)
        store.finish(key, second["token"], {"value": 1}, cost_seconds=3, simulated_ticks=10)
        assert store.seeds()[0]["search_seconds"] == 3
        assert store.seeds()[0]["jobs"] == 1
        assert store.seeds()[0]["attempts"] == 1
        assert store.report()["abandoned_attempts"] == 1
        with pytest.raises(ValueError, match="settled differently"):
            store.finish(key, second["token"], {}, cost_seconds=3)


def test_retry_accounts_each_attempt_once_and_dedupes_jobs(tmp_path):
    with initialized(tmp_path) as store:
        key = store.enqueue("bootstrap", 7, payload={"n": 1}, dedupe_key="seed7")
        assert key == store.enqueue("bootstrap", 7, payload={"n": 1}, dedupe_key="seed7")
        with pytest.raises(ValueError, match="different work"):
            store.enqueue("bootstrap", 7, payload={"n": 2}, dedupe_key="seed7")
        first = store.claim("worker")
        store.fail(key, first["token"], "restart", retry=True, cost_seconds=2)
        store.fail(key, first["token"], "restart", retry=True, cost_seconds=2)
        second = store.claim("worker")
        store.finish(key, second["token"], {}, cost_seconds=3)
        assert store.seeds()[0]["search_seconds"] == 5
        # A resumed logical bootstrap still explores one search seed.
        assert store.seeds()[0]["attempts"] == 1


def test_promotion_uses_canonical_score_allows_game_over_and_survives_prune(tmp_path):
    with initialized(tmp_path) as store:
        key = store.add_checkpoint(7, [0, 0], summary(terminal=True))
        receipt = {"verified": True, "seed": 7, "tick": 2, "canonical_score": 100}
        with pytest.raises(ValueError, match="canonical"):
            store.promote(key, dict(receipt, canonical_score=150))
        with pytest.raises(ValueError, match="canonical"):
            store.promote(key, dict(receipt, verified=False))
        assert store.promote(key, receipt)
        assert not store.promote(key, receipt)
        assert store.champion()["score"] == 100
        # This is level 99 game-over, still a legitimate score candidate.
        assert store.get_checkpoint(key)["summary"]["level"] == 99
        result = store.prune_archive(max_trace_bytes=1)
        assert result["protected_over_budget"]
        assert store.get_checkpoint(key)["active"]
        assert len(store.db.execute("SELECT * FROM verifications").fetchall()) == 1


def test_archive_retains_distinct_states_and_cold_recipes(tmp_path):
    with initialized(tmp_path) as store:
        keys = [store.add_checkpoint(7, [n, 0], summary(score=100+n)) for n in range(5)]
        assert len({archive_cell(store.get_checkpoint(k)["summary"]) for k in keys}) == 1
        result = store.prune_archive(per_cell=2, max_checkpoints=3)
        assert result["active"] == 2
        assert len(store.checkpoints(active=False)) == 3
        assert len(store.checkpoints(active=None)) == 5
        assert all(len(store.get_checkpoint(k)["words"]) == 2 for k in keys)


def test_examples_content_addressed_and_nonfinite_rejected(tmp_path):
    with initialized(tmp_path) as store:
        assert store.add_example({"seed": 7, "value": 2}) == store.add_example({"value": 2, "seed": 7})
        assert len(store.examples()) == 1
        with pytest.raises(ValueError):
            store.add_example({"value": float("nan")})
        with pytest.raises(ValueError):
            store.enqueue("test", 7, priority=float("inf"))


def test_seed_and_trace_boundaries(tmp_path):
    with initialized(tmp_path) as store:
        store.ensure_seed(0)
        store.ensure_seed(0xFFFFFFFF)
        with pytest.raises(ValueError):
            store.ensure_seed(True)
        with pytest.raises(ValueError):
            store.ensure_seed(1 << 32)
        with pytest.raises(ValueError):
            store.add_checkpoint(7, [1 << 21], summary(tick=1))
        with pytest.raises(ValueError):
            store.add_checkpoint(7, [0], summary(tick=2))


def test_permanent_failed_attempt_settles_known_ticks_exactly_once(tmp_path):
    with initialized(tmp_path) as store:
        key = store.enqueue("bootstrap", 7)
        job = store.claim("worker")
        store.fail(key, job["token"], "failed after completed batch", cost_seconds=2, simulated_ticks=8)
        store.fail(key, job["token"], "failed after completed batch", cost_seconds=2, simulated_ticks=8)
        assert store.report()["simulated_ticks"] == 8
        assert store.seeds()[0]["search_seconds"] == 2
        assert store.seeds()[0]["attempts"] == 0


def test_bounded_sample_stratifies_seeds_and_donates_unused_quota(tmp_path):
    with initialized(tmp_path) as store:
        for seed, count in ((7, 1), (8, 100), (9, 100)):
            for index in range(count):
                store.add_example({"seed": seed, "index": index, "targets": {"extension": index % 2}, "censored": True})
        sample = store.sample_examples(12, seed=4, supervised_only=True)
        assert len(sample) == 12
        sizes = {seed: sum(row["seed"] == seed for row in sample) for seed in (7, 8, 9)}
        assert sizes[7] == 1
        assert sorted((sizes[8], sizes[9])) == [5, 6]
        assert sample == store.sample_examples(12, seed=4, supervised_only=True)


def test_bounded_sample_is_insertion_order_independent_with_many_seeds(tmp_path):
    examples = [{"seed": seed, "index": index, "targets": {"extension": index % 2}}
                for seed in range(40) for index in range(3)]
    samples = []
    for name, rows in (("forward", examples), ("reverse", examples[::-1])):
        with Store(tmp_path / name) as store:
            for row in rows:
                store.add_example(row)
            sample = store.sample_examples(7, seed=17, supervised_only=True)
            assert len(sample) == len({row["seed"] for row in sample}) == 7
            samples.append(sample)
    assert samples[0] == samples[1]


def test_bounded_sample_excludes_unobserved_and_malformed_seed_rows(tmp_path):
    with initialized(tmp_path) as store:
        store.add_example({"seed": 7, "targets": {"gain": 100}, "censored": True})
        store.add_example({"seed": 8, "targets": {"extension": None}, "censored": False})
        store.add_example({"seed": [], "targets": {"extension": 1}})
        store.add_example({"seed": 9, "targets": []})
        observed = {"seed": 0, "targets": {"extension": 0}, "censored": True}
        store.add_example(observed)
        assert store.sample_examples(10, supervised_only=True) == [observed]


def test_attempt_metadata_is_immutable_fenced_and_retained_across_retries(tmp_path):
    with initialized(tmp_path) as store:
        key = store.enqueue("bootstrap", 7)
        first = store.claim("first")
        model_a = {"path": "guide-a.json", "sha256": "a" * 64}
        model_b = {"path": "guide-b.json", "sha256": "b" * 64}
        store.set_attempt_metadata(key, first["token"], {"model_identity": model_a})
        store.set_attempt_metadata(key, first["token"], {"model_identity": model_a})
        with pytest.raises(ValueError, match="immutable"):
            store.set_attempt_metadata(key, first["token"], {"model_identity": model_b})
        store.fail(key, first["token"], "retry", retry=True)
        second = store.claim("second")
        with pytest.raises(ValueError, match="stale"):
            store.set_attempt_metadata(key, first["token"], {"model_identity": model_b})
        store.set_attempt_metadata(key, second["token"], {"model_identity": model_b})
        assert store.get_attempt_metadata(key, first["token"])["model_identity"] == model_a
        assert store.get_attempt_metadata(key, second["token"])["model_identity"] == model_b
        assert {model["sha256"] for model in store.job_model_identities(key)} == {"a" * 64, "b" * 64}
