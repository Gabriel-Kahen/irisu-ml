from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from irisu_search.config import CampaignConfig
from irisu_search.coordinator import Coordinator
from irisu_search.proposals import select_candidates, select_survivors
from irisu_search.scheduler import next_job
from irisu_search.store import Store


class FakeEngine:
    """Deterministic checkpoint engine that catches use-after-release handles."""

    identity = {"backend": "test-exact", "protocol": 1, "max_checkpoints": 64}

    def __init__(self, terminal_tick=6):
        self.terminal_tick = terminal_tick
        self.handles = {}
        self.next_handle = 1
        self.loads = []
        self.released = []
        self.batches = []
        self.batch_starts = []

    def summary(self, seed, words):
        tick = len(words)
        score = (seed % 17) + sum(1 + (w & 3) for w in words)
        terminal = tick >= self.terminal_tick
        level = 100 if terminal and not any(words) else 99
        return {"seed": seed, "tick": tick, "score": score + (10 if terminal else 0),
                "canonical_score": score if terminal else 0, "canonical_level": level if terminal else 0,
                "canonical_chain": 3 if terminal else 0, "level": level,
                "qualifying": 980, "highest_chain": 3, "gauge": 1000,
                "terminated": terminal, "truncated": False,
                "groups": [{"id": 1, "chain": 3 + tick, "num": tick // 2, "members": [8]}],
                "bodies": [{"id": 8, "kind": "piece", "color": 0, "x": 150,
                            "y": 200, "chain_id": 1, "hits": 0, "size": 30}]}

    def _retain(self, seed, words, suffix):
        if len(self.handles) >= self.identity["max_checkpoints"]:
            raise RuntimeError("fake resident capacity exhausted")
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = (seed, list(words))
        return {"handle": handle, "summary": self.summary(seed, words), "words": list(suffix)}

    def load(self, seed, words):
        assert len(words) <= self.terminal_tick
        self.loads.append((seed, list(words)))
        return self._retain(seed, words, words)

    def batch(self, handle, horizon, candidates):
        assert handle in self.handles, "parent checkpoint was released"
        seed, prefix = self.handles[handle]
        self.batch_starts.append(len(prefix))
        before = list(prefix)
        count = min(horizon, self.terminal_tick - len(prefix))
        results = []
        for candidate in candidates:
            suffix = list(candidate["words"][:count])
            suffix.extend([candidate["mode"]] * (count - len(suffix)))
            results.append(self._retain(seed, prefix + suffix, suffix))
        assert self.handles[handle][1] == before
        self.batches.append((handle, deepcopy(candidates)))
        return {"results": results, "simulated_ticks": count * len(candidates), "wall_seconds": .001}

    def release(self, handles):
        for handle in handles:
            assert handle in self.handles, "double-release of checkpoint"
            del self.handles[handle]
            self.released.append(handle)


class FakeVerifier:
    identity = {"oracle": "fake-independent-native-oracle-v1"}

    def __init__(self, engine):
        self.engine = engine
        self.verified = []

    def verify(self, checkpoint):
        assert checkpoint["summary"] == self.engine.summary(checkpoint["seed"], checkpoint["words"])
        summary = checkpoint["summary"]
        assert summary["terminated"] and not summary["truncated"]
        self.verified.append(checkpoint["id"])
        return {"verified": True, "seed": checkpoint["seed"], "tick": summary["tick"],
                "canonical_score": summary["canonical_score"], "canonical_level": summary["canonical_level"],
                "canonical_chain": summary["canonical_chain"]}


def config(**kwargs):
    values = dict(engine=("fake",), oracle="fake", initial_seeds=2, control_seed=7,
                  attempts_per_seed=1, candidates=4, survivors=2, bootstrap_ticks=2,
                  horizon=2, maximum_ticks=8, job_seconds=.05, engine_timeout=1,
                  minimum_seed_seconds=.001, finalist_seconds=.1)
    values.update(kwargs)
    return CampaignConfig(**values)


class RejectingEngine(FakeEngine):
    """A healthy worker rejects candidates without retaining their invalid states."""

    def __init__(self, rejected=(0,), reject_batches=None, **kwargs):
        super().__init__(**kwargs)
        self.reject_indices = rejected
        self.reject_batches = reject_batches

    def batch(self, handle, horizon, candidates):
        output = super().batch(handle, horizon, candidates)
        reject = self.reject_indices if self.reject_batches is None or len(self.batches) <= self.reject_batches else ()
        accepted, rejected = [], []
        for index, row in enumerate(output["results"]):
            if index in reject:
                self.release([row["handle"]])
                rejected.append({"candidate_index": index, "words": row["words"],
                    "diagnostic": {"physics_status": 1, "status": "capacity", "tick": row["summary"]["tick"]}})
            else:
                accepted.append({**row, "candidate_index": index})
        return {**output, "results": accepted, "rejected": rejected}


def test_mixed_rejections_keep_candidate_lineage_and_do_not_train_on_failed_states(tmp_path, monkeypatch):
    import irisu_search.coordinator as module
    from irisu_search.neural import features
    candidates = [{"mode": 0, "words": [i], "search_seed": 10 + i, "horizon": 2,
                   "operator": operator} for i, operator in enumerate(("recruit", "preserve", "payout", "repair"))]
    monkeypatch.setattr(module, "generate", lambda *_: deepcopy(candidates))
    monkeypatch.setattr(module, "select_candidates", lambda _, rows, *args: rows)
    with Store(tmp_path) as store:
        engine = RejectingEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, config(maximum_ticks=2), FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        before = engine.summary(7, [])
        coordinator.run(seconds=1, max_jobs=1)
        assert store.jobs()[0]["status"] == "done"
        assert store.report()["simulated_ticks"] == 8
        descendants = [store.get_checkpoint(row["id"]) for row in store.checkpoints(active=None) if row["summary"]["tick"]]
        assert {tuple(row["words"]) for row in descendants} == {(1, 0), (2, 0), (3, 0)}
        assert all(row["operator"] == candidates[row["words"][0]]["operator"] for row in descendants)
        proposal_examples = [row for row in store.examples() if row["role"] == "proposal"]
        assert len(proposal_examples) == 3
        assert [row["features"] for row in proposal_examples] == [
            features(before, {**candidate, "budget_seconds": .05}) for candidate in candidates[1:]]
        events = store.db.execute("SELECT payload FROM events WHERE kind='rollout_rejected'").fetchall()
        assert len(events) == 1
        evidence = json.loads(events[0][0])
        assert evidence["job_id"] == key and evidence["candidate_index"] == 0
        assert evidence["candidate"]["operator"] == "recruit"
        assert evidence["diagnostic"]["status"] == "capacity"
        assert store.get_blob(evidence["executed_words_blob"]) == struct.pack("<2I", 0, 0)
        assert store.get_checkpoint(evidence["parent_checkpoint_id"])["words"] == []
        coordinator.close()


def test_all_rejected_job_settles_work_and_campaign_continues(tmp_path):
    with Store(tmp_path) as store:
        engine = RejectingEngine(rejected=range(4), reject_batches=1, terminal_tick=2)
        coordinator = Coordinator(store, engine, config(maximum_ticks=2), FakeVerifier(engine))
        first = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        store.ensure_seed(8)
        second = store.enqueue("bootstrap", 8, payload={"search_seed": 6, "horizon": 2})
        coordinator.run(seconds=1, max_jobs=2)
        jobs = {row["id"]: row for row in store.jobs()}
        assert jobs[first]["status"] == "failed" and jobs[first]["simulated_ticks"] == 8
        assert jobs[second]["status"] == "done" and jobs[second]["simulated_ticks"] == 8
        assert store.report()["simulated_ticks"] == 16
        assert store.get_meta(f"progress:{first}") is None
        assert next(row for row in store.seeds() if row["seed"] == 7)["attempts"] == 1
        assert store.db.execute("SELECT COUNT(*) FROM events WHERE kind='rollout_rejected'").fetchone()[0] == 4
        assert all(row["seed"] == 8 for row in store.examples())
        assert not any(row["summary"]["tick"] for row in store.checkpoints() if row["seed"] == 7)
        coordinator.close()
        assert not engine.handles


def test_exhausted_seed_probes_do_not_block_discovery_of_new_seeds(tmp_path, monkeypatch):
    import irisu_search.scheduler as scheduler
    monkeypatch.setattr(scheduler, "choose_lane", lambda *_: "seed")
    with Store(tmp_path) as store:
        engine = RejectingEngine(rejected=range(4), terminal_tick=2)
        cfg = config(initial_seeds=1, attempts_per_seed=1, maximum_ticks=2)
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        known = {row["seed"] for row in store.seeds()}
        # This checks two logical exhausted probes, not host filesystem latency.
        coordinator.run(seconds=30, max_jobs=2)
        assert len(store.jobs("failed")) == 2
        assert all(row["attempts"] == 1 for row in store.seeds())
        key = scheduler.next_job(store, cfg)
        fresh = next(row for row in store.jobs() if row["id"] == key)
        assert fresh["seed"] not in known
        assert fresh["payload"]["attempt"] == 0
        coordinator.close()


def test_all_rejected_batch_accounts_preceding_successful_batch(tmp_path):
    class RejectSecondBatch(RejectingEngine):
        def batch(self, handle, horizon, candidates):
            self.reject_indices = range(4) if len(self.batches) == 1 else ()
            return super().batch(handle, horizon, candidates)
    with Store(tmp_path) as store:
        engine = RejectSecondBatch(terminal_tick=6)
        coordinator = Coordinator(store, engine, config(maximum_ticks=6, job_seconds=.2), FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        coordinator.run(seconds=1, max_jobs=1)
        assert engine.batch_starts == [0, 2]
        assert store.jobs()[0]["status"] == "failed"
        assert store.jobs()[0]["simulated_ticks"] == store.report()["simulated_ticks"] == 16
        assert store.get_meta(f"progress:{key}") is None
        coordinator.close()


def test_interruption_after_rejected_batch_does_not_reexecute_or_lose_ticks(tmp_path, monkeypatch):
    cfg = config(maximum_ticks=2)
    with Store(tmp_path) as store:
        engine = RejectingEngine(rejected=range(4), terminal_tick=2)
        first = Coordinator(store, engine, cfg, FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        def interrupt_prune():
            raise KeyboardInterrupt("interrupted after rejected trace publication")
        monkeypatch.setattr(first, "_prune", interrupt_prune)
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        assert store.get_meta(f"progress:{key}")["rollout_rejected"]
        assert store.jobs()[0]["status"] == "pending" and store.report()["simulated_ticks"] == 0
        first.close()
        resumed_engine = RejectingEngine(rejected=range(4), terminal_tick=2)
        resumed = Coordinator(store, resumed_engine, cfg, FakeVerifier(resumed_engine))
        resumed.run(seconds=1, max_jobs=1)
        assert resumed_engine.batches == []
        assert store.jobs()[0]["status"] == "failed"
        assert store.report()["simulated_ticks"] == 8
        assert store.get_meta(f"progress:{key}") is None
        assert store.db.execute("SELECT COUNT(*) FROM events WHERE kind='rollout_rejected'").fetchone()[0] == 4
        resumed.close()


def test_multi_seed_campaign_resumes_and_continues_after_terminal_archives(tmp_path):
    cfg = config(lanes={"seed": .97, "combo": .01, "probe": .01, "wildcard": .01})
    with Store(tmp_path) as store:
        first_engine = FakeEngine(terminal_tick=2)
        first = Coordinator(store, first_engine, cfg, FakeVerifier(first_engine))
        manifest = store.get_meta("seed_manifest")
        first.run(seconds=2, max_jobs=5)
        original_ids = {j["id"] for j in store.jobs("done")}
        assert len(original_ids) == 5
        assert len({j["seed"] for j in store.jobs("done")}) >= 3
        assert store.champion()["receipt"]["verified"] is True
        assert store.champion()["score"] < store.get_checkpoint(store.champion()["checkpoint_id"])["summary"]["score"]
        first.close()
        assert not first_engine.handles
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        resumed = Coordinator(store, engine, cfg, FakeVerifier(engine))
        assert store.get_meta("seed_manifest") == manifest
        resumed.run(seconds=2, max_jobs=3)
        assert len(store.jobs("done")) == 8
        assert original_ids <= {j["id"] for j in store.jobs("done")}
        assert len({j["seed"] for j in store.jobs("done")}) >= 4
        assert store.report()["simulated_ticks"] > 0
        resumed.close()


def test_expired_job_is_recovered_without_duplicate_work_accounting(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, config(), FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 1, "horizon": 2})
        stale = store.claim("dead-coordinator")
        store.db.execute("UPDATE jobs SET lease_until=-1 WHERE id=?", (key,))
        coordinator.run(seconds=1, max_jobs=1)
        assert store.jobs()[0]["status"] == "done"
        assert store.jobs()[0]["attempts"] == 2
        assert store.seeds()[0]["jobs"] == 1
        with pytest.raises(ValueError, match="stale"):
            store.finish(key, stale["token"], {})
        coordinator.close()


def test_cold_recipes_reconstruct_and_resident_handles_are_reused(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine()
        coordinator = Coordinator(store, engine, config(), FakeVerifier(engine))
        words = [0, 1]
        key = store.add_checkpoint(7, words, engine.summary(7, words))
        row = store.get_checkpoint(key)
        handle = coordinator._load(row)
        assert coordinator._load(row) == handle
        assert engine.loads == [(7, words)]
        coordinator.close()
        assert not engine.handles
        # Snapshot bytes and process handles are dispensable; the recipe is not.
        store.db.execute("UPDATE checkpoints SET active=0 WHERE id=?", (key,))
        fresh = FakeEngine()
        resumed = Coordinator(store, fresh, config(), FakeVerifier(fresh))
        rebuilt = resumed._load(store.get_checkpoint(key))
        assert rebuilt in fresh.handles
        assert fresh.loads == [(7, words)]
        resumed.close()


def test_inconsistent_recipe_reconstruction_releases_new_handle(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine()
        coordinator = Coordinator(store, engine, config(), FakeVerifier(engine))
        bad = engine.summary(7, [0])
        bad["groups"][0]["num"] += 1
        key = store.add_checkpoint(7, [0], bad)
        with pytest.raises(RuntimeError, match="reconstruction disagrees"):
            coordinator._load(store.get_checkpoint(key))
        assert not engine.handles
        coordinator.close()


def test_game_over_score_can_replace_completed_champion(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, config(), FakeVerifier(engine))
        first = store.add_checkpoint(7, [0, 0], engine.summary(7, [0, 0]))
        assert store.get_checkpoint(first)["summary"]["level"] == 100
        assert coordinator._verify(first)
        higher = store.add_checkpoint(7, [3, 3], engine.summary(7, [3, 3]))
        assert coordinator._verify(higher)
        assert store.champion()["checkpoint_id"] == higher
        assert store.get_checkpoint(higher)["summary"]["level"] == 99
        coordinator.close()


def test_candidate_selection_retains_random_exploration_against_model(tmp_path):
    class Guide:
        def predict(self, summary, candidate):
            return {"priority": -candidate["index"]}
    candidates = [{"index": n, "words": [], "mode": 1} for n in range(100)]
    selected = select_candidates({}, candidates, 8, Guide(), exploration_fraction=.5, seed=7)
    assert selected[0] is candidates[0]
    assert len(selected) == len({id(x) for x in selected}) == 8
    assert sum(row["index"] >= 8 for row in selected) >= 3


def test_scheduler_creates_new_seed_when_existing_states_are_terminal(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        cfg = config()
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        initial = {s["seed"] for s in store.seeds()}
        for seed in initial:
            store.update_seed(seed, attempts=cfg.attempts_per_seed, search_seconds=1)
            store.add_checkpoint(seed, [0, 0], engine.summary(seed, [0, 0]))
        key = next_job(store, cfg)
        job = next(j for j in store.jobs() if j["id"] == key)
        assert job["kind"] == "bootstrap"
        assert job["seed"] not in initial
        coordinator.close()


def test_engine_capacity_rejected_before_campaign_identity_is_written(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine()
        engine.identity = dict(engine.identity, max_checkpoints=4)
        with pytest.raises(ValueError, match="parent plus every candidate"):
            Coordinator(store, engine, config(candidates=4), FakeVerifier(engine))
        assert store.get_meta("campaign_config") is None
        assert store.seeds() == []


def test_tight_capacity_retains_chosen_continuation_without_prefix_replays(tmp_path):
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=6)
        engine.identity = dict(engine.identity, max_checkpoints=5)
        cfg = config(candidates=4, survivors=2, maximum_ticks=6, job_seconds=.2)
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        store.enqueue("bootstrap", 7, payload={"search_seed": 1, "horizon": 2})
        coordinator.run(seconds=1, max_jobs=1)
        assert engine.batch_starts == [0, 2, 4]
        assert engine.loads == [(7, [])]
        assert len(engine.handles) == 1
        coordinator.close()
        assert not engine.handles


def test_interrupted_job_resumes_last_completed_batch_and_counts_ticks_once(tmp_path):
    class InterruptedEngine(FakeEngine):
        def batch(self, handle, horizon, candidates):
            if len(self.batches) == 1:
                raise KeyboardInterrupt("simulated process interruption")
            return super().batch(handle, horizon, candidates)
    cfg = config(candidates=4, maximum_ticks=6, job_seconds=.2)
    with Store(tmp_path) as store:
        interrupted = InterruptedEngine(terminal_tick=6)
        first = Coordinator(store, interrupted, cfg, FakeVerifier(interrupted))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        assert interrupted.batch_starts == [0]
        saved = store.get_meta(f"progress:{key}")
        assert store.get_checkpoint(saved["checkpoint_id"])["summary"]["tick"] == 2
        assert saved["simulated_ticks"] == 8
        assert store.jobs()[0]["status"] == "pending"
        assert next(s for s in store.seeds() if s["seed"] == 7)["attempts"] == 0
        first.close()
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=6)
        resumed = Coordinator(store, engine, cfg, FakeVerifier(engine))
        resumed.run(seconds=1, max_jobs=1)
        assert engine.batch_starts == [2, 4]
        assert store.jobs()[0]["status"] == "done"
        assert store.jobs()[0]["attempts"] == 2
        assert store.jobs()[0]["simulated_ticks"] == 24
        assert store.report()["simulated_ticks"] == 24
        assert next(s for s in store.seeds() if s["seed"] == 7)["attempts"] == 1
        assert store.get_meta(f"progress:{key}") is None
        assert store.champion()["receipt"]["verified"] is True
        resumed.close()


def test_terminal_batch_completed_before_settlement_is_not_reexecuted(tmp_path, monkeypatch):
    cfg = config(candidates=4, maximum_ticks=2, job_seconds=.2)
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        first = Coordinator(store, engine, cfg, FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        original_finish = store.finish
        def interrupted_finish(*args, **kwargs):
            raise KeyboardInterrupt("interrupted after durable terminal batch")
        monkeypatch.setattr(store, "finish", interrupted_finish)
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        monkeypatch.setattr(store, "finish", original_finish)
        progress = store.get_meta(f"progress:{key}")
        assert progress["completed"] and progress["observed_terminal"]
        assert store.get_checkpoint(progress["checkpoint_id"])["summary"]["terminated"]
        assert engine.batch_starts == [0]
        assert next(s for s in store.seeds() if s["seed"] == 7)["attempts"] == 0
        first.close()
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        resumed = Coordinator(store, engine, cfg, FakeVerifier(engine))
        resumed.run(seconds=1, max_jobs=1)
        assert engine.batch_starts == []
        assert store.jobs()[0]["simulated_ticks"] == 8
        assert store.jobs()[0]["result"]["batches"] == 1
        assert next(s for s in store.seeds() if s["seed"] == 7)["attempts"] == 1
        assert store.get_meta(f"progress:{key}") is None
        resumed.close()


def test_permanent_failure_accounts_previously_completed_batches(tmp_path):
    class FailedEngine(FakeEngine):
        def batch(self, handle, horizon, candidates):
            if len(self.batches) == 1:
                raise RuntimeError("second batch failed")
            return super().batch(handle, horizon, candidates)
    cfg = config(maximum_ticks=6, job_seconds=.2)
    with Store(tmp_path) as store:
        engine = FailedEngine(terminal_tick=6)
        coordinator = Coordinator(store, engine, cfg, FakeVerifier(engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        with pytest.raises(RuntimeError, match="second batch"):
            coordinator.run(seconds=1, max_jobs=1)
        assert store.get_meta(f"progress:{key}")["simulated_ticks"] == 8
        assert store.jobs()[0]["status"] == "failed"
        assert store.jobs()[0]["simulated_ticks"] == 8
        assert store.report()["simulated_ticks"] == 8
        assert next(s for s in store.seeds() if s["seed"] == 7)["attempts"] == 0
        coordinator.close()
    with Store(tmp_path) as store:
        assert store.report()["simulated_ticks"] == 8


def test_permanent_failure_after_retry_accounts_cumulative_progress_once(tmp_path):
    class FailedEngine(FakeEngine):
        interruption = KeyboardInterrupt

        def batch(self, handle, horizon, candidates):
            if len(self.batches) == 1:
                raise self.interruption("second batch failed")
            return super().batch(handle, horizon, candidates)
    cfg = config(maximum_ticks=6, job_seconds=.2)
    with Store(tmp_path) as store:
        first_engine = FailedEngine(terminal_tick=6)
        first = Coordinator(store, first_engine, cfg, FakeVerifier(first_engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        assert store.report()["simulated_ticks"] == 0
        first.close()
        resumed_engine = FailedEngine(terminal_tick=6)
        resumed_engine.interruption = RuntimeError
        resumed = Coordinator(store, resumed_engine, cfg, FakeVerifier(resumed_engine))
        with pytest.raises(RuntimeError, match="second batch"):
            resumed.run(seconds=1, max_jobs=1)
        assert first_engine.batch_starts == [0]
        assert resumed_engine.batch_starts == [2]
        assert store.get_meta(f"progress:{key}")["simulated_ticks"] == 16
        assert store.jobs()[0]["simulated_ticks"] == 16
        assert store.report()["simulated_ticks"] == 16
        assert store.jobs()[0]["attempts"] == 2
        resumed.close()


def test_shadow_training_samples_history_without_materializing_all_examples(tmp_path, monkeypatch):
    from irisu_search.neural import features
    with Store(tmp_path) as store:
        engine = FakeEngine()
        coordinator = Coordinator(store, engine, config(max_training_examples=6, training_epochs=2), FakeVerifier(engine))
        for seed in (7, 8, 9):
            for index in range(20):
                store.add_example({"seed": seed, "features": features(engine.summary(seed, [])),
                                   "index": index, "targets": {"extension": index % 2}, "censored": True,
                                   "planner_version": "irisu-search-v1", "budget_seconds": .2})
        def unbounded_read():
            raise AssertionError("automatic training must not load all historical examples")
        monkeypatch.setattr(store, "examples", unbounded_read)
        result = coordinator.train_shadow()
        assert result is not None
        report = result["report"]
        assert report["train_examples"] + report["validation_examples"] == 6
        assert set(report["train_seeds"]).isdisjoint(report["validation_seeds"])
        coordinator.close()


def test_scheduling_does_not_materialize_historical_job_results(tmp_path, monkeypatch):
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        coordinator = Coordinator(store, engine, config(), FakeVerifier(engine))
        def unbounded_read(*args, **kwargs):
            raise AssertionError("scheduling must use aggregates instead of complete result histories")
        monkeypatch.setattr(store, "jobs", unbounded_read)
        coordinator.run(seconds=1, max_jobs=2)
        assert store.completed_jobs() == 2
        coordinator.close()


def test_retry_keeps_each_collection_model_identity_and_training_provenance(tmp_path):
    from irisu_search.neural import features, train_guide
    from irisu_search.verification import sha256_file
    class InterruptedEngine(FakeEngine):
        def batch(self, handle, horizon, candidates):
            if len(self.batches) == 1:
                raise KeyboardInterrupt("switch model after durable batch")
            return super().batch(handle, horizon, candidates)
    cfg = config(maximum_ticks=6, job_seconds=.2)
    with Store(tmp_path / "campaign") as store:
        initial_engine = FakeEngine(terminal_tick=6)
        initial = Coordinator(store, initial_engine, cfg, FakeVerifier(initial_engine))
        identity = initial.identity
        initial.close()
        paths = []
        for index in range(2):
            model, _ = train_guide([
                {"seed": seed, "features": features(initial_engine.summary(seed, [])),
                 "targets": {"extension": seed % 2}, "censored": True,
                 "planner_version": "irisu-search-v1", "budget_seconds": .2}
                for seed in (7, 8)
            ], epochs=1, hidden=3, seed=index)
            path = tmp_path / f"guide-{index}.json"
            model.save(path, {"engine_identity": identity})
            paths.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        store.set_meta("active_model", paths[0])
        first_engine = InterruptedEngine(terminal_tick=6)
        first = Coordinator(store, first_engine, cfg, FakeVerifier(first_engine))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        first_token = store.jobs()[0]["token"]
        assert store.get_attempt_metadata(key, first_token)["model_identity"] == paths[0]
        first.close()
        store.set_meta("active_model", paths[1])
        engine = FakeEngine(terminal_tick=6)
        resumed = Coordinator(store, engine, cfg, FakeVerifier(engine))
        resumed.run(seconds=1, max_jobs=1)
        job = store.jobs()[0]
        assert job["token"] != first_token
        assert store.get_attempt_metadata(key, first_token)["model_identity"] == paths[0]
        assert store.get_attempt_metadata(key, job["token"])["model_identity"] == paths[1]
        assert job["result"]["model_identity"] == paths[1]
        assert {model["sha256"] for model in job["result"]["collection_models"]} == {row["sha256"] for row in paths}
        assert {row["model_identity"]["sha256"] for row in store.examples()} == {row["sha256"] for row in paths}
        # Add a second seed so a shadow model can exercise its whole-seed split.
        store.add_example({"seed": 8, "features": features(engine.summary(8, [])),
                           "targets": {"extension": 0}, "censored": True,
                           "planner_version": "irisu-search-v1", "budget_seconds": .2,
                           "model_identity": paths[0]})
        report = resumed.train_shadow()["report"]
        assert {model["sha256"] for model in report["collection_models"]} == {row["sha256"] for row in paths}
        resumed.close()


def test_prior_attempt_terminal_does_not_uncensor_resumed_partial_allocation(tmp_path):
    class MixedEngine(FakeEngine):
        def summary(self, seed, words):
            result = super().summary(seed, words)
            if len(words) >= 2 and not any(words):
                result.update(terminated=True, canonical_score=result["score"],
                              canonical_level=result["level"], canonical_chain=3)
            return result
    class InterruptedEngine(MixedEngine):
        def batch(self, handle, horizon, candidates):
            if len(self.batches) == 1:
                raise KeyboardInterrupt("prior attempt observed a terminal alternative")
            return super().batch(handle, horizon, candidates)
    cfg = config(maximum_ticks=4, job_seconds=.2)
    with Store(tmp_path) as store:
        interrupted = InterruptedEngine(terminal_tick=6)
        first = Coordinator(store, interrupted, cfg, FakeVerifier(interrupted))
        key = store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        with pytest.raises(KeyboardInterrupt):
            first.run(seconds=1, max_jobs=1)
        assert store.get_meta(f"progress:{key}")["observed_terminal"]
        first.close()
        engine = MixedEngine(terminal_tick=6)
        resumed = Coordinator(store, engine, cfg, FakeVerifier(engine))
        resumed.run(seconds=1, max_jobs=1)
        allocation = [row for row in store.examples() if row.get("role") == "compute_allocation"]
        assert len(allocation) == 1
        assert allocation[0]["censored"] is True
        assert allocation[0]["targets"]["compute_gain"] == 0
        resumed.close()


@pytest.mark.parametrize("candidate_count", [4, 32])
def test_terminal_ranking_decodes_only_the_verified_recipe(tmp_path, monkeypatch, candidate_count):
    import irisu_search.coordinator as module
    candidates = [{"mode": 0, "words": [i], "search_seed": i, "horizon": 2,
                   "operator": "payout"} for i in range(1, candidate_count + 1)]
    monkeypatch.setattr(module, "generate", lambda *_: deepcopy(candidates))
    with Store(tmp_path) as store:
        engine = FakeEngine(terminal_tick=2)
        verifier = FakeVerifier(engine)
        coordinator = Coordinator(store, engine, config(candidates=candidate_count), verifier)
        store.enqueue("bootstrap", 7, payload={"search_seed": 5, "horizon": 2})
        reads = []
        original_get = store.get_checkpoint

        def get_checkpoint(key):
            reads.append(key)
            return original_get(key)

        monkeypatch.setattr(store, "get_checkpoint", get_checkpoint)
        coordinator.run(seconds=1, max_jobs=1)
        terminals = [row for row in store.checkpoints(active=None) if row["summary"]["terminated"]]
        assert len(terminals) == candidate_count
        best = max(terminals, key=lambda row: row["summary"]["canonical_score"])
        assert verifier.verified == [best["id"]]
        assert store.champion()["checkpoint_id"] == best["id"]
        # One reset parent load, then the winner for verification and promotion.
        assert len(reads) == 3
        assert reads.count(best["id"]) == 2
        coordinator.close()
