"""Persistent search campaigns; no per-tick Python calls or learned record scores."""
from __future__ import annotations

import fcntl
import json
import math
import os
import random
import struct
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from . import PLANNER_VERSION
from .neural import NeuralGuide, features, train_guide, validate_guide
from .proposals import (
    generate,
    live_groups,
    select_candidates,
    select_survivors,
    target_group,
)
from .scheduler import next_job
from .verification import NativeVerifier, read_replay, replay_bytes, sha256_file


class RolloutBatchRejected(RuntimeError):
    """Every candidate was rejected while the resident worker remained healthy."""


@contextmanager
def campaign_lock(root):
    with (Path(root) / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another coordinator owns this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def run_accounting(store):
    started = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        store.set_meta("run_wall_seconds", store.get_meta("run_wall_seconds", 0.) + elapsed)


def planner_identity(engine_identity, verifier_identity):
    source = Path(__file__).parent
    modules = ("__init__", "config", "coordinator", "engine", "neural", "proposals", "scheduler", "store", "verification")
    return {"planner": PLANNER_VERSION, "engine": engine_identity, "verifier": verifier_identity,
            "planner_sources": {name: sha256_file(source / f"{name}.py") for name in modules}}


def collection_models(examples):
    """Collection policies are data provenance, not rollout-engine identities."""
    identities = {}
    for row in examples:
        model = row.get("model_identity")
        identities[json.dumps(model, sort_keys=True)] = model
    return [identities[key] for key in sorted(identities)]


class Coordinator:
    def __init__(self, store, engine, config, verifier=None):
        capacity = engine.identity.get("max_checkpoints", 64)
        if type(capacity) is not int or capacity < config.candidates + 1:
            raise ValueError("engine checkpoint capacity must fit the parent plus every candidate")
        self.store, self.engine, self.config = store, engine, config
        self.verifier = verifier or NativeVerifier(config.oracle, config.oracle_library, config.engine_timeout)
        self.identity = planner_identity(engine.identity, self.verifier.identity)
        self.cache = OrderedDict()
        self.owner = f"{os.getpid()}:{uuid.uuid4().hex}"
        self.guide = None
        self.model_identity = None
        bound_model = None
        active = store.get_meta("active_model", {})
        path = active.get("path") or config.model
        if path:
            model_hash = sha256_file(path)
            if active.get("sha256") and model_hash != active["sha256"]:
                raise ValueError("active neural guide changed since activation")
            self.guide = NeuralGuide.load(path)
            if sha256_file(path) != model_hash:
                raise ValueError("neural guide changed while it was loaded")
            validate_guide(self.guide, self.identity, PLANNER_VERSION)
            self.model_identity = {"path": str(Path(path).resolve()), "sha256": model_hash}
            if not active:
                bound_model = self.model_identity
        store.init_campaign(config.manifest(), self.identity)
        if bound_model:
            store.set_meta("active_model", bound_model)
        self.seed_manifest()

    def seed_manifest(self):
        if self.store.get_meta("seed_manifest") is not None:
            return
        rng = random.Random(self.config.seed_generator)
        seeds = []
        seen = {self.config.control_seed}
        while len(seeds) < self.config.initial_seeds:
            seed = rng.getrandbits(32)
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
        self.store.ensure_seed(self.config.control_seed, control=True)
        for seed in seeds:
            self.store.ensure_seed(seed)
        self.store.set_meta("seed_manifest", seeds)

    def _trim(self, reserve=0):
        maximum = max(1, int(self.engine.identity.get("max_checkpoints", 64)) - reserve)
        while len(self.cache) > maximum:
            _, handle = self.cache.popitem(last=False)
            self.engine.release([handle])

    def _remember(self, checkpoint_id, handle):
        old = self.cache.pop(checkpoint_id, None)
        if old is not None and old != handle:
            self.engine.release([old])
        self.cache[checkpoint_id] = handle
        self._trim(self.config.candidates + 1)

    def _load(self, checkpoint):
        key = checkpoint["id"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self._trim(self.config.candidates + 2)
        row = self.engine.load(checkpoint["seed"], checkpoint["words"])
        # Reconstructing recipes may not silently change observable group state.
        if row["summary"] != checkpoint["summary"]:
            self.engine.release([row["handle"]])
            raise RuntimeError("checkpoint reconstruction disagrees with stored summary")
        self._remember(key, row["handle"])
        return row["handle"]

    def _publish(self, parent, row, operator, cost, search_seed):
        suffix = row["words"]
        words = parent["words"] + suffix
        summary = row["summary"]
        if summary.get("seed") != parent["seed"] or summary["tick"] != len(words):
            raise RuntimeError("engine returned inconsistent input lineage")
        return self.store.add_checkpoint(parent["seed"], words, summary, parent["id"],
                                         operator=operator, cost_seconds=cost, search_seed=search_seed)

    def _verify(self, checkpoint_id, force=False):
        checkpoint = self.store.get_checkpoint(checkpoint_id)
        s = checkpoint["summary"]
        if not s.get("terminated") or s.get("truncated"):
            return False
        seed_best = next((row["best_score"] for row in self.store.seeds() if row["seed"] == checkpoint["seed"]), -1)
        if not force and s["canonical_score"] <= seed_best:
            # This remains an unverified local result, never seed best_score.
            return False
        try:
            receipt = self.verifier.verify(checkpoint)
        except Exception as error:
            self.store.record_verification(checkpoint_id, {"verified": False, "error": str(error)}, False)
            self.store.event("verification_failed", {"checkpoint_id": checkpoint_id, "error": str(error)})
            raise
        promoted = self.store.promote(checkpoint_id, receipt)
        if promoted:
            data = replay_bytes(checkpoint["seed"], checkpoint["words"], s)
            blob = self.store.put_blob(data)
            self.store.set_meta("champion_replay", blob)
        return promoted

    def import_replay(self, path, stride=4096):
        if stride < 1:
            raise ValueError("stride must be positive")
        seed, words, header = read_replay(path)
        self.store.ensure_seed(seed, control=seed == self.config.control_seed)
        parent = None
        imported = []
        for stop in sorted({0, *range(stride, len(words), stride), len(words)}):
            self._trim(self.config.candidates + 2)
            row = self.engine.load(seed, words[:stop])
            if row["summary"]["tick"] != stop:
                self.engine.release([row["handle"]])
                raise RuntimeError("replay continues after natural termination")
            key = self.store.add_checkpoint(seed, words[:stop], row["summary"], parent,
                                             operator="import")
            self._remember(key, row["handle"])
            imported.append(key)
            parent = key
        final = self.store.get_checkpoint(imported[-1])
        if any(final["summary"].get(k) != v for k, v in header.items()):
            raise RuntimeError("imported replay header disagrees with engine")
        self._verify(imported[-1], force=True)
        # The original suffix is an explicit baseline option at imported roots.
        self.store.set_meta(f"baseline:{seed}", final["id"])
        self.store.event("import", {"path": str(Path(path).resolve()), "seed": seed,
                                    "checkpoints": imported})
        self._prune()
        return imported

    def _prune(self):
        result = self.store.prune_archive(self.config.per_cell, self.config.checkpoint_limit,
                                          self.config.archive_bytes)
        active = {cp["id"] for cp in self.store.checkpoints()}
        for key in list(self.cache):
            if key not in active:
                self.engine.release([self.cache.pop(key)])
        return result

    def _example(self, before, after, candidate, cost, group_deltas=(), metrics=None):
        target = target_group(before)
        new = {g["id"]: g for g, members in live_groups(after)}
        if target:
            old = target[0]
            current = new.get(old["id"])
            survival = bool(current)
            delta = next((d for d in group_deltas if d["id"] == old["id"]), None)
            extension = (bool(delta["chain_after"] > delta["chain_before"]) if delta else
                         bool(current["chain"] > old["chain"]) if current else None)
        else:
            survival = None
            old_ids = {g["id"] for g in before.get("groups", [])}
            if metrics is not None and "max_new_chain" in metrics:
                extension = metrics["max_new_chain"] > 2
            else:
                extension = True if any(g["chain"] > 2 and gid not in old_ids for gid, g in new.items()) else None
        terminal = after.get("terminated") and not after.get("truncated")
        gain = (after["canonical_score"] - before["score"]) if terminal else 0
        return {"seed": before["seed"], "features": features(before, candidate),
                "targets": {"gain": gain, "extension": int(extension) if extension is not None else None,
                            "survival": int(survival) if survival is not None else None},
                "role": "proposal",
                "budget_seconds": candidate["budget_seconds"], "actual_seconds": cost,
                "planner_version": PLANNER_VERSION, "engine_identity": self.identity,
                "model_identity": getattr(self, "model_identity", None),
                "censored": not terminal, "horizon": candidate["horizon"]}

    def execute(self, job, budget_seconds=None):
        budget_seconds = self.config.job_seconds if budget_seconds is None else budget_seconds
        start = time.monotonic()
        payload = job["payload"]
        if job["checkpoint_id"]:
            parent = self.store.get_checkpoint(job["checkpoint_id"])
        else:
            self._trim(self.config.candidates + 2)
            row = self.engine.load(job["seed"], [])
            key = self.store.add_checkpoint(job["seed"], [], row["summary"], operator="reset")
            self._remember(key, row["handle"])
            parent = self.store.get_checkpoint(key)
        best_before = next(s["best_score"] for s in self.store.seeds() if s["seed"] == job["seed"])
        observed_terminal = False
        progress = self.store.get_meta(f"progress:{job['id']}")
        if progress:
            parent = self.store.get_checkpoint(progress["checkpoint_id"])
            if progress.get("rollout_rejected"):
                raise RolloutBatchRejected("all candidate rollouts were rejected; diagnostics preserved")
        allocation_state = parent["summary"]
        total_ticks = progress.get("simulated_ticks", 0) if progress else 0
        published = progress.get("published", []) if progress else []
        batches = progress.get("batches", 0) if progress else 0
        observed_terminal = bool(progress and progress.get("observed_terminal"))
        attempt_observed_terminal = False
        initial_batches = batches
        job_horizon = payload.get("horizon", self.config.horizon)
        while time.monotonic() - start < budget_seconds and not (progress and progress.get("completed")):
            self.store.renew(job["id"], job["token"], self.config.engine_timeout * 3 + self.config.job_seconds)
            horizon = min(job_horizon, self.engine.identity.get("max_horizon", 100000),
                          self.config.maximum_ticks - parent["summary"]["tick"])
            if horizon < 1 or parent["summary"].get("terminated") or parent["summary"].get("truncated"):
                break
            baseline = []
            baseline_id = self.store.get_meta(f"baseline:{job['seed']}")
            if baseline_id:
                reference = self.store.get_checkpoint(baseline_id)["words"]
                if reference[:len(parent["words"])] == parent["words"]:
                    baseline = reference[len(parent["words"]):len(parent["words"]) + horizon]
            search_seed = (payload["search_seed"] + batches) & 0xffffffff
            candidates = generate(parent["summary"], self.config.candidates * 2, search_seed,
                                  payload.get("operator", "recruit"), baseline, horizon)
            held = parent["words"][-1] & 3 if parent["words"] else 0
            for c in candidates:
                if c["mode"] and c["words"] and held & c["words"][0] & 3:
                    c["words"] = [0, *c["words"]][:horizon]
                c["budget_seconds"] = budget_seconds
            candidates = select_candidates(parent["summary"], candidates, self.config.candidates,
                                           self.guide, self.config.exploration_fraction, search_seed)
            self._trim(self.config.candidates + 1)
            handle = self._load(parent)
            batch_start = time.monotonic()
            output = self.engine.batch(handle, horizon, candidates)
            elapsed = time.monotonic() - batch_start
            results = output["results"]
            rejected = output.get("rejected", [])
            if len(results) + len(rejected) != len(candidates):
                raise RuntimeError("engine returned wrong candidate count")
            total_ticks += output["simulated_ticks"]
            paired = []
            indices = set()
            for position, row in enumerate(results):
                index = row.get("candidate_index", None if rejected else position)
                if type(index) is not int or not 0 <= index < len(candidates) or index in indices:
                    raise RuntimeError("engine returned invalid candidate indices")
                indices.add(index)
                paired.append((candidates[index], row))
            for failure in rejected:
                index = failure.get("candidate_index")
                if type(index) is not int or not 0 <= index < len(candidates) or index in indices:
                    raise RuntimeError("engine returned invalid candidate indices")
                indices.add(index)
                trace = failure["words"]
                blob = self.store.put_blob(struct.pack(f"<{len(trace)}I", *trace))
                self.store.event("rollout_rejected", {"job_id": job["id"], "attempt_token": job["token"],
                    "parent_checkpoint_id": parent["id"], "seed": parent["seed"], "horizon": horizon,
                    "candidate_index": index, "candidate": candidates[index],
                    "executed_words_blob": blob, "executed_word_count": len(trace),
                    "diagnostic": failure["diagnostic"]})
            if not results:
                self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                    "batches": batches + 1, "simulated_ticks": total_ticks, "published": published,
                    "rollout_rejected": True, "observed_terminal": observed_terminal})
                self._prune()
                raise RolloutBatchRejected("all candidate rollouts were rejected; diagnostics preserved")
            survivors = select_survivors(results, self.config.survivors, self.guide,
                                         horizon=horizon, budget_seconds=budget_seconds)
            survivor_handles = {r["handle"] for r in survivors}
            rows = {}
            retained = []
            terminal_ids = []
            for candidate, row in paired:
                self.store.add_example(self._example(parent["summary"], row["summary"], candidate,
                                                     elapsed / max(1, len(results)), row.get("group_deltas", ()), row.get("metrics")))
                if row["handle"] in survivor_handles or row["summary"].get("terminated"):
                    key = self._publish(parent, row, candidate["operator"], elapsed / len(results), candidate["search_seed"])
                    rows[row["handle"]] = key
                    published.append(key)
                    if row["handle"] in survivor_handles:
                        retained.append((key, row["handle"]))
                    if row["summary"].get("terminated"):
                        terminal_ids.append(key)
                if row["handle"] not in survivor_handles:
                    self.engine.release([row["handle"]])
            if terminal_ids:
                best = max(terminal_ids, key=lambda key: self.store.get_checkpoint(key)["summary"]["canonical_score"])
                self._verify(best)
                observed_terminal = True
                attempt_observed_terminal = True
            live = [row for row in survivors if not row["summary"].get("terminated")
                    and not row["summary"].get("truncated")]
            # Cache the chosen live continuation last. This preserves it even
            # when the worker has room for only parent + one candidate batch.
            preferred = live[0]["handle"] if live else None
            for key, resident in sorted(retained, key=lambda pair: pair[1] == preferred):
                self._remember(key, resident)
            batches += 1
            if not live:
                self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": best if terminal_ids else parent["id"],
                    "batches": batches, "simulated_ticks": total_ticks, "published": published,
                    "completed": True, "observed_terminal": observed_terminal})
                break
            parent = self.store.get_checkpoint(rows[live[0]["handle"]])
            self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                "batches": batches, "simulated_ticks": total_ticks, "published": published,
                "observed_terminal": observed_terminal})
        best_after = next(s["best_score"] for s in self.store.seeds() if s["seed"] == job["seed"])
        allocation = {"operator": "allocation", "budget_seconds": budget_seconds, "horizon": job_horizon}
        if batches > initial_batches:
            self.store.add_example({"seed": job["seed"], "features": features(allocation_state, allocation),
                "targets": {"compute_gain": max(0, best_after - best_before) / max(.001, time.monotonic() - start)},
                "budget_seconds": budget_seconds, "actual_seconds": time.monotonic() - start,
                "planner_version": PLANNER_VERSION, "engine_identity": self.identity,
                "model_identity": self.model_identity,
                "censored": not attempt_observed_terminal, "role": "compute_allocation"})
        archive = self._prune()
        return {"checkpoints": published, "batches": batches, "archive": archive,
                "attempt_token": job["token"], "model_identity": self.model_identity,
                "collection_models": self.store.job_model_identities(job["id"])}, total_ticks

    def train_shadow(self):
        examples = self.store.sample_examples(self.config.max_training_examples,
                                              self.config.seed_generator, supervised_only=True)
        if len({row["seed"] for row in examples}) < 2:
            return None
        started = time.monotonic()
        model, report = train_guide(examples, epochs=self.config.training_epochs,
                                    seed=self.config.seed_generator)
        report["collection_models"] = collection_models(examples)
        path = self.store.root / "models" / f"shadow-{uuid.uuid4().hex}.json"
        model.save(path, {"engine_identity": self.identity,
                          "collection_models": report["collection_models"]})
        result = {"path": str(path), "sha256": sha256_file(path), "report": report,
                  "seconds": time.monotonic() - started, "active": False}
        self.store.set_meta("latest_shadow_model", result)
        self.store.event("shadow_model", result)
        return result

    def run(self, *, seconds=3600., max_jobs=None, progress=None):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0 or max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
            raise ValueError("positive run budget required")
        deadline = time.monotonic() + seconds
        count = 0
        with campaign_lock(self.store.root), run_accounting(self.store):
            while time.monotonic() < deadline and (max_jobs is None or count < max_jobs):
                job = self.store.claim(self.owner, self.config.engine_timeout * 3 + self.config.job_seconds)
                if not job:
                    next_job(self.store, self.config, self.guide)
                    job = self.store.claim(self.owner, self.config.engine_timeout * 3 + self.config.job_seconds)
                if not job:
                    break
                started = time.monotonic()
                try:
                    self.store.set_attempt_metadata(job["id"], job["token"], {
                        "planner_version": PLANNER_VERSION, "engine_identity": self.identity,
                        "model_identity": self.model_identity,
                    })
                    result, ticks = self.execute(job, min(self.config.job_seconds, max(.001, deadline - time.monotonic())))
                    self.store.finish(job["id"], job["token"], result,
                                      cost_seconds=time.monotonic() - started, simulated_ticks=ticks)
                    self.store.set_meta(f"progress:{job['id']}", None)
                except BaseException as error:
                    retry = isinstance(error, KeyboardInterrupt)
                    saved = self.store.get_meta(f"progress:{job['id']}") or {}
                    # Retried work settles cumulative ticks on eventual completion.
                    # A permanent failure has no later settlement, so account its
                    # durable completed batches now, without counting them twice.
                    ticks = 0 if retry else max(0, saved.get("simulated_ticks", 0) - job.get("simulated_ticks", 0))
                    self.store.fail(job["id"], job["token"], str(error), retry=retry,
                                    cost_seconds=time.monotonic() - started, simulated_ticks=ticks,
                                    rollout_rejected=isinstance(error, RolloutBatchRejected))
                    if not isinstance(error, RolloutBatchRejected):
                        raise
                    self.store.set_meta(f"progress:{job['id']}", None)
                    count += 1
                    if progress:
                        progress({"job": job["id"], "kind": job["kind"], "seed": job["seed"],
                                  "completed_jobs": count, "status": "failed", "result": {"rejected": True},
                                  "champion": self.store.champion()})
                    continue
                count += 1
                completed = self.store.completed_jobs()
                if self.config.train_every_jobs and completed % self.config.train_every_jobs == 0 and time.monotonic() < deadline:
                    self.train_shadow()
                if progress:
                    progress({"job": job["id"], "kind": job["kind"], "seed": job["seed"],
                              "completed_jobs": count, "result": {"batches": result["batches"],
                                  "checkpoints": len(result["checkpoints"]), "archive": result["archive"]},
                              "champion": self.store.champion()})
        return self.store.report()

    def close(self):
        if self.cache:
            try:
                self.engine.release(list(self.cache.values()))
            finally:
                self.cache.clear()
