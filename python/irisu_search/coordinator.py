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
from .combos import combo_key, event_ticks, outcome, select_frontier, terminal_candidates
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
    modules = ("__init__", "config", "coordinator", "engine", "neural", "proposals", "scheduler", "store", "verification", "combos", "diagnostics")
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
        self.replay_references = {}
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

    def _replay_inputs(self, parent, horizon, reference_id=None):
        baseline_id = reference_id or self.store.get_meta(f"baseline:{parent['seed']}")
        if not baseline_id:
            return [], []
        cached = self.replay_references.get(parent["seed"])
        if not cached or cached[0] != baseline_id:
            cached = (baseline_id, self.store.get_checkpoint(baseline_id)["words"])
            self.replay_references[parent["seed"]] = cached
        reference = cached[1]
        tick = parent["summary"]["tick"]
        suffix = reference[tick:tick + horizon]
        baseline = suffix if reference[:tick] == parent["words"] else []
        return baseline, suffix

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
        if self.config.combo_search and (s["canonical_score"] > seed_best
                                        or not self.store.get_meta(f"baseline:{checkpoint['seed']}")):
            self.store.set_meta(f"baseline:{checkpoint['seed']}", checkpoint_id)
            if promoted:
                # Parent recipes are exact prefixes, including after deduplicated imports.
                roots, current = [], checkpoint.get("parent_id")
                while current:
                    row = self.store.checkpoint_summaries([current])[0]
                    if not row["summary"].get("terminated") and not row["summary"].get("truncated"):
                        roots.append(row)
                    current = row.get("parent_id")
                if roots:
                    roots.sort(key=lambda row: combo_key(row["summary"]), reverse=True)
                    self.store.set_meta(f"combo_roots:{checkpoint['seed']}",
                                        [row["id"] for row in roots[:self.config.combo_root_limit]])
        if promoted:
            data = replay_bytes(checkpoint["seed"], checkpoint["words"], s)
            blob = self.store.put_blob(data)
            self.store.set_meta("champion_replay", blob)
        return promoted

    def import_replay(self, path, stride=4096, events_path=None):
        if stride < 1:
            raise ValueError("stride must be positive")
        seed, words, header = read_replay(path)
        self.store.ensure_seed(seed, control=seed == self.config.control_seed)
        parent = None
        imported = []
        roots = (event_ticks(events_path, sha256_file(path), seed, header["canonical_score"],
                             self.config.combo_root_limit, self.config.combo_root_lead,
                             self.config.combo_min_chain) if events_path and self.config.combo_search else [])
        for stop in sorted({0, *range(stride, len(words), stride), len(words), *roots}):
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
        if self.config.combo_search:
            candidates = [self.store.get_checkpoint(key) for key in imported[:-1]]
            preferred = [cp for cp in candidates if cp["summary"]["tick"] in roots]
            if not preferred:
                preferred = sorted(candidates, key=lambda cp: combo_key(cp["summary"]), reverse=True)
            self.store.set_meta(f"combo_roots:{seed}", [cp["id"] for cp in preferred[:self.config.combo_root_limit]])
            self.store.event("combo_roots_imported", {"seed": seed, "roots": len(preferred),
                              "event_receipt": str(events_path) if events_path else None})
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
        terminal_job = self.config.combo_search and job["kind"] == "combo_terminal"
        origin_id = (progress.get("origin_id") if progress else None) or payload.get("origin_id") or job["checkpoint_id"] or parent["id"]
        while time.monotonic() - start < budget_seconds and not (progress and progress.get("completed")):
            self.store.renew(job["id"], job["token"], self.config.engine_timeout * 3 + self.config.job_seconds)
            horizon = min(job_horizon, self.engine.identity.get("max_horizon", 100000),
                          self.config.maximum_ticks - parent["summary"]["tick"])
            if horizon < 1 or parent["summary"].get("terminated") or parent["summary"].get("truncated"):
                break
            baseline, reference = self._replay_inputs(parent, horizon, payload.get("reference_id"))
            if terminal_job and payload.get("plan_blob"):
                data = self.store.get_blob(payload["plan_blob"])
                reference = [word[0] for word in struct.iter_unpack("<I", data)][:horizon]
                baseline = []  # An exploration plan is not the verified incumbent.
            held = parent["words"][-1] & 3 if parent["words"] else 0
            search_seed = (payload["search_seed"] + batches) & 0xffffffff
            if terminal_job:
                candidates = []
                signatures = set()
                for refill in range(4):
                    catalog = terminal_candidates(parent["summary"], self.config.candidates,
                                                  (search_seed + refill * 104729) & 0xffffffff,
                                                  reference, horizon, payload.get("operator", "combo"), held)
                    for index, candidate in enumerate(catalog):
                        signature = (candidate["mode"], candidate["search_seed"] if candidate["mode"] else 0,
                                     tuple(candidate["words"]))
                        if signature in signatures:
                            continue
                        if candidates and (index == 0 or self.store.combo_candidate_seen(parent["id"], candidate, horizon)):
                            continue
                        signatures.add(signature)
                        candidates.append(candidate)
                        if len(candidates) >= self.config.candidates:
                            break
                    if len(candidates) >= self.config.candidates:
                        break
            else:
                candidates = generate(parent["summary"], self.config.candidates * 2, search_seed,
                                      payload.get("operator", "recruit"), baseline, horizon, reference, held)
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
            # A returned, validated batch is charged even if publication or verification is interrupted.
            self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                "batches": batches, "simulated_ticks": total_ticks, "published": published,
                "origin_id": origin_id, "observed_terminal": observed_terminal})
            if terminal_job:
                self.store.record_combo_candidates(parent["id"], candidates, horizon)
            if not results:
                self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                    "batches": batches + 1, "simulated_ticks": total_ticks, "published": published,
                    "rollout_rejected": True, "observed_terminal": observed_terminal})
                self._prune()
                raise RolloutBatchRejected("all candidate rollouts were rejected; diagnostics preserved")
            if self.config.combo_search:
                survivors = select_frontier(results, self.config.combo_frontier_width)
                # Terminal publication and verification are independent of the live frontier.
            else:
                survivors = select_survivors(results, self.config.survivors, self.guide,
                                             horizon=horizon, budget_seconds=budget_seconds)
            survivor_handles = {r["handle"] for r in survivors}
            rows = {}
            retained = []
            terminal_scores = {}
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
                        terminal_scores[key] = row["summary"]["canonical_score"]
                if row["handle"] not in survivor_handles:
                    self.engine.release([row["handle"]])
            # Settle completed simulation work even if independent verification fails.
            self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                "batches": batches, "simulated_ticks": total_ticks, "published": published, "origin_id": origin_id,
                "observed_terminal": observed_terminal})
            if terminal_scores:
                best = max(terminal_scores, key=terminal_scores.__getitem__)
                self._verify(best)
                observed_terminal = True
                attempt_observed_terminal = True
            live = [row for row in survivors if not row["summary"].get("terminated")
                    and not row["summary"].get("truncated")]
            if self.config.combo_search:
                observation = f"{job['id']}:{job['token']}:{batches}:{self.model_identity or 'unlearned'}"
                self.store.mark_combo_expanded(parent["id"], observation_key=observation)
                if terminal_job and payload.get("plan_blob"):
                    self.store.mark_combo_plan_expanded(parent["id"], payload["plan_blob"], observation_key=observation)
                self.store.update_combo_frontier(job["seed"], [key for key, _ in retained], origin_id,
                                                  self.config.combo_frontier_width, observation_key=observation)
                if terminal_job:
                    reference_row = next((row for candidate, row in paired
                                          if candidate is candidates[0]), None)
                    reference_score = (reference_row["summary"].get("canonical_score")
                                       if reference_row and reference_row["summary"].get("terminated")
                                       and not reference_row["summary"].get("truncated") else None)
                    if reference_score is not None:
                        reference_key = rows[reference_row["handle"]]
                        if not self.store.db.execute("SELECT 1 FROM verifications WHERE checkpoint_id=? AND verified=1", (reference_key,)).fetchone():
                            self._verify(reference_key, force=True)
                    novel = []
                    baseline_replays = 0
                    for candidate, row in paired:
                        if candidate is candidates[0]:
                            continue
                        executed_reference = list(reference[:len(row["words"])])
                        executed_reference.extend([0] * (len(row["words"]) - len(executed_reference)))
                        if row["words"] == executed_reference:
                            baseline_replays += 1
                        elif row["summary"].get("terminated") and not row["summary"].get("truncated"):
                            novel.append(row)
                    # Preserve full intended futures, including inputs after an early death.
                    # A second edit can then repair a losing first edit without losing its tail.
                    plans = []
                    novel_handles = {row["handle"] for row in novel}
                    for candidate, row in paired:
                        if candidate is candidates[0] or candidate["mode"] != 0 or row["handle"] not in novel_handles:
                            continue
                        data = struct.pack(f"<{len(candidate['words'])}I", *candidate["words"])
                        plans.append({"plan_blob": self.store.put_blob(data),
                                      "score": row["summary"]["canonical_score"]})
                    self.store.record_combo_plans(parent["id"], plans, self.config.combo_frontier_width,
                                                 observation_key=observation)
                    best_novel = max(novel, key=lambda row: row["summary"]["canonical_score"], default=None)
                    if best_novel:
                        novel_id = rows[best_novel["handle"]]
                        verified = self.store.db.execute("SELECT 1 FROM verifications WHERE checkpoint_id=? AND verified=1", (novel_id,)).fetchone()
                        if not verified:
                            self._verify(novel_id, force=True)
                    marker = f"combo-observation:{observation}"
                    if not self.store.get_meta(marker):
                        value = best_novel["summary"]["canonical_score"] if best_novel else None
                        self.store.event("combo_terminal_evaluated", {
                            "job_id": job["id"], "batch": batches, "origin_id": origin_id,
                            "root_id": parent["id"], "reference_id": payload.get("reference_id"),
                            "plan_blob": payload.get("plan_blob"),
                            "reference_score": reference_score, "terminals": len(terminal_scores),
                            "novel_terminals": len(novel), "best_novel_score": value,
                            "gain": value - reference_score if value is not None and reference_score is not None else None,
                            "censored": sum(not row["summary"].get("terminated") or row["summary"].get("truncated") for _, row in paired),
                            "baseline_ties": baseline_replays,
                            "novel_score_ties": sum(row["summary"].get("canonical_score") == reference_score for row in novel) if reference_score is not None else 0,
                            "simulated_ticks": output["simulated_ticks"],
                            "combo_outcome": outcome(parent["summary"], best_novel["summary"],
                                                     best_novel.get("group_deltas", ())) if best_novel else None})
                        self.store.set_meta(marker, True)
            # Cache the chosen live continuation last. This preserves it even
            # when the worker has room for only parent + one candidate batch.
            preferred = live[0]["handle"] if live else None
            for key, resident in sorted(retained, key=lambda pair: pair[1] == preferred):
                self._remember(key, resident)
            batches += 1
            pending_frontier = ([row for row in self.store.combo_frontier(job["seed"])
                                 if row["summary"]["tick"] < self.config.maximum_ticks]
                                if self.config.combo_search and not terminal_job else [])
            if (not live and not pending_frontier) or terminal_job:
                self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": best if terminal_scores else parent["id"],
                    "batches": batches, "simulated_ticks": total_ticks, "published": published, "origin_id": origin_id,
                    "completed": True, "observed_terminal": observed_terminal})
                break
            if self.config.combo_search:
                frontier = pending_frontier
                next_state = min(frontier, key=lambda row: (row["visits"], row["created"], row["checkpoint_id"]), default=None)
                parent = self.store.get_checkpoint(next_state["checkpoint_id"] if next_state else rows[live[0]["handle"]])
                if next_state:
                    origin_id = next_state["origin_id"]
            else:
                parent = self.store.get_checkpoint(rows[live[0]["handle"]])
            self.store.set_meta(f"progress:{job['id']}", {"checkpoint_id": parent["id"],
                "batches": batches, "simulated_ticks": total_ticks, "published": published, "origin_id": origin_id,
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
                planning_started = time.monotonic()
                job = self.store.claim(self.owner, self.config.engine_timeout * 3 + self.config.job_seconds)
                if not job:
                    next_job(self.store, self.config, self.guide)
                    job = self.store.claim(self.owner, self.config.engine_timeout * 3 + self.config.job_seconds)
                if not job:
                    break
                started = planning_started
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
