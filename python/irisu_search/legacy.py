"""Run the recovered October event beam on the resident exact simulator."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
from itertools import count
import json
import os
from pathlib import Path
import random
import queue
import subprocess
import threading
import time

from .engine import Engine
from .historical import growth, opportunity_value, search
from .historical.common import snapshot, trace_bytes
from .verification import NativeVerifier, read_replay, replay_bytes


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def observation(summary):
    """Expose the original policy observation, including actual lifecycle."""
    result = {k: summary[k] for k in (
        "tick", "score", "gauge", "level", "highest_chain", "terminated", "truncated",
        "left_held", "right_held")}
    result.update(gauge_max=40000, qualifying_clear_count=summary["qualifying"])
    if summary["terminated"]:
        result.update(canonical_score=summary["canonical_score"],
                      canonical_level=summary["canonical_level"],
                      canonical_highest_chain=summary["canonical_chain"])
    kinds = ("piece", "projectile", "bonus")
    shapes = ("circle", "box", "triangle")
    lifecycles = ("scripted_falling", "dynamic_fresh", "confirmed", "rotten", "deleted")
    result["bodies"] = [{
        **{key: body[key] for key in ("id", "color", "x", "y", "vx", "vy", "angle",
            "angular_velocity", "size", "chain_id", "age_ticks", "rot_timer")},
        "kind": kinds[body["kind"]], "shape": shapes[body["shape"]],
        "lifecycle": lifecycles[body["lifecycle"]],
        "projectile_hits": body["hits"], "remaining_lifetime": body["lifetime"],
    } for body in summary["bodies"]]
    return result


class RejectedOption(RuntimeError):
    pass


class Batcher:
    """Coalesce imperative option steps while retaining their independent worlds."""
    def __init__(self, engine, on_batch=None):
        self.engine, self.queue = engine, queue.Queue()
        self.on_batch = on_batch
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def request(self, operation, value):
        future = Future()
        self.queue.put((operation, value, future))
        return future.result()

    def batch_events(self, handle, horizon, candidates):
        if len(candidates) != 1:
            raise ValueError("option cursors submit one independent future")
        return self.request("step", {"handle": handle, "horizon": horizon,
                                    "words": candidates[0]["words"]})

    def release(self, handles):
        values = list(handles)
        if values:
            self.request("release", values)

    def run(self):
        failure = None
        while True:
            first = self.queue.get()
            if first is None:
                return
            items = [first]
            stopping = False
            deadline = time.monotonic() + .001
            while len(items) < 128:
                try:
                    item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                    if item is None:
                        stopping = True
                        break
                    items.append(item)
                except queue.Empty:
                    break
            try:
                if failure:
                    raise failure
                releases = [item for item in items if item[0] == "release"]
                steps = [item for item in items if item[0] == "step"]
                # Reclaim completed branch storage before admitting another wave.
                if releases:
                    self.engine.release({h for _, values, _ in releases for h in values})
                    for _, _, future in releases:
                        future.set_result(None)
                if steps:
                    response = self.engine.multi_batch_events([item[1] for item in steps])
                    if self.on_batch:
                        self.on_batch(len(steps))
                    for key in ("results", "rejected"):
                        for row in response[key]:
                            index = row["candidate_index"]
                            steps[index][2].set_result({"results": [row] if key == "results" else [],
                                "rejected": [row] if key == "rejected" else [],
                                "simulated_ticks": len(row["words"])})
            except BaseException as error:
                failure = error
                for _, _, future in items:
                    if not future.done():
                        future.set_exception(error)
            if stopping:
                return

    def close(self):
        self.queue.put(None)
        self.thread.join()


class Environment:
    """Mutable branch cursor over immutable resident checkpoint handles."""
    def __init__(self, engine, result, account, owned=False):
        self.engine, self.result, self.account, self.owned = engine, result, account, owned

    def observation(self):
        return observation(self.result["summary"])

    def state_hash(self):
        # Handles identify complete immutable worlds, including hidden physics.
        return self.result["handle"]

    def advance(self, words):
        response = self.engine.batch_events(self.result["handle"], len(words), [{"words": words}])
        self.account(response)
        if response["rejected"]:
            raise RejectedOption(str(response["rejected"][0]["diagnostic"]))
        previous, owned = self.result, self.owned
        self.result, self.owned = response["results"][0], True
        if owned:
            self.engine.release([previous["handle"]])
        return self.observation(), snapshot(self.observation()), self.result["events"]

    def legacy_step(self, word, duration):
        if duration < 1 or word & 3 and duration != 1:
            raise ValueError("invalid legacy action span")
        events = []
        while duration:
            span = min(512, duration)
            before = self.result["summary"]["tick"]
            obs, final, batch = self.advance([word] + [0] * (span - 1))
            events.extend(batch)
            duration -= final["tick"] - before
            if final["terminated"] or final["truncated"]:
                break
            word = 0
        return obs, final, events

    @contextmanager
    def fast_checkpoint(self):
        # Pin this handle while child cursors advance and release their own worlds.
        parent = self.result
        environment = self

        class Checkpoint:
            @contextmanager
            def branch(self):
                branch = Environment(environment.engine, parent, environment.account)
                try:
                    yield branch
                finally:
                    branch.close()

        yield Checkpoint()

    def close(self):
        if self.owned:
            self.engine.release([self.result["handle"]])
            self.owned = False


class Deadline(Exception):
    pass


class Budget:
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds
        self.cancelled = False

    def check(self):
        if self.cancelled or time.monotonic() >= self.deadline:
            raise Deadline("search budget")


def node_data(node):
    return {"suffix": node.suffix, "observation": node.observation,
            "history": node.history.to_dict(), "labels": node.labels, "family": node.family,
            "maximum_score_debt": node.maximum_score_debt}


def restore_node(data, args):
    obs = data["observation"]
    history = opportunity_value.ComboHistory.from_dict(data["history"])
    return growth.Node(tuple(data["suffix"]), obs, snapshot(obs), history,
        search.evaluate_value(obs, history, args, elapsed_ticks=obs["tick"] - args.tick),
        data["family"], tuple(data["labels"]), data.get("maximum_score_debt", 0))


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def trace_histories(path, ticks):
    """Read only prefix events from independently replayed native genealogy."""
    history, result = opportunity_value.ComboHistory(), {}
    rows = [json.loads(line) for line in Path(path).open()]
    cursor = 0
    for tick in sorted(set(ticks)):
        while cursor < len(rows):
            row = rows[cursor]
            if "tick" in row and row["tick"] > tick:
                break
            cursor += 1
            kind = {"join": "chain_joined", "qualify": "confirmed", "score": "cleared",
                    "deletion_mark": "destroyed", "ejected": "ejected"}.get(row["kind"])
            if kind:
                history.update([{"kind": kind, "tick": row["tick"], "a": row["body_id"],
                                 "value": row["group_id"] if kind == "chain_joined" else row["num"]}])
        result[tick] = history.copy()
    return result


class Campaign:
    def __init__(self, root, config, incumbent, engine):
        self.root, self.config, self.engine = Path(root), config, engine
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        identity = {"config": config, "engine": engine.identity,
                    "planner": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), Path(__file__).with_name("engine.py"),
                                  Path(__file__).with_name("verification.py"),
                                  *sorted(Path(__file__).with_name("historical").glob("*.py"))]}}
        identity_path = self.root / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("historical campaign runtime or configuration changed")
        atomic_json(identity_path, identity)
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "jobs_done": 0, "branches": 0, "stages": 0, "simulated_ticks": 0,
            "batches": 0, "rejected_candidates": 0, "failed_verifications": 0,
            "verified_improvements": 0, "incumbent": str(Path(incumbent).resolve())}
        self.verifier = NativeVerifier(config["oracle"], config["oracle_library"], 600)
        self.best_path = Path(self.state["incumbent"])
        seed, words, header = read_replay(self.best_path)
        loaded = engine.load(seed, words)
        try:
            receipt = self.verifier.verify({"seed": seed, "words": loaded["words"],
                                            "summary": loaded["summary"]})
            if receipt["canonical_score"] != header["canonical_score"]:
                raise RuntimeError("incumbent header disagrees with independent verifier")
        finally:
            engine.release([loaded["handle"]])
        self.state.update(verified_score=receipt["canonical_score"], verified_seed=seed,
                          incumbent=str(self.best_path), incumbent_sha256=receipt["replay_sha256"])
        atomic_json(self.root / "incumbent-verification.json", receipt)
        self.publish("initializing")

    def publish(self, phase="searching", **extra):
        with self.lock:
            self.state.update(extra)
            atomic_json(self.state_path, self.state)
            atomic_json(self.root / "progress-source.json", {**self.state, "phase": phase,
                        "updated": time.time(), **extra})
        atomic_json(self.root / "process-status.json", {"phase": phase, "pid": os.getpid(),
                    "unbounded": True, "time": time.time()})

    def account(self, response):
        with self.lock:
            self.state["simulated_ticks"] += response["simulated_ticks"]
            self.state["batches"] += 1
            self.state["rejected_candidates"] += len(response["rejected"])
            if response["rejected"]:
                with (self.root / "rejections.jsonl").open("a") as stream:
                    stream.write(json.dumps(response["rejected"]) + "\n")

    def mixed_batch(self, count):
        with self.lock:
            self.state["mixed_batches"] = self.state.get("mixed_batches", 0) + 1
            self.state["mixed_futures"] = self.state.get("mixed_futures", 0) + count
            self.state["largest_mixed_batch"] = max(self.state.get("largest_mixed_batch", 0), count)

    def histories(self, source, ticks):
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        path = self.root / "traces" / (digest + ".jsonl")
        if not path.exists() or not path.with_suffix(".receipt.json").exists():
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(".tmp")
            env = dict(os.environ, LD_LIBRARY_PATH=str(Path(self.config["oracle_library"]).parent))
            env.pop("LD_PRELOAD", None)
            with temporary.open("w") as stream:
                subprocess.run([self.config["trace"], str(source)], stdout=stream,
                               stderr=subprocess.PIPE, env=env, timeout=600, check=True)
            rows = [json.loads(line) for line in temporary.open()]
            seed, _, header = read_replay(source)
            if (rows[0]["seed"] != seed or rows[-1].get("verified") is not True
                    or rows[-1]["canonical_score"] != header["canonical_score"]):
                raise RuntimeError("native prefix genealogy did not verify")
            temporary.replace(path)
            atomic_json(path.with_suffix(".receipt.json"), {"replay_sha256": digest,
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        if receipt != {"replay_sha256": digest, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}:
            raise RuntimeError("native genealogy identity changed")
        return trace_histories(path, ticks)

    def terminal(self, node, prefix, job):
        if node.final["canonical_score"] <= self.state["verified_score"]:
            return
        words = prefix + list(node.suffix)
        summary = {**node.final, "terminated": True, "truncated": False,
                   "canonical_chain": node.final["canonical_highest_chain"]}
        checkpoint = {"seed": self.state["verified_seed"], "words": words, "summary": summary}
        candidate = self.root / f"candidate-{node.identifier}.rpy"
        candidate.write_bytes(replay_bytes(checkpoint["seed"], words, summary))
        try:
            receipt = self.verifier.verify(checkpoint)
        except Exception as error:
            self.state["failed_verifications"] += 1
            atomic_json(candidate.with_suffix(".failure.json"), {"error": str(error), "job": job})
            self.publish("failed")
            raise
        destination = self.root / f"best-verified-{receipt['canonical_score']}.rpy"
        candidate.replace(destination)
        atomic_json(destination.with_suffix(".receipt.json"), receipt)
        self.state.update(verified_score=receipt["canonical_score"], incumbent=str(destination),
                          incumbent_sha256=receipt["replay_sha256"])
        self.state["verified_improvements"] += 1
        self.publish()

    def reconstruct(self, root_result, nodes, args):
        """Restore selected exact prefixes in batches; histories remain measured."""
        results = {}
        for length in sorted({len(n.suffix) for n in nodes}):
            group = [n for n in nodes if len(n.suffix) == length]
            if not length:
                results[group[0].identifier] = root_result
                continue
            response = self.engine.batch(root_result["handle"], length,
                                         [{"words": list(n.suffix)} for n in group])
            self.account(response)
            if response["rejected"]:
                raise RuntimeError("previously accepted beam prefix could not be restored")
            for node, result in zip(group, response["results"]):
                if observation(result["summary"]) != node.observation:
                    raise RuntimeError("selected beam prefix changed on exact reconstruction")
                results[node.identifier] = result
        return results

    def expand(self, transport, parent_result, parent, option, words, prefix, args, budget, limit):
        env = Environment(transport, parent_result, self.account)
        try:
            budget.check()
            with env.fast_checkpoint() as checkpoint, checkpoint.branch() as branch:
                if option is None or option.mode == "macro":
                    span = min(args.segment, limit - parent.final["tick"])
                    plan = words[parent.final["tick"]:parent.final["tick"] + span] if option is None else [
                        dict(option.macro.sparse).get(i, 0) for i in range(option.duration)]
                    obs, final, events = branch.advance(plan + [0] * (span - len(plan)))
                    history = parent.history.copy()
                    history.update(events)
                    suffix = parent.suffix + tuple(branch.result["words"])
                else:
                    outcome = search.execute_candidate(branch, parent.observation,
                        parent.history, option, args, budget, limit)
                    if not outcome["actions"]:
                        return None
                    obs, final, history = (outcome[k] for k in ("observation", "final", "history"))
                    suffix = parent.suffix + tuple(outcome["actions"])
                full, obs, final, exhausted = search.finish_exhausted(branch,
                    prefix + list(suffix), obs, final, history, args.max_ticks, budget)
                if exhausted:
                    suffix = tuple(full[args.tick:])
                return growth.Node(suffix, obs, final, history,
                    search.evaluate_value(obs, history, args, elapsed_ticks=final["tick"] - args.tick),
                    option.family if option else "reference",
                    parent.labels + (option.label if option else "reference",), parent.maximum_score_debt)
        except (RejectedOption, Deadline):
            return None

    def job(self):
        job = self.state["jobs_done"]
        directory = self.root / "jobs" / f"{job:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        saved_path = directory / "frontier.json"
        saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
        source = Path(saved["source"] if saved else self.state["incumbent"])
        if saved and hashlib.sha256(source.read_bytes()).hexdigest() != saved["source_sha256"]:
            raise RuntimeError("saved beam source replay changed")
        seed, words, _ = read_replay(source)
        roots = [min(tick, len(words) - 1) for tick in self.config["roots"]]
        tick = saved["args"]["tick"] if saved else roots[job % len(roots)]
        args = search.defaults(**self.config.get("search", {}), tick=tick, seed=seed,
                               random_seed=661003 + job)
        root_history = self.histories(source, [tick])[tick]
        root_result = self.engine.load(seed, words[:tick])
        for group in root_result["summary"]["groups"]:
            past = root_history.groups.get(group["id"], {})
            if (past.get("cumulative_members"), past.get("successful_clears")) != (group["chain"], group["num"]):
                raise RuntimeError("root genealogy disagrees with simulator group counters")
        root_env = Environment(self.engine, root_result, self.account)
        frontier = [restore_node(n, args) for n in saved["frontier"]] if saved else [
            search.initial_node(root_env.observation(), root_history, args)]
        rng = random.Random(args.random_seed)
        if saved:
            rng.setstate(tuples(saved["rng"]))
        stage = saved["stage"] if saved else 0
        budget = Budget(self.config.get("search_seconds", 1200))
        # Stop between complete stages so no unexpanded parent loses its niche.
        work_budget = Budget(float("inf"))
        prefix, limit = words[:tick], min(args.max_ticks, tick + args.horizon)
        active = {}
        incomplete = False
        self.publish(job=job, root_tick=tick, stage=stage, beam=len(frontier))
        try:
            while frontier and any(n.final["tick"] < limit for n in frontier):
                budget.check()
                active = self.reconstruct(root_result, frontier, args)
                children = []
                transport = Batcher(self.engine, self.mixed_batch)
                futures = []
                processed = set()
                executor = ThreadPoolExecutor(max_workers=64)
                try:
                    # Proposal RNG and result tie order match the original parent order.
                    for parent in frontier:
                        if parent.final["tick"] >= limit:
                            continue
                        try:
                            work_budget.check()
                            env = Environment(transport, active[parent.identifier], self.account)
                            options = search.candidate_options(env, parent.observation, parent.history, args,
                                min(args.segment, limit - parent.final["tick"]), rng, work_budget)
                        except Deadline:
                            break
                        for option in [None, *options]:
                            future = executor.submit(self.expand, transport, active[parent.identifier],
                                parent, option, words, prefix, args, work_budget, limit)
                            futures.append((parent.identifier, future))
                        self.publish(job=job, root_tick=tick, stage=stage, beam=len(frontier))
                    with (directory / "candidates.jsonl").open("a") as journal:
                        for parent_id, future in futures:
                            node = future.result()
                            processed.add(id(future))
                            if node is None:
                                continue
                            self.state["branches"] += 1
                            journal.write(json.dumps({"stage": stage, "parent": parent_id,
                                "candidate": node.identifier, "family": node.family, "final": node.final}) + "\n")
                            if node.natural:
                                self.terminal(node, prefix, job)
                            elif not node.final["truncated"]:
                                children.append(node)
                            if self.state["branches"] % 64 == 0:
                                self.publish(job=job, root_tick=tick, stage=stage, beam=len(frontier))
                except BaseException:
                    work_budget.cancelled = True
                    raise
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
                    transport.close()
                    # Completed later futures may contain a record even if the
                    # main thread was interrupted while reading an earlier one.
                    for _, future in futures:
                        if (id(future) not in processed and not future.cancelled()
                                and future.done() and future.exception() is None):
                            node = future.result()
                            if node is not None and node.natural:
                                self.terminal(node, prefix, job)
                for result in active.values():
                    if result["handle"] != root_result["handle"]:
                        self.engine.release([result["handle"]])
                active = {}
                frontier = search.select_diverse(children, args.beam_width, root_history)
                stage += 1
                self.state["stages"] += 1
                atomic_json(saved_path, {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "args": vars(args), "stage": stage, "rng": rng.getstate(),
                    "frontier": [node_data(n) for n in frontier]})
                self.publish(job=job, root_tick=tick, stage=stage, beam=len(frontier))
        except Deadline:
            incomplete = True
        finally:
            for result in active.values():
                if result["handle"] != root_result["handle"]:
                    self.engine.release([result["handle"]])
            self.engine.release([root_result["handle"]])
        atomic_json(directory / "result.json", {"stages": stage, "finished": time.time(),
                    "continuing": incomplete, "verified_score": self.state["verified_score"]})
        self.state["slices_done"] = self.state.get("slices_done", 0) + 1
        if not incomplete:
            self.state["jobs_done"] += 1
        self.publish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if not isinstance(config.get("roots"), list) or not config["roots"] or any(
            type(t) is not int or t < 0 for t in config["roots"]):
        parser.error("roots must be a nonempty list of nonnegative ticks")
    settings = search.defaults(**config.get("search", {}))
    if (not 1 <= settings.segment <= 512 or not 1 <= settings.beam_width <= 64
            or not 1 <= settings.candidates <= 128 or settings.horizon < 1
            or config.get("search_seconds", 1200) <= 0):
        parser.error("invalid historical search limits")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Engine(config["engine"], timeout_seconds=600,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": config.get("cuda_device", "0")}) as engine:
            campaign = Campaign(args.root, config, args.incumbent, engine)
            try:
                for _ in range(args.max_jobs) if args.max_jobs is not None else count():
                    campaign.job()
                campaign.publish("complete")
            except KeyboardInterrupt:
                campaign.publish("stopped")
                raise
            except BaseException:
                campaign.publish("failed")
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
