"""Durable search jobs, immutable input recipes, and a bounded active archive."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sqlite3
import struct
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _uint(value: Any, name: str, maximum: int = 0xFFFFFFFF) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 0..{maximum}")
    return value


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return float(value)


def _bin(value: int) -> int:
    return max(0, int(value)).bit_length()


def archive_cell(summary: dict[str, Any]) -> str:
    """Coarse selection descriptors, never an assertion of state equivalence."""
    groups = [g for g in summary.get("groups", []) if g.get("members")]
    mature = max(groups, key=lambda g: (g.get("chain", 0) * (g.get("num", 0) + 1), len(g["members"])), default={})
    members = set(mature.get("members", []))
    bodies = [b for b in summary.get("bodies", []) if b.get("id") in members]
    useful = [b for b in bodies if not b.get("pending_delete") and not b.get("pending_clear")]
    height = min((float(b.get("y", 480)) for b in useful), default=480)
    side = min((abs(float(b.get("x", 290)) - 290) for b in useful), default=0)
    return _json([
        int(summary.get("qualifying", summary.get("qualifying_clear_count", 0))) // 10,
        _bin(mature.get("chain", 0)), _bin(mature.get("num", 0)),
        min(3, len(useful)), int(height // 80), int(side // 80),
        bool(summary.get("terminated")),
    ])


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "objects"
        self.blobs.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "search.sqlite3", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS seeds(
              seed INTEGER PRIMARY KEY, control INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'new', search_seconds REAL NOT NULL DEFAULT 0,
              jobs INTEGER NOT NULL DEFAULT 0, best_score INTEGER NOT NULL DEFAULT 0,
              attempts INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS checkpoints(
              id TEXT PRIMARY KEY, seed INTEGER NOT NULL REFERENCES seeds(seed),
              trace_blob TEXT NOT NULL, trace_bytes INTEGER NOT NULL, summary TEXT NOT NULL,
              parent_id TEXT REFERENCES checkpoints(id), operator TEXT NOT NULL,
              cost_seconds REAL NOT NULL, search_seed INTEGER NOT NULL, cell TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 1, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS checkpoints_seed ON checkpoints(seed, active);
            CREATE INDEX IF NOT EXISTS checkpoints_import ON checkpoints(active) WHERE operator='import';
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, seed INTEGER NOT NULL REFERENCES seeds(seed),
              checkpoint_id TEXT REFERENCES checkpoints(id), payload TEXT NOT NULL,
              lane TEXT NOT NULL, priority REAL NOT NULL, dedupe_key TEXT UNIQUE,
              status TEXT NOT NULL DEFAULT 'pending', owner TEXT, token TEXT,
              lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
              result TEXT, error TEXT, cost_seconds REAL NOT NULL DEFAULT 0,
              simulated_ticks INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,
              updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(status, priority, created);
            CREATE TABLE IF NOT EXISTS attempts(
              job_id TEXT NOT NULL REFERENCES jobs(id), token TEXT NOT NULL,
              status TEXT NOT NULL, started REAL NOT NULL, settled REAL,
              cost_seconds REAL NOT NULL DEFAULT 0, simulated_ticks INTEGER NOT NULL DEFAULT 0,
              settlement TEXT, PRIMARY KEY(job_id,token));
            CREATE TABLE IF NOT EXISTS attempt_metadata(
              job_id TEXT NOT NULL, token TEXT NOT NULL, payload TEXT NOT NULL,
              PRIMARY KEY(job_id,token),
              FOREIGN KEY(job_id,token) REFERENCES attempts(job_id,token));
            CREATE TABLE IF NOT EXISTS verifications(
              id TEXT PRIMARY KEY, checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
              verified INTEGER NOT NULL, receipt TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS examples(id TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
        """)
        version = self.get_meta("schema_version")
        if version not in (None, 1):
            raise ValueError("unsupported search store schema")
        self.set_meta("schema_version", 1)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        if key in {"campaign_config", "engine_identity", "schema_version"}:
            old = self.get_meta(key)
            if old is not None and old != value:
                raise ValueError(f"immutable campaign metadata differs: {key}")
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, _json(value)))

    def init_campaign(self, config: dict[str, Any], engine_identity: dict[str, Any]) -> None:
        with self._transaction():
            self.set_meta("campaign_config", config)
            self.set_meta("engine_identity", engine_identity)
            if self.get_meta("created") is None:
                self.set_meta("created", time.time())

    def ensure_seed(self, seed: int, control: bool = False) -> None:
        _uint(seed, "seed")
        self.db.execute("INSERT INTO seeds(seed,control) VALUES(?,?) ON CONFLICT(seed) DO UPDATE SET control=MAX(control,excluded.control)", (seed, int(bool(control))))

    def seeds(self) -> list[dict[str, Any]]:
        return [dict(r, control=bool(r["control"])) for r in self.db.execute("SELECT * FROM seeds ORDER BY seed")]

    def update_seed(self, seed: int, **fields: Any) -> None:
        allowed = {"control", "status", "search_seconds", "jobs", "best_score", "attempts"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("unknown or empty seed update")
        for k, v in fields.items():
            if k == "search_seconds":
                _seconds(v)
            elif k in {"jobs", "best_score", "attempts"}:
                _uint(v, k, (1 << 63) - 1)
            elif k == "status" and (not isinstance(v, str) or not v):
                raise ValueError("seed status must be nonempty")
        self.ensure_seed(seed)
        self.db.execute("UPDATE seeds SET " + ",".join(f"{key}=?" for key in fields) + " WHERE seed=?", (*fields.values(), seed))

    def _blob_path(self, digest: str) -> Path:
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid object digest")
        return self.blobs / digest[:2] / digest[2:]

    def put_blob(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("artifact must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        dest = self._blob_path(digest)
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            self.get_blob(digest)
            return digest
        fd, name = tempfile.mkstemp(prefix=".object-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, dest)
            except FileExistsError:
                self.get_blob(digest)
            directory = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(name)
        return digest

    def get_blob(self, digest: str) -> bytes:
        path = self._blob_path(digest)
        if path.is_symlink():
            raise ValueError("artifact must not be a symlink")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("artifact checksum mismatch")
        return data

    def add_checkpoint(self, seed: int, words: list[int], summary: dict[str, Any], parent_id: str | None = None,
                       operator: str = "load", cost_seconds: float = 0, search_seed: int = 0) -> str:
        _uint(seed, "seed")
        _uint(search_seed, "search_seed", (1 << 63) - 1)
        _seconds(cost_seconds)
        if summary.get("seed", seed) != seed or summary.get("tick") != len(words):
            raise ValueError("checkpoint seed/tick does not match its input recipe")
        for word in words:
            _uint(word, "input word", (1 << 21) - 1)
        trace = b"".join(struct.pack("<I", word) for word in words)
        digest = self.put_blob(trace)
        identity = {"engine": self.get_meta("engine_identity"), "config": self.get_meta("campaign_config"), "seed": seed, "trace": digest}
        checkpoint_id = hashlib.sha256(_json(identity).encode()).hexdigest()
        encoded = _json(summary)
        with self._transaction():
            self.ensure_seed(seed)
            if parent_id:
                parent = self.db.execute("SELECT seed,trace_blob FROM checkpoints WHERE id=?", (parent_id,)).fetchone()
                if parent is None or parent["seed"] != seed:
                    raise ValueError("checkpoint parent belongs to another seed or is missing")
                prefix = self.get_blob(parent["trace_blob"])
                if not trace.startswith(prefix):
                    raise ValueError("checkpoint recipe does not extend its parent")
            previous = self.db.execute("SELECT summary FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
            if previous:
                if previous[0] != encoded:
                    raise ValueError("identical input recipe produced a different state summary")
                self.db.execute("UPDATE checkpoints SET active=1 WHERE id=?", (checkpoint_id,))
                return checkpoint_id
            self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                checkpoint_id, seed, digest, len(trace), encoded, parent_id, operator,
                float(cost_seconds), search_seed, archive_cell(summary), 1, time.time()))
        return checkpoint_id

    @staticmethod
    def _checkpoint(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["summary"] = json.loads(result["summary"])
        result["active"] = bool(result["active"])
        return result

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        result = self._checkpoint(row)
        data = self.get_blob(result["trace_blob"])
        if len(data) != result["trace_bytes"] or len(data) % 4:
            raise ValueError("checkpoint trace length mismatch")
        result["words"] = [x[0] for x in struct.iter_unpack("<I", data)]
        return result

    def checkpoints(self, seed: int | None = None, active: bool | None = True) -> list[dict[str, Any]]:
        where, args = [], []
        if seed is not None:
            where.append("seed=?")
            args.append(seed)
        if active is not None:
            where.append("active=?")
            args.append(int(active))
        sql = "SELECT * FROM checkpoints" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created,id"
        return [self._checkpoint(row) for row in self.db.execute(sql, args)]

    def prune_archive(self, per_cell: int = 3, max_checkpoints: int = 1024,
                      max_trace_bytes: int = 256 * 1024 * 1024) -> dict[str, Any]:
        for value in (per_cell, max_checkpoints, max_trace_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("archive quotas must be positive integers")
        rows = self.checkpoints(active=True)
        # Imported branching roots remain eligible even after a temporary
        # quota squeeze. Cold search descendants are still evidence only.
        rows += [self._checkpoint(row) for row in self.db.execute(
            "SELECT * FROM checkpoints WHERE active=0 AND operator='import'")]
        def rank(row: dict[str, Any]) -> tuple[Any, ...]:
            s = row["summary"]
            live = [g for g in s.get("groups", []) if g.get("members")]
            value = max((g.get("chain", 0) * (g.get("num", 0) + 1) for g in live), default=0)
            return (not bool(s.get("truncated")), value, s.get("canonical_score", 0) if s.get("terminated") else s.get("score", 0), s.get("gauge", 0), row["created"])
        protected: set[str] = set()
        champion = self.champion()
        if champion:
            protected.add(champion["checkpoint_id"])
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        controls = {s["seed"] for s in self.seeds() if s["control"]}
        for seed, values in by_seed.items():
            terminal = [r for r in values if r["summary"].get("terminated") and not r["summary"].get("truncated")]
            if terminal:
                protected.add(max(terminal, key=lambda r: r["summary"].get("canonical_score", 0))["id"])
            if seed in controls:
                live = [r for r in values if not r["summary"].get("terminated")]
                if live:
                    protected.add(max(live, key=rank)["id"])
        cells: Counter[tuple[int, str]] = Counter()
        kept = set(protected)
        by_id = {r["id"]: r for r in rows}
        used_bytes = sum(by_id[i]["trace_bytes"] for i in kept if i in by_id)
        for row in rows:
            if row["id"] in kept:
                cells[(row["seed"], row["cell"])] += 1

        def retain(row: dict[str, Any], *, baseline: bool = False) -> None:
            nonlocal used_bytes
            cell = (row["seed"], row["cell"])
            if row["id"] in kept or (not baseline and cells[cell] >= per_cell):
                return
            if len(kept) >= max_checkpoints or used_bytes + row["trace_bytes"] > max_trace_bytes:
                return
            kept.add(row["id"])
            cells[cell] += 1
            used_bytes += row["trace_bytes"]

        # Reserve replay coverage before ranking descendants. If quotas are
        # tight, visit both ends and progressively bisect the replay instead
        # of keeping only its highest-potential late states.
        roots = {}
        for seed, values in by_seed.items():
            ordered = sorted((r for r in values if r["operator"] == "import"
                              and not r["summary"].get("terminated")
                              and not r["summary"].get("truncated")),
                             key=lambda r: (r["summary"]["tick"], r["id"]))
            indices = [len(ordered) - 1, 0] if len(ordered) > 1 else [0] if ordered else []
            gaps = [(-len(ordered), 0, len(ordered) - 1)] if len(ordered) > 2 else []
            while gaps:
                _, left, right = heapq.heappop(gaps)
                middle = (left + right) // 2
                indices.append(middle)
                for lo, hi in ((left, middle), (middle, right)):
                    if hi - lo > 1:
                        heapq.heappush(gaps, (lo - hi, lo, hi))
            roots[seed] = [ordered[index] for index in indices]
        for depth in range(max((len(values) for values in roots.values()), default=0)):
            for seed in sorted(roots):
                if depth < len(roots[seed]):
                    retain(roots[seed][depth], baseline=True)
        # Round-robin across seeds stops one large archive consuming the entire pool.
        # Terminals have no continuation to explore. Keep each seed's best
        # active and leave the remaining immutable terminal evidence cold.
        pools = {seed: sorted((r for r in values if not r["summary"].get("terminated")),
                              key=rank, reverse=True) for seed, values in by_seed.items()}
        for depth in range(max((len(v) for v in pools.values()), default=0)):
            for seed in sorted(pools):
                if depth >= len(pools[seed]):
                    continue
                row = pools[seed][depth]
                retain(row)
        with self._transaction():
            for row in rows:
                active = row["id"] in kept
                if row["active"] != active:
                    self.db.execute("UPDATE checkpoints SET active=? WHERE id=?", (int(active), row["id"]))
        return {"active": len(kept), "deactivated": sum(r["active"] and r["id"] not in kept for r in rows), "trace_bytes": used_bytes,
                "protected": len(protected), "protected_over_budget": len(kept) > max_checkpoints or used_bytes > max_trace_bytes}

    def enqueue(self, kind: str, seed: int, checkpoint_id: str | None = None, payload: dict[str, Any] | None = None,
                lane: str = "combo", priority: float = 0, dedupe_key: str | None = None) -> str:
        if not isinstance(kind, str) or not kind or not isinstance(lane, str) or not lane or isinstance(priority, bool) or not math.isfinite(priority):
            raise ValueError("invalid job kind/lane/priority")
        if dedupe_key is not None and (not isinstance(dedupe_key, str) or not dedupe_key):
            raise ValueError("dedupe key must be nonempty or None")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("job payload must be an object")
        encoded = _json(payload or {})
        with self._transaction():
            self.ensure_seed(seed)
            if checkpoint_id:
                row = self.db.execute("SELECT seed FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
                if row is None or row[0] != seed:
                    raise ValueError("job checkpoint seed mismatch")
            if dedupe_key:
                row = self.db.execute("SELECT * FROM jobs WHERE dedupe_key=?", (dedupe_key,)).fetchone()
                if row:
                    if (row["kind"], row["seed"], row["checkpoint_id"], row["payload"], row["lane"]) != (kind, seed, checkpoint_id, encoded, lane):
                        raise ValueError("job dedupe key reused for different work")
                    return row["id"]
            job_id, now = uuid.uuid4().hex, time.time()
            self.db.execute("INSERT INTO jobs(id,kind,seed,checkpoint_id,payload,lane,priority,dedupe_key,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (job_id, kind, seed, checkpoint_id, encoded, lane, priority, dedupe_key, now, now))
        return job_id

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("payload", "result"):
            if result[key] is not None:
                result[key] = json.loads(result[key])
        return result

    def claim(self, owner: str, lease_seconds: float = 300) -> dict[str, Any] | None:
        if not owner or _seconds(lease_seconds) <= 0:
            raise ValueError("owner and positive lease required")
        with self._transaction():
            now = time.time()
            row = self.db.execute("SELECT * FROM jobs WHERE status='pending' OR (status='running' AND lease_until<=?) ORDER BY priority DESC,created,id LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                self.db.execute("UPDATE attempts SET status='abandoned',settled=? WHERE job_id=? AND token=? AND status='running'", (now, row["id"], row["token"]))
            token = uuid.uuid4().hex
            self.db.execute("UPDATE jobs SET status='running',owner=?,token=?,lease_until=?,attempts=attempts+1,updated=? WHERE id=?",
                            (owner, token, now + lease_seconds, now, row["id"]))
            self.db.execute("INSERT INTO attempts(job_id,token,status,started) VALUES(?,?,'running',?)", (row["id"], token, now))
            return self._job(self.db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def _owned(self, job_id: str, token: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] != "running" or row["token"] != token or row["lease_until"] <= time.time():
            raise ValueError("job lease expired or ownership token is stale")
        return row

    def renew(self, job_id: str, token: str, seconds: float = 300) -> None:
        if _seconds(seconds) <= 0:
            raise ValueError("lease must be positive")
        with self._transaction():
            self._owned(job_id, token)
            self.db.execute("UPDATE jobs SET lease_until=?,updated=? WHERE id=?", (time.time() + seconds, time.time(), job_id))

    def set_attempt_metadata(self, job_id: str, token: str, metadata: dict[str, Any]) -> None:
        encoded = _json(metadata)
        with self._transaction():
            self._owned(job_id, token)
            previous = self.db.execute("SELECT payload FROM attempt_metadata WHERE job_id=? AND token=?", (job_id, token)).fetchone()
            if previous:
                if previous[0] != encoded:
                    raise ValueError("attempt metadata is immutable")
                return
            self.db.execute("INSERT INTO attempt_metadata VALUES(?,?,?)", (job_id, token, encoded))

    def get_attempt_metadata(self, job_id: str, token: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT payload FROM attempt_metadata WHERE job_id=? AND token=?", (job_id, token)).fetchone()
        return json.loads(row[0]) if row else None

    def job_model_identities(self, job_id: str) -> list[dict[str, Any] | None]:
        identities = {}
        for row in self.db.execute("SELECT payload FROM attempt_metadata WHERE job_id=?", (job_id,)):
            model = json.loads(row[0]).get("model_identity")
            identities[_json(model)] = model
        return [identities[key] for key in sorted(identities)]

    def _settle(self, job_id: str, token: str, status: str, payload: dict[str, Any], cost_seconds: float, ticks: int, retry: bool = False) -> None:
        cost = _seconds(cost_seconds)
        _uint(ticks, "simulated_ticks", (1 << 63) - 1)
        settlement = _json({"status": status, "payload": payload, "cost_seconds": cost, "simulated_ticks": ticks, "retry": retry})
        with self._transaction():
            old = self.db.execute("SELECT settlement FROM attempts WHERE job_id=? AND token=?", (job_id, token)).fetchone()
            if old and old[0] is not None:
                if old[0] != settlement:
                    raise ValueError("attempt already settled differently")
                return
            row = self._owned(job_id, token)
            now = time.time()
            self.db.execute("UPDATE attempts SET status=?,settled=?,cost_seconds=?,simulated_ticks=?,settlement=? WHERE job_id=? AND token=?",
                            (status, now, cost, ticks, settlement, job_id, token))
            self.db.execute("UPDATE jobs SET status=?,result=?,error=?,cost_seconds=cost_seconds+?,simulated_ticks=simulated_ticks+?,lease_until=NULL,updated=? WHERE id=?",
                            ("pending" if retry else status, _json(payload) if status == "done" else None, payload.get("error") if status == "failed" else None, cost, ticks, now, job_id))
            completed_probe = status == "done" or (
                status == "failed" and not retry and payload.get("rollout_rejected") is True)
            self.db.execute("UPDATE seeds SET search_seconds=search_seconds+?,jobs=jobs+1,attempts=attempts+? WHERE seed=?",
                            (cost, int(row["kind"] == "bootstrap" and completed_probe), row["seed"]))

    def finish(self, job_id: str, token: str, result: dict[str, Any], cost_seconds: float = 0, simulated_ticks: int = 0) -> None:
        self._settle(job_id, token, "done", result, cost_seconds, simulated_ticks)

    def fail(self, job_id: str, token: str, error: str, retry: bool = False, cost_seconds: float = 0,
             simulated_ticks: int = 0, *, rollout_rejected: bool = False) -> None:
        if type(rollout_rejected) is not bool or rollout_rejected and retry:
            raise ValueError("rollout rejection must identify a permanently exhausted probe")
        payload = {"error": str(error)}
        if rollout_rejected:
            payload["rollout_rejected"] = True
        self._settle(job_id, token, "failed", payload, cost_seconds, simulated_ticks, retry)

    def jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs" + (" WHERE status=?" if status else "") + " ORDER BY created,id"
        return [self._job(row) for row in self.db.execute(sql, (status,) if status else ())]

    def record_verification(self, checkpoint_id: str, receipt: dict[str, Any], verified: bool) -> None:
        encoded = _json(receipt)
        key = hashlib.sha256(_json([checkpoint_id, bool(verified), receipt]).encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO verifications VALUES(?,?,?,?,?)", (key, checkpoint_id, int(bool(verified)), encoded, time.time()))

    def promote(self, checkpoint_id: str, receipt: dict[str, Any]) -> bool:
        checkpoint = self.get_checkpoint(checkpoint_id)
        summary = checkpoint["summary"]
        score = summary.get("canonical_score")
        if (receipt.get("verified") is not True or not summary.get("terminated") or summary.get("truncated")
                or type(score) is not int or score < 0 or receipt.get("canonical_score") != score
                or receipt.get("seed") != checkpoint["seed"] or receipt.get("tick") != summary.get("tick")):
            raise ValueError("record promotion requires matching verified canonical terminal result")
        with self._transaction():
            self.record_verification(checkpoint_id, receipt, True)
            self.db.execute("UPDATE seeds SET best_score=MAX(best_score,?) WHERE seed=?", (score, checkpoint["seed"]))
            old = self.champion()
            if old is not None and old["score"] >= score:
                return False
            self.set_meta("champion", {"checkpoint_id": checkpoint_id, "seed": checkpoint["seed"], "score": score, "receipt": receipt, "promoted": time.time()})
            self.db.execute("UPDATE checkpoints SET active=1 WHERE id=?", (checkpoint_id,))
            self.event("champion", {"checkpoint_id": checkpoint_id, "seed": checkpoint["seed"], "score": score})
        return True

    def champion(self) -> dict[str, Any] | None:
        return self.get_meta("champion")

    def add_example(self, example: dict[str, Any]) -> str:
        payload = _json(example)
        key = hashlib.sha256(payload.encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO examples VALUES(?,?,?)", (key, payload, time.time()))
        return key

    def examples(self) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM examples ORDER BY created,id")]

    def sample_examples(self, limit: int, seed: int = 0, *, supervised_only: bool = False) -> list[dict[str, Any]]:
        """Sample whole-seed strata without loading the accumulated history.

        Two streaming passes keep at most ``limit`` seed counters and samples.
        Stable priorities make the result independent of insertion order, and
        small seed strata donate unused quota to the remaining selected seeds.
        """
        if type(limit) is not int or limit < 1 or type(seed) is not int:
            raise ValueError("positive sample limit and integer seed required")

        def eligible(example: dict[str, Any]) -> bool:
            if not isinstance(example, dict):
                return False
            value = example.get("seed")
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                return False
            targets = example.get("targets", {})
            if supervised_only and not isinstance(targets, dict):
                return False
            return not supervised_only or any(
                targets.get(name) is not None and
                (name not in {"gain", "compute_gain"} or not example.get("censored"))
                for name in ("gain", "extension", "survival", "compute_gain")
            )

        def priority(kind: str, value: Any) -> int:
            return int.from_bytes(hashlib.sha256(f"{seed}:{kind}:{value}".encode()).digest(), "big")

        counts: dict[int, int] = {}
        seed_heap: list[tuple[int, int]] = []
        for row in self.db.execute("SELECT payload FROM examples"):
            example = json.loads(row[0])
            if not eligible(example):
                continue
            game_seed = example["seed"]
            if game_seed in counts:
                counts[game_seed] += 1
                continue
            key = priority("seed", game_seed)
            if len(counts) >= limit:
                if key >= -seed_heap[0][0]:
                    continue
                _, removed = heapq.heappop(seed_heap)
                del counts[removed]
            counts[game_seed] = 1
            heapq.heappush(seed_heap, (-key, game_seed))
        if not counts:
            return []
        ordered = sorted(counts, key=lambda value: priority("seed", value))
        quotas = dict.fromkeys(ordered, 0)
        available = list(ordered)
        remaining = limit
        while available and remaining:
            next_available = []
            for game_seed in available:
                quotas[game_seed] += 1
                remaining -= 1
                if quotas[game_seed] < counts[game_seed]:
                    next_available.append(game_seed)
                if not remaining:
                    break
            available = next_available
        samples: dict[int, list[tuple[int, str, dict[str, Any]]]] = {value: [] for value in ordered}
        for row in self.db.execute("SELECT id,payload FROM examples"):
            example = json.loads(row[1])
            if not eligible(example):
                continue
            game_seed = example["seed"]
            if game_seed not in quotas:
                continue
            bucket = samples[game_seed]
            key = (-priority("example", row[0]), row[0], example)
            if len(bucket) < quotas[game_seed]:
                heapq.heappush(bucket, key)
            elif key > bucket[0]:
                heapq.heapreplace(bucket, key)
        return [entry[2] for game_seed in ordered for entry in sorted(samples[game_seed], key=lambda item: item[1])]

    def lane_costs(self) -> dict[str, float]:
        return dict(self.db.execute("""
            SELECT lane,SUM(MAX(cost_seconds,0.001)) FROM jobs
            WHERE status IN ('done','failed') GROUP BY lane
        """))

    def checkpoint_uses(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT checkpoint_id,COUNT(*) FROM jobs WHERE checkpoint_id IS NOT NULL GROUP BY checkpoint_id"))

    def completed_jobs(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM jobs WHERE status='done'").fetchone()[0]

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO events(kind,payload,created) VALUES(?,?,?)", (kind, _json(payload), time.time()))

    def report(self) -> dict[str, Any]:
        statuses = dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
        active = self.db.execute("SELECT COUNT(*),COALESCE(SUM(trace_bytes),0) FROM checkpoints WHERE active=1").fetchone()
        seeds = self.seeds()
        return {"schema": "irisu-search-report-v1", "champion": self.champion(), "seeds": seeds,
                "run_wall_seconds": self.get_meta("run_wall_seconds", 0.),
                "latest_shadow_model": self.get_meta("latest_shadow_model"),
                "job_counts": statuses, "search_seconds": sum(s["search_seconds"] for s in seeds),
                "simulated_ticks": self.db.execute("SELECT COALESCE(SUM(simulated_ticks),0) FROM attempts").fetchone()[0],
                "abandoned_attempts": self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='abandoned'").fetchone()[0],
                "active_checkpoints": active[0], "active_trace_bytes": active[1],
                "cold_checkpoints": self.db.execute("SELECT COUNT(*) FROM checkpoints WHERE active=0").fetchone()[0],
                "recipe_bytes": self.db.execute("SELECT COALESCE(SUM(trace_bytes),0) FROM checkpoints").fetchone()[0],
                "examples": self.db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]}
