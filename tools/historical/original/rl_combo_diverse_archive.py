"""Persistent combo niches and cost-aware exploration of exact replay prefixes.

Behavior descriptors decide retention and allocation, never state equivalence.
Only identical action prefixes in one explicitly named environment are deduped.
This is a search archive, not a calibrated estimate of eventual record score.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from rl_combo_opportunity_value import ComboHistory, _health, evaluate_opportunity

VERSION = "combo-diverse-archive-v2"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bucket(value: float, boundaries: tuple) -> int:
    return sum(value >= boundary for boundary in boundaries)


def _carrier_layout(group: dict) -> list:
    """Broad relay geometry and safe first-hit budget; unknown fields stay unknown."""
    return [[int(body["id"]),
             math.floor(float(body.get("effect_x", body["x"])) / 64) if "x" in body else None,
             math.floor(float(body.get("effect_y", body["y"])) / 64) if "y" in body else None,
             max(0, 1 - int(body["projectile_hits"])) if "projectile_hits" in body else None]
            for body in sorted(group.get("carrier_states", group.get("carriers", ())), key=lambda b: b["id"])]


@dataclass(frozen=True)
class ArchiveConfig:
    max_cells: int = 256
    representatives: int = 2
    recent_window: int = 8
    nominal_cpu_seconds: float = 120.0
    gain_scale: float = 100000.0
    exploration: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for key in ("max_cells", "representatives", "recent_window"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.representatives < 2:
            raise ValueError("at least two representatives protect different histories")
        for key in ("nominal_cpu_seconds", "gain_scale", "exploration"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


def _features(row: dict) -> dict:
    groups = row.get("groups", ())
    dominant = max(groups, key=lambda g: (g["cumulative_members"] *
        (g["successful_clears"] + 1), g.get("carrier_health", 0), -g["chain_id"]), default={})
    chain, clears = int(dominant.get("cumulative_members", 0)), int(dominant.get("successful_clears", 0))
    health = float(dominant.get("carrier_health", 0))
    lifetime = int(dominant.get("youngest_remaining_lifetime", -1))
    remaining = max(0, 990 - int(row.get("qualifying_clear_count", 0)))
    feeders = len(dominant.get("feeder_ids", ()))
    # Fine structure breaks ties *within* a niche. Group ids and birth times
    # record genealogy; bodies with similar geometry can have different futures.
    structure = [[int(g["chain_id"]), int(g.get("first_tick", row["tick"])) // 256,
                  int(g["cumulative_members"]), int(g["successful_clears"]),
                  sorted(int(v) for v in g.get("carrier_ids", ())),
                  int(g.get("youngest_remaining_lifetime", -1)) // 128, _carrier_layout(g)]
                 for g in sorted(groups, key=lambda g: g["chain_id"])]
    inventory = sorted([int(k), int(v)] for k, v in row.get("fresh_inventory_by_color", {}).items())
    level = int(row["level"])
    # Late levels have sharply different, nonmonotonic spawn cadence; merging
    # 91 and 94 would discard the very setup opportunities this archive seeks.
    descriptor = [level if level >= 85 else level // 10,
                  -remaining if remaining <= 25 else _bucket(remaining, (50, 100, 200, 400, 700)),
                  _bucket(chain, (2, 4, 8, 16, 32, 48, 64, 96)),
                  _bucket(clears, (1, 4, 8, 16, 32, 48, 64)),
                  _bucket(health, (.25, .5, .75)),
                  _bucket(lifetime, (0, 32, 128, 512, 2048)), min(3, feeders)]
    return dict(descriptor=descriptor, structure=structure, inventory=inventory,
                chain=chain, successful_clears=clears, carrier_health=health,
                youngest_remaining_lifetime=lifetime, visible_feeders=feeders,
                qualifications_remaining=remaining, scoring_clears_remaining=max(0, remaining - 1),
                live_groups=len(groups))


def describe_state(observation: dict, history: ComboHistory) -> dict:
    """History-based archive row using events only through observation.tick."""
    value = evaluate_opportunity(observation, history)
    bodies = {int(b["id"]): b for b in observation.get("bodies", ()) if b.get("kind") == "piece"}
    groups = []
    for detail in value.group_details:
        carriers = [bodies[ident] for ident in detail["carrier_ids"]]
        past = history.groups.get(detail["chain_id"], {})
        groups.append({**detail, "first_tick": int(past.get("first_tick", observation["tick"])),
            "carrier_health": max(map(_health, carriers), default=0),
            "carrier_states": carriers,
            "youngest_remaining_lifetime": max((int(b.get("remaining_lifetime", -1))
                                                for b in carriers), default=-1)})
    inventory = Counter(int(b["color"]) for b in bodies.values() if not b.get("chain_id")
        and _health(b) > 0 and b.get("lifecycle") in ("scripted_falling", "dynamic_fresh", "fresh", "falling"))
    row = dict(tick=int(observation["tick"]), level=int(observation["level"]),
        score=value.realized_score, gauge=int(observation["gauge"]), terminal=value.terminal,
        qualifying_clear_count=int(observation.get("qualifying_clear_count", 0)), groups=groups,
        fresh_inventory_by_color=dict(inventory), components={"potential": value.potential},
        ranks={"extend": value.extension_value, "open": value.opening_value,
               "sustain": max((g["cumulative_members"] * g["carrier_health"] for g in groups), default=0)})
    row["features"] = _features(row)
    row["descriptor"] = row["features"]["descriptor"]
    return row


def _distance(a: dict, b: dict) -> float:
    """Genealogy/inventory difference, deliberately insensitive to pixel jitter."""
    left = {r[0]: r for r in a["features"]["structure"]}
    right = {r[0]: r for r in b["features"]["structure"]}
    distance = len(left.keys() ^ right.keys()) * 4.0
    for gid in left.keys() & right.keys():
        x, y = left[gid], right[gid]
        distance += min(4, abs(x[1] - y[1])) + min(4, abs(x[2] - y[2]) / 4)
        distance += min(4, abs(x[3] - y[3]) / 4) + min(2, abs(x[5] - y[5]))
        distance += min(4, len(set(x[4]) ^ set(y[4])))
        gx, gy = {r[0]: r for r in x[6]}, {r[0]: r for r in y[6]}
        for ident in gx.keys() & gy.keys():
            px, py = gx[ident], gy[ident]
            distance += min(4, sum(abs(px[i] - py[i]) for i in (1, 2)
                                  if px[i] is not None and py[i] is not None))
            if px[3] is not None and py[3] is not None:
                distance += 2 * abs(px[3] - py[3])
    ix, iy = dict(a["features"]["inventory"]), dict(b["features"]["inventory"])
    return distance + sum(min(3, abs(ix.get(k, 0) - iy.get(k, 0))) for k in ix.keys() | iy.keys())


def _stats() -> dict:
    return dict(visits=0, completed=0, in_flight=0, cpu_seconds=0.0, admissions=0, score_gain=0.0, recent=[])


class DiverseArchive:
    """A single-coordinator archive; callers serialize mutations across workers.

    ``source`` in returned entries is an immutable, copied .u32le prefix and
    ``source_path`` identifies the original trace. Changing a source's future
    tail cannot reset visits. Pass a seed/runtime namespace as environment_key.
    """

    def __init__(self, directory: Path, config: ArchiveConfig | None = None, *, environment_key: str):
        if not environment_key:
            raise ValueError("an explicit environment identity is required")
        self.directory = Path(directory).resolve()
        self.path = self.directory / "archive.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("version") != VERSION or self.state.get("environment_key") != environment_key:
                raise ValueError("archive version/environment identity mismatch")
            saved = ArchiveConfig(**self.state["config"])
            if config is not None and config != saved:
                raise ValueError("archive config mismatch")
            self.config = saved
            self.state.setdefault("receipts", {})
            self.state.setdefault("job_receipts", {})
            self.state.setdefault("admissions_by_producer", {})
            for identity in self.state["pending"]:
                if identity in self.state["entries"]:
                    self.state["pending"][identity] = self.state["entries"][identity]
        else:
            self.config = config or ArchiveConfig()
            self.state = dict(version=VERSION, environment_key=environment_key,
                config=asdict(self.config), selection_index=0, offered=0, entries={}, emitters={}, pending={}, receipts={},
                job_receipts={}, admissions_by_producer={})

    @property
    def entries(self) -> dict:
        return self.state["entries"]

    def save(self):
        # Small receipts survive niche eviction. They prevent a later discovery
        # of an identical prefix from resetting its exploration cost or visits.
        for identity, entry in {**self.state["pending"], **self.entries}.items():
            self.state["receipts"][identity] = {key: entry[key] for key in _stats()}
        _atomic(self.path, (json.dumps(self.state, sort_keys=True, allow_nan=False) + "\n").encode())
        # Delete only after the new manifest commits. A crash beforehand leaves
        # recoverable extra files instead of removing an old manifest's prefix.
        live = self.entries.keys() | self.state["pending"].keys()
        for path in (self.directory / "prefixes").glob("*.u32le"):
            if path.stem not in live:
                path.unlink()

    def _discard(self, identity: str):
        entry = self.entries.pop(identity, None)
        if entry is not None:
            self.state["receipts"][identity] = {key: entry[key] for key in _stats()}

    def _tie(self, key: str) -> str:
        return digest(f"{self.config.seed}:{self.state['selection_index']}:{key}".encode())

    @staticmethod
    def _quality(entry: dict) -> tuple:
        return (entry["score"] + entry["potential"], entry["score"], entry["features"]["carrier_health"])

    def offer(self, source: Path, observation: dict, history: ComboHistory, **kwargs) -> dict:
        return self.offer_root(source, describe_state(observation, history), **kwargs)

    def offer_root(self, source: Path, row: dict, *, parent: str | None = None,
                   source_bytes: bytes | None = None, metadata: dict | None = None) -> dict:
        source = Path(source).resolve()
        payload = source.read_bytes() if source_bytes is None else source_bytes
        tick = row["tick"]
        if type(tick) is not int or tick < 0 or len(payload) % 4 or tick > len(payload) // 4:
            raise ValueError("root tick must lie in an aligned source trace")
        prefix, identity = payload[:tick * 4], digest(payload[:tick * 4])
        if row.get("prefix_sha256", identity) != identity:
            raise ValueError("root prefix identity mismatch")
        if not 1 <= int(row["level"]) <= 100 or int(row.get("qualifying_clear_count", 0)) < 0:
            raise ValueError("invalid game phase or qualification count")
        if (row.get("terminal") or row.get("terminated") or row.get("truncated") or
                int(row["level"]) >= 100 or int(row.get("qualifying_clear_count", 0)) >= 990
                or int(row.get("gauge", 0)) <= 0):
            return dict(admitted=False, reason="terminal", prefix_sha256=identity, entry=None)
        if int(row.get("qualifying_clear_count", 0)) == 989:
            # Qualification 990 records the result before the actor payout pass.
            # These prefixes can be cheaply finished outside the search archive.
            return dict(admitted=False, reason="no_scoring_budget", prefix_sha256=identity, entry=None)
        if identity in self.entries:
            return dict(admitted=False, reason="duplicate_prefix", prefix_sha256=identity, entry=self.entries[identity])
        potential = float(row.get("components", {}).get("potential", row.get("potential", 0)))
        if not math.isfinite(potential) or potential < 0 or int(row["score"]) < 0:
            raise ValueError("score and potential must be finite and nonnegative")
        self.state["offered"] += 1
        features = _features(row)
        entry = dict(prefix_sha256=identity, source_path=str(source), tick=tick, level=int(row["level"]),
            score=int(row["score"]), potential=potential, gauge=int(row.get("gauge", 0)), features=features,
            descriptor=features["descriptor"], parent=parent, added_index=self.state["offered"],
            ranks=dict(row.get("ranks", {})), metadata=metadata or {},
            **self.state["receipts"].get(identity, _stats()))
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in entry["ranks"].values()):
            raise ValueError("ranks must be finite and nonnegative")
        cell = [e for e in self.entries.values() if e["descriptor"] == entry["descriptor"]] + [entry]
        leader = max(cell, key=lambda e: (self._quality(e), e["prefix_sha256"]))
        keep = [leader]
        while len(keep) < min(len(cell), self.config.representatives):
            keep.append(max((e for e in cell if e not in keep), key=lambda e: (
                min(_distance(e, k) for k in keep), self._quality(e), e["prefix_sha256"])))
        if entry not in keep:
            return dict(admitted=False, reason="niche_dominated", prefix_sha256=identity, entry=None)
        path = self.directory / "prefixes" / f"{identity}.u32le"
        if path.exists() and digest(path.read_bytes()) != identity:
            raise ValueError("archived prefix integrity failure")
        if not path.exists():
            _atomic(path, prefix)
        entry["source"] = entry["prefix_path"] = str(path)
        removed = [e["prefix_sha256"] for e in cell if e not in keep]
        for key in removed:
            self._discard(key)
        self.entries[identity] = entry
        if identity in self.state["pending"]:
            self.state["pending"][identity] = entry
        # Bound cells without a global score cutoff: first retire the most
        # repeatedly unproductive cell in the most represented game phase.
        cells = {}
        for e in self.entries.values():
            cells.setdefault(tuple(e["descriptor"]), []).append(e)
        if len(cells) > self.config.max_cells:
            phases = Counter(c[0] for c in cells)
            eligible = [(c, es) for c, es in cells.items() if c != tuple(entry["descriptor"])]
            victim, rows = max(eligible, key=lambda item: (
                phases[item[0][0]], sum(e["visits"] for e in item[1]) /
                (1 + sum(e["admissions"] + e["score_gain"] / self.config.gain_scale for e in item[1])),
                -min(e["added_index"] for e in item[1]), item[0]))
            for e in rows:
                self._discard(e["prefix_sha256"])
                removed.append(e["prefix_sha256"])
        if (metadata or {}).get("producer"):
            producer = str(Path(metadata["producer"]).resolve())
            admitted = self.state["admissions_by_producer"].setdefault(producer, [])
            if identity not in admitted:
                admitted.append(identity)
        self.save()
        return dict(admitted=True, reason="new_niche" if len(cell) == 1 else "niche_representative",
                    prefix_sha256=identity, removed=removed, entry=entry)

    def _priority(self, stats: dict, total: int) -> float:
        recent = stats["recent"]
        cost = sum(r["cpu_seconds"] for r in recent)
        reward = sum(r["reward"] for r in recent)
        average_cost = cost / len(recent) if recent else self.config.nominal_cpu_seconds
        scale = max(1, average_cost / self.config.nominal_cpu_seconds)
        exploitation = reward / max(1, cost / self.config.nominal_cpu_seconds)
        exploration = self.config.exploration * math.sqrt(math.log(total + 2) / (stats["visits"] + 1)) / math.sqrt(scale)
        return exploitation + exploration

    def select(self, profile: str | None = None) -> dict:
        candidates = [e for e in self.entries.values() if e["features"]["scoring_clears_remaining"] > 0
                      and (profile is None or e["ranks"].get(profile, 0) > 0)]
        if not candidates:
            raise ValueError("archive has no matching nonterminal parents")
        cells = Counter()
        phases = Counter()
        for e in self.entries.values():
            cells[tuple(e["descriptor"])] += e["visits"]
            phases[e["descriptor"][0]] += e["visits"]
        unvisited = [e for e in candidates if not e["visits"]]
        if unvisited:
            chosen = min(unvisited, key=lambda e: (phases[e["descriptor"][0]],
                cells[tuple(e["descriptor"])], self._tie(e["prefix_sha256"])))
        else:
            total = sum(e["visits"] for e in self.entries.values())
            chosen = max(candidates, key=lambda e: (self._priority(e, total) +
                1 / math.sqrt(1 + cells[tuple(e["descriptor"])]), self._tie(e["prefix_sha256"])))
        payload = Path(chosen["prefix_path"]).read_bytes()
        if len(payload) != chosen["tick"] * 4 or digest(payload) != chosen["prefix_sha256"]:
            raise ValueError("archived prefix integrity failure")
        chosen["visits"] += 1
        chosen["in_flight"] += 1
        self.state["pending"][chosen["prefix_sha256"]] = chosen
        self.state["selection_index"] += 1
        self.save()
        return json.loads(json.dumps(chosen))

    def select_emitter(self, names=("beam", "options", "nrpa")) -> str:
        if not names or len(set(names)) != len(names):
            raise ValueError("emitters must be a nonempty unique collection")
        stats = self.state["emitters"]
        for name in names:
            stats.setdefault(name, _stats())
        unvisited = [name for name in names if not stats[name]["visits"]]
        chosen = min(unvisited, key=self._tie) if unvisited else max(names, key=lambda name: (
            self._priority(stats[name], sum(s["visits"] for s in stats.values())), self._tie(name)))
        stats[chosen]["visits"] += 1
        stats[chosen]["in_flight"] += 1
        self.state["selection_index"] += 1
        self.save()
        return chosen

    def reserve_root(self, source: Path, row: dict, *, emitter: str) -> dict:
        """Reserve a fixed benchmark root even if its niche representative moved.

        A pending control is not admitted to the exploration population and
        earns no novelty reward. Its receipt survives normal niche eviction.
        """
        if not isinstance(emitter, str) or not emitter:
            raise ValueError("control emitter must be a nonempty name")
        source = Path(source).resolve()
        payload = source.read_bytes()
        tick = row["tick"]
        if type(tick) is not int or tick < 0 or len(payload) % 4 or tick > len(payload) // 4:
            raise ValueError("root tick must lie in an aligned source trace")
        if not 1 <= int(row["level"]) <= 100 or int(row.get("qualifying_clear_count", 0)) < 0:
            raise ValueError("invalid game phase or qualification count")
        if (row.get("terminal") or row.get("terminated") or row.get("truncated") or
                int(row["level"]) >= 100 or int(row.get("qualifying_clear_count", 0)) >= 989
                or int(row.get("gauge", 0)) <= 0):
            raise ValueError("control root has no search budget")
        prefix, identity = payload[:tick*4], digest(payload[:tick*4])
        if row.get("prefix_sha256", identity) != identity:
            raise ValueError("root prefix identity mismatch")
        potential = float(row.get("components", {}).get("potential", row.get("potential", 0)))
        ranks = dict(row.get("ranks", {}))
        if not math.isfinite(potential) or potential < 0 or int(row["score"]) < 0:
            raise ValueError("score and potential must be finite and nonnegative")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in ranks.values()):
            raise ValueError("ranks must be finite and nonnegative")
        entry = self.entries.get(identity, self.state["pending"].get(identity))
        path = Path(entry["prefix_path"]) if entry else self.directory/"prefixes"/f"{identity}.u32le"
        if path.exists() and (len(path.read_bytes()) != tick*4 or digest(path.read_bytes()) != identity):
            raise ValueError("control prefix integrity failure")
        if not path.exists():
            _atomic(path, prefix)
        if entry is None:
            features = _features(row)
            entry = dict(prefix_sha256=identity, source_path=str(source), source=str(path),
                prefix_path=str(path), tick=tick, level=int(row["level"]), score=int(row["score"]),
                potential=potential, gauge=int(row["gauge"]),
                features=features, descriptor=features["descriptor"], parent=None,
                added_index=self.state["offered"], ranks=ranks,
                metadata={"fixed_control":True}, **self.state["receipts"].get(identity,_stats()))
        entry["visits"] += 1
        entry["in_flight"] += 1
        self.state["pending"][identity] = entry
        stats = self.state["emitters"].setdefault(emitter,_stats())
        stats["visits"] += 1
        stats["in_flight"] += 1
        self.save()
        return json.loads(json.dumps(entry))

    def record_result(self, prefix_sha256: str, cpu_seconds: float, *, score_gain: float = 0,
                      admissions: int = 0, emitter: str | None = None, receipt_id: str | None = None):
        if (not math.isfinite(cpu_seconds) or cpu_seconds < 0 or not math.isfinite(score_gain)
                or score_gain < 0 or type(admissions) is not int or admissions < 0):
            raise ValueError("feedback must have finite nonnegative cost, gain and admission count")
        if receipt_id is not None:
            if not isinstance(receipt_id, str) or not receipt_id:
                raise ValueError("receipt id must be a nonempty string")
            previous = self.state["job_receipts"].get(receipt_id)
            if previous is not None:
                if previous["prefix_sha256"] != prefix_sha256 or previous["emitter"] != emitter:
                    raise ValueError("receipt identity was reused for a different job")
                return {**previous, "applied": False}
        entry = self.entries.get(prefix_sha256, self.state["pending"].get(prefix_sha256))
        if entry is None:
            raise KeyError("feedback parent was not retained or selected")
        # Admission rewards are capped so geometry jitter cannot swamp verified gains.
        reward = min(2, admissions) * .25 + min(10, score_gain / self.config.gain_scale)
        targets = [entry]
        if emitter is not None:
            targets.append(self.state["emitters"].setdefault(emitter, _stats()))
        for stats in targets:
            stats["completed"] += 1
            stats["in_flight"] = max(0, stats["in_flight"] - 1)
            stats["cpu_seconds"] += cpu_seconds
            stats["admissions"] += admissions
            stats["score_gain"] += score_gain
            stats["recent"] = (stats["recent"] + [dict(cpu_seconds=cpu_seconds, reward=reward)])[-self.config.recent_window:]
        # A selected parent can be evicted while its worker is running. Preserve
        # its receipt until feedback arrives without retaining its niche forever.
        if not entry["in_flight"]:
            self.state["pending"].pop(prefix_sha256, None)
        self.state["receipts"][prefix_sha256] = {key: entry[key] for key in _stats()}
        receipt = dict(prefix_sha256=prefix_sha256, emitter=emitter, cpu_seconds=cpu_seconds,
                       score_gain=score_gain, admissions=admissions)
        if receipt_id is not None:
            self.state["job_receipts"][receipt_id] = receipt
        self.save()
        return {**receipt, "applied": True}
