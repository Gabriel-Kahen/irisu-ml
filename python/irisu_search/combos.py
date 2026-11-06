"""Combo continuity and replay neighborhoods derived from normal-mode rules.

The native score path uses (num + 1) / 2 * chain per scored actor, with
level/size factors. Chain counts recruitment, num counts qualification; neither
is the number of surviving carriers. These descriptors are search heuristics,
not terminal scores. See clone/core/{simulator,normal_rules}.cpp.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .proposals import body_key, encode, generate, live_groups


def _risk(body):
    margin = min(body.get("x", 320) - 10, 630 - body.get("x", 320),
                 440 - body.get("y", 200))
    edge = max(0., min(1., 1 - margin / 40))
    lifetime = max(0., min(1., 1 - body.get("lifetime", 100000) / 256))
    falling = (max(0., min(1., 1 - (440 - body.get("y", 200)) /
               (128 * body["vy"]))) if body.get("vy", 0) > 0 else 0.)
    return max(edge, lifetime, falling, .65 if body.get("hits", 0) else 0.,
               .75 if body.get("rot_timer", 0) else 0.)


def _dominant(summary):
    return max(live_groups(summary), key=lambda p: (
        p[0]["chain"] * (p[0]["num"] + 1),
        sum(_risk(body) < .5 for body in p[1]), p[0]["chain"],
        tuple(body_key(body) for body in p[1])), default=None)


def descriptor(summary):
    """Describe the strongest viable group without using IDs as features."""
    pair = _dominant(summary)
    if pair is None:
        return {"chain": 0, "num": 0, "carriers": 0, "safe_carriers": 0,
                "one_hit_carriers": 0, "rot_carriers": 0,
                "last_carrier_risk": 1., "recruit_distance": None,
                "group_value": 0, "color": -1, "x": 0., "y": 0.}
    group, members = pair
    fresh = [body for body in summary.get("bodies", [])
             if not body.get("chain_id") and not body.get("special")
             and body.get("kind", "piece") not in ("projectile", 1)
             and not body.get("pending_clear") and not body.get("pending_delete")
             and body.get("lifetime", 100000) > 1]
    distances = [math.hypot(a.get("x", 0) - b.get("x", 0),
                            a.get("y", 0) - b.get("y", 0))
                 for a in members for b in fresh
                 if a.get("color", -1) >= 0 and a.get("color") == b.get("color")]
    count = len(members)
    return {"chain": group["chain"], "num": group["num"], "carriers": count,
            "safe_carriers": sum(_risk(body) < .5 for body in members),
            "one_hit_carriers": sum(body.get("hits", 0) == 1 for body in members),
            "rot_carriers": sum(bool(body.get("rot_timer", 0)) for body in members),
            "last_carrier_risk": min(_risk(body) for body in members),
            "recruit_distance": min(distances, default=None),
            "group_value": group["chain"] * (group["num"] + 1),
            "color": members[0].get("color", -1),
            "x": math.fsum(body.get("x", 0) for body in members) / count,
            "y": math.fsum(body.get("y", 0) for body in members) / count}


def combo_key(summary):
    d = descriptor(summary)
    return (d["group_value"], min(3, d["safe_carriers"]), -d["last_carrier_risk"],
            d["chain"], d["num"], summary.get("score", 0), summary.get("gauge", 0))


def outcome(before, after, group_deltas=()):
    """Observed combo growth survives extinction; IDs are only lineage joins."""
    target = _dominant(before)
    result = {"target_group_id": None, "chain_gain": None, "num_gain": None,
              "carrier_survived": None, "extinct": None,
              "realized_score_gain": after.get("score", 0) - before.get("score", 0)}
    if target is None:
        return result
    group = target[0]
    delta = next((row for row in group_deltas if row["id"] == group["id"]), None)
    current = next((row for row in after.get("groups", []) if row["id"] == group["id"]), None)
    survived = any(row["id"] == group["id"] for row, _ in live_groups(after))
    result.update(target_group_id=group["id"], carrier_survived=survived, extinct=not survived)
    if delta is not None:
        result.update(chain_gain=max(0, delta["chain_after"] - group["chain"]),
                      num_gain=max(0, delta["num_after"] - group["num"]))
    elif current is not None:
        result.update(chain_gain=max(0, current["chain"] - group["chain"]),
                      num_gain=max(0, current["num"] - group["num"]))
    return result


def select_frontier(rows, count):
    """Retain live combo, score, safety and geometry alternatives in one batch.

    Words are executed suffixes from the same parent. Callers combining parents
    must provide parent_id so identical suffixes are not mistaken for one recipe.
    """
    if type(count) is not int or count < 1:
        raise ValueError("frontier count must be a positive integer")
    distinct = {}
    for row in rows:
        s = row["summary"]
        if s.get("terminated") or s.get("truncated"):
            continue
        identity = ((row.get("parent_id"), s.get("seed"), tuple(row["words"]))
                    if "words" in row else ("handle", row.get("handle", id(row))))
        previous = distinct.get(identity)
        if previous is None or combo_key(s) > combo_key(previous["summary"]):
            distinct[identity] = row
    ranked = sorted(distinct.values(), key=lambda row: combo_key(row["summary"]), reverse=True)
    if not ranked:
        return []
    selected = []
    seen = set()

    def add(row):
        if id(row) not in seen and len(selected) < count:
            seen.add(id(row))
            selected.append(row)

    add(ranked[0])
    add(max(ranked, key=lambda row: row["summary"].get("score", 0)))
    add(max(ranked, key=lambda row: (
        descriptor(row["summary"])["safe_carriers"],
        -descriptor(row["summary"])["last_carrier_risk"], combo_key(row["summary"]))))
    cells = set()
    for row in selected:
        d = descriptor(row["summary"])
        cells.add((d["chain"].bit_length(), d["num"].bit_length(), d["carriers"],
                   d["color"], int(d["x"] // 64), int(d["y"] // 64)))
    for row in ranked:
        d = descriptor(row["summary"])
        cell = (d["chain"].bit_length(), d["num"].bit_length(), d["carriers"],
                d["color"], int(d["x"] // 64), int(d["y"] // 64))
        if cell not in cells:
            add(row)
            cells.add(cell)
    for row in ranked:
        add(row)
    return selected


def event_ticks(path, replay_sha256, seed, score, limit=64, lead=48, min_chain=3):
    """Select replay roots before combo events from a hash-bound trace receipt.

    Native JSONL traces do not themselves contain the replay hash. Prefer a
    saved trace_replay() receipt; unbound raw traces are rejected.
    """
    if (type(limit) is not int or limit < 1 or type(lead) is not int or lead < 0
            or type(min_chain) is not int or min_chain < 1):
        raise ValueError("invalid combo event selection bounds")
    if (not isinstance(replay_sha256, str) or len(replay_sha256) != 64
            or any(char not in "0123456789abcdef" for char in replay_sha256)):
        raise ValueError("a replay SHA-256 is required")
    path = Path(path)
    with path.open() as stream:
        try:
            first = json.loads(stream.readline())
        except json.JSONDecodeError:
            stream.seek(0)
            first = json.load(stream)
    receipt = first if "output" in first and "replay_sha256" in first else None
    trace = Path(receipt["output"]) if receipt else path
    if receipt and not trace.is_absolute():
        trace = path.parent / trace
    if receipt and receipt["replay_sha256"] != replay_sha256:
        raise ValueError("combo trace belongs to a different replay")
    if receipt and receipt.get("trace_sha256"):
        if hashlib.sha256(trace.read_bytes()).hexdigest() != receipt["trace_sha256"]:
            raise ValueError("combo trace hash mismatch")
    with trace.open() as stream:
        identity = json.loads(stream.readline())
        events = [json.loads(line) for line in stream if line.strip()]
    summary = events.pop() if events else {}
    if not receipt and identity.get("replay_sha256") != replay_sha256:
        raise ValueError("raw combo trace has no matching replay hash; use a trace receipt")
    if (identity.get("kind") != "identity" or identity.get("seed") != seed
            or identity.get("schema") != "irisu-search-genealogy-v1"
            or identity.get("header_score") != score
            or summary.get("kind") != "summary" or summary.get("verified") is not True
            or summary.get("seed") != seed or summary.get("canonical_score") != score
            or summary.get("unassigned_score_events", 0) != 0
            or summary.get("canonical_attributed_score", score) != score
            or identity.get("input_words") != summary.get("tick")):
        raise ValueError("combo genealogy identity or verified result disagrees")
    if receipt and (receipt.get("identity", identity) != identity
                    or receipt.get("summary", summary) != summary):
        raise ValueError("combo trace disagrees with its receipt")
    groups = [g for g in summary.get("groups", []) if g.get("chain", 0) >= min_chain]
    if not groups:
        return []
    best = max(groups, key=lambda g: (g.get("score_canonical", 0), g["chain"], g.get("num", 0)))
    relevant = {g["id"] for g in groups
                if g.get("score_canonical", 0) >= best.get("score_canonical", 0) * .1}
    first_tick = min(g["first_tick"] for g in groups if g["id"] in relevant)
    last_tick = max(g.get("last_tick", g["first_tick"]) for g in groups if g["id"] in relevant)
    selected = {max(0, first_tick - lead)}
    if limit > 1:
        selected.add(max(0, last_tick - lead))
    bands = defaultdict(dict)
    width = max(1, math.ceil((last_tick - first_tick + 1) / min(8, limit)))
    for row in events:
        if row.get("group_id") not in relevant or row.get("tick", 0) >= summary["tick"]:
            continue
        kind = row.get("kind")
        if kind not in ("join", "qualify", "score", "carriers", "ejected", "rotten", "projectile_hit"):
            continue
        value = row.get("chain", 0) * (row.get("num", 0) + 1)
        risk = kind in ("ejected", "rotten") or (kind == "carriers" and row.get("after", 2) <= 1)
        weight = value * (3 if risk else 1) + row.get("delta", 0) / 100
        tick = max(0, row["tick"] - lead)
        band = (row["tick"] - first_tick) // width
        bands[band][tick] = max(weight, bands[band].get(tick, 0))
    ordered = {band: sorted(values, key=lambda tick: (-values[tick], tick))
               for band, values in bands.items()}
    for depth in range(max((len(values) for values in ordered.values()), default=0)):
        for band in sorted(ordered):
            if depth < len(ordered[band]) and len(selected) < limit:
                selected.add(ordered[band][depth])
    return sorted(selected)


def terminal_candidates(summary, count, search_seed, reference, horizon,
                        operator="combo", held=0):
    """Mutate distributed reference shots, then keep their absolute future timing."""
    if type(count) is not int or count < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("candidate count and horizon must be positive integers")
    reference = list(reference[:horizon])
    if not reference:
        rows = generate(summary, count, search_seed, "recruit", (), horizon)
        rows[0]["operator"] = "combo-reference"
        return rows
    rng = random.Random(search_seed)
    output, seen = [], set()

    def add(words, label, edit=None):
        words = list(words[:horizon])
        identity = tuple(words)
        if identity in seen or len(output) >= count:
            return
        seen.add(identity)
        output.append({"mode": 0, "words": words, "operator": label,
                       "search_seed": rng.getrandbits(32), "horizon": horizon,
                       "combo_edit": edit})

    add(reference, "combo-reference")
    edges, edge_kinds, previous = [], {}, held
    for index, word in enumerate(reference):
        kind = word & 3
        if kind & ~previous:
            edges.append(index)
            edge_kinds[index] = kind & ~previous
        previous = kind
    positions = sorted(rng.sample(edges, min(len(edges), max(1, count // 2))))
    rng.shuffle(positions)
    operations = [("delete", 0), ("timing", -8), ("timing", 8), ("strength", 0),
                  ("x", -2), ("x", 2), ("y", -2), ("y", 2),
                  ("timing", -2), ("timing", 2), ("x", -4), ("x", 4),
                  ("y", -4), ("y", 4), ("timing", -1), ("timing", 1),
                  ("x", -12), ("x", 12), ("y", -12), ("y", 12)]
    mutation_limit = count if count < 4 else max(2, count - max(1, count // 4))

    def mutate(index, name, amount):
        changed = reference.copy()
        kind = edge_kinds[index]
        x, y = (reference[index] >> 2) & 1023, (reference[index] >> 12) & 511
        if name in ("delete", "timing", "strength"):
            for bit in (1, 2):
                if not kind & bit:
                    continue
                end = index
                while end < len(changed) and reference[end] & bit:
                    changed[end] &= ~bit
                    end += 1
        if name == "timing":
            destination = index + amount
            if not 0 <= destination < len(changed) or changed[destination] & 3:
                return
            previous = (changed[destination - 1] & 3) if destination else held
            following = (changed[destination + 1] & 3) if destination + 1 < len(changed) else 0
            if (previous | following) & kind:
                return
            changed[destination] = encode(kind, x, y)
        elif name == "strength":
            kind = 2 if kind == 1 else 1
            previous = (changed[index - 1] & 3) if index else held
            if previous & kind or changed[index] & kind:
                return
            changed[index] = encode(kind | (changed[index] & 3), x, y)
        elif name in ("x", "y"):
            changed[index] = encode(reference[index] & 3, x + (amount if name == "x" else 0),
                                    y + (amount if name == "y" else 0))
        add(changed, f"{operator}-{name}", {"kind": name, "tick": index, "amount": amount})

    # Reserve the same fine neighborhood in space and time; larger edits and
    # event coverage still receive the rest of the mutation budget.
    if count >= 8 and edges:
        for name in ("x", "y", "timing"):
            for amount in (-2, 2):
                mutate(edges[0], name, amount)
        if edges[-1] != edges[0]:
            for amount in (-2, 2):
                mutate(edges[-1], "x", amount)
    for cycle in range(len(operations)):
        for offset, index in enumerate(positions):
            if len(output) >= mutation_limit:
                break
            name, amount = operations[(cycle + offset) % len(operations)]
            mutate(index, name, amount)
        if len(output) >= mutation_limit:
            break
    for candidate in generate(summary, max(8, count), search_seed, "handoff", (), horizon):
        prefix = candidate["words"]
        if not candidate["mode"] or not any(word & 3 for word in prefix):
            continue
        if prefix and held & prefix[0] & 3:
            prefix = [0, *prefix]
        prefix = prefix[:horizon]
        add(prefix + reference[len(prefix):], f"{operator}-handoff", {"kind": "handoff", "tick": 0})
        if len(output) >= count:
            break
    return output
