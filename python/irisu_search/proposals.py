"""Mechanics-aware interventions, archive diversity, and exploration floors."""
from __future__ import annotations

import math
import random

OPERATORS = ("recruit", "preserve", "handoff", "payout", "prepare", "timing", "repair")


def body_key(body):
    """Physical ordering only: IDs are join keys, never ranking features."""
    return (body.get("color", -1), str(body.get("kind", "piece")),
            str(body.get("shape", "")), body.get("x", 0), body.get("y", 0),
            body.get("vx", 0), body.get("vy", 0), body.get("size", 0),
            body.get("hits", 0), body.get("rot_timer", 0), body.get("lifetime", 100000),
            bool(body.get("pending_clear")), bool(body.get("pending_delete")))


def encode(kind, x=0, y=0):
    if type(kind) is not int or not 0 <= kind <= 3 or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("shot kind and coordinates must be finite legal values")
    return int(kind) | (max(0, min(640, round(x))) << 2) | (max(0, min(480, round(y))) << 12)


def live_groups(summary):
    bodies = {b["id"]: b for b in summary.get("bodies", [])}
    result = []
    for group in summary.get("groups", []):
        members = sorted((bodies[i] for i in group.get("members", []) if i in bodies
                          and not bodies[i].get("pending_delete") and not bodies[i].get("pending_clear")
                          and bodies[i].get("lifetime", 100000) > 1 and bodies[i].get("hits", 0) < 2),
                         key=body_key)
        if members:
            result.append((group, members))
    return result


def group_key(pair):
    group, members = pair
    return (group["chain"] * (group["num"] + 1), group["chain"], group["num"],
            len(members), tuple(body_key(body) for body in members))


def target_group(summary):
    """The same physically canonical target is used for features and labels."""
    return max(live_groups(summary), key=group_key, default=None)


def potential(summary):
    """An exploration proxy, never a score or a terminal admission criterion."""
    terms = []
    for group, members in live_groups(summary):
        accessible = math.fsum(sorted(1 / (1 + b.get("hits", 0) + (4 if b.get("rot_timer", 0) else 0)) for b in members))
        terms.append(math.log1p(group["chain"] * (group["num"] + 1)) * min(accessible, 3.))
    return math.fsum(sorted(terms))


def exploration_key(summary):
    return (potential(summary), math.log1p(max(0, summary.get("score", 0))),
            min(summary.get("gauge", 0), 20000) / 20000)


def generate(summary, count, search_seed, operator="recruit", baseline=(), horizon=384):
    """Return variable-length prefixes followed by board-responsive continuation.

    Grouped pieces with one existing hit are not direct-shot targets. Pairing
    proposals recruit ungrouped pieces; two existing groups are never merged.
    """
    if type(count) is not int or count < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("candidate count and horizon must be positive integers")
    rng = random.Random(search_seed)
    mode = {"recruit": 1, "preserve": 2, "handoff": 2, "payout": 3,
            "prepare": 1, "timing": 1, "repair": 1}.get(operator, 1)
    output = []
    seen = set()

    def add(words, candidate_mode=mode, label=operator):
        words = list(words[:horizon])
        identity = (candidate_mode, tuple(words))
        if identity not in seen:
            seen.add(identity)
            output.append({"mode": candidate_mode, "search_seed": rng.getrandbits(32),
                           "words": list(words), "operator": label, "horizon": horizon})

    add(list(baseline[:horizon]), 0, "incumbent")
    add([], mode, operator)
    add([], 0, "wait")
    bodies = sorted((b for b in summary.get("bodies", []) if not b.get("pending_delete")
              and not b.get("pending_clear") and not b.get("special")
              and b.get("kind", "piece") not in ("projectile", 1)
              and b.get("lifetime", 100000) > 1), key=body_key)
    fresh = [b for b in bodies if not b.get("chain_id") and b.get("color", -1) >= 0]
    groups = sorted(live_groups(summary), key=group_key, reverse=True)
    # Reserve part of the proposal budget for the requested carrier
    # intervention before generic pair fanout can consume every slot.
    carrier_words = []
    if operator in ("preserve", "handoff", "payout", "repair"):
        for group, members in groups:
            for body in sorted(members, key=lambda b: (b.get("hits", 0), body_key(b))):
                if body.get("hits", 0) >= 1:
                    continue
                direction = -.35 if body["x"] < 290 else .35
                x = body["x"] + (direction if operator == "payout" else 0) * body.get("size", 30)
                carrier_words.append([encode(1 if operator == "preserve" else 2, x,
                                             body["y"] + .55 * body.get("size", 30)), 0])
    reserved = min(len(carrier_words), max(1, math.ceil((count - 1) / 4)), count - 1)
    if reserved:
        output = output[:max(1, count - reserved)]
        for words in carrier_words[:reserved]:
            add(words)
    pairs = [(source, target) for group, members in groups for target in members for source in fresh
             if source["color"] == target["color"] and source["id"] != target["id"]]
    pairs += [(a, b) for i, a in enumerate(fresh) for b in fresh[i + 1:]
              if a["color"] == b["color"]]
    rng.shuffle(pairs)
    for source, target in pairs:
        if len(output) >= count:
            break
        # A projectile placed below/alongside the fresh source redirects it
        # toward the carrier; exact physics, not this geometry, judges success.
        dx = target["x"] - source["x"]
        offsets = (-.3, 0., .3) if abs(dx) < 4 else (-math.copysign(.3, dx),)
        for offset in offsets:
            for strength in (1, 2):
                if len(output) >= count:
                    break
                delay = rng.choice((0, 2, 8, 16, 32)) if operator in ("prepare", "timing") else 0
                x = source["x"] + offset * source.get("size", 30)
                y = source["y"] + .55 * source.get("size", 30)
                add([0] * delay + [encode(strength, x, y), 0])
    for words in carrier_words[reserved:]:
        if len(output) >= count:
            break
        add(words)
    # Distinct controller seeds diversify reactive continuations even when no
    # safe geometry proposal exists. No synthetic/random untested state scores.
    while len(output) < count:
        output.append({"mode": rng.choice((1, 2, 3)), "search_seed": rng.getrandbits(32),
                       "words": [0] * rng.randrange(0, min(33, horizon)),
                       "operator": operator, "horizon": horizon})
    return output[:count]


def select_candidates(summary, candidates, count, guide=None, exploration_fraction=.25, seed=0):
    if type(count) is not int or count < 1:
        raise ValueError("candidate count must be positive")
    if not math.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
        raise ValueError("exploration fraction must be in [0,1]")
    if len(candidates) <= count:
        return candidates
    if count == 1:
        return candidates[:1]
    rng = random.Random(seed)
    # Incumbent and genuine random exploration cannot be pruned by a model.
    fixed = candidates[:1]
    remaining = candidates[1:]
    explore = min(len(remaining), max(1, math.ceil((count - 1) * exploration_fraction)))
    random_part = rng.sample(remaining, explore)
    rest = [x for x in remaining if all(x is not y for y in random_part)]
    if guide:
        rest.sort(key=lambda x: guide.predict(summary, x)["priority"], reverse=True)
    return fixed + rest[:max(0, count - len(fixed) - len(random_part))] + random_part


def select_survivors(results, count, guide=None, *, horizon=384, budget_seconds=30):
    """Keep heuristic, realized-score and learned-value exploration lanes.

    Learned return estimates concern additional payout under reactive
    recruitment. They may retain an unfinished state, never score a terminal
    record. A protected heuristic live state survives even a hostile model.
    """
    if type(count) is not int or count < 1:
        raise ValueError("survivor count must be positive")
    if not results:
        return []
    terminals = sorted((r for r in results if r["summary"].get("terminated")
                        and not r["summary"].get("truncated")),
                       key=lambda r: r["summary"].get("canonical_score", 0), reverse=True)
    live = [r for r in results if not r["summary"].get("terminated")
            and not r["summary"].get("truncated")]
    by_potential = sorted(live, key=lambda r: exploration_key(r["summary"]), reverse=True)
    by_score = sorted(live, key=lambda r: r["summary"].get("score", 0), reverse=True)
    selected, seen = [], set()
    # Terminal candidates are independently published and verified by the
    # coordinator; even a one-slot beam must keep making progress when live
    # alternatives remain.
    streams = [by_potential[:1], terminals[:1], by_score[:1], by_potential]
    counts = getattr(guide, "head_counts", (0, 0, 0, 0))
    if guide and count >= 2 and live and counts[0] >= 32:
        continuation = {"mode": 1, "operator": "recruit", "words": [],
                        "horizon": horizon, "budget_seconds": budget_seconds}

        def predicted_total(row):
            gain = guide.predict(row["summary"], continuation)["predicted_gain"]
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain):
                raise ValueError("neural leaf value must be finite")
            return row["summary"].get("score", 0) + gain

        by_value = sorted(live, key=predicted_total, reverse=True)
        # Completed results have their own publication/verification path. Use
        # scarce live beam slots for promising continuations first.
        streams = [by_potential[:1], by_value[:1], by_score[:1],
                   terminals[:1], by_potential]
    for stream in streams:
        for row in stream:
            identity = row.get("handle", id(row))
            if identity not in seen:
                selected.append(row)
                seen.add(identity)
            if len(selected) >= count:
                return selected
    return selected
