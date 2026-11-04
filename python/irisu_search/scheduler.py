"""Wall-time lane scheduling and seed tournament with persistent exploration."""
from __future__ import annotations

import hashlib
import math
import random

from .proposals import OPERATORS, potential


def random_for(*parts):
    return random.Random(int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little"))


def seed_ranking(store):
    opportunities = {}
    for cp in store.checkpoints():
        s = cp["summary"]
        if not s.get("terminated") and not s.get("truncated"):
            opportunities[cp["seed"]] = max(opportunities.get(cp["seed"], 0), potential(s))
    # A shortlist is a search allocation, not a claim of inherent seed quality.
    return sorted(store.seeds(), key=lambda s: (
        opportunities.get(s["seed"], 0), math.log1p(max(0, s.get("best_score", 0)))
    ), reverse=True)


def rebalance(store, config):
    ranked = seed_ranking(store)
    experienced = [s for s in ranked if s.get("search_seconds", 0) >= config.minimum_seed_seconds
                   and s.get("attempts", 0) >= config.attempts_per_seed]
    deep = [s for s in experienced[:config.promoted_seeds] if s.get("search_seconds", 0) >= config.finalist_seconds]
    finalists = {s["seed"] for s in deep[:config.finalists]}
    promoted = {s["seed"] for s in experienced[:config.promoted_seeds]}
    for seed in ranked:
        status = ("discovery" if seed not in experienced else "finalist" if seed["seed"] in finalists
                  else "promoted" if seed["seed"] in promoted else "dormant")
        if seed.get("status") != status:
            store.update_seed(seed["seed"], status=status)


def choose_lane(store, config):
    spent = {lane: 0. for lane in config.lanes}
    for lane, cost in store.lane_costs().items():
        if lane in spent:
            spent[lane] += cost
    total = sum(spent.values())
    return max(config.lanes, key=lambda lane: config.lanes[lane] * (total + config.job_seconds) - spent[lane])


def next_job(store, config, guide=None):
    sequence = store.get_meta("schedule_sequence", 0)
    store.set_meta("schedule_sequence", sequence + 1)
    rng = random_for(config.seed_generator, sequence)
    rebalance(store, config)
    seeds = store.seeds()
    lane = choose_lane(store, config)
    bootstrap = [s for s in seeds if s.get("attempts", 0) < config.attempts_per_seed]
    live = [cp for cp in store.checkpoints() if not cp["summary"].get("terminated")
            and not cp["summary"].get("truncated") and cp["summary"]["tick"] < config.maximum_ticks]
    if lane == "seed" or not live:
        if bootstrap:
            seed = min(bootstrap, key=lambda s: (s.get("attempts", 0), s.get("search_seconds", 0)))
        elif not config.discover_new_seeds:
            seed = min(seeds, key=lambda s: (s.get("attempts", 0), s.get("search_seconds", 0)))
        else:
            known = {s["seed"] for s in seeds}
            value = rng.getrandbits(32)
            while value in known:
                value = rng.getrandbits(32)
            store.ensure_seed(value)
            seed = {"seed": value, "attempts": 0}
        return store.enqueue("bootstrap", seed["seed"], payload={
            "search_seed": rng.getrandbits(32), "attempt": seed.get("attempts", 0),
            "horizon": config.bootstrap_ticks, "operator": OPERATORS[sequence % len(OPERATORS)],
        }, lane="seed", dedupe_key=f"bootstrap:{seed['seed']}:{seed.get('attempts',0)}:{sequence}")
    live_seeds = {cp["seed"] for cp in live}
    ranked = [s for s in seed_ranking(store) if s["seed"] in live_seeds]
    if lane == "wildcard":
        seed = rng.choice(ranked)
    else:
        under = [s for s in ranked if s.get("search_seconds", 0) < config.minimum_seed_seconds]
        pool = under or [s for s in ranked if s.get("status") in ("promoted", "finalist") or s.get("control")]
        if not pool:
            pool = ranked[:config.promoted_seeds]
        if under:
            seed = min(pool, key=lambda s: s.get("search_seconds", 0))
        elif guide and guide.head_counts[3] >= 32 and rng.random() >= config.exploration_fraction:
            def benefit(seed_row):
                return max((guide.predict(cp["summary"], {"operator": "allocation", "horizon": config.horizon,
                            "budget_seconds": config.job_seconds})["compute_gain"]
                            for cp in live if cp["seed"] == seed_row["seed"]), default=0)
            seed = max(pool, key=benefit)
        else:
            seed = rng.choice(pool)
    candidates = [cp for cp in live if cp["seed"] == seed["seed"]]
    uses = store.checkpoint_uses()
    if lane == "wildcard":
        checkpoint = rng.choice(candidates)
    else:
        candidates.sort(key=lambda cp: (potential(cp["summary"]), cp["summary"].get("score", 0)), reverse=True)
        # Include a random checkpoint and favor unexplored descendants. Historical
        # high score alone cannot erase an unusual live group configuration.
        pool = candidates[:max(4, config.survivors * 2)]
        if rng.random() < config.exploration_fraction:
            checkpoint = rng.choice(candidates)
        else:
            checkpoint = min(pool, key=lambda cp: uses.get(cp["id"], 0))
    horizon = config.horizon
    if lane == "probe":
        horizon = max(config.bootstrap_ticks, horizon * 4)
        if seed.get("status") == "finalist":
            horizon = max(horizon, config.bootstrap_ticks * 4)
    elif lane == "wildcard":
        horizon *= rng.choice((1, 2, 4))
    horizon = min(horizon, 100000, config.maximum_ticks - checkpoint["summary"]["tick"])
    operator = "repair" if lane == "probe" else rng.choice(OPERATORS)
    return store.enqueue("probe" if lane == "probe" else "expand", seed["seed"],
                         checkpoint_id=checkpoint["id"], lane=lane, payload={
                             "search_seed": rng.getrandbits(32), "operator": operator,
                             "horizon": horizon,
                         }, dedupe_key=f"schedule:{sequence}")
