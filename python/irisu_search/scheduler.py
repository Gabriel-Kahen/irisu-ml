"""Wall-time lane scheduling and seed tournament with persistent exploration."""
from __future__ import annotations

import hashlib
import math
import random

from .proposals import OPERATORS, potential


def random_for(*parts):
    return random.Random(int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little"))


def seed_ranking(store, combo=False):
    if combo:
        from .combos import combo_key
    opportunities = {}
    for cp in store.checkpoints():
        s = cp["summary"]
        if not s.get("terminated") and not s.get("truncated"):
            opportunities[cp["seed"]] = max(opportunities.get(cp["seed"], (0,) if combo else 0),
                                                combo_key(s) if combo else potential(s))
    # A shortlist is a search allocation, not a claim of inherent seed quality.
    return sorted(store.seeds(), key=lambda s: (
        opportunities.get(s["seed"], (0,) if combo else 0), math.log1p(max(0, s.get("best_score", 0)))
    ), reverse=True)


def rebalance(store, config):
    ranked = seed_ranking(store, getattr(config, "combo_search", False))
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


def champion_seed(store, config):
    if not config.champion_only:
        return None
    champion = store.champion()
    if not champion:
        raise ValueError("champion-only search requires an imported, verified champion")
    return champion["seed"]


def choose_lane(store, config):
    spent = {lane: 0. for lane in config.lanes}
    for lane, cost in store.lane_costs().items():
        if lane in spent:
            spent[lane] += cost
    shares = dict(config.lanes)
    if config.champion_only:
        spent["combo"] += spent.pop("seed")
        shares["combo"] += shares.pop("seed")
    total = sum(spent.values())
    return max(shares, key=lambda lane: shares[lane] * (total + config.job_seconds) - spent[lane])


def next_job(store, config, guide=None):
    focus = champion_seed(store, config)
    sequence = store.get_meta("schedule_sequence", 0)
    store.set_meta("schedule_sequence", sequence + 1)
    rng = random_for(config.seed_generator, sequence)
    if getattr(config, "combo_search", False):
        return _combo_next_job(store, config, sequence, rng, focus)
    rebalance(store, config)
    seeds = [s for s in store.seeds() if focus is None or s["seed"] == focus]
    lane = choose_lane(store, config)
    if focus is not None and lane == "seed":
        lane = "combo"
    bootstrap = [s for s in seeds if s.get("attempts", 0) < config.attempts_per_seed]
    live = [cp for cp in store.checkpoints(seed=focus) if not cp["summary"].get("terminated")
            and not cp["summary"].get("truncated") and cp["summary"]["tick"] < config.maximum_ticks]
    if lane == "seed" or not live:
        if bootstrap:
            seed = min(bootstrap, key=lambda s: (s.get("attempts", 0), s.get("search_seconds", 0)))
        elif focus is not None or not config.discover_new_seeds:
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
        # Discovery continues while bootstrap attempts are outstanding. Those
        # seeds still need live development; otherwise an imported control is
        # the only eligible seed after everybody's minimum wall-time budget.
        pool = under or [s for s in ranked if s.get("status") in ("discovery", "promoted", "finalist")
                         or s.get("control")]
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
        # Probe depth follows actual investment, independently of tournament
        # classification: outstanding bootstrap attempts must not cap a
        # developed discovery seed at the short probe horizon for hours.
        if seed.get("search_seconds", 0) >= config.finalist_seconds:
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


def _combo_next_job(store, config, sequence, rng, focus=None):
    """Give record combos a budget independent of discovery tournament status."""
    from .combos import combo_key
    rebalance(store, config)
    lane = choose_lane(store, config)
    if focus is not None and lane == "seed":
        lane = "combo"
    seeds = [s for s in store.seeds() if focus is None or s["seed"] == focus]
    live = [row for row in store.checkpoints(seed=focus) if not row["summary"].get("terminated")
            and not row["summary"].get("truncated")
            and row["summary"]["tick"] < config.maximum_ticks]
    champion = store.champion()
    record_seed = champion["seed"] if champion else config.control_seed
    uses = store.checkpoint_uses()

    if lane == "seed" or not live:
        # Complete each small cohort instead of making three breadth-first
        # passes over the full manifest before any seed can qualify.
        manifest = list(dict.fromkeys(store.get_meta("seed_manifest", [])
                                     + [row["seed"] for row in seeds]))
        by_seed = {row["seed"]: row for row in seeds}
        pending = []
        for offset in range(0, len(manifest), config.seed_cohort_size):
            cohort = [by_seed[key] for key in manifest[offset:offset + config.seed_cohort_size]
                      if key in by_seed]
            pending = [row for row in cohort if row["attempts"] < config.attempts_per_seed]
            if pending:
                break
        if pending:
            seed = min(pending, key=lambda row: (row["attempts"], row["search_seconds"], row["seed"]))
        elif focus is None and config.discover_new_seeds:
            known = set(by_seed)
            value = rng.getrandbits(32)
            while value in known:
                value = rng.getrandbits(32)
            store.ensure_seed(value)
            seed = {"seed": value, "attempts": 0}
        else:
            seed = min(seeds, key=lambda row: (row["search_seconds"], row["attempts"]))
        return store.enqueue("bootstrap", seed["seed"], lane="seed", payload={
            "search_seed": rng.getrandbits(32), "attempt": seed["attempts"],
            "horizon": config.bootstrap_ticks, "operator": OPERATORS[sequence % len(OPERATORS)],
        }, dedupe_key=f"combo-bootstrap:{seed['seed']}:{seed['attempts']}:{sequence}")

    record_live = [row for row in live if row["seed"] == record_seed]
    reference_id = (champion["checkpoint_id"] if champion
                    else store.get_meta(f"baseline:{record_seed}"))
    if lane == "combo" and record_live and reference_id:
        root_ids = store.get_meta(f"combo_roots:{record_seed}", [])
        roots = [row for row in store.checkpoint_summaries(root_ids)
                 if not row["summary"].get("terminated") and not row["summary"].get("truncated")
                 and row["summary"]["tick"] < config.maximum_ticks]
        if not roots:
            roots = [row for row in record_live if row["operator"] == "import"] or record_live
        unvisited = [row for row in roots if not uses.get(row["id"], 0)]
        frontier = [row for row in store.combo_frontier(record_seed)
                    if row["summary"]["tick"] < config.maximum_ticks]
        counter_key = f"combo_schedule_sequence:{record_seed}"
        turn = store.get_meta(counter_key, 0)
        store.set_meta(counter_key, turn + 1)
        plan_roots = [row for row in roots if store.combo_plans(row["id"])]
        # Revisit useful branches while coverage is still underway. Otherwise
        # a large event bank delays every second edit until its full first pass.
        exploit = turn % 2 and (frontier or plan_roots)
        use_frontier = exploit and frontier and (not plan_roots or (turn // 2) % 2 == 0)
        if use_frontier:
            selected = min(frontier, key=lambda row: (row["visits"], row["created"]))
            checkpoint = {"id": selected["checkpoint_id"], "summary": selected["summary"]}
            origin_id = selected["origin_id"]
        elif exploit and plan_roots:
            checkpoint = min(plan_roots, key=lambda row: (uses.get(row["id"], 0), row["summary"]["tick"]))
            origin_id = checkpoint["id"]
        elif unvisited:
            checkpoint = min(unvisited, key=lambda row: row["summary"]["tick"])
            origin_id = checkpoint["id"]
        else:
            checkpoint = min(roots, key=lambda row: (uses.get(row["id"], 0), row["summary"]["tick"]))
            origin_id = checkpoint["id"]
        plans = store.combo_plans(checkpoint["id"])
        reference_turn_key = f"combo_reference_sequence:{checkpoint['id']}"
        reference_turn = store.get_meta(reference_turn_key, 0)
        store.set_meta(reference_turn_key, reference_turn + 1)
        plan = min(plans, key=lambda row: (row["visits"], row["created"])) if plans and reference_turn % 2 else None
        horizon = min(config.combo_terminal_horizon, 100000,
                      config.maximum_ticks - checkpoint["summary"]["tick"])
        payload = {"search_seed": rng.getrandbits(32), "operator": "combo-terminal",
                   "horizon": horizon, "reference_id": reference_id, "origin_id": origin_id}
        if plan:
            payload["plan_blob"] = plan["plan_blob"]
        return store.enqueue("combo_terminal", record_seed, checkpoint_id=checkpoint["id"], lane="combo",
                             payload=payload, dedupe_key=f"combo-terminal:{sequence}")

    # Only a bounded shortlist receives the challenger lane. The wildcard lane
    # remains the route for unusual low-score states outside that shortlist.
    by_seed = {}
    for row in live:
        by_seed.setdefault(row["seed"], []).append(row)
    ranked = sorted(by_seed, key=lambda seed: max(combo_key(row["summary"])
                                                for row in by_seed[seed]), reverse=True)
    if lane == "wildcard":
        seed = rng.choice(ranked)
        checkpoint = rng.choice(by_seed[seed])
        origin_id = checkpoint["id"]
    else:
        pool = ([record_seed] if lane == "combo" and record_seed in by_seed else
                [seed for seed in ranked if seed != record_seed][:config.promoted_seeds] or ranked[:1])
        counts = dict(store.db.execute("SELECT seed,COUNT(*) FROM jobs WHERE lane='probe' GROUP BY seed"))
        seed = min(pool, key=lambda key: (counts.get(key, 0), -max(combo_key(row["summary"])[0]
                                                                     for row in by_seed[key])))
        frontier = [row for row in store.combo_frontier(seed)
                    if row["summary"]["tick"] < config.maximum_ticks]
        if frontier:
            selected = min(frontier, key=lambda row: (row["visits"], row["created"]))
            checkpoint = {"id": selected["checkpoint_id"], "summary": selected["summary"]}
            origin_id = selected["origin_id"]
        else:
            candidates = sorted(by_seed[seed], key=lambda row: combo_key(row["summary"]), reverse=True)
            checkpoint = min(candidates[:config.combo_frontier_width],
                             key=lambda row: uses.get(row["id"], 0))
            origin_id = checkpoint["id"]
    counter_key = f"combo_probe_sequence:{seed}"
    turn = store.get_meta(counter_key, 0)
    store.set_meta(counter_key, turn + 1)
    operator = ("recruit", "preserve", "handoff", "payout")[turn % 4]
    horizon = config.horizon
    # Pay out a developed combo through the same terminal evaluator even when
    # a fresh seed has no reference replay yet.
    if lane == "probe" and operator == "payout":
        from .combos import descriptor
        if descriptor(checkpoint["summary"])["chain"] >= config.combo_min_chain:
            return store.enqueue("combo_terminal", seed, checkpoint_id=checkpoint["id"], lane="probe",
                                 payload={"search_seed": rng.getrandbits(32), "operator": "combo-terminal",
                                          "horizon": min(config.combo_terminal_horizon, 100000,
                                                         config.maximum_ticks - checkpoint["summary"]["tick"]),
                                          "origin_id": origin_id,
                                          "reference_id": store.get_meta(f"baseline:{seed}")},
                                 dedupe_key=f"combo-payout:{sequence}")
    if lane == "probe" and operator == "payout":
        horizon = max(config.bootstrap_ticks, config.horizon * 4)
        if next(row for row in seeds if row["seed"] == seed)["search_seconds"] >= config.finalist_seconds:
            horizon = max(horizon, config.bootstrap_ticks * 4)
    elif lane == "wildcard":
        horizon *= rng.choice((1, 2, 4))
    horizon = min(horizon, 100000, config.maximum_ticks - checkpoint["summary"]["tick"])
    return store.enqueue("probe" if lane == "probe" else "expand", seed,
                         checkpoint_id=checkpoint["id"], lane=lane,
                         payload={"search_seed": rng.getrandbits(32), "operator": operator,
                                  "horizon": horizon, "origin_id": origin_id},
                         dedupe_key=f"combo-develop:{sequence}")
