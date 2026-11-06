"""Recovered October state descriptors used by the event beam."""
from __future__ import annotations
from collections import Counter
import math
from .opportunity_value import ComboHistory, _health, evaluate_opportunity

def _bucket(value: float, boundaries: tuple) -> int:
    return sum(value >= boundary for boundary in boundaries)



def _carrier_layout(group: dict) -> list:
    """Broad relay geometry and safe first-hit budget; unknown fields stay unknown."""
    return [[int(body["id"]),
             math.floor(float(body.get("effect_x", body["x"])) / 64) if "x" in body else None,
             math.floor(float(body.get("effect_y", body["y"])) / 64) if "y" in body else None,
             max(0, 1 - int(body["projectile_hits"])) if "projectile_hits" in body else None]
            for body in sorted(group.get("carrier_states", group.get("carriers", ())), key=lambda b: b["id"])]



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

