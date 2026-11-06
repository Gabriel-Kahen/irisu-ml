"""Opt-in endgame heuristic for feeding before a shared qualification.

Joining does not consume the clear budget. Multiple actors can score after one
qualification, so a one-clear ending can still benefit from visible feeders.
This is deliberately optimistic ranking credit, never a promised game score.
"""
from dataclasses import replace
import math

from rl_combo_opportunity_value import evaluate_opportunity, nominal_score, _factor, _health, _position


def evaluate_diverse(observation, history, config, *, elapsed_ticks=0, co_clear_confidence=.4):
    if not math.isfinite(co_clear_confidence) or not 0 <= co_clear_confidence <= 1:
        raise ValueError("co-clear confidence must be between zero and one")
    value = evaluate_opportunity(observation, history, config, elapsed_ticks=elapsed_ticks)
    remaining = max(0,config.qualifying_clear_limit-int(observation.get("qualifying_clear_count",0))-1)
    if value.terminal or not remaining or not co_clear_confidence:
        return value
    pieces = [b for b in observation.get("bodies",()) if b.get("kind")=="piece" and _health(b)>0]
    # Existing nonrotating carriers generally cost one qualification each.
    # A new rot-running feeder can share its target's payment, but simply
    # keeping six carriers alive does not cash out six under one qualification.
    if remaining <= 8:
        offers=[]
        for group in value.group_details:
            for carrier in pieces:
                if int(carrier.get("chain_id",0)) == group["chain_id"]:
                    amount=nominal_score(int(observation["level"]),group["successful_clears"]+1,
                        group["cumulative_members"],_factor(carrier))*_health(carrier)
                    offers.append((amount,int(carrier["id"]),group["chain_id"]))
        selected=sorted(offers,reverse=True)[:remaining]
        capped=[]
        for group in value.group_details:
            row=dict(group)
            paid=[o for o in selected if o[2]==row["chain_id"]]
            row["cashout_carrier_ids"]=[o[1] for o in paid]
            row["cashout"]=sum(o[0] for o in paid)*config.cashout_confidence*(1 if row["history_known"] else config.missing_history_confidence)
            capped.append(row)
        cashout=sum(g["cashout"] for g in capped)
        adjustment=config.cashout_weight*(cashout-value.cashout_value)
        value=replace(value,cashout_value=cashout,potential=value.potential+adjustment,
                      objective=value.objective+adjustment,group_details=tuple(capped))
    if remaining not in (1,2):
        return value
    assigned = {bid for g in value.group_details for bid in g["feeder_ids"]}
    assigned.update(bid for opening in value.opening_details for bid in opening["member_ids"])
    details, extra = [],0.0
    for group in sorted(value.group_details,key=lambda g:g["cashout"],reverse=True):
        row = dict(group)
        carriers = [b for b in pieces if int(b.get("chain_id",0))==row["chain_id"]
                    and b.get("lifecycle")!="rotten" and _health(b)>=.25]
        if not carriers or not row["history_known"]:
            details.append(row)
            continue
        feeders = [b for b in pieces if not b.get("chain_id") and int(b["id"]) not in assigned
                   and b.get("lifecycle") in
                   ("scripted_falling","dynamic_fresh","fresh","falling")
                   and any(b["color"]==c["color"] for c in carriers)]
        # An observed rot-running fresh actor is already prepared for a shared
        # payout. Keep it ahead of ordinary feeders; rotten actors cannot join.
        feeders.sort(key=lambda b:(not bool(b.get("rot_timer")),
            min(math.dist(_position(b),_position(c)) for c in carriers)))
        selected = []
        for feeder in feeders:
            distance = min(math.dist(_position(feeder),_position(c)) for c in carriers)
            if distance >= config.feed_radius or len(selected) >= config.max_extensions:
                continue
            selected.append(feeder)
            assigned.add(int(feeder["id"]))
            members = row["cumulative_members"]+len(selected)
            clears = row["successful_clears"]+1
            paying_carriers=[c for c in carriers if int(c["id"]) in row.get("cashout_carrier_ids",())]
            gained = nominal_score(int(observation["level"]),clears,members,_factor(feeder)) if len(selected)<=remaining else 0
            gained += sum(nominal_score(int(observation["level"]),clears,members,_factor(c))-
                          nominal_score(int(observation["level"]),clears,members-1,_factor(c))
                          for c in paying_carriers+selected[:min(len(selected)-1,remaining)])
            readiness = 2 if feeder.get("rot_timer") else 1
            confidence = min(1,co_clear_confidence*readiness)*(1-distance/config.feed_radius)*_health(feeder)*max(map(_health,carriers))
            extra += gained*confidence
        row["co_clear_feeder_ids"] = [int(b["id"]) for b in selected]
        row["prepared_feeder_ids"] = [int(b["id"]) for b in selected if b.get("rot_timer")]
        details.append(row)
    weighted = config.extension_weight*extra
    return replace(value,extension_value=value.extension_value+extra,
        potential=value.potential+weighted,objective=value.objective+weighted,group_details=tuple(details))
