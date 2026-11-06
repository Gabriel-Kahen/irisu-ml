"""Closed-loop combo options executed against exact, event-bearing transitions.

Options describe intentions, not promised outcomes. Re-aiming uses the current
body positions and returns only committed input frames; checkpoint probes never
enter the replay. Public observations are not treated as exact state identities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "benchmarks")]

from irisu_env.exact_ipc import _BODY, _OBSERVATION_HEADER, _TRANSITION
from rl_combo_growth_proposals import (GrowthMacro, propose_macros, _color,
                                       _group, _key, _num, _pos, _value)
from rl_combo_opportunity_value import _health
from rl_combo_replay_audit import unpack_events
from rl_exact_suffix_beam import snapshot
from rl_record_rollout import _snapshot


@dataclass(frozen=True, slots=True)
class EventOption:
    label: str
    family: str
    duration: int
    mode: str = "macro"
    macro: GrowthMacro | None = None
    source_id: int = 0
    target_id: int = 0
    group_id: int = 0
    angle: float = 0.0
    radius: float = .6
    strength: int = 2
    observation_period: int = 4
    min_wait: int = 4
    max_shots: int = 2
    inventory_target: int = 3
    probe_ticks: int = 192
    relay_angle: float = 60.0
    relay_radius: float = .35
    prediction_tick: int = 0

    def __post_init__(self):
        if type(self.duration) is not int or self.duration < 1:
            raise ValueError("duration must be a positive integer")
        if self.mode not in ("macro", "wait_spawn", "wait_join", "feed_spawn", "wait_inventory",
                              "paired", "relay", "relay_separate", "reposition", "suppress_minor", "eject_fresh", "causal_protect", "retire_carrier", "prepare_feeder"):
            raise ValueError("unknown option mode")
        if self.mode == "macro" and (self.macro is None or self.macro.duration != self.duration):
            raise ValueError("macro option requires a matching GrowthMacro")
        if not 1 <= self.observation_period <= 64 or self.min_wait < 1 or not 1 <= self.max_shots <= 3:
            raise ValueError("invalid option timing")
        if self.strength not in (1, 2) or not math.isfinite(self.angle) or not 0 < self.radius <= 1.5:
            raise ValueError("invalid shot geometry")
        if self.inventory_target not in (3, 4) or not 1 <= self.probe_ticks <= 512:
            raise ValueError("invalid inventory or probe horizon")
        if not math.isfinite(self.relay_angle) or not 0 < self.relay_radius <= 1:
            raise ValueError("invalid relay geometry")
        if type(self.prediction_tick) is not int or self.prediction_tick < 0:
            raise ValueError("invalid prediction tick")


def _pieces(observation):
    return sorted((b for b in observation.get("bodies", ())
                   if b.get("kind") == "piece" and _num(b.get("size")) > 0 and _health(b) > 0), key=_key)


def _fresh(body):
    return not _group(body) and _num(body.get("rot_timer")) == 0 and body.get("lifecycle") in {
        "scripted_falling", "dynamic_fresh", "falling", "fresh"}


def _aim(source, target, option, carriers, *, protect=False, direct=False):
    sx, sy = _pos(source)
    tx, ty = _pos(target)
    angle = math.atan2(sy - ty, sx - tx) + math.radians(option.angle) + (math.pi if protect else 0)
    x = round(sx + math.cos(angle) * _num(source.get("size")) * option.radius)
    y = round(sy + math.sin(angle) * _num(source.get("size")) * option.radius)
    if not (0 <= x < 640 and 0 <= y < 480):
        return None
    if any(math.dist((x, y), _pos(b)) <= .72 * _num(b.get("size")) + 8
           for b in carriers if not (direct and int(b["id"]) == int(source["id"]))):
        return None
    return (y << 12) | (x << 2) | option.strength


def propose_options(observation, history, duration=96, cap=32, rng=None):
    """Reserve ordinary actions, then timed waits, paired feeds and relays.

    Repositioning guards a first carrier hit. Suppression guards deliberate
    deletion of a minor group member against neutral, protecting valuable
    groups and measuring consumption of the qualification budget.
    """
    macros = propose_macros(observation, history, duration, cap, rng)
    if not macros:
        return []
    conservative = [EventOption(m.label, m.family, duration, macro=m) for m in macros]
    pieces = _pieces(observation)
    carriers = [b for b in pieces if _group(b)]
    fresh = [b for b in pieces if _fresh(b)]
    counts = Counter(_group(b) for b in carriers)
    priority = {_group(b): _value(history, _group(b), counts[_group(b)]) for b in carriers}
    groups = sorted(priority, key=lambda g: (-priority[g], g))
    options = [EventOption(f"wait-spawn-{duration}", "wait", duration, "wait_spawn")]
    for rank, gid in enumerate(groups[:2]):
        options.extend([
            EventOption(f"wait-join-g{rank}", "extend", duration, "wait_join", group_id=gid),
            EventOption(f"feed-spawn-g{rank}", "extend", duration, "feed_spawn", group_id=gid),
        ])
        if max((_health(b) for b in carriers if _group(b) == gid), default=0) >= .25:
            for amount in (3, 4):
                options.append(EventOption(f"inventory-g{rank}-n{amount}", "extend", duration,
                    "wait_inventory", group_id=gid, inventory_target=amount))
    pairs = [(a, b, "extend") for a in fresh for b in carriers if _color(a) == _color(b)]
    pairs += [(a, b, "open") for i, a in enumerate(fresh) for b in fresh[i+1:] if _color(a) == _color(b)]
    pairs.sort(key=lambda p: (-priority.get(_group(p[1]), 0), math.dist(_pos(p[0]), _pos(p[1])), _key(p[0])))
    valid_pairs, examined_pairs = Counter(), Counter()
    for index, (source, target, family) in enumerate(pairs):
        if valid_pairs[family] >= 4 or examined_pairs[family] >= 16:
            continue
        examined_pairs[family] += 1
        safe = []
        for radius in (.6, 1.15):
            for angle in (0, -45, 45, 90):
                jitter = rng.uniform(-10, 10) if rng is not None else 0
                option = EventOption(f"paired-{family}-p{index}-r{radius}-a{angle}", family, duration, "paired",
                    source_id=int(source["id"]), target_id=int(target["id"]), group_id=_group(target),
                    angle=angle+jitter, radius=radius, min_wait=(4, 8, 16, 24)[(index+angle//45) % 4])
                if _aim(source, target, option, carriers) is not None:
                    safe.append(option)
        if not safe:
            continue
        valid_pairs[family] += 1
        options.extend(safe)
        if family == "extend" and priority.get(_group(target), 0) >= 12:
            first = safe[0]
            options.append(EventOption(f"relay-g{groups.index(_group(target))}-p{index}", "relay", duration, "relay",
                source_id=int(source["id"]), target_id=int(target["id"]), group_id=_group(target),
                angle=first.angle, radius=first.radius, max_shots=3))
            for angle in (-60, 60, -90, 90):
                options.append(EventOption(f"relay-feed-g{groups.index(_group(target))}-p{index}-s{angle}",
                    "relay", duration, "relay_separate", source_id=int(source["id"]), target_id=int(target["id"]),
                    group_id=_group(target), angle=first.angle, radius=first.radius,
                    relay_angle=angle, max_shots=3))
    for rank, gid in enumerate(groups[:2]):
        if priority[gid] < 12:
            continue
        members = [b for b in carriers if _group(b) == gid]
        young = sorted((b for b in members if not _num(b.get("projectile_hits"))),
                       key=lambda b: (_num(b.get("age_ticks")), _key(b)))
        for member_rank, younger in enumerate(young[:2]):
            older = [b for b in members if int(b["id"]) != int(younger["id"]) and
                     _num(b.get("age_ticks")) > _num(younger.get("age_ticks"))+6 and
                     math.dist(_pos(b), _pos(younger)) < .72*(_num(b.get("size"))+_num(younger.get("size")))+80]
            if not older:
                continue
            target = max(older, key=lambda b: _num(b.get("age_ticks")))
            for angle in (-60, 60, -90, 90):
                option = EventOption(f"relay-carrier-g{rank}-p{member_rank}-s{angle}", "relay", duration,
                    "relay_separate", source_id=int(younger["id"]), target_id=int(target["id"]),
                    group_id=gid, relay_angle=angle, max_shots=2)
                if _aim(younger, target, replace(option, angle=angle, radius=option.relay_radius),
                        carriers, protect=True, direct=True) is not None:
                    options.append(option)
    for member_rank, carrier in enumerate(sorted(carriers, key=lambda b: -priority[_group(b)])[:2]):
        threats = [b for b in pieces if not _group(b) and
                   (_color(b) != _color(carrier) or _num(b.get("rot_timer")) > 0) and
                   math.dist(_pos(b), _pos(carrier)) < 160 + _num(carrier.get("size"))]
        if (not _num(carrier.get("projectile_hits")) and _num(carrier.get("age_ticks")) > 100 and
                (_num(carrier.get("rot_timer")) > 0 or _pos(carrier)[1] > 400 or threats)):
            target = min(threats, key=lambda b: math.dist(_pos(b), _pos(carrier))) if threats else {
                "x": _pos(carrier)[0], "y": 560}
            option = EventOption(f"guarded-relocate-g{groups.index(_group(carrier))}-p{member_rank}", "reposition", duration,
                "reposition", source_id=int(carrier["id"]), target_id=int(target.get("id", 0)),
                group_id=_group(carrier), max_shots=1)
            if _aim(carrier, target, option, carriers, protect=True, direct=True) is not None:
                options.append(option)
    # The limited terminal budget makes minor clear suppression most relevant
    # late in the game. Two hit buttons are hypotheses until exact probes show
    # target removal and preserved valuable carriers.
    if groups and int(observation.get("qualifying_clear_count", 0)) >= 800:
        dominant = groups[0]
        for rank, gid in enumerate(groups[1:4], 1):
            if priority[gid] > .10*priority[dominant]:
                continue
            protected = [b for b in carriers if _group(b) == dominant or
                         priority[_group(b)] >= max(12, .25*priority[dominant])]
            template = EventOption("suppression-probe", "budget", duration, "suppress_minor", radius=.15)
            targets = sorted((b for b in carriers if _group(b) == gid and _num(b.get("age_ticks")) > 100 and
                              _budget_word(b, template, protected) is not None),
                key=lambda b: (-_num(b.get("rot_timer")), -_pos(b)[1], _key(b)))
            if not targets:
                continue
            for strength in (1, 2):
                options.append(EventOption(f"suppress-g{rank}-k{strength}", "budget", duration,
                    "suppress_minor", source_id=int(targets[0]["id"]), group_id=gid,
                    radius=.15, strength=strength, min_wait=2))
        dominant_colors = {_color(b) for b in carriers if _group(b) == dominant}
        unwanted = [b for b in fresh if _color(b) not in dominant_colors and
                    (_pos(b)[0] < 120 or _pos(b)[0] > 520 or _pos(b)[1] > 350)]
        unwanted.sort(key=lambda b: (min(_pos(b)[0], 640-_pos(b)[0], 480-_pos(b)[1]), _key(b)))
        for rank, body in enumerate(unwanted[:2]):
            options.append(EventOption(f"eject-fresh-c{_color(body)}-p{rank}", "budget", duration,
                "eject_fresh", source_id=int(body["id"]), strength=2, min_wait=2))
    # Family reservations retain the original safe fallback even with small caps.
    reserve = []
    for family in ("wait", "extend", "open", "protect", "cashout"):
        match = next((m for m in conservative if m.family == family), None)
        if match is not None:
            reserve.append(match)
    remainder = [m for m in conservative if m not in reserve]
    # Interleave hypothesis families rather than spending every event slot on
    # angular variants of the first pair.
    ordered, pools = [], {}
    for mode in ("relay_separate", "wait_inventory", "suppress_minor", "eject_fresh"):
        for option in options:
            if option.mode == mode:
                pools.setdefault(option.mode, []).append(option)
        if mode == "relay_separate" and mode in pools:
            pools[mode].sort(key=lambda o: not o.label.startswith("relay-carrier"))
    for option in options:
        if option.mode in ("relay_separate", "wait_inventory", "suppress_minor", "eject_fresh"):
            continue
        pools.setdefault(option.mode, []).append(option)
    while pools:
        for mode in list(pools):
            ordered.append(pools[mode].pop(0))
            if not pools[mode]:
                del pools[mode]
    event_slots = max(0, cap-len(reserve))
    event_slots = min(len(ordered), max((event_slots+1)//2, min(event_slots, 8)))
    return (reserve + ordered[:event_slots] + remainder)[:cap]


def _step(env, word, duration):
    native = getattr(env, "_native", env)
    native.send_step(word & 3, (word >> 2) & 1023, (word >> 12) & 511, duration)
    payload = native.receive_step_raw()
    header = _OBSERVATION_HEADER.unpack_from(payload)
    trans = _TRANSITION.unpack_from(payload, _OBSERVATION_HEADER.size + header[15] * _BODY.size)
    if trans[15]:
        raise RuntimeError("invalid input during event option")
    final = _snapshot(header, trans)
    observation = native.observation()
    observation.update({k: v for k, v in final.items() if k.startswith("canonical_")})
    return observation, final, unpack_events(payload)


def _valuable_groups(observation, history):
    counts = Counter(_group(b) for b in _pieces(observation) if _group(b))
    values = {gid: _value(history, gid, count) for gid, count in counts.items()}
    groups = sorted(values, key=lambda gid: (-values[gid], gid))
    if not groups:
        return []
    return [gid for gid in groups if gid == groups[0] or values[gid] >= max(12, .25*values[groups[0]])]


def _group_confirmations(events, history, groups):
    memberships = dict(history.memberships)
    result = []
    for event in events:
        if event["kind"] == "chain_joined":
            memberships[int(event["a"])] = int(event["value"])
        elif event["kind"] == "confirmed" and memberships.get(int(event["a"])) in groups:
            result.append({**event, "group_id": memberships[int(event["a"])]})
    return result


def propose_protection_options(env, observation, history, duration=32, cap=8, budget=None):
    """Probe neutral once, then target present bodies that cause valuable bursts.

    This is opt-in: ordinary proposals retain their original behavior. Future
    actors and boundary contacts are not actionable present-body hypotheses.
    Each proposed displacement receives a paired exact guard when executed.
    """
    groups = _valuable_groups(observation, history)
    if cap <= 0 or not groups or not hasattr(env, "fast_checkpoint") or observation.get("terminated") or observation.get("truncated"):
        return []
    events, current, final = [], observation, snapshot(observation)
    limit = int(observation["tick"])+192
    with env.fast_checkpoint() as checkpoint, checkpoint.branch() as branch:
        while int(final["tick"]) < limit and not (final["terminated"] or final["truncated"]):
            if budget is not None:
                budget.check()
            current, final, batch = _step(branch, 0, min(16, limit-int(final["tick"])))
            events.extend(batch)
    bodies = {int(b["id"]): b for b in _pieces(observation)}
    carriers = [b for b in bodies.values() if _group(b)]
    result, seen_causes, seen_words = [], set(), set()
    for event in _group_confirmations(events, history, groups):
        source = bodies.get(int(event["b"]))
        gid = event["group_id"]
        if source is None or int(source["id"]) in seen_causes or _group(source) in groups:
            continue
        seen_causes.add(int(source["id"]))
        target = bodies.get(int(event["a"]))
        if target is None:
            target = min((b for b in carriers if _group(b) == gid),
                         key=lambda b: math.dist(_pos(source), _pos(b)))
        for radius, strength in ((.6, 2), (1.15, 2), (.6, 1), (1.15, 1)):
            for angle in (0, -60, 60, 180):
                option = EventOption(f"causal-protect-g{groups.index(gid)}-c{len(seen_causes)-1}-r{radius}-a{angle}-k{strength}",
                    "protect", duration, "causal_protect", source_id=int(source["id"]),
                    target_id=int(target["id"]), group_id=gid, radius=radius, angle=angle,
                    strength=strength, max_shots=1, prediction_tick=int(event["tick"]))
                word = _aim(source, target, option, carriers, protect=True)
                if word is not None and word not in seen_words:
                    seen_words.add(word)
                    result.append(option)
                    if len(result) >= cap:
                        return result
    return result


def _protection_plan(env, observation, history, option, budget, end, probe_end, *, attack):
    current, final, ledger = observation, snapshot(observation), history.copy()
    actions, events, shots, saved = [], [], [], None
    while int(final["tick"]) < probe_end and not (final["terminated"] or final["truncated"]):
        if budget is not None:
            budget.check()
        tick, word = int(final["tick"]), 0
        span = min(16, probe_end-tick)
        if attack and not shots and tick < end:
            bodies = {int(b["id"]): b for b in _pieces(current)}
            source, target = bodies.get(option.source_id), bodies.get(option.target_id)
            if source is None or target is None:
                return None
            span = 1
            if not current.get("left_held" if option.strength == 1 else "right_held"):
                word = _aim(source, target, option, [b for b in bodies.values() if _group(b)], protect=True)
                if word is None:
                    return None
        if tick < end:
            span = min(span, end-tick)
        current, final, batch = _step(env, word, span)
        elapsed = int(final["tick"])-tick
        if not 0 < elapsed <= span or (elapsed < span and not (final["terminated"] or final["truncated"])):
            raise RuntimeError("protection probe tick accounting mismatch")
        actions.extend([word] if word else [0]*elapsed)
        events.extend(batch)
        ledger.update(batch)
        if word:
            shots.append(dict(tick=tick, word=word, source_id=option.source_id, target_id=option.target_id))
        if int(final["tick"]) == end:
            saved = dict(observation=current, final=final, history=ledger.copy(), actions=tuple(actions))
    if saved is None:
        saved = dict(observation=current, final=final, history=ledger.copy(), actions=tuple(actions))
    return dict(observation=current, final=final, events=events, shots=shots, committed=saved)


def _execute_protection(env, observation, history, option, budget, end, stop_tick):
    start = int(observation["tick"])
    horizon = option.probe_ticks
    if option.prediction_tick > start:
        horizon = min(horizon, option.prediction_tick-start+16)
    probe_end = min(start+max(option.duration, horizon), stop_tick)
    groups = _valuable_groups(observation, history)
    original = {int(b["id"]): b for b in _pieces(observation) if _group(b) in groups}
    source = next((b for b in _pieces(observation) if int(b["id"]) == option.source_id), None)
    eligible = bool(option.group_id in groups and source is not None and _group(source) not in groups
                    and end > start and hasattr(env, "fast_checkpoint"))
    verdict, chosen = dict(accepted=False, reason="ineligible", probe_end_tick=probe_end), None
    shots = []
    if eligible:
        arms = []
        with env.fast_checkpoint() as checkpoint:
            for attack in (True, False):
                with checkpoint.branch() as branch:
                    arms.append(_protection_plan(branch, observation, history, option, budget,
                                                 end, probe_end, attack=attack))
        shot, neutral = arms
        reason = "unsafe_geometry" if shot is None else "accepted"
        if shot is not None:
            live = {int(b["id"]): b for b in _pieces(shot["observation"])}
            reference = {int(b["id"]): b for b in _pieces(neutral["observation"])}
            cleared = {int(e["a"]) for e in shot["events"] if e["kind"] == "cleared"}
            lost = {int(e["a"]) for e in shot["events"] if e["kind"] in ("destroyed", "ejected")}
            saving = int(neutral["final"]["qualifying_clear_count"])-int(shot["final"]["qualifying_clear_count"])
            confirmations = [_group_confirmations(arm["events"], history, [option.group_id]) for arm in arms]
            first = [min((int(e["tick"]) for e in batch), default=probe_end+1) for batch in confirmations]
            main_live = [b for b in live.values() if _group(b) == option.group_id]
            neutral_main = [b for b in reference.values() if _group(b) == option.group_id]
            if not shot["shots"]:
                reason = "no_committed_shot"
            elif int(shot["final"]["gauge"]) <= 0 or shot["final"]["terminated"] or shot["final"]["truncated"]:
                reason = "terminal_or_gauge_death"
            elif not main_live:
                reason = "main_group_lost"
            elif lost & original.keys() - cleared:
                reason = "valuable_unscored_loss"
            elif any(body_id in live and (_group(live[body_id]) != _group(body) or
                     _num(live[body_id].get("projectile_hits")) > _num(body.get("projectile_hits")))
                     for body_id, body in original.items()):
                reason = "valuable_direct_hit"
            elif any(body_id in reference and body_id not in live and body_id not in cleared for body_id in original):
                reason = "valuable_loss"
            elif saving < 0:
                reason = "qualification_cost"
            elif not (saving > 0 or first[0] > first[1] or not neutral_main):
                reason = "no_observed_asset_benefit"
            classification = "saved_qualification" if saving > 0 else "delayed_burst" if first[0] > first[1] else "preserved_group"
            verdict.update(accepted=reason == "accepted", reason=reason, qualification_saving=saving,
                classification=classification if reason == "accepted" else "rejected",
                attack_first_confirmation=first[0], neutral_first_confirmation=first[1],
                attack_carriers=len(main_live), neutral_carriers=len(neutral_main),
                attack_final=shot["final"], neutral_final=neutral["final"])
            if verdict["accepted"]:
                chosen, shots = shot["committed"], shot["shots"]
        else:
            verdict["reason"] = reason
    current, final, ledger = observation, snapshot(observation), history.copy()
    final.update({k: v for k, v in observation.items() if k.startswith("canonical_")})
    actions, events = [], []
    if chosen is not None:
        for word in chosen["actions"]:
            if budget is not None:
                budget.check()
            current, final, batch = _step(env, word, 1)
            actions.append(word)
            events.extend(batch)
            ledger.update(batch)
        if current != chosen["observation"] or final != chosen["final"] or ledger.to_dict() != chosen["history"].to_dict():
            raise RuntimeError("accepted protection differs from probed prefix")
    natural = bool(final["terminated"] and not final["truncated"])
    if natural and "canonical_score" not in final:
        raise RuntimeError("natural protection lacks canonical metadata")
    return dict(actions=tuple(actions), observation=current, final=final, history=ledger, natural=natural,
        diagnostics=dict(label=option.label, family=option.family, mode=option.mode, start_tick=start,
            end_tick=int(final["tick"]), stop_reason="protection_committed" if chosen is not None else "guard_rejected",
            guard_checked=eligible, guard=verdict, prediction_tick=option.prediction_tick, shots=shots, events=events))


def _guard_reposition(env, observation, option, word, budget, horizon):
    """Paired probe rejects unscored losses and spending a second direct hit."""
    original = {int(b["id"]): b for b in _pieces(observation) if _group(b)}
    if not horizon or not hasattr(env, "fast_checkpoint"):
        return False
    arms = []
    with env.fast_checkpoint() as checkpoint:
        for input_word in (word, 0):
            if budget is not None:
                budget.check()
            with checkpoint.branch() as branch:
                obs, final, events = _step(branch, input_word, 1)
                if horizon > 1 and not (final["terminated"] or final["truncated"]):
                    obs, final, later = _step(branch, 0, horizon-1)
                    events.extend(later)
                arms.append((obs, events))
    shot, neutral = ({int(b["id"]): b for b in _pieces(arm[0])} for arm in arms)
    cleared = {int(e["a"]) for e in arms[0][1] if e["kind"] == "cleared"}
    for body_id, body in original.items():
        if body_id in neutral and body_id not in shot and body_id not in cleared:
            return False
        if body_id in shot and (_group(shot[body_id]) != _group(body) or
                                _num(shot[body_id].get("projectile_hits")) >= 2):
            return False
    return option.source_id in shot and _group(shot[option.source_id]) == option.group_id


def _guard_relay_separation(env, observation, option, word, budget, horizon):
    if not horizon or not hasattr(env, "fast_checkpoint"):
        return dict(accepted=False, reason="no_checkpoint")
    arms = []
    with env.fast_checkpoint() as checkpoint:
        for input_word in (word, 0):
            if budget is not None:
                budget.check()
            with checkpoint.branch() as branch:
                obs, final, events = _step(branch, input_word, 1)
                if horizon > 1 and not (final["terminated"] or final["truncated"]):
                    obs, final, later = _step(branch, 0, horizon-1)
                    events.extend(later)
                arms.append((obs, final, events))
    shot, neutral = ({int(b["id"]): b for b in _pieces(arm[0])} for arm in arms)
    young, old = shot.get(option.source_id), shot.get(option.target_id)
    cleared = {int(e["a"]) for e in arms[0][2] if e["kind"] == "cleared"}
    reason = "accepted"
    if young is None or _group(young) != option.group_id or _num(young.get("projectile_hits")) != 1:
        reason = "young_not_preserved_with_first_hit"
    elif int(arms[0][1]["gauge"]) <= 0 or arms[0][1]["terminated"]:
        reason = "terminal_or_gauge_death"
    else:
        for body_id, body in neutral.items():
            if not _group(body) or body_id == option.source_id:
                continue
            if body_id not in shot and body_id not in cleared:
                reason = "collateral_loss"; break
            if body_id in shot and (_group(shot[body_id]) != _group(body) or
                    _num(shot[body_id].get("projectile_hits")) > _num(body.get("projectile_hits"))):
                reason = "collateral_hit"; break
    distance = math.dist(_pos(young), _pos(old)) if young is not None and old is not None else None
    reference = math.dist(_pos(neutral[option.source_id]), _pos(neutral[option.target_id])) if (
        option.source_id in neutral and option.target_id in neutral) else None
    if reason == "accepted" and not (option.target_id in cleared or
            (distance is not None and reference is not None and distance > reference+2)):
        reason = "no_extra_separation"
    return dict(accepted=reason == "accepted", reason=reason, distance=distance,
                neutral_distance=reference, older_scored=option.target_id in cleared)


def _budget_word(body, option, protected):
    x, y = _pos(body)
    if option.mode == "suppress_minor":
        x += math.cos(math.radians(option.angle))*_num(body.get("size"))*option.radius
        y += math.sin(math.radians(option.angle))*_num(body.get("size"))*option.radius
    elif y > 350 and 480-y < min(x, 640-x):
        y -= _num(body.get("size"))*option.radius  # push toward lower boundary
    else:
        x += (1 if x < 320 else -1)*_num(body.get("size"))*option.radius
    x, y = round(x), round(y)
    if not (0 <= x < 640 and 0 <= y < 480) or any(
            math.dist((x, y), _pos(b)) <= .72*_num(b.get("size"))+8 for b in protected):
        return None
    return (y << 12) | (x << 2) | option.strength


def _budget_plan(env, observation, history, option, budget, end, probe_end, protected_ids, *, attack):
    """Re-aim up to two shots, then observe neutral through the probe boundary."""
    current, final, ledger = observation, snapshot(observation), history.copy()
    actions, events, shots = [], [], []
    saved = None
    last_shot = int(observation["tick"])-option.min_wait
    start = int(observation["tick"])
    while int(final["tick"]) < probe_end and not (final["terminated"] or final["truncated"]):
        if budget is not None:
            budget.check()
        tick = int(final["tick"])
        pieces = _pieces(current)
        source = next((b for b in pieces if int(b["id"]) == option.source_id), None)
        eligible = source is not None and ((_group(source) == option.group_id and
            _num(source.get("projectile_hits")) < 2) if option.mode == "suppress_minor" else _fresh(source))
        word, span = 0, min(16, probe_end-tick)
        if attack and eligible and len(shots) < option.max_shots and tick-last_shot >= option.min_wait:
            if current.get("left_held" if option.strength == 1 else "right_held"):
                span = 1
            else:
                word = _budget_word(source, option, [b for b in pieces if int(b["id"]) in protected_ids]) or 0
                span = 1
        elif attack and eligible and len(shots) < option.max_shots:
            span = min(span, max(1, option.min_wait-(tick-last_shot)))
        if tick < end:
            span = min(span, end-tick)
        current, final, batch = _step(env, word, span)
        elapsed = int(final["tick"])-tick
        if not 0 < elapsed <= span or (elapsed < span and not (final["terminated"] or final["truncated"])):
            raise RuntimeError("budget probe tick accounting mismatch")
        actions.extend([word] if word else [0]*elapsed)
        events.extend(batch)
        ledger.update(batch)
        if word:
            shots.append(dict(tick=tick, word=word, source_id=option.source_id, source_position=_pos(source)))
            last_shot = tick
        if int(final["tick"]) == end:
            saved = dict(observation=current, final=final, history=ledger.copy(), actions=tuple(actions),
                         events=list(events), shots=list(shots))
    if saved is None:
        saved = dict(observation=current, final=final, history=ledger.copy(), actions=tuple(actions),
                     events=list(events), shots=list(shots))
    return dict(observation=current, final=final, history=ledger, actions=tuple(actions),
                events=events, shots=shots, committed=saved, start_tick=start)


def _execute_budget_option(env, observation, history, option, budget, end, stop_tick):
    pieces = _pieces(observation)
    carriers = [b for b in pieces if _group(b)]
    counts = Counter(_group(b) for b in carriers)
    priorities = {gid: _value(history, gid, count) for gid, count in counts.items()}
    dominant = max(priorities, key=priorities.get) if priorities else 0
    protected_groups = {gid for gid, value in priorities.items()
                        if gid == dominant or value >= max(12, .25*priorities.get(dominant, 0))}
    protected_ids = {int(b["id"]) for b in carriers if _group(b) in protected_groups}
    source = next((b for b in pieces if int(b["id"]) == option.source_id), None)
    eligible = bool(dominant and source is not None and (
        _group(source) not in protected_groups and _group(source) == option.group_id and
        priorities.get(option.group_id, 0) <= .10*priorities[dominant] if option.mode == "suppress_minor" else
        _fresh(source) and _color(source) not in {_color(b) for b in carriers if _group(b) == dominant}))
    probe_end = min(int(observation["tick"])+max(option.duration, option.probe_ticks), stop_tick)
    verdict = dict(accepted=False, reason="ineligible", qualification_saving=0, classification="rejected",
                   probe_end_tick=probe_end)
    chosen = None
    if eligible and end > int(observation["tick"]) and hasattr(env, "fast_checkpoint"):
        with env.fast_checkpoint() as checkpoint:
            arms = []
            for attack in (True, False):
                with checkpoint.branch() as branch:
                    arms.append(_budget_plan(branch, observation, history, option, budget,
                        end, probe_end, protected_ids, attack=attack))
        shot, neutral = arms
        shot_live = {int(b["id"]): b for b in _pieces(shot["observation"])}
        neutral_live = {int(b["id"]): b for b in _pieces(neutral["observation"])}
        cleared = {int(e["a"]) for e in shot["events"] if e["kind"] == "cleared"}
        destroyed = {int(e["a"]) for e in shot["events"] if e["kind"] == "destroyed"}
        removed = {int(e["a"]) for e in shot["events"] if e["kind"] ==
                   ("destroyed" if option.mode == "suppress_minor" else "ejected")}
        saving = int(neutral["final"]["qualifying_clear_count"])-int(shot["final"]["qualifying_clear_count"])
        reason = "accepted"
        if option.source_id not in removed or option.source_id in cleared:
            reason = "target_not_removed"
        elif saving < 0:
            reason = "qualification_cost"
        elif int(shot["final"]["gauge"]) <= 0 or (shot["final"]["terminated"] and
                int(shot["final"]["qualifying_clear_count"]) < 990):
            reason = "gauge_death"
        elif protected_ids & destroyed - cleared:
            reason = "valuable_unscored_loss"
        elif any(body_id in neutral_live and body_id not in shot_live and body_id not in cleared
                 for body_id in protected_ids):
            reason = "valuable_loss"
        elif any(body_id in shot_live and body_id in neutral_live and
                 (_group(shot_live[body_id]) != _group(neutral_live[body_id]) or
                  _num(shot_live[body_id].get("projectile_hits")) > _num(neutral_live[body_id].get("projectile_hits")))
                 for body_id in protected_ids):
            reason = "valuable_direct_hit"
        verdict.update(accepted=reason == "accepted", reason=reason, qualification_saving=saving,
            classification=("observed_saving" if saving > 0 else "preventative") if reason == "accepted" else "rejected",
            attack_final=shot["final"], neutral_final=neutral["final"])
        if verdict["accepted"]:
            chosen = shot["committed"]
    current, final, ledger = observation, snapshot(observation), history.copy()
    actions, events = [], []
    if chosen is not None:
        index = 0
        while index < len(chosen["actions"]):
            if budget is not None:
                budget.check()
            word = chosen["actions"][index]
            span = 1
            if not word:
                while index+span < len(chosen["actions"]) and not chosen["actions"][index+span] and span < 32:
                    span += 1
            before = int(final["tick"])
            current, final, batch = _step(env, word, span)
            elapsed = int(final["tick"])-before
            if elapsed != span:
                raise RuntimeError("accepted budget probe failed exact reconstruction")
            actions.extend([word] if word else [0]*elapsed)
            events.extend(batch)
            ledger.update(batch)
            index += elapsed
        if final != chosen["final"] or current != chosen["observation"] or ledger.to_dict() != chosen["history"].to_dict():
            raise RuntimeError("accepted budget option differs from probed prefix")
    natural = bool(final["terminated"] and not final["truncated"])
    if natural and "canonical_score" not in final:
        raise RuntimeError("natural budget result lacks terminal metadata")
    return dict(actions=tuple(actions), observation=current, final=final, history=ledger, natural=natural,
        diagnostics=dict(label=option.label, family=option.family, mode=option.mode,
            start_tick=int(observation["tick"]), end_tick=int(final["tick"]),
            stop_reason="budget_committed" if chosen is not None else "guard_rejected", guard_checked=eligible,
            guard=verdict, events=events, shots=chosen["shots"] if chosen is not None else []))


def propose_retirement_options(env, observation, history, duration=32, cap=4, budget=None):
    """Offer retirement only for a predicted solo payer with younger backups.

    This intentionally loses one already-hit carrier without a payout. Its
    executor must observe a saved qualification and preserve the group and all
    remaining valuable carriers' direct-hit budgets in paired exact probes.
    """
    groups = _valuable_groups(observation, history)
    if cap <= 0 or not groups or not hasattr(env, "fast_checkpoint") or observation.get("terminated") or observation.get("truncated"):
        return []
    events, final = [], snapshot(observation)
    with env.fast_checkpoint() as checkpoint, checkpoint.branch() as branch:
        while final["tick"] < observation["tick"]+192 and not (final["terminated"] or final["truncated"]):
            if budget is not None:
                budget.check()
            _, final, batch = _step(branch, 0, min(16, observation["tick"]+192-final["tick"]))
            events.extend(batch)
    bodies = {int(b["id"]): b for b in _pieces(observation)}
    pools, seen = [], set()
    for event in _group_confirmations(events, history, groups[:1]):
        source = bodies.get(int(event["a"]))
        cause = bodies.get(int(event["b"]))
        if source is None or int(source["id"]) in seen or _num(source.get("projectile_hits")) != 1 or _num(source.get("size")) >= 140:
            continue
        if event["b"] and (cause is None or _color(cause) == _color(source)):
            continue  # Preserve an already available two-actor payout.
        backups = [b for b in bodies.values() if _group(b) == event["group_id"] and
                   int(b["id"]) != int(source["id"]) and _health(b) >= .25 and
                   _num(b.get("age_ticks")) < _num(source.get("age_ticks"))]
        if not backups:
            continue
        seen.add(int(source["id"]))
        protected = [b for b in bodies.values() if _group(b) in groups and int(b["id"]) != int(source["id"])]
        variants = []
        for strength in (1, 2):
            for angle in (0, 90, -90):
                option = EventOption(f"retire-g0-p{len(seen)-1}-a{angle}-k{strength}", "budget", duration,
                    "retire_carrier", source_id=int(source["id"]), group_id=event["group_id"],
                    radius=.10, angle=angle, strength=strength, max_shots=1, prediction_tick=int(event["tick"]))
                if _budget_word(source, replace(option, mode="suppress_minor"), protected) is not None:
                    variants.append(option)
        if variants:
            pools.append(variants)
    result = []
    while pools and len(result) < cap:
        for variants in pools:
            result.append(variants.pop(0))
            if len(result) == cap:
                break
        pools = [variants for variants in pools if variants]
    return result


def _execute_retirement(env, observation, history, option, budget, end, stop_tick):
    start = int(observation["tick"])
    groups = _valuable_groups(observation, history)
    original = {int(b["id"]): b for b in _pieces(observation) if _group(b) in groups}
    source = original.get(option.source_id)
    backups = [b for body_id, b in original.items() if body_id != option.source_id and _group(b) == option.group_id
               and _health(b) >= .25 and source is not None and _num(b.get("age_ticks")) < _num(source.get("age_ticks"))]
    eligible = bool(groups and option.group_id == groups[0] and source is not None and
                    _num(source.get("projectile_hits")) == 1 and _num(source.get("size")) < 140 and backups and
                    end > start and not (observation.get("terminated") or observation.get("truncated")) and
                    hasattr(env, "fast_checkpoint"))
    horizon = min(option.probe_ticks, option.prediction_tick-start+16) if option.prediction_tick > start else option.probe_ticks
    probe_end = min(start+max(option.duration, horizon), stop_tick)
    verdict, chosen = dict(accepted=False, reason="ineligible", probe_end_tick=probe_end), None
    if eligible:
        protected_ids = set(original)-{option.source_id}
        template = replace(option, mode="suppress_minor", max_shots=1)
        arms = []
        with env.fast_checkpoint() as checkpoint:
            for attack in (True, False):
                with checkpoint.branch() as branch:
                    arms.append(_budget_plan(branch, observation, history, template, budget,
                                             end, probe_end, protected_ids, attack=attack))
        shot, neutral = arms
        live = {int(b["id"]): b for b in _pieces(shot["observation"])}
        reference = {int(b["id"]): b for b in _pieces(neutral["observation"])}
        destroyed = {int(e["a"]) for e in shot["events"] if e["kind"] == "destroyed"}
        lost = {int(e["a"]) for e in shot["events"] if e["kind"] in ("destroyed", "ejected")}
        cleared = {int(e["a"]) for e in shot["events"] if e["kind"] == "cleared"}
        confirmed = {int(e["a"]) for e in neutral["events"] if e["kind"] == "confirmed"}
        saving = int(neutral["final"]["qualifying_clear_count"])-int(shot["final"]["qualifying_clear_count"])
        survivors = [b for b in live.values() if _group(b) == option.group_id and _health(b) >= .25]
        reason = "accepted"
        if option.source_id not in destroyed or option.source_id in cleared:
            reason = "target_not_retired_unscored"
        elif option.source_id not in confirmed or saving < 1:
            reason = "no_observed_qualification_saving"
        elif shot["final"]["terminated"] or shot["final"]["truncated"] or int(shot["final"]["gauge"]) <= 0:
            reason = "terminal_or_gauge_death"
        elif not survivors or not any(int(b["id"]) in protected_ids for b in survivors):
            reason = "no_durable_backup"
        elif protected_ids & (lost-cleared):
            reason = "collateral_unscored_loss"
        elif any(body_id in reference and body_id not in live and body_id not in cleared for body_id in protected_ids):
            reason = "collateral_loss"
        elif any(body_id in live and (_group(live[body_id]) != _group(original[body_id]) or
                 _num(live[body_id].get("projectile_hits")) > _num(original[body_id].get("projectile_hits")))
                 for body_id in protected_ids):
            reason = "collateral_direct_hit"
        verdict.update(accepted=reason == "accepted", reason=reason, qualification_saving=saving,
            retired_id=option.source_id, surviving_carriers=len(survivors), attack_final=shot["final"], neutral_final=neutral["final"])
        if verdict["accepted"]:
            chosen = shot["committed"]
    current, final, ledger = observation, snapshot(observation), history.copy()
    final.update({k:v for k,v in observation.items() if k.startswith("canonical_")})
    actions, events = [], []
    if chosen is not None:
        index = 0
        while index < len(chosen["actions"]):
            if budget is not None:
                budget.check()
            word = chosen["actions"][index]
            span = 1
            if not word:
                while index+span < len(chosen["actions"]) and not chosen["actions"][index+span] and span < 32:
                    span += 1
            before = int(final["tick"])
            current, final, batch = _step(env, word, span)
            if int(final["tick"])-before != span:
                raise RuntimeError("retirement failed exact tick accounting")
            actions.extend([word] if word else [0]*span)
            events.extend(batch);ledger.update(batch);index += span
        if current != chosen["observation"] or final != chosen["final"] or ledger.to_dict() != chosen["history"].to_dict():
            raise RuntimeError("retirement differs from probed prefix")
    natural = bool(final["terminated"] and not final["truncated"])
    if natural and "canonical_score" not in final:
        raise RuntimeError("natural retirement lacks canonical metadata")
    return dict(actions=tuple(actions), observation=current, final=final, history=ledger, natural=natural,
        diagnostics=dict(label=option.label, family=option.family, mode=option.mode, start_tick=start, end_tick=final["tick"],
            stop_reason="retirement_committed" if chosen is not None else "guard_rejected", guard_checked=eligible,
            guard=verdict, events=events, shots=chosen["shots"] if chosen is not None else []))


def propose_preparation_options(observation, history, duration=32, cap=8, rng=None):
    """Steer an aged fresh feeder into different-color rot before grouping.

    Starting rot on an ungrouped actor consumes no qualification. A later
    same-color group contact can pay the feeder alongside a carrier. Exact
    execution must observe the preparation before committing this hypothesis.
    """
    if cap <= 0 or observation.get("terminated") or observation.get("truncated"):
        return []
    groups = _valuable_groups(observation, history)
    pieces = _pieces(observation)
    carriers = [b for b in pieces if _group(b)]
    pairs = []
    for rank, gid in enumerate(groups[:2]):
        colors = {_color(b) for b in carriers if _group(b) == gid and _health(b) >= .25}
        for source in (b for b in pieces if _fresh(b) and _num(b.get("age_ticks")) > 100 and _color(b) in colors):
            targets = sorted((b for b in pieces if not _group(b) and _num(b.get("rot_timer")) > 0 and
                              _color(b) != _color(source)), key=lambda b: (math.dist(_pos(b), _pos(source)), _key(b)))
            for target in targets[:3]:
                pairs.append((rank, gid, source, target))
    pairs.sort(key=lambda p: (p[0], math.dist(_pos(p[2]), _pos(p[3])), _key(p[2]), _key(p[3])))
    pools = []
    for index, (rank, gid, source, target) in enumerate(pairs[:8]):
        variants = []
        for radius in (.15, .6, 1.15):
            for angle in (-45, 45, 0, -90, 90):
                for strength in (1, 2):
                    option = EventOption(f"prepare-g{rank}-p{index}-r{radius}-a{angle}-k{strength}",
                        "prepare", duration, "prepare_feeder", source_id=int(source["id"]),
                        target_id=int(target["id"]), group_id=gid, radius=radius, angle=angle,
                        strength=strength, min_wait=8, max_shots=1)
                    if _aim(source, target, option, carriers) is not None:
                        variants.append(option)
        if variants:
            pools.append(variants)
    result = []
    while pools and len(result) < cap:
        for variants in pools:
            result.append(variants.pop(0))
            if len(result) == cap:
                break
        pools = [variants for variants in pools if variants]
    return result


def _execute_preparation(env, observation, history, option, budget, end):
    start = int(observation["tick"])
    groups = _valuable_groups(observation, history)
    bodies = {int(b["id"]): b for b in _pieces(observation)}
    source, target = bodies.get(option.source_id), bodies.get(option.target_id)
    original = {body_id: b for body_id, b in bodies.items() if _group(b) in groups}
    eligible = bool(source is not None and _fresh(source) and _num(source.get("age_ticks")) > 100 and
        target is not None and not _group(target) and _num(target.get("rot_timer")) > 0 and
        _color(source) != _color(target) and option.group_id in groups and
        any(_group(b) == option.group_id and _color(b) == _color(source) and _health(b) >= .25 for b in original.values()) and
        end > start and not (observation.get("terminated") or observation.get("truncated")) and hasattr(env, "fast_checkpoint"))
    verdict, chosen = dict(accepted=False, reason="ineligible"), None
    if eligible:
        with env.fast_checkpoint() as checkpoint:
            with checkpoint.branch() as attack:
                current, final, ledger = observation, snapshot(observation), history.copy()
                actions, events, shots = [], [], []
                reason, last_shot = "not_prepared", start-option.min_wait
                while int(final["tick"]) < end and not (final["terminated"] or final["truncated"]):
                    if budget is not None:
                        budget.check()
                    pieces = _pieces(current)
                    by_id = {int(b["id"]): b for b in pieces}
                    source, target = by_id.get(option.source_id), by_id.get(option.target_id)
                    if source is None:
                        reason = "source_lost"; break
                    if _group(source):
                        reason = "grouped_before_prepared"; break
                    if _num(source.get("rot_timer")) > 0:
                        reason = "prepared"; break
                    tick = int(final["tick"])
                    word, span = 0, min(option.observation_period, end-tick)
                    if target is not None and len(shots) < option.max_shots and tick-last_shot >= option.min_wait:
                        if current.get("left_held" if option.strength == 1 else "right_held"):
                            span = 1
                        else:
                            word = _aim(source, target, option, [b for b in pieces if _group(b)]) or 0
                            span = 1
                    current, final, batch = _step(attack, word, span)
                    elapsed = int(final["tick"])-tick
                    if not 0 < elapsed <= span or (elapsed < span and not (final["terminated"] or final["truncated"])):
                        raise RuntimeError("preparation probe tick accounting mismatch")
                    actions.extend([word] if word else [0]*elapsed)
                    events.extend(batch); ledger.update(batch)
                    if word:
                        shots.append(dict(tick=tick, word=word, source_id=option.source_id,
                                          target_id=option.target_id, source_position=_pos(source)))
                        last_shot = tick
                live = {int(b["id"]): b for b in _pieces(current)}
                prepared = live.get(option.source_id)
                if prepared is not None and not _group(prepared) and _num(prepared.get("rot_timer")) > 0:
                    reason = "prepared"
                lost = {int(e["a"]) for e in events if e["kind"] in ("destroyed", "ejected")}
                anchors = [b for b in live.values() if _group(b) == option.group_id and _health(b) >= .25]
                if reason != "prepared":
                    pass
                elif final["terminated"] or final["truncated"] or int(final["gauge"]) <= 0:
                    reason = "terminal_or_gauge_death"
                elif int(final["qualifying_clear_count"]) != int(observation["qualifying_clear_count"]):
                    reason = "qualification_spent_before_preparation"
                elif set(original) & lost or any(body_id not in live for body_id in original):
                    reason = "valuable_loss"
                elif not anchors:
                    reason = "no_durable_anchor"
                elif any(_group(live[body_id]) != _group(body) or
                         _num(live[body_id].get("projectile_hits")) > _num(body.get("projectile_hits"))
                         for body_id, body in original.items()):
                    reason = "valuable_direct_hit"
                neutral_final = snapshot(observation)
                with checkpoint.branch() as neutral:
                    while int(neutral_final["tick"]) < int(final["tick"]) and not (neutral_final["terminated"] or neutral_final["truncated"]):
                        if budget is not None:
                            budget.check()
                        _, neutral_final, _ = _step(neutral, 0, min(16, int(final["tick"])-int(neutral_final["tick"])))
                verdict.update(accepted=reason == "prepared", reason=reason, prepared_id=option.source_id,
                    probe_end_tick=int(final["tick"]), surviving_carriers=len(anchors), attack_final=final,
                    neutral_final=neutral_final, qualification_saving=int(neutral_final["qualifying_clear_count"])-int(final["qualifying_clear_count"]))
                if verdict["accepted"]:
                    chosen = dict(actions=tuple(actions), observation=current, final=final, history=ledger, shots=shots)
    current, final, ledger = observation, snapshot(observation), history.copy()
    final.update({k:v for k,v in observation.items() if k.startswith("canonical_")})
    actions, events = [], []
    if chosen is not None:
        for word in chosen["actions"]:
            if budget is not None:
                budget.check()
            before = int(final["tick"])
            current, final, batch = _step(env, word, 1)
            if int(final["tick"])-before != 1:
                raise RuntimeError("preparation failed exact tick accounting")
            actions.append(word);events.extend(batch);ledger.update(batch)
        if current != chosen["observation"] or final != chosen["final"] or ledger.to_dict() != chosen["history"].to_dict():
            raise RuntimeError("preparation differs from probed prefix")
    natural = bool(final["terminated"] and not final["truncated"])
    if natural and "canonical_score" not in final:
        raise RuntimeError("natural preparation lacks canonical metadata")
    return dict(actions=tuple(actions), observation=current, final=final, history=ledger, natural=natural,
        diagnostics=dict(label=option.label, family=option.family, mode=option.mode, start_tick=start,
            end_tick=int(final["tick"]), stop_reason="preparation_committed" if chosen is not None else "guard_rejected",
            guard_checked=eligible, guard=verdict, events=events, shots=chosen["shots"] if chosen is not None else []))


def execute_option(env, observation, history, option, budget=None, stop_tick=None):
    """Commit a bounded option and return its exact, possibly short suffix.

    ``history`` is copied. A neutral input is committed before any shot whose
    button is already held. Terminal metadata is retained from the first
    terminal transition, and neutral spans are shortened to actual native time.
    """
    start = int(observation["tick"])
    end = min(start + option.duration, int(stop_tick) if stop_tick is not None else start + option.duration)
    if end < start:
        raise ValueError("stop_tick precedes option start")
    if option.mode == "prepare_feeder":
        return _execute_preparation(env, observation, history, option, budget, end)
    if option.mode == "retire_carrier":
        return _execute_retirement(env, observation, history, option, budget, end,
            int(stop_tick) if stop_tick is not None else start+max(option.duration, option.probe_ticks))
    if option.mode == "causal_protect":
        return _execute_protection(env, observation, history, option, budget, end,
            int(stop_tick) if stop_tick is not None else start+max(option.duration, option.probe_ticks))
    if option.mode in ("suppress_minor", "eject_fresh"):
        return _execute_budget_option(env, observation, history, option, budget, end,
            int(stop_tick) if stop_tick is not None else start+max(option.duration, option.probe_ticks))
    ledger = history.copy()
    current, final = observation, snapshot(observation)
    final.update({k: v for k, v in observation.items() if k.startswith("canonical_")})
    actions, events, shots = [], [], []
    reason, last_shot, last_contact = "duration", start-option.min_wait, -1
    initial_ids = {int(b["id"]) for b in _pieces(observation)}
    initial_members = int(ledger.groups.get(option.group_id, {}).get("cumulative_members", 0))
    relay_joined = bool(option.mode == "relay_separate" and any(
        int(b["id"]) == option.source_id and _group(b) == option.group_id for b in _pieces(observation)))
    relay_guards = []
    guard_checked = False
    initial_level = int(observation.get("level", 0))
    gauge_floor = max(1000, .60*float(observation.get("gauge", 0)))

    def commit(word, duration):
        nonlocal current, final, last_contact
        if budget is not None:
            budget.check()
        before = int(final["tick"])
        current, final, new = _step(env, word, min(duration, end-before))
        elapsed = int(final["tick"])-before
        if not 0 < elapsed <= duration or (elapsed < min(duration, end-before) and
                not (final["terminated"] or final["truncated"])):
            raise RuntimeError("event option tick accounting mismatch")
        actions.extend([word] if word & 3 else [0]*elapsed)
        ledger.update(new)
        events.extend(new)
        for event in new:
            if event["kind"] in ("contact", "projectile_hit") and (
                    int(event.get("a", 0)) in (option.source_id, option.target_id) or
                    int(event.get("b", 0)) in (option.source_id, option.target_id)):
                last_contact = max(last_contact, int(event["tick"]))
        return new

    while int(final["tick"]) < end and not (final["terminated"] or final["truncated"]):
        tick = int(final["tick"])
        if option.mode == "macro":
            offset = tick-start
            future = [(t, w) for t, w in option.macro.sparse if t >= offset]
            word = future[0][1] if future and future[0][0] == offset else 0
            span = 1 if word else min(option.observation_period, future[0][0]-offset if future else end-tick)
            commit(word, span)
            if word:
                shots.append({"tick": tick, "word": word})
            continue
        pieces = _pieces(current)
        by_id = {int(b["id"]): b for b in pieces}
        carriers = [b for b in pieces if _group(b)]
        spawned = bool(set(by_id)-initial_ids)
        members = int(ledger.groups.get(option.group_id, {}).get("cumulative_members", 0))
        joined = members > initial_members
        if option.mode == "wait_spawn" and spawned:
            reason = "spawn"; break
        if option.mode == "wait_join" and joined:
            reason = "join"; break
        if option.mode in ("paired", "feed_spawn", "wait_inventory") and option.group_id and joined:
            reason = "join"; break
        if option.mode == "paired" and not option.group_id and option.source_id in by_id and _group(by_id[option.source_id]):
            reason = "opened"; break
        live_group = [b for b in carriers if _group(b) == option.group_id]
        if option.group_id and not live_group:
            reason = "group_lost"; break
        if option.mode == "wait_inventory" and (int(current.get("level", 0)) != initial_level or
                float(current.get("gauge", 0)) < gauge_floor or
                (live_group and max(_health(b) for b in live_group) < .25)):
            reason = "inventory_health_or_phase"; break
        if option.mode in ("relay", "relay_separate") and option.source_id in by_id and _group(by_id[option.source_id]) == option.group_id:
            relay_joined = True
            if option.target_id not in by_id and option.source_id in by_id and _group(by_id[option.source_id]) == option.group_id:
                reason = "relay"; break
        contact_ready = bool(shots and last_contact > last_shot and tick-last_shot >= 2)
        if option.mode in ("wait_spawn", "wait_join") or len(shots) >= option.max_shots or (tick-last_shot < option.min_wait and not contact_ready):
            commit(0, min(option.observation_period, end-tick)); continue
        source, target = by_id.get(option.source_id), by_id.get(option.target_id)
        chosen_word = None
        protect, direct = False, option.mode == "reposition"
        if option.mode in ("feed_spawn", "wait_inventory"):
            matching = [b for b in pieces if _fresh(b) and any(_color(b) == _color(c) for c in live_group)]
            if (option.mode == "feed_spawn" and not spawned) or (option.mode == "wait_inventory" and
                    not shots and len(matching) < option.inventory_target):
                commit(0, min(option.observation_period, end-tick)); continue
            targets = sorted(live_group, key=lambda b: (_num(b.get("age_ticks")), _key(b)))
            pairs = [(s, t) for s in pieces if _fresh(s) for t in targets if _color(s) == _color(t)]
            pairs.sort(key=lambda p: math.dist(_pos(p[0]), _pos(p[1])))
            for s, t in pairs[:12]:
                for radius in (option.radius, 1.15):
                    for angle in (option.angle, -45, 45, -90, 90):
                        word = _aim(s, t, replace(option, radius=radius, angle=angle), carriers)
                        if word is not None:
                            source, target, chosen_word = s, t, word
                            break
                    if chosen_word is not None:
                        break
                if chosen_word is not None:
                    break
            if chosen_word is None:
                commit(0, min(option.observation_period, end-tick)); continue
        if option.mode in ("relay", "relay_separate") and relay_joined:
            # Route a rot-running trigger into the older member only when the
            # new carrier is spatially separate; its history remains valuable.
            young, old = by_id.get(option.source_id), by_id.get(option.target_id)
            triggers = [b for b in pieces if not _group(b) and _num(b.get("rot_timer")) > 0]
            separation = .72*(_num(young.get("size"))+_num(old.get("size")))+24 if young is not None and old is not None else 0
            if (option.mode == "relay_separate" and young is not None and old is not None and
                    math.dist(_pos(young), _pos(old)) <= separation and not _num(young.get("projectile_hits"))):
                separate = replace(option, angle=option.relay_angle, radius=option.relay_radius)
                separation_word = _aim(young, old, separate, carriers, protect=True, direct=True)
                if separation_word is not None:
                    if current.get("left_held" if option.strength == 1 else "right_held"):
                        commit(0, 1); continue
                    guard_checked = True
                    verdict = _guard_relay_separation(env, current, separate, separation_word,
                        budget, min(16, end-tick))
                    relay_guards.append({"tick": tick, **verdict})
                    if verdict["accepted"]:
                        commit(separation_word, 1)
                        shots.append(dict(tick=tick, word=separation_word, source_id=option.source_id,
                            target_id=option.target_id, source_position=_pos(young), target_position=_pos(old),
                            purpose="separate_young_carrier"))
                        last_shot = tick
                        continue
                commit(0, min(option.observation_period, end-tick)); continue
            if young is not None and old is not None and math.dist(_pos(young), _pos(old)) > .72*(_num(young.get("size"))+_num(old.get("size")))+24 and triggers:
                source, target = min(triggers, key=lambda b: math.dist(_pos(b), _pos(old))), old
            else:
                commit(0, min(option.observation_period, end-tick)); continue
        elif option.mode != "reposition" and (source is None or not _fresh(source)):
            reason = "source_unavailable"; break
        if direct:
            if source is None or _group(source) != option.group_id or _num(source.get("projectile_hits")):
                reason = "guard_ineligible"; break
            target = target or {"x": _pos(source)[0], "y": 560}
            protect = True
        if source is None or target is None:
            reason = "target_unavailable"; break
        word = chosen_word if chosen_word is not None else _aim(source, target, option, carriers, protect=protect, direct=direct)
        if word is None:
            reason = "unsafe_geometry"; break
        held = current.get("left_held" if option.strength == 1 else "right_held")
        if held:
            commit(0, 1); continue
        if direct and not guard_checked:
            guard_checked = True
            if not _guard_reposition(env, current, option, word, budget, min(12, end-tick)):
                reason = "guard_rejected"; break
        commit(word, 1)
        shots.append({"tick": tick, "word": word, "source_id": int(source.get("id", 0)),
                      "target_id": int(target.get("id", 0)), "source_position": _pos(source), "target_position": _pos(target)})
        last_shot = tick
    if final["terminated"] or final["truncated"]:
        reason = "terminal" if final["terminated"] else "truncated"
    natural = bool(final["terminated"] and not final["truncated"])
    if natural and "canonical_score" not in final:
        raise RuntimeError("natural option result lacks terminal metadata")
    return {"actions": tuple(actions), "observation": current, "final": final, "history": ledger,
            "natural": natural, "diagnostics": {"label": option.label, "family": option.family,
                "mode": option.mode, "start_tick": start, "end_tick": int(final["tick"]),
                "stop_reason": reason, "shots": shots, "events": events, "guard_checked": guard_checked,
                "relay_guards": relay_guards}}
