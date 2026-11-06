"""Interpretable, deliberately optimistic value for unfinished combos.

This is a search heuristic, not a learned value or guaranteed future score.
The counters come from exact transition events; geometry only estimates the
possibility of retaining carriers and feeding them. No unseen future spawns
are invented. Compare objectives from one root using its elapsed tick count.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import math
import struct
from typing import Any


# QWORD stores from clone/core/normal_rules.cpp, v2.03 normal levels 1..99.
_SCALE_BITS = (
    0x4010000000000000, 0x4019FDF8BCCE533D, 0x402142E81C889914,
    0x40251CB453B9536C, 0x4028AE6D3FC447A1, 0x402C0A88FB9717F1,
    0x402F3C7995604382, 0x403125FBEE250664, 0x40329F4504B4E7E6,
    0x40340C28430012E7, 0x40356E3D8CCA85AF, 0x4036C6C846AB7CA7,
    0x403816CE65CFE3C4, 0x40395F27EAF32C5B, 0x403AA089A39BA854,
    0x403BDB8CDADBE120, 0x403D10B4FC1EEC63, 0x403E4073CB226E4F,
    0x403F6B2C9B687921, 0x4040489B672A7D8D, 0x4040D96FE46AF637,
    0x40416836416E33C9, 0x4041F50D649383D7, 0x4042801121978C6C,
    0x4043095AA530DCE5, 0x40439100CE4040AD, 0x404417187852199D,
    0x40449BB4BA4F8B2F, 0x40451EE71B998F87, 0x4045A0BFC14CAD7D,
    0x4046214D950E908D, 0x4046A09E667F3BCD, 0x40471EBF08304CB3,
    0x40479BBB68D9AD8B, 0x4048179EA9613A93, 0x4048927330300A96,
    0x40490C42BA3AA4FC, 0x404985166A103EC4, 0x4049FCF6D5373157,
    0x404A73EC101190A5, 0x404AE9FDB87B85FE, 0x404B5F32FF4D7A6A,
    0x404BD392B0E5D159, 0x404C47233CD8BB74, 0x404CB9EABCE04D70,
    0x404D2BEEFB235F59, 0x404D9D3577E6A580, 0x404E0DC36EB8DAD9,
    0x404E7D9DDB28A02C, 0x404EECC97D10D2CC, 0x404F5B4ADC868348,
    0x404FC9264D72541E, 0x40501B2FF96EED30, 0x4050517DE0FE4A4C,
    0x4050877EC27B956A, 0x4050BD346EBF7347, 0x4050F2A0A397535A,
    0x405127C50CDEFFC1, 0x40515CA3458559DE, 0x4051913CD87E217C,
    0x4051C59341A2723E, 0x4051F9A7EE817706, 0x40522D7C3F22AC1C,
    0x4052611186BAE675, 0x405294690C5537FA, 0x4052C7840B70ADA3,
    0x4052FA63B493CC19, 0x40532D092DD69A27, 0x40535F759363F51E,
    0x405391A9F7F2DA08, 0x4053C3A765383F5A, 0x4053F56EDC520CF8,
    0x40542701562BB3E1, 0x4054585FC3DCDBF0, 0x4054898B0F0293CB,
    0x4054BA841A136649, 0x4054EB4BC0AEB020, 0x40551BE2D7E7898A,
    0x40554C4A2E8B907E, 0x40557C828D65DA56, 0x4055AC8CB77E4DDF,
    0x4055DC696A55A1F6, 0x40560C195E1E3824, 0x40563B9D45F20673,
    0x40566AF5D005BFED, 0x40569A23A5D967B3, 0x4056C9276C667751,
    0x4056F801C44BBE14, 0x405726B349F71C7D, 0x4057553C95CD3C54,
    0x4057839E3C4F63B2, 0x4057B1D8CE3F7F32, 0x4057DFECD8C27D8F,
    0x40580DDAE5811544, 0x40583BA37AC70ADE, 0x405869471BA10D87,
    0x405896C647F93DA4, 0x4058C4217CB27030, 0x4058F15933C24051,
)
_SCALES = tuple(struct.unpack("<d", struct.pack("<Q", bits))[0] for bits in _SCALE_BITS)


def nominal_score(level: int, successful_clears: int, cumulative_members: int,
                  size_factor: int = 1) -> int:
    """Exact nominal normal-rule arithmetic, assuming the supplied counters."""
    if (type(level) is not int or not 1 <= level <= 99 or
            any(type(v) is not int or v < 0 for v in
                (successful_clears, cumulative_members)) or
            type(size_factor) is not int or size_factor < 1):
        raise ValueError("invalid normal score parameters")
    value = float(successful_clears) * 0.5
    value += 0.5
    value *= float(cumulative_members)
    value *= _SCALES[level - 1]
    value *= float(size_factor)
    if not math.isfinite(value) or value >= 2**63:
        raise ValueError("nominal score overflow")
    return int(value)


class ComboHistory:
    """Incremental historical counters with cheap, independent branch copies.

    Groups persist after their last body dies, but valuation never credits a
    group without a living observed carrier. Rows are replaced, never mutated;
    callers must treat ``groups`` as read-only. Consume each event batch once.
    """

    def __init__(self) -> None:
        self.groups: dict[int, dict[str, Any]] = {}
        self.memberships: dict[int, int] = {}

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]]) -> ComboHistory:
        history = cls()
        history.update(events)
        return history

    @classmethod
    def from_groups(cls, groups: Mapping | Iterable[Mapping]) -> ComboHistory:
        """Load audit rows cut off at the evaluated state, never final-run rows."""
        history = cls()
        for row in groups.values() if isinstance(groups, Mapping) else groups:
            group = dict(row)
            gid = int(group["chain_id"])
            group["member_ids"] = tuple(int(v) for v in group.get("member_ids", ()))
            history.groups[gid] = group
            history.memberships.update((body, gid) for body in group["member_ids"])
        return history

    def copy(self) -> ComboHistory:
        result = ComboHistory()
        result.groups = self.groups.copy()
        result.memberships = self.memberships.copy()
        return result

    def update(self, events: Iterable[Mapping[str, Any]]) -> None:
        for event in events:
            kind, body = event["kind"], int(event.get("a", 0))
            tick = int(event["tick"])
            if kind == "chain_joined":
                gid = int(event["value"])
                prior = self.groups.get(gid, dict(chain_id=gid, first_tick=tick,
                    cumulative_members=0, successful_clears=0, member_ids=()))
                members = prior["member_ids"]
                new = body not in members
                self.groups[gid] = {**prior, "last_join_tick": tick,
                    "cumulative_members": prior["cumulative_members"] + int(new),
                    "member_ids": (*members, body) if new else members}
                self.memberships[body] = gid
            elif kind in ("confirmed", "cleared") and body in self.memberships:
                gid = self.memberships[body]
                prior = self.groups[gid]
                self.groups[gid] = {**prior, "last_clear_tick": tick,
                    "successful_clears": max(prior["successful_clears"], int(event["value"]))}
            if kind in ("cleared", "destroyed", "ejected"):
                self.memberships.pop(body, None)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "groups": {str(gid): dict(row) for gid, row in self.groups.items()},
                "memberships": {str(body): gid for body, gid in self.memberships.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComboHistory:
        if data.get("version") != 1:
            raise ValueError("unknown combo history format")
        result = cls.from_groups(data["groups"])
        result.memberships = {int(body): int(gid) for body, gid in data["memberships"].items()}
        return result


@dataclass(frozen=True, slots=True)
class OpportunityConfig:
    cashout_weight: float = 1.0
    extension_weight: float = 1.5
    opening_weight: float = 1.25
    cashout_confidence: float = 0.9
    extension_confidence: float = 0.75
    opening_confidence: float = 0.6
    missing_history_confidence: float = 0.45
    feed_radius: float = 420.0
    max_extensions: int = 4
    max_opening_members: int = 6
    stale_half_life_ticks: float = 768.0
    elapsed_tick_cost: float = 1.0
    qualifying_clear_limit: int = 990

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("cashout_confidence", "extension_confidence", "opening_confidence",
                    "missing_history_confidence"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be at most one")
        if self.feed_radius <= 0 or self.stale_half_life_ticks <= 0:
            raise ValueError("feed radius and stale half life must be positive")
        for key in ("max_extensions", "max_opening_members", "qualifying_clear_limit"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.max_opening_members < 2 or self.qualifying_clear_limit < 1:
            raise ValueError("opening members and clear limit are too small")


@dataclass(frozen=True, slots=True)
class OpportunityValue:
    realized_score: int
    cashout_value: float
    extension_value: float
    opening_value: float
    potential: float
    objective: float
    time_cost: float
    group_details: tuple[dict[str, Any], ...]
    opening_details: tuple[dict[str, Any], ...]
    terminal: bool = False
    guaranteed_future_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position(body: Mapping) -> tuple[float, float]:
    return float(body.get("effect_x", body["x"])), float(body.get("effect_y", body["y"]))


def _factor(body: Mapping) -> int:
    # Only the 140-unit ordinary piece has the release-default factor two.
    return 2 if float(body.get("size", 0)) == 140.0 else 1


def _health(body: Mapping) -> float:
    """Survival discount, not a calibrated probability or hard safety gate."""
    if body.get("lifecycle") == "deleted" or int(body.get("remaining_lifetime", -1)) in (0, 1):
        return 0.0
    x, y = _position(body)
    if not (0 <= x <= 640 and -30 <= y <= 560) or body.get("projectile_hits", 0) >= 2:
        return 0.0
    health = 1.0 if y <= 340 else max(0.1, (540 - y) / 200)
    health *= 0.92 if body.get("projectile_hits", 0) else 1.0
    health *= max(0.2, 1.0 - float(body.get("rot_timer", 0)) / 50)
    return health * (0.25 if body.get("lifecycle") == "rotten" else 1.0)


def evaluate_opportunity(observation: Mapping[str, Any], history: ComboHistory,
                         config: OpportunityConfig = OpportunityConfig(), *,
                         elapsed_ticks: int = 0) -> OpportunityValue:
    """Rank a state while allowing score debt backed by live combo assets.

    Credit is inventory-bounded and replaces previous potential; never add it
    as a per-step reward. Time is measured from a common root, not last action.
    Terminal/truncated states have no potential, including final-frame bodies
    whose actor score has not entered the canonical result.
    """
    if type(elapsed_ticks) is not int or elapsed_ticks < 0:
        raise ValueError("elapsed_ticks must be a nonnegative integer")
    level = int(observation["level"])
    clears_so_far = int(observation.get("qualifying_clear_count", 0))
    # The final qualification records the canonical result before actor scoring.
    remaining = max(0, config.qualifying_clear_limit - clears_so_far - 1)
    terminal = bool(observation.get("terminated") or observation.get("truncated") or
                    level >= 100 or clears_so_far >= config.qualifying_clear_limit)
    score = int(observation.get("canonical_score", observation["score"])) if terminal else int(observation["score"])
    time_cost = elapsed_ticks * config.elapsed_tick_cost
    if terminal:
        return OpportunityValue(score, 0, 0, 0, 0, float(score), 0, (), (), True)
    if not 1 <= level <= 99:
        raise ValueError("normal levels must be in 1..99")
    if not remaining:
        return OpportunityValue(score, 0, 0, 0, 0, score - time_cost, time_cost, (), ())
    tick = int(observation["tick"])
    pieces = [body for body in observation.get("bodies", ())
              if body.get("kind") == "piece" and _health(body) > 0]
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    fresh: list[Mapping] = []
    for body in pieces:
        gid = int(body.get("chain_id", 0))
        if gid:
            grouped[gid].append(body)
        elif body.get("lifecycle") in ("scripted_falling", "dynamic_fresh", "fresh", "falling"):
            fresh.append(body)
    details = []
    for gid, carriers in sorted(grouped.items()):
        past = history.groups.get(gid)
        if past and any(int(past.get(key) or 0) > tick for key in ("last_join_tick", "last_clear_tick")):
            raise ValueError("combo history contains events after the evaluated state")
        members = max(len(carriers), int(past["cumulative_members"]) if past else 0)
        clears = int(past["successful_clears"]) if past else 0
        known = 1.0 if past else config.missing_history_confidence
        nominal = sum(nominal_score(level, clears + 1, members, _factor(b)) for b in carriers)
        cashout = sum(nominal_score(level, clears + 1, members, _factor(b)) * _health(b) for b in carriers)
        idle = max(0, tick - int(past.get("last_join_tick", tick))) if past else 0
        details.append(dict(chain_id=gid, cumulative_members=members, successful_clears=clears,
            history_known=bool(past), carrier_ids=[int(b["id"]) for b in carriers], feeder_ids=[],
            nominal_cashout=nominal, cashout=cashout * config.cashout_confidence * known,
            extension=0.0, confidence=known,
            stale_discount=0.25 + 0.75 * 2 ** (-idle / config.stale_half_life_ticks)))
    # Reserve at least one qualifying clear per proposed carrier cashout. This
    # understates rare co-clears, while preventing many endgame phantom groups.
    details = sorted(details, key=lambda row: (-row["cashout"], row["chain_id"]))[:remaining]
    # Allocate each visible feeder once. Several groups cannot all consume it.
    allocations: dict[int, list[tuple[Mapping, float]]] = defaultdict(list)
    candidates = []
    for feeder in fresh:
        for row in details:
            carriers = [b for b in grouped[row["chain_id"]] if b["color"] == feeder["color"]
                        and b.get("lifecycle") != "rotten"]
            if not carriers:
                continue
            distance = min(math.dist(_position(feeder), _position(b)) for b in carriers)
            if distance >= config.feed_radius:
                continue
            confidence = (1 - distance / config.feed_radius) * _health(feeder) * max(map(_health, carriers))
            marginal = nominal_score(level, row["successful_clears"] + 1,
                                     row["cumulative_members"] + 1, _factor(feeder))
            candidates.append((marginal * confidence, int(feeder["id"]), row["chain_id"], feeder, confidence))
    used: set[int] = set()
    # This conservative cap reserves one qualifying clear to cash out the carrier.
    extension_budget = max(0, remaining - len(details))
    for _, ident, gid, feeder, confidence in sorted(candidates, key=lambda item: item[:3], reverse=True):
        if ident not in used and len(allocations[gid]) < config.max_extensions and extension_budget:
            allocations[gid].append((feeder, confidence))
            used.add(ident)
            extension_budget -= 1
    for row in details:
        feeds = allocations[row["chain_id"]]
        members, clears = row["cumulative_members"], row["successful_clears"]
        # Feeding scores itself and increases the retained carrier's payout.
        previous_cashout = row["nominal_cashout"]
        for index, (feeder, confidence) in enumerate(feeds, 1):
            future_cashout = sum(nominal_score(level, clears + index + 1, members + index, _factor(b))
                                 for b in grouped[row["chain_id"]])
            earned = nominal_score(level, clears + index, members + index, _factor(feeder))
            row["extension"] += (earned + future_cashout - previous_cashout) * confidence
            previous_cashout = future_cashout
        row["extension"] *= config.extension_confidence * row["confidence"] * row["stale_discount"]
        row["feeder_ids"] = [int(b["id"]) for b, _ in feeds]
    # Remaining same-color local clusters offer openings, never unseen spawns.
    available = {int(b["id"]): b for b in fresh if int(b["id"]) not in used}
    opening_details = []
    opening_budget = max(0, remaining - len(used) - len(details))
    while len(available) >= 2 and opening_budget >= 2:
        ordered = sorted(available.items())
        pairs = [(math.dist(_position(a), _position(b)), aid, bid)
                 for i, (aid, a) in enumerate(ordered) for bid, b in ordered[i + 1:]
                 if a["color"] == b["color"]]
        if not pairs:
            break
        distance, aid, bid = min(pairs)
        if distance >= config.feed_radius:
            break
        cluster = [available.pop(aid), available.pop(bid)]
        for ident, body in sorted(available.items(), key=lambda item: (
                math.dist(_position(item[1]), _position(cluster[0])), item[0])):
            if len(cluster) >= min(config.max_opening_members, opening_budget):
                break
            if body["color"] == cluster[0]["color"] and math.dist(_position(body), _position(cluster[0])) < config.feed_radius:
                cluster.append(body)
                available.pop(ident)
        opening_budget -= len(cluster)
        confidence = config.opening_confidence * (1 - distance / config.feed_radius) * min(map(_health, cluster))
        value = sum(nominal_score(level, index, len(cluster), _factor(body))
                    for index, body in enumerate(cluster, 1)) * confidence
        opening_details.append(dict(color=int(cluster[0]["color"]), member_ids=[int(b["id"]) for b in cluster],
                                    confidence=confidence, value=value))
    cashout = sum(row["cashout"] for row in details)
    extension = sum(row["extension"] for row in details)
    opening = sum(row["value"] for row in opening_details)
    potential = (config.cashout_weight * cashout + config.extension_weight * extension +
                 config.opening_weight * opening)
    return OpportunityValue(score, cashout, extension, opening, potential,
                            score + potential - time_cost, time_cost,
                            tuple(details), tuple(opening_details))
