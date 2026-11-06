"""Geometry proposals for exact, history-aware combo growth search.

Clicks are hypotheses, not predicted successes: projectiles travel upwards and
contact impulses depend on shape and rotation. Exact branches decide the result.
Only current geometry is used; the planner reobserves after every macro.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


_FRESH = {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
_FAMILIES = ("wait", "extend", "open", "protect", "cashout")


@dataclass(frozen=True, slots=True)
class GrowthMacro:
    label: str
    family: str
    sparse: tuple[tuple[int, int], ...]
    duration: int

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("unknown macro family")
        if type(self.duration) is not int or self.duration < 1:
            raise ValueError("duration must be a positive integer")
        previous, held = -1, 0
        for offset, word in self.sparse:
            if type(offset) is not int or not previous < offset < self.duration:
                raise ValueError("sparse offsets must increase within duration")
            if type(word) is not int or not 0 < word < 1 << 21 or not word & 3:
                raise ValueError("sparse words must encode shots")
            if ((word >> 2) & 1023) >= 640 or ((word >> 12) & 511) >= 480:
                raise ValueError("new shot coordinates must fit the live client")
            if offset == previous + 1 and word & held:
                raise ValueError("repeated buttons require a neutral release frame")
            previous, held = offset, word & 3
        if self.family == "wait" and self.sparse:
            raise ValueError("a wait macro cannot contain shots")
        if self.family != "wait" and not self.sparse:
            raise ValueError("a shot macro must contain a shot")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _pos(body: Mapping) -> tuple[float, float]:
    return (_num(body.get("effect_x", body.get("x"))),
            _num(body.get("effect_y", body.get("y"))))


def _group(body: Mapping) -> int:
    return max(0, int(_num(body.get("chain_id"))))


def _color(body: Mapping) -> int:
    return int(_num(body.get("color"), -1))


def _key(body: Mapping) -> tuple:
    x, y = _pos(body)
    return (_color(body), -y, x, _num(body.get("size")),
            _num(body.get("projectile_hits")), str(body.get("shape", "")))


def _value(history: Any, group: int, live: int) -> float:
    groups = getattr(history, "groups", history)
    if not isinstance(groups, Mapping):
        groups = {}
    record = groups.get(group, groups.get(str(group), {}))
    get = record.get if isinstance(record, Mapping) else lambda name, default: getattr(record, name, default)
    members = max(live, _num(get("cumulative_members", live)))
    clears = max(0, _num(get("successful_clears", 0)))
    return members * (clears + 1)


def _color_round_robin(pairs: list[tuple]) -> list[tuple]:
    buckets: dict[int, list[tuple]] = defaultdict(list)
    for pair in pairs:
        buckets[_color(pair[0])].append(pair)
    # First appearance follows global priority; colors do not get numeric bias.
    return [bucket[index] for index in range(max(map(len, buckets.values()), default=0))
            for bucket in buckets.values() if index < len(bucket)]


def propose_macros(observation: Mapping[str, Any], history: Any, duration: int = 96,
                   cap: int = 32, rng: Any = None) -> list[GrowthMacro]:
    """Reserve growth, opening, protection and scored cashout alternatives.

    A wait advances a whole segment, allowing new inventory to arrive before
    reobservation. A held firing button gets one neutral frame before its shot.
    History changes target priorities, never introduces fixed actor IDs.
    ``rng`` may be a seeded ``random.Random`` to vary the angular/radial grid.
    """
    if type(duration) is not int or duration < 1:
        raise ValueError("duration must be a positive integer")
    if type(cap) is not int:
        raise ValueError("cap must be an integer")
    if cap <= 0:
        return []
    result = [GrowthMacro(f"wait-{duration}", "wait", (), duration)]
    if cap == 1:
        return result
    pieces = sorted((body for body in observation.get("bodies", ())
                     if isinstance(body, Mapping) and body.get("kind") == "piece"
                     and body.get("lifecycle") != "deleted"
                     and _num(body.get("size")) > 0), key=_key)
    carriers = [body for body in pieces if _group(body)]
    counts = Counter(_group(body) for body in carriers)
    priorities = {_group(body): _value(history, _group(body), counts[_group(body)])
                  for body in carriers}
    fresh = [body for body in pieces if not _group(body)
             and body.get("lifecycle") in _FRESH and 0 <= _color(body) <= 5
             and _num(body.get("rot_timer")) == 0]
    triggers = [body for body in pieces if not _group(body)
                and (_num(body.get("rot_timer")) > 0 or body.get("lifecycle") == "rotten")]

    extend = [(source, target) for source in fresh for target in carriers
              if _color(source) == _color(target)]
    extend.sort(key=lambda pair: (-priorities[_group(pair[1])],
                                  math.dist(_pos(pair[0]), _pos(pair[1])), _key(pair[0]), _key(pair[1])))
    opening = [(source, target) for source in fresh for target in fresh if source is not target
               and _color(source) == _color(target)]
    opening.sort(key=lambda pair: (math.dist(_pos(pair[0]), _pos(pair[1])), _key(pair[0]), _key(pair[1])))
    # A rot-running body can qualify a group burst. Fresh mismatches alone do
    # not, so they are never advertised as a cashout trigger.
    cashout = [(source, target) for source in triggers for target in carriers]
    cashout.sort(key=lambda pair: (-priorities[_group(pair[1])],
                                   math.dist(_pos(pair[0]), _pos(pair[1])), _key(pair[0])))
    protect = [(source, target) for source in triggers + fresh for target in carriers
               if (_color(source) != _color(target) or source in triggers)
               and math.dist(_pos(source), _pos(target)) < 240 + _num(target.get("size"))]
    protect.sort(key=lambda pair: (-priorities[_group(pair[1])],
                                   math.dist(_pos(pair[0]), _pos(pair[1])), _key(pair[0])))
    pairs_by_family = {"extend": _color_round_robin(extend), "open": _color_round_robin(opening),
                       "protect": _color_round_robin(protect), "cashout": _color_round_robin(cashout)}

    def candidates(family: str):
        # Keep multiple sources, without spending the whole cap on one color.
        pairs = pairs_by_family[family][:4]
        if not pairs:
            return
        angles = [0, -30, 30, -60, 60, -90, 90, 180, -120, 120, -150, 150]
        radii = [.60, .35, .90, 1.15, .15]
        jitter = 0.0
        if rng is not None:
            jitter = rng.uniform(-15, 15)
            tail = angles[1:]
            rng.shuffle(tail)
            angles[1:] = tail
            rng.shuffle(radii)
        for index in range(len(angles) * len(radii) * 2):
            angle = math.radians(angles[(index // 2) % len(angles)] + jitter)
            radius = radii[(index // (2 * len(angles))) % len(radii)]
            for pair_index, (source, target) in enumerate(pairs):
                strength = 2 if (index + pair_index) % 2 == 0 else 1
                sx, sy = _pos(source)
                tx, ty = _pos(target)
                # Put the projectile opposite the desired displacement. For
                # protection, send the threat away from the carrier instead.
                base = math.atan2(sy - ty, sx - tx)
                if family == "protect":
                    base += math.pi
                x = round(sx + math.cos(base + angle) * _num(source.get("size")) * radius)
                y = round(sy + math.sin(base + angle) * _num(source.get("size")) * radius)
                if not (0 <= x < 640 and 0 <= y < 480):
                    continue
                # Conservatively exclude clicks inside any carrier's bounding
                # circle. This guards direct damage, not every later collision.
                if any(math.dist((x, y), _pos(body)) <= .72 * _num(body.get("size")) + 8
                       for body in carriers):
                    continue
                offset = int(bool(observation.get("left_held" if strength == 1 else "right_held")))
                if offset >= duration:
                    continue
                word = (y << 12) | (x << 2) | strength
                label = f"{family}-p{pair_index}-r{radius:g}-a{angles[(index // 2) % len(angles)]}-k{strength}"
                yield GrowthMacro(label, family, ((offset, word),), duration)

    streams = {family: iter(candidates(family)) for family in _FAMILIES[1:]}
    seen: set[tuple] = {()}

    def take(family: str) -> bool:
        stream = streams.get(family)
        if stream is None:
            return False
        for macro in stream:
            if macro.sparse not in seen:
                seen.add(macro.sparse)
                result.append(macro)
                return True
        streams.pop(family, None)
        return False

    # Family reservations survive even very small caps; remaining slots favor
    # growth without erasing ordinary pair creation or deliberate cashout.
    for family in ("extend", "open", "protect", "cashout"):
        if len(result) == cap:
            return result
        take(family)
    cycle = ("extend", "open", "extend", "protect", "extend", "open", "extend", "cashout")
    while len(result) < cap and streams:
        for family in cycle:
            if len(result) == cap:
                break
            take(family)
    return result
