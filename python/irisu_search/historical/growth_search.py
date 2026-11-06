"""Recovered October event beam nodes and frontier selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from .opportunity_value import ComboHistory, _health
from .common import snapshot, trace_bytes

@dataclass
class Node:
    suffix: tuple[int, ...]
    observation: dict
    final: dict
    history: ComboHistory
    value: object
    family: str
    labels: tuple[str, ...]
    maximum_score_debt: int = 0

    @property
    def natural(self):
        return bool(self.final["terminated"] and not self.final["truncated"])

    @property
    def identifier(self):
        return hashlib.sha256(trace_bytes(self.suffix)).hexdigest()[:20]



def endpoint_key(node: Node):
    """Deduplicate exact observable endpoints and history, never score alone.

    Action prefix remains in the key because public observations omit hidden
    physics and random-generator state. Distinct histories must not be merged.
    """
    return node.identifier



def live_achievements(node: Node, root_history: ComboHistory):
    live = {int(b.get("chain_id", 0)) for b in node.observation.get("bodies", ())
            if b.get("kind") == "piece" and _health(b) > 0}
    existing, opened = [], []
    for gid in live - {0}:
        row = node.history.groups.get(gid, {})
        members = int(row.get("cumulative_members", 0))
        old = root_history.groups.get(gid)
        if old:
            existing.append(max(0, members - int(old["cumulative_members"])))
        else:
            opened.append(members)
    return max(existing, default=0), max(opened, default=0), sum(max(0, n-2) for n in opened)



def select_frontier(nodes, width, root_history, *, preserve_reference=True):
    """Round-robin protected growth/opening/value/score/gauge slots.

    There is deliberately no best-score-minus-X floor or wait-relative debt
    veto. A finite horizon and explicit potential decay bound patience.
    """
    if width < 1:
        raise ValueError("beam width must be positive")
    viable = [n for n in nodes if not (n.final["terminated"] or n.final["truncated"])]
    achievements = {id(n): live_achievements(n, root_history) for n in viable}
    keys = (
        lambda n: (n.value.objective, n.final["score"]),
        lambda n: (achievements[id(n)][0], n.value.extension_value, n.value.objective),
        lambda n: (achievements[id(n)][1:], n.value.opening_value, n.value.objective),
        lambda n: (n.value.extension_value, n.value.cashout_value, n.value.objective),
        lambda n: (n.final["score"], n.final["gauge"]),
        lambda n: (n.value.opening_value, n.value.objective),
        lambda n: (n.final["gauge"], n.value.objective),
    )
    pools = [sorted(viable, key=key, reverse=True) for key in keys]
    # Keep one exact inherited route as a measured control and reusable parent.
    reference = next((n for n in viable if preserve_reference and n.labels and
                      all(label == "reference" for label in n.labels)), None)
    selected = [reference] if reference is not None else []
    seen = {endpoint_key(reference)} if reference is not None else set()
    if len(selected) == width:
        return selected
    for rank in range(len(viable)):
        for pool in pools:
            node = pool[rank]
            key = endpoint_key(node)
            if key not in seen:
                seen.add(key)
                selected.append(node)
                if len(selected) == width:
                    return selected
    return selected



def node_report(node, root_history):
    new_sizes = [int(row["cumulative_members"]) for gid, row in node.history.groups.items()
                 if gid not in root_history.groups]
    return dict(id=node.identifier, final=node.final, family=node.family, labels=node.labels,
                maximum_score_debt=node.maximum_score_debt,
                achievements=live_achievements(node, root_history),
                new_combos=dict(largest=max(new_sizes, default=0),
                                at_least_three=sum(n >= 3 for n in new_sizes),
                                at_least_five=sum(n >= 5 for n in new_sizes)),
                value=asdict(node.value))



def replay_events(env, actions, start, stop, history, budget=None):
    """Resident equivalent of the original event-preserving neutral-span replay."""
    from .event_options import _step
    current = env.observation()
    final = snapshot(current)
    tick = start
    while tick < stop and not (final["terminated"] or final["truncated"]):
        if budget:
            budget.check()
        word = actions[tick] if tick < len(actions) else 0
        end = tick + 1
        if not word:
            while end < stop and not (actions[end] if end < len(actions) else 0):
                end += 1
        current, final, events = _step(env, word, end-tick)
        history.update(events)
        tick = int(final["tick"])
    return current, final
