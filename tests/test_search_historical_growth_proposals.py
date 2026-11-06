from copy import deepcopy
import math
import random
from types import SimpleNamespace

import pytest

from irisu_search.historical.growth_proposals import GrowthMacro, propose_macros


def piece(id, color, x, y, group=0, **extra):
    return dict(id=id, kind="piece", lifecycle="dynamic_fresh", color=color,
                x=x, y=y, size=32, chain_id=group, **extra)


def unpack(macro):
    offset, word = macro.sparse[0]
    return offset, word & 3, (word >> 2) & 1023, (word >> 12) & 511


def board():
    return {"bodies": [piece(1, 0, 260, 180), piece(2, 0, 420, 170, 10),
                       piece(3, 1, 160, 300), piece(4, 1, 245, 315),
                       piece(5, 2, 350, 330, rot_timer=30)]}


def test_every_available_family_keeps_a_slot_and_strengths_are_diverse():
    choices = propose_macros(board(), {}, cap=32)
    assert len(choices) == 32
    assert {m.family for m in choices[:5]} == {"wait", "extend", "open", "protect", "cashout"}
    assert len({m.sparse for m in choices}) == len(choices)
    assert {unpack(m)[1] for m in choices if m.family == "extend"} == {1, 2}
    assert {unpack(m)[1] for m in choices if m.family == "open"} == {1, 2}
    assert all(m.duration == 96 for m in choices)
    assert len(propose_macros(board(), {}, cap=3)) == 3
    assert propose_macros(board(), {}, cap=0) == []


def test_historical_value_prioritizes_a_single_mature_carrier():
    observation = {"bodies": [piece(1, 0, 200, 150), piece(2, 0, 450, 150, 10),
                              piece(3, 1, 200, 350), piece(4, 1, 450, 350, 20)]}
    history = SimpleNamespace(groups={10: {"cumulative_members": 2, "successful_clears": 0},
                                      20: SimpleNamespace(cumulative_members=17, successful_clears=15)})
    choices = propose_macros(observation, history, cap=2)
    assert choices[1].family == "extend"
    assert unpack(choices[1])[3] == 350  # Mature group's source gets the reserved slot.
    remapped = deepcopy(observation)
    for body in remapped["bodies"]:
        body["id"] += 9000
        if body["chain_id"]:
            body["chain_id"] += 100
    remapped["bodies"].reverse()
    remapped_history = {110: history.groups[10], 120: history.groups[20]}
    assert choices == propose_macros(remapped, remapped_history, cap=2)


def test_lateral_geometry_and_no_direct_carrier_shots():
    observation = {"bodies": [piece(1, 0, 200, 250), piece(2, 0, 430, 250, 1)]}
    choices = propose_macros(observation, {1: {"cumulative_members": 17}}, cap=32)
    first = next(m for m in choices if m.family == "extend")
    _, _, x, y = unpack(first)
    assert x < 200 and y == 250  # Offset lies along actual destination vector.
    points = [(unpack(m)[2], unpack(m)[3]) for m in choices if m.sparse]
    assert any(y < 250 for _, y in points) and any(y > 250 for _, y in points)
    assert all(math.dist((x, y), (430, 250)) > .72 * 32 + 8 for x, y in points)
    assert propose_macros({"bodies": [piece(1, 0, 320, 180, 1)]}, {}, cap=32) == [
        GrowthMacro("wait-96", "wait", (), 96)]


def test_cashout_requires_rot_and_fresh_mismatch_is_only_protection():
    observation = {"bodies": [piece(1, 0, 220, 200), piece(2, 1, 400, 200, 1)]}
    choices = propose_macros(observation, {}, cap=16)
    assert {m.family for m in choices} == {"wait", "protect"}
    observation["bodies"][0]["rot_timer"] = 1
    choices = propose_macros(observation, {}, cap=16)
    assert {m.family for m in choices} == {"wait", "protect", "cashout"}


def test_held_buttons_release_and_repeated_macro_shots_require_release():
    observation = {**board(), "left_held": True, "right_held": True}
    assert all(unpack(m)[0] == 1 for m in propose_macros(observation, {}, duration=12) if m.sparse)
    assert len(propose_macros(observation, {}, duration=1)) == 1
    word = (200 << 12) | (200 << 2) | 2
    GrowthMacro("two", "open", ((0, word), (2, word)), 3)
    with pytest.raises(ValueError, match="release"):
        GrowthMacro("two", "open", ((0, word), (1, word)), 3)
    with pytest.raises(ValueError, match="within"):
        GrowthMacro("late", "extend", ((3, word),), 3)
    with pytest.raises(ValueError, match="live client"):
        GrowthMacro("outside", "extend", ((0, (500 << 12) | 2),), 3)


def test_seeded_geometry_diversity_is_reproducible_and_inputs_unchanged():
    observation = board()
    history = {10: {"cumulative_members": 15, "successful_clears": 12}}
    before = deepcopy((observation, history))
    a = propose_macros(observation, history, rng=random.Random(123))
    b = propose_macros(observation, history, rng=random.Random(123))
    c = propose_macros(observation, history, rng=random.Random(321))
    assert a == b and a != c
    assert (observation, history) == before
    for macro in a + c:
        if macro.sparse:
            _, _, x, y = unpack(macro)
            assert 0 <= x < 640 and 0 <= y < 480
    with pytest.raises(ValueError):
        propose_macros(observation, history, duration=0)
