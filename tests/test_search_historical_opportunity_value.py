import json

import pytest

from irisu_search.historical.opportunity_value import (
    ComboHistory, OpportunityConfig, evaluate_opportunity, nominal_score,
)


def event(tick, kind, body, value=0):
    return dict(tick=tick, kind=kind, a=body, value=value)


def piece(ident, *, group=0, color=0, x=300, y=120, **changes):
    return dict(id=ident, kind="piece", chain_id=group, color=color, x=x, y=y,
                size=54, projectile_hits=0, rot_timer=0, remaining_lifetime=10000,
                lifecycle="confirmed" if group else "scripted_falling") | changes


def observation(*bodies, **changes):
    return dict(tick=85558, score=254112, level=88, qualifying_clear_count=870,
                terminated=False, truncated=False, bodies=list(bodies)) | changes


def mature_history(gid=399, members=15, clears=11, last_join=85520):
    return ComboHistory.from_groups([dict(chain_id=gid, cumulative_members=members,
        successful_clears=clears, first_tick=81803, last_join_tick=last_join,
        member_ids=list(range(1, members + 1)))])


def test_nominal_score_matches_native_fixtures_and_audited_anchor_payouts():
    assert nominal_score(1, 1, 1) == 4
    assert nominal_score(1, 1, 1, 2) == 8
    assert nominal_score(2, 1, 25) == 162
    # Full exact replay audit: group 399, large carrier body 6276.
    assert nominal_score(88, 12, 15, 2) == 17915
    assert nominal_score(88, 14, 17, 2) == 23428


def test_historical_counters_survive_single_carrier_and_branches_are_independent():
    history = ComboHistory.from_events([
        event(10, "chain_joined", 1, 77), event(10, "chain_joined", 2, 77),
        event(15, "confirmed", 2, 1), event(15, "cleared", 2, 1),
        event(20, "chain_joined", 3, 77), event(25, "confirmed", 3, 2),
        event(25, "cleared", 3, 2),
    ])
    assert history.groups[77]["cumulative_members"] == 3
    assert history.groups[77]["successful_clears"] == 2
    assert history.memberships == {1: 77}
    branch = history.copy()
    branch.update([event(30, "chain_joined", 4, 77), event(31, "confirmed", 4, 3)])
    assert history.groups[77]["cumulative_members"] == 3
    assert branch.groups[77]["cumulative_members"] == 4
    assert branch.groups[77]["successful_clears"] == 3
    restored = ComboHistory.from_dict(json.loads(json.dumps(branch.to_dict())))
    assert restored.groups == branch.groups and restored.memberships == branch.memberships


def test_clear_counter_is_not_number_of_scored_bodies_and_duplicate_join_is_ignored():
    history = ComboHistory.from_events([
        event(1, "chain_joined", 1, 9), event(1, "chain_joined", 1, 9),
        event(1, "chain_joined", 2, 9), event(2, "confirmed", 1, 1),
        event(2, "cleared", 1, 1), event(2, "cleared", 2, 1),
    ])
    assert history.groups[9]["cumulative_members"] == 2
    assert history.groups[9]["successful_clears"] == 1
    assert not history.memberships


def test_live_mature_carrier_and_feeders_can_outweigh_large_realized_score_debt():
    waiting = observation(piece(1, group=399, size=140), piece(30, x=290), piece(31, x=320))
    delayed = evaluate_opportunity(waiting, mature_history(), elapsed_ticks=316)
    cashed = evaluate_opportunity(observation(score=272664), mature_history())
    assert delayed.realized_score == cashed.realized_score - 18552
    assert delayed.objective > cashed.objective
    assert delayed.cashout_value > 15000 and delayed.extension_value > 10000
    assert delayed.guaranteed_future_value == 0  # This ranking is explicitly optimistic.
    assert delayed.group_details[0]["feeder_ids"]


def test_finished_and_destroyed_groups_have_no_potential_even_with_large_history():
    history = mature_history()
    for bodies in ([], [piece(1, group=399, lifecycle="deleted")],
                   [piece(1, group=399, projectile_hits=2)],
                   [piece(1, group=399, remaining_lifetime=1)]):
        value = evaluate_opportunity(observation(*bodies), history)
        assert value.potential == 0 and value.objective == value.realized_score


@pytest.mark.parametrize("terminal", [dict(terminated=True), dict(truncated=True),
                                      dict(level=100), dict(qualifying_clear_count=990)])
def test_terminal_states_never_credit_phantom_final_frame_score_or_potential(terminal):
    value = evaluate_opportunity(observation(piece(1, group=399, size=140), piece(30),
        score=389217, canonical_score=389018, **terminal), mature_history(), elapsed_ticks=200)
    assert value.terminal
    assert value.realized_score == value.objective == 389018
    assert value.potential == value.time_cost == 0


def test_stationary_waiting_cannot_accumulate_potential_or_improve_objective():
    board = observation(piece(1, group=399), piece(30, x=310))
    first = evaluate_opportunity(board, mature_history(), elapsed_ticks=10)
    later = evaluate_opportunity(board | {"tick": board["tick"] + 1000}, mature_history(), elapsed_ticks=1010)
    assert later.cashout_value == first.cashout_value
    assert later.extension_value < first.extension_value
    assert later.objective < first.objective - 1000


def test_new_combo_openings_need_visible_compatible_inventory():
    history = ComboHistory()
    pair = evaluate_opportunity(observation(piece(1), piece(2, x=320)), history)
    five = evaluate_opportunity(observation(*(piece(i, x=280 + 8 * i) for i in range(1, 6))), history)
    assert pair.opening_value > 0
    assert five.opening_value > pair.opening_value * 4
    for bodies in ([piece(1)], [piece(1), piece(2, color=1)],
                   [piece(1, x=10), piece(2, x=620)],
                   [piece(1), piece(2, lifecycle="rotten")]):
        assert evaluate_opportunity(observation(*bodies), history).opening_value == 0


def test_visible_inventory_is_not_double_counted_across_extensions_or_openings():
    history = mature_history()
    history.groups[400] = dict(chain_id=400, cumulative_members=10, successful_clears=7,
                               last_join_tick=85520, member_ids=(20,))
    board = observation(piece(1, group=399), piece(20, group=400, x=310),
                        *(piece(i, x=290 + i) for i in range(30, 40)))
    value = evaluate_opportunity(board, history)
    allocated = [ident for row in value.group_details for ident in row["feeder_ids"]]
    allocated += [ident for row in value.opening_details for ident in row["member_ids"]]
    assert len(allocated) == len(set(allocated)) <= 10
    assert all(len(row["feeder_ids"]) <= 4 for row in value.group_details)


def test_endgame_budget_disables_extensions_that_cannot_cash_out():
    value = evaluate_opportunity(observation(piece(1, group=399), piece(30), piece(31),
        qualifying_clear_count=988), mature_history())
    assert value.cashout_value > 0
    assert value.extension_value == value.opening_value == 0
    last = evaluate_opportunity(observation(piece(1, group=399), piece(30), piece(31),
        qualifying_clear_count=989), mature_history())
    assert not last.terminal and last.potential == 0


def test_history_uncertainty_and_danger_discount_potential_without_score_gates():
    board = observation(piece(1, group=399, size=140), piece(30, x=310))
    known = evaluate_opportunity(board, mature_history())
    unknown = evaluate_opportunity(board, ComboHistory())
    low_gauge = evaluate_opportunity(board | {"gauge": 1}, mature_history())
    risky = evaluate_opportunity(observation(piece(1, group=399, y=450, rot_timer=35),
                                           piece(30, x=310)), mature_history())
    assert known.potential > unknown.potential > 0
    assert low_gauge.potential == known.potential
    assert risky.potential < known.potential
    assert not unknown.group_details[0]["history_known"]


def test_validation_and_finite_serializable_output():
    for kwargs in ({"feed_radius": 0}, {"elapsed_tick_cost": float("nan")},
                   {"max_extensions": 1.2}, {"cashout_confidence": 1.1}):
        with pytest.raises(ValueError):
            OpportunityConfig(**kwargs)
    with pytest.raises(ValueError):
        evaluate_opportunity(observation(), ComboHistory(), elapsed_ticks=-1)
    with pytest.raises(ValueError, match="after the evaluated state"):
        evaluate_opportunity(observation(piece(1, group=399)), mature_history(last_join=90000))
    value = evaluate_opportunity(observation(piece(1, group=399), piece(30)), mature_history())
    assert json.loads(json.dumps(value.to_dict(), allow_nan=False))["objective"] > 0
