"""Event feedback, replay boundaries and exact option reconstruction."""
from copy import deepcopy
import random

import pytest

from irisu_search.historical import event_options as options
from irisu_search.historical.growth_proposals import GrowthMacro
from irisu_search.historical.opportunity_value import ComboHistory


def piece(body_id, x, y=220, color=0, group=0, **extra):
    return dict(id=body_id, kind="piece", lifecycle="dynamic_fresh", x=x, y=y,
                size=32, chain_id=group, color=color, age_ticks=200,
                projectile_hits=0, remaining_lifetime=1000, rot_timer=0, **extra)


def board(tick=0):
    return dict(tick=tick, score=10000, gauge=8000, level=90, highest_chain=12,
        qualifying_clear_count=890, terminated=0, truncated=0, left_held=False,
        right_held=False, bodies=[piece(1, 200), piece(2, 430, group=7),
            piece(3, 150, 320, color=1), piece(4, 240, 320, color=1),
            {**piece(5, 350, 300, color=2), "rot_timer": 8}])


def ledger():
    return ComboHistory.from_groups([dict(chain_id=7, first_tick=0, last_join_tick=0,
        cumulative_members=12, successful_clears=8, member_ids=[2])])


class Fake:
    def __init__(self, observation, callback=None, terminal_tick=None, truncate=False):
        self.observation = deepcopy(observation)
        self.callback = callback
        self.terminal_tick = terminal_tick
        self.truncate = truncate
        self.calls = []

    def step(self, word, duration):
        before = self.observation["tick"]
        end = min(before+duration, self.terminal_tick) if self.terminal_tick else before+duration
        self.observation["tick"] = end
        self.observation["left_held"] = bool(word & 1)
        self.observation["right_held"] = bool(word & 2)
        events = self.callback(self.observation, word, before, end) if self.callback else []
        final = options.snapshot(self.observation)
        if self.terminal_tick and end == self.terminal_tick:
            final["truncated" if self.truncate else "terminated"] = 1
            if not self.truncate:
                final.update(canonical_score=9900, canonical_level=100,
                             canonical_highest_chain=12, canonical_clears=990)
            self.observation.update(final)
        self.calls.append((before, word, duration, end))
        return deepcopy(self.observation), final, events


class ProbeFake(Fake):
    def fast_checkpoint(self):
        from contextlib import contextmanager
        @contextmanager
        def checkpoint():
            class Checkpoint:
                @contextmanager
                def branch(inner):
                    yield Fake(self.observation, self.callback)
            yield Checkpoint()
        return checkpoint()


@pytest.fixture
def fake_step(monkeypatch):
    monkeypatch.setattr(options, "_step", lambda env, word, duration: env.step(word, duration))


def test_proposals_reserve_conservative_families_and_do_not_mutate():
    obs, history = board(), ledger()
    before = deepcopy(obs), history.to_dict()
    a = options.propose_options(obs, history, 96, 32, random.Random(7))
    b = options.propose_options(obs, history, 96, 32, random.Random(7))
    assert a == b and len(a) == 32
    assert {o.family for o in a[:5]} == {"wait", "extend", "open", "protect", "cashout"}
    assert {o.mode for o in a} >= {"macro", "wait_spawn", "wait_join", "feed_spawn", "paired", "relay"}
    assert (obs, history.to_dict()) == before
    assert len(options.propose_options(obs, history, 96, 1)) == 1
    assert options.propose_options(obs, history, 96, 0) == []


def test_macro_stops_at_actual_terminal_and_keeps_canonical_metadata(fake_step):
    obs, history = board(10), ledger()
    env = Fake(obs, terminal_tick=17)
    option = options.EventOption("wait", "wait", 30, macro=GrowthMacro("wait", "wait", (), 30))
    result = options.execute_option(env, obs, history, option, stop_tick=50)
    assert result["actions"] == (0,)*7
    assert result["final"]["tick"] == 17 and result["natural"]
    assert result["final"]["canonical_score"] == 9900
    assert result["diagnostics"]["stop_reason"] == "terminal"
    assert history.to_dict() == ledger().to_dict()


def test_stop_tick_and_partial_truncation_are_exact(fake_step):
    obs = board(10)
    option = options.EventOption("wait", "wait", 30, macro=GrowthMacro("wait", "wait", (), 30))
    result = options.execute_option(Fake(obs), obs, ledger(), option, stop_tick=19)
    assert len(result["actions"]) == 9 and result["final"]["tick"] == 19
    result = options.execute_option(Fake(obs, terminal_tick=17, truncate=True), obs, ledger(), option)
    assert len(result["actions"]) == 7 and result["final"]["truncated"]
    assert not result["natural"] and "canonical_score" not in result["final"]
    with pytest.raises(ValueError, match="precedes"):
        options.execute_option(Fake(obs), obs, ledger(), option, stop_tick=9)


def test_spawn_wait_ends_on_observed_new_inventory(fake_step):
    def spawn(obs, word, start, end):
        if end >= 7 and not any(b["id"] == 6 for b in obs["bodies"]):
            obs["bodies"].append(piece(6, 250, 30))
        return []
    obs = board()
    result = options.execute_option(Fake(obs, spawn), obs, ledger(),
        options.EventOption("spawn", "wait", 40, "wait_spawn", observation_period=4))
    assert result["actions"] == (0,)*8
    assert result["diagnostics"]["stop_reason"] == "spawn"


def test_paired_shot_reaims_moving_source_and_releases_held_button(fake_step):
    def move(obs, word, start, end):
        obs["bodies"][0]["x"] += end-start
        return []
    obs = board()
    obs["right_held"] = True
    result = options.execute_option(Fake(obs, move), obs, ledger(),
        options.EventOption("paired", "extend", 16, "paired", source_id=1,
            target_id=2, group_id=7, min_wait=4))
    shots = result["diagnostics"]["shots"]
    assert len(shots) == 2 and shots[0]["tick"] == 1
    assert shots[0]["source_position"][0] < shots[1]["source_position"][0]
    assert shots[0]["word"] != shots[1]["word"]
    assert result["actions"][0] == 0
    assert all(result["actions"][i-1] == 0 for i, w in enumerate(result["actions"]) if w and i)


def test_join_event_updates_copied_history_and_finishes_feed(fake_step):
    def join(obs, word, start, end):
        if end >= 5 and not obs["bodies"][0]["chain_id"]:
            obs["bodies"][0]["chain_id"] = 7
            return [dict(kind="chain_joined", tick=end, a=1, b=2, value=7)]
        return []
    obs, history = board(), ledger()
    result = options.execute_option(Fake(obs, join), obs, history,
        options.EventOption("paired", "extend", 40, "paired", source_id=1, target_id=2, group_id=7))
    assert result["final"]["tick"] == 5
    assert result["history"].groups[7]["cumulative_members"] == 13
    assert history.groups[7]["cumulative_members"] == 12
    assert result["diagnostics"]["stop_reason"] == "join"


def test_contact_reobserves_before_nominal_second_shot_delay(fake_step):
    def contact(obs, word, start, end):
        if start < 3 <= end:
            return [dict(kind="contact", tick=3, a=1, b=99, value=0)]
        return []
    obs = board()
    result = options.execute_option(Fake(obs, contact), obs, ledger(),
        options.EventOption("paired", "extend", 24, "paired", source_id=1,
            target_id=2, group_id=7, min_wait=16))
    assert [s["tick"] for s in result["diagnostics"]["shots"]] == [0, 5]


def test_feed_waits_past_irrelevant_spawn_for_matching_color(fake_step):
    def spawn(obs, word, start, end):
        if end >= 4 and not any(b["id"] == 6 for b in obs["bodies"]):
            obs["bodies"].append(piece(6, 250, 50, color=2))
        if end >= 12 and not any(b["id"] == 7 for b in obs["bodies"]):
            obs["bodies"].append(piece(7, 250, 70, color=0))
        return []
    obs = board()
    obs["bodies"] = [b for b in obs["bodies"] if b["chain_id"]]
    result = options.execute_option(Fake(obs, spawn), obs, ledger(),
        options.EventOption("feed", "extend", 20, "feed_spawn", group_id=7))
    assert result["diagnostics"]["shots"][0]["tick"] == 12
    assert result["diagnostics"]["shots"][0]["source_id"] == 7


def test_reposition_requires_zero_hit_and_concrete_pressure():
    obs = board()
    choices = options.propose_options(obs, ledger(), cap=64)
    # Existing nearby mismatched trigger is collision pressure.
    assert any(o.mode == "reposition" for o in choices)
    obs["bodies"][1]["projectile_hits"] = 1
    assert not any(o.mode == "reposition" for o in options.propose_options(obs, ledger(), cap=64))
    obs["bodies"] = [piece(2, 430, group=7)]
    assert not any(o.mode == "reposition" for o in options.propose_options(obs, ledger(), cap=64))


def test_relay_keeps_joined_carrier_and_routes_trigger_to_old_member(fake_step):
    def relay(obs, word, start, end):
        if end >= 5 and not obs["bodies"][0]["chain_id"]:
            obs["bodies"][0]["chain_id"] = 7
            return [dict(kind="chain_joined", tick=end, a=1, b=2, value=7)]
        if word and start >= 5:
            obs["bodies"][1]["lifecycle"] = "deleted"
            obs["score"] += 5000
            return [dict(kind="cleared", tick=end, a=2, b=5, value=9)]
        return []
    obs, history = board(), ledger()
    result = options.execute_option(Fake(obs, relay), obs, history,
        options.EventOption("relay", "relay", 40, "relay", source_id=1,
            target_id=2, group_id=7, max_shots=3))
    assert result["diagnostics"]["stop_reason"] == "relay"
    assert len(result["diagnostics"]["shots"]) == 2
    assert result["diagnostics"]["shots"][-1]["source_id"] == 5
    assert result["history"].memberships[1] == 7 and 2 not in result["history"].memberships
    assert result["history"].groups[7]["successful_clears"] == 9


def test_relay_does_not_mistake_another_members_join_for_its_selected_feeder(fake_step):
    def other_join(obs, word, start, end):
        if end >= 5 and not any(b["id"] == 6 for b in obs["bodies"]):
            obs["bodies"].append(piece(6, 460, group=7))
            return [dict(kind="chain_joined", tick=end, a=6, b=2, value=7)]
        return []
    obs = board()
    result = options.execute_option(Fake(obs, other_join), obs, ledger(),
        options.EventOption("relay", "relay", 24, "relay", source_id=1,
            target_id=2, group_id=7, max_shots=3))
    assert result["diagnostics"]["shots"]
    assert {s["source_id"] for s in result["diagnostics"]["shots"]} == {1}


def test_guard_probe_rejects_unscored_loss_and_never_commits_probe_frames(fake_step):
    from contextlib import contextmanager
    class Checkpoint:
        def __init__(self, env):
            self.env = env
        @contextmanager
        def branch(self):
            yield Fake(self.env.observation, self.env.callback)
    class ProbeEnv(Fake):
        @contextmanager
        def fast_checkpoint(self):
            yield Checkpoint(self)
    def destroy(obs, word, start, end):
        if word:
            obs["bodies"][1]["lifecycle"] = "deleted"
            return [dict(kind="destroyed", tick=end, a=2, b=0, value=0)]
        return []
    obs = board()
    env = ProbeEnv(obs, destroy)
    option = options.EventOption("reposition", "reposition", 40, "reposition", source_id=2,
        target_id=5, group_id=7, max_shots=1)
    result = options.execute_option(env, obs, ledger(), option)
    assert result["diagnostics"]["stop_reason"] == "guard_rejected"
    assert not result["actions"] and env.observation == obs and not env.calls


def test_budget_checked_before_every_committed_step(fake_step):
    class Budget:
        checks = 0
        def check(self):
            self.checks += 1
            if self.checks == 3:
                raise TimeoutError("budget")
    obs, budget, env = board(), Budget(), Fake(board())
    option = options.EventOption("wait", "wait", 30, macro=GrowthMacro("wait", "wait", (), 30))
    with pytest.raises(TimeoutError):
        options.execute_option(env, obs, ledger(), option, budget)
    assert env.observation["tick"] == 8 and len(env.calls) == 2


def test_inventory_waits_for_three_matching_fresh_bodies_then_feeds(fake_step):
    def spawn(obs, word, start, end):
        for tick, body_id in ((4, 6), (8, 7)):
            if end >= tick and not any(b["id"] == body_id for b in obs["bodies"]):
                obs["bodies"].append(piece(body_id, 100+body_id*10, 60))
        return []
    obs = board()
    result = options.execute_option(Fake(obs, spawn), obs, ledger(),
        options.EventOption("inventory-g0-n3", "extend", 20, "wait_inventory", group_id=7))
    assert result["diagnostics"]["shots"][0]["tick"] == 8


def test_inventory_stops_when_wait_changes_level(fake_step):
    def advance_level(obs, word, start, end):
        if end >= 4:
            obs["level"] += 1
        return []
    obs = board()
    result = options.execute_option(Fake(obs, advance_level), obs, ledger(),
        options.EventOption("inventory", "extend", 20, "wait_inventory", group_id=7))
    assert result["actions"] == (0,)*4
    assert result["diagnostics"]["stop_reason"] == "inventory_health_or_phase"


def minor_board():
    obs, history = board(), ledger()
    obs["bodies"].append(piece(8, 100, 420, color=2, group=8))
    history.groups[8] = dict(chain_id=8, first_tick=0, last_join_tick=0,
        cumulative_members=2, successful_clears=0, member_ids=(8,))
    history.memberships[8] = 8
    return obs, history


def suppression_callback(*, qualifies=True, collateral=False, eject=False):
    def callback(obs, word, start, end):
        source = next(b for b in obs["bodies"] if b["id"] == 8)
        if word and source["lifecycle"] != "deleted":
            source["projectile_hits"] += 1
            if source["projectile_hits"] == 2 or eject:
                source["lifecycle"] = "deleted"
                events = [dict(kind="ejected" if eject else "destroyed", tick=end, a=8, b=0, value=0)]
                if collateral:
                    obs["bodies"][1]["lifecycle"] = "deleted"
                    events.append(dict(kind="destroyed", tick=end, a=2, b=0, value=0))
                return events
        if qualifies and end >= 8 and source["lifecycle"] != "deleted" and not source.get("qualified"):
            source["qualified"] = True
            obs["qualifying_clear_count"] += 1
        return []
    return callback


@pytest.mark.parametrize("qualifies,classification,saving", [(True,"observed_saving",1), (False,"preventative",0)])
def test_minor_suppression_probes_two_hits_and_commits_only_tested_prefix(fake_step, qualifies, classification, saving):
    obs, history = minor_board()
    env = ProbeFake(obs, suppression_callback(qualifies=qualifies))
    option = options.EventOption("suppress-g1-k1", "budget", 12, "suppress_minor",
        source_id=8, group_id=8, radius=.15, strength=1, min_wait=2, probe_ticks=24)
    result = options.execute_option(env, obs, history, option, stop_tick=30)
    assert len(result["actions"]) == 12 and result["final"]["tick"] == 12
    assert [s["tick"] for s in result["diagnostics"]["shots"]] == [0, 2]
    assert result["diagnostics"]["guard"]["qualification_saving"] == saving
    assert result["diagnostics"]["guard"]["classification"] == classification
    assert 8 not in result["history"].memberships and 8 in history.memberships
    assert result["history"].memberships[2] == 7
    assert len(result["diagnostics"]["events"]) == 1  # probe events never duplicate committed events


def test_minor_suppression_rejects_valuable_unscored_loss_and_dominant_target(fake_step):
    obs, history = minor_board()
    option = options.EventOption("suppress", "budget", 12, "suppress_minor", source_id=8,
        group_id=8, radius=.15, strength=1, min_wait=2, probe_ticks=24)
    env = ProbeFake(obs, suppression_callback(collateral=True))
    result = options.execute_option(env, obs, history, option, stop_tick=30)
    assert not result["actions"] and env.observation == obs and not env.calls
    assert result["diagnostics"]["guard"]["reason"] == "valuable_unscored_loss"
    option = options.EventOption("suppress", "budget", 12, "suppress_minor", source_id=2, group_id=7)
    result = options.execute_option(env, obs, history, option, stop_tick=30)
    assert result["diagnostics"]["guard"]["reason"] == "ineligible"


def test_fresh_ejection_requires_exact_ejected_event(fake_step):
    obs, history = minor_board()
    obs["bodies"][-1]["chain_id"] = 0
    history = ledger()
    env = ProbeFake(obs, suppression_callback(qualifies=False, eject=True))
    result = options.execute_option(env, obs, history,
        options.EventOption("eject", "budget", 12, "eject_fresh", source_id=8, probe_ticks=24), stop_tick=30)
    assert result["diagnostics"]["guard"]["accepted"]
    assert result["diagnostics"]["events"][0]["kind"] == "ejected"


def test_relay_spends_only_young_first_hit_to_create_separation(fake_step):
    def separate(obs, word, start, end):
        if word:
            young = obs["bodies"][0]
            young["projectile_hits"] += 1
            young["x"] -= 50
            return [dict(kind="projectile_hit", tick=end, a=99, b=1, value=0)]
        return []
    obs = board()
    obs["bodies"] = [piece(1,300,group=7), piece(2,350,group=7)]
    obs["bodies"][0]["age_ticks"] = 20
    history = ledger()
    history.memberships[1] = 7
    result = options.execute_option(ProbeFake(obs, separate), obs, history,
        options.EventOption("relay-carrier-g0-s90", "relay", 24, "relay_separate",
            source_id=1, target_id=2, group_id=7, relay_angle=90, max_shots=2))
    assert len(result["diagnostics"]["shots"]) == 1
    assert result["diagnostics"]["shots"][0]["purpose"] == "separate_young_carrier"
    assert result["diagnostics"]["relay_guards"][0]["accepted"]
    assert result["observation"]["bodies"][0]["projectile_hits"] == 1
    assert result["observation"]["bodies"][1]["projectile_hits"] == 0
    assert result["history"].memberships[1] == 7


def test_proposed_options_have_unique_coarse_labels():
    obs = board()
    obs["bodies"].extend([piece(6,350,group=7), piece(7,410,group=7)])
    obs["bodies"][-2]["age_ticks"] = 30
    obs["bodies"][-1]["age_ticks"] = 100
    proposed = options.propose_options(obs, ledger(), duration=96, cap=128, rng=random.Random(4))
    labels = [o.label for o in proposed]
    assert len(labels) == len(set(labels))
    assert any(o.label.startswith("relay-carrier-g0-p") for o in proposed)


def causal_callback(*, collateral=False, cause=5, save=True):
    def callback(obs, word, start, end):
        if word:
            obs["protected"] = save
            if collateral:
                obs["bodies"][1]["projectile_hits"] += 1
        if end >= 8 and not obs.get("qualified") and not obs.get("protected"):
            obs["qualified"] = True
            obs["qualifying_clear_count"] += 1
            obs["bodies"][1]["lifecycle"] = "deleted"
            return [dict(kind="confirmed", tick=8, a=2, b=cause, value=9),
                    dict(kind="cleared", tick=8, a=2, b=0, value=9)]
        return []
    return callback


def test_causal_discovery_uses_exact_confirmed_source_without_mutating_root(fake_step):
    obs, history = board(), ledger()
    env = ProbeFake(obs, causal_callback())
    before = deepcopy(obs), history.to_dict()
    choices = options.propose_protection_options(env, obs, history, duration=12, cap=8)
    assert len(choices) == 8 and {o.source_id for o in choices} == {5}
    assert {o.target_id for o in choices} == {2} and {o.prediction_tick for o in choices} == {8}
    assert len({o.label for o in choices}) == len(choices)
    assert all(o.mode == "causal_protect" for o in choices)
    assert env.observation == obs and not env.calls
    assert (obs, history.to_dict()) == before


def test_causal_discovery_skips_future_causes_and_zero_caps(fake_step):
    obs, history = board(), ledger()
    env = ProbeFake(obs, causal_callback(cause=99))
    assert options.propose_protection_options(env, obs, history) == []
    assert options.propose_protection_options(env, obs, history, cap=0) == []
    assert options.propose_protection_options(Fake(obs), obs, history) == []


def test_causal_guard_commits_only_prefix_releasing_held_shot(fake_step):
    obs, history = board(), ledger()
    obs["right_held"] = True
    env = ProbeFake(obs, causal_callback())
    option = options.propose_protection_options(env, obs, history, duration=12, cap=1)[0]
    result = options.execute_option(env, obs, history, option, stop_tick=40)
    assert len(result["actions"]) == 12 and result["actions"][0] == 0 and result["actions"][1] != 0
    assert result["final"]["tick"] == 12 and not result["natural"]
    guard = result["diagnostics"]["guard"]
    assert guard["accepted"] and guard["qualification_saving"] == 1
    assert guard["probe_end_tick"] == 24  # original contact plus sixteen ticks
    assert result["diagnostics"]["shots"][0]["tick"] == 1
    assert result["diagnostics"]["events"] == []
    assert result["history"].to_dict() == history.to_dict()
    assert history.to_dict() == ledger().to_dict()


@pytest.mark.parametrize("collateral,save,reason", [(True,True,"valuable_direct_hit"), (False,False,"main_group_lost")])
def test_causal_rejection_has_no_committed_actions(fake_step, collateral, save, reason):
    obs, history = board(), ledger()
    env = ProbeFake(obs, causal_callback(collateral=collateral, save=save))
    option = options.EventOption("causal", "protect", 12, "causal_protect", source_id=5,
        target_id=2, group_id=7, prediction_tick=8)
    result = options.execute_option(env, obs, history, option, stop_tick=40)
    assert result["diagnostics"]["guard"]["reason"] == reason
    assert not result["actions"] and env.observation == obs and not env.calls


def test_causal_uncommitted_terminal_preserves_first_canonical_metadata(fake_step):
    obs, history = board(), ledger()
    obs.update(terminated=1, canonical_score=9900, canonical_level=100,
               canonical_highest_chain=12, canonical_clears=990)
    option = options.EventOption("causal", "protect", 12, "causal_protect", source_id=5,
        target_id=2, group_id=7, prediction_tick=8)
    result = options.execute_option(Fake(obs), obs, history, option)
    assert result["natural"] and result["final"]["canonical_score"] == 9900
    assert not result["actions"]


def retirement_board():
    obs, history = board(), ledger()
    obs["bodies"][1]["projectile_hits"] = 1
    backup = piece(6, 350, 170, group=7)
    backup["age_ticks"] = 20
    obs["bodies"].append(backup)
    history.memberships[6] = 7
    return obs, history


def retirement_callback(*, collateral=None, save=True, cause=5):
    def callback(obs, word, start, end):
        source, backup = obs["bodies"][1], obs["bodies"][-1]
        if word and source["lifecycle"] != "deleted":
            source["projectile_hits"] += 1
            source["lifecycle"] = "deleted"
            events = [dict(kind="destroyed", tick=end, a=2, b=0, value=0)]
            if collateral == "hit":
                backup["projectile_hits"] += 1
            elif collateral in ("destroyed", "ejected"):
                backup["lifecycle"] = "deleted"
                events.append(dict(kind=collateral, tick=end, a=6, b=0, value=0))
            return events
        if end >= 8 and not obs.get("qualified") and (source["lifecycle"] != "deleted" or not save):
            obs["qualified"] = True
            obs["qualifying_clear_count"] += 1
            if source["lifecycle"] == "deleted":
                return [dict(kind="confirmed", tick=8, a=3, b=5, value=1)]
            source["lifecycle"] = "deleted"
            return [dict(kind="confirmed", tick=8, a=2, b=cause, value=9),
                    dict(kind="cleared", tick=8, a=2, b=0, value=9)]
        return []
    return callback


def test_retirement_discovery_requires_predicted_solo_payer_and_young_backup(fake_step):
    obs, history = retirement_board()
    env = ProbeFake(obs, retirement_callback())
    before = deepcopy(obs), history.to_dict()
    choices = options.propose_retirement_options(env, obs, history, duration=12, cap=4)
    assert len(choices) == 4 and {o.source_id for o in choices} == {2}
    assert all(o.mode == "retire_carrier" and o.prediction_tick == 8 for o in choices)
    assert len({o.label for o in choices}) == len(choices)
    assert (obs, history.to_dict()) == before and env.observation == obs and not env.calls
    assert options.propose_retirement_options(env, obs, history, cap=0) == []
    obs["bodies"][4]["color"] = 0  # An available same-color co-payer should be retained.
    assert options.propose_retirement_options(ProbeFake(obs, retirement_callback()), obs, history) == []
    obs["bodies"] = obs["bodies"][:-1]
    assert options.propose_retirement_options(ProbeFake(obs, retirement_callback()), obs, history) == []


def test_retirement_saves_qualification_and_commits_only_exact_prefix(fake_step):
    obs, history = retirement_board()
    obs["right_held"] = True
    env = ProbeFake(obs, retirement_callback())
    option = options.EventOption("retire-g0-p0", "budget", 12, "retire_carrier", source_id=2,
        group_id=7, radius=.1, prediction_tick=8)
    result = options.execute_option(env, obs, history, option, stop_tick=40)
    assert len(result["actions"]) == 12 and result["actions"][0] == 0 and result["actions"][1]
    assert result["final"]["tick"] == 12 and result["final"]["qualifying_clear_count"] == 890
    guard = result["diagnostics"]["guard"]
    assert guard["accepted"] and guard["qualification_saving"] == 1
    assert guard["probe_end_tick"] == 24 and guard["surviving_carriers"] == 1
    assert [e["kind"] for e in result["diagnostics"]["events"]] == ["destroyed"]
    assert 2 not in result["history"].memberships and result["history"].memberships[6] == 7
    assert history.memberships[2] == 7 and history.memberships[6] == 7


def test_small_retirement_cap_covers_distinct_predicted_payers(fake_step):
    obs, history = retirement_board()
    obs["bodies"][-1]["projectile_hits"] = 1
    youngest = piece(7, 150, 80, group=7)
    youngest["age_ticks"] = 10
    obs["bodies"].append(youngest)
    history.memberships[7] = 7
    def two_payers(o, word, start, end):
        if end < 8 or o.get("qualified"):
            return []
        o["qualified"] = True
        return [dict(kind="confirmed", tick=8, a=2, b=5, value=9),
                dict(kind="confirmed", tick=9, a=6, b=5, value=10)]
    choices = options.propose_retirement_options(ProbeFake(obs, two_payers), obs, history, cap=2)
    assert {o.source_id for o in choices} == {2, 6}


@pytest.mark.parametrize("collateral,save,reason", [
    ("hit", True, "collateral_direct_hit"), ("destroyed", True, "no_durable_backup"),
    ("ejected", True, "no_durable_backup"), (None, False, "no_observed_qualification_saving")])
def test_retirement_guard_rejects_collateral_loss_or_no_saving(fake_step, collateral, save, reason):
    obs, history = retirement_board()
    env = ProbeFake(obs, retirement_callback(collateral=collateral, save=save))
    option = options.EventOption("retire", "budget", 12, "retire_carrier", source_id=2,
        group_id=7, radius=.1, prediction_tick=8)
    result = options.execute_option(env, obs, history, option, stop_tick=40)
    assert result["diagnostics"]["guard"]["reason"] == reason
    assert not result["actions"] and not env.calls and env.observation == obs


def test_retirement_terminal_preserves_canonical_metadata_without_probing(fake_step):
    obs, history = retirement_board()
    obs.update(terminated=1, canonical_score=9900, canonical_level=100,
               canonical_highest_chain=12, canonical_clears=990)
    env = ProbeFake(obs, retirement_callback())
    option = options.EventOption("retire", "budget", 12, "retire_carrier", source_id=2, group_id=7)
    result = options.execute_option(env, obs, history, option)
    assert result["natural"] and result["final"]["canonical_score"] == 9900 and not result["actions"]
    assert not result["diagnostics"]["guard_checked"] and not env.calls


def preparation_callback(*, group=False, spend=False, collateral=None):
    def callback(obs, word, start, end):
        if word:
            obs["preparing"] = True
            if collateral == "hit":
                obs["bodies"][1]["projectile_hits"] += 1
            elif collateral == "ejected":
                obs["bodies"][1]["lifecycle"] = "deleted"
                return [dict(kind="ejected", tick=end, a=2, b=0, value=0)]
        if end >= 7 and obs.get("preparing") and not obs["bodies"][0]["rot_timer"]:
            obs["bodies"][0]["rot_timer"] = 1
            if group:
                obs["bodies"][0]["chain_id"] = 7
            if spend:
                obs["qualifying_clear_count"] += 1
            return [dict(kind="contact", tick=7, a=1, b=5, value=0)]
        return []
    return callback


def test_preparation_proposals_require_aged_matching_fresh_and_different_rot():
    obs, history = board(), ledger()
    before = deepcopy(obs), history.to_dict()
    choices = options.propose_preparation_options(obs, history, duration=20, cap=8, rng=random.Random(3))
    assert len(choices) == 8 and {o.source_id for o in choices} == {1}
    assert {o.target_id for o in choices} == {5}
    assert all(o.mode == "prepare_feeder" and o.group_id == 7 for o in choices)
    assert (obs, history.to_dict()) == before
    assert choices == options.propose_preparation_options(obs, history, duration=20, cap=8, rng=random.Random(3))
    obs["bodies"][0]["age_ticks"] = 100
    assert options.propose_preparation_options(obs, history) == []
    obs["bodies"][0]["age_ticks"] = 200
    obs["bodies"][4]["color"] = 0
    assert options.propose_preparation_options(obs, history) == []


def test_preparation_commits_only_through_observed_rot_start_and_releases_hold(fake_step):
    obs, history = board(), ledger()
    obs["left_held"] = True
    env = ProbeFake(obs, preparation_callback())
    option = options.EventOption("prepare", "prepare", 20, "prepare_feeder", source_id=1,
        target_id=5, group_id=7, strength=1, radius=.15, max_shots=1, observation_period=2)
    result = options.execute_option(env, obs, history, option, stop_tick=30)
    assert result["diagnostics"]["guard"]["accepted"]
    assert result["final"]["tick"] == 8 and len(result["actions"]) == 8
    assert result["actions"][0] == 0 and result["actions"][1]
    assert result["observation"]["bodies"][0]["rot_timer"] == 1
    assert result["observation"]["bodies"][0]["chain_id"] == 0
    assert result["final"]["qualifying_clear_count"] == obs["qualifying_clear_count"]
    assert len(result["diagnostics"]["events"]) == 1 and result["history"].to_dict() == history.to_dict()
    assert result["diagnostics"]["guard"]["probe_end_tick"] == 8


@pytest.mark.parametrize("group,spend,collateral,reason", [
    (True, False, None, "grouped_before_prepared"),
    (False, True, None, "qualification_spent_before_preparation"),
    (False, False, "hit", "valuable_direct_hit"),
    (False, False, "ejected", "valuable_loss")])
def test_preparation_rejects_premature_grouping_or_collateral_cost(fake_step, group, spend, collateral, reason):
    obs, history = board(), ledger()
    env = ProbeFake(obs, preparation_callback(group=group, spend=spend, collateral=collateral))
    option = options.EventOption("prepare", "prepare", 20, "prepare_feeder", source_id=1,
        target_id=5, group_id=7, radius=.15, max_shots=1, observation_period=2)
    result = options.execute_option(env, obs, history, option)
    assert result["diagnostics"]["guard"]["reason"] == reason
    assert not result["actions"] and not env.calls and env.observation == obs


def test_preparation_respects_short_budget_and_stop_boundary(fake_step):
    obs, history = board(), ledger()
    env = ProbeFake(obs, preparation_callback())
    option = options.EventOption("prepare", "prepare", 20, "prepare_feeder", source_id=1,
        target_id=5, group_id=7, radius=.15, max_shots=1)
    result = options.execute_option(env, obs, history, option, stop_tick=5)
    assert not result["actions"] and result["diagnostics"]["guard"]["probe_end_tick"] == 5
    class Budget:
        checks = 0
        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise TimeoutError("budget")
    with pytest.raises(TimeoutError):
        options.execute_option(env, obs, history, option, Budget())
    assert env.observation == obs and not env.calls


