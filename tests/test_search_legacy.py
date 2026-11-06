"""Historical genealogy and resident adapter preserve the original event path."""
from copy import deepcopy
import json

import pytest

from irisu_search.historical import event_options, opportunity_value, search
from irisu_search.historical.growth_proposals import GrowthMacro
from irisu_search.legacy import Environment, observation, trace_histories


def summary(tick=200):
    return dict(seed=42, tick=tick, score=10000, gauge=8000, level=90,
        highest_chain=12, qualifying=890, terminated=False, truncated=False,
        left_held=False, right_held=False, canonical_score=0, canonical_level=0,
        canonical_chain=0, bodies=[dict(id=1, kind=0, shape=1, lifecycle=1,
            color=0, x=200.0, y=220.0, vx=0.0, vy=1.0, angle=.5,
            angular_velocity=.01, size=32.0, chain_id=0, age_ticks=200,
            rot_timer=0, hits=0, lifetime=1000)])


class MemoryEngine:
    """Immutable worlds with exact input-edge behavior for adapter tests."""
    def __init__(self, terminal_tick=None):
        self.nodes = {1: dict(handle=1, summary=summary(), words=[])}
        self.next_handle = 2
        self.released = []
        self.terminal_tick = terminal_tick

    def batch_events(self, handle, horizon, candidates):
        parent = self.nodes[handle]
        results = []
        for index, candidate in enumerate(candidates):
            state = deepcopy(parent["summary"])
            words, events = [], []
            for word in list(candidate["words"]) + [0] * horizon:
                if len(words) >= horizon or state["terminated"]:
                    break
                state["tick"] += 1
                if word & 3 and not (state["left_held"] if word & 1 else state["right_held"]):
                    events.append(dict(tick=state["tick"], kind="shot_fired", a=90,
                                       b=0, value=word & 3, sequence=state["tick"]))
                state["left_held"], state["right_held"] = bool(word & 1), bool(word & 2)
                words.append(word)
                if self.terminal_tick == state["tick"]:
                    state.update(terminated=True, canonical_score=9900,
                                 canonical_level=100, canonical_chain=12)
            result = dict(handle=self.next_handle, summary=state, words=words,
                          events=events, candidate_index=index)
            self.next_handle += 1
            self.nodes[result["handle"]] = result
            results.append(result)
        return dict(results=results, rejected=[], simulated_ticks=sum(len(r["words"]) for r in results))

    def release(self, handles):
        for handle in handles:
            assert handle in self.nodes, "released a handle twice or used an expired world"
            self.released.append(handle)
            del self.nodes[handle]


def test_body_observation_matches_historical_enums_and_fields():
    body = observation(summary())["bodies"][0]
    assert (body["kind"], body["shape"], body["lifecycle"]) == ("piece", "box", "dynamic_fresh")
    assert body["projectile_hits"] == 0 and body["remaining_lifetime"] == 1000
    assert (body["age_ticks"], body["angle"], body["angular_velocity"]) == (200, .5, .01)
    assert observation(summary())["qualifying_clear_count"] == 890


def test_root_history_matches_original_events_without_future_information(tmp_path):
    events = [
        dict(tick=2, kind="chain_joined", a=1, value=7),
        dict(tick=2, kind="chain_joined", a=2, value=7),
        dict(tick=5, kind="confirmed", a=2, value=1),
        dict(tick=5, kind="cleared", a=2, value=1),
        dict(tick=6, kind="chain_joined", a=3, value=7),
        dict(tick=7, kind="destroyed", a=1, value=0),
        dict(tick=8, kind="confirmed", a=3, value=2),
        dict(tick=9, kind="ejected", a=3, value=0),
    ]
    labels = dict(chain_joined="join", confirmed="qualify", cleared="score",
                  destroyed="deletion_mark", ejected="ejected")
    rows = [dict(kind="identity", seed=42)]
    for event in events:
        rows.append(dict(tick=event["tick"], kind=labels[event["kind"]], body_id=event["a"],
                         group_id=7, num=event["value"]))
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    histories = trace_histories(path, [2, 5, 6, 7, 8, 9])
    for tick, history in histories.items():
        expected = opportunity_value.ComboHistory.from_events(e for e in events if e["tick"] <= tick)
        assert history.to_dict() == expected.to_dict()
    assert histories[5].groups[7]["member_ids"] == (1, 2)
    assert histories[5].memberships == {1: 7}
    assert histories[8].groups[7]["last_clear_tick"] == 8
    assert histories[8].groups[7]["last_join_tick"] == 6
    assert histories[8].memberships == {3: 7}
    assert not histories[9].memberships


def test_nested_branch_handles_keep_parent_immutable_and_release_owned_worlds():
    engine = MemoryEngine()
    accounted = []
    root = Environment(engine, engine.nodes[1], accounted.append)
    with root.fast_checkpoint() as checkpoint:
        with checkpoint.branch() as child:
            child.legacy_step(0, 4)
            first = child.state_hash()
            with child.fast_checkpoint() as nested:
                with nested.branch() as probe:
                    probe.legacy_step(0, 3)
                    assert probe.observation()["tick"] == 207
                assert child.state_hash() == first
            child.legacy_step(0, 2)
            assert first in engine.released
            assert child.observation()["tick"] == 206
        assert root.observation()["tick"] == 200
        assert list(engine.nodes) == [1]
    assert sum(r["simulated_ticks"] for r in accounted) == 9


@pytest.mark.parametrize("terminal_tick", [None, 205])
def test_batched_macro_matches_original_executor_with_held_release_and_terminal(terminal_tick):
    engine = MemoryEngine(terminal_tick=terminal_tick)
    engine.nodes[1]["summary"]["right_held"] = True
    env = Environment(engine, engine.nodes[1], lambda _: None)
    shot = (220 << 12) | (200 << 2) | 2
    macro = GrowthMacro("two-shots", "extend", ((1, shot), (5, shot)), 8)
    option = event_options.EventOption(macro.label, macro.family, macro.duration, macro=macro)
    history = opportunity_value.ComboHistory()
    expected = search.execute_candidate(env, env.observation(), history, option,
                                        search.defaults(), type("Budget", (), {"check": lambda self: None})(), 208)
    plans = [dict(macro.sparse).get(tick, 0) for tick in range(8)]
    actual = engine.batch_events(1, 8, [{"words": plans}])["results"][0]
    assert tuple(actual["words"]) == expected["actions"]
    assert observation(actual["summary"]) == expected["observation"]
    assert opportunity_value.ComboHistory.from_events(actual["events"]).to_dict() == expected["history"].to_dict()


def test_parallel_event_options_match_serial_results_and_keep_inputs_immutable():
    from concurrent.futures import ThreadPoolExecutor
    import random
    from threading import RLock

    class SerializedEngine(MemoryEngine):
        def __init__(self):
            super().__init__()
            self.lock = RLock()

        def batch_events(self, *args):
            with self.lock:
                return super().batch_events(*args)

        def release(self, handles):
            with self.lock:
                return super().release(handles)

    engine = SerializedEngine()
    carrier = {**engine.nodes[1]["summary"]["bodies"][0], "id": 2, "x": 430,
               "chain_id": 7, "lifecycle": 2}
    engine.nodes[1]["summary"]["bodies"].append(carrier)
    root_result = engine.nodes[1]
    obs = observation(root_result["summary"])
    history = opportunity_value.ComboHistory.from_groups([dict(chain_id=7,
        cumulative_members=12, successful_clears=8, first_tick=1,
        last_join_tick=100, member_ids=[2])])
    options = event_options.propose_options(obs, history, duration=16, cap=32,
                                           rng=random.Random(661003))
    assert {option.mode for option in options} >= {"macro", "wait_spawn", "wait_join"}
    before = deepcopy((obs, history.to_dict(), options))
    budget = type("Budget", (), {"check": lambda self: None})()

    def execute(option):
        env = Environment(engine, root_result, lambda _: None)
        try:
            outcome = search.execute_candidate(env, obs, history, option,
                search.defaults(segment=16), budget, 216)
            return {**outcome, "history": outcome["history"].to_dict()}
        finally:
            env.close()

    serial = [execute(option) for option in options]
    with ThreadPoolExecutor(max_workers=12) as executor:
        parallel = list(executor.map(execute, options))
    assert parallel == serial
    assert (obs, history.to_dict(), options) == before
    assert list(engine.nodes) == [1]


def test_batcher_keeps_mixed_parent_horizons_and_candidate_results_separate():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from irisu_search.legacy import Batcher

    class MixedEngine(MemoryEngine):
        def __init__(self):
            super().__init__()
            self.nodes[90] = dict(handle=90, summary=summary(900), words=[])
            self.next_handle = 100
            self.waves = []

        def multi_batch_events(self, requests):
            self.waves.append(deepcopy(requests))
            results = []
            for index, request in enumerate(requests):
                result = super().batch_events(request["handle"], request["horizon"],
                    [{"words": request["words"]}])["results"][0]
                result["candidate_index"] = index
                results.append(result)
            # Reply order must not control caller association.
            return dict(results=list(reversed(results)), rejected=[],
                        simulated_ticks=sum(len(row["words"]) for row in results))

    engine, ready = MixedEngine(), Barrier(16)
    transport = Batcher(engine)
    try:
        def execute(index):
            ready.wait()
            handle = 1 if index % 2 else 90
            horizon = 1 + index % 4
            response = transport.batch_events(handle, horizon, [{"words": [0] * horizon}])
            result = response["results"][0]
            assert result["summary"]["tick"] == engine.nodes[handle]["summary"]["tick"] + horizon
            assert response["simulated_ticks"] == horizon
            transport.release([result["handle"]])
            return result

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(execute, range(16)))
    finally:
        transport.close()
    assert any(len(wave) > 1 for wave in engine.waves)
    assert len({result["handle"] for result in results}) == 16
    assert set(engine.nodes) == {1, 90}
    assert not transport.thread.is_alive()


def test_batcher_delivers_rejection_to_only_its_request_and_releases_successes():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from irisu_search.legacy import Batcher

    class RejectingEngine(MemoryEngine):
        def multi_batch_events(self, requests):
            results, rejected = [], []
            for index, request in enumerate(requests):
                if request["words"] == [1]:
                    rejected.append(dict(candidate_index=index, words=[1],
                                         diagnostic={"reason": "fixture rejection"}))
                else:
                    row = super().batch_events(request["handle"], request["horizon"],
                        [{"words": request["words"]}])["results"][0]
                    row["candidate_index"] = index
                    results.append(row)
            return dict(results=results, rejected=rejected,
                        simulated_ticks=sum(len(row["words"]) for row in results + rejected))

    engine, ready = RejectingEngine(), Barrier(2)
    transport = Batcher(engine)
    try:
        def execute(word):
            ready.wait()
            return transport.batch_events(1, 1, [{"words": [word]}])
        with ThreadPoolExecutor(max_workers=2) as executor:
            accepted, rejected = list(executor.map(execute, (0, 1)))
        assert not accepted["rejected"] and len(accepted["results"]) == 1
        assert not rejected["results"] and len(rejected["rejected"]) == 1
        assert accepted["simulated_ticks"] == rejected["simulated_ticks"] == 1
        transport.release([accepted["results"][0]["handle"]])
    finally:
        transport.close()
    assert list(engine.nodes) == [1]


def test_batcher_failure_unblocks_every_waiter_and_future_requests():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from irisu_search.legacy import Batcher

    class FailedEngine:
        def multi_batch_events(self, requests):
            raise RuntimeError("fixture transport failure")

    ready, transport = Barrier(8), Batcher(FailedEngine())
    try:
        def execute(_):
            ready.wait()
            with pytest.raises(RuntimeError, match="fixture transport failure"):
                transport.batch_events(1, 1, [{"words": [0]}])
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(execute, range(8)))
        with pytest.raises(RuntimeError, match="fixture transport failure"):
            transport.batch_events(1, 1, [{"words": [0]}])
    finally:
        transport.close()
    assert not transport.thread.is_alive()


def test_completed_terminal_is_retained_when_deadline_expires_during_rollout():
    from irisu_search.legacy import Budget, Campaign

    engine = MemoryEngine(terminal_tick=205)
    args = search.defaults(tick=200, segment=8, horizon=8)
    history = opportunity_value.ComboHistory()
    env = Environment(engine, engine.nodes[1], lambda _: None)
    parent = search.initial_node(env.observation(), history, args)
    budget = Budget(100)
    campaign = Campaign.__new__(Campaign)
    campaign.account = lambda response: setattr(budget, "cancelled", True)
    node = campaign.expand(engine, engine.nodes[1], parent, None, [0] * 208,
                           [0] * 200, args, budget, 208)
    assert budget.cancelled
    assert node is not None and node.natural
    assert node.final["canonical_score"] == 9900
    assert len(node.suffix) == 5
    assert list(engine.nodes) == [1]
