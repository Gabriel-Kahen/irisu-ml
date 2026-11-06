"""Opt-in endgame qualification of the actual October planner on native and GPU.

Requires exclusive GPU ownership. Set IRISU_SEARCH_CUDA_ENGINE,
IRISU_SEARCH_ENGINE, IRISU_SEARCH_TRACE and optionally
IRISU_SEARCH_HISTORICAL_REPLAY (defaults to the published record).
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import pytest

from irisu_search.engine import Engine
from irisu_search.historical import growth, search
from irisu_search.historical.common import snapshot
from irisu_search.legacy import Batcher, Budget, Campaign, Environment, observation, trace_histories
from irisu_search.verification import read_replay

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("IRISU_SEARCH_CUDA_ENGINE", "IRISU_SEARCH_ENGINE", "IRISU_SEARCH_TRACE")


def original_child(env, option, parent, history, words, args, budget):
    """Original event_search branch path, without the new driver's macro batching."""
    root, limit = args.tick, min(args.max_ticks, args.tick + args.horizon)
    with env.fast_checkpoint() as checkpoint, checkpoint.branch() as child:
        if option is None:
            end = min(root + args.segment, limit)
            full = words[:end] + [0] * max(0, end-len(words))
            child_history = history.copy()
            obs, final = growth.replay_events(child, full, root, end, child_history, budget)
            suffix = tuple(full[root:final["tick"]])
        else:
            outcome = search.execute_candidate(child, parent.observation, history,
                                                option, args, budget, limit)
            if not outcome["actions"]:
                return None
            obs, final, child_history = (outcome[key] for key in ("observation", "final", "history"))
            suffix = tuple(outcome["actions"])
        full, obs, final, exhausted = search.finish_exhausted(child,
            words[:root] + list(suffix), obs, final, child_history, args.max_ticks, budget)
        if exhausted:
            suffix = tuple(full[root:])
        return growth.Node(suffix, obs, final, child_history,
            search.evaluate_value(obs, child_history, args, elapsed_ticks=final["tick"]-root),
            option.family if option else "reference", (option.label if option else "reference",))


def node_result(node):
    if node is None:
        return None
    return dict(suffix=node.suffix, final=node.final, observation=node.observation,
                history=node.history.to_dict(), family=node.family, labels=node.labels,
                value=node.value.to_dict())


@pytest.mark.skipif(any(not os.environ.get(key) for key in REQUIRED),
                    reason="set native/CUDA/trace paths for exclusive device qualification")
def test_october_endgame_menu_and_diverse_selection_match_native(tmp_path):
    replay = Path(os.environ.get("IRISU_SEARCH_HISTORICAL_REPLAY", ROOT / "irisu-high-score-4393101.rpy"))
    seed, words, header = read_replay(replay)
    args = search.defaults(seed=seed, tick=96000, random_seed=661003)
    assert len(words) > args.tick + args.segment
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if os.environ.get("IRISU_SEARCH_ORACLE_LIBRARY"):
        environment["LD_LIBRARY_PATH"] = str(Path(os.environ["IRISU_SEARCH_ORACLE_LIBRARY"]).parent)
    trace = tmp_path / "native-genealogy.jsonl"
    with trace.open("w") as stream:
        completed = subprocess.run([os.environ["IRISU_SEARCH_TRACE"], str(replay)],
            stdout=stream, stderr=subprocess.PIPE, env=environment, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    rows = [json.loads(line) for line in trace.open()]
    assert rows[0]["seed"] == seed and rows[0]["input_words"] == len(words)
    assert rows[0]["header_score"] == header["canonical_score"]
    assert rows[-1]["verified"] and rows[-1]["canonical_score"] == header["canonical_score"]
    assert rows[-1]["tick"] == len(words)
    history = trace_histories(trace, [args.tick])[args.tick]
    totals = {"native": 0, "cuda": 0}
    totals_lock = threading.Lock()

    def account(backend):
        def record(response):
            assert not response["rejected"], response["rejected"]
            with totals_lock:
                totals[backend] += response["simulated_ticks"]
        return record

    native_command = [os.environ["IRISU_SEARCH_ENGINE"], "--max-checkpoints", "512"]
    device_command = [os.environ["IRISU_SEARCH_CUDA_ENGINE"], "--max-checkpoints", "512", "--host-prefix"]
    started = time.monotonic()
    with Engine(native_command, timeout_seconds=600, env=environment) as native, \
            Engine(device_command, timeout_seconds=600, env=environment) as cuda:
        a, b = native.load(seed, words[:args.tick]), cuda.load(seed, words[:args.tick])
        try:
            assert observation(a["summary"]) == observation(b["summary"])
            for group in a["summary"]["groups"]:
                past = history.groups[group["id"]]
                assert (past["cumulative_members"], past["successful_clears"]) == (group["chain"], group["num"])
            native_env = Environment(native, a, account("native"))
            cuda_env = Environment(cuda, b, account("cuda"))
            parent = search.initial_node(native_env.observation(), history, args)
            menus = []
            for env in (native_env, cuda_env):
                menus.append(search.candidate_options(env, parent.observation, history,
                    args, args.segment, random.Random(args.random_seed), Budget(600)))
            assert menus[0] == menus[1]
            assert len(menus[0]) == args.candidates == 48
            options = [None, *menus[0]]
            expected = [original_child(native_env, option, parent, history, words,
                                        args, Budget(600)) for option in options]
            campaign = Campaign.__new__(Campaign)
            campaign.account = account("cuda")
            transport = Batcher(cuda)
            budget = Budget(600)
            try:
                with ThreadPoolExecutor(max_workers=49) as executor:
                    futures = [executor.submit(campaign.expand, transport, b, parent, option,
                        words, words[:args.tick], args, budget, args.tick+args.horizon)
                        for option in options]
                    actual = [future.result() for future in futures]
            finally:
                transport.close()
            assert [node_result(node) for node in actual] == [node_result(node) for node in expected]
            left = search.select_diverse([node for node in expected if node], args.beam_width, history)
            right = search.select_diverse([node for node in actual if node], args.beam_width, history)
            assert [node.identifier for node in left] == [node.identifier for node in right]
            assert observation(a["summary"]) == native_env.observation()
            assert observation(b["summary"]) == cuda_env.observation()
            receipt = dict(schema="irisu-historical-planner-device-qualification-v1",
                replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                native=native.identity, cuda=cuda.identity, seed=seed, root=args.tick,
                random_seed=args.random_seed, candidate_cap=args.candidates,
                menu_size=len(menus[0]), compared_branches=len(options),
                accepted_branches=sum(node is not None for node in actual),
                selected_identifiers=[node.identifier for node in right],
                exact_actions_observations_history_and_values=True,
                simulated_ticks=totals, seconds=time.monotonic()-started)
            (tmp_path / "historical-device-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        finally:
            native.release([a["handle"]])
            cuda.release([b["handle"]])


@pytest.mark.skipif(any(not os.environ.get(key) for key in (
    "IRISU_SEARCH_CUDA_ENGINE", "IRISU_SEARCH_ORACLE", "IRISU_SEARCH_ORACLE_LIBRARY")),
    reason="set CUDA/oracle/library paths for exclusive terminal qualification")
def test_full_tail_gpu_payouts_pass_independent_native_verification(tmp_path):
    from irisu_search.verification import NativeVerifier

    replay = Path(os.environ.get("IRISU_SEARCH_HISTORICAL_REPLAY", ROOT / "irisu-high-score-4393101.rpy"))
    seed, words, header = read_replay(replay)
    root_tick, horizon = 96000, 24000
    assert root_tick < len(words) <= root_tick + horizon
    suffix = words[root_tick:]
    shots = [index for index, word in enumerate(suffix) if word & 3]
    assert shots, "qualification replay must contain endgame shots"
    removed = list(suffix)
    removed[shots[0]] = 0
    retargeted = list(suffix)
    last = shots[-1]
    x = (suffix[last] >> 2) & 1023
    delta = 2 if x + 2 < 640 else -2
    assert 0 <= x + delta < 640
    retargeted[last] = (suffix[last] & ~(1023 << 2)) | ((x + delta) << 2)
    candidates = [{"mode": 0, "words": plan} for plan in (suffix, removed, retargeted)]
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["LD_LIBRARY_PATH"] = str(Path(os.environ["IRISU_SEARCH_ORACLE_LIBRARY"]).parent)
    verifier = NativeVerifier(os.environ["IRISU_SEARCH_ORACLE"],
                              os.environ["IRISU_SEARCH_ORACLE_LIBRARY"], timeout=600)
    receipt_path = tmp_path / "historical-terminal-device-receipt.json"
    receipt = dict(schema="irisu-historical-terminal-device-qualification-v1",
        replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
        seed=seed, root_tick=root_tick, horizon=horizon, candidate_count=3,
        mutation_labels=["reference", "delete-first-tail-shot", f"last-tail-shot-x{delta:+}"],
        verified=[], censored=[], rejected=[], published=False)

    def persist():
        receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")

    command = [os.environ["IRISU_SEARCH_CUDA_ENGINE"], "--max-checkpoints", "16", "--host-prefix"]
    persist()
    try:
        with Engine(command, timeout_seconds=600, env=environment) as engine:
            receipt["cuda"] = engine.identity
            root = engine.load(seed, words[:root_tick])
            handles = [root["handle"]]
            try:
                response = engine.batch(root["handle"], horizon, candidates)
                handles.extend(row["handle"] for row in response["results"])
                receipt.update(rejected=response["rejected"],
                               simulated_ticks=response["simulated_ticks"])
                persist()
                assert not response["rejected"], response["rejected"]
                baseline_verified = False
                for result in response["results"]:
                    index, final = result["candidate_index"], result["summary"]
                    if not final["terminated"] or final["truncated"]:
                        receipt["censored"].append(dict(candidate_index=index, final=final,
                            executed_words=len(result["words"])))
                        persist()
                        continue
                    full = words[:root_tick] + result["words"]
                    verified = verifier.verify(dict(seed=seed, words=full, summary=final))
                    receipt["verified"].append(dict(candidate_index=index, receipt=verified))
                    persist()
                    if index == 0:
                        assert result["words"] == suffix
                        assert final["canonical_score"] == header["canonical_score"]
                        assert final["canonical_level"] == header["canonical_level"]
                        assert final["canonical_chain"] == header["canonical_chain"]
                        assert verified["replay_sha256"] == receipt["replay_sha256"]
                        baseline_verified = True
                assert baseline_verified, "reference full tail did not naturally terminate and verify"
                receipt["passed"] = True
                persist()
            finally:
                engine.release(handles)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise
