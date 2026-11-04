"""Real persistent-worker integration and transport failure tests."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from irisu_search.engine import Engine, EngineError

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("IRISU_SEARCH_ENGINE", ROOT / "build-project-exact/irisu-search-engine"))
CUDA = os.environ.get("IRISU_SEARCH_CUDA_ENGINE")


def record() -> tuple[int, list[int], tuple[int, int, int]]:
    payload = (ROOT / "irisu-high-score-4393101.rpy").read_bytes()
    seed, level, score, chain, mode = struct.unpack_from("<5I", payload)
    assert mode == 0
    words = list(struct.unpack_from(f"<{(len(payload)-52)//4}I", payload, 52))
    return seed, words, (score, level, chain)


@unittest.skipUnless(NATIVE.is_file(), "build irisu-search-engine to run native integration")
class NativeEngineTests(unittest.TestCase):
    def test_closed_loop_trace_replays_and_parent_is_immutable(self) -> None:
        with Engine([str(NATIVE)]) as engine:
            parent = engine.load(42, [0] * 100)
            candidates = [{"mode": 1, "search_seed": 711}, {"mode": 1, "search_seed": 711}]
            response = engine.batch(parent["handle"], 400, candidates)
            first, second = response["results"]
            self.assertEqual(first["words"], second["words"])
            self.assertEqual(first["summary"], second["summary"])
            self.assertTrue(any(first["words"]))
            replayed = engine.load(42, [0] * 100 + first["words"])
            self.assertEqual(first["summary"], replayed["summary"])
            again = engine.batch(parent["handle"], 400, candidates[:1])["results"][0]
            self.assertEqual(first["summary"], again["summary"])

    def test_interval_metrics_retain_fully_paid_new_groups(self) -> None:
        seed, words, _ = record()
        with Engine([str(NATIVE)]) as engine:
            for start, horizon, expected in (
                (270, 130, {"recruitment_events": 8, "new_groups": 2,
                            "max_new_chain": 5, "qualifications": 5}),
                (953, 1, {"recruitment_events": 2, "new_groups": 1,
                          "max_new_chain": 2, "qualifications": 1}),
            ):
                parent = engine.load(seed, words[:start])
                self.assertEqual(parent["summary"]["groups"], [])
                child = engine.batch(parent["handle"], horizon,
                                     [{"mode": 0, "words": words[start:start+horizon]}])["results"][0]
                self.assertEqual(child["summary"]["groups"], [])
                self.assertEqual(child["group_deltas"], [])
                self.assertEqual(child["metrics"], expected)

    def test_releasing_parent_preserves_descendant_and_capacity_recovers(self) -> None:
        with Engine([str(NATIVE), "--max-checkpoints", "3"]) as engine:
            parent = engine.load(0)
            child = engine.batch(parent["handle"], 4, [{"mode": 0}])["results"][0]
            engine.release([parent["handle"]])
            leaf = engine.batch(child["handle"], 4, [{"mode": 0}])["results"][0]
            self.assertEqual(leaf["summary"]["tick"], 8)
            with self.assertRaises(EngineError):
                engine.batch(child["handle"], 1, [{"mode": 0}, {"mode": 0}])
            engine.release([leaf["handle"]])
            self.assertEqual(engine.load(1)["summary"]["seed"], 1)

    def test_record_tail_uses_canonical_score_and_retains_extinct_group_counters(self) -> None:
        seed, words, expected = record()
        with Engine([str(NATIVE)]) as engine:
            parent = engine.load(seed, words[:96000])
            self.assertEqual(parent["summary"]["score"], 3135391)
            self.assertTrue(any(g["chain"] > parent["summary"]["highest_chain"]
                                for g in parent["summary"]["groups"]))
            result = engine.batch(parent["handle"], 2000,
                                  [{"mode": 0, "words": words[96000:]}])["results"][0]
            summary = result["summary"]
            self.assertTrue(summary["terminated"])
            self.assertEqual((summary["canonical_score"], summary["canonical_level"],
                              summary["canonical_chain"]), expected)
            self.assertGreater(summary["score"], summary["canonical_score"])
            self.assertEqual(len(result["words"]), summary["tick"] - 96000)
            self.assertEqual(len(result["group_deltas"]), len(parent["summary"]["groups"]))
            self.assertTrue(any(g["chain_after"] > g["chain_before"] for g in result["group_deltas"]))
            self.assertTrue(all(g["chain_after"] >= g["chain_before"] and
                                g["num_after"] >= g["num_before"] for g in result["group_deltas"]))


@unittest.skipUnless(CUDA and NATIVE.is_file(), "set IRISU_SEARCH_CUDA_ENGINE for device integration")
class CudaEngineTests(unittest.TestCase):
    def test_native_device_replay_and_closed_loop_equivalence(self) -> None:
        seed, words, _ = record()
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        for host_prefix, threads in ((False, None), (True, None), (True, 4), (True, 32)):
            command = [str(CUDA), "--max-checkpoints", "8"]
            if threads is not None:
                command.extend(["--threads", str(threads)])
            if host_prefix:
                command.append("--host-prefix")
            with Engine([str(NATIVE)]) as native, Engine(command, env=env) as device:
                a = native.load(seed, words[:400])
                b = device.load(seed, words[:400])
                self.assertEqual(a["summary"], b["summary"])
                candidates = [{"mode": 0, "words": words[400:480]},
                              {"mode": 1, "search_seed": 711},
                              {"mode": 2, "search_seed": 913},
                              {"mode": 3, "search_seed": 113}]
                left = native.batch(a["handle"], 80, candidates)
                right = device.batch(b["handle"], 80, candidates)
                self.assertEqual(right["launch_threads"], 1 if threads is None else threads)
                self.assertEqual(right["rollout_blocks"], (len(candidates) + right["launch_threads"] - 1)
                                 // right["launch_threads"])
                for x, y in zip(left["results"], right["results"]):
                    self.assertEqual(x["words"], y["words"])
                    self.assertEqual(x["summary"], y["summary"])
                    self.assertEqual(x["group_deltas"], y["group_deltas"])
                    self.assertEqual(x["metrics"], y["metrics"])
                device.release([r["handle"] for r in right["results"]])
                self.assertEqual(device.load(0)["summary"]["seed"], 0)
                for start, horizon in ((270, 130), (953, 1)):
                    a = native.load(seed, words[:start])
                    b = device.load(seed, words[:start])
                    proposal = [{"mode": 0, "words": words[start:start+horizon]}]
                    x = native.batch(a["handle"], horizon, proposal)["results"][0]
                    y = device.batch(b["handle"], horizon, proposal)["results"][0]
                    self.assertEqual(x["metrics"], y["metrics"])
                    self.assertEqual(x["summary"], y["summary"])
                    device.release([b["handle"], y["handle"]])


class TransportTests(unittest.TestCase):
    def test_timeout_terminates_worker(self) -> None:
        code = ('import json,sys,time; '
                'sys.stdin.readline(); '
                'print(json.dumps({"ok":True,"identity":{"protocol":1}}),flush=True); '
                'sys.stdin.readline(); time.sleep(30)')
        with Engine([sys.executable, "-u", "-c", code], timeout_seconds=.2) as engine:
            with self.assertRaises(EngineError):
                engine.load(0)
            self.assertIsNotNone(engine._process.poll())

    def test_invalid_values_never_reach_worker(self) -> None:
        with self.assertRaises(ValueError):
            Engine([], timeout_seconds=1)
        if NATIVE.is_file():
            with Engine([str(NATIVE)]) as engine:
                for invalid in (-1, 1 << 32, True):
                    with self.assertRaises(ValueError):
                        engine.load(invalid)
                with self.assertRaises(ValueError):
                    engine.load(0, [1 << 21])


if __name__ == "__main__":
    unittest.main()
