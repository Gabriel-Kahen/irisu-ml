from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metsu_replay_fixtures", ROOT / "tools/metsu/check_replay_fixtures.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(GATE)


class ReplayFixtureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "trace.jsonl"
        self.meta = {
            "type": "metadata", "schema": GATE.comparison.SCHEMA, "source": "native",
            "mode": 1, "boundary": "after-actor-pass", "physics_backend": GATE.BACKEND,
            "config_hash": "config", "seed": 41, "replay_sha256": "a" * 64,
        }
        actor = dict.fromkeys(GATE.comparison.ACTOR_FIELDS, 0)
        actor.update(role=4, slot=5, group_id=7, group_ptr=1000)
        self.frames = []
        for frame in range(3):
            row = dict.fromkeys(GATE.comparison.SCALARS, 0)
            row.update(dict.fromkeys(GATE.comparison.FINISH_FIELDS, 0))
            row.update(type="frame", frame=frame, tick=frame, level=1,
                       pool_colors=[-1] * 200, terminal_metadata_recorded=False,
                       actors=[copy.deepcopy(actor), dict(actor, slot=6)])
            self.frames.append(row)
        self.frames[-1].update(terminated=True, truncated=False, active_scene_state=1,
                               finish_call_count=1, terminal_metadata_recorded=True,
                               recorded_final_level=1)
        self.write()
        golden = GATE.fingerprint_trace(self.path, "native", 2)
        golden.pop("metadata")
        self.fixture = {"replay": {"inputs": 2, "seed": 41, "sha256": "a" * 64,
                                   "score": 0, "level": 1, "chain": 0},
                        "original_states": golden}

    def write(self, frames=None, original=False):
        frames = copy.deepcopy(self.frames if frames is None else frames)
        rows = [dict(self.meta, source="original" if original else "native")]
        if original:
            for row in frames:
                for actor in row["actors"]:
                    for field in GATE.comparison.ACTOR_FIELDS:
                        if field.endswith("_bits"):
                            actor[field] = f"{actor[field]:08x}"
                row["actors"].reverse()
            rows += frames[:-1]
            rows.append({"type": "finish", "finish_call": 1, "scene_frame": 1,
                         "input_index": 2, "score": 0, "level": 1, "highest_chain": 0})
            rows += frames[-1:]
            rows.append({"type": "complete", "frame": 2})
        else:
            rows += frames
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_original_hex_bits_group_addresses_and_actor_order_normalize(self):
        self.write(original=True)
        measured = GATE.fingerprint_trace(self.path, "original", 2)
        self.assertEqual(measured["normalized_state_sha256"],
                         self.fixture["original_states"]["normalized_state_sha256"])

    def test_one_ulp_and_wrong_group_membership_fail(self):
        for field, value in (("x_bits", 1), ("group_id", 8)):
            with self.subTest(field=field):
                frames = copy.deepcopy(self.frames)
                frames[1]["actors"][0][field] = value
                self.write(frames)
                with self.assertRaisesRegex(GATE.comparison.TraceError, "normalized_state_sha256"):
                    GATE.check_trace(self.path, self.fixture, "config")

    def test_header_canonical_finish_mismatch_fails(self):
        self.fixture["replay"]["score"] = 10
        with self.assertRaisesRegex(GATE.comparison.TraceError, "header"):
            GATE.check_trace(self.path, self.fixture, "config")

    def test_missing_frame_fails(self):
        self.write([self.frames[0], self.frames[2]])
        with self.assertRaisesRegex(GATE.comparison.TraceError, "frame 1"):
            GATE.check_trace(self.path, self.fixture, "config")

    def test_wrong_golden_digest_and_missing_natural_finish_fail(self):
        self.fixture["original_states"]["normalized_state_sha256"] = "f" * 64
        with self.assertRaisesRegex(GATE.comparison.TraceError, "normalized_state_sha256"):
            GATE.check_trace(self.path, self.fixture, "config")
        self.frames[-1]["terminated"] = False
        self.write()
        with self.assertRaisesRegex(GATE.comparison.TraceError, "naturally finish"):
            GATE.check_trace(self.path, self.fixture, "config")


if __name__ == "__main__":
    unittest.main()
