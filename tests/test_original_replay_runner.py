from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("original_metsu_runner", ROOT / "tools/metsu/run_original_replay.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class OriginalRunValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dll = self.root / "data/dll"
        self.dll.mkdir(parents=True)
        self.source = {"seed": 41, "frames": 2}
        initial = dict(type="frame", phase="reset", frame=0, mode=1, x87_cw="137f",
                       seed=41, input_count=2, input_index=0, playback=1, score=0,
                       gauge=1000, level=1, qualifying_clears=0, highest_chain=0,
                       rng_calls=51, active_scene_state=0)
        middle = dict(initial, phase="tick", frame=1, input_index=1)
        final = dict(initial, phase="tick", frame=2, input_index=2, input_count=0,
                     playback=0, score=123, gauge=1, level=2, qualifying_clears=6,
                     highest_chain=3, active_scene_state=1, finish_call_count=1,
                     terminal_metadata_recorded=True, recorded_final_score=123,
                     recorded_final_level=2, recorded_final_chain=3, recorded_final_clears=6)
        self.rows = [dict(type="metadata", source="original", gameplay_stubs=False),
                     dict(type="bootstrap", title_passes=61), initial, middle,
                     dict(type="finish", finish_call=1, scene_frame=1, input_index=2,
                          score=123, level=2, highest_chain=3, qualifying_clears=6),
                     final, dict(type="complete", frame=2, reason="original-finished")]
        physics = [dict(type="proxy_loaded", real_loaded=True, ok=True),
                   dict(type="step"), dict(type="step")]
        (self.dll / "box2d-trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in physics))

    def check(self, rows=None):
        (self.dll / "metsu-state.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (self.rows if rows is None else rows)))
        return RUNNER.checked_trace(self.root, 2, self.source, 1)

    def test_natural_completion_and_recording_header_bind_to_first_finish(self):
        trace = self.check()
        header = dict(seed=41, frames=2, score=123, level=2, chain=3)
        validation = RUNNER.validate_recording(trace, header)
        self.assertEqual(validation["canonical_tuple"], "matched-original-finish-hook")
        for key, value in (("seed", 42), ("frames", 1), ("score", 124), ("level", 3), ("chain", 4)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    RUNNER.validate_recording(trace, dict(header, **{key: value}))

    def test_rejects_natural_completion_while_original_scene_is_active(self):
        rows = copy.deepcopy(self.rows)
        rows[-2].update(active_scene_state=0, playback=1, input_count=2)
        with self.assertRaisesRegex(ValueError, "finished scene state"):
            self.check(rows)

    def test_rejects_missing_finish_hook_when_observer_advertises_it(self):
        with self.assertRaisesRegex(ValueError, "finish hook"):
            self.check([row for row in self.rows if row["type"] != "finish"])

    def test_rejects_canonical_metadata_changed_after_first_finish(self):
        rows = copy.deepcopy(self.rows)
        rows[-2]["recorded_final_score"] += 1
        with self.assertRaisesRegex(ValueError, "first finish call"):
            self.check(rows)

    def test_older_observer_retains_honest_unavailable_canonical_status(self):
        rows = [copy.deepcopy(row) for row in self.rows if row["type"] != "finish"]
        for key in tuple(rows[-2]):
            if key.startswith("recorded_final_") or key in ("terminal_metadata_recorded", "finish_call_count"):
                del rows[-2][key]
        trace = self.check(rows)
        validation = RUNNER.validate_recording(trace, dict(seed=41, frames=2, score=123, level=2, chain=3))
        self.assertEqual(validation["canonical_tuple"], "unavailable-in-older-observer")

    def test_missing_completion_and_reordered_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion record"):
            self.check(self.rows[:-1])
        rows = copy.deepcopy(self.rows)
        rows[3]["frame"] = 2
        with self.assertRaisesRegex(ValueError, "unordered"):
            self.check(rows)

    def test_actual_wrong_mode_cannot_pass_via_metadata(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["mode"] = 0
        with self.assertRaisesRegex(ValueError, "did not load mode 1"):
            self.check(rows)

    def test_wrong_mode_replay_is_rejected_before_launch(self):
        path = self.root / "normal.rpy"
        path.write_bytes(struct.pack("<Iiiii", 41, 1, 0, 0, 0) + bytes(32) + bytes(4))
        with self.assertRaisesRegex(ValueError, "mode-1"):
            RUNNER.replay_info(path)


if __name__ == "__main__":
    unittest.main()
