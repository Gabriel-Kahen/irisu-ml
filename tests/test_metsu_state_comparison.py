from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metsu_state_comparison", ROOT / "tools/metsu/compare_state_traces.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class StateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.replay = self.base / "input.rpy"
        self.replay.write_bytes(struct.pack("<Iiiii32sII", 41, 1, 0, 0, 1, b"", 0, 0))
        self.replay_hash = hashlib.sha256(self.replay.read_bytes()).hexdigest()
        self.original = self.base / "original.jsonl"
        self.native = self.base / "native.jsonl"
        actor = {field: 0 for field in MODULE.ACTOR_FIELDS}
        actor.update(slot=5, role=4, ex_timer=-1, remaining_lifetime=9999,
                     freshness=2, group_ptr=0, group_id=0)
        self.frames = [
            {"type": "frame", "frame": frame, "tick": frame, "score": 0, "gauge": 1000,
             "level": 1, "qualifying_clears": 0, "highest_chain": 0,
             "rng_calls": 51, "pool_colors": [-1] * 200, "actors": [copy.deepcopy(actor)]}
            for frame in range(3)
        ]
        self.meta = {"type": "metadata", "schema": MODULE.SCHEMA, "mode": 1,
                     "boundary": "after-actor-pass", "seed": 41,
                     "replay_sha256": self.replay_hash}
        self.original_meta = dict(self.meta, source="original", execution="original-process",
                                  game_exe_sha256=MODULE.ORIGINAL_EXE, gameplay_stubs=False)
        self.native_meta = dict(self.meta, source="native",
                                physics_backend="exact-msvc9-r58-metsu-forward")

    def write(self, original=None, native=None, complete=True):
        for path, metadata, frames in [
            (self.original, self.original_meta, self.frames if original is None else original),
            (self.native, self.native_meta, self.frames if native is None else native),
        ]:
            rows = [metadata] + copy.deepcopy(frames)
            if path == self.original and complete and frames:
                rows.append({"type": "complete", "frame": frames[-1]["frame"]})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def compare(self, **kwargs):
        return MODULE.compare(self.original, self.native, self.replay, **kwargs)

    def test_full_coverage_and_raw_hex_bits_pass(self):
        original = copy.deepcopy(self.frames)
        for row in original:
            for actor in row["actors"]:
                for field in MODULE.ACTOR_FIELDS:
                    if field.endswith("_bits"):
                        actor[field] = f"{actor[field]:08x}"
        self.write(original=original)
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["compared_frames"], 3)
        self.assertEqual(result["compared_actor_states"], 3)

    def test_one_float32_bit_difference_is_not_rounded_away(self):
        native = copy.deepcopy(self.frames)
        native[1]["actors"][0]["x_bits"] = 1
        self.write(native=native)
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_mismatch"]["frame"], 1)
        self.assertEqual(result["first_mismatch"]["field"], "actors[5].x_bits")

    def test_missing_required_field_is_not_a_common_field_pass(self):
        native = copy.deepcopy(self.frames)
        del native[1]["actors"][0]["ex_timer"]
        self.write(native=native)
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIn("ex_timer", result["validation_error"])

    def test_same_truncation_in_both_traces_does_not_pass(self):
        self.write(original=self.frames[:2], native=self.frames[:2])
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIn("ended before", result["validation_error"])

    def test_original_completion_record_is_required(self):
        self.write(complete=False)
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIn("completion record", result["validation_error"])

    def test_duplicate_actor_slots_are_rejected(self):
        original = copy.deepcopy(self.frames)
        original[1]["actors"].append(copy.deepcopy(original[1]["actors"][0]))
        self.write(original=original)
        self.assertIn("duplicate actor slot", self.compare()["validation_error"])

    def test_frame_gap_is_rejected(self):
        self.write(original=[self.frames[0], self.frames[2]])
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIn("frame 1", result["validation_error"])

    def test_empty_body_capture_cannot_pass_by_absence_of_evidence(self):
        rows = copy.deepcopy(self.frames)
        for row in rows:
            row["actors"] = []
        self.write(original=rows, native=rows)
        self.assertIn("no body-state comparisons", self.compare()["validation_error"])

    def test_actor_order_and_implementation_body_ids_do_not_define_identity(self):
        original, native = copy.deepcopy(self.frames), copy.deepcopy(self.frames)
        for a, b in zip(original, native):
            a["actors"].append(dict(a["actors"][0], slot=6, body_id=100))
            b["actors"].append(dict(b["actors"][0], slot=6, body_id=999))
            b["actors"].reverse()
        self.write(original=original, native=native)
        self.assertTrue(self.compare()["passed"])

    def test_equal_counters_with_wrong_group_sharing_are_rejected(self):
        original, native = copy.deepcopy(self.frames), copy.deepcopy(self.frames)
        for a, b in zip(original, native):
            a["actors"][0].update(group_ptr=1000, grouped=1)
            b["actors"][0].update(group_id=1, grouped=1)
            a["actors"].append(dict(a["actors"][0], slot=6))
            b["actors"].append(dict(b["actors"][0], slot=6, group_id=2))
        self.write(original=original, native=native)
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_mismatch"]["field"], "group_partition")

    def test_group_pointer_values_can_differ_when_live_slot_partition_matches(self):
        original, native = copy.deepcopy(self.frames), copy.deepcopy(self.frames)
        for a, b in zip(original, native):
            # EX countdown links affect future chains before grouped becomes true.
            a["actors"][0].update(group_ptr=1000, grouped=0)
            b["actors"][0].update(group_id=7, grouped=0)
            a["actors"].append(dict(a["actors"][0], slot=6))
            b["actors"].append(dict(b["actors"][0], slot=6))
        self.write(original=original, native=native)
        self.assertTrue(self.compare()["passed"])

    def test_unbound_original_requires_a_hash_bound_receipt(self):
        self.original_meta.pop("replay_sha256")
        self.write()
        self.assertFalse(self.compare()["passed"])
        receipt = {
            "schema": "irisu-metsu-original-capture-v1", "replay_sha256": self.replay_hash,
            "seed": 41, "mode": 1, "replay_input_count": 2,
            "original_trace_sha256": MODULE.sha(self.original),
            "original_executable_sha256": MODULE.ORIGINAL_EXE,
            "execution": "original-process", "gameplay_stubs": False,
            "boundary": "after-actor-pass",
        }
        path = self.base / "receipt.json"
        path.write_text(json.dumps(receipt))
        self.assertTrue(self.compare(receipt_path=path)["passed"])
        receipt["original_trace_sha256"] = "0" * 64
        path.write_text(json.dumps(receipt))
        self.assertFalse(self.compare(receipt_path=path)["passed"])

    def test_diagnostic_prefix_cannot_be_reported_as_passing(self):
        self.write()
        result = self.compare(through_frame=1, diagnostic=True)
        self.assertTrue(result["coverage_complete"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["scope"], "bounded-prefix")

    def terminal_rows(self):
        original, native = copy.deepcopy(self.frames), copy.deepcopy(self.frames)
        original[-1].update(active_scene_state=1, first_finish={"score": 0, "level": 1, "chain": 0})
        native[-1].update(terminated=True, truncated=False, terminal_metadata_recorded=True,
                          recorded_final_score=0, recorded_final_level=1, recorded_final_chain=0)
        return original, native

    def test_state_match_at_active_eof_is_not_saved_replay_interoperability(self):
        self.write()
        result = self.compare(require_natural_terminal=True)
        self.assertTrue(result["state_parity_passed"])
        self.assertFalse(result["interop_passed"])
        self.assertFalse(result["passed"])
        self.assertIn("still active", result["validation_error"])

    def test_natural_terminal_requires_both_canonical_first_finishes(self):
        original, native = self.terminal_rows()
        self.write(original=original, native=native)
        self.assertTrue(self.compare(require_natural_terminal=True)["interop_passed"])
        native[-1]["recorded_final_score"] = 1
        self.write(original=original, native=native)
        self.assertIn("native canonical", self.compare(require_natural_terminal=True)["validation_error"])
        native[-1]["recorded_final_score"] = 0
        original[-1]["first_finish"]["score"] = 1
        self.write(original=original, native=native)
        self.assertIn("original canonical", self.compare(require_natural_terminal=True)["validation_error"])

    def test_latest_post_finish_score_does_not_replace_canonical_score(self):
        original, native = self.terminal_rows()
        original[-1]["score"] = native[-1]["score"] = 8
        self.write(original=original, native=native)
        result = self.compare(require_natural_terminal=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["canonical_first_finish"]["score"], 0)
        self.assertEqual(result["last_state"]["score"], 8)

    def test_missing_original_first_finish_cannot_be_inferred_from_live_score(self):
        original, native = self.terminal_rows()
        original[-1].pop("first_finish")
        self.write(original=original, native=native)
        result = self.compare(require_natural_terminal=True)
        self.assertFalse(result["passed"])
        self.assertIn("actual original first-finish", result["validation_error"])

    def test_bootstrap_and_transparent_finish_entry_are_preserved(self):
        original, native = self.terminal_rows()
        original[-1].pop("first_finish")
        self.write(original=original, native=native)
        rows = [json.loads(line) for line in self.original.read_text().splitlines()]
        rows.insert(1, {"type": "bootstrap", "title_passes": 61,
                        "first_title_cw": "137f", "warm_title_cw": "137f"})
        rows.insert(-2, {"type": "finish", "finish_call": 1, "scene_frame": 1,
                         "input_index": 2, "score": 0, "level": 1, "highest_chain": 0})
        self.original.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.compare(require_natural_terminal=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["original_bootstrap"]["title_passes"], 61)
        self.assertEqual(result["original_first_finish_source"], "original-process-finish-hook-entry")

    def test_hash_bound_actual_writer_receipt_supplies_canonical_finish(self):
        original, native = self.terminal_rows()
        original[-1].pop("first_finish")
        self.write(original=original, native=native)
        receipt = {
            "schema": "irisu-metsu-original-capture-v1", "replay_sha256": self.replay_hash,
            "seed": 41, "mode": 1, "replay_input_count": 2,
            "original_trace_sha256": MODULE.sha(self.original),
            "original_executable_sha256": MODULE.ORIGINAL_EXE,
            "execution": "original-process", "gameplay_stubs": False,
            "boundary": "after-actor-pass", "execution_mode": "live-original-recorder",
            "original_generated_replay": {"sha256": self.replay_hash, "score": 0, "level": 1, "chain": 0},
        }
        path = self.base / "writer-receipt.json"
        path.write_text(json.dumps(receipt))
        result = self.compare(receipt_path=path, require_natural_terminal=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["original_first_finish_source"], "hash-bound-original-writer-receipt")


if __name__ == "__main__":
    unittest.main()
