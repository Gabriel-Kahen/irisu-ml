"""Record-level reconciliation of exact group genealogy and score attribution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACE = Path(os.environ.get("IRISU_SEARCH_TRACE", ROOT / "build-project-exact/irisu-search-trace"))
REPLAY = ROOT / "irisu-high-score-4393101.rpy"


@unittest.skipUnless(TRACE.is_file(), "build exact irisu-search-trace for genealogy tests")
class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        completed = subprocess.run([str(TRACE), str(REPLAY)], capture_output=True,
                                   text=True, timeout=60, check=True)
        cls.rows = [json.loads(line) for line in completed.stdout.splitlines()]
        cls.identity, cls.summary = cls.rows[0], cls.rows[-1]

    def test_every_score_reconciles_with_live_and_first_terminal_capture(self) -> None:
        summary = self.summary
        scores = [row for row in self.rows if row["kind"] == "score"]
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["unassigned_score_events"], 0)
        self.assertEqual(len(scores), summary["score_events"])
        self.assertEqual(sum(row["delta"] for row in scores), 4_393_699)
        self.assertEqual(sum(row["delta"] for row in scores if row["canonical_eligible"]), 4_393_101)
        self.assertEqual(sum(group["score_live"] for group in summary["groups"]), 4_393_699)
        self.assertEqual(sum(group["score_canonical"] for group in summary["groups"]), 4_393_101)
        self.assertTrue(all(row["canonical_eligible"] == (row["tick"] < summary["first_finish_tick"])
                            for row in scores))
        self.assertEqual(summary["joins"], sum(group["chain"] for group in summary["groups"]))
        self.assertEqual(summary["qualifications"], sum(group["num"] for group in summary["groups"]))
        self.assertEqual(summary["joins"] - summary["removals"],
                         sum(group["members"] for group in summary["groups"]))

    def test_dominant_chain_is_a_real_shared_group_with_remaining_carriers(self) -> None:
        dominant = max(self.summary["groups"], key=lambda group: group["score_canonical"])
        self.assertEqual(dominant["id"], 399)
        self.assertEqual((dominant["chain"], dominant["num"], dominant["scored_pieces"]), (70, 54, 62))
        self.assertEqual(dominant["score_canonical"], 4_155_740)
        self.assertEqual(dominant["first_tick"], 81_803)
        self.assertEqual(dominant["members"], 2)
        self.assertIsNone(dominant["extinct_tick"])
        self.assertEqual(self.identity["source_sha256"],
                         hashlib.sha256((ROOT / "tools/orchestrator/trace_record.cpp").read_bytes()).hexdigest())

    def test_wrong_header_and_nonterminal_replay_fail(self) -> None:
        payload = REPLAY.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.rpy"
            path.write_bytes(payload[:52+4*4])
            failed = subprocess.run([str(TRACE), str(path)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("did not naturally terminate", failed.stderr)
            altered = bytearray(payload)
            struct.pack_into("<I", altered, 8, 4_393_102)
            path.write_bytes(altered)
            failed = subprocess.run([str(TRACE), str(path)], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("disagrees with header", failed.stderr)


if __name__ == "__main__":
    unittest.main()
