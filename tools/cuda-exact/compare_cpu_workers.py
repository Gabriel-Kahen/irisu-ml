#!/usr/bin/env python3
"""Compare every response byte of two exact CPU workers on one padded replay.

This is an engineering equivalence check, not a change to the historical
training/promotion identity. Body states, events and terminal metadata all enter
the compared payload and its SHA-256 transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from irisu_env.exact_ipc import ExactWorkerClient, _decode_transition  # noqa: E402


def compare(left_path: Path, right_path: Path, replay: Path) -> dict[str, object]:
    data = replay.read_bytes()
    if len(data) <= 52 or (len(data) - 52) % 4:
        raise ValueError("expected a padded v2.03 replay")
    seed, level, score, chain, mode = struct.unpack_from("<5I", data)
    if mode != 0:
        raise ValueError("normal-mode replay required")
    digest = hashlib.sha256()
    total = 0
    start = time.perf_counter()
    with ExactWorkerClient(left_path.resolve()) as left, ExactWorkerClient(right_path.resolve()) as right:
        if left.info.exact_library_sha256 != right.info.exact_library_sha256:
            raise AssertionError("exact library identities differ")
        if left.reset(seed) != right.reset(seed):
            raise AssertionError("reset observation mismatch")
        for frame, (word,) in enumerate(struct.iter_unpack("<I", data[52:])):
            for worker in (left, right):
                worker.send_step(word & 3, (word >> 2) & 0x3ff, (word >> 12) & 0x1ff,
                                 suppress_fresh_edges=frame < 2)
            a, b = left.receive_step_payload(), right.receive_step_payload()
            if a != b:
                offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                raise AssertionError(f"step {frame + 1}: payload differs at byte {offset}, lengths {len(a)}/{len(b)}")
            digest.update(struct.pack("<I", len(a)))
            digest.update(a)
            total += len(a)
        result = _decode_transition(a)
        if not result.terminated or result.truncated or not result.terminal_metadata_recorded:
            raise AssertionError("replay did not reach a canonical natural terminal")
        if (result.latest_final_score, result.latest_final_level, result.latest_final_highest_chain) != (score, level, chain):
            raise AssertionError("canonical terminal metadata differs from replay header")
        return {
            "version": "cpu-worker-byte-differential-v1",
            "replay_sha256": hashlib.sha256(data).hexdigest(),
            "left_worker_sha256": left.executable_sha256,
            "right_worker_sha256": right.executable_sha256,
            "exact_library_sha256": left.info.exact_library_sha256,
            "steps": frame + 1,
            "compared_payload_bytes": total,
            "length_prefixed_transcript_sha256": digest.hexdigest(),
            "mismatches": 0,
            "canonical_score": result.latest_final_score,
            "level": result.latest_final_level,
            "highest_chain": result.latest_final_highest_chain,
            "elapsed_seconds": time.perf_counter() - start,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=ROOT / "irisu-high-score-4393101.rpy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(compare(args.left, args.right, args.replay), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
