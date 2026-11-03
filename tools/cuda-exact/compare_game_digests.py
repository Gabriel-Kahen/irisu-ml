#!/usr/bin/env python3
"""Compare game-digest-v1 trajectories, reporting the first differing field."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

FIELDS = "tick body_hash observation_hash event_hash event_count transition_hash body_count score gauge level highest_chain latest_final_score".split()
ROW = struct.Struct("<12Q")


def compare(expected: Path, actual: Path) -> dict[str, object]:
    a, b = expected.read_bytes(), actual.read_bytes()
    if not a or len(a) % ROW.size or len(b) % ROW.size:
        raise ValueError("empty or incomplete game-digest-v1 trajectory")
    if len(a) != len(b):
        raise ValueError(f"trajectory lengths differ: {len(a)//ROW.size} / {len(b)//ROW.size} rows")
    for index, (left, right) in enumerate(zip(ROW.iter_unpack(a), ROW.iter_unpack(b))):
        for field, x, y in zip(FIELDS, left, right):
            if x != y:
                raise ValueError(f"row {index}, expected tick {left[0]}: {field} differs: {x:#x} / {y:#x}")
    return {"version": "game-digest-v1-comparison", "rows": len(a)//ROW.size,
            "mismatches": 0, "trajectory_sha256": hashlib.sha256(a).hexdigest(),
            "scope": "All encoded scalar/body/event/transition digests match; FNV-1a is not a collision-resistant proof of unobserved internal state."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.expected, args.actual), indent=2))


if __name__ == "__main__":
    main()
