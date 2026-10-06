#!/usr/bin/env python3
"""Rank precomputed exact probe candidates with a development reserve band."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from irisu_pointer.development_reserve_band import (  # noqa: E402,F401
    ProbeCandidate,
    RankedCandidate,
    ReserveBandConfig,
    VERSION,
    choose_candidate,
    evaluate_manifest,
    rank_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("input JSON root must be an object")
    report = evaluate_manifest(source)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
