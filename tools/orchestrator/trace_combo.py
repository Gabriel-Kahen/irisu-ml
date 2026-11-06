#!/usr/bin/env python3
"""Produce a replay-bound native combo genealogy and integrity receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from irisu_search.diagnostics import trace_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--oracle-library", default="")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = SimpleNamespace(oracle=str(args.trace.resolve().with_name("irisu-game-oracle")),
                             oracle_library=args.oracle_library, engine_timeout=args.timeout)
    store = SimpleNamespace(root=args.output.parent, event=lambda *_: None)
    receipt = trace_replay(store, config, args.replay, args.output)
    print(json.dumps({"receipt": str(Path(str(args.output) + ".receipt.json").resolve()),
                      "replay_sha256": receipt["replay_sha256"],
                      "score": receipt["summary"]["canonical_score"]}))


if __name__ == "__main__":
    main()
