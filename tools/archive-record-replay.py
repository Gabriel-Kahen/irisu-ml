#!/usr/bin/env python3
"""Exact-verify and atomically save a new record replay by score."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from irisu_rl.replay_archive import archive_record_replay  # noqa: E402


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "irisu_record_evaluate_rpy", ROOT / "tools" / "evaluate-rpy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load exact replay evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path.home() / "Downloads" / "replays",
    )
    parser.add_argument(
        "--score-floor",
        type=int,
        default=0,
        help="known record not yet present in the archive",
    )
    args = parser.parse_args()
    if not args.worker.is_absolute():
        parser.error("--worker must be an explicit absolute path")
    if args.score_floor < 0:
        parser.error("--score-floor must be nonnegative")

    evaluator = _load_evaluator()

    def verify(path: Path):
        return evaluator.evaluate_path(
            path,
            layout="padded",
            worker_path=str(args.worker),
            purpose="promotion",
        )

    try:
        result = archive_record_replay(
            args.replay,
            args.archive_dir.expanduser(),
            verify,
            score_floor=args.score_floor,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
