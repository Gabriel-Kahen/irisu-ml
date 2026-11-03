#!/usr/bin/env python3
"""Require unchanged normal-mode responses against the pinned released worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from irisu_env.exact_ipc import ExactWorkerClient  # noqa: E402

BASELINE_WORKER_SHA256 = "8ef81521e81a5b2a764c305ac48dad997b28476bcd2fccbd1c9aed9603322854"
EXACT_LIBRARY_SHA256 = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"
NORMAL_CONFIG_HASH = 0xEC0E8463FEAF2670
REPLAY_SHA256 = "81fe723b6b825e62afb471d7517355fef2b8bf0fb1dbe636069c97c70ce2ca79"


def check(worker: Path, baseline: Path) -> dict[str, object]:
    replay = ROOT / "irisu-high-score-4393101.rpy"
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_WORKER_SHA256:
        raise ValueError("baseline must be the pinned published normal-mode worker")
    if hashlib.sha256(replay.read_bytes()).hexdigest() != REPLAY_SHA256:
        raise ValueError("published normal-mode regression replay changed")
    for path in (baseline, worker):
        with ExactWorkerClient(path.resolve()) as client:
            if client.info.config_hash != NORMAL_CONFIG_HASH:
                raise ValueError(f"normal-mode default configuration changed: {path}")
            if client.info.exact_library_sha256 != EXACT_LIBRARY_SHA256:
                raise ValueError(f"pinned exact physics host required: {path}")
    spec = importlib.util.spec_from_file_location(
        "normal_cpu_differential", ROOT / "tools/cuda-exact/compare_cpu_workers.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compare(baseline, worker, replay)
    result["guard"] = "pinned-normal-mode-regression-v1"
    result["normal_config_hash"] = str(NORMAL_CONFIG_HASH)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True,
                        help="newly built normal worker to validate")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="published runtime guest/irisu-exact-worker")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(check(args.worker, args.baseline), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
