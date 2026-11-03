#!/usr/bin/env python3
"""Bounded-memory, resumable binary32 CUDA-versus-x87 exhaustive gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--first-exponent", type=int, default=0)
parser.add_argument("--last-exponent", type=int, default=145)
parser.add_argument("--signs", choices=("positive", "negative", "both"), default="both")
args = parser.parse_args()
if not 0 <= args.first_exponent <= args.last_exponent <= 145:
    parser.error("supported exhaustive range is exponent bins 0..145")
args.output.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment.pop("CUDA_VISIBLE_DEVICES", None)
signs = (0, 1) if args.signs == "both" else (int(args.signs == "negative"),)
started = time.time()
binary_hash = hashlib.sha256(args.executable.read_bytes()).hexdigest()
source_hashes = {
    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in ("arithmetic_validate.cu", "x87_numeric.cuh", "arithmetic_sweep.py")
}
reports = []
for sign in signs:
    for exponent in range(args.first_exponent, args.last_exponent + 1):
        start = (sign << 31) | (exponent << 23)
        path = args.output / f"sign{sign}-exponent{exponent:03d}.json"
        if path.exists():
            report = json.loads(path.read_text())
        else:
            result = subprocess.run(
                [str(args.executable.resolve()), "--trig-sweep", hex(start), "8388608"],
                env=environment, text=True, capture_output=True, check=True,
            )
            report = json.loads(result.stdout)
            report["validation_binary_sha256"] = binary_hash
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(path)
        assert report["pass"] and report["samples"] == 8388608
        assert report.get("validation_binary_sha256", binary_hash) == binary_hash
        assert report["trig"]["supported"] == 8388608
        assert report["contiguous_range"] == {"start": start, "count": 8388608}
        reports.append(report)
        print(f"sign={sign} exponent={exponent}: exact {report['samples']:,}", flush=True)
summary = {
    "schema": 1,
    "scope": "final_binary32_sin_cos_values_only_not_full_x87_or_physics_parity",
    "first_exponent": args.first_exponent,
    "last_exponent": args.last_exponent,
    "signs": list(signs),
    "compared": sum(r["samples"] for r in reports),
    "gpu_mismatches": sum(r["trig"]["gpu_mismatches"] for r in reports),
    "host_mismatches": sum(r["trig"]["host_mismatches"] for r in reports),
    "cuda_libm_mismatches": sum(r["trig"]["cuda_libm_mismatches_on_supported"] for r in reports),
    "run_seconds": time.time() - started,
    "resumed_chunks_included": True,
    "gpu": reports[0]["gpu"],
    "validation_binary_sha256": binary_hash,
    "source_sha256_at_start": source_hashes,
    "pass": True,
}
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
