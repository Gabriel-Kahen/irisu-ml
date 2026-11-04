#!/usr/bin/env python3
"""Build/validate/measure CUDA arena scheduling; this does not run physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ROOT / "clone/cuda/batch.hpp",
    ROOT / "clone/cuda/batch.cu",
    ROOT / "tools/cuda-exact/batch_validate.cu",
]


def capture(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="/opt/cuda/bin/nvcc")
    parser.add_argument("--host-compiler", default="g++-14")
    parser.add_argument("--gpu", default="0", help="physical CUDA device for this child process")
    parser.add_argument("--parents", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--stride", type=int, default=65536)
    parser.add_argument("--depths", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--matrix", action="store_true", help="also measure 64 and 1024 parents")
    parser.add_argument("--sanitize", action="store_true", help="also run memcheck, racecheck, and synccheck")
    parser.add_argument("--sanitizer", default="/opt/cuda/compute-sanitizer/compute-sanitizer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    build = ROOT / ".tmp/cuda-exact"
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "irisu-cuda-batch-validate"
    compile_command = [
        args.nvcc, "-std=c++20", "-O3", "-arch=sm_70", "-ccbin", args.host_compiler,
        "--fmad=false", "-Xcompiler=-Wall,-Wextra,-fopenmp",
        str(SOURCES[1]), str(SOURCES[2]), "-o", str(binary),
    ]
    print("Building CUDA arena validator", file=sys.stderr)
    subprocess.run(compile_command, cwd=ROOT, check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    # Pinned CPU workers help make repeated runs comparable without changing
    # affinities, clocks, power limits, drivers, or any other process.
    env.update(OMP_PROC_BIND="true", OMP_PLACES="cores")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--id=" + args.gpu,
         "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits", "--loop-ms=100"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    runs, commands = [], []
    try:
        parent_counts = sorted({64, args.parents, 1024}) if args.matrix else [args.parents]
        for parents in parent_counts:
            for cpu_threads in args.cpu_threads:
                command = [str(binary), "--parents", str(parents),
                           "--fanout", str(args.fanout), "--stride", str(args.stride),
                           "--depths", str(args.depths), "--repeats", str(args.repeats),
                           "--cpu-threads", str(cpu_threads)]
                print(f"Validating/measuring {parents} parents, CPU threads={cpu_threads}", file=sys.stderr)
                completed = subprocess.run(command, cwd=ROOT, env=env, check=True,
                                           text=True, capture_output=True)
                runs.append(json.loads(completed.stdout))
                commands.append(command)
    finally:
        monitor.terminate()
        try:
            telemetry, monitor_error = monitor.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            telemetry, monitor_error = monitor.communicate()

    sanitizer_results = []
    if args.sanitize:
        for tool in ("memcheck", "racecheck", "synccheck"):
            command = [args.sanitizer, "--tool", tool, "--error-exitcode", "99"]
            if tool == "memcheck":
                command += ["--leak-check", "full"]
            command += [str(binary), "--parents", "16", "--fanout", "4", "--stride", "256",
                        "--depths", "2", "--repeats", "1", "--cpu-threads", "1"]
            print(f"Checking {tool}", file=sys.stderr)
            completed = subprocess.run(command, cwd=ROOT, env=env, check=True,
                                       text=True, capture_output=True)
            sanitizer_results.append({"tool": tool, "command": command,
                                      "exit_code": completed.returncode,
                                      "stdout": completed.stdout.splitlines(),
                                      "stderr": completed.stderr.splitlines()})

    result = {
        "schema": "irisu-cuda-batch-benchmark-v1",
        "scope": "device-resident checkpoint/branch/rank/compact with synthetic integer scoring",
        "physics_parity": False,
        "promotable": False,
        "not_a_simulation_speedup": True,
        "timestamp_unix": time.time(),
        "git_head": capture("git", "rev-parse", "HEAD"),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in SOURCES},
        "binary_sha256": digest(binary),
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown"),
        "nvcc": capture(args.nvcc, "--version"),
        "gpu": capture("nvidia-smi", "--id=" + args.gpu,
                       "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,power.limit",
                       "--format=csv,noheader,nounits"),
        "compile_command": compile_command,
        "process_environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_PROC_BIND", "OMP_PLACES")},
        "commands": commands,
        "timing_contract": {
            "warmup": "one CPU pipeline, one GPU pipeline, one CUDA graph pipeline before samples",
            "included": "fixed-budget branch copies, synthetic integer scoring, stable rank, survivor copies; GPU wall includes final synchronization",
            "excluded": "allocation, initial upload, graph construction, final download, correctness checks",
            "cpu_parallelism": "OpenMP independent branch/evaluation/compaction loops; deterministic serial stable sort",
            "transfers_in_depth_loop": "none; branch specs and ranking indices remain on device",
            "metrics_limit": "telemetry is whole-device, 100ms sampling; peaks are not process-exclusive",
        },
        "runs": runs,
        "sanitizer_results": sanitizer_results,
        "telemetry_columns": ["timestamp", "utilization_percent", "memory_used_mib", "power_w", "temperature_c"],
        "telemetry_csv": telemetry.splitlines(),
        "telemetry_stderr": monitor_error,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            stream.write(serialized)
            temporary = Path(stream.name)
        temporary.replace(args.output)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
