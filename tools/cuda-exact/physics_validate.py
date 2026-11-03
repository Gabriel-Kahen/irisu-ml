#!/usr/bin/env python3
"""Differential/throughput gate for a bounded native-vs-translated physics fixture."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
HOST_SHA = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def command_run(command: list[str], env: dict[str, str], timeout: float):
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}\n"
                           + completed.stderr.decode(errors="replace"))
    stderr = completed.stderr.decode(errors="replace")
    metadata = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
    return completed.stdout, elapsed, metadata, stderr


def assert_lanes(output: bytes, reference: bytes, lanes: int) -> None:
    if len(output) != lanes * len(reference):
        raise AssertionError(f"output length {len(output)} != {lanes} * {len(reference)}")
    for lane in range(lanes):
        actual = output[lane * len(reference):(lane + 1) * len(reference)]
        if actual != reference:
            word = next(i // 4 for i, (a, b) in enumerate(zip(actual, reference)) if a != b)
            raise AssertionError(f"physics mismatch lane={lane} tick={word // 15 + 1} "
                                 f"body={(word % 15) // 5} field={word % 5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT / "build-gpu-guest/physics-probe-gpu")
    parser.add_argument("--backend", choices=("cuda", "translated-host"), default="cuda")
    parser.add_argument("--oracle", type=Path, default=ROOT / ".tmp/physics-oracle")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=ROOT / "build-gpu-guest/generated/guest_image.bin")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--arena-bytes", type=int, help="optional per-world arena size for the current probe")
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mutations", choices=("both", "off", "on"), default="both")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.repeats <= 20:
        parser.error("steps must be 1..100000, repeats 1..20")
    if any(not 1 <= n <= 65536 for n in args.lanes) or any(not 1 <= n <= 64 for n in args.cpu_threads):
        parser.error("lanes must be 1..65536 and CPU threads 1..64")
    if args.arena_bytes is not None and (args.arena_bytes < 0x80000 or args.arena_bytes > 0x40000000 or args.arena_bytes % 16):
        parser.error("arena-bytes must be aligned to16 and512KiB..1GiB")
    for name in ("probe", "oracle", "library", "image"):
        setattr(args, name, getattr(args, name).resolve())
    if digest(args.library) != HOST_SHA:
        parser.error("the native oracle library must match the pinned exact host")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    source_paths = [ROOT / "tools/cuda-exact/physics_validate.py",
                    ROOT / "tools/cuda-exact/physics_probe.cu",
                    ROOT / "tools/cuda-exact/physics_oracle.c",
                    ROOT / "clone/cuda/guest.hpp", ROOT / "clone/cuda/guest_program.cuh",
                    ROOT / "tools/cuda-exact/translate.py", ROOT / "tools/cuda-exact/x87_numeric.cuh"]
    result = {
        "schema": "irisu-cuda-physics-fixture-v1",
        "scope": "three dynamic bodies plus floor; x/y/angle/vx/vy every tick; identical initial fixture per lane",
        "full_game_or_search_parity": False,
        "probe_backend": args.backend,
        "arena_bytes": args.arena_bytes,
        "not_comparable_to_historical_dense_decisions_per_second": True,
        "timestamp_unix": time.time(),
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "binary_sha256": {name: digest(getattr(args, name)) for name in ("probe", "oracle", "library", "image")},
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown"),
        "gpu": capture(["nvidia-smi", "--id=" + args.gpu,
                        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                        "--format=csv,noheader,nounits"]),
        "timing_contract": {
            "warmup": "one invocation per case/backend/config before timed repetitions",
            "process_wall_seconds": "process creation, runtime startup, world creation, simulation, serialization, download and stdout capture; comparison excluded",
            "cpu_parallelism": "independent native oracle processes launched by a bounded thread pool; process startup is included",
            "gpu_kernel_ms": "probe-reported CUDA events; world setup plus all simulation steps; excludes allocation/upload/download",
            "telemetry": "whole-device nvidia-smi 100ms samples; peaks are not process-exclusive",
            "fixture_limits": "lanes repeat one fixture, no game RNG/spawns/scoring, no dense-board or candidate-search speedup established",
        },
        "commands": [], "references": [], "warmups": [], "runs": [], "pass": False,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    telemetry_file = tempfile.TemporaryFile(mode="w+")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--id=" + args.gpu,
         "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits", "--loop-ms=100"],
        stdout=telemetry_file, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        mutations = (False, True) if args.mutations == "both" else (args.mutations == "on",)
        for mutated in mutations:
            flags = ["--mutations"] if mutated else []
            native = [str(args.oracle), str(args.library), str(args.steps), *flags]
            reference, _, _, _ = command_run(native, env, args.timeout)
            if len(reference) != args.steps * 15 * 4:
                raise AssertionError("unexpected native oracle output size")
            result["references"].append({"mutations": mutated, "bytes_per_lane": len(reference),
                                          "sha256": hashlib.sha256(reference).hexdigest()})
            for lanes in args.lanes:
                gpu = [str(args.probe), str(args.image), str(args.steps),
                       *(["--gpu"] if args.backend == "cuda" else []), "--lanes", str(lanes),
                       *(["--arena-bytes", str(args.arena_bytes)] if args.arena_bytes is not None else []), *flags]
                result["commands"].append(gpu)
                if digest(args.probe) != result["binary_sha256"]["probe"]:
                    raise RuntimeError("probe binary changed during the benchmark")
                warm, warm_seconds, warm_metadata, _ = command_run(gpu, env, args.timeout)
                assert_lanes(warm, reference, lanes)
                result["warmups"].append({"backend": args.backend, "lanes": lanes,
                                           "steps_per_lane": args.steps, "mutations": mutated,
                                           "wall_seconds": warm_seconds, "probe_metadata": warm_metadata,
                                           "compared_words": lanes * args.steps * 15, "mismatches": 0})
                save()
                for backend in (args.backend, "native"):
                    for threads in ([0] if backend != "native" else args.cpu_threads):
                        samples, metadata = [], []
                        if backend == "native":
                            result["commands"].append({"native_command": native, "parallel_processes": threads, "lanes": lanes})
                            with ThreadPoolExecutor(max_workers=threads) as pool:
                                def cpu_lane(_):
                                    return command_run(native, env, args.timeout)[0]
                                for payload in pool.map(cpu_lane, range(lanes)):
                                    assert_lanes(payload, reference, 1)
                                for _ in range(args.repeats):
                                    started = time.perf_counter()
                                    outputs = list(pool.map(cpu_lane, range(lanes)))
                                    samples.append(time.perf_counter() - started)
                                    for payload in outputs:
                                        assert_lanes(payload, reference, 1)
                        else:
                            for _ in range(args.repeats):
                                if digest(args.probe) != result["binary_sha256"]["probe"]:
                                    raise RuntimeError("probe binary changed during the benchmark")
                                payload, seconds, info, _ = command_run(gpu, env, args.timeout)
                                samples.append(seconds)
                                metadata.append(info)
                                assert_lanes(payload, reference, lanes)
                        run = {"backend": backend, "cpu_processes": threads, "lanes": lanes,
                               "steps_per_lane": args.steps, "mutations": mutated,
                               "compared_words_per_repetition": lanes * args.steps * 15,
                               "process_wall_seconds": samples,
                               "median_process_wall_ticks_per_second": lanes * args.steps / statistics.median(samples),
                               "probe_metadata": metadata, "mismatches": 0}
                        kernels = [item["kernel_ms"] for sample in metadata for item in sample
                                   if backend == "cuda" and "kernel_ms" in item]
                        if kernels:
                            run["kernel_ms"] = kernels
                            run["median_kernel_ticks_per_second"] = lanes * args.steps * 1000 / statistics.median(kernels)
                        result["runs"].append(run)
                        print(f"{backend} lanes={lanes} processes={threads} mutations={mutated} "
                              f"median={statistics.median(samples):.6f}s", file=sys.stderr, flush=True)
                        save()
        result["pass"] = True
    except (RuntimeError, AssertionError, subprocess.TimeoutExpired) as error:
        result["failure"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill(); monitor.wait()
        telemetry_file.seek(0)
        result["telemetry_columns"] = ["timestamp", "gpu_utilization_percent", "memory_used_mib", "power_w", "temperature_c"]
        result["telemetry_csv"] = telemetry_file.read().splitlines()
        telemetry_file.close()
        save()
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
