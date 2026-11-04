#!/usr/bin/env python3
"""Capture every native Metsu replay state without requesting physics snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(tracer: Path, replay: Path, output: Path, maximum: int | None = None) -> dict:
    data = replay.read_bytes()
    if len(data) < 56 or (len(data) - 52) % 4 or struct.unpack_from("<i", data, 16)[0] != 1:
        raise ValueError("a nonempty original-format v2.03 mode-1 replay is required")
    inputs = (len(data) - 52) // 4
    if maximum is not None and not 1 <= maximum <= inputs:
        raise ValueError("maximum input count is outside the replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LD_") and key != "GLIBC_TUNABLES"}
    environment["IRISU_EXACT_CW"] = "0x027f"
    replay_hash, tracer_hash = sha(replay), sha(tracer)
    command = [str(tracer.resolve()), str(replay.resolve()), replay_hash]
    if maximum is not None:
        command.append(str(maximum))
    with tempfile.TemporaryDirectory(prefix="metsu-trace-", dir=output.parent) as temporary:
        raw = Path(temporary) / "raw.jsonl"
        with raw.open("wb") as destination:
            subprocess.run(command, cwd="/", env=environment, stdout=destination, check=True)
        if sha(tracer) != tracer_hash or sha(replay) != replay_hash:
            raise ValueError("tracer or replay changed during capture")
        staged = Path(temporary) / "trace.jsonl"
        with raw.open() as source, staged.open("w") as destination:
            metadata = json.loads(source.readline())
            if metadata.get("physics_backend") != "exact-msvc9-r58-metsu-forward":
                raise ValueError("the original-game comparison requires the exact Metsu tracer")
            metadata.update(tracer_sha256=tracer_hash, replay_bytes=len(data),
                            executed_command=command)
            destination.write(json.dumps(metadata, sort_keys=True) + "\n")
            shutil.copyfileobj(source, destination)
        staged.replace(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-inputs", type=int)
    args = parser.parse_args()
    print(json.dumps(capture(args.tracer, args.replay, args.output, args.max_inputs), indent=2))


if __name__ == "__main__":
    main()
