#!/usr/bin/env python3
"""Execute hash-pinned original Metsu contact components with isolated actors.

Requires x86 Linux and GCC's 32-bit libc toolchain. Never starts the game or
loads original presentation resources. Output is component evidence only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

EXE_SHA256 = "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"


def dump(executable: Path) -> dict[str, object]:
    if hashlib.sha256(executable.read_bytes()).hexdigest() != EXE_SHA256:
        raise ValueError("the pinned v2.03 original executable is required")
    source = Path(__file__).with_name("original_contacts_probe.c")
    with tempfile.TemporaryDirectory(prefix="irisu-metsu-contacts-") as tmp:
        probe = Path(tmp) / "probe"
        subprocess.run(["gcc", "-m32", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(source), "-o", str(probe)], check=True)
        result = subprocess.run([str(probe), str(executable.resolve())], check=True,
                                capture_output=True, text=True, timeout=10)
    raw = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    if len(raw) != 21 or len({row["case"] for row in raw}) != 21:
        raise AssertionError("unexpected component case inventory")
    cases = [{key: value if key == "case" else int(value)
              for key, value in row.items()} for row in raw]
    return {
        "schema": "metsu-original-contacts-component-oracle-v1",
        "original_executable_sha256": EXE_SHA256,
        "harness_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "x87_control_word": "0x027f",
        "case_count": len(cases),
        "original_routines": {
            "actor_post_contact_update": "0x4033d0",
            "clear_color": "0x4032d0",
            "directional_grouping": "0x40a35c",
            "burst": "0x40a510",
            "rot_start": "0x40a66c",
            "top_gate": "0x40a6f0",
            "special": "0x40a748",
            "activation": "0x40a7d8",
            "direct_hit": "0x40a89c",
            "both_scripted": "0x40a958",
            "clear_reward": "0x4098fc",
            "mark_clear": "0x403ea4",
        },
        "stubs": {
            "0x4160e0": "this-pointer assertion helper: return",
            "0x416268": "configuration lookup: shipped statge_level_norma=6 only",
            "0x418964": "debug logging: return",
            "0x419080": "allocate a zeroed 3-int CHAIN record",
            "0x4077c4": "base actor transform, age and lifetime update: excluded",
            "0x407c98": "particle effect: return",
            "0x431946 / 0x43194c": "sound APIs: return",
            "actor virtual +0x1c": "destruction hook: record delete=1 only",
            "actor virtual +0x54": "post-contact cleanup: return true",
            "actor virtual +0x5c": "presentation refresh: return",
            "manager virtual +0x20": "sound lookup: return zero",
            "manager virtual +0x48": "scene mode: Metsu=1, normal positive control=0",
        },
        "scope": "Original contact/update component execution with isolated actor state. "
                 "Physics, base actor integration, complete post-contact cleanup, full-game "
                 "replay parity and long-run trajectories are excluded. Normal positive "
                 "color-clear control keeps reward unit120 to isolate the scene-mode branch.",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dump(args.executable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case_count": result["case_count"], "output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
