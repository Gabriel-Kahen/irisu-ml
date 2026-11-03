#!/usr/bin/env python3
"""Dump Metsu's exact level stores using only the pinned original setters.

Needs x86 Linux, GCC and its 32-bit libc development/runtime packages. The
original executable remains an ignored local input; no entrypoint is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXE_SHA256 = "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"


def dump(executable: Path) -> tuple[str, dict[str, object]]:
    if hashlib.sha256(executable.read_bytes()).hexdigest() != EXE_SHA256:
        raise ValueError("the hash-pinned v2.03 original executable is required")
    source = Path(__file__).with_name("level_probe.c")
    with tempfile.TemporaryDirectory(prefix="irisu-metsu-levels-") as tmp:
        probe = Path(tmp) / "probe"
        subprocess.run(["gcc", "-m32", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(source), "-o", str(probe)], check=True)
        result = subprocess.run([str(probe), str(executable.resolve())], check=True,
                                capture_output=True, text=True, timeout=10)
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    normal, metsu = rows[:99], rows[99:]
    if len(rows) != 198 or any(row["mode"] != str(i // 99) or int(row["level"]) != i % 99 + 1
                               for i, row in enumerate(rows)):
        raise AssertionError("unexpected setter row inventory")
    original_scale = list(csv.DictReader(
        (line for line in (ROOT / "reference/score-scale-binary64.tsv").read_text().splitlines()
         if not line.startswith("#")), delimiter="\t"))
    normal_source = (ROOT / "clone/core/normal_rules.cpp").read_text()
    descent_section = normal_source.split("kScriptedDescentBits{", 1)[1].split("};", 1)[0]
    original_descent = re.findall(r"0x([0-9a-f]+)U", descent_section)
    if len(original_scale) != 99 or len(original_descent) != 99:
        raise AssertionError("normal reference inventory changed")
    for row, scale, descent in zip(normal, original_scale, original_descent, strict=True):
        if int(row["score_scale_bits"], 16) != int(scale["binary64_bits"], 16):
            raise AssertionError(f"normal original-runtime scale differs at level {row['level']}")
        if int(row["descent_bits"], 16) != int(descent, 16):
            raise AssertionError(f"normal original-runtime descent differs at level {row['level']}")
    output = io.StringIO()
    columns = [key for key in rows[0] if key != "mode"]
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in columns} for row in metsu)
    table = output.getvalue()
    report = {
        "schema": "metsu-original-setter-component-oracle-v1",
        "original_executable_sha256": EXE_SHA256,
        "harness_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "x87_control_word": "0x027f",
        "normal_reference_scale_rows_matched": len(normal),
        "normal_reference_descent_rows_matched": len(normal),
        "metsu_level_rows": len(metsu),
        "table_sha256": hashlib.sha256(table.encode()).hexdigest(),
        "original_routines": {"normal_setter": "0x415624", "metsu_setter": "0x415800",
                              "power_wrapper": "0x41a5e0", "power": "0x427110", "absolute": "0x41a578"},
        "stubs": {"0x416268": "config lookup: shipped float -0.05 and int120 only",
                  "0x418964": "debug logging: return",
                  "0x40ffc4": "normal shape RNG: zero (not used by Metsu setter)"},
        "scope": "Original pure parameter component execution; no game entrypoint, scene, physics or replay parity claim.",
    }
    return table, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    table, report = dump(args.executable)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
