#!/usr/bin/env python3
"""Audit float load/store boundaries at direct trig calls in the pinned host.

This is a static audit of one fixed image, not a proof that arbitrary x87
transcendentals can be replaced by binary32 math. Numeric helper accuracy still
needs independent differential validation and an explicit admitted input range.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


HOST_SHA256 = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"
INSTRUCTION = re.compile(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*([a-z][a-z0-9]*)\s*(.*)")


def audit(host: Path) -> dict[str, object]:
    if hashlib.sha256(host.read_bytes()).hexdigest() != HOST_SHA256:
        raise ValueError("audit requires the pinned exact host image")
    listing = subprocess.check_output(["objdump", "-d", "-Mintel", str(host)], text=True)
    instructions = []
    for line in listing.splitlines():
        match = INSTRUCTION.fullmatch(line)
        if match:
            instructions.append((int(match[1], 16), match[2], match[3]))
    calls, categories = [], Counter()
    for index, (address, mnemonic, operand) in enumerate(instructions):
        if mnemonic != "call" or not re.search(r"<msvc_CI(cos|sin)>", operand):
            continue
        successor = instructions[index + 1]
        if successor[1] != "fstp" or not successor[2].startswith("DWORD PTR"):
            raise AssertionError(f"unrounded trig result at {address:#x}")
        prior = [item for item in instructions[max(0, index - 14):index] if item[1].startswith("f")]
        is_load = lambda item: item[1] == "fld" and item[2].startswith("DWORD PTR")
        if is_load(prior[-1]):
            category = "immediate_binary32_load"
        elif prior[-1][1] == "fst" and prior[-1][2].startswith("DWORD PTR") and is_load(prior[-2]):
            category = "binary32_load_then_nonpopping_store"
        elif address == 0x5e56 and [item[0] for item in prior[-4:]] == [0x5e3a, 0x5e42, 0x5e4a, 0x5e52]:
            # FLD f32; FST f32; FLD f32; FSTP f32 leaves original f32 at ST0.
            if not (is_load(prior[-4]) and prior[-3][1] == "fst" and is_load(prior[-2]) and prior[-1][1] == "fstp"):
                raise AssertionError("unexpected nested-load trig input")
            category = "binary32_load_retained_below_nested_load_store"
        else:
            raise AssertionError(f"unproved trig input at {address:#x}")
        categories[category] += 1
        calls.append({"call": hex(address), "target": operand.split("<")[1].rstrip(">"),
                      "input_proof": category, "return_store": hex(successor[0])})
    if len(calls) != 94:
        raise AssertionError(f"unexpected direct call inventory: {len(calls)}")
    sites = {address: mnemonic for address, mnemonic, _ in instructions
             if mnemonic in ("fsin", "fcos", "fsincos")}
    expected = {0x15309: "fsincos", 0x15316: "fcos", 0x1531a: "fsin", 0x15349: "fsin"}
    if sites != expected:
        raise AssertionError(f"unexpected hardware trig sites: {sites}")
    return {
        "version": "fixed-host-trig-load-store-audit-v1",
        "exact_library_sha256": HOST_SHA256,
        "direct_calls": len(calls),
        "immediate_binary32_result_stores": len(calls),
        "binary32_input_proofs": dict(categories),
        "hardware_trig_sites": {hex(k): v for k, v in sites.items()},
        "runtime_flow": {
            "0x15309": "FSINCOS; FXCH; FSTP binary32 sine cache; RET cosine; immediate caller FSTP binary32",
            "0x15316": "fallback FCOS; FXCH; FSIN; FSTP binary32 sine cache; RET cosine; immediate caller FSTP binary32",
            "0x15349": "standalone FSIN; RET; immediate caller FSTP binary32",
        },
        "limits": [
            "Only the identified direct call sites in this exact hashed image are audited.",
            "Float rounding boundaries do not prove numeric equivalence of any replacement helper.",
            "CORE-MATH bounded x87 corrections do not establish unrestricted-angle equivalence.",
            "No GPU instruction, full solver, or game-state parity is claimed by this audit.",
        ],
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(audit(args.host), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
