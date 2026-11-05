#!/usr/bin/env python3
"""Check a newly compiled exact Metsu engine against recorded original states.

This command only reads the committed original-state manifest. It never
generates or updates golden values and needs no original game installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import compare_state_traces as comparison
from trace_replay import capture

SCHEMA = "irisu-metsu-original-state-digests-v1"
BACKEND = "exact-msvc9-r58-metsu-forward"
REQUIRED_FIXTURES = {"browser-export", "original-generated", "controlled-level4"}
PROJECTION = {
    "schema": "irisu-metsu-observed-frame-v1",
    "scalar_fields": list(comparison.SCALARS),
    "actor_fields": list(comparison.ACTOR_FIELDS),
    "finish_fields": list(comparison.FINISH_FIELDS),
    "actor_identity": "ascending original actor-pool slot",
    "group_identity": "partition of all live non-null group links, including pending EX",
    "encoding": "SHA256 over uint32-le byte length then UTF-8 compact sorted-key JSON for each frame",
}
FLAG_FIELDS = {"ex", "grouped", "physics_owned", "special", "clear_pending",
               "delete_marked", "gate_seen", "gate_enabled"}


def integer(value, field: str) -> int:
    if type(value) is not int:
        raise comparison.TraceError(f"integer state field missing or invalid: {field}")
    return value


def normalized_frame(row: dict, source: str) -> dict:
    """Project only the observed state, excluding addresses and build metadata."""
    frame = integer(row.get("frame"), "frame")
    scalars = {}
    for field in comparison.SCALARS:
        value = row.get(field)
        if field == "pool_colors":
            if not isinstance(value, list) or len(value) != 200:
                raise comparison.TraceError("all 200 actor-pool colors are required")
            value = [integer(color, "pool_colors") for color in value]
        else:
            value = integer(value, field)
        scalars[field] = value
    finish = {}
    for field in comparison.FINISH_FIELDS:
        value = row.get(field)
        if field == "terminal_metadata_recorded":
            if type(value) is not bool:
                raise comparison.TraceError("boolean first-finish latch is required")
        else:
            value = integer(value, field)
        finish[field] = value
    live = comparison.actors(row)
    actors = []
    for slot, actor in sorted(live.items()):
        projected = {"slot": slot}
        for field in comparison.ACTOR_FIELDS:
            if field not in actor:
                raise comparison.TraceError(f"frame {frame} slot {slot}: missing {field}")
            value = comparison.normalized(field, actor[field])
            if field in FLAG_FIELDS and type(value) is bool:
                value = int(value)
            projected[field] = integer(value, field)
        actors.append(projected)
    group_key = "group_ptr" if source == "original" else "group_id"
    return {"frame": frame, "scalars": scalars, "actors": actors, "finish": finish,
            "groups": comparison.group_partition(live, group_key)}


def fingerprint_trace(path: Path, source: str, input_count: int) -> dict:
    """Fingerprint every reset/update row and require an observed natural finish."""
    if source not in {"original", "native"}:
        raise comparison.TraceError("expected original or native trace source")
    trace = comparison.Trace(path)
    digest = hashlib.sha256()
    frames = actor_states = 0
    last = None
    try:
        if trace.metadata.get("source") != source or trace.metadata.get("mode") != 1:
            raise comparison.TraceError("trace source/mode is incompatible")
        if trace.metadata.get("boundary") != "after-actor-pass":
            raise comparison.TraceError("trace update boundary is incompatible")
        if source == "native" and trace.metadata.get("physics_backend") != BACKEND:
            raise comparison.TraceError("the exact Metsu trace backend is required")
        for row in trace.frames():
            projected = normalized_frame(row, source)
            encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            frames += 1
            actor_states += len(projected["actors"])
            last = row
        if frames != input_count + 1 or not actor_states:
            raise comparison.TraceError("trace does not cover reset and every replay input")
        if source == "original":
            if trace.complete is None or last.get("active_scene_state") != 1 or not trace.finishes:
                raise comparison.TraceError("actual original process natural-finish evidence is required")
            first = trace.finishes[0]
            canonical = {"score": first["score"], "level": first["level"], "chain": first["highest_chain"]}
        else:
            if last.get("terminated") is not True or last.get("truncated") is not False:
                raise comparison.TraceError("native replay did not naturally finish")
            canonical = {"score": last.get("recorded_final_score"), "level": last.get("recorded_final_level"),
                         "chain": last.get("recorded_final_chain")}
        if last.get("terminal_metadata_recorded") is not True:
            raise comparison.TraceError("canonical first-finish state is missing")
        if canonical != {"score": last["recorded_final_score"], "level": last["recorded_final_level"],
                         "chain": last["recorded_final_chain"]}:
            raise comparison.TraceError("finish-entry tuple and retained canonical state disagree")
        return {"normalized_state_sha256": digest.hexdigest(), "frames": frames,
                "actor_states": actor_states, "group_partitions": frames,
                "canonical_first_finish": canonical, "metadata": trace.metadata}
    finally:
        trace.source.close()


def replay_header(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 56 or (len(data) - 52) % 4:
        raise comparison.TraceError("expected a nonempty padded v2.03 replay")
    seed, level, score, chain, mode = struct.unpack_from("<Iiiii", data)
    if mode != 1:
        raise comparison.TraceError("a mode-1 replay is required")
    return {"seed": seed, "level": level, "score": score, "chain": chain, "mode": mode,
            "inputs": (len(data) - 52) // 4, "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def check_trace(path: Path, fixture: dict, expected_config: str) -> dict:
    measured = fingerprint_trace(path, "native", fixture["replay"]["inputs"])
    metadata = measured["metadata"]
    if (metadata.get("config_hash") != expected_config
            or metadata.get("seed") != fixture["replay"]["seed"]
            or metadata.get("replay_sha256") != fixture["replay"]["sha256"]):
        raise comparison.TraceError("native trace configuration or replay identity differs")
    expected_finish = {key: fixture["replay"][key] for key in ("score", "level", "chain")}
    if measured["canonical_first_finish"] != expected_finish:
        raise comparison.TraceError("replay header differs from canonical native first finish")
    expected = fixture["original_states"]
    for field in ("frames", "actor_states", "group_partitions", "canonical_first_finish",
                  "normalized_state_sha256"):
        if measured[field] != expected[field]:
            raise comparison.TraceError(
                f"original golden mismatch: {field}; expected {expected[field]}, got {measured[field]}")
    return measured


def check(tracer: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("projection") != PROJECTION:
        raise comparison.TraceError("unsupported original-state manifest/projection")
    if comparison.sha(ROOT / manifest["qualification_report"]) != manifest["qualification_report_sha256"]:
        raise comparison.TraceError("tracked original qualification report changed")
    fixtures = manifest.get("fixtures")
    if (not isinstance(fixtures, list) or len(fixtures) != 3
            or {fixture.get("name") for fixture in fixtures} != REQUIRED_FIXTURES):
        raise comparison.TraceError("all three original-qualified replay fixtures are required")
    report = {"schema": "irisu-metsu-original-fixture-check-v1",
              "scope": "recorded original-state regression; not a new original-process execution",
              "manifest_sha256": comparison.sha(manifest_path),
              "tracer_sha256": comparison.sha(tracer), "config_hash": manifest["config_hash"],
              "projection": PROJECTION, "fixtures": [], "passed": False}
    with tempfile.TemporaryDirectory(prefix="metsu-original-fixtures-") as directory:
        for fixture in fixtures:
            result = {"name": fixture["name"], "passed": False}
            try:
                base = manifest_path.parent.resolve()
                replay = (base / fixture["path"]).resolve()
                if not replay.is_relative_to(base) or replay_header(replay) != fixture["replay"]:
                    raise comparison.TraceError("tracked replay content or header differs from the golden manifest")
                evidence = fixture["evidence"]
                original_report_path = ROOT / evidence["comparison_report"]
                if comparison.sha(original_report_path) != evidence["comparison_report_sha256"]:
                    raise comparison.TraceError("tracked original comparison receipt changed")
                original_report = json.loads(original_report_path.read_text())
                if (original_report.get("passed") is not True
                        or original_report.get("interop_passed") is not True
                        or original_report.get("original_trace_sha256") != evidence["original_trace_sha256"]
                        or original_report.get("original_receipt_sha256") != evidence["original_receipt_sha256"]
                        or original_report.get("replay_sha256") != fixture["replay"]["sha256"]):
                    raise comparison.TraceError("original comparison does not bind the expected trace/receipt/replay")
                trace = Path(directory) / (fixture["name"] + ".jsonl")
                capture(tracer, replay, trace)
                measured = check_trace(trace, fixture, manifest["config_hash"])
                result.update(measured)
                result["passed"] = True
            except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
                result["error"] = str(error)
            report["fixtures"].append(result)
    report["passed"] = all(fixture["passed"] for fixture in report["fixtures"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "tests/fixtures/metsu/original-state-digests.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(args.manifest.resolve().parent):
        parser.error("--output cannot overwrite the fixture/golden directory")
    report = check(args.tracer.resolve(), args.manifest.resolve())
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
