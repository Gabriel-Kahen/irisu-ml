#!/usr/bin/env python3
"""Compare actual original-game and native Metsu states at every replay update.

Raw float32 bits are compared exactly. Missing state, duplicate slots, coverage
gaps, incomplete captures and unbound original traces cannot produce a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
from typing import Iterator

SCHEMA = "irisu-metsu-state-trace-v1"
ORIGINAL_EXE = "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"
SCALARS = ("tick", "score", "gauge", "level", "qualifying_clears",
           "highest_chain", "rng_calls", "pool_colors")
ACTOR_FIELDS = ("role", "ex", "ex_timer", "color", "size_slot", "size_bits",
                "x_bits", "y_bits", "angle_bits", "vx_bits", "vy_bits",
                "physics_updates", "age", "remaining_lifetime", "freshness",
                "grouped", "physics_owned", "special", "rot_timer", "projectile_hits",
                "clear_pending", "delete_marked", "non_wall_contacts",
                "gate_seen", "gate_enabled", "group_chain", "group_shadow", "group_num")
FINISH_FIELDS = ("finish_call_count", "terminal_metadata_recorded", "recorded_final_score",
                 "recorded_final_level", "recorded_final_chain", "recorded_final_clears")


def sha(path: Path) -> str:
    with path.open("rb") as source:
        result = hashlib.sha256()
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


class TraceError(ValueError):
    pass


class Trace:
    def __init__(self, path: Path):
        self.path = path
        self.source = path.open()
        self.metadata = self._read(self.source.readline(), 1)
        if self.metadata.get("type") != "metadata" or self.metadata.get("schema") != SCHEMA:
            raise TraceError(f"{path}: missing versioned trace metadata")
        self.complete = None
        self.last = None
        self.bootstrap = None
        self.finishes = []

    def _read(self, line: str, number: int) -> dict:
        try:
            value = json.loads(line)
        except (ValueError, TypeError) as error:
            raise TraceError(f"{self.path}:{number}: incomplete or invalid JSON") from error
        if not isinstance(value, dict):
            raise TraceError(f"{self.path}:{number}: expected an object")
        return value

    def frames(self) -> Iterator[dict]:
        expected = 0
        for number, line in enumerate(self.source, 2):
            value = self._read(line, number)
            if value.get("type") == "bootstrap" and not expected and self.bootstrap is None:
                if type(value.get("title_passes")) is not int or value["title_passes"] < 0:
                    raise TraceError(f"{self.path}:{number}: invalid bootstrap receipt")
                self.bootstrap = value
                continue
            if value.get("type") == "finish" and self.complete is None:
                if value.get("finish_call") != len(self.finishes) + 1:
                    raise TraceError(f"{self.path}:{number}: invalid finish-call sequence")
                for field in ("score", "level", "highest_chain", "scene_frame", "input_index"):
                    if type(value.get(field)) is not int:
                        raise TraceError(f"{self.path}:{number}: finish-entry field missing: {field}")
                self.finishes.append(value)
                continue
            if value.get("type") == "complete":
                if self.complete is not None or value.get("frame") != expected - 1:
                    raise TraceError(f"{self.path}:{number}: invalid completion record")
                self.complete = value
                continue
            if self.complete is not None or value.get("type") != "frame":
                raise TraceError(f"{self.path}:{number}: unexpected record")
            if type(value.get("frame")) is not int or value["frame"] != expected:
                raise TraceError(f"{self.path}:{number}: missing, duplicated or reordered frame {expected}")
            self.last = value
            expected += 1
            yield value


def normalized(field: str, value):
    if field.endswith("_bits"):
        if isinstance(value, str):
            if len(value) != 8:
                raise TraceError(f"invalid float32 bit string for {field}")
            try:
                value = int(value, 16)
            except ValueError as error:
                raise TraceError(f"invalid float32 bits for {field}") from error
        if type(value) is not int or not 0 <= value <= 0xffffffff:
            raise TraceError(f"invalid float32 word for {field}")
    return value


def actors(row: dict) -> dict[int, dict]:
    if not isinstance(row.get("actors"), list):
        raise TraceError(f"frame {row['frame']}: missing actor list")
    result = {}
    for actor in row["actors"]:
        if not isinstance(actor, dict) or "role" not in actor:
            raise TraceError(f"frame {row['frame']}: actor role is missing")
        if actor["role"] not in (4, 5):
            continue  # Static field actors are not native Body objects.
        slot = actor.get("slot")
        if type(slot) is not int or not 0 <= slot < 200 or slot in result:
            raise TraceError(f"frame {row['frame']}: invalid or duplicate actor slot")
        result[slot] = actor
    return result


def mismatch(frame: int, field: str, original, candidate) -> dict:
    result = {"frame": frame, "field": field, "original": original, "candidate": candidate}
    if field.endswith("_bits") and isinstance(original, int) and isinstance(candidate, int):
        result["original_float"] = struct.unpack("<f", struct.pack("<I", original))[0]
        result["candidate_float"] = struct.unpack("<f", struct.pack("<I", candidate))[0]
    return result


def group_partition(live: dict[int, dict], key: str) -> list[list[int]]:
    groups = {}
    for slot, actor in live.items():
        identity = actor.get(key)
        if type(identity) is not int or not 0 <= identity <= 0xffffffff:
            raise TraceError(f"slot {slot}: missing or invalid group-link identity: {key}")
        if identity:
            groups.setdefault(identity, []).append(slot)
    return sorted(sorted(slots) for slots in groups.values())


def compare(original: Path, candidate: Path, replay: Path, receipt_path: Path | None = None,
            through_frame: int | None = None, diagnostic: bool = False,
            require_natural_terminal: bool = False) -> dict:
    data = replay.read_bytes()
    if len(data) < 56 or (len(data) - 52) % 4:
        raise TraceError("a nonempty padded v2.03 replay is required")
    seed, level, score, chain, mode = struct.unpack_from("<Iiiii", data)
    if mode != 1:
        raise TraceError("mode-1 replay required; normal is a separate gate")
    count = (len(data) - 52) // 4
    expected_last = count if through_frame is None else through_frame
    if not 1 <= expected_last <= count:
        raise TraceError("comparison must include reset and at least one valid input update")
    expected_hash = hashlib.sha256(data).hexdigest()
    report = {"schema": "irisu-metsu-original-state-comparison-v1", "mode": 1,
              "scope": "complete-replay" if through_frame is None else "bounded-prefix",
              "diagnostic_only": diagnostic, "replay_sha256": expected_hash,
              "qualification": "saved-replay-interchange" if require_natural_terminal else "state-parity",
              "require_natural_terminal": require_natural_terminal,
              "original_trace_sha256": sha(original), "candidate_trace_sha256": sha(candidate),
              "replay_header": {"seed": seed, "level": level, "score": score, "chain": chain},
              "requested_last_frame": expected_last, "scalar_fields": list(SCALARS),
              "actor_fields": list(ACTOR_FIELDS), "excluded_static_roles": [2, 3, 6],
              "compared_frames": 0, "compared_actor_states": 0,
              "compared_group_partitions": 0,
              "group_identity": "live non-null links, including pending EX: original group_ptr versus native group_id",
              "first_mismatch": None, "coverage_complete": False, "state_parity_passed": False,
              "interop_passed": False if require_natural_terminal else None, "passed": False}
    left, right = Trace(original), Trace(candidate)
    try:
        for trace, source in [(left, "original"), (right, "native")]:
            meta = trace.metadata
            if meta.get("source") != source or meta.get("mode") != 1 or meta.get("boundary") != "after-actor-pass":
                raise TraceError(f"{source}: wrong source, mode or update boundary")
        original_meta = left.metadata
        exe = original_meta.get("game_exe_sha256", original_meta.get("original_executable_sha256"))
        if exe != ORIGINAL_EXE or original_meta.get("execution") != "original-process" or original_meta.get("gameplay_stubs") is not False:
            raise TraceError("original trace is not an attested un-stubbed v2.03 process capture")
        if right.metadata.get("physics_backend") != "exact-msvc9-r58-metsu-forward":
            raise TraceError("candidate trace must use the exact Metsu backend")
        if right.metadata.get("seed") != seed or right.metadata.get("replay_sha256") != expected_hash:
            raise TraceError("native trace does not identify this replay")
        bound = original_meta.get("replay_sha256") == expected_hash
        receipt = None
        if receipt_path:
            receipt = json.loads(receipt_path.read_text())
            checks = {"schema": "irisu-metsu-original-capture-v1", "replay_sha256": expected_hash,
                      "seed": seed, "mode": 1, "replay_input_count": count,
                      "original_trace_sha256": report["original_trace_sha256"],
                      "original_executable_sha256": ORIGINAL_EXE,
                      "execution": "original-process", "gameplay_stubs": False,
                      "boundary": "after-actor-pass"}
            for field, expected in checks.items():
                if receipt.get(field) != expected:
                    raise TraceError(f"capture receipt mismatch: {field}")
            report["original_receipt_sha256"] = sha(receipt_path)
            bound = True
        report["provenance_complete"] = bound
        if not bound and not diagnostic:
            raise TraceError("original trace is not bound to this replay; supply its capture receipt")
        if original_meta.get("replay_sha256") not in (None, "", expected_hash):
            raise TraceError("original trace identifies a different replay")
        report["environment"] = {
            "original_observer_preload_cw": original_meta.get("observer_preload_cw"),
            "candidate_x87_cw": right.metadata.get("x87_cw")}
        report["candidate_identity"] = {key: right.metadata.get(key) for key in
                                        ("physics_backend", "config_hash", "tracer_sha256", "x87_cw")}
        missing = object()
        left_frames, right_frames = left.frames(), right.frames()
        for a, b in itertools.zip_longest(left_frames, right_frames, fillvalue=missing):
            if a is missing or b is missing:
                raise TraceError("trace frame coverage differs")
            frame = a["frame"]
            if a.get("seed", seed) != seed:
                raise TraceError("original frame seed differs from replay")
            for field in SCALARS:
                if field not in a or field not in b:
                    raise TraceError(f"frame {frame}: required scalar missing: {field}")
                if field == "pool_colors" and (len(a[field]) != 200 or len(b[field]) != 200):
                    raise TraceError(f"frame {frame}: actor-pool color coverage is incomplete")
                if a[field] != b[field]:
                    report["first_mismatch"] = mismatch(frame, field, a[field], b[field])
                    return report
            for field in FINISH_FIELDS:
                if field not in a:
                    continue
                if field not in b:
                    raise TraceError(f"frame {frame}: native finish-state field missing: {field}")
                if a[field] != b[field]:
                    report["first_mismatch"] = mismatch(frame, field, a[field], b[field])
                    return report
            aa, bb = actors(a), actors(b)
            if aa.keys() != bb.keys():
                report["first_mismatch"] = mismatch(frame, "actor_slots", sorted(aa), sorted(bb))
                return report
            for slot in sorted(aa):
                for field in ACTOR_FIELDS:
                    if field not in aa[slot] or field not in bb[slot]:
                        raise TraceError(f"frame {frame} slot {slot}: required actor field missing: {field}")
                    av, bv = normalized(field, aa[slot][field]), normalized(field, bb[slot][field])
                    if av != bv:
                        report["first_mismatch"] = mismatch(frame, f"actors[{slot}].{field}", av, bv)
                        return report
                report["compared_actor_states"] += 1
            original_groups = group_partition(aa, "group_ptr")
            candidate_groups = group_partition(bb, "group_id")
            if original_groups != candidate_groups:
                report["first_mismatch"] = mismatch(frame, "group_partition", original_groups, candidate_groups)
                return report
            report["compared_group_partitions"] += 1
            report["compared_frames"] += 1
            if frame == expected_last:
                if through_frame is None:
                    # Exhaust both generators to verify their explicit ending and
                    # reject hidden extra/reordered rows after the expected frame.
                    if next(left_frames, None) is not None or next(right_frames, None) is not None:
                        raise TraceError("trace contains extra frames beyond the replay")
                    if left.complete is None:
                        raise TraceError("original process capture has no completion record")
                report["coverage_complete"] = True
                if not report["compared_actor_states"]:
                    raise TraceError("trace contains no body-state comparisons")
                report["state_parity_passed"] = True
                report["last_state"] = {field: a[field] for field in SCALARS if field != "pool_colors"}
                report["candidate_natural_terminal"] = bool(b.get("terminated"))
                report["original_bootstrap"] = left.bootstrap
                report["original_finish_events"] = left.finishes
                report["compared_finish_fields"] = [field for field in FINISH_FIELDS if field in a]
                if require_natural_terminal:
                    if through_frame is not None or diagnostic:
                        raise TraceError("saved-replay qualification requires the complete replay, not a diagnostic prefix")
                    if a.get("active_scene_state") != 1:
                        raise TraceError("original game is still active at replay EOF")
                    if b.get("terminated") is not True or b.get("truncated") is not False:
                        raise TraceError("native replay did not reach a natural terminal")
                    if b.get("terminal_metadata_recorded") is not True:
                        raise TraceError("native first-finish metadata is missing")
                    header = (score, level, chain)
                    native_finish = tuple(b.get(field) for field in
                                          ("recorded_final_score", "recorded_final_level", "recorded_final_chain"))
                    if native_finish != header:
                        raise TraceError("replay header differs from native canonical first-finish metadata")
                    finish = a.get("first_finish")
                    if left.finishes:
                        first = left.finishes[0]
                        original_finish = (first["score"], first["level"], first["highest_chain"])
                        report["original_first_finish_source"] = "original-process-finish-hook-entry"
                    elif a.get("terminal_metadata_recorded") is True:
                        original_finish = tuple(a.get(field) for field in
                                                ("recorded_final_score", "recorded_final_level", "recorded_final_chain"))
                        report["original_first_finish_source"] = "original-process-first-finish-latch"
                    elif isinstance(finish, dict):
                        original_finish = (finish.get("score"), finish.get("level"),
                                           finish.get("highest_chain", finish.get("chain")))
                        report["original_first_finish_source"] = "original-process-finish-hook"
                    elif (receipt and receipt.get("execution_mode") == "live-original-recorder"
                          and receipt.get("original_generated_replay", {}).get("sha256") == expected_hash):
                        written = receipt["original_generated_replay"]
                        original_finish = (written.get("score"), written.get("level"), written.get("chain"))
                        report["original_first_finish_source"] = "hash-bound-original-writer-receipt"
                    else:
                        raise TraceError("actual original first-finish metadata or original-writer receipt is required")
                    if original_finish != header:
                        raise TraceError("replay header differs from original canonical first-finish metadata")
                    report["canonical_first_finish"] = {"score": score, "level": level, "chain": chain}
                    report["interop_passed"] = True
                report["passed"] = not diagnostic and bound
                return report
        raise TraceError("trace ended before the requested replay frame")
    except TraceError as error:
        report["validation_error"] = str(error)
        return report
    finally:
        left.source.close()
        right.source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--through-frame", type=int)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--require-natural-terminal", action="store_true",
                        help="qualify a saved replay only after both games naturally finish and canonical headers match")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.original, args.candidate, args.replay, args.receipt,
                     args.through_frame, args.diagnostic, args.require_natural_terminal)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
