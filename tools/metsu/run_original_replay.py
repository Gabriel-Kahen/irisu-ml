#!/usr/bin/env python3
"""Run the authentic v2.03 engine in a fresh game tree, Wine prefix and Xvfb.

The observer selects the original scene/loader and forwards every physics call
into the shipped DLL. This tool validates trace completion; parity with the
clone is a separate check. A live recording is accepted only when the original
process creates replay/new.rpy and reports a natural finish.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PINNED = {
    "irisu.exe": "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255",
    "data/dll/Box2D.dll": "34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd",
    "data/dll/DxLib.dll": "d8ef638a078a8b4d24b53b174ca179623fed3690027d3f4acfe71a7d61c8b5c9",
    "data/doc/irisu.ini": "1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35",
}
OBSERVER_SOURCES = (
    "tools/metsu/original_state_proxy.c", "tools/metsu/state_proxy_kernel32.def",
    "tools/metsu/build_original_state_proxy.sh",
    "reference/trace-proxy/src/box2d_trace_proxy.c",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def replay_info(path: Path, expected_mode: int = 1, *, data: bytes | None = None) -> dict[str, object]:
    data = path.read_bytes() if data is None else data
    if len(data) < 56 or (len(data) - 52) % 4:
        raise ValueError("expected a nonempty padded v2.03 replay")
    seed, level, score, chain, mode = struct.unpack_from("<Iiiii", data)
    if mode != expected_mode:
        raise ValueError(f"the original runner requires a mode-{expected_mode} replay")
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "seed": seed, "level": level, "score": score, "chain": chain,
            "mode": mode, "frames": (len(data) - 52) // 4}


def checked_trace(run: Path, limit: int, source_replay: dict | None, mode: int) -> dict:
    state = run / "data/dll/metsu-state.jsonl"
    physics = run / "data/dll/box2d-trace.jsonl"
    metadata = None
    bootstrap = None
    completion = None
    finish_records = []
    frames = 0
    controls = set()
    first = last = None
    with state.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "metadata":
                if metadata is not None or frames:
                    raise ValueError("duplicate or misplaced original metadata")
                metadata = row
            elif row.get("type") == "bootstrap":
                if bootstrap is not None or frames:
                    raise ValueError("duplicate or misplaced original bootstrap record")
                bootstrap = row
            elif row.get("type") == "frame":
                if metadata is None or completion is not None or row.get("frame") != frames:
                    raise ValueError("original state frames are missing, duplicated or unordered")
                if row.get("mode") != mode:
                    raise ValueError(f"the actual original scene did not load mode {mode}")
                if not re.fullmatch(r"[0-9a-fA-F]{4}", row.get("x87_cw", "")):
                    raise ValueError("missing actual original x87 control word")
                controls.add(row["x87_cw"])
                if source_replay and not (row.get("active_scene_state") and not row.get("playback")):
                    if row.get("seed") != source_replay["seed"] or row.get("input_count") != source_replay["frames"]:
                        raise ValueError("actual original loader seed/input count differs from replay")
                    if row.get("input_index") != min(frames, source_replay["frames"]):
                        raise ValueError("original replay input index is not aligned with the actor boundary")
                if first is None:
                    first = row
                last = row
                frames += 1
            elif row.get("type") == "finish":
                if completion is not None or row.get("finish_call") != len(finish_records) + 1:
                    raise ValueError("original finish records are missing or unordered")
                finish_records.append(row)
            elif row.get("type") == "complete":
                if completion is not None:
                    raise ValueError("duplicate completion record")
                completion = row
            else:
                raise ValueError("unexpected original state record")
    if not metadata or metadata.get("source") != "original" or metadata.get("gameplay_stubs") is not False:
        raise ValueError("missing authentic original-process observer metadata")
    if first is None or first.get("phase") != "reset" or not completion:
        raise ValueError("missing reset or completion record")
    if completion.get("frame") != last["frame"] or completion.get("reason") not in ("original-finished", "frame-limit"):
        raise ValueError("inconsistent original completion record")
    if completion["reason"] == "frame-limit" and last["frame"] != limit:
        raise ValueError("original trace stopped before the requested frame boundary")
    if completion["reason"] == "original-finished":
        if last.get("active_scene_state") != 1:
            raise ValueError("natural completion lacks the original finished scene state")
        if "terminal_metadata_recorded" in last or "finish_call_count" in last:
            if not finish_records or last.get("terminal_metadata_recorded") is not True:
                raise ValueError("natural completion lacks the instrumented original finish hook")
            first_finish = finish_records[0]
            for recorded, observed in (("recorded_final_score", "score"), ("recorded_final_level", "level"),
                                       ("recorded_final_chain", "highest_chain"), ("recorded_final_clears", "qualifying_clears")):
                if last.get(recorded) != first_finish.get(observed):
                    raise ValueError("original canonical metadata disagrees with its first finish call")
    if source_replay and first.get("playback") != 1:
        raise ValueError("the original Input object did not enter replay playback")
    step_count = 0
    proxy = None
    with physics.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "proxy_loaded":
                proxy = row
            elif row.get("type") == "step":
                step_count += 1
    if not proxy or not proxy.get("real_loaded") or not proxy.get("ok") or step_count < last["frame"]:
        raise ValueError("the authentic shipped physics DLL was not fully observed")
    if finish_records and last.get("finish_call_count") != len(finish_records):
        raise ValueError("final original state disagrees with observed finish calls")
    return {"finish_records": finish_records, "frame_rows": frames, "last_frame": last["frame"], "completion": completion,
            "actual_mode": mode, "bootstrap": bootstrap, "x87_control_words": sorted(controls),
            "physics_steps": step_count, "observer_metadata": metadata,
            "state_trace_sha256": digest(state), "physics_trace_sha256": digest(physics),
            "initial_state": {key: first[key] for key in ("seed", "score", "gauge", "level", "rng_calls")},
            "final_state": {key: last[key] for key in ("seed", "score", "gauge", "level", "qualifying_clears", "highest_chain", "active_scene_state")}}



def validate_recording(trace: dict, header: dict) -> dict:
    if trace["completion"]["reason"] != "original-finished":
        raise ValueError("a bounded trace prefix is not a completed original recording")
    if header["seed"] != trace["initial_state"]["seed"]:
        raise ValueError("original recording header seed differs from the observed episode")
    if header["frames"] != trace["last_frame"]:
        raise ValueError("original recording input count differs from the observed episode")
    validation = {"seed_matches": True, "input_count_matches": True,
                  "canonical_tuple": "unavailable-in-older-observer"}
    if trace["finish_records"]:
        finish = trace["finish_records"][0]
        expected = (finish["score"], finish["level"], finish["highest_chain"])
        actual = (header["score"], header["level"], header["chain"])
        if actual != expected:
            raise ValueError("original recording header differs from the first canonical finish tuple")
        validation["canonical_tuple"] = "matched-original-finish-hook"
    return validation


def run(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.experiment):
        raise ValueError("invalid experiment identifier")
    source = args.game.resolve(strict=True)
    wine = args.wine.resolve(strict=True)
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("preserved source game must not contain symlinks")
    for name, expected in PINNED.items():
        if digest(source / name) != expected:
            raise ValueError(f"wrong original v2.03 input: {name}")
    replay_data = args.replay.read_bytes() if args.replay else None
    replay = replay_info(args.replay, args.mode, data=replay_data) if args.replay else None
    limit = args.frames if args.frames is not None else int(replay["frames"]) if replay else 100_000
    if not 1 <= limit <= 100_000 or args.timeout <= 0:
        raise ValueError("invalid frame limit or timeout")
    run_dir = ROOT / "reference/runs" / args.experiment
    capture = ROOT / "reference/captures" / args.experiment
    prefix = ROOT / ".tmp/metsu-original-prefixes" / args.experiment
    if any(path.exists() for path in (run_dir, capture, prefix)):
        raise ValueError("experiment game tree, receipt or Wine prefix already exists")
    capture.mkdir(parents=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "irisu-metsu-original-capture-v1", "experiment": args.experiment,
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
               "mode": args.mode, "execution": "original-process", "boundary": "after-actor-pass",
               "execution_mode": "live-original-recorder" if args.record else "original-replay-playback",
               "original_executable_sha256": PINNED["irisu.exe"],
               "replay_sha256": replay["sha256"] if replay else None,
               "seed": replay["seed"] if replay else None,
               "replay_input_count": replay["frames"] if replay else None,
               "source_game": str(source), "run_directory": str(run_dir), "wine_prefix": str(prefix),
               "wine_executable": str(wine), "wine_executable_sha256": digest(wine),
               "preserved_input_replay": str(args.replay.resolve()) if args.replay else None,
               "wine_version": subprocess.check_output([str(wine), "--version"], text=True).strip(),
               "requested_frames": limit, "timeout_seconds": args.timeout, "input_replay": replay,
               "gameplay_stubs": False, "observer_sources": {p: digest(ROOT / p) for p in OBSERVER_SOURCES}}
    process = server = None
    before = {}
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("WAYLAND_DISPLAY", "WAYLAND_SOCKET") and
                   not key.startswith(("IRISU_METSU_", "WINE", "LD_")) and key != "GLIBC_TUNABLES"}
    wine_arch = args.winearch or ("win32" if wine.read_bytes()[:5] == b"\x7fELF\x01" else "win64")
    environment.update(WINEPREFIX=str(prefix), WINEARCH=wine_arch, WINEDEBUG="-all",
                       WINEDLLOVERRIDES="winewayland.drv=d;mscoree,mshtml=",
                       LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", MESA_LOADER_DRIVER_OVERRIDE="llvmpipe",
                       IRISU_METSU_MAX_FRAME=str(limit), IRISU_ORIGINAL_MODE=str(args.mode))
    if args.title_warmup is not None:
        if not 0 <= args.title_warmup <= 10_000:
            raise ValueError("invalid title warmup")
        environment["IRISU_METSU_TITLE_WARMUP"] = str(args.title_warmup)
    try:
        shutil.copytree(source, run_dir)
        (run_dir / ".irisu-reference-run").write_text("fresh original v2.03 observer run\n")
        before = hashes(run_dir)
        # Never mistake an inherited recording for a file produced by this run.
        (run_dir / "replay/new.rpy").unlink(missing_ok=True)
        if args.replay:
            (run_dir / "input.rpy").write_bytes(replay_data)
            if digest(run_dir / "input.rpy") != replay["sha256"]:
                raise ValueError("staged replay bytes differ from the captured input")
            environment["IRISU_METSU_REPLAY"] = "input.rpy"
            environment["IRISU_METSU_REPLAY_SHA256"] = replay["sha256"]
        else:
            environment["IRISU_METSU_LAUNCH"] = "live"
        build = capture / "observer-build"
        if args.proxy:
            proxy = args.proxy.resolve(strict=True)
        else:
            with (capture / "observer-build.log").open("w") as log:
                subprocess.run([str(ROOT / "tools/metsu/build_original_state_proxy.sh"), str(build)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            proxy = build / "Box2D.dll"
        if receipt["observer_sources"] != {p: digest(ROOT / p) for p in OBSERVER_SOURCES}:
            raise ValueError("observer source changed during the build")
        proxy_bytes = proxy.read_bytes()
        dll = run_dir / "data/dll/Box2D.dll"
        dll.rename(dll.with_name("Box2D.real.dll"))
        dll.write_bytes(proxy_bytes)
        subprocess.run([sys.executable, str(ROOT / "reference/trace-proxy/validate_build.py"), str(dll)],
                       check=True, capture_output=True)
        receipt["observer_dll_sha256"] = digest(dll)
        receipt["original_files_sha256"] = {name: before[name] for name in PINNED}
        read_fd, write_fd = os.pipe()
        with (capture / "xvfb.log").open("wb") as log:
            server = subprocess.Popen(["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "1024x768x24",
                                       "-nolisten", "tcp", "-noreset"], pass_fds=(write_fd,), stdout=log, stderr=subprocess.STDOUT)
        os.close(write_fd)
        try:
            if not select.select([read_fd], [], [], 15)[0]:
                raise RuntimeError("private Xvfb did not announce a display")
            display = os.read(read_fd, 32).decode().strip()
        finally:
            os.close(read_fd)
        if not display.isdigit() or server.poll() is not None:
            raise RuntimeError("private Xvfb failed to start")
        environment["DISPLAY"] = ":" + display
        receipt.update(display=environment["DISPLAY"], xvfb_pid=server.pid,
                       command=[str(wine), "irisu.exe"], environment={key: environment[key] for key in
                       ("DISPLAY", "WINEPREFIX", "WINEARCH", "WINEDLLOVERRIDES", "IRISU_METSU_MAX_FRAME")})
        (capture / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        started = time.monotonic()
        with (capture / "wine.log").open("wb") as log:
            process = subprocess.Popen([str(wine), "irisu.exe"], cwd=run_dir, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("original game timed out; partial traces retained") from error
        receipt.update(exit_code=code, elapsed_seconds=time.monotonic() - started)
        if code:
            raise RuntimeError(f"original game exited with status {code}")
        receipt["trace"] = checked_trace(run_dir, limit, replay, args.mode)
        receipt["original_trace_sha256"] = receipt["trace"]["state_trace_sha256"]
        if digest(run_dir / "irisu.exe") != PINNED["irisu.exe"] or digest(dll.with_name("Box2D.real.dll")) != PINNED["data/dll/Box2D.dll"]:
            raise ValueError("original executable or shipped physics file changed")
        if args.record:
            generated = run_dir / "replay/new.rpy"
            if receipt["trace"]["completion"]["reason"] != "original-finished" or not generated.is_file():
                raise ValueError("original engine did not naturally finish and create replay/new.rpy")
            receipt["original_generated_replay"] = replay_info(generated, args.mode)
            generated_info = receipt["original_generated_replay"]
            receipt["recording_header_validation"] = validate_recording(receipt["trace"], generated_info)
            receipt.update(replay_sha256=generated_info["sha256"], seed=generated_info["seed"],
                           replay_input_count=generated_info["frames"])
            shutil.copyfile(generated, capture / "original-generated.rpy")
        receipt["status"] = f"complete_original_mode{args.mode}_trace"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        except Exception as error:
            cleanup_errors.append(f"game process: {error}")
        try:
            subprocess.run([str(wine.with_name("wineserver")), "-k"], env=environment,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        except Exception as error:
            cleanup_errors.append(f"isolated Wine server: {error}")
        try:
            if server is not None and server.poll() is None:
                server.terminate()
                server.wait(timeout=10)
        except Exception as error:
            cleanup_errors.append(f"private Xvfb: {error}")
        try:
            if run_dir.exists():
                after = hashes(run_dir)
                receipt["changed_game_files"] = {key: {"before": before.get(key), "after": after.get(key)}
                                                 for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
        except Exception as error:
            cleanup_errors.append(f"final game-tree hashes: {error}")
        if cleanup_errors:
            receipt.update(status="failed", cleanup_errors=cleanup_errors)
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (capture / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt.get("cleanup_errors"):
        raise RuntimeError("original-run cleanup failed; failure receipt retained")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--game", type=Path, default=ROOT / "reference/game/irisu-v2.03-jp/irisu203")
    parser.add_argument("--wine", type=Path, default=ROOT / ".tmp/metsu-wine/root/usr/bin/wine")
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--winearch", choices=["win32", "win64"])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--mode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--title-warmup", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        parser.exit(1, f"original run failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
