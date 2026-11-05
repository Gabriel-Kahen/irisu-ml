"""Persistent, batched engine transport with bounded waits and process cleanup.

No simulation or tensor-library imports: orchestration operates at batch boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self


class EngineError(RuntimeError):
    """A worker failed, rejected a request, or violated the wire contract."""


def _integer(value: Any, maximum: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in 0..{maximum}")
    return value


def _words(values: Iterable[int]) -> list[int]:
    return [_integer(value, 0x1FFFFF, "input word") for value in values]


class Engine:
    """Own a resident native or CUDA engine process.

    Checkpoint handles are immutable and local to this process. Persist the
    seed and executed words to reconstruct them after a restart.
    """

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 120.0,
                 env: Mapping[str, str] | None = None) -> None:
        if not command or not all(isinstance(s, str) and s for s in command):
            raise ValueError("engine command must contain nonempty strings")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be finite and positive")
        self.command = list(command)
        self.timeout_seconds = float(timeout_seconds)
        self._buffer = bytearray()
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - lifetime is owned by close()
        self._closed = False
        self._process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, start_new_session=True,
            env=None if env is None else dict(env), bufsize=0,
        )
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdout.fileno(), False)
        assert self._process.stdin is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        try:
            self.identity = self._request("identity\n")["identity"]
            if self.identity.get("protocol") != 1:
                raise EngineError("unsupported engine protocol")
            executable = Path(self.command[0]).expanduser()
            if not executable.is_file():
                import shutil
                resolved = shutil.which(self.command[0])
                if not resolved:
                    raise EngineError("cannot identify engine executable")
                executable = Path(resolved)
            self.identity = {**self.identity,
                             "executable": str(executable.resolve()),
                             "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                             "command": self.command}
        except BaseException:
            self.close()
            raise

    def _diagnostic(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0, os.SEEK_END)
        end = self._stderr.tell()
        self._stderr.seek(max(0, end - 4096))
        return self._stderr.read().decode("utf-8", "replace")

    def _request(self, text: str) -> dict[str, Any]:
        if self._closed:
            raise EngineError("engine is closed")
        assert self._process.stdin is not None and self._process.stdout is not None
        deadline = time.monotonic() + self.timeout_seconds
        data = memoryview(text.encode("ascii"))
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                selector.register(self._process.stdin, selectors.EVENT_WRITE)
                while True:
                    if not data and b"\n" in self._buffer:
                        line, _, remaining = self._buffer.partition(b"\n")
                        self._buffer = bytearray(remaining)
                        try:
                            reply = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                        except (ValueError, UnicodeError) as exc:
                            self.close()
                            raise EngineError("engine emitted malformed JSON") from exc
                        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
                            self.close()
                            raise EngineError("engine response lacks boolean status")
                        if not reply["ok"]:
                            raise EngineError(str(reply.get("error", "engine rejected request")))
                        return reply
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError("engine request timed out")
                    events = selector.select(remaining_seconds)
                    if not events:
                        raise TimeoutError("engine request timed out")
                    for key, _ in events:
                        if key.fileobj is self._process.stdin:
                            try:
                                sent = os.write(key.fd, data[:65536])
                            except BlockingIOError:
                                continue
                            data = data[sent:]
                            if not data:
                                selector.unregister(self._process.stdin)
                        else:
                            try:
                                chunk = os.read(key.fd, 65536)
                            except BlockingIOError:
                                continue
                            if not chunk:
                                diagnostic = self._diagnostic()
                                self.close()
                                raise EngineError(f"engine exited: {diagnostic}")
                            self._buffer.extend(chunk)
                            if len(self._buffer) > 256 * 1024 * 1024:
                                self.close()
                                raise EngineError("engine response exceeds 256 MiB")
        except (OSError, TimeoutError, UnicodeError) as exc:
            self.close()
            raise EngineError(str(exc)) from exc

    @staticmethod
    def _result(result: Any) -> dict[str, Any]:
        if (not isinstance(result, dict) or type(result.get("handle")) is not int
                or not 1 <= result["handle"] <= 0xFFFFFFFF):
            raise EngineError("malformed checkpoint result")
        summary = result.get("summary")
        if not isinstance(summary, dict) or not isinstance(result.get("words"), list):
            raise EngineError("missing checkpoint summary or trace")
        if summary.get("failed", False) is not False:
            raise EngineError("engine accepted a failed simulation state")
        try:
            _words(result["words"])
        except ValueError as exc:
            raise EngineError("engine returned an invalid replay word") from exc
        for key in ("seed", "tick", "score", "gauge", "level", "qualifying",
                    "highest_chain", "canonical_score", "canonical_level", "canonical_chain"):
            if type(summary.get(key)) is not int:
                raise EngineError(f"invalid summary field {key}")
        for key in ("terminated", "truncated"):
            if type(summary.get(key)) is not bool:
                raise EngineError(f"invalid summary flag {key}")
        if not isinstance(summary.get("groups"), list) or not isinstance(summary.get("bodies"), list):
            raise EngineError("missing group/body summary")
        metrics = result.get("metrics")
        if metrics is not None and (not isinstance(metrics, dict) or any(
            type(metrics.get(key)) is not int or metrics[key] < 0
            for key in ("recruitment_events", "new_groups", "max_new_chain", "qualifications")
        )):
            raise EngineError("invalid interval metrics")
        return result

    def load(self, seed: int, words: Iterable[int] = ()) -> dict[str, Any]:
        seed = _integer(seed, 0xFFFFFFFF, "game seed")
        values = _words(words)
        reply = self._request(f"load {seed} {len(values)} " + " ".join(map(str, values)) + "\n")
        result = self._result(reply.get("result"))
        if result["summary"]["seed"] != seed or result["words"] != values[:len(result["words"])]:
            raise EngineError("load trace or seed disagrees with request")
        if result["summary"]["tick"] != len(result["words"]):
            raise EngineError("loaded checkpoint tick disagrees with trace")
        return result

    def batch(self, handle: int, horizon: int,
              candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        handle = _integer(handle, 0xFFFFFFFF, "checkpoint handle")
        horizon = _integer(horizon, 100000, "horizon")
        if not horizon or not candidates or len(candidates) > 1024:
            raise ValueError("batch requires a positive horizon and 1..1024 candidates")
        lines = [f"batch {handle} {horizon} {len(candidates)}"]
        for candidate in candidates:
            mode = _integer(candidate.get("mode", 0), 3, "continuation mode")
            seed = _integer(candidate.get("search_seed", 0), 0xFFFFFFFF, "search seed")
            values = _words(candidate.get("words", ()))
            if len(values) > horizon:
                raise ValueError("candidate prefix exceeds rollout horizon")
            lines.append(f"{mode} {seed} {len(values)} " + " ".join(map(str, values)))
        reply = self._request("\n".join(lines) + "\n")
        results = reply.get("results")
        rejected = reply.get("rejected", [])
        partial = "rejected" in reply
        if (not isinstance(results, list) or not isinstance(rejected, list)
                or len(results) + len(rejected) != len(candidates)):
            raise EngineError("engine returned wrong candidate count")
        indices = set()
        for position, result in enumerate(results):
            self._result(result)
            if len(result["words"]) > horizon:
                raise EngineError("engine exceeded requested horizon")
            index = result.get("candidate_index", None if partial else position)
            if type(index) is not int or not 0 <= index < len(candidates) or index in indices:
                raise EngineError("invalid or duplicate candidate index")
            indices.add(index)
            result["candidate_index"] = index
        for failure in rejected:
            if (not isinstance(failure, dict) or "handle" in failure
                    or not isinstance(failure.get("words"), list)
                    or not isinstance(failure.get("diagnostic"), dict) or not failure["diagnostic"]):
                raise EngineError("malformed rejected rollout")
            index = failure.get("candidate_index")
            if type(index) is not int or not 0 <= index < len(candidates) or index in indices:
                raise EngineError("invalid or duplicate candidate index")
            indices.add(index)
            try:
                _words(failure["words"])
            except ValueError as exc:
                raise EngineError("rejected rollout contains an invalid replay word") from exc
            if len(failure["words"]) > horizon:
                raise EngineError("rejected rollout exceeded requested horizon")
        handles = [r["handle"] for r in results]
        if len(set(handles)) != len(handles) or handle in handles:
            raise EngineError("engine reused an immutable checkpoint handle")
        ticks = sum(len(r["words"]) for r in results + rejected)
        if (partial or "simulated_ticks" in reply) and (
            type(reply.get("simulated_ticks")) is not int or reply["simulated_ticks"] != ticks
        ):
            raise EngineError("engine simulated tick accounting disagrees with traces")
        reply["results"] = sorted(results, key=lambda row: row["candidate_index"])
        reply["rejected"] = sorted(rejected, key=lambda row: row["candidate_index"])
        reply["simulated_ticks"] = ticks
        return reply

    def release(self, handles: Iterable[int]) -> None:
        values = [_integer(h, 0xFFFFFFFF, "checkpoint handle") for h in handles]
        if values:
            self._request(f"release {len(values)} " + " ".join(map(str, values)) + "\n")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait()
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
