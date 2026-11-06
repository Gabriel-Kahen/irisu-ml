"""Independent native reset replay, distinct from exploration and live score."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import struct
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def replay_bytes(seed, words, summary):
    if not 0 <= seed <= 0xffffffff or any(type(w) is not int or not 0 <= w < (1 << 21) for w in words):
        raise ValueError("invalid normal-mode replay inputs")
    if len(words) != summary["tick"]:
        raise ValueError("replay length must equal actual executed tick")
    if not summary.get("terminated") or summary.get("truncated"):
        raise ValueError("only naturally terminated runs can be exported as records")
    score = summary["canonical_score"]
    level = summary.get("canonical_level", summary["level"])
    chain = summary.get("canonical_chain", summary["highest_chain"])
    if any(type(x) is not int or not 0 <= x <= 0x7fffffff for x in (score, level, chain)):
        raise ValueError("replay metadata must fit original signed32 format")
    return struct.pack("<I4i", seed, level, score, chain, 0) + bytes(32) + struct.pack(f"<{len(words)}I", *words)


def read_replay(path):
    data = Path(path).read_bytes()
    if len(data) < 56 or (len(data) - 52) % 4:
        raise ValueError("expected nonempty padded v2.03 replay")
    seed, level, score, chain, mode = struct.unpack_from("<I4i", data)
    if mode != 0:
        raise ValueError("only normal-mode replay is supported")
    words = list(struct.unpack_from(f"<{(len(data)-52)//4}I", data, 52))
    if any(w >= 1 << 21 for w in words):
        raise ValueError("invalid replay input word")
    return seed, words, {"canonical_level": level, "canonical_score": score, "canonical_chain": chain}


class NativeVerifier:
    def __init__(self, oracle, library="", timeout=120):
        self.oracle = str(Path(oracle).resolve(strict=True))
        self.library = str(Path(library).resolve(strict=True)) if library else ""
        self.timeout = timeout
        self.identity = {"oracle": self.oracle, "sha256": sha256_file(self.oracle)}
        if self.library:
            self.identity["library_sha256"] = sha256_file(self.library)

    def verify(self, checkpoint):
        summary = checkpoint["summary"]
        data = replay_bytes(checkpoint["seed"], checkpoint["words"], summary)
        if sha256_file(self.oracle) != self.identity["sha256"] or self.library and sha256_file(self.library) != self.identity["library_sha256"]:
            raise RuntimeError("native verification runtime changed")
        environment = dict(os.environ)
        environment.pop("LD_PRELOAD", None)
        if self.library:
            environment["LD_LIBRARY_PATH"] = str(Path(self.library).parent)
        with tempfile.TemporaryDirectory(prefix="irisu-verify-") as temporary:
            replay = Path(temporary) / "candidate.rpy"
            replay.write_bytes(data)
            with (Path(temporary) / "frames.bin").open("w+b") as frames:
                completed = subprocess.run([self.oracle, str(replay)], stdout=frames,
                                           stderr=subprocess.PIPE, env=environment,
                                           timeout=self.timeout, check=False)
                if completed.returncode:
                    reason = f"exit status {completed.returncode}"
                    if completed.returncode < 0:
                        try:
                            name = signal.Signals(-completed.returncode).name
                        except ValueError:
                            name = f"signal {-completed.returncode}"
                        reason = f"{name} (returncode {completed.returncode})"
                    detail = completed.stderr.decode(errors="replace")[-3000:]
                    raise RuntimeError("native replay verification failed: " + reason +
                                       (f": {detail}" if detail else " (empty stderr)"))
                frames.seek(0, 2)
                if frames.tell() != 96 * (len(checkpoint["words"]) + 1):
                    raise RuntimeError("native oracle returned an incomplete frame stream")
            reports = []
            for line in completed.stderr.decode().splitlines():
                try:
                    reports.append(json.loads(line))
                except ValueError:
                    pass
            if not reports or not isinstance(reports[-1], dict):
                raise RuntimeError("native oracle returned no verification receipt")
            report = reports[-1]
        expected = {"tick": summary["tick"], "canonical_score": summary["canonical_score"],
                    "level": summary["level"],
                    "highest_chain": summary["highest_chain"],
                    "steps": len(checkpoint["words"])}
        if report.get("backend") != "native-msvc-cpu-oracle" or any(report.get(k) != v for k, v in expected.items()):
            raise RuntimeError("independent replay disagrees with candidate metadata")
        return {"verified": True, "seed": checkpoint["seed"], **expected,
                "canonical_level": summary.get("canonical_level", summary["level"]),
                "canonical_chain": summary.get("canonical_chain", summary["highest_chain"]),
                "replay_sha256": hashlib.sha256(data).hexdigest(),
                "runtime": self.identity, "oracle_report": report}
