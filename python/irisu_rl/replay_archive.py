"""Atomic high-score replay archiving.

The archive itself is the scoreboard: accepted files are named ``<score>.rpy``.
Callers must supply an exact verifier; header metadata alone is never enough to
promote a replay.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path


_HEADER = struct.Struct("<5i")
_PADDING_SIZE = 32
_RECORD_NAME = re.compile(r"([0-9]+)\.rpy\Z")


def replay_header(path: Path) -> dict[str, int]:
    """Read the five signed v2.03 replay header fields."""

    with path.open("rb") as stream:
        data = stream.read(_HEADER.size + _PADDING_SIZE)
        size = os.fstat(stream.fileno()).st_size
    minimum = _HEADER.size + _PADDING_SIZE
    if len(data) != minimum or (size - minimum) % 4:
        raise ValueError("replay is not a complete padded v2.03 trace")
    if data[_HEADER.size:] != bytes(_PADDING_SIZE):
        raise ValueError("replay does not have padded v2.03 layout")
    seed, level, score, chain, mode = _HEADER.unpack_from(data)
    if mode != 0:
        raise ValueError(f"record archive only accepts normal mode, got mode {mode}")
    if score < 0 or level < 0 or chain < 0:
        raise ValueError("replay header has negative outcome metadata")
    return {
        "seed": seed,
        "level": level,
        "score": score,
        "highest_chain": chain,
        "mode": mode,
    }


def archived_scores(archive_dir: Path) -> tuple[int, ...]:
    """Return scores represented by canonical archive filenames."""

    if not archive_dir.exists():
        return ()
    scores = []
    for path in archive_dir.iterdir():
        match = _RECORD_NAME.fullmatch(path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            filename_score = int(match.group(1))
            try:
                header = replay_header(path)
            except (OSError, ValueError):
                continue
            if header["score"] == filename_score:
                scores.append(filename_score)
    return tuple(sorted(scores))


def _accepted_result(report: Mapping[str, object]) -> tuple[int, str]:
    status = report.get("status")
    outcome = report.get("outcome")
    hashes = report.get("hashes")
    if not isinstance(status, Mapping) or status.get("accepted") is not True:
        raise ValueError("exact verifier did not accept the replay")
    if not isinstance(outcome, Mapping):
        raise ValueError("exact verifier report lacks outcome")
    if not isinstance(hashes, Mapping):
        raise ValueError("exact verifier report lacks hashes")
    score = outcome.get("score")
    clone = outcome.get("clone")
    if not isinstance(score, Mapping) or not isinstance(clone, Mapping):
        raise ValueError("exact verifier report has malformed outcome")
    value = score.get("clone_final")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("exact verifier report lacks an integer final score")
    if (
        score.get("matches") is not True
        or clone.get("terminated") is not True
        or clone.get("truncated") is not False
        or clone.get("terminal_metadata_recorded") is not True
    ):
        raise ValueError("exact verifier report is not a natural terminal closure")
    digest = hashes.get("replay_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("exact verifier report lacks a valid replay SHA-256")
    return value, digest


def archive_record_replay(
    replay: Path,
    archive_dir: Path,
    verifier: Callable[[Path], Mapping[str, object]],
    *,
    score_floor: int = 0,
) -> dict[str, object]:
    """Verify and atomically archive ``replay`` if it beats the current record."""

    if replay.is_symlink():
        raise ValueError("replay must not be a symlink")
    source = replay.resolve(strict=True)
    if not source.is_file():
        raise ValueError("replay must be a regular file")
    header = replay_header(source)
    score = header["score"]
    current = max((score_floor, *archived_scores(archive_dir)))
    if score <= current:
        return {
            "archived": False,
            "reason": "not-a-new-record",
            "score": score,
            "record_before": current,
        }

    report = verifier(source)
    verified_score, verified_digest = _accepted_result(report)
    if verified_score != score:
        raise ValueError(
            f"replay header score {score} differs from exact score {verified_score}"
        )
    # From here onward the verified exact outcome, not untrusted header metadata,
    # is authoritative for record comparison and the destination filename.
    score = verified_score

    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != verified_digest:
        raise RuntimeError("replay changed during or after exact verification")
    archive_dir.mkdir(parents=True, exist_ok=True)
    current = max((score_floor, *archived_scores(archive_dir)))
    if score <= current:
        return {
            "archived": False,
            "reason": "record-changed-during-verification",
            "score": score,
            "record_before": current,
            "sha256": digest,
        }

    destination = archive_dir / f"{score}.rpy"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{score}.", suffix=".part", dir=archive_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(archive_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archived": True,
        "reason": "new-record",
        "score": score,
        "record_before": current,
        "destination": str(destination),
        "sha256": digest,
        "seed": header["seed"],
        "level": header["level"],
        "highest_chain": header["highest_chain"],
    }
