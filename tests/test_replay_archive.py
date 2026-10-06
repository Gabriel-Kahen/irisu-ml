from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from irisu_rl.replay_archive import archive_record_replay, archived_scores


def replay(path: Path, score: int, *, mode: int = 0) -> Path:
    path.write_bytes(struct.pack("<5i", 41, 100, score, 7, mode) + bytes(32))
    return path


def accepted(score: int, path: Path):
    return {
        "status": {"accepted": True},
        "hashes": {"replay_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "outcome": {
            "score": {"clone_final": score, "matches": True},
            "clone": {
                "terminated": True,
                "truncated": False,
                "terminal_metadata_recorded": True,
            },
        },
    }


def test_archives_exact_verified_record_atomically(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 250_243)
    archive = tmp_path / "replays"
    result = archive_record_replay(source, archive, lambda path: accepted(250_243, path))
    assert result["archived"] is True
    assert (archive / "250243.rpy").read_bytes() == source.read_bytes()
    assert archived_scores(archive) == (250_243,)
    assert not list(archive.glob("*.part"))


def test_archive_discovery_ignores_noncanonical_and_malformed_files(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "replays"
    archive.mkdir()
    replay(archive / "100.rpy", 100)
    replay(archive / "not-a-score.rpy", 999_999)
    replay(archive / "200.rpy", 201)
    (archive / "300.rpy").write_bytes(b"malformed")
    assert archived_scores(archive) == (100,)


def test_does_not_verify_non_record(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 250_242)

    def unexpected(_path: Path):
        raise AssertionError("non-record should not spend time in exact verification")

    result = archive_record_replay(
        source, tmp_path / "replays", unexpected, score_floor=250_242
    )
    assert result == {
        "archived": False,
        "reason": "not-a-new-record",
        "score": 250_242,
        "record_before": 250_242,
    }


def test_rejects_unverified_or_mismatched_record(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 300_000)
    report = accepted(299_999, source)
    report["outcome"]["score"]["matches"] = False
    with pytest.raises(ValueError, match="natural terminal"):
        archive_record_replay(source, tmp_path / "replays", lambda _path: report)
    assert not (tmp_path / "replays").exists()


def test_rejects_score_disagreement_even_if_report_claims_match(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 300_000)
    with pytest.raises(ValueError, match="differs from exact score"):
        archive_record_replay(
            source,
            tmp_path / "replays",
            lambda path: accepted(300_001, path),
        )


def test_rejects_replay_changed_after_verification(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 300_000)

    def mutate(path: Path):
        report = accepted(300_000, path)
        path.write_bytes(path.read_bytes() + bytes(4))
        return report

    with pytest.raises(RuntimeError, match="changed during or after"):
        archive_record_replay(source, tmp_path / "replays", mutate)


def test_rejects_non_normal_mode(tmp_path: Path) -> None:
    source = replay(tmp_path / "candidate.rpy", 300_000, mode=1)
    with pytest.raises(ValueError, match="normal mode"):
        archive_record_replay(
            source, tmp_path / "replays", lambda path: accepted(300_000, path)
        )
