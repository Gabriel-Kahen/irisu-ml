"""Independent replay initialization must precede variable-size input allocation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from irisu_search.config import default_paths
from irisu_search.verification import NativeVerifier, replay_bytes


PATHS = default_paths()
ORACLE = Path(os.environ.get("IRISU_SEARCH_ORACLE", PATHS["oracle"]))
LIBRARY = Path(os.environ.get("IRISU_SEARCH_ORACLE_LIBRARY", PATHS["oracle_library"]))
FIXTURE = Path(__file__).parent / "fixtures/search/oracle-initialization-391769539.json"
pytestmark = pytest.mark.skipif(not ORACLE.is_file() or not LIBRARY.is_file(),
                                reason="build the independently linked native oracle")


def archived_checkpoint():
    fixture = json.loads(FIXTURE.read_text())
    words = [0] * fixture["ticks"]
    for index, word in fixture["nonzero_words"]:
        words[index] = word
    summary = {key: fixture[key] for key in (
        "canonical_score", "canonical_level", "canonical_chain", "level", "highest_chain")}
    summary.update(tick=len(words), terminated=True, truncated=False)
    return fixture, {"seed": fixture["seed"], "words": words, "summary": summary}


def run_oracle(path, maximum_steps=None):
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = str(LIBRARY.parent)
    command = [str(ORACLE), str(path)]
    if maximum_steps is not None:
        command.append(str(maximum_steps))
    return subprocess.run(command, capture_output=True, env=environment, timeout=120)


def test_archived_full_replay_frames_and_independent_verification(tmp_path):
    fixture, checkpoint = archived_checkpoint()
    data = replay_bytes(checkpoint["seed"], checkpoint["words"], checkpoint["summary"])
    assert hashlib.sha256(data).hexdigest() == fixture["replay_sha256"]
    path = tmp_path / "archived.rpy"
    path.write_bytes(data)
    result = run_oracle(path)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert len(result.stdout) == 96 * (fixture["ticks"] + 1)
    assert hashlib.sha256(result.stdout).hexdigest() == fixture["native_frames_sha256"]
    receipt = NativeVerifier(ORACLE, LIBRARY).verify(checkpoint)
    assert receipt["verified"] and receipt["canonical_score"] == 7752
    # A valid full frame stream must still fail if the recorded header is wrong.
    invalid = bytearray(data)
    struct.pack_into("<i", invalid, 8, fixture["canonical_score"] + 1)
    path.write_bytes(invalid)
    result = run_oracle(path)
    assert result.returncode != 0
    assert b"full replay canonical terminal disagrees with header" in result.stderr


@pytest.mark.parametrize("seed", [0, 42, 391769539, 0xFFFFFFFF])
def test_initial_frame_does_not_depend_on_unexecuted_input_length(tmp_path, seed):
    # The old oracle SIGSEGVs even with maximum-steps=0 for the archived
    # 22,508-word buffer. No replay update can explain that startup difference.
    fixture, checkpoint = archived_checkpoint()
    initial = None
    for length in (1, 128, 2048, fixture["ticks"], 97219):
        summary = dict(checkpoint["summary"], tick=length)
        words = (checkpoint["words"] * (length // len(checkpoint["words"]) + 1))[:length]
        path = tmp_path / f"startup-{length}.rpy"
        path.write_bytes(replay_bytes(seed, words, summary))
        result = run_oracle(path, 0)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert len(result.stdout) == 96
        report = json.loads(result.stderr)
        assert report["steps"] == report["tick"] == 0
        if initial is None:
            initial = result.stdout
        assert result.stdout == initial
