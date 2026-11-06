"""Archived reentrant level-up must survive native body-vector reallocation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from irisu_search.config import default_paths
from irisu_search.verification import NativeVerifier, replay_bytes


PATHS = default_paths()
ORACLE = Path(os.environ.get("IRISU_SEARCH_ORACLE", PATHS["oracle"]))
LIBRARY = Path(os.environ.get("IRISU_SEARCH_ORACLE_LIBRARY", PATHS["oracle_library"]))
FIXTURE = Path(__file__).parent / "fixtures/search/native-participant-lifetime-3943786419.json"


@pytest.mark.skipif(not ORACLE.is_file() or not LIBRARY.is_file(),
                    reason="build the independently linked native oracle")
def test_archived_level_up_reallocation_matches_typed_frames_and_verifies(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    words = [0] * fixture["ticks"]
    for index, word in fixture["nonzero_words"]:
        words[index] = word
    checkpoint = {"seed": fixture["seed"], "words": words, "summary": fixture["summary"]}
    data = replay_bytes(checkpoint["seed"], words, checkpoint["summary"])
    assert hashlib.sha256(data).hexdigest() == fixture["replay_sha256"]
    replay = tmp_path / "candidate.rpy"
    replay.write_bytes(data)
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = str(LIBRARY.parent)
    result = subprocess.run([str(ORACLE), str(replay)], capture_output=True,
                            env=environment, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert len(result.stdout) == 96 * (len(words) + 1)
    assert hashlib.sha256(result.stdout).hexdigest() == fixture["frames_sha256"]
    assert NativeVerifier(ORACLE, LIBRARY).verify(checkpoint)["canonical_score"] == 252
