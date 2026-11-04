"""Fresh native worlds must not inherit dirty protocol/reply allocator state."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from irisu_search.config import default_paths
from irisu_search.engine import Engine
from irisu_search.verification import NativeVerifier, read_replay


PATHS = default_paths()
NATIVE = Path(os.environ.get("IRISU_SEARCH_ENGINE", PATHS["engine"]))
ORACLE = Path(PATHS["oracle"])
TRACE = ORACLE.with_name("irisu-search-trace")
REPLAY = Path(PATHS["incumbent"])


@pytest.mark.skipif(not NATIVE.is_file(), reason="build native search engine")
def test_reset_after_large_retained_checkpoint_reply_uses_pristine_initialization():
    seed, words, _ = read_replay(REPLAY)
    with Engine([str(NATIVE), "--max-checkpoints", "8"]) as engine:
        initial = engine.load(seed, [])
        middle = engine.load(seed, words[:48_000])
        # The large reply dirties the coordinator's heap. A fresh world must
        # not consume its bytes as the legacy PairManager's initial counters.
        reset = engine.load(seed, [])
        assert reset["summary"] == initial["summary"]
        assert middle["summary"]["tick"] == 48_000
        first = engine.batch(initial["handle"], 8, [{"mode": 0}])["results"][0]
        second = engine.batch(reset["handle"], 8, [{"mode": 0}])["results"][0]
        assert first["summary"] == second["summary"]


@pytest.mark.skipif(not all(p.is_file() for p in (NATIVE, ORACLE, TRACE)),
                    reason="build native engine, oracle, and genealogy trace")
def test_retained_record_import_verification_trace_then_new_seed_batches():
    seed, words, header = read_replay(REPLAY)
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(Path(PATHS["oracle_library"]).parent)
    with Engine([str(NATIVE), "--max-checkpoints", "16"], env=environment) as engine:
        retained = [engine.load(seed, words[:stop]) for stop in (0, 48_000, 96_000, len(words))]
        final = retained[-1]
        assert all(final["summary"][key] == value for key, value in header.items())
        receipt = NativeVerifier(ORACLE, PATHS["oracle_library"]).verify(
            {"seed": seed, "words": words, "summary": final["summary"]})
        assert receipt["canonical_score"] == 4_393_101
        with tempfile.TemporaryFile() as destination:
            trace = subprocess.run([str(TRACE), str(REPLAY)], stdout=destination,
                                   stderr=subprocess.PIPE, env=environment, timeout=60)
            assert trace.returncode == 0, trace.stderr.decode(errors="replace")
        for new_seed in (seed, 42, 0xFFFFFFFF):
            root = engine.load(new_seed, [])
            if new_seed == seed:
                assert root["summary"] == retained[0]["summary"]
            outcomes = engine.batch(root["handle"], 96,
                                    [{"mode": 1, "search_seed": 71}, {"mode": 2, "search_seed": 91}])
            assert outcomes["simulated_ticks"] == 192
            assert all(row["summary"]["seed"] == new_seed for row in outcomes["results"])
            engine.release([root["handle"], *(row["handle"] for row in outcomes["results"])])
        # Imported keepers stayed resident throughout all external processes
        # and fresh-seed batches; the original prefix remains usable.
        tail = engine.batch(retained[2]["handle"], 2000,
                            [{"mode": 0, "words": words[96_000:]}])["results"][0]
        assert tail["summary"]["canonical_score"] == 4_393_101
        assert tail["summary"]["terminated"]
