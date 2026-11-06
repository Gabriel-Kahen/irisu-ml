"""Stream exact group genealogy without materializing per-tick state in Python."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .verification import sha256_file


def trace_replay(store, config, replay, output=None):
    replay = Path(replay).resolve(strict=True)
    binary = Path(config.oracle).with_name("irisu-search-trace")
    binary_hash = sha256_file(binary)
    replay_hash = sha256_file(replay)
    destination = Path(output) if output else store.root / "diagnostics" / f"groups-{replay_hash[:12]}-{binary_hash[:12]}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    if config.oracle_library:
        environment["LD_LIBRARY_PATH"] = str(Path(config.oracle_library).parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            completed = subprocess.run([str(binary), str(replay)], stdout=stream, stderr=subprocess.PIPE,
                                       timeout=config.engine_timeout, env=environment, check=False)
            if completed.returncode:
                raise RuntimeError("group trace failed: " + completed.stderr.decode(errors="replace")[-3000:])
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            identity = json.loads(stream.readline())
            summary = None
            for line in stream:
                summary = json.loads(line)
            if not summary or summary.get("kind") != "summary" or not summary.get("verified"):
                raise RuntimeError("group trace produced no verified summary")
        # Exclusive publication preserves an existing diagnostic file.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    receipt = {"output": str(destination.resolve()), "binary_sha256": binary_hash,
               "replay_sha256": replay_hash, "trace_sha256": sha256_file(destination),
               "identity": identity, "summary": summary}
    with Path(str(destination) + ".receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    store.event("group_trace", receipt)
    return receipt
