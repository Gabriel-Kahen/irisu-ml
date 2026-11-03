#!/usr/bin/env python3
"""Recover the published CPU oracle without substituting its physics identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request


TAG = "web-exact-runtime-virtio-v1-20260824"
ASSET = "irisu-exact-runtime-virtio-v1-20260824.tar.gz"
URL = f"https://github.com/Gabriel-Kahen/irisu-ml/releases/download/{TAG}/{ASSET}"
ARCHIVE_SHA256 = "0340660c5e5ccd1bbc47bd75eb60ec96349a0d046fa9cf1e5923ff78ee77838f"
HOST_SHA256 = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"
WORKER_SHA256 = "8ef81521e81a5b2a764c305ac48dad997b28476bcd2fccbd1c9aed9603322854"
FILES = (
    "guest/libirisu_box2d_msvc_exact_multiworld.so",
    "guest/irisu-exact-worker",
    "LICENSE.Box2D",
    "PROVENANCE.md",
    "SOURCE_OFFER.md",
    "SHA256SUMS",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recover(archive: Path, output: Path) -> dict[str, object]:
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("release archive SHA-256 mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Extract only named regular files. Never materialize links or archive paths.
    with tempfile.TemporaryDirectory(prefix=".exact-recovery-", dir=output.parent) as tmp:
        stage = Path(tmp) / "runtime"
        stage.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            for name in FILES:
                member = bundle.getmember("exact-runtime/" + name)
                if not member.isfile() or member.size > 1 << 20:
                    raise ValueError(f"unexpected release member: {name}")
                source = bundle.extractfile(member)
                assert source is not None
                destination = stage / name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(source.read())
                destination.chmod(0o755 if name.startswith("guest/") else 0o644)
        host = stage / FILES[0]
        worker = stage / FILES[1]
        if sha256(host) != HOST_SHA256 or sha256(worker) != WORKER_SHA256:
            raise ValueError("recovered executable identity mismatch")
        metadata: dict[str, object] = {
            "version": "recovered-exact-cpu-oracle-v1",
            "source_url": URL,
            "archive_sha256": ARCHIVE_SHA256,
            "exact_library_sha256": HOST_SHA256,
            "release_worker_sha256": WORKER_SHA256,
            "historical_cpu_worker_sha256": "4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261",
            "changes_historical_training_identity": False,
            "note": "The browser release relinked the worker. The exact MSVC physics host is byte-identical to the CPU production pin. Rebuilt workers need explicit independent validation and provenance.",
        }
        (stage / "recovery.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # A single rename makes only a complete, checked recovery visible.
        os.rename(stage, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="use an already downloaded release archive")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="irisu-exact-download-") as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / ASSET
            request = urllib.request.Request(URL, headers={"User-Agent": "irisu-exact-runtime-recovery"})
            with urllib.request.urlopen(request, timeout=60) as source, archive.open("wb") as target:
                total = 0
                while chunk := source.read(1 << 20):
                    total += len(chunk)
                    if total > 16 << 20:
                        raise ValueError("release archive exceeds bounded expected size")
                    target.write(chunk)
        print(json.dumps(recover(archive, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
