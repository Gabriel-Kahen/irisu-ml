#!/usr/bin/env python3
"""Package and verify an isolated Metsu exact worker for the pinned v86 guest."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HOST_HASH = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"
NORMAL_CONFIG_HASH = 17009678407634462320
BACKEND = "exact-msvc9-r58-metsu-forward"
BASELINE_HASHES = {
    "libv86.js": "730d3e4e4bd1d9c7ede4a4995f23da87261a723b3524da9a75c90e6e0610f46b",
    "v86.wasm": "73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403",
    "buildroot-bzimage68.bin": "c745fcde490cb8e4db59e5b4e1ce5e00bfdae3bcb6bf34048d16dcb15e3dfe30",
    "seabios.bin": "73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98",
    "vgabios.bin": "a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880",
    "guest/libirisu_box2d_msvc_exact_multiworld.so": HOST_HASH,
    "guest/libstdc++.so.6": "90abf0c6a98625b812132ea9af960ee8b036e3f519e368de1084fb50532ba021",
}
DOWNLOADS = {
    "libc6_2.43-7_i386.deb": (
        "https://deb.debian.org/debian/pool/main/g/glibc/libc6_2.43-7_i386.deb",
        "e18975c43109a3c6b2fc92c010ce7508d2379b60c1b312f778b32f69fb62f1a4"),
    "libgcc-s1_14.2.0-19_i386.deb": (
        "https://deb.debian.org/debian/pool/main/g/gcc-14/libgcc-s1_14.2.0-19_i386.deb",
        "a4c71fd856d2a48a7505a087b4186e3cca23f94603c05e3fb7c799b27e72f761"),
    "LGPL-2.1.txt": (
        "https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt",
        "20e50fe7aae3e56378ebf0417d9de904f55a0e61e4df315333e632a4d3555d95"),
}
GUEST_FILES = ["irisu-exact-worker", "irisu-metsu-worker.bin",
               "libirisu_box2d_msvc_exact_multiworld.so", "libstdc++.so.6",
               "ld-linux.so.2", "libc.so.6", "libm.so.6", "libgcc_s.so.1"]
RUNTIME_ASSETS = ["libv86.js", "v86.wasm", "buildroot-bzimage68.bin", "seabios.bin", "vgabios.bin"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch or missing input: {path}")


def stamp_identity(identity: dict, runtime: Path) -> None:
    """Bind the cache key to every byte the browser fetches, including launchers."""
    identity["guest_file_sha256"] = {name: digest(runtime / "guest" / name)
                                     for name in identity["guest_files"]}
    identity["runtime_asset_sha256"] = {name: digest(runtime / name) for name in RUNTIME_ASSETS}
    versioned = {key: value for key, value in identity.items() if key not in {"prepared", "runtime_version"}}
    identity["runtime_version"] = hashlib.sha256(json.dumps(versioned, sort_keys=True).encode()).hexdigest()[:24]


def write_checksums(runtime: Path) -> None:
    files = sorted(path for path in runtime.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (runtime / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(runtime)}\n" for path in files))


def framed(opcode: int, request: int) -> bytes:
    return struct.pack("<IHHII", 0x43505249, 1, opcode, request, 0)


def worker_identity(worker: Path) -> dict:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LD_") and key != "GLIBC_TUNABLES"}
    environment["IRISU_EXACT_CW"] = "0x027f"
    result = subprocess.run([str(worker)], cwd="/", env=environment,
                            input=framed(1, 1) + framed(5, 2), capture_output=True,
                            timeout=20, check=True)
    data = result.stdout
    if len(data) < 52 or struct.unpack_from("<IHHI", data) != (0x43505249, 1, 1, 1):
        raise ValueError("worker emitted an invalid Hello response")
    size = struct.unpack_from("<I", data, 12)[0]
    if size < 36 or len(data) < 16 + size or struct.unpack_from("<i", data, 16)[0]:
        raise ValueError("worker rejected Hello")
    payload = data[20:16 + size]
    protocol, bits, capacity, _, config, control, processes = struct.unpack_from("<IIIIQII", payload)
    strings = []
    offset = 32
    for _ in range(3):
        length = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        strings.append(payload[offset:offset + length].decode("utf8"))
        offset += length
    if (offset != len(payload) or (protocol, bits, capacity, control, processes) != (1, 32, 196, 0x027f, 1)
            or strings[0] != BACKEND or strings[2] != HOST_HASH
            or config in (0, NORMAL_CONFIG_HASH)):
        raise ValueError("worker is not the required mode-1 Metsu exact backend")
    return {"mode": 1, "backend": strings[0], "config_hash": str(config),
            "compiler": strings[1], "exact_library_sha256": strings[2]}


def fetch(cache: Path, name: str) -> Path:
    url, expected = DOWNLOADS[name]
    target = cache / name
    if not target.is_file() or digest(target) != expected:
        temporary = target.with_suffix(target.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=60) as response:
            temporary.write_bytes(response.read())
        verify(temporary, expected)
        temporary.replace(target)
    verify(target, expected)
    return target


def extract_deb(package: Path, names: dict[str, Path]) -> None:
    members = subprocess.check_output(["ar", "t", str(package)], text=True).splitlines()
    member = next(name for name in members if name.startswith("data.tar."))
    data = subprocess.check_output(["ar", "p", str(package), member])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for source, target in names.items():
            entry = archive.getmember(source)
            if not entry.isfile():
                raise ValueError(f"package input is not a regular file: {source}")
            target.write_bytes(archive.extractfile(entry).read())


def verify_elf(path: Path) -> None:
    header = path.read_bytes()[:20]
    if header[:5] != b"\x7fELF\x01" or header[5] != 1 or struct.unpack_from("<H", header, 18)[0] != 3:
        raise ValueError(f"not a little-endian i386 ELF: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, default=os.environ.get("IRISU_METSU_BASE_RUNTIME"))
    parser.add_argument("--worker", type=Path, default=os.environ.get(
        "IRISU_METSU_WORKER", str(ROOT / "build-metsu-exact/irisu-metsu-exact-worker")))
    parser.add_argument("--cache", type=Path, default=ROOT / ".tmp/metsu-runtime-deps")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", default="0,41,123")
    parser.add_argument("--policy", choices=["scripted", "noinput"], default="scripted")
    args = parser.parse_args()
    if not args.baseline:
        parser.error("provide --baseline or IRISU_METSU_BASE_RUNTIME (the pinned normal runtime)")
    output, baseline, worker = args.output.resolve(), args.baseline.resolve(), args.worker.resolve()
    protected = [ROOT, ROOT / "apps/web/static", ROOT / "apps/web/metsu-static", baseline,
                 args.cache.resolve(), worker]
    if output == Path("/") or any(output == item or output in item.parents or item in output.parents
                                  for item in protected if item != ROOT):
        raise ValueError("refusing unsafe or overlapping Metsu runtime path")
    if output == ROOT or output in ROOT.parents:
        raise ValueError("refusing to replace the repository")
    if output.exists() and not (output / ".irisu-metsu-runtime").is_file():
        raise ValueError("refusing to replace an unmarked Metsu runtime directory")
    verify_elf(worker)
    original_worker_hash = digest(worker)
    identity = worker_identity(worker)  # Reject normal workers before downloads.
    for name, expected in BASELINE_HASHES.items():
        verify(baseline / name, expected)
    args.cache.mkdir(parents=True, exist_ok=True)
    dependencies = {name: fetch(args.cache, name) for name in DOWNLOADS}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        shutil.copytree(baseline, stage, dirs_exist_ok=True)
        (stage / ".irisu-exact-runtime").unlink(missing_ok=True)
        for name in ["PROVENANCE.md", "SHA256SUMS"]:
            (stage / name).rename(stage / ("BASELINE_" + name))
        guest = stage / "guest"
        extract_deb(dependencies["libc6_2.43-7_i386.deb"], {
            **{f"./usr/lib/i386-linux-gnu/{name}": guest / name
               for name in ["ld-linux.so.2", "libc.so.6", "libm.so.6"]},
            "./usr/share/doc/libc6/copyright": stage / "COPYRIGHT.glibc"})
        extract_deb(dependencies["libgcc-s1_14.2.0-19_i386.deb"], {
            "./usr/lib/i386-linux-gnu/libgcc_s.so.1": guest / "libgcc_s.so.1"})
        shutil.copyfile(dependencies["LGPL-2.1.txt"], stage / "LICENSE.LGPL-2.1")
        binary = guest / "irisu-metsu-worker.bin"
        subprocess.run(["objcopy", "--remove-section", ".note.gnu.property", str(worker), str(binary)], check=True)
        # GCC's host policy note advertises x86-64-v3 even for -m32 objects;
        # retain the actual i386 code and verify it in the emulated CPU below.
        launcher = guest / "irisu-exact-worker"
        launcher.write_text("#!/bin/sh\nchmod +x /mnt/ld-linux.so.2 /mnt/irisu-metsu-worker.bin\n"
                            "exec /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-metsu-worker.bin\n")
        for name in GUEST_FILES[1:]:
            verify_elf(guest / name)
        allowed = {"ld-linux.so.2", "libc.so.6", "libm.so.6", "libgcc_s.so.1",
                   "libstdc++.so.6", "libirisu_box2d_msvc_exact_multiworld.so"}
        for name in GUEST_FILES[1:]:
            dynamic = subprocess.check_output(["readelf", "-dW", str(guest / name)], text=True)
            needed = set(re.findall(r"Shared library: \[([^\]]+)\]", dynamic))
            if not needed <= allowed:
                raise ValueError(f"unpackaged dynamic dependencies in {name}: {needed - allowed}")
        identity.update(prepared=False, guest_files=GUEST_FILES, source_worker_sha256=original_worker_hash,
                        packaged_worker_sha256=digest(binary),
                        dependencies={name: {"url": url, "sha256": sha} for name, (url, sha) in DOWNLOADS.items()},
                        source_routes=["https://deb.debian.org/debian/pool/main/g/glibc/glibc_2.43-7.dsc",
                                       "https://deb.debian.org/debian/pool/main/g/gcc-14/gcc-14_14.2.0-19.dsc"])
        stamp_identity(identity, stage)
        manifest = stage / "metsu-runtime.json"
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
        subprocess.run(["node", str(ROOT / "apps/web/verify-metsu-v86.mjs"), "--runtime", str(stage),
                        "--worker", str(worker), "--steps", str(args.steps), "--seed", args.seeds,
                        "--policy", args.policy,
                        "--output", str(stage / "native-v86-parity.json")], check=True)
        if digest(worker) != original_worker_hash:
            raise ValueError("worker changed during packaging; rebuild the runtime from the final worker")
        identity["prepared"] = True
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
        (stage / "PROVENANCE.md").write_text(
            "# Isolated Metsu exact browser runtime\n\n"
            "Mode 1 uses a separate gameplay worker and runtime directory. The pinned v86, kernel, "
            "BIOS, MSVC physics host, and libstdc++ are preserved from the normal release. "
            "BASELINE_PROVENANCE.md and SOURCE_OFFER.md retain their provenance.\n\n"
            "The launcher explicitly runs the packaged Debian i386 loader with --library-path /mnt; "
            "it does not depend on the older guest's libc ABI. The only executable transformation "
            "removes .note.gnu.property. native-v86-parity.json compares actual response bytes "
            "before this runtime is marked prepared. This is browser/native equivalence evidence, "
            "not a claim of original-game Metsu fidelity.\n\n"
            "metsu-runtime.json records worker and dependency hashes, binary download URLs, and "
            "exact Debian source-package routes. COPYRIGHT.glibc and LICENSE.LGPL-2.1 cover libc, "
            "libm, and the loader. COPYRIGHT.gcc-runtime and SOURCE_OFFER.md cover the GCC runtimes. "
            "The .dsc source routes identify the corresponding original source and Debian patches.\n")
        (stage / ".irisu-metsu-runtime").write_text("verified isolated Metsu mode-1 runtime\n")
        write_checksums(stage)
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"built verified Metsu mode-1 runtime at {output}")


if __name__ == "__main__":
    main()
