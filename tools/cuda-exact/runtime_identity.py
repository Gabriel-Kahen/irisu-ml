#!/usr/bin/env python3
"""Record and verify experimental CUDA inputs without impersonating CPU lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HOST_SHA = 'ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'
SCHEMA = 'irisu-cuda-experimental-identity-v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check_generated(generated: Path, host: Path) -> None:
    manifest = json.loads((generated / 'manifest.json').read_text())
    if sha256(host) != HOST_SHA or manifest['source_sha256'] != HOST_SHA:
        raise ValueError('physics image does not derive from the pinned MSVC host')
    if manifest.get('image_sha256') != sha256(generated / 'guest_image.bin'):
        raise ValueError('generated image bytes do not match translation manifest')
    if manifest.get('unsupported'):
        raise ValueError('translation contains unsupported instructions')
    check_game()


def check_game() -> None:
    game = json.loads((ROOT / 'clone/cuda/game_source.json').read_text())
    for name, expected in game['inputs'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'generated game input changed: {name}')
    for name, expected in game['outputs'].items():
        if sha256(ROOT / 'clone/cuda' / name) != expected:
            raise ValueError(f'generated game output changed: {name}')


def create(binary: Path, generated: Path, host: Path, build_command: str) -> dict:
    check_generated(generated, host)
    paths = {binary.resolve(), host.resolve(), ROOT / 'CMakeLists.txt',
             ROOT / 'tools/exact-physics-prototype/physics_wrapper_forward.cpp'}
    for directory in ('clone/cuda', 'clone/core', 'clone/include/irisu', 'tools/cuda-exact'):
        paths.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in {'.hpp', '.cuh', '.cu', '.cpp', '.c', '.py', '.json', '.txt', '.inc'})
    paths.update(p for p in generated.iterdir() if p.is_file())
    capture = lambda command: subprocess.check_output(command, cwd=ROOT, text=True).strip()
    return {
        'schema': SCHEMA,
        'backend': 'cuda-translated-msvc-experimental',
        'production_exact_training_admitted': False,
        'parity_claim': 'none; attach separate workload-specific differential evidence',
        'git_head': capture(['git', 'rev-parse', 'HEAD']),
        'git_status': capture(['git', 'status', '--porcelain']),
        'binary': named(binary), 'generated': named(generated), 'host': named(host),
        'build_command': build_command,
        'files_sha256': {named(p): sha256(p) for p in sorted(paths)},
        'device': capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,compute_cap',
                           '--format=csv,noheader,nounits']),
    }


def verify(receipt: dict) -> None:
    if receipt.get('schema') != SCHEMA or receipt.get('production_exact_training_admitted') is not False:
        raise ValueError('unsupported identity or attempted production promotion')
    paths = receipt['files_sha256']
    for name in (receipt['binary'], receipt['host']):
        if name not in paths:
            raise ValueError(f'missing runtime identity: {name}')
    for name, expected in paths.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'identity changed: {name}')
    check_generated(ROOT / receipt['generated'], ROOT / receipt['host'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    write = sub.add_parser('create')
    write.add_argument('--binary', type=Path, required=True)
    write.add_argument('--generated', type=Path, required=True)
    write.add_argument('--host', type=Path, required=True)
    write.add_argument('--build-command', required=True)
    write.add_argument('--output', type=Path, required=True)
    sub.add_parser('check-game')
    read = sub.add_parser('verify')
    read.add_argument('receipt', type=Path)
    args = parser.parse_args()
    if args.action == 'create':
        receipt = create(args.binary, args.generated, args.host, args.build_command)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        verify(receipt)
        print(f'Experimental identity saved: {args.output}')
    elif args.action == 'check-game':
        check_game()
        print('Generated game source hashes verified')
    else:
        verify(json.loads(args.receipt.read_text()))
        print('Experimental identity verified; production admission remains false')


if __name__ == '__main__':
    main()
