#!/usr/bin/env python3
"""Compare native and typed game-digest-v1 frames; no search/performance admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

LIBRARY_SHA = 'ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'
RECORD_SHA = '81fe723b6b825e62afb471d7517355fef2b8bf0fb1dbe636069c97c70ce2ca79'
FIELDS = ('tick', 'body_hash', 'observation_hash', 'event_hash', 'event_count',
          'transition_hash', 'body_count', 'score', 'gauge', 'level',
          'highest_chain', 'latest_final_score')
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_snapshot():
    files = {Path(__file__).resolve(), ROOT / 'CMakeLists.txt',
             ROOT / 'tools/cuda-exact/game_digest.hpp',
             ROOT / 'tools/cuda-exact/game_oracle.cpp',
             ROOT / 'tools/exact-physics-prototype/physics_wrapper_forward.cpp'}
    for directory in ('clone/cuda_native', 'clone/cuda', 'clone/include', 'clone/core'):
        files.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.hpp', '.cuh', '.cpp', '.cu', '.inc', '.txt'))
    files.update((ROOT / 'tools/cuda-native').glob('*.hpp'))
    files.add(ROOT / 'tools/cuda-native/game_runner.cu')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files) if p.is_file()}


def metadata(stderr):
    rows = []
    for line in stderr.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            pass
    return rows


def execute(command, output, env, timeout):
    start = time.monotonic()
    with output.open('wb') as stream:
        try:
            result = subprocess.run(list(map(str, command)), stdout=stream,
                                    stderr=subprocess.PIPE, env=env, timeout=timeout)
            stderr = result.stderr.decode(errors='replace')
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired as error:
            stderr = (error.stderr or b'').decode(errors='replace')
            code, timed_out = None, True
    return {'command': list(map(str, command)), 'returncode': code,
            'timed_out': timed_out, 'wall_seconds': time.monotonic() - start,
            'stderr': stderr, 'metadata': metadata(stderr),
            'output': str(output), 'output_bytes': output.stat().st_size,
            'output_sha256': digest(output)}


def first_difference(expected, actual):
    limit = min(len(expected), len(actual))
    at = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    frame, within = divmod(at, 96)
    word = within // 8
    start = frame * 96 + word * 8
    def bits(data):
        return f'{struct.unpack_from("<Q", data, start)[0]:016x}' if start + 8 <= len(data) else None
    return {'byte': at, 'frame': frame, 'word64': word, 'word32': within // 4,
            'field': FIELDS[word], 'native_u64_hex': bits(expected),
            'typed_u64_hex': bits(actual), 'length_difference': len(actual) - len(expected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-oracle', 'runner', 'library', 'replay', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--steps', type=int)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--clone-at', type=int)
    p.add_argument('--timeout', type=float, default=900)
    a = p.parse_args()
    for key in ('native_oracle', 'runner', 'library', 'replay'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    a.output = a.output.resolve()
    if a.timeout <= 0 or a.clone_at is not None and a.gpu:
        p.error('timeout must be positive; clone-at is host only')
    if digest(a.library) != LIBRARY_SHA:
        p.error('native DSO does not match the authoritative SHA-256 pin')
    replay = a.replay.read_bytes()
    if len(replay) <= 52 or (len(replay) - 52) % 4 or struct.unpack_from('<I', replay, 16)[0]:
        p.error('expected nonempty padded normal-mode replay')
    seed, level, score, chain = struct.unpack_from('<4I', replay)
    available = (len(replay) - 52) // 4
    steps = available if a.steps is None else a.steps
    if not 0 <= steps <= available or a.clone_at is not None and not 0 <= a.clone_at < steps:
        p.error('steps must fit replay and clone-at must precede steps')
    replay_sha = digest(a.replay)
    record_claim = seed == 1298144938 and score == 4393101 and level == 100
    if record_claim and replay_sha != RECORD_SHA:
        p.error('canonical record header requires the unchanged original replay SHA-256')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    native_env = dict(os.environ)
    native_env.pop('LD_PRELOAD', None)
    native_env['LD_LIBRARY_PATH'] = str(a.library.parent) + (
        os.pathsep + native_env['LD_LIBRARY_PATH'] if native_env.get('LD_LIBRARY_PATH') else '')
    linkage = subprocess.run(['ldd', str(a.native_oracle)], env=native_env,
                             capture_output=True, text=True, timeout=30, check=True)
    mapped = []
    for line in linkage.stdout.splitlines():
        if '=>' in line and a.library.name in line.split('=>', 1)[0]:
            target = Path(line.split('=>', 1)[1].strip().split()[0]).resolve(strict=True)
            mapped.append({'path': str(target), 'sha256': digest(target)})
    if len(mapped) != 1 or mapped[0]['sha256'] != LIBRARY_SHA:
        p.error('native oracle does not resolve exactly one pinned native DSO under the child environment')
    inputs = {name: {'path': str(getattr(a, name)), 'sha256': digest(getattr(a, name))}
              for name in ('native_oracle', 'runner', 'library', 'replay')}
    sources = source_snapshot()
    report = {'schema': 'irisu-cuda-native-game-qualification-v1', 'passed': False,
              'promotion_eligible': False,
              'scope': 'Exact complete game-digest-v1 wire equality for this replay prefix, including initial frame, body/observation/event/transition digests. Digest equality is not raw hidden-state equality. No search, throughput, arbitrary-input, or production admission.',
              'source_attestation_scope': 'Source hashes observed at invocation; not a receipt proving these binaries were built from this source snapshot.',
              'backend': 'cuda-native-full-game' if a.gpu else 'typed-host-full-game',
              'inputs': inputs, 'source_sha256': sources,
              'replay': {'seed': seed, 'header_level': level, 'header_score': score,
                         'header_chain': chain, 'available_steps': available, 'steps': steps,
                         'canonical_record_header': record_claim,
                         'original_record_sha256_verified': replay_sha == RECORD_SHA},
              'wire': {'format': 'game-digest-v1', 'row_bytes': 96,
                       'words': list(FIELDS), 'expected_rows': steps + 1},
              'clone_at': a.clone_at,
              'clone_scope': 'Same-input continuation/rebind only, not divergent-branch isolation' if a.clone_at is not None else None,
              'native_linkage': {'command': ['ldd', str(a.native_oracle)],
                                 'stdout': linkage.stdout, 'stderr': linkage.stderr,
                                 'pinned_mapping': mapped, 'LD_LIBRARY_PATH': native_env['LD_LIBRARY_PATH'],
                                 'LD_PRELOAD_removed': 'LD_PRELOAD' in os.environ},
              'gpu_child_cuda_visible_devices_removed': a.gpu}
    native_command = [a.native_oracle, a.replay, str(steps)]
    typed_command = [a.runner, a.replay, str(steps)]
    if a.gpu:
        typed_command.append('--gpu')
    if a.clone_at is not None:
        typed_command.extend(['--clone-at', str(a.clone_at)])
    typed_env = dict(os.environ)
    if a.gpu:
        typed_env.pop('CUDA_VISIBLE_DEVICES', None)
    report['native'] = execute(native_command, a.output.with_suffix('.native.frames.bin'), native_env, a.timeout)
    report['typed'] = execute(typed_command, a.output.with_suffix('.typed.frames.bin'), typed_env, a.timeout)
    expected = Path(report['native']['output']).read_bytes()
    actual = Path(report['typed']['output']).read_bytes()
    report['exact_frame_bytes_equal'] = actual == expected
    report['output_lengths_valid'] = len(expected) == len(actual) == (steps + 1) * 96
    if actual != expected:
        report['first_difference'] = first_difference(expected, actual)
    report['inputs_unchanged'] = all(digest(Path(value['path'])) == value['sha256'] for value in inputs.values())
    report['sources_unchanged'] = source_snapshot() == sources
    typed_metadata = report['typed']['metadata']
    report['typed_status_valid'] = bool(typed_metadata) and typed_metadata[-1].get('failed') is False
    report['clone_verified'] = a.clone_at is None or bool(typed_metadata) and typed_metadata[-1].get('clone_verified') is True
    report['verification_scope'] = ('native_reset_full_replay_with_header_terminal_check'
                                    if steps == available else 'native_reset_bounded_replay_prefix')
    report['passed'] = all((report['exact_frame_bytes_equal'], report['output_lengths_valid'],
                            report['inputs_unchanged'], report['sources_unchanged'],
                            report['typed_status_valid'], report['clone_verified'],
                            report['native']['returncode'] == 0, report['typed']['returncode'] == 0))
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(a.output)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
