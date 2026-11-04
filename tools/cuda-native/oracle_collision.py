#!/usr/bin/env python3
"""Generate and compare native circle-stage packets; no world/search admission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess

PIN = 'ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'
HEADER = struct.Struct('<4I')
CASE = struct.Struct('<22I')
MANIFOLD = struct.Struct('<15I')
MAGIC = 0x31434e49


def word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def generate(count, seed):
    rng = random.Random(seed)
    rows, labels = [], []

    def add(label, x1, y1, r1, x2, y2, r2, conservative=0):
        sentinel = [0x3f000000 | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
                    for _ in range(15)]
        sentinel[5], sentinel[11], sentinel[14] = (rng.getrandbits(32) for _ in range(3))
        rows.append(CASE.pack(*(word(x) for x in (x1, y1, r1, x2, y2, r2)),
                              conservative, *sentinel))
        labels.append(label)

    for conservative in (0, 1):
        for distance in (0., value(1), .5, value(word(2.) - 1), 2.,
                         value(word(2.) + 1), 3.):
            for sign in (-1, 1):
                add(f'axial-{distance.hex()}-c{conservative}-s{sign}',
                    0, 0, 1, sign * distance, 0, 1, conservative)
                add(f'vertical-{distance.hex()}-c{conservative}-s{sign}',
                    0, 0, 1, 0, sign * distance, 1, conservative)
        for radius in (0., -0., value(1), value(0x007fffff), value(0x00800000), 1.):
            add(f'coincident-r{word(radius):08x}-c{conservative}',
                -0., 0., radius, 0., -0., radius, conservative)
        threshold = word(math.sqrt(2. ** -23))
        for bits in range(threshold - 3, threshold + 4):
            add(f'epsilon-distance-{bits:08x}-c{conservative}',
                0, 0, 1, value(bits), 0, 1, conservative)
        add(f'large-origin-small-radius-c{conservative}',
            2. ** 20, -(2. ** 20), 2. ** -16,
            2. ** 20, -(2. ** 20), 2. ** -16, conservative)
    for i in range(count):
        scale = 2. ** rng.randrange(-12, 13)
        x, y = (rng.uniform(-64, 64) * scale for _ in range(2))
        r1, r2 = (rng.uniform(.01, 4) * scale for _ in range(2))
        angle, distance = rng.uniform(-math.pi, math.pi), (r1 + r2) * rng.uniform(0, 2)
        add(f'random-{i}', x, y, r1, x + math.cos(angle) * distance,
            y + math.sin(angle) * distance, r2, int(i % 4 == 0))
    return HEADER.pack(MAGIC, 1, len(rows), 22) + b''.join(rows), labels


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(command, timeout, gpu=False):
    env = dict(os.environ) if gpu else None
    if gpu:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=True, env=env)
    return result.stdout, result.stderr.decode(errors='replace')


def validate_sentinels(packet, output):
    _, _, count, _ = HEADER.unpack_from(packet)
    if len(output) != count * MANIFOLD.size:
        raise RuntimeError('wrong native manifold output length')
    counts = [0, 0]
    for i in range(count):
        source = CASE.unpack_from(packet, HEADER.size + i * CASE.size)[7:]
        actual = MANIFOLD.unpack_from(output, i * MANIFOLD.size)
        if actual[14] not in (0, 1):
            raise RuntimeError(f'invalid point count at case {i}')
        counts[actual[14]] += 1
        written = {0, 1, 2, 5, 12, 13, 14} if actual[14] else {14}
        if any(source[j] != actual[j] for j in range(15) if j not in written):
            raise RuntimeError(f'unused/warm-impulse sentinel changed at case {i}')
    return {'disjoint': counts[0], 'contact': counts[1],
            'unused_second_points_and_warm_impulses_preserved': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', required=True, type=Path)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--random', type=int, default=4096)
    p.add_argument('--seed', type=int, default=661003)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--timeout', type=float, default=60)
    a = p.parse_args()
    if not 0 <= a.random <= 999000 or a.timeout <= 0:
        p.error('invalid corpus/timeout bounds')
    if sha(a.library.read_bytes()) != PIN:
        p.error('native DSO does not match authoritative pin')
    packet, labels = generate(a.random, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cases = a.output.with_suffix('.cases.bin')
    cases.write_bytes(packet)
    native_command = [a.oracle, a.library, cases]
    expected, native_metadata = run(native_command, a.timeout)
    a.output.with_suffix('.native.bin').write_bytes(expected)
    sentinel_check = validate_sentinels(packet, expected)
    report = {'schema': 'irisu-native-circle-stage-parity-v1', 'promotion_eligible': False,
              'scope': 'Hidden pinned native circle narrow-phase stage only; full 60-byte manifold equality including untouched warm impulses/second point. No full world or search admission or performance claim.',
              'corpus': {'generator': 'circle-float-boundaries-v1', 'random_cases': a.random,
                         'seed': a.seed, 'cases': len(labels), 'labels': labels},
              'native_command': list(map(str, native_command)), 'native_metadata': native_metadata,
              'inputs_sha256': {'native_library': PIN, 'native_oracle': sha(a.oracle.read_bytes()),
                                'cases': sha(packet)}, 'native_result_sha256': sha(expected),
              'sentinel_check': sentinel_check, 'typed_parity': None}
    if a.runner:
        command = [a.runner, cases] + (['--gpu'] if a.gpu else [])
        actual, metadata = run(command, a.timeout, gpu=a.gpu)
        report['typed_command'] = list(map(str, command))
        report['typed_metadata'] = metadata
        report['typed_result_sha256'] = sha(actual)
        report['inputs_sha256']['typed_runner'] = sha(a.runner.read_bytes())
        report['typed_parity'] = actual == expected
        if actual != expected:
            limit = min(len(actual), len(expected))
            at = next((i for i in range(limit) if actual[i] != expected[i]), limit)
            report['first_mismatch'] = {'byte': at, 'case': at // MANIFOLD.size,
                                        'word': (at % MANIFOLD.size) // 4,
                                        'label': labels[min(at // MANIFOLD.size, len(labels) - 1)]}
        a.output.with_suffix('.typed.bin').write_bytes(actual)
    report['native_dso_unchanged'] = sha(a.library.read_bytes()) == PIN
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(a.output)
    if report['typed_parity'] is False or not report['native_dso_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
