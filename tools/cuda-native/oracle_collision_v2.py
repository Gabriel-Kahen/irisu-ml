#!/usr/bin/env python3
"""Independent pinned native geometry-wire v2 collision corpus and comparator."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
from oracle_collision import PIN, sha, word, value

CASE = struct.Struct('<98I')
HEADER = struct.Struct('<4I')
MANIFOLD = struct.Struct('<15I')


def shape(kind, x=0, y=0, radius=1, vertices=None, angle=0):
    points = vertices or []
    normals = []
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        normals.extend([dy / length, -dx / length])
    packed_vertices = [v for point in points for v in point]
    packed_vertices += [0] * (16 - len(packed_vertices))
    normals += [0] * (16 - len(normals))
    geometry = [x, y, math.cos(angle), math.sin(angle), -math.sin(angle),
                math.cos(angle), radius, *packed_vertices, *normals]
    return [kind, len(points), *(word(v) for v in geometry)]


def generate(count, seed, family):
    rng = random.Random(seed)
    rows, labels = [], []
    square = [(-1, 1), (-1, -1), (1, -1), (1, 1)]

    def add(label, a, b, conservative):
        initial = [0x3f000000 | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
                   for _ in range(15)]
        initial[5], initial[11], initial[14] = (rng.getrandbits(32) for _ in range(3))
        rows.append(CASE.pack(*a, *b, *initial, conservative))
        labels.append(label)

    if family == 'poly-circle':
        for conservative in (0, 1):
            for radius in (0., .25, 1.):
                tangent = word(1 + radius)
                for bits in (tangent - 1, tangent, tangent + 1):
                    for axis in (0, 1):
                        for sign in (-1, 1):
                            pos = [0., 0.]
                            pos[axis] = sign * value(bits)
                            add(f'face-r{radius}-b{bits:08x}-a{axis}-s{sign}-c{conservative}',
                                shape(1, vertices=square), shape(0, *pos, radius), conservative)
                add(f'inside-r{radius}-c{conservative}', shape(1, vertices=square),
                    shape(0, 0, 0, radius), conservative)
            for bits in range(word(math.sqrt(2)) - 2, word(math.sqrt(2)) + 3):
                add(f'corner-r{bits:08x}-c{conservative}', shape(1, vertices=square),
                    shape(0, 2, 2, value(bits)), conservative)
    else:
        for conservative in (0, 1):
            for dx in (0., value(word(2.) - 1), 2., value(word(2.) + 1), 3.):
                for angle in (0., math.pi / 4, math.pi / 2):
                    add(f'boxes-d{dx}-a{angle}-c{conservative}',
                        shape(1, vertices=square), shape(1, dx, 0, vertices=square, angle=angle), conservative)
    for i in range(count):
        scale = 2. ** rng.randrange(-4, 5)
        n = rng.randrange(3, 9)
        points = [(scale * math.cos(2 * math.pi * k / n),
                   scale * math.sin(2 * math.pi * k / n)) for k in range(n)]
        x, y, angle = rng.uniform(-10, 10) * scale, rng.uniform(-10, 10) * scale, rng.uniform(-math.pi, math.pi)
        first = shape(1, x, y, vertices=points, angle=angle)
        if family == 'poly-circle':
            second = shape(0, x + rng.uniform(-3, 3) * scale,
                           y + rng.uniform(-3, 3) * scale, rng.uniform(.01, 2) * scale)
        else:
            second = shape(1, x + rng.uniform(-3, 3) * scale,
                           y + rng.uniform(-3, 3) * scale, vertices=points,
                           angle=rng.uniform(-math.pi, math.pi))
        add(f'random-{i}-n{n}', first, second, i % 2)
    return HEADER.pack(0x32534e49, 2, len(rows), 98) + b''.join(rows), labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--runner', type=Path)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--family', choices=['poly-circle', 'poly-poly'], default='poly-circle')
    p.add_argument('--random', type=int, default=1024)
    p.add_argument('--seed', type=int, default=661003)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=60)
    a = p.parse_args()
    if not 0 <= a.random <= 999000 or sha(a.library.read_bytes()) != PIN:
        p.error('invalid corpus bound or authoritative library pin')
    packet, labels = generate(a.random, a.seed, a.family)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cases = a.output.with_suffix('.cases.bin')
    cases.write_bytes(packet)
    command = [a.oracle, a.library, cases]
    native = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=a.timeout, check=True)
    if len(native.stdout) != len(labels) * 60:
        raise RuntimeError('native v2 output length mismatch')
    warm_sentinels = True
    for i in range(len(labels)):
        initial = CASE.unpack_from(packet, 16 + i * 392)[82:97]
        result = MANIFOLD.unpack_from(native.stdout, i * 60)
        # Narrow phase never owns accumulated impulses. Unused point fields
        # have function/path-specific behavior and are compared in full below.
        warm_sentinels &= all(initial[j] == result[j] for j in [3, 4, 9, 10])
    if not warm_sentinels:
        raise RuntimeError('native collision changed accumulated impulses')
    a.output.with_suffix('.native.bin').write_bytes(native.stdout)
    report = {'schema': 'irisu-native-collision-wire-v2-parity', 'promotion_eligible': False,
              'scope': 'Independent native narrow-phase stage only; complete60Bmanifold and inputshapeimmutability. No world/search/performance admission.',
              'family': a.family, 'cases': len(labels), 'seed': a.seed, 'random_cases': a.random,
              'labels': labels, 'native_command': list(map(str, command)),
              'native_metadata': native.stderr.decode(), 'native_result_sha256': sha(native.stdout),
              'cases_sha256': sha(packet), 'native_dso_sha256': PIN,
              'native_oracle_sha256': sha(a.oracle.read_bytes()),
              'accumulated_impulses_preserved': warm_sentinels, 'typed_parity': None}
    if a.runner:
        command = [a.runner, cases] + (['--gpu'] if a.gpu else [])
        env = dict(os.environ) if a.gpu else None
        if a.gpu:
            env.pop('CUDA_VISIBLE_DEVICES', None)
        typed = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=a.timeout, check=True, env=env)
        report.update(typed_command=list(map(str, command)), typed_metadata=typed.stderr.decode(),
                      typed_result_sha256=sha(typed.stdout), typed_runner_sha256=sha(a.runner.read_bytes()),
                      typed_parity=typed.stdout == native.stdout)
        if typed.stdout != native.stdout:
            limit = min(len(typed.stdout), len(native.stdout))
            offset = next((i for i in range(limit) if typed.stdout[i] != native.stdout[i]), limit)
            report['first_mismatch'] = {'byte': offset, 'case': offset // 60,
                                        'word': offset % 60 // 4,
                                        'label': labels[min(offset // 60, len(labels) - 1)]}
        a.output.with_suffix('.typed.bin').write_bytes(typed.stdout)
    report['native_library_unchanged'] = sha(a.library.read_bytes()) == PIN
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(a.output)
    if report['typed_parity'] is False or not report['native_library_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
