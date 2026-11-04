#!/usr/bin/env python3
"""Native SAP history oracle: callbacks, query order, and full normalized state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
from oracle_collision import PIN, sha

OP = struct.Struct('<3I4fI')
RESULT = struct.Struct('<5I')
STATE_BYTES = 0x1743c


def generate(count, seed, null_callback=False):
    rng = random.Random(seed)
    operations, labels = [], []
    active = {}

    def add(kind, slot=0, user=0, box=(0, 0, 1, 1), label=''):
        operations.append(OP.pack(kind, slot, user, *box, 0))
        labels.append(label or f'op{kind}-slot{slot}')
        if kind == 1:
            active[slot] = box
        if kind == 2:
            active[slot] = box
        if kind == 3:
            del active[slot]

    add(1, 0, 101, (-2, -2, 0, 0), 'first')
    add(1, 1, 202, (-1, -1, 1, 1), 'overlap-add')
    add(2, 1, box=(5, 5, 6, 6), label='move-remove-buffered')
    add(4, label='commit-remove')
    add(2, 1, box=(-1, -1, 1, 1), label='move-add-buffered')
    add(4, label='commit-add')
    add(5, user=512, box=(-100, -100, 100, 100), label='query-all')
    add(3, 0, label='destroy-remove')
    add(1, 2, 303, (-1, -1, 1, 1), 'proxy-reuse')
    for slot in range(3, 11):
        add(1, slot, 1000 + slot, (-1, -1, 1, 1), 'identical-bound-ties')
    for i in range(count):
        x, y = rng.uniform(-125, 125), rng.uniform(-125, 125)
        dx, dy = rng.uniform(.01, 12), rng.uniform(.01, 12)
        box = (x - dx, y - dy, x + dx, y + dy)
        choice = rng.random()
        free = sorted(set(range(64)) - active.keys())
        if not active or choice < .15 and free and len(active) < 32:
            add(1, rng.choice(free), 10000 + i, box, f'random-create-{i}')
        elif choice < .60:
            add(2, rng.choice(sorted(active)), box=box, label=f'random-move-{i}')
        elif choice < .70:
            add(3, rng.choice(sorted(active)), label=f'random-destroy-{i}')
        elif choice < .82:
            add(4, label=f'random-commit-{i}')
        else:
            add(5, user=rng.choice([0, 1, 2, 16, 512]), box=box, label=f'random-query-{i}')
    add(4, label='final-buffer-commit')
    if null_callback:
        free = sorted(set(range(64)) - active.keys())
        first, zero = free[:2]
        add(1, first, 424242, (-1, -1, 1, 1), 'null-callback-partner')
        add(1, zero, 0, (-1, -1, 1, 1), 'null-callback-add')
        add(2, zero, box=(80, 80, 81, 81), label='null-callback-move-away')
        add(4, label='null-callback-remove-zero-token')
        add(2, zero, box=(-1, -1, 1, 1), label='null-callback-reinsert')
        add(4, label='null-callback-readd-zero-token')
        add(3, zero, label='null-callback-destroy-zero-token')
    add(6, user=65535, label='valid-private-timestamp-wrap-precondition')
    add(5, user=512, box=(-100, -100, 100, 100), label='wrap-reset-query')
    for slot in sorted(active):
        add(3, slot, label='ordered-final-destroy')
    add(5, user=512, box=(-100, -100, 100, 100), label='empty-query')
    header = struct.pack('<4I4f', 0x3153424e, 1, len(operations), 8, -100, -100, 100, 100)
    return header + b''.join(operations), labels


def decode(data, count):
    rows, offset = [], 0
    for i in range(count):
        ordinal, result, queries, events, words = RESULT.unpack_from(data, offset)
        offset += RESULT.size
        if ordinal != i or words * 4 != STATE_BYTES or queries > 512 or events > 4096:
            raise RuntimeError('invalid SAP result header')
        query = list(struct.unpack_from(f'<{queries}I', data, offset))
        offset += queries * 4
        callbacks = [list(struct.unpack_from('<4I', data, offset + j * 16)) for j in range(events)]
        offset += events * 16
        state = data[offset:offset + STATE_BYTES]
        if len(state) != STATE_BYTES or state[:8] != bytes(8):
            raise RuntimeError('short/non-normalized SAP state')
        offset += STATE_BYTES
        rows.append({'operation': i, 'result': result, 'queries': query,
                     'callbacks': callbacks, 'state_sha256': sha(state)})
    if offset != len(data):
        raise RuntimeError('trailing SAP output')
    return rows


def capacity_history():
    operations, labels = [], []
    def add(kind, slot=0, user=0, box=(0, 0, 1, 1), label=''):
        operations.append(OP.pack(kind, slot, user, *box, 0))
        labels.append(label)
    for slot in range(512):
        x, y = -96 + 6 * (slot % 32), -96 + 12 * (slot // 32)
        add(1, slot, slot + 1, (x - 1, y - 1, x + 1, y + 1), f'disjoint-create-{slot}')
    for limit in [0, 1, 16, 511, 512]:
        add(5, user=limit, box=(-100, -100, 100, 100), label=f'full-query-limit-{limit}')
    for slot in range(0, 512, 2):
        add(3, slot, label=f'free-even-{slot}')
    for slot in range(0, 512, 2):
        x, y = -96 + 6 * (slot % 32), -96 + 12 * (slot // 32)
        add(1, slot, 10000 + slot, (x - 1, y - 1, x + 1, y + 1), f'reuse-even-{slot}')
    add(5, user=512, box=(-100, -100, 100, 100), label='query-after-reuse')
    for slot in range(511, -1, -1):
        add(3, slot, label=f'final-reverse-destroy-{slot}')
    header = struct.pack('<4I4f', 0x3153424e, 1, len(operations), 8, -100, -100, 100, 100)
    return header + b''.join(operations), labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--runner', type=Path)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--random', type=int, default=200)
    p.add_argument('--seed', type=int, default=661003)
    p.add_argument('--null-callback', action='store_true')
    p.add_argument('--capacity-512', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=60)
    a = p.parse_args()
    if not 0 <= a.random <= 3900 or sha(a.library.read_bytes()) != PIN:
        p.error('invalid bounded SAP corpus or native pin')
    if a.capacity_512 and a.null_callback:
        p.error('capacity and null-callback fixtures are separate scopes')
    packet, labels = capacity_history() if a.capacity_512 else generate(a.random, a.seed, a.null_callback)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    operations = a.output.with_suffix('.ops.bin')
    operations.write_bytes(packet)
    command = [str(a.oracle), str(a.library), str(operations)]
    if a.capacity_512:
        command.append('--capacity-512')
    native = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=a.timeout, check=True)
    a.output.with_suffix('.native.bin').write_bytes(native.stdout)
    rows = decode(native.stdout, len(labels))
    report = {'schema': 'irisu-native-SAP-history-parity-v1', 'promotion_eligible': False,
              'scope': 'Exact native SAP stage, default <=64 live proxies or opt-in <=512 with conservative pair/buffer guards; calloc-backed explicit initial state, 95,292B snapshots with only first two native owner/callback pointer words zeroed; callback/query ordering retained. Timestamp force is an explicit private-state precondition to exercise wrap. No world/search admission.',
              'seed': a.seed, 'random_operations': 0 if a.capacity_512 else a.random, 'operations': len(labels),
              'null_callback_fixture': a.null_callback,
              'capacity_512_disjoint_fixture': a.capacity_512,
              'callback_policy': 'PairAdded returns/logs token0 if either proxy user ID is0; other additions consume nextpositive token. PairRemoved logs the stored token including0.',
              'labels': labels, 'native_command': command, 'native_metadata': native.stderr.decode(),
              'native_oracle_sha256': sha(a.oracle.read_bytes()), 'native_pin': PIN,
              'input_sha256': sha(packet), 'native_output_sha256': sha(native.stdout),
              'steps': rows, 'typed_parity': None}
    if a.runner:
        command = [str(a.runner), str(operations)] + (['--gpu'] if a.gpu else [])
        env = dict(os.environ) if a.gpu else None
        if a.gpu:
            env.pop('CUDA_VISIBLE_DEVICES', None)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=a.timeout, check=True, env=env)
        report.update(typed_command=command, typed_metadata=result.stderr.decode(),
                      typed_runner_sha256=sha(a.runner.read_bytes()), typed_output_sha256=sha(result.stdout),
                      typed_parity=result.stdout == native.stdout)
        if result.stdout != native.stdout:
            limit = min(len(result.stdout), len(native.stdout))
            at = next((i for i in range(limit) if result.stdout[i] != native.stdout[i]), limit)
            report['first_difference_byte'] = at
            # Decoding exposes which semantic stage first differs even when a
            # different result count moves all subsequent byte offsets.
            try:
                actual = decode(result.stdout, len(labels))
                report['first_difference_operation'] = next(i for i, (x, y) in enumerate(zip(rows, actual)) if x != y)
            except (RuntimeError, struct.error):
                report['typed_decode_failed'] = True
        a.output.with_suffix('.typed.bin').write_bytes(result.stdout)
    report['native_library_unchanged'] = sha(a.library.read_bytes()) == PIN
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(a.output)
    if report['typed_parity'] is False or not report['native_library_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
