#!/usr/bin/env python3
"""Compare original ELF32 circle stage CPU1/8/16 with the new CUDA stage.

Reports stage timings and complete process wall separately. This microbenchmark
cannot establish complete world, game, or search performance.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from oracle_collision import PIN, generate, sha, validate_sentinels


def invoke(command, timeout, expected, gpu=False):
    env = dict(os.environ) if gpu else None
    if gpu:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    start = time.monotonic()
    result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=True, env=env)
    wall = time.monotonic() - start
    if result.stdout != expected:
        raise RuntimeError('complete native manifold packet differs: ' + str(command))
    metadata = json.loads(result.stderr)
    return {'wall_seconds': wall, 'metadata': metadata,
            'result_sha256': sha(result.stdout), 'full_manifold_parity': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'native', 'gpu', 'library', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--random', type=int, default=262144)
    p.add_argument('--seed', type=int, default=661003)
    p.add_argument('--threads', nargs='+', type=int, default=[1, 8, 16])
    p.add_argument('--gpu-threads', type=int, default=128)
    p.add_argument('--process-warmups', type=int, default=1)
    p.add_argument('--process-repeats', type=int, default=3)
    p.add_argument('--stage-warmups', type=int, default=1)
    p.add_argument('--stage-repeats', type=int, default=7)
    p.add_argument('--timeout', type=float, default=120)
    a = p.parse_args()
    if not 0 <= a.random <= 999916 or a.process_warmups < 0 or a.process_repeats < 1:
        p.error('invalid benchmark bounds')
    if sha(a.library.read_bytes()) != PIN:
        p.error('native library pin mismatch')
    packet, labels = generate(a.random, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cases = a.output.with_suffix('.cases.bin')
    cases.write_bytes(packet)
    reference = subprocess.run([str(a.oracle), str(a.library), str(cases)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=a.timeout, check=True)
    expected = reference.stdout
    sentinel_check = validate_sentinels(packet, expected)
    configurations = [('native-cpu', threads, [a.native, a.library, cases,
                          '--threads', str(threads)]) for threads in a.threads]
    configurations.append(('cuda', a.gpu_threads, [a.gpu, cases, '--gpu',
                            '--threads', str(a.gpu_threads)]))
    rows = []
    for backend, threads, base in configurations:
        command = base + ['--warmups', str(a.stage_warmups),
                          '--repeats', str(a.stage_repeats)]
        warmups, runs = [], []
        for repetition in range(a.process_warmups + a.process_repeats):
            row = invoke(command, a.timeout, expected, gpu=backend == 'cuda')
            (warmups if repetition < a.process_warmups else runs).append(row)
        rows.append({'backend': backend, 'threads': threads,
                     'command': list(map(str, command)), 'warmups': warmups, 'runs': runs,
                     'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs),
                     'median_stage_ms': statistics.median(r['metadata']['median_stage_ms'] for r in runs)})
    cpu = [r for r in rows if r['backend'] == 'native-cpu']
    gpu = rows[-1]
    report = {'schema': 'irisu-native-circle-stage-benchmark-v1', 'promotion_eligible': False,
              'scope': 'Circle narrow phase only; exact pinned original native ELF32 functions, persistent CPU pthread workers with PC53 per thread, and new typed CUDA. No world/game/search speed claim.',
              'timing_contract': {'process': 'One process warmup then timed process repeats; startup, input parse/preparation, native pin or CUDA init, all stage warmups/repeats, last full output and cleanup included. Python comparison/hashing excluded.',
                                  'stage': 'Native timer includes barrier dispatch, fresh60Bmanifold input and actual original function; native76Bshape inputs prepared before stage timer. GPU kernel includes geometry adapter, fresh60Bmanifold and typed function; CUDA allocation/transfers excluded from kernel event samples. Native shape immutability checked by the separate authoritative oracle outside all process timers.',
                                  'stage_warmups': a.stage_warmups, 'stage_repeats': a.stage_repeats,
                                  'process_warmups': a.process_warmups, 'process_repeats': a.process_repeats},
              'corpus': {'seed': a.seed, 'random_cases': a.random, 'cases': len(labels),
                         'generator': 'circle-float-boundaries-v1'},
              'inputs_sha256': {'cases': sha(packet), 'native_library': PIN,
                                'oracle': sha(a.oracle.read_bytes()), 'native_benchmark': sha(a.native.read_bytes()),
                                'gpu_runner': sha(a.gpu.read_bytes())},
              'native_reference_metadata': reference.stderr.decode(),
              'native_result_sha256': sha(expected), 'sentinel_check': sentinel_check,
              'cases': rows,
              'fastest_cpu_stage_ms_divided_by_gpu_stage_ms': min(r['median_stage_ms'] for r in cpu) / gpu['median_stage_ms'],
              'fastest_cpu_wall_divided_by_gpu_wall': min(r['median_wall_seconds'] for r in cpu) / gpu['median_wall_seconds'],
              'native_library_unchanged': sha(a.library.read_bytes()) == PIN}
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
