#!/usr/bin/env python3
"""Compare complete GPU campaigns with identical candidates and native output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_conditions():
    fields = 'name,driver_version,power.limit,power.draw,temperature.gpu,utilization.gpu,memory.used'
    output = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=15, check=True)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader'], capture_output=True, text=True,
                          timeout=15, check=True)
    return dict(fields=fields, gpu=output.stdout.strip(), compute_apps=apps.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'replay', 'specs', 'expected', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-flag', action='append', default=[])
    parser.add_argument('--baseline-flag', action='append', default=[])
    parser.add_argument('--candidate-comparison', choices=('exact', 'outcome'), default='exact',
                        help='outcome permits hash changes, but requires identical discrete results and ranks')
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--prefix', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--cooldown', type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.batches <= 20 or not 1 <= args.rounds <= 10:
        parser.error('bounded campaign required')
    if not 0 <= args.cooldown <= 30:
        parser.error('cooldown must be between zero and 30 seconds')
    count, remainder = divmod(args.specs.stat().st_size, 40)
    if remainder or not 1 <= count <= 16384:
        parser.error('invalid candidate packet')
    pins = {name: dict(path=str(getattr(args, name).resolve()), sha256=sha(getattr(args, name)))
            for name in ('baseline', 'candidate', 'replay', 'specs', 'expected')}
    report = dict(schema='irisu-paired-gpu-efficiency-v1', status='running', inputs=pins,
                  candidates=count, batches=args.batches, horizon=args.horizon,
                  prefix=args.prefix, future_ticks=count * args.batches * args.horizon,
                  future_ticks_scope='Logical candidate ticks; shared execution is reported separately per run.',
                  cooldown_seconds_outside_timer=args.cooldown,
                  candidate_comparison=args.candidate_comparison,
                  timing_scope='Full child process including CUDA setup, output and teardown; '
                               'native packet comparison and conditions sampled outside timer.',
                  warmups=[], runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def run(backend, batches, round_index):
        binary = args.baseline if backend == 'baseline' else args.candidate
        command = [str(binary.resolve()), str(args.replay.resolve()), str(args.specs.resolve()),
                   str(args.prefix), str(args.horizon), '--final-only', '--copy-u64',
                   '--reuse-prefix', '--physical-snapshot', '--warmups', '0',
                   '--repeats', str(batches), '--skip-compaction-check']
        flags = args.candidate_flag if backend == 'candidate' else args.baseline_flag
        command += flags
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        time.sleep(args.cooldown)
        before = Path('/proc/pressure/cpu').read_text().strip()
        gpu_before = gpu_conditions()
        start = time.perf_counter()
        process = subprocess.run(command, env=env, capture_output=True, timeout=600)
        wall = time.perf_counter() - start
        after = Path('/proc/pressure/cpu').read_text().strip()
        gpu_after = gpu_conditions()
        if process.returncode:
            raise RuntimeError(process.stderr.decode()[-4000:])
        metadata = json.loads(process.stderr.decode().splitlines()[-1])
        digest = hashlib.sha256(process.stdout).hexdigest()
        equal = digest == pins['expected']['sha256']
        differences = {}
        if backend == 'candidate' and args.candidate_comparison == 'outcome':
            expected = args.expected.read_bytes()
            assert len(process.stdout) == len(expected) == 16 + count * 260
            assert process.stdout[:16] == expected[:16] == struct.pack('<4I', 0x32535249, 2, count, 256)
            names = {1: 'body_hash', 2: 'observation_hash', 3: 'event_hash',
                     5: 'transition_hash', 12: 'trace_hash'}
            for candidate in range(count):
                actual_words = struct.unpack_from('<32Q', process.stdout, 16 + candidate * 256)
                expected_words = struct.unpack_from('<32Q', expected, 16 + candidate * 256)
                for word, (actual, reference) in enumerate(zip(actual_words, expected_words)):
                    if actual != reference:
                        assert word in names, f'native discrete result differs: candidate {candidate}, word {word}'
                        key = names[word]
                        differences[key] = differences.get(key, 0) + 1
            assert process.stdout[16 + count * 256:] == expected[16 + count * 256:], 'native ranking differs'
        else:
            assert equal, 'native packet differs'
        assert metadata['candidates'] == count and metadata['horizon'] == args.horizon
        assert metadata['prefix_ticks'] == args.prefix and metadata['prefix_executions'] == 1
        assert metadata['repeats'] == batches and metadata['simulated_ticks'] == count * args.horizon
        assert sum(s.get('batches', 1) for s in metadata['samples']) == batches
        if 'total_future_ticks' in metadata:
            assert metadata['total_future_ticks'] == count * args.horizon * batches
        logical_ticks = count * args.horizon * batches
        executed_ticks = metadata.get('measured_executed_future_ticks', logical_ticks)
        assert 0 < executed_ticks <= logical_ticks
        if 'planned_executed_future_ticks' in metadata:
            planned_gpu_ticks = metadata['planned_executed_future_ticks']
            if metadata.get('host_shared_checkpoints'):
                assert metadata['host_shared_checkpoint_executions'] == 1
                planned_gpu_ticks -= metadata['host_shared_checkpoint_ticks']
                if 'shared_checkpoint_immutability_checks' in metadata:
                    assert metadata['shared_checkpoint_immutability_checks']
                    assert metadata['shared_checkpoint_guard_executions'] == batches
                    assert not metadata['shared_checkpoint_immutability_errors']
            assert metadata.get('planned_gpu_future_ticks_per_batch', planned_gpu_ticks) == planned_gpu_ticks
            assert executed_ticks == planned_gpu_ticks * batches
            assert metadata['executed_future_ticks'] * batches == executed_ticks
        expected_host_prefix = '--host-prefix' in flags
        assert metadata['host_prefix'] == expected_host_prefix and metadata['prefix_immutability_checks']
        assert not any(metadata[k] for k in ('failed_candidates', 'compaction_errors',
                                            'prefix_immutability_errors', 'guest_calls', 'fallback_calls'))
        row = dict(backend=backend, batches=batches, round=round_index, command=command,
                   wall_seconds=wall, output_sha256=digest, native_packet_equal=equal,
                   logical_future_ticks=logical_ticks, executed_future_ticks=executed_ticks,
                   host_shared_checkpoint_ticks=metadata.get('host_shared_checkpoint_ticks', 0),
                   native_discrete_results_and_ranks_equal=True, changed_hash_fields=differences,
                   cpu_pressure_before=before, cpu_pressure_after=after, metadata=metadata)
        row.update(gpu_before=gpu_before, gpu_after=gpu_after)
        phases = {key: statistics.median(s[key] / s.get('batches', 1) for s in metadata['samples'])
                  for key in ('branch_ms', 'rollout_ms', 'rank_compact_ms')}
        print(json.dumps(dict(backend=backend, round=round_index, wall_seconds=wall,
                              phase_medians=phases)), flush=True)
        return row

    try:
        for backend in ('baseline', 'candidate'):
            report['warmups'].append(run(backend, 1, -1)); save()
        for index in range(args.rounds):
            for backend in (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')):
                report['runs'].append(run(backend, args.batches, index)); save()
        medians = {backend: statistics.median(r['wall_seconds'] for r in report['runs']
                                             if r['backend'] == backend)
                   for backend in ('baseline', 'candidate')}
        report['summary'] = dict(median_seconds=medians, speedup=medians['baseline'] / medians['candidate'])
        report['inputs_unchanged'] = all(sha(Path(pin['path'])) == pin['sha256'] for pin in pins.values())
        assert report['inputs_unchanged']
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
