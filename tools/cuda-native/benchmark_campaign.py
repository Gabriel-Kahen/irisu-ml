#!/usr/bin/env python3
"""Paired actual-native/GPU cached-prefix campaigns; comparisons outside timers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions():
    frequencies = []
    for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'):
        try:
            frequencies.append(int(path.read_text()))
        except OSError:
            pass
    return dict(cpu_pressure=Path('/proc/pressure/cpu').read_text().strip(),
                load=Path('/proc/loadavg').read_text().strip(),
                affinity=sorted(os.sched_getaffinity(0)),
                frequency_khz=frequencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'gpu', 'replay', 'specs', 'expected', 'library', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prefix', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.batches <= 20 or not 1 <= args.repeats <= 10:
        parser.error('bounded campaign required')
    count = args.specs.stat().st_size // 40
    if args.specs.stat().st_size != count * 40 or not 1 <= count <= 16384:
        parser.error('invalid candidate packet')
    inputs = {name: dict(path=str(getattr(args, name).resolve()), sha256=sha(getattr(args, name)))
              for name in ('native', 'gpu', 'replay', 'specs', 'expected', 'library')}
    report = dict(schema='irisu-paired-cached-prefix-campaign-v1', status='running',
                  promotion_eligible=False, inputs=inputs, candidates=count,
                  batches=args.batches, future_ticks=count*args.batches*args.horizon,
                  scope='One process and one prefix per campaign on both backends. '
                        'External wall includes loading, allocation, execution, output and teardown. '
                        'CPU verification-free wall is an additional conservative control.',
                  warmups=[], runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def run(backend, workers, batches, round_index):
        common = [str(args.replay.resolve()), str(args.specs.resolve()),
                  str(args.prefix), str(args.horizon)]
        env = dict(os.environ)
        if backend == 'native':
            command = [str(args.native.resolve()), *common, str(workers), str(batches),
                       '4393102', '--final-only']
            env['LD_LIBRARY_PATH'] = str(args.library.resolve().parent)
        else:
            command = [str(args.gpu.resolve()), *common, '--final-only', '--copy-u64',
                       '--reuse-prefix', '--physical-snapshot', '--warmups', '0',
                       '--repeats', str(batches), '--skip-compaction-check']
            env.pop('CUDA_VISIBLE_DEVICES', None)
        before = conditions()
        start = time.perf_counter()
        process = subprocess.run(command, env=env, capture_output=True, timeout=args.timeout)
        wall = time.perf_counter() - start
        after = conditions()
        rows = [json.loads(line) for line in process.stderr.decode().splitlines()
                if line.startswith('{')]
        if process.returncode or not rows:
            raise RuntimeError(f'{backend} failed: {process.stderr.decode()[-2000:]}')
        metadata = rows[-1]
        output_sha = hashlib.sha256(process.stdout).hexdigest()
        if output_sha != inputs['expected']['sha256']:
            raise RuntimeError(f'{backend} native packet mismatch')
        if backend == 'native':
            assert metadata['prefix_preparations'] == 1
            assert metadata['simulated_future_ticks'] == count*batches*args.horizon
            assert all(metadata[k] for k in ('all_batch_packets_and_ranks_identical',
                                            'parent_checkpoint_frame_unchanged', 'loaded_inputs_unchanged'))
            verification = sum(x['verification_ms'] for x in metadata['batch_timings'])/1000
        else:
            assert metadata['prefix_executions'] == 1 and not metadata['host_prefix']
            assert metadata['numeric_profile'] == 'bounded-device-add-sub-mul'
            assert metadata['physical_snapshot'] and metadata['reuse_prefix']
            assert metadata['prefix_immutability_checks']
            assert not any(metadata[k] for k in ('failed_candidates', 'compaction_errors',
                                                'prefix_immutability_errors', 'fallback_calls'))
            verification = 0
        result = dict(backend=backend, workers=workers, batches=batches, round=round_index,
                      command=command, wall_seconds=wall,
                      verification_free_seconds=wall-verification,
                      output_sha256=output_sha, output_bytes=len(process.stdout),
                      native_packet_equal=True, before=before, after=after, metadata=metadata)
        print(json.dumps({k: result[k] for k in ('backend', 'workers', 'batches', 'round',
                                               'wall_seconds', 'verification_free_seconds')}), flush=True)
        return result

    try:
        for backend, workers in [('native', 8), ('gpu', 0), ('native', 16)]:
            report['warmups'].append(run(backend, workers, 1, -1)); save()
        for index in range(args.repeats):
            for workers in ([8, 16] if index % 2 == 0 else [16, 8]):
                report['runs'].append(run('native', workers, args.batches, index)); save()
                report['runs'].append(run('gpu', 0, args.batches, index)); save()
        medians = {workers: statistics.median(x['verification_free_seconds'] for x in report['runs']
                                             if x['backend'] == 'native' and x['workers'] == workers)
                   for workers in (8, 16)}
        gpu = [x['wall_seconds'] for x in report['runs'] if x['backend'] == 'gpu']
        best = min(medians, key=medians.get)
        report['summary'] = dict(best_cpu_workers=best, cpu_verification_free_medians=medians,
                                 gpu_full_wall_median_seconds=statistics.median(gpu),
                                 gpu_full_wall_range_seconds=[min(gpu), max(gpu)],
                                 conservative_speedup=medians[best]/statistics.median(gpu))
        report['inputs_unchanged'] = all(sha(Path(v['path'])) == v['sha256'] for v in inputs.values())
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
