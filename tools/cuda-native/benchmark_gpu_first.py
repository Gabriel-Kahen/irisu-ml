#!/usr/bin/env python3
"""Measure fully GPU-prepared searches against complete native Result packets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions():
    def query(fields):
        return subprocess.check_output(['nvidia-smi', *fields], text=True, timeout=15).strip()
    return dict(gpu=query(['--query-gpu=name,power.draw,temperature.gpu,memory.used,utilization.gpu', '--format=csv,noheader']),
                processes=query(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']),
                cpu_pressure=Path('/proc/pressure/cpu').read_text().strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['runner', 'replay', 'specs', 'expected', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--runner-flag', action='append', default=[])
    p.add_argument('--baseline-flag', action='append', default=[])
    p.add_argument('--prefix', type=int, default=100)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--batches', type=int, default=20)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--cooldown', type=float, default=1)
    p.add_argument('--timeout', type=float, default=600)
    a = p.parse_args()
    count, tail = divmod(a.specs.stat().st_size, 40)
    if tail or not 1 <= count <= 131072 or not 1 <= a.batches <= 20 or not 1 <= a.rounds <= 10:
        p.error('invalid bounded input or campaign')
    if not 0 <= a.cooldown <= 30 or a.timeout <= 0:
        p.error('invalid cooldown or timeout')
    if any(flag in ('--host-prefix', '--host-shared-checkpoints', '--cpu')
           for flag in a.runner_flag + a.baseline_flag):
        p.error('this campaign requires GPU physics preparation')
    expected = a.expected.read_bytes()
    assert len(expected) == 16 + count * 260
    assert struct.unpack_from('<4I', expected) == (0x32535249, 2, count, 256)
    inputs = {name: dict(path=str(getattr(a, name).resolve()), sha256=sha(getattr(a, name)))
              for name in ['runner', 'replay', 'specs', 'expected'] + (['baseline'] if a.baseline else [])}
    report = dict(schema='irisu-gpu-first-campaign-v1', status='running', inputs=inputs,
                  candidates=count, horizon=a.horizon, prefix=a.prefix, batches=a.batches,
                  timing_scope='Full child process: GPU reset/prefix/checkpoints/futures/ranking, setup, guards, output and teardown; native comparison outside timer.',
                  warmups=[], runs=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + '\n')

    def run(backend, batches, round_index):
        binary = a.runner if backend == 'candidate' else a.baseline
        flags = a.runner_flag if backend == 'candidate' else a.baseline_flag
        command = [str(binary.resolve()), str(a.replay.resolve()), str(a.specs.resolve()),
                   str(a.prefix), str(a.horizon), '--final-only', '--copy-u64', '--reuse-prefix',
                   '--physical-snapshot', '--warmups', '0', '--repeats', str(batches),
                   '--skip-compaction-check', '--compact-clone', '--merge-rank',
                   '--tiled-selection', '--parallel-prefix-guard', '--adaptive-shared-prefixes', *flags]
        time.sleep(a.cooldown)
        before = conditions()
        assert not before['processes'], 'another GPU compute process is active'
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        start = time.perf_counter()
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        wall = time.perf_counter() - start
        after = conditions()
        if process.returncode:
            raise RuntimeError(stderr.decode(errors='replace')[-4000:])
        assert stdout == expected, 'complete native Result/rank packet differs'
        metadata = json.loads(stderr.splitlines()[-1])
        assert metadata['candidates'] == count and metadata['horizon'] == a.horizon
        assert metadata['repeats'] == batches and not metadata.get('host_prefix', False)
        assert not metadata.get('host_shared_checkpoints', False)
        for field in ['host_physics_calls', 'host_shared_checkpoint_ticks', 'guest_calls',
                      'failed_candidates', 'compaction_errors', 'prefix_immutability_errors',
                      'selected_state_errors', 'shared_checkpoint_immutability_errors']:
            assert not metadata.get(field, 0), (field, metadata[field])
        actual = metadata.get('measured_gpu_executed_future_ticks')
        assert actual is not None and 0 < actual <= count * a.horizon * batches
        row = dict(backend=backend, round=round_index, batches=batches, command=command,
                   wall_seconds=wall, native_packet_equal=True, output_sha256=hashlib.sha256(stdout).hexdigest(),
                   logical_candidate_futures=count * batches, actual_gpu_future_ticks=actual,
                   candidate_futures_per_second=count * batches / wall,
                   gpu_conditions_before=before, gpu_conditions_after=after, metadata=metadata)
        print(json.dumps({k: row[k] for k in ['backend', 'round', 'wall_seconds', 'candidate_futures_per_second']}), flush=True)
        return row

    backends = ['baseline', 'candidate'] if a.baseline else ['candidate']
    try:
        for backend in backends:
            report['warmups'].append(run(backend, 1, -1))
            save()
        for i in range(a.rounds):
            for backend in (backends if i % 2 == 0 else list(reversed(backends))):
                report['runs'].append(run(backend, a.batches, i))
                save()
        medians = {b: statistics.median(r['wall_seconds'] for r in report['runs'] if r['backend'] == b)
                   for b in backends}
        report['summary'] = dict(median_seconds=medians,
                                candidate_futures_per_second={b: count * a.batches / t for b, t in medians.items()})
        if a.baseline:
            report['summary']['speedup'] = medians['baseline'] / medians['candidate']
        report['inputs_unchanged'] = all(sha(Path(v['path'])) == v['sha256'] for v in inputs.values())
        assert report['inputs_unchanged']
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
