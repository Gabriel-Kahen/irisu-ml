#!/usr/bin/env python3
"""Qualify opt-in GPU search against actual native packets and state continuations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LIBRARY_SHA = 'ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'runner', 'library', 'replay', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runner-flag', action='append', default=[])
    parser.add_argument('--all-small-selected', action='store_true',
                        help='Independently validate every leaf when candidate count is at most 512')
    parser.add_argument('--late-prefixes', action='store_true')
    parser.add_argument('--force-host-prefix', action='store_true')
    parser.add_argument('--spread-small-candidates', action='store_true',
                        help='Sample small fixtures across the mutation menu to exercise shared checkpoints')
    args = parser.parse_args()
    work = args.output.parent
    work.mkdir(parents=True, exist_ok=True)
    pins = {name: dict(path=str(getattr(args, name).resolve()), sha256=sha(getattr(args, name)))
            for name in ('native', 'runner', 'library', 'replay')}
    report = dict(schema='irisu-gpu-efficiency-search-qualification-v1', status='running',
                  promotion_eligible=False, inputs=pins, cases=[])
    gpu_env = dict(os.environ)
    gpu_env.pop('CUDA_VISIBLE_DEVICES', None)
    native_env = dict(os.environ, LD_LIBRARY_PATH=str(args.library.resolve().parent))
    native_env.pop('LD_PRELOAD', None)
    assert pins['library']['sha256'] == LIBRARY_SHA, 'unexpected native library'
    linkage = subprocess.run(['ldd', str(args.native.resolve())], env=native_env,
                             capture_output=True, text=True, timeout=30, check=True)
    mapped = []
    for line in linkage.stdout.splitlines():
        if '=>' in line and args.library.name in line.split('=>', 1)[0]:
            path = Path(line.split('=>', 1)[1].strip().split()[0]).resolve(strict=True)
            mapped.append(dict(path=str(path), sha256=sha(path)))
    assert len(mapped) == 1 and mapped[0]['sha256'] == LIBRARY_SHA, 'native mapping differs'
    report['native_linkage'] = dict(stdout=linkage.stdout, mapped=mapped, LD_PRELOAD_removed=True)
    inputs, expected = {}, {}

    def execute(command, env, output):
        process = subprocess.run(command, env=env, capture_output=True, timeout=600)
        output.write_bytes(process.stdout)
        output.with_suffix('.log').write_bytes(process.stderr)
        if process.returncode:
            raise RuntimeError(f'{command[0]} failed: {process.stderr.decode()[-2000:]}')
        return process.stdout, json.loads(process.stderr.splitlines()[-1])

    cases = [(8, 100, trace, host, 0) for trace in (False, True) for host in (False, True)]
    cases += [(8, 2000, True, True, 2), (512, 100, True, False, 0),
              (512, 100, False, False, 1), (12288, 100, False, False, 0)]
    if args.late_prefixes:
        cases += [(8, 96000, False, True, 1), (8, 97150, True, True, 2),
                  (8, 97218, True, True, 0)]
    if args.force_host_prefix:
        cases = list(dict.fromkeys((count, prefix, trace, True, poison)
                                   for count, prefix, trace, _, poison in cases))
    try:
        for count, prefix, trace, host, poison in cases:
            key = (count, prefix)
            if key not in inputs:
                specs = work / f'specs-{count}-{prefix}.bin'
                subprocess.run([sys.executable, 'tools/cuda-native/search_specs.py',
                                str(args.replay), str(specs), '--prefix', str(prefix),
                                '--horizon', '4', '--count', str(512 if args.spread_small_candidates and count < 512 else count)],
                               check=True, capture_output=True)
                if args.spread_small_candidates and count < 512:
                    pool = specs.read_bytes()
                    indices = [i * 512 // count for i in range(count)]
                    specs.write_bytes(b''.join(pool[i * 40:(i + 1) * 40] for i in indices))
                    specs.with_suffix('.bin.json').write_text(json.dumps(dict(
                        sampled_from=512, indices=indices, specifications=count,
                        specs_sha256=sha(specs)), indent=2) + '\n')
                inputs[key] = specs
            specs = inputs[key]
            common = [str(args.replay.resolve()), str(specs.resolve()), str(prefix), '4']
            final = [] if trace else ['--final-only']
            native_key = (count, prefix, trace)
            if native_key not in expected:
                packet, _ = execute([str(args.native.resolve()), *common, '8', '4393102', *final],
                                    native_env, work / f'native-{count}-{prefix}-{int(trace)}.bin')
                expected[native_key] = packet
            command = [str(args.runner.resolve()), *common, *final, '--copy-u64', '--reuse-prefix',
                       '--physical-snapshot', '--compact-clone', '--merge-rank', '--tiled-selection',
                       '--parallel-prefix-guard', '--warmups', '0', '--repeats', '3']
            if host:
                command += ['--host-prefix']
            if poison:
                command += ['--scratch-poison', str(poison)]
            command += args.runner_flag
            if args.all_small_selected and count <= 512:
                command += ['--keep', str(count)]
            packet, metadata = execute(command, gpu_env,
                                       work / f'gpu-{count}-{prefix}-{int(trace)}-{int(host)}-{poison}.bin')
            assert packet == expected[native_key], 'actual native Result/rank packet differs'
            assert metadata['selected_state_checks'] and metadata['prefix_immutability_checks']
            assert all(metadata[k] for k in ('compact_clone', 'merge_rank', 'tiled_selection',
                                             'parallel_prefix_guard', 'physical_snapshot', 'reuse_prefix'))
            assert metadata['prefix_executions'] == 1 and metadata['repeats'] == 3
            assert not any(metadata[k] for k in ('failed_candidates', 'compaction_errors',
                                                'prefix_immutability_errors', 'guest_calls', 'fallback_calls'))
            if args.spread_small_candidates and '--adaptive-shared-prefixes' in args.runner_flag:
                assert metadata['shared_checkpoint_count'] > 1, 'fixture did not activate shared execution'
                assert metadata['independent_selected_replays_per_batch'] == (count if count <= 512 and args.all_small_selected else metadata['keep'])
                if metadata.get('host_shared_checkpoints'):
                    assert metadata['shared_checkpoint_immutability_checks']
                    assert metadata['shared_checkpoint_guard_executions'] == 3
                    assert not metadata['shared_checkpoint_immutability_errors']
            row = dict(candidates=count, prefix=prefix, horizon=4, trace=trace, host_prefix=host,
                       scratch_poison=poison, native_packet_equal=True, packet_sha256=hashlib.sha256(packet).hexdigest(),
                       specs_sha256=sha(specs), command=command, metadata=metadata)
            report['cases'].append(row)
            print(json.dumps({k: row[k] for k in ('candidates', 'prefix', 'trace', 'host_prefix',
                                                'scratch_poison', 'native_packet_equal')}), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        report['inputs_unchanged'] = all(sha(pin['path']) == pin['sha256'] for pin in pins.values())
        assert report['inputs_unchanged']
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
