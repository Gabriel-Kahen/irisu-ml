#!/usr/bin/env python3
"""Warm native resident sessions against captured exact GPU Session-v4 bytes.

The GPU capture must use the same inputs, depth, selection, and digest mode.
This does not run or time the GPU and cannot establish a GPU speedup itself.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', default='build-project-exact/irisu-search-session-oracle')
    for name in ('replay', 'specs', 'gpu-results', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    for name in ('prefix', 'horizon', 'beam', 'depths'):
        p.add_argument('--' + name, required=True, type=int)
    p.add_argument('--workers', nargs='+', type=int, default=[1, 8, 16])
    p.add_argument('--target', type=int, default=4393102)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--final-only', action='store_true')
    p.add_argument('--greedy-beam', action='store_true')
    p.add_argument('--no-protect-incumbent', action='store_true')
    a = p.parse_args()
    if a.repeats < 1 or a.warmups < 0 or a.timeout <= 0:
        p.error('invalid repeat or timeout limits')
    expected = a.gpu_results.read_bytes()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    cases = []
    for workers in a.workers:
        command = [a.native, str(a.replay), str(a.specs), str(a.prefix),
                   str(a.horizon), str(workers), str(a.beam), str(a.depths), str(a.target)]
        for flag in ('final_only', 'greedy_beam', 'no_protect_incumbent'):
            if getattr(a, flag):
                command.append('--' + flag.replace('_', '-'))
        runs, warmups = [], []
        for i in range(a.warmups + a.repeats):
            start = time.monotonic()
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=a.timeout, check=True)
            elapsed = time.monotonic() - start
            if result.stdout != expected:
                raise RuntimeError(f'native resident bytes differ for workers={workers}')
            row = {'wall_seconds': elapsed, 'metadata': json.loads(result.stderr),
                   'result_sha256': digest(result.stdout), 'byte_parity': True}
            (warmups if i < a.warmups else runs).append(row)
        cases.append({'workers': workers, 'command': command, 'warmups': warmups,
                      'runs': runs, 'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs)})
    receipt = {'schema': 'irisu-native-resident-session-comparison-v1',
               'promotion_eligible': False,
               'scope': 'Native exact prefix once; selected fork/COW states retained across depths; same Session-v4 results, lifecycle and selection as the supplied GPU capture. Python parity/hash work excluded from process wall timers. GPU not timed by this tool.',
               'inputs': {name: {'path': str(path), 'sha256': digest(path.read_bytes())}
                          for name, path in [('replay', a.replay), ('specs', a.specs),
                                             ('gpu_results', a.gpu_results)]},
               'native_sha256': digest(Path(a.native).read_bytes()),
               'cases': cases}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
