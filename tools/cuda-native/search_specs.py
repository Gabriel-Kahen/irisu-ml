#!/usr/bin/env python3
"""Reproduce distinct sparse search candidates; repetition is explicit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]


def generate(replay, prefix, horizon, count, repeat=1):
    if (len(replay) <= 52 or (len(replay)-52) % 4 or
            struct.unpack_from('<I', replay, 16)[0] != 0):
        raise ValueError('expected padded normal-mode replay')
    if not (0 <= prefix <= (len(replay)-52)//4 and 1 <= horizon <= 100000 and
            prefix+horizon <= 0xffffffff and 1 <= count <= 16384 and
            1 <= repeat and count*repeat <= 16384):
        raise ValueError('invalid bounded prefix/horizon/candidate counts')
    source = ROOT / 'tools/cuda-exact/search_validate.py'
    spec = importlib.util.spec_from_file_location('shared_search_specs', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    distinct = module.candidates(replay, prefix, horizon, count)
    assert len(distinct) == count*40
    assert len({distinct[i:i+40] for i in range(0, len(distinct), 40)}) == count
    return distinct*repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--prefix', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--count', type=int, default=12288)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    replay = args.replay.read_bytes()
    packet = generate(replay, args.prefix, args.horizon, args.count, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(packet)
    metadata = dict(prefix=args.prefix, horizon=args.horizon,
                    distinct_specifications=args.count, repeats=args.repeat,
                    total_candidates=args.count*args.repeat, bytes=len(packet),
                    replay_sha256=hashlib.sha256(replay).hexdigest(),
                    specs_sha256=hashlib.sha256(packet).hexdigest(),
                    candidate_format='<10I',
                    scope='Distinct effective input edits; result diversity needs native qualification.')
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
