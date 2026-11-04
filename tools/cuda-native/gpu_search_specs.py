#!/usr/bin/env python3
"""Distinct wide candidate menu, without changing the existing bounded reader."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]

def generate(replay,prefix,horizon,count):
    if len(replay)<=52 or (len(replay)-52)%4 or struct.unpack_from('<I',replay,16)[0]:
        raise ValueError('expected padded normal-mode replay')
    if not (0<=prefix<=(len(replay)-52)//4 and 1<=horizon<=100000 and prefix+horizon<=0xffffffff and 1<=count<=131072):
        raise ValueError('invalid wide candidate bounds')
    spec=importlib.util.spec_from_file_location('wide_menu',ROOT/'tools/cuda-exact/search_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    packet=module.candidates(replay,prefix,horizon,count)
    assert len(packet)==count*40 and len({packet[i:i+40] for i in range(0,len(packet),40)})==count
    return packet

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--prefix',type=int,default=100);p.add_argument('--horizon',type=int,default=4);p.add_argument('--count',type=int,default=24576)
    a=p.parse_args();replay=a.replay.read_bytes();packet=generate(replay,a.prefix,a.horizon,a.count)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(packet)
    receipt=dict(prefix=a.prefix,horizon=a.horizon,distinct_specifications=a.count,bytes=len(packet),candidate_format='<10I',replay_sha256=hashlib.sha256(replay).hexdigest(),specs_sha256=hashlib.sha256(packet).hexdigest())
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
