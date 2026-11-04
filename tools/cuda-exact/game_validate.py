#!/usr/bin/env python3
"""Bounded full-game native/translated trajectory differential with receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from compare_game_digests import compare

ROOT=Path(__file__).resolve().parents[2]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--replay',type=Path,default=ROOT/'irisu-high-score-4393101.rpy')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--clone-at',type=int)
    p.add_argument('--arena-mib',type=int,default=8)
    p.add_argument('--timeout',type=float,default=300)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    replay=args.replay.read_bytes()
    if len(replay)<56 or (len(replay)-52)%4: p.error('padded normal replay required')
    available=(len(replay)-52)//4
    steps=available if args.steps is None else args.steps
    if not 0<steps<=available: p.error('steps outside replay')
    if args.clone_at is not None and (args.gpu or not 0<args.clone_at<=steps):p.error('clone-at requires host and 0<tick<=steps')
    seeds=args.seed or [struct.unpack_from('<I',replay)[0]]
    if any(not 0<=seed<2**32 for seed in seeds):p.error('seed outside uint32')
    if steps==available and seeds!=[struct.unpack_from('<I',replay)[0]]:p.error('alternate seed gates require partial replay to avoid canonical header assertion')
    env=os.environ.copy()
    if args.gpu:env.pop('CUDA_VISIBLE_DEVICES',None)
    report={'schema':'irisu-cuda-game-differential-v1','exact_gpu_promoted':False,
      'backend':'cuda' if args.gpu else 'host-translated','steps_per_case':steps,
      'inputs':{str(path):sha(path) for path in [args.oracle,args.probe,args.image,args.replay]},
      'cases':[]}
    report_path=args.output/'comparison.json'
    try:
      for seed in seeds:
        path=args.output/f'seed-{seed}.rpy';path.write_bytes(struct.pack('<I',seed)+replay[4:])
        case={'seed':seed,'replay_sha256':sha(path),'runs':[]};report['cases'].append(case)
        for name,command in [('native',[str(args.oracle.resolve()),str(path.resolve()),str(steps)]),
          ('probe',[str(args.probe.resolve()),str(args.image.resolve()),str(path.resolve()),str(steps),'--arena-mib',str(args.arena_mib)])]:
          if name=='probe':
            if args.gpu:command.append('--gpu')
            if args.clone_at is not None:command.extend(['--clone-at',str(args.clone_at)])
          output=args.output/f'{name}-{seed}.digest';metadata=args.output/f'{name}-{seed}.json'
          start=time.perf_counter()
          with output.open('wb') as stdout,metadata.open('wb') as stderr:
            subprocess.run(command,stdout=stdout,stderr=stderr,check=True,timeout=args.timeout,env=env,cwd=ROOT)
          case['runs'].append({'name':name,'command':command,'wall_seconds':time.perf_counter()-start,
            'digest_sha256':sha(output),'metadata':json.loads(metadata.read_text())})
        case['comparison']=compare(args.output/f'native-{seed}.digest',args.output/f'probe-{seed}.digest')
      report['status']='passed'
    except Exception as error:
      report['status']='failed';report['failure']=str(error)
      raise
    finally:report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(seeds),'rows_per_case':steps+1,'receipt':str(report_path)},indent=2))

if __name__=='__main__':main()
