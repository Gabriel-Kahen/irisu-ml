#!/usr/bin/env python3
"""Compose exact wide/multi-root Results and global ranks from native chunks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

RESULT=struct.Struct('<21Q11q')
LIBRARY_SHA='ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ranking(rows):return sorted(range(len(rows)),key=lambda i:(bool(rows[i][17]),tuple(-x for x in rows[i][21:]),i))
def packet(rows):return struct.pack('<4I',0x32535249,2,len(rows),256)+b''.join(RESULT.pack(*row) for row in rows)+struct.pack('<'+'I'*len(rows),*ranking(rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['oracle','library','replay','specs','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prefix',type=int,default=100);p.add_argument('--horizon',type=int,default=4);p.add_argument('--roots',type=int,default=1);p.add_argument('--root-seed-step',type=int,default=1);p.add_argument('--root-prefix-step',type=int,default=0)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--chunk',type=int,default=16384);p.add_argument('--target',type=int,default=4393102);p.add_argument('--trace-every-tick',action='store_true');a=p.parse_args()
    assert sha(a.library)==LIBRARY_SHA and 1<=a.roots<=32 and 1<=a.chunk<=16384
    specs=a.specs.read_bytes();assert specs and not len(specs)%40;count=len(specs)//40;assert a.roots<=count<=131072
    replay=a.replay.read_bytes();seed=struct.unpack_from('<I',replay)[0];rows=[None]*count
    work=a.output.parent/(a.output.stem+'-native-chunks');work.mkdir(parents=True,exist_ok=True)
    pins={k:dict(path=str(getattr(a,k).resolve()),sha256=sha(getattr(a,k))) for k in ['oracle','library','replay','specs']};report=dict(status='running',inputs=pins,candidates=count,roots=a.roots,chunks=[])
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.library.resolve().parent));env.pop('LD_PRELOAD',None)
    linkage=subprocess.run(['ldd',str(a.oracle.resolve())],env=env,capture_output=True,text=True,timeout=30,check=True)
    mapped=[]
    for line in linkage.stdout.splitlines():
        if '=>' in line and a.library.name in line.split('=>',1)[0]:
            path=Path(line.split('=>',1)[1].strip().split()[0]).resolve(strict=True)
            mapped.append(dict(path=str(path),sha256=sha(path)))
    assert len(mapped)==1 and mapped[0]['sha256']==LIBRARY_SHA,'native DSO mapping differs'
    report['native_linkage']=dict(stdout=linkage.stdout,mapped=mapped,LD_PRELOAD_removed=True)
    for root in range(a.roots):
        root_seed=seed+root*a.root_seed_step;prefix=a.prefix+root*a.root_prefix_step;assert root_seed<=0xffffffff and prefix<=(len(replay)-52)//4
        source=bytearray(replay);struct.pack_into('<I',source,0,root_seed)
        if root_seed!=seed:struct.pack_into('<3I',source,4,0,0,0)
        root_replay=work/f'root-{root}.rpy';root_replay.write_bytes(source)
        indices=list(range(root,count,a.roots))
        for offset in range(0,len(indices),a.chunk):
            chosen=indices[offset:offset+a.chunk];path=work/f'root-{root}-chunk-{offset}.bin';path.write_bytes(b''.join(specs[i*40:(i+1)*40] for i in chosen))
            command=[str(a.oracle.resolve()),str(root_replay.resolve()),str(path.resolve()),str(prefix),str(a.horizon),str(a.workers),str(a.target)]
            if not a.trace_every_tick:command.append('--final-only')
            start=time.perf_counter();proc=subprocess.run(command,env=env,capture_output=True,timeout=900)
            (path.with_suffix('.stderr')).write_bytes(proc.stderr);assert proc.returncode==0,proc.stderr.decode(errors='replace')
            assert len(proc.stdout)==16+len(chosen)*260 and struct.unpack_from('<4I',proc.stdout)==(0x32535249,2,len(chosen),256)
            for local,logical in enumerate(chosen):rows[logical]=RESULT.unpack_from(proc.stdout,16+local*256)
            report['chunks'].append(dict(root=root,root_seed=root_seed,prefix=prefix,indices=chosen,command=command,seconds=time.perf_counter()-start,output_sha256=hashlib.sha256(proc.stdout).hexdigest()))
    assert all(row is not None for row in rows);result=packet(rows);a.output.write_bytes(result)
    report.update(status='passed',output_sha256=hashlib.sha256(result).hexdigest(),output_bytes=len(result),distinct_results=len(set(RESULT.pack(*row) for row in rows)),inputs_unchanged=all(sha(Path(v['path']))==v['sha256'] for v in pins.values()),scope='Native chunks preserve complete Results; ranking recomputed globally with original logical candidate IDs and failure-first ordering.')
    assert report['inputs_unchanged'];a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['chunks','inputs']}))

if __name__=='__main__':main()
