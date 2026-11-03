#!/usr/bin/env python3
"""Fixed sparse-mutation full-game search parity and warmed timing matrix.

Binary formats are explicitly little endian: candidate '<10I', result '<31Q'
(last 11 signed objective keys), followed by one '<I' rank per candidate.
Each result folds reset/prefix and every simulated frame, including events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import signal
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
SPEC=struct.Struct("<10I")
RESULT=struct.Struct("<20Q11q")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates(replay, prefix, horizon, count):
    """Incumbent plus deterministic shot deletion/injection/geometry variants."""
    words=list(struct.unpack(f"<{(len(replay)-52)//4}I",replay[52:]))
    output=[(0,0,*([0]*8))]
    for ordinal in range(1,count):
        offset=((ordinal-1)//8)%horizon
        original=words[prefix+offset] if prefix+offset<len(words) else 0
        mode=(ordinal-1)%8
        if mode==0:
            value=0
        else:
            kind=1+(mode%2)
            x=(250+(mode-4)*8+(ordinal//(8*horizon))*4)%640
            y=350+((ordinal//8)%5)*8
            if original&3:
                x=max(0,min(639,((original>>2)&1023)+(mode-4)*4))
                y=min(479,(original>>12)&511)
            value=(y<<12)|(x<<2)|kind
        # Include two-edit timing displacement when an actual source shot exists.
        if mode==0 and original&3 and offset+1<horizon:
            output.append((2,0,offset,0,offset+1,original,0,0,0,0))
        else:
            output.append((1,0,offset,value,0,0,0,0,0,0))
    return b"".join(SPEC.pack(*row) for row in output)


def decode(data,count):
    expected=count*(RESULT.size+4)
    if len(data)!=expected:
        raise RuntimeError(f"search result length {len(data)} != {expected}")
    results=[RESULT.unpack_from(data,i*RESULT.size) for i in range(count)]
    ranks=list(struct.unpack_from(f"<{count}I",data,count*RESULT.size))
    if sorted(ranks)!=list(range(count)):
        raise RuntimeError("rank output is not a permutation")
    for i,row in enumerate(results):
        if row[17] or row[18]:
            raise RuntimeError(f"candidate {i} failed guest={row[18]} pc={row[19]}")
    expected_ranks=sorted(range(count),key=lambda i:(results[i][20:],-i),reverse=True)
    if ranks!=expected_ranks:
        raise RuntimeError("stable canonical ranking disagrees with scalar reference")
    return results,ranks


def invoke(command,timeout,count):
    env=dict(os.environ);env.pop("CUDA_VISIBLE_DEVICES",None)
    start=time.perf_counter()
    proc=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          env=env,start_new_session=True)
    try:
        stdout,stderr=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.communicate()
        raise
    elapsed=time.perf_counter()-start
    lines=stderr.decode(errors="replace").splitlines()
    if proc.returncode:
        raise RuntimeError(f"command failed {proc.returncode}: {command}\n"+"\n".join(lines[-10:]))
    metadata=next((json.loads(line) for line in reversed(lines) if line.startswith("{")),{})
    decode(stdout,count)
    return stdout,{"wall_seconds":elapsed,"metadata":metadata}


class Telemetry:
    def __init__(self):
        self.samples=[];self.stop=threading.Event();self.thread=None

    def __enter__(self):
        def poll():
            while not self.stop.is_set():
                p=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True)
                if p.returncode==0:
                    try:self.samples.append([time.time(),*map(float,p.stdout.strip().split(","))])
                    except ValueError:pass
                self.stop.wait(.2)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self

    def __exit__(self,*_):
        self.stop.set();self.thread.join()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle",type=Path,required=True)
    p.add_argument("--gpu",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--replay",type=Path,default=ROOT/"irisu-high-score-4393101.rpy")
    p.add_argument("--prefix",type=int,default=0)
    p.add_argument("--horizon",type=int,default=1)
    p.add_argument("--candidates",type=int,default=8)
    p.add_argument("--arena-mib",type=int,default=8)
    p.add_argument("--keep",type=int,default=2)
    p.add_argument("--workers",type=int,nargs="+",default=[1,8,16])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--timeout",type=float,default=600)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--host-prefix",action="store_true",help="prepare one translated prefix on CPU then upload/rebind it; CPU preparation remains timed")
    p.add_argument("--isolated-warps",action="store_true",help="one active thread per warp avoids sharing divergent guest dispatch across worlds")
    args=p.parse_args()
    if args.horizon<1 or not 1<=args.candidates<=4096 or not 1<=args.keep<=args.candidates or args.repeats<1:
        p.error("invalid bounded workload")
    if args.receipt:
        from runtime_identity import verify
        verify(json.loads(args.receipt.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    specs=args.output.with_suffix(".specs.bin")
    specs.write_bytes(candidates(args.replay.read_bytes(),args.prefix,args.horizon,args.candidates))
    commands=[("cpu",w,[args.oracle,args.replay,specs,args.prefix,args.horizon,w,4393102]) for w in args.workers]
    gpu_command=[args.gpu,args.image,args.replay,specs,args.prefix,args.horizon,args.arena_mib,args.keep,4393102]
    if args.host_prefix:gpu_command.append("--host-prefix")
    if args.isolated_warps:gpu_command.append("--isolated-warps")
    commands.append(("cuda",0,gpu_command))
    baseline=None;cases=[]
    report={"schema":"irisu-full-game-search-benchmark-v1","promotion_eligible":False,
      "scope":"bounded published-replay prefix, sparse mutations, fixed continuation, full game rules and exact-host translation",
      "record_search_provenance":"original final record runner not established; this is a new fixed workload",
      "inputs":{"replay_sha256":sha(args.replay),"specs_sha256":sha(specs),"image_sha256":sha(args.image)},
      "binaries":{"oracle_sha256":sha(args.oracle),"gpu_sha256":sha(args.gpu)},
      "prefix":args.prefix,"horizon":args.horizon,"candidates":args.candidates,"host_prefix":args.host_prefix,"isolated_warps":args.isolated_warps,
      "timing_contract":{"warmups":1,"timed_repeats":args.repeats,
        "wall":"process startup, runtime initialization, prefix preparation/upload, branches, complete futures, stable ranking, GPU selected-state compaction and result download; Python parity comparison excluded",
        "gpu_component":"CUDA events; setup/prefix, snapshot copy, future rollout, rank+compact+verification separately",
        "telemetry":"whole-device nvidia-smi samples at approximately 200 ms; not process exclusive",
        "validation":"warmup additionally compares compacted hidden state/arena and steps selected state plus independent source; timed repetitions skip that extra validation workload",
        "cpu":"native ELF32 exact core, prefix once then bounded fork/COW candidate workers; all frame digests included"},"cases":cases}
    for backend,workers,command in commands:
        warm,wm=invoke(command,args.timeout,args.candidates)
        if baseline is None:baseline=warm
        if warm!=baseline:raise RuntimeError(f"{backend}/{workers} warmup differs from native full per-candidate frame digest/results/rank oracle")
        timed_command=command+(["--skip-compaction-check"] if backend=="cuda" else [])
        runs=[]
        with Telemetry() as telemetry:
            for repeat in range(args.repeats):
                data,run=invoke(timed_command,args.timeout,args.candidates)
                if data!=baseline:raise RuntimeError(f"{backend}/{workers} repeat {repeat} parity mismatch")
                runs.append(run)
        rows,ranks=decode(warm,args.candidates)
        median=statistics.median(x["wall_seconds"] for x in runs)
        ticks=sum(row[13] for row in rows)
        case={"backend":backend,"workers":workers,"command":list(map(str,timed_command)),"validation_command":list(map(str,command)),"warmup":wm,"runs":runs,
          "parity":True,"result_sha256":hashlib.sha256(warm).hexdigest(),"winner":ranks[0],
          "actual_simulated_ticks":ticks,"median_wall_seconds":median,
          "search_branches_per_second":args.candidates/median,"simulation_ticks_per_second":ticks/median,
          "telemetry":telemetry.samples}
        if telemetry.samples:
            case["peak_vram_mib"]=max(x[2] for x in telemetry.samples)
            case["peak_gpu_util_percent"]=max(x[1] for x in telemetry.samples)
        cases.append(case)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in case.items() if k in {"backend","workers","parity","median_wall_seconds","search_branches_per_second","simulation_ticks_per_second","winner"}}),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
