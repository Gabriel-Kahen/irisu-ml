#!/usr/bin/env python3
"""Validate resident dependent GPU beam depths against independent native forks.

Small gates compose at most four sparse mutations per leaf, so every candidate
at every depth can be independently evaluated from the native root checkpoint.
The exported winning replay is also checked from reset by game-oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

from search_validate import SPEC,RESULT,Telemetry,candidates,decode,sha


def invoke(command,timeout):
    env=dict(os.environ);env.pop("CUDA_VISIBLE_DEVICES",None)
    start=time.perf_counter()
    p=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       start_new_session=True,env=env)
    try:out,err=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.communicate();raise
    if p.returncode:raise RuntimeError(f"failed {p.returncode}: {command}\n"+err.decode(errors="replace")[-6000:])
    return out,{"wall_seconds":time.perf_counter()-start,"stderr":err.decode(errors="replace")}


def decode_session(data,depths):
    if len(data)<8 or struct.unpack_from("<2I",data)!=(2,depths):raise RuntimeError("bad session version/depth header")
    offset=8;rounds=[]
    for depth in range(depths):
        count,keep=struct.unpack_from("<2I",data,offset);offset+=8
        size=16+count*(RESULT.size+4);payload=data[offset:offset+size];offset+=size
        results,ranks=decode(payload,count)
        if not 1<=keep<=count:raise RuntimeError("invalid compacted beam width")
        rounds.append({"count":count,"keep":keep,"results":results,"ranks":ranks,"payload":payload})
    if offset!=len(data):raise RuntimeError("trailing session bytes")
    return rounds


def specs_for_paths(specs,parents,depth,horizon):
    rows=[];paths=[]
    for parent in parents:
        for ordinal,spec in enumerate(specs):
            path=parent+[ordinal];mutations=[]
            for d,index in enumerate(path):
                source=specs[index]
                mutations.extend((d*horizon+source[2+2*i],source[3+2*i]) for i in range(source[0]))
            if len(mutations)>4:raise ValueError("native flattened gate supports at most four mutations per path")
            encoded=[len(mutations),0]
            for tick,value in mutations:encoded.extend((tick,value))
            encoded.extend([0]*(10-len(encoded)));rows.append(SPEC.pack(*encoded));paths.append(path)
    return b"".join(rows),paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",type=Path,required=True);p.add_argument("--oracle",type=Path,required=True)
    p.add_argument("--game-oracle",type=Path,required=True);p.add_argument("--image",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True);p.add_argument("--prefix",type=int,default=100)
    p.add_argument("--horizon",type=int,default=2);p.add_argument("--fanout",type=int,default=3)
    p.add_argument("--depths",type=int,default=3);p.add_argument("--beam",type=int,default=1)
    p.add_argument("--arena-mib",type=int,default=8);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--timeout",type=float,default=600);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--host-prefix",action="store_true");p.add_argument("--isolated-warps",action="store_true")
    p.add_argument("--final-only",action="store_true");p.add_argument("--global-guest",action="store_true")
    p.add_argument("--block-threads",type=int,default=4);p.add_argument("--stack-kib",type=int,default=16)
    p.add_argument("--verify-winner-only",action="store_true",
                   help="allow up to16 resident depths; validate stable ranks/lineage and independently replay the winner, omitting flattened native checks for every leaf")
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    if not 1<=args.fanout<=16384 or not 1<=args.beam<=64 or args.fanout*args.beam>16384 or not 1<=args.horizon<=100000:
        p.error("invalid bounded session fanout, beam or horizon")
    if not 1<=args.depths<=(16 if args.verify_winner_only else 4):
        p.error("independent flattened gate requires1..4 depths; winner-only mode allows1..16")
    specs_path=args.output.with_suffix(".specs.bin")
    spec_bytes=candidates(args.replay.read_bytes(),args.prefix,args.horizon,args.fanout)
    specs_path.write_bytes(spec_bytes);specs=list(SPEC.iter_unpack(spec_bytes))
    winner=args.output.with_suffix(".winner.rpy");lineage=args.output.with_suffix(".lineage.json")
    command=[args.gpu,args.image,args.replay,specs_path,args.prefix,args.horizon,args.arena_mib,args.beam,args.depths,4393102,winner,lineage]
    if args.host_prefix:command.append("--host-prefix")
    if args.isolated_warps:command.append("--isolated-warps")
    if args.final_only:command.append("--final-only")
    if args.global_guest:command.append("--global-guest")
    command.extend(["--block-threads",args.block_threads,"--stack-kib",args.stack_kib])
    with Telemetry() as telemetry:data,run=invoke(command,args.timeout)
    rounds=decode_session(data,args.depths)
    args.output.with_suffix(".results.bin").write_bytes(data)
    report={"schema":"irisu-resident-search-session-parity-v2","promotion_eligible":False,
      "inputs":{"replay_sha256":sha(args.replay),"specs_sha256":sha(specs_path),"image_sha256":sha(args.image)},
      "binaries":{"gpu_sha256":sha(args.gpu),"oracle_sha256":sha(args.oracle),"game_oracle_sha256":sha(args.game_oracle)},
      "command":list(map(str,command)),"gpu":run,"rounds":[],"all_candidate_parity":False,
      "prefix":args.prefix,"horizon":args.horizon,"beam":args.beam,"fanout":args.fanout,"depths":args.depths}
    report["trace_every_tick"]=not args.final_only
    report["distinct_normalized_specs"]=len(set(specs))
    report["verification_mode"]="independent_winner_replay" if args.verify_winner_only else "every_native_candidate_and_winner_replay"
    report["telemetry"]=telemetry.samples
    if telemetry.samples:
        report["peak_vram_mib"]=max(row[2] for row in telemetry.samples)
        report["peak_gpu_util_percent"]=max(row[1] for row in telemetry.samples)
    report["gpu_metadata"]=next((json.loads(line) for line in reversed(run["stderr"].splitlines()) if line.startswith("{")),{})
    parents=[[]]
    for depth,round in enumerate(rounds):
        if args.verify_winner_only:
            paths=[parent+[ordinal] for parent in parents for ordinal in range(len(specs))]
            native_command=[];native_run={};native=None
        else:
            native_specs,paths=specs_for_paths(specs,parents,depth,args.horizon)
            path=args.output.with_suffix(f".depth-{depth}.native-specs.bin");path.write_bytes(native_specs)
            native_command=[args.oracle,args.replay,path,args.prefix,args.horizon*(depth+1),args.workers,4393102]
            if args.final_only:native_command.append("--final-only")
            native,native_run=invoke(native_command,args.timeout)
            if native!=round["payload"]:
                cpu_results,cpu_ranks=decode(native,round["count"])
                mismatch=next(((i,j,a,b) for i,(cr,gr) in enumerate(zip(cpu_results,round["results"]))
                               for j,(a,b) in enumerate(zip(cr,gr)) if a!=b),None)
                raise RuntimeError(f"depth{depth} full candidate/rank mismatch: {mismatch}; native ranks{cpu_ranks}, GPU ranks{round['ranks']}")
        parents=[paths[i] for i in round["ranks"][:round["keep"]]]
        report["rounds"].append({"depth":depth,"candidates":round["count"],"selected_paths":parents,
          "distinct_final_frame_fingerprints":len({row[:12] for row in round["results"]}),
          "distinct_trace_fingerprints":len({row[12] for row in round["results"]}),
          "native_command":list(map(str,native_command)),"native":native_run,"native_candidate_parity":None if args.verify_winner_only else True,
          "result_sha256":hashlib.sha256(round["payload"]).hexdigest()})
    manifest=json.loads(lineage.read_text())
    if manifest["winner_candidates"]!=parents[0]:raise RuntimeError("exported winning lineage disagrees with stable ranked beam")
    final=rounds[-1]["results"][rounds[-1]["ranks"][0]]
    replay_data=winner.read_bytes();steps=(len(replay_data)-52)//4
    if steps!=final[0]:raise RuntimeError("winning replay length disagrees with selected tick")
    verify_path=winner
    verify_command=[args.game_oracle,verify_path]
    if not final[16]:
        verify_path=args.output.with_suffix(".partial-verification.rpy")
        verify_path.write_bytes(replay_data+bytes(4));verify_command=[args.game_oracle,verify_path,steps]
    frames,verification=invoke(verify_command,args.timeout)
    if len(frames)!=(steps+1)*96 or struct.unpack_from("<12Q",frames,len(frames)-96)!=final[:12]:
        raise RuntimeError("winning trace independent native reset replay disagrees with final GPU frame")
    manifest["native_replay_verified"]=True;manifest["native_verification_scope"]="natural canonical terminal" if final[16] else "bounded nonterminal trace"
    manifest["winner_replay_sha256"]=sha(winner);lineage.write_text(json.dumps(manifest,indent=2)+"\n")
    report.update(all_candidate_parity=None if args.verify_winner_only else True,
      winning_lineage_consistent=True,winning_lineage_native_parity=None if args.verify_winner_only else True,
      winner_replay_sha256=sha(winner),native_reset_replay_verified=True,
      verification_scope="natural_terminal" if final[16] else "bounded_nonterminal",natural_terminal=bool(final[16]),
      canonical_score=final[20],winner_trace_ticks=steps,independent_verification=verification)
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k in {"all_candidate_parity","native_reset_replay_verified","verification_scope","canonical_score","winner_trace_ticks"}}))
    return 0
if __name__=="__main__":raise SystemExit(main())
