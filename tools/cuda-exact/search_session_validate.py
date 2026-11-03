#!/usr/bin/env python3
"""Validate resident dependent GPU beam depths against independent native forks.

Small gates independently evaluate every candidate from the native root
checkpoint. Longer edit histories are materialized in grouped base replays;
the root prefix and cumulative hash history remain unchanged. The exported
winning replay is also checked from reset by game-oracle.
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
    if len(data)<8:raise RuntimeError("short session header")
    version,declared=struct.unpack_from("<2I",data)
    if version not in (2,3,4) or declared!=depths:raise RuntimeError("bad session version/depth header")
    offset=8;rounds=[]
    for depth in range(depths):
        count,keep=struct.unpack_from("<2I",data,offset);offset+=8
        size=16+count*(RESULT.size+4);payload=data[offset:offset+size];offset+=size
        results,ranks=decode(payload,count)
        if not 1<=keep<=count:raise RuntimeError("invalid compacted beam width")
        selected=ranks[:keep];protected_rank=None
        if version>=3:
            selected=list(struct.unpack_from(f"<{keep}I",data,offset));offset+=4*keep
            protected_rank=struct.unpack_from("<I",data,offset)[0];offset+=4
            if len(set(selected))!=keep or any(i>=count for i in selected):raise RuntimeError("invalid selected child indices")
        lifecycle=None
        if version>=4:
            lifecycle=list(struct.unpack_from(f"<{count}I",data,offset));offset+=4*count
            if any(flag&~3 for flag in lifecycle):raise RuntimeError("invalid lifecycle flags")
        rounds.append({"version":version,"count":count,"keep":keep,"results":results,"ranks":ranks,
                       "selected":selected,"protected_rank":protected_rank,"lifecycle":lifecycle,"payload":payload})
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


def native_round(args,specs,parents,depth):
    """Keep the root checkpoint/hash seed while admitting longer edit history."""
    paths=[parent+[ordinal] for parent in parents for ordinal in range(len(specs))]
    base_command=[args.oracle,args.replay,None,args.prefix,args.horizon*(depth+1),args.workers,4393102]
    if args.final_only:base_command.append("--final-only")
    try:
        encoded,_=specs_for_paths(specs,parents,depth,args.horizon)
    except ValueError:
        # Previous segments become part of the immutable base replay. Only
        # current-segment edits occupy the native oracle's four-edit record.
        original=args.replay.read_bytes();all_rows=[];all_flags=[];runs=[]
        for parent_index,parent in enumerate(parents):
            replay=bytearray(original)
            required=52+4*(args.prefix+depth*args.horizon)
            if len(replay)<required:replay.extend(bytes(required-len(replay)))
            for d,ordinal in enumerate(parent):
                source=specs[ordinal]
                for i in range(source[0]):
                    tick=args.prefix+d*args.horizon+source[2+2*i]
                    struct.pack_into("<I",replay,52+4*tick,source[3+2*i])
            rp=args.output.with_suffix(f".depth-{depth}.parent-{parent_index}.rpy");rp.write_bytes(replay)
            sp=args.output.with_suffix(f".depth-{depth}.parent-{parent_index}.specs.bin")
            encoded=[]
            for source in specs:
                row=list(source)
                for i in range(source[0]):row[2+2*i]+=depth*args.horizon
                encoded.append(SPEC.pack(*row))
            sp.write_bytes(b"".join(encoded))
            lp=args.output.with_suffix(f".depth-{depth}.parent-{parent_index}.lifecycle.bin")
            command=base_command.copy();command[1]=rp;command[2]=sp;command.extend(["--emit-lifecycle",lp])
            data,run=invoke(command,args.timeout);rows,_=decode(data,len(specs));all_rows.extend(rows)
            all_flags.extend(struct.unpack(f"<{len(specs)}I",lp.read_bytes()))
            runs.append({"command":list(map(str,command)),"replay_sha256":sha(rp),"specs_sha256":sha(sp),**run})
        ranks=sorted(range(len(all_rows)),key=lambda i:(all_rows[i][21:],-i),reverse=True)
        payload=struct.pack("<4I",0x32535249,2,len(all_rows),RESULT.size)+b"".join(RESULT.pack(*row) for row in all_rows)+struct.pack(f"<{len(ranks)}I",*ranks)
        return payload,all_flags,[],{"grouped_parent_oracles":runs},paths
    sp=args.output.with_suffix(f".depth-{depth}.native-specs.bin");sp.write_bytes(encoded)
    lp=args.output.with_suffix(f".depth-{depth}.native-lifecycle.bin")
    command=base_command.copy();command[2]=sp;command.extend(["--emit-lifecycle",lp])
    data,run=invoke(command,args.timeout)
    flags=list(struct.unpack(f"<{len(paths)}I",lp.read_bytes()))
    return data,flags,command,run,paths


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
    p.add_argument("--no-protect-incumbent",action="store_true",help="allow the zero-edit source lineage to be pruned")
    p.add_argument("--greedy-beam",action="store_true",help="use legacy canonical top-K survivors instead of retaining the best child plus live futures")
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
    if args.no_protect_incumbent:command.append("--no-protect-incumbent")
    if args.greedy_beam:command.append("--greedy-beam")
    command.extend(["--block-threads",args.block_threads,"--stack-kib",args.stack_kib])
    with Telemetry() as telemetry:data,run=invoke(command,args.timeout)
    rounds=decode_session(data,args.depths)
    args.output.with_suffix(".results.bin").write_bytes(data)
    report={"schema":"irisu-resident-search-session-parity-v4","promotion_eligible":False,
      "inputs":{"replay_sha256":sha(args.replay),"specs_sha256":sha(specs_path),"image_sha256":sha(args.image)},
      "binaries":{"gpu_sha256":sha(args.gpu),"oracle_sha256":sha(args.oracle),"game_oracle_sha256":sha(args.game_oracle)},
      "command":list(map(str,command)),"gpu":run,"rounds":[],"all_candidate_parity":False,
      "prefix":args.prefix,"horizon":args.horizon,"beam":args.beam,"fanout":args.fanout,"depths":args.depths}
    report["trace_every_tick"]=not args.final_only
    report["protect_incumbent"]=not args.no_protect_incumbent
    report["greedy_beam"]=args.greedy_beam
    report["distinct_normalized_specs"]=len(set(specs))
    report["verification_mode"]="independent_winner_replay" if args.verify_winner_only else "every_native_candidate_and_winner_replay"
    report["telemetry"]=telemetry.samples
    if telemetry.samples:
        report["peak_vram_mib"]=max(row[2] for row in telemetry.samples)
        report["peak_gpu_util_percent"]=max(row[1] for row in telemetry.samples)
    report["gpu_metadata"]=next((json.loads(line) for line in reversed(run["stderr"].splitlines()) if line.startswith("{")),{})
    parents=[[]];protected_parent=0;best_path=[]
    for depth,round in enumerate(rounds):
        if args.verify_winner_only:
            paths=[parent+[ordinal] for parent in parents for ordinal in range(len(specs))]
            native_command=[];native_run={};native=None
        else:
            native,native_flags,native_command,native_run,paths=native_round(args,specs,parents,depth)
            if native!=round["payload"]:
                cpu_results,cpu_ranks=decode(native,round["count"])
                mismatch=next(((i,j,a,b) for i,(cr,gr) in enumerate(zip(cpu_results,round["results"]))
                               for j,(a,b) in enumerate(zip(cr,gr)) if a!=b),None)
                raise RuntimeError(f"depth{depth} full candidate/rank mismatch: {mismatch}; native ranks{cpu_ranks}, GPU ranks{round['ranks']}")
            if native_flags!=round["lifecycle"]:raise RuntimeError(f"depth{depth} native/GPU lifecycle mismatch")
        if round["lifecycle"] is None:raise RuntimeError("runtime lacks live-beam lifecycle flags")
        if args.greedy_beam:expected_selected=round["ranks"][:round["keep"]]
        else:
            expected_selected=round["ranks"][:1]
            expected_selected.extend(i for i in round["ranks"][1:] if round["lifecycle"][i]==0)
            expected_selected=expected_selected[:round["keep"]]
            expected_selected.extend(i for i in round["ranks"] if i not in expected_selected)
            expected_selected=expected_selected[:round["keep"]]
        expected_protected=0xffffffff
        if not args.no_protect_incumbent:
            if round["version"]<3:raise RuntimeError("runtime does not support protected-incumbent selection")
            protected_child=protected_parent*len(specs)
            if protected_child not in expected_selected:expected_selected[-1]=protected_child
            expected_protected=expected_selected.index(protected_child)
        if round["selected"]!=expected_selected or round["protected_rank"]!=expected_protected:
            raise RuntimeError(f"depth{depth} incumbent-selection policy mismatch")
        protected_parent=expected_protected
        best_path=paths[round["ranks"][0]]
        parents=[paths[i] for i in round["selected"]]
        report["rounds"].append({"depth":depth,"candidates":round["count"],"selected_paths":parents,
          "distinct_final_frame_fingerprints":len({row[:12] for row in round["results"]}),
          "distinct_trace_fingerprints":len({row[12] for row in round["results"]}),
          "selected_children":round["selected"],"protected_parent_rank":round["protected_rank"],
          "live_children":sum(flag==0 for flag in round["lifecycle"]),
          "selected_live_children":sum(round["lifecycle"][i]==0 for i in round["selected"]),
          "native_command":list(map(str,native_command)),"native":native_run,"native_candidate_parity":None if args.verify_winner_only else True,
          "result_sha256":hashlib.sha256(round["payload"]).hexdigest()})
    manifest=json.loads(lineage.read_text())
    if manifest["winner_candidates"]!=best_path:raise RuntimeError("exported winning lineage disagrees with best final canonical leaf")
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
    incumbent_verification={};incumbent_floor=None
    if not args.no_protect_incumbent:
        source_frames,incumbent_verification=invoke([args.game_oracle,args.replay],args.timeout)
        source_header=struct.unpack_from("<5I",args.replay.read_bytes())
        source_metadata=next(json.loads(line) for line in reversed(incumbent_verification["stderr"].splitlines()) if line.startswith("{"))
        # For incomplete games the comparator prioritizes survival before
        # score. A numeric score floor is promised only for level100 sources.
        if source_header[1]>=100 and args.prefix+args.horizon*args.depths>=source_metadata["tick"]:
            if not final[16] or final[20]<source_header[2]:
                raise RuntimeError("protected resident search returned less than its independently verified source incumbent")
            incumbent_floor=source_header[2]
            manifest["native_verified_incumbent_score_floor"]=incumbent_floor
            lineage.write_text(json.dumps(manifest,indent=2)+"\n")
    report.update(all_candidate_parity=None if args.verify_winner_only else True,
      winning_lineage_consistent=True,winning_lineage_native_parity=None if args.verify_winner_only else True,
      winner_replay_sha256=sha(winner),native_reset_replay_verified=True,
      verification_scope="natural_terminal" if final[16] else "bounded_nonterminal",natural_terminal=bool(final[16]),
      canonical_score=final[20],winner_trace_ticks=steps,independent_verification=verification)
    report["native_verified_incumbent_score_floor"]=incumbent_floor
    report["incumbent_native_verification"]=incumbent_verification
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k in {"all_candidate_parity","native_reset_replay_verified","verification_scope","canonical_score","winner_trace_ticks"}}))
    return 0
if __name__=="__main__":raise SystemExit(main())
