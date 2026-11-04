#!/usr/bin/env python3
"""Verify chunk composition, signed ranking, and native multi-root references."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from gpu_search_native_reference import RESULT,packet,ranking
from gpu_search_specs import generate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['oracle','library','replay']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rows=[[0]*32 for _ in range(6)];rows[0][17]=1
    for i,value in enumerate([100,-2,-1,5,5,0]):rows[i][21]=value
    assert ranking(rows)==[3,4,5,2,1,0]
    encoded=packet(rows);assert len(encoded)==16+260*len(rows)
    assert [RESULT.unpack_from(encoded,16+256*i) for i in range(len(rows))]==[tuple(row) for row in rows]
    assert list(struct.unpack_from('<6I',encoded,16+6*256))==ranking(rows)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='irisu-gpu-native-reference-') as temporary:
        work=Path(temporary);specs=work/'specs.bin';specs.write_bytes(generate(a.replay.read_bytes(),100,4,37))
        for roots,trace in [(1,False),(3,False),(3,True)]:
            outputs=[]
            for chunk in [7,64]:
                output=work/f'expected-{roots}-{int(trace)}-{chunk}.bin'
                command=[sys.executable,str(Path(__file__).with_name('gpu_search_native_reference.py')),'--oracle',str(a.oracle.resolve()),'--library',str(a.library.resolve()),'--replay',str(a.replay.resolve()),'--specs',str(specs),'--output',str(output),'--roots',str(roots),'--root-seed-step','5','--root-prefix-step','2','--chunk',str(chunk)]
                if trace:command.append('--trace-every-tick')
                process=subprocess.run(command,capture_output=True,timeout=120)
                assert process.returncode==0,process.stderr.decode(errors='replace')
                outputs.append(output.read_bytes())
            assert outputs[0]==outputs[1],'chunk partition changed exact native packet'
            cases.append(dict(roots=roots,trace=trace,candidates=37,sha256=hashlib.sha256(outputs[0]).hexdigest()))
    print(json.dumps(dict(passed=True,signed_failure_first_ranking=True,native_chunk_partition_cases=cases)))

if __name__=='__main__':main()
