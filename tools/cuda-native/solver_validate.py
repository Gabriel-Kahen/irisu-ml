#!/usr/bin/env python3
"""Native hidden-method vs independent typed CPU/CUDA solver stage packets."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--host',type=Path,required=True);p.add_argument('--cuda',type=Path);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--memcheck',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
report={'schema':'irisu-independent-solver-stage-v1','promotion_eligible':False,'scope':'Actual pinned ELF32 MSVC constructor/PreSolve/velocity/position/PostSolve, independent typed CPU/CUDA solver; contact-list stage snapshots, not whole independent scene backend.','inputs_sha256':{str(f):sha(f) for f in [a.oracle,a.host,a.library]+([a.cuda] if a.cuda else [])},'runs':[],'status':'running'}
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
def run(name,command):
 start=time.perf_counter();proc=subprocess.run(list(map(str,command)),capture_output=True,env=env,timeout=60);stdout=a.output/(name+'.bin');stderr=a.output/(name+'.stderr');stdout.write_bytes(proc.stdout);stderr.write_bytes(proc.stderr)
 row={'name':name,'command':list(map(str,command)),'returncode':proc.returncode,'wall_seconds':time.perf_counter()-start,'stdout_bytes':len(proc.stdout),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)};report['runs'].append(row)
 metadata=[json.loads(x) for x in proc.stderr.decode().splitlines() if x.startswith('{')];row['metadata']=metadata[-1] if metadata else None
 if proc.returncode:raise RuntimeError(f'{name} failed: '+proc.stderr.decode()[-2000:])
 return proc.stdout,row
try:
 cases=a.output/'cases.bin';expected,native=run('native',[a.oracle.resolve(),a.library.resolve(),cases.resolve()]);report['cases_sha256']=sha(cases)
 for name,binary in [('host',a.host),('cuda',a.cuda)]:
  if binary is None:continue
  actual,row=run(name,[binary.resolve(),cases.resolve()]+(['--gpu'] if name=='cuda' else []));row['all_physical_output_bytes_equal_native']=actual==expected
  if actual!=expected:
   first=next((i for i,(x,y) in enumerate(zip(actual,expected)) if x!=y),min(len(actual),len(expected)));row['first_different_byte']=first;raise RuntimeError(f'{name} differs from actual native atbyte{first}')
 if a.memcheck:
  if a.cuda is None:raise ValueError('memcheck requires CUDA binary')
  actual,row=run('memcheck',['/opt/cuda/compute-sanitizer/compute-sanitizer','--tool','memcheck','--error-exitcode','42','--log-file',str((a.output/'memcheck.log').resolve()),a.cuda.resolve(),cases.resolve(),'--gpu']);row['all_physical_output_bytes_equal_native']=actual==expected
  if actual!=expected:raise RuntimeError('memcheck execution differs from native')
  log=(a.output/'memcheck.log').read_text();row['memcheck_error_summary_zero']='ERROR SUMMARY: 0 errors' in log
  if not row['memcheck_error_summary_zero']:raise RuntimeError('memcheck missing zero-error summary')
 report['status']='passed'
except BaseException as error:report['status']='failed';report['failure']=str(error);raise
finally:
 (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
