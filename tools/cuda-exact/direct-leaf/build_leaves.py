#!/usr/bin/env python3
"""Bounded isolated build of selected scalar leaves and matched control consumers."""
import argparse,concurrent.futures,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build-cuda-function-calls');p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/gpu-exact-speedup-20261103/function-calls/unpatched');p.add_argument('--typed',type=Path,required=True);p.add_argument('--entries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
for name in ['build','baseline','typed','entries','output']:setattr(a,name,getattr(a,name).resolve())
a.output.mkdir(parents=True,exist_ok=True);base=a.build/'clone/cuda';commands=[];entries=json.loads(a.entries.read_text());snapshots={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c,log=None):
 c=list(map(str,c));commands.append(c)
 if log:
  with log.open('w') as f:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=90,cwd=ROOT)
 else:subprocess.run(c,check=True,cwd=ROOT)
def expand(c):
 result=[]
 for t in c:result.extend(shlex.split((base/t[1:]).read_text()) if t.startswith('@') else [t])
 return result
links={};device={}
for target in ['irisu-cuda-physics-probe','irisu-cuda-search','irisu-cuda-search-session']:
 directory=base/f'CMakeFiles/{target}.dir';links[target]=expand(shlex.split((directory/'link.txt').read_text()));device[target]=[]
 for name in ['deviceObjects1.rsp','deviceLinkLibs.rsp']:device[target]+=shlex.split((directory/name).read_text())
 for token in links[target]+device[target]:
  source=(base/token).resolve()
  if source.is_file() and source.is_relative_to(base) and source.suffix in ('.o','.a') and 'cmake_device_link' not in str(source):
   dest=a.output/'snapshot'/source.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);snapshots[str(source)]=dest
shutil.copytree(a.typed,a.output/'typed',dirs_exist_ok=True)
run(['python3',Path(__file__).with_name('patch_leaves.py'),a.baseline/'generated',a.output/'typed',a.output/'generated','--entries',*entries])
control=a.output/'control';typed=a.output/'scalar';control.mkdir(exist_ok=True);typed.mkdir(exist_ok=True)
for d in [control,typed]:shutil.copy2(a.baseline/'libirisu_cuda_regions.a',d/'libirisu_cuda_regions.a')
flags=['/opt/cuda/bin/nvcc','-ccbin','g++-14','-std=c++20','-O2','-DNDEBUG','-gencode=arch=compute_70,code=[compute_70,sm_70]','-dc','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--expt-extended-lambda','-diag-suppress=68,177','-Xptxas=-O1,-v','-I'+str(a.output/'generated')]
def compile_leaf(entry):
 obj=a.output/f'guest_region_{entry}.cu.o';start=time.perf_counter();run([*flags,a.output/f'generated/guest_region_{entry}.cu','-o',obj],a.output/f'leaf-{entry}-build.log');return {'entry':entry,'seconds':time.perf_counter()-start,'object':str(obj)}
start=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:compiled=list(pool.map(compile_leaf,entries))
run(['ar','r',typed/'libirisu_cuda_regions.a',*[x['object'] for x in compiled]])
def transform(token,archive):
 source=(base/token).resolve()
 if source==base/'libirisu_cuda_regions.a':return str(archive)
 if str(source) in snapshots:return str(snapshots[str(source)])
 if not token.startswith('-') and source.is_file():return str(source)
 return token
for variant in [control,typed]:
 archive=variant/'libirisu_cuda_regions.a'
 for target in links:
  dlink=variant/(target+'-device-link.o');run(['/opt/cuda/bin/nvcc','-ccbin','g++-14','-gencode=arch=compute_70,code=[compute_70,sm_70]','-shared','-dlink',*[transform(t,archive) for t in device[target]],'-o',dlink])
  final=[];skip=False
  for token in links[target]:
   if skip:skip=False;continue
   if token=='-o':final+=['-o',str(variant/target)];skip=True;continue
   if token.startswith('-Wl,--dependency-file='):continue
   final.append(str(dlink) if 'cmake_device_link.o' in token else transform(token,archive))
  run(final)
report={'schema':'irisu-selected-scalar-leaves-build-v1','elapsed_seconds':time.perf_counter()-start,'entries':entries,'compiled':compiled,'commands':commands,'snapshots_sha256':{str(f.relative_to(a.output)):sha(f) for f in a.output.rglob('*') if f.is_file() and not f.name.endswith('.log')}}
(a.output/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'compiled':len(compiled),'seconds':report['elapsed_seconds']}))
