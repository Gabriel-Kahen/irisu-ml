#!/usr/bin/env python3
"""Build only one typed solver object and relink copied physics/search consumers."""
import argparse,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build-cuda-function-calls');p.add_argument('--preserved',type=Path,default=ROOT/'artifacts/gpu-exact-speedup-20261103/function-calls/unpatched');p.add_argument('--output',type=Path,default=ROOT/'build-cuda-direct-leaf/replacement');a=p.parse_args()
a.build=a.build.resolve();a.preserved=a.preserved.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);base=a.build/'clone/cuda';commands=[]
def run(cmd):
 commands.append(list(map(str,cmd)));print(shlex.join(commands[-1]),flush=True);subprocess.run(commands[-1],check=True,cwd=ROOT)
def expand(cmd):
 out=[]
 for token in cmd:
  if token.startswith('@'):out+=shlex.split((base/token[1:]).read_text())
  else:out.append(token)
 return out
# Snapshot consumer inputs before another workstream rebuilds production files.
links={};snapshots={}
for target in ['irisu-cuda-physics-probe','irisu-cuda-search']:
 directory=base/f'CMakeFiles/{target}.dir';links[target]=expand(shlex.split((directory/'link.txt').read_text()))
 for name in ['deviceObjects1.rsp','deviceLinkLibs.rsp']:
  file=directory/name
  if file.exists():
   for token in shlex.split(file.read_text()):
    source=base/token
    if source.is_file() and source.suffix in ['.o','.a'] and source.is_relative_to(base):
     dest=a.output/'snapshot'/source.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);snapshots[str(source.resolve())]=dest
 for token in links[target]:
  source=base/token
  if source.is_file() and source.suffix=='.o' and 'cmake_device_link' not in str(source):
   dest=a.output/'snapshot'/source.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);snapshots[str(source.resolve())]=dest
for name in ['libirisu_cuda_regions.a','libirisu_cuda_search.a']:
 shutil.copy2(a.preserved/name,a.output/name);snapshots[str((base/name).resolve())]=a.output/name
shutil.copytree(a.preserved/'generated',a.output/'generated',dirs_exist_ok=True)
run([ROOT/'.venv-gpu/bin/python',Path(__file__).with_name('generate_velocity.py'),ROOT/'.local/exact-runtime-20260824/exact-runtime/guest/libirisu_box2d_msvc_exact_multiworld.so',a.output/'typed','--baseline-generated',a.output/'generated'])
run(['python3',Path(__file__).with_name('patch_function_unit.py'),a.output/'generated',a.output/'typed/velocity.cuh'])
obj=a.output/'guest_region_30224.cu.o'
run(['/opt/cuda/bin/nvcc','-ccbin','g++-14','-std=c++20','-O2','-DNDEBUG','-gencode=arch=compute_70,code=[compute_70,sm_70]','-dc','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--expt-extended-lambda','-diag-suppress=68,177','-Xptxas=-O1,-v','-I'+str(a.output/'generated'),a.output/'generated/guest_region_30224.cu','-o',obj])
run(['ar','r',a.output/'libirisu_cuda_regions.a',obj])
def transform(token):
 source=(base/token).resolve()
 if str(source) in snapshots:return str(snapshots[str(source)])
 if not token.startswith('-') and source.is_file():return str(source)
 return token
for target,original in links.items():
 directory=base/f'CMakeFiles/{target}.dir';dev=[]
 for name in ['deviceObjects1.rsp','deviceLinkLibs.rsp']:
  dev+=shlex.split((directory/name).read_text())
 dev=[transform(t) for t in dev]
 dlink=a.output/(target+'-device-link.o')
 run(['/opt/cuda/bin/nvcc','-ccbin','g++-14','-gencode=arch=compute_70,code=[compute_70,sm_70]','-shared','-dlink',*dev,'-o',dlink])
 final=[];skip=False
 for index,token in enumerate(original):
  if skip:skip=False;continue
  if token=='-o':final+=['-o',str(a.output/target)];skip=True;continue
  if token.startswith('-Wl,--dependency-file='):continue
  if 'cmake_device_link.o' in token:final.append(str(dlink));continue
  final.append(transform(token))
 run(final)
inputs=[Path(__file__),Path(__file__).with_name('generate_velocity.py'),Path(__file__).with_name('support.cuh'),Path(__file__).with_name('patch_function_unit.py'),ROOT/'clone/cuda/guest.hpp',ROOT/'tools/cuda-exact/x87_numeric.cuh']
receipt={'inputs_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},'schema':'irisu-direct-leaf-replacement-build-v1','commands':commands,'files_sha256':{str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()}}
(a.output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
